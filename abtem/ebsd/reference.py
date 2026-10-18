"""Reference patterns: EBSD intensity over the whole hemisphere.

A single reciprocity calculation is only valid for directions near the beam
direction it was set up for, because the slab is cut with that direction along
``z`` and the propagator is accurate over a limited angle. Covering the whole
hemisphere therefore means tiling it into patches, running one calculation per
patch with its own slab, and stitching the results back together in the crystal
frame.

:class:`EBSDReferencePattern` does the tiling, the bookkeeping between the
crystal frame and each slab frame, and the stitching.
"""

from __future__ import annotations

import warnings
from typing import Optional

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.array import validate_lazy
from abtem.core.axes import EnergyAxis
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.ebsd.detectors import BackscatterDetector
from abtem.ebsd.measurements import SphericalPattern
from abtem.ebsd.orientations import (
    _crystal_and_displacements,
    _displaced,
    bulk_block,
    central_origin,
    estimate_repetitions,
    fibonacci_hemisphere,
    is_centrosymmetric,
    rotated_slab,
    zone_axis_rotation,
)
from abtem.ebsd.projections import HemisphereProjection, validate_projection
from abtem.ebsd.reciprocity import (
    EBSD,
    DepthWeight,
    _energy_metadata,
    _validate_backscatter_energy,
)
from abtem.ebsd.sampling import (
    AntialiasLossWarning,
    _warn_if_undersampled,
    potential_sampling,
)
from abtem.inelastic.phonons import FrozenPhonons
from abtem.potentials.iam import Potential

__all__ = [
    "EBSDReferencePattern",
    "patch_half_angle",
    "fft_friendly_gpts",
]

#: Density of the direction grid relative to the output image, so that binning
#: leaves no empty pixels at the rim where the grid thins out.
_DIRECTION_OVERSAMPLING = 1.15


def fft_friendly_gpts(extent: tuple[float, float], sampling: float) -> tuple[int, int]:
    """Grid points covering `extent` at `sampling` or finer, sized for the FFT.

    The calculation is almost entirely Fresnel propagation, so its cost is set
    by how fast an FFT of this size is rather than by how many points there
    are. Those are not the same thing: a grid of 71 points -- prime, so the
    transform falls back to Bloom/Rader -- takes five times as long as one of
    72, and longer than one of 128. Rounding *up* to the next size with only
    small prime factors makes the sampling finer, so it can only help, and it
    is typically several times faster.

    Parameters
    ----------
    extent : two float
        Lateral extent of the grid [Å].
    sampling : float
        Largest acceptable sampling [Å].

    Returns
    -------
    gpts : two int
    """
    from scipy.fft import next_fast_len  # type: ignore[import-untyped]

    return tuple(  # type: ignore[return-value]
        int(next_fast_len(int(np.ceil(length / sampling)))) for length in extent[:2]
    )


def patch_half_angle(n_patches: int, coverage: float = 1.2) -> float:
    """Angular radius of each patch needed to tile the hemisphere.

    A Fibonacci lattice of `n_patches` points on the hemisphere gives each point
    a cell of solid angle :math:`2\\pi / n`. Treating the cell as a regular
    hexagon and covering it with a disc through its corners gives the angular
    radius below, which `coverage` then inflates so neighbouring patches overlap
    rather than leaving seams.

    Parameters
    ----------
    n_patches : int
        Number of patches tiling the hemisphere.
    coverage : float, optional
        Factor by which the discs are enlarged past the bare covering radius
        (default 1.2).

    Returns
    -------
    half_angle : float
        Angular radius of one patch [mrad].
    """
    if n_patches < 1:
        raise ValueError(f"n_patches must be at least 1, got {n_patches}")

    return float(
        np.sqrt(8.0 * np.pi / (3.0 * np.sqrt(3.0) * n_patches)) * 1e3 * coverage
    )


def _patch_intensities(
    builder: "EBSDReferencePattern",
    block: Atoms,
    slab_axis: np.ndarray,
    local: np.ndarray,
    origin: Optional[np.ndarray],
    gpts: tuple[int, int],
    max_batch_directions: int | str,
) -> np.ndarray:
    """Cut one patch's slab and collect its directions, eagerly.

    The unit of work of a reference pattern, whether the patches run one after
    another or as tasks of a lazy one. Returns one value per direction of
    `local`, after an energy axis if there are several backscattered energies.
    """
    slab, _ = rotated_slab(
        block, slab_axis, builder.slab_cell, repetitions=(1, 1, 1), origin=origin
    )

    # A coarse sampling is warned about once, by the builder; per patch it
    # would be several hundred copies of the same warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntialiasLossWarning)
        result = EBSD(
            Potential(
                _displaced(slab, builder._frozen_phonons),
                gpts=gpts,
                slice_thickness=builder._slice_thickness,
                projection="finite",
                device=builder._device,
            ),
            BackscatterDetector(directions=local),
            energy=builder.energy,
            depth_weight=builder._depth_weight,
            backscatter_energy=builder._backscatter_energy,
            device=builder._device,
        ).build(max_batch_directions=max_batch_directions, lazy=False)

    return np.asarray(result.array)


class EBSDReferencePattern(CopyMixin, EqualityMixin):
    """A hemisphere-wide EBSD reference pattern, tiled from zone-axis patches.

    Parameters
    ----------
    atoms : ase.Atoms or FrozenPhonons
        The unit cell of the crystal. Give :class:`~abtem.FrozenPhonons` of it
        for thermal vibrations -- EMsoft's Debye-Waller factor: every slab is
        displaced configuration by configuration, as the unit cell would be,
        and the pattern is the average over them, of the scattering and the
        emitting atoms alike. It costs `num_configs` times as much.
    energy : float
        Energy of the incident beam [eV].
    slab_cell : tuple of three float
        Dimensions of the slab cut for each patch [Å]; the third entry is the
        thickness along the beam. There is no default: the right size depends
        on the crystal and on how converged the pattern has to be, and a
        default is how an undersized slab goes unnoticed.

        The thickness matters most. The Kikuchi bands build up with depth, and
        a thin slab leaves them too weak to hide the small differences between
        neighbouring patches: silicon at 30 keV, 40 Å thick, breaks into facets
        along the patch boundaries around ``[101]`` -- as badly 20 Å wide as
        10 Å -- while 100 Å gives continuous bands at either width (measured
        with the original implementation's source). That implementation was
        validated at 40 x 40 x 100 Å.

        The width sets how far a wave travels before it meets a seam. A slab
        cut at an arbitrary orientation is not periodic across its faces, and
        a reciprocity wave at an angle ``θ`` to the slab's axis drifts ``θ T``
        sideways over the thickness ``T``, into the misaligned crystal beyond a
        seam, where it stops channelling. Silicon near ``[101]``, 40 Å thick,
        loses 14% of its yield 150 mrad off the slab's axis at 12 Å wide, and
        7% at 24 Å. The default 400 patches keep every direction within 90
        mrad of its slab's axis, and at 20 x 20 x 100 Å the pattern then dims
        by less than 1% towards the patch edges, averaged over the hemisphere.
        10 Å wide also swings by 12% in atom count from one patch to the next.

        To make a run cheaper, lower `gpts`, which sets how many directions are
        calculated, rather than the slab.
    n_patches : int, optional
        Number of zone-axis patches tiling each hemisphere calculated (default
        400). More patches keep every direction closer to its slab's axis, at
        the cost of more slabs.
    gpts : int, optional
        Pixels along each axis of the projected image (default 128). Only sets
        the default density of the sampled directions; the projection itself
        happens in :meth:`SphericalPattern.project`.
    projection : str or HemisphereProjection, optional
        Projection whose even grid the directions are drawn from. One of
        ``'lambert'`` (default) or ``'stereographic'``.

        Lambert is the better choice and the default for two reasons. It is
        equal-area, so it samples the hemisphere uniformly: a stereographic
        grid puts three times more directions per steradian at the equator than
        at the pole, which is the wrong way round, since the poles carry the
        zone-axis detail. And it covers the whole hemisphere including the
        near-equator azimuths that fall in the corners of its square, which a
        stereographic disk never reaches -- so a Lambert-sampled pattern can be
        interpolated to either projection without gaps, while the reverse
        leaves holes.
    sampling : float, optional
        Real-space sampling of the potentials [Å]. Defaults to
        :func:`potential_sampling` for `atoms`; the collected angles set no
        condition of their own.
    slice_thickness : float, optional
        Multislice slice thickness [Å] (default 1.0).
    direction_gpts : int, optional
        Number of points per axis of the grid the directions are drawn from.
        Defaults to ``ceil(1.15 * gpts)``, slightly denser than the output image
        so binning leaves no empty pixels.
    max_angle : float, optional
        Angular radius of each patch [mrad]: directions farther than this from
        their patch's axis are left out. Defaults to :func:`patch_half_angle`
        for `n_patches`, which leaves out none.
    repetitions : tuple of three int, optional
        Repetitions of `atoms` used to build the block the slabs are cut from.
        Defaults to the smallest block that can contain the slab at any
        orientation. Note that this also fixes which point of the crystal sits
        at the centre of each slab.
    origin : np.ndarray, optional
        Point of `atoms` to place at the centre of every slab, passed to
        :func:`rotated_slab`. Defaults to the centroid.
    hemisphere : {'north', 'south', 'both'}, optional
        Which directions to calculate (default 'north'). A centrosymmetric
        crystal needs only one: the other is its inversion image, which the
        pattern supplies wherever it is asked about it
        (:attr:`SphericalPattern.southern`). A crystal without an inversion
        centre -- GaN, GaAs -- differs between the two, and the difference is
        what an EBSD polarity measurement reads; calculate 'both' for it, at
        twice the cost. The southern directions are laid out as EMsoft lays
        them out, ``(x, y, -z)`` below each northern one.
    centrosymmetric : bool, optional
        Whether the crystal has an inversion centre. The pattern records it, and
        it decides whether the hemisphere that was not calculated stands in as
        the inversion image silently or with a warning. If not given, it is
        determined with spglib (:func:`is_centrosymmetric`) when first needed,
        which can be slow, or fail, for a large cell without lattice symmetry --
        an amorphous or molecular-dynamics model; give True or False to skip
        the search.
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energies of the backscattered electrons [eV]; several add a leading
        energy axis. See :class:`~abtem.ebsd.reciprocity.EBSD`.
    depth_weight, device :
        Passed to :class:`~abtem.ebsd.reciprocity.EBSD`.

    Notes
    -----
    Every atom emits, lit evenly, by its cross-section for scattering straight
    back (see :mod:`abtem.ebsd.emission`): the source EMsoft builds its master
    patterns from, and the one that makes a reference pattern a property of the
    crystal alone -- no incident direction singles out a patch.

    A direction ``d`` is where the backscattered electrons go. By reciprocity
    it is calculated with a plane wave travelling the other way, along ``-d``,
    into the crystal from the side the detector is on -- so the directions
    about a patch's centre are calculated in a slab cut along the opposite
    direction. For a centrosymmetric crystal the distinction makes no
    difference; for one without an inversion centre it is what puts each
    polar face in its right hemisphere.

    The intensity leaving a crystal along ``-d`` is the intensity leaving its
    inversion image along ``d``, so the southern hemisphere is calculated as
    the northern hemisphere of the crystal inverted about the origin: every
    southern patch in the slab of the northern patch opposite it, cut from the
    inverted crystal. A direction and its opposite then differ by what
    inverting the crystal changes and nothing else -- its polarity, for GaN.
    For a crystal with an inversion centre at the origin, nothing: the two
    calculations are the same.
    """

    def __init__(
        self,
        atoms: Atoms | FrozenPhonons,
        energy: float,
        slab_cell: tuple[float, float, float],
        n_patches: int = 400,
        gpts: int = 128,
        projection: str | HemisphereProjection = "lambert",
        sampling: Optional[float] = None,
        slice_thickness: float = 1.0,
        direction_gpts: Optional[int] = None,
        max_angle: Optional[float] = None,
        repetitions: Optional[tuple[int, int, int]] = None,
        origin: Optional[np.ndarray] = None,
        hemisphere: str = "north",
        centrosymmetric: Optional[bool] = None,
        backscatter_energy=None,
        depth_weight: Optional[DepthWeight] = None,
        device: Optional[str] = None,
    ):
        if hemisphere not in ("north", "south", "both"):
            raise ValueError(
                f"hemisphere must be 'north', 'south' or 'both', got {hemisphere!r}"
            )

        if centrosymmetric not in (None, True, False):
            raise ValueError(
                f"centrosymmetric must be None, True or False, got {centrosymmetric!r}"
            )

        slab_cell = tuple(float(x) for x in slab_cell)
        if len(slab_cell) != 3 or min(slab_cell) <= 0.0:
            raise ValueError(
                f"slab_cell must be three positive lengths [Å], got {slab_cell}"
            )

        self._atoms, self._frozen_phonons = _crystal_and_displacements(atoms)
        self._energy = float(energy)
        self._n_patches = int(n_patches)
        self._slab_cell = slab_cell
        self._gpts = int(gpts)
        self._projection = validate_projection(projection)
        self._slice_thickness = float(slice_thickness)
        self._repetitions = repetitions
        self._origin = origin

        self._hemisphere = hemisphere
        self._backscatter_energy = backscatter_energy
        self._centrosymmetric = (
            None if centrosymmetric is None else bool(centrosymmetric)
        )
        self._depth_weight = depth_weight
        self._device = device

        self._max_angle = (
            patch_half_angle(self._n_patches) if max_angle is None else float(max_angle)
        )

        if sampling is None:
            self._sampling = potential_sampling(self._atoms)
        else:
            self._sampling = float(sampling)
            _warn_if_undersampled(self._atoms, self._sampling, "a")

        self._direction_gpts = (
            int(np.ceil(_DIRECTION_OVERSAMPLING * self._gpts))
            if direction_gpts is None
            else int(direction_gpts)
        )

    @property
    def atoms(self) -> Atoms:
        """The unit cell of the crystal, undisplaced."""
        return self._atoms

    @property
    def frozen_phonons(self) -> Optional[FrozenPhonons]:
        """The thermal displacements of the crystal, if any."""
        return self._frozen_phonons

    @property
    def energy(self) -> float:
        """Energy of the incident beam [eV]."""
        return self._energy

    @property
    def slab_cell(self) -> tuple[float, float, float]:
        """Dimensions of the slab cut for each patch [Å]."""
        return self._slab_cell

    @property
    def n_patches(self) -> int:
        """Number of zone-axis patches tiling the hemisphere."""
        return self._n_patches

    @property
    def gpts(self) -> int:
        """Pixels per axis of the projected image."""
        return self._gpts

    @property
    def projection(self) -> HemisphereProjection:
        """Projection the sampled directions are drawn from."""
        return self._projection

    @property
    def hemisphere(self) -> str:
        """Which directions are calculated: 'north', 'south' or 'both'."""
        return self._hemisphere

    @property
    def centrosymmetric(self) -> bool:
        """Whether the crystal has an inversion centre, so ``I(-k) = I(k)``.

        As given, or else determined with spglib on first use.
        """
        if self._centrosymmetric is None:
            self._centrosymmetric = is_centrosymmetric(self._atoms)
        return self._centrosymmetric

    def _in_hemisphere(self, northern: np.ndarray, southern: np.ndarray) -> np.ndarray:
        if self._hemisphere == "north":
            return northern
        if self._hemisphere == "south":
            return southern
        return np.concatenate([northern, southern])

    @property
    def zone_axes(self) -> np.ndarray:
        """The centre of each patch, as unit vectors of shape ``(M, 3)``.

        Each patch is calculated in a slab cut along the opposite direction,
        and the southern ones, opposite the northern ones, as those in the
        inverted crystal: see the notes.
        """
        northern = fibonacci_hemisphere(self._n_patches)
        return self._in_hemisphere(northern, -northern)

    @property
    def directions(self) -> np.ndarray:
        """Every sampled direction in the crystal frame, of shape ``(N, 3)``.

        The southern ones lie ``(x, y, -z)`` below the northern ones, as
        EMsoft lays out its southern master pattern; -0.0 on the equator.
        """
        northern = self._projection.grid(self._direction_gpts)
        return self._in_hemisphere(northern, northern * np.array([1.0, 1.0, -1.0]))

    @property
    def max_angle(self) -> float:
        """Angular radius of each patch [mrad]."""
        return self._max_angle

    @property
    def sampling(self) -> float:
        """Largest real-space sampling of the potentials [Å]."""
        return self._sampling

    @property
    def potential_gpts(self) -> tuple[int, int]:
        """Grid the potential is built on.

        Derived from :attr:`sampling` and rounded up to a size the FFT handles
        quickly; see :func:`fft_friendly_gpts`. The actual sampling is
        therefore a little finer than :attr:`sampling`.
        """
        return fft_friendly_gpts(self._slab_cell[:2], self._sampling)

    def _assign_directions(self) -> list[np.ndarray]:
        """Group the sampled directions by the patch that will calculate them.

        Each direction goes to its nearest zone axis.
        """
        zone_axes = self.zone_axes
        nearest = np.argmax(self.directions @ zone_axes.T, axis=1)
        return [np.where(nearest == j)[0] for j in range(len(zone_axes))]

    def build(
        self,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = True,
    ) -> SphericalPattern:
        """Run every patch and stitch the results into one pattern.

        Named for the abTEM builders it follows -- :meth:`abtem.Probe.build`,
        :meth:`abtem.SMatrix.build` -- which leaves ``compute`` to mean what it
        means everywhere else: realize a lazy array. So ``build(lazy=True)``
        returns a pattern whose ``compute()`` runs it.

        Parameters
        ----------
        max_batch_directions : int or str, optional
            Directions propagated at once within each patch. Passed to
            :meth:`~abtem.ebsd.reciprocity.EBSD.build`.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph with one task per
            patch, rather than running them here. Defaults to the abTEM
            configuration.
        pbar : bool, optional
            If True (default), show a progress bar over the patches. Ignored
            when lazy, where nothing runs yet.

        Returns
        -------
        pattern : SphericalPattern
            Backscattered intensity at every sampled direction, in the crystal
            frame. Use :meth:`SphericalPattern.project` to turn it into an
            image.
        """
        lazy = validate_lazy(lazy)

        directions = self.directions
        zone_axes = self.zone_axes
        assignment = self._assign_directions()

        repetitions = (
            estimate_repetitions(self._atoms, self._slab_cell)
            if self._repetitions is None
            else self._repetitions
        )
        block = bulk_block(self._atoms, self._slab_cell, repetitions)

        # The origin is given in the frame of `atoms`, but that copy of the
        # cell is the corner of the block, and a slab cut about it would be
        # mostly empty. The equivalent point of the central copy is the same
        # place in the crystal with the whole block around it.
        origin = (
            None
            if self._origin is None
            else central_origin(self._atoms, repetitions, self._origin)
        )

        # The southern patches are the northern ones of the inverted crystal
        # (see the notes). Inverted about the point the slabs are cut about,
        # the block still holds the crystal around it.
        blocks = {False: block}
        if self._hemisphere != "north":
            centre_of_inversion = (
                np.mean(block.positions, axis=0) if origin is None else origin
            )
            blocks[True] = block.copy()
            blocks[True].positions = 2.0 * centre_of_inversion - block.positions

        # Every patch shares a slab shape and a sampling, so the grid is chosen
        # once -- at a size the FFT likes, which dominates the runtime.
        gpts = self.potential_gpts

        cutoff = np.cos(self._max_angle * 1e-3)

        energies, energy_weights = _validate_backscatter_energy(
            self._backscatter_energy, self._energy
        )
        ensemble_shape: tuple[int, ...] = (len(energies),) if len(energies) > 1 else ()
        ensemble_axes = (
            [EnergyAxis(values=tuple(float(e) for e in energies))]
            if ensemble_shape
            else []
        )

        patterns: list[SphericalPattern] = []

        if lazy:
            import dask

            # One node per block, which every patch's task cuts from.
            blocks = {south: dask.delayed(b) for south, b in blocks.items()}

        progress = TqdmWrapper(
            total=len(zone_axes), enabled=pbar and not lazy, leave=False
        )
        try:
            for centre, indices in zip(zone_axes, assignment):
                # A southern patch is the northern one opposite it, in the
                # inverted crystal.
                south = bool(centre[2] < 0.0)
                sign = -1.0 if south else 1.0

                # The electrons leave along d, and the reciprocity waves come
                # in along -d: in a slab cut along -centre, then. Crystal frame
                # -> slab frame, and drop whatever this patch cannot
                # legitimately collect.
                slab_axis = -sign * centre
                local = -sign * directions[indices] @ zone_axis_rotation(slab_axis).T
                keep = local[:, 2] > cutoff
                local, indices = local[keep], indices[keep]

                if len(local) == 0:
                    continue

                arguments = dict(
                    slab_axis=slab_axis,
                    local=local,
                    origin=origin,
                    gpts=gpts,
                    max_batch_directions=max_batch_directions,
                )

                if lazy:
                    # One task per patch, run eagerly inside. Hundreds of
                    # patches already occupy the workers; splitting each into
                    # blocks as well, as a lone lazy scan does, repeats the
                    # per-slice work of every block for nothing: 1.7 times the
                    # cost, measured on 100 patches of 55 directions each.
                    task = dask.delayed(_patch_intensities, pure=True)(
                        self, blocks[south], **arguments
                    )
                    array = da.from_delayed(
                        task,
                        shape=ensemble_shape + (len(local),),
                        dtype=get_dtype(complex=False),
                    )
                else:
                    array = _patch_intensities(self, blocks[south], **arguments)

                progress.update_if_exists(1)

                patterns.append(
                    SphericalPattern(
                        array,
                        directions=directions[indices],
                        ensemble_axes_metadata=ensemble_axes,
                    )
                )
        finally:
            progress.close_if_exists()

        if not patterns:
            raise RuntimeError("no directions were assigned to any patch")

        pattern = SphericalPattern.concatenate(patterns)
        pattern.metadata.update(
            {
                **_energy_metadata(self._energy, energies, energy_weights),
                "label": "backscattered intensity",
                "n_patches": self._n_patches,
                "max_angle": self._max_angle,
                "sampling": self._sampling,
                "gpts": gpts,
                "projection": self._projection.name,
                "source": "uniform",
                "num_configs": (
                    1
                    if self._frozen_phonons is None
                    else self._frozen_phonons.num_configs
                ),
                "hemisphere": self._hemisphere,
                "centrosymmetric": self.centrosymmetric,
            }
        )

        return pattern
