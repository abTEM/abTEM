"""Reference patterns: EBSD intensity over the whole hemisphere.

A reciprocity calculation is accurate for directions near its slab's axis, so
:class:`EBSDReferencePattern` tiles the hemisphere into patches, calculates
each in its own slab, and joins the results in the crystal frame.
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

#: Density of the direction grid relative to the output image.
_DIRECTION_OVERSAMPLING = 1.15


def fft_friendly_gpts(extent: tuple[float, float], sampling: float) -> tuple[int, int]:
    """Grid points covering `extent` at `sampling` or finer, sized for the FFT.

    Rounds the number of grid points up to the next size with only small
    prime factors, for which the FFT is fast.

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

    Each of `n_patches` points of a Fibonacci lattice has a cell of solid angle
    :math:`2\\pi / n`; the radius is that of a disc through the corners of a
    regular hexagon of that area, enlarged by `coverage`.

    Parameters
    ----------
    n_patches : int
        Number of patches tiling the hemisphere.
    coverage : float, optional
        Factor by which the covering radius is enlarged (default 1.2).

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
    """Cut one patch's slab and calculate its directions, eagerly.

    Returns one value per direction of `local`, after an energy axis if there
    are several backscattered energies.
    """
    slab, _ = rotated_slab(
        block, slab_axis, builder.slab_cell, repetitions=(1, 1, 1), origin=origin
    )

    # the builder warns about a coarse sampling once
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
        for thermal vibrations: every slab is displaced as the unit cell would
        be, and the pattern is averaged over the configurations, at
        `num_configs` times the cost.
    energy : float
        Energy of the incident beam [eV].
    slab_cell : tuple of three float
        Dimensions of the slab cut for each patch [Å]; the third entry is the
        thickness along the slab's axis. There is no default, since the right
        size depends on the crystal and the accuracy needed.

        The bands build up over the first ~100 Å, so 60-100 Å is a typical
        thickness. The width controls the seams: a wave at an angle ``θ`` to
        the slab's axis moves ``θ T`` sideways over a thickness ``T`` and then
        meets the slab's non-periodic edge. 30-40 Å wide slabs keep this error
        to a few percent. To make a run cheaper, lower `gpts` rather than the
        slab size.
    n_patches : int, optional
        Number of zone-axis patches tiling each hemisphere (default 400). More
        patches keep every direction closer to its slab's axis.
    gpts : int, optional
        Pixels along each axis of a projected image (default 128). Sets the
        default density of the sampled directions.
    projection : str or HemisphereProjection, optional
        Projection whose even grid the directions are drawn from, ``'lambert'``
        (default) or ``'stereographic'``. The equal-area Lambert grid samples
        the hemisphere uniformly and can be interpolated to either projection.
    sampling : float, optional
        Real-space sampling of the potentials [Å]. Defaults to
        :func:`~abtem.ebsd.sampling.potential_sampling` for `atoms`.
    slice_thickness : float, optional
        Multislice slice thickness [Å] (default 1.0).
    direction_gpts : int, optional
        Number of points per axis of the grid the directions are drawn from.
        Defaults to ``ceil(1.15 * gpts)``.
    max_angle : float, optional
        Angular radius of each patch [mrad]; directions farther than this from
        their patch's axis are left out. Defaults to :func:`patch_half_angle`.
    repetitions : tuple of three int, optional
        Repetitions of `atoms` in the block the slabs are cut from; see
        :func:`~abtem.ebsd.orientations.rotated_slab`.
    origin : np.ndarray, optional
        Point of `atoms` to place at the centre of every slab, passed to
        :func:`rotated_slab`. Defaults to the centroid.
    hemisphere : {'north', 'south', 'both'}, optional
        Which directions to calculate (default 'north'). For a centrosymmetric
        crystal the other hemisphere is the inversion image, which the pattern
        provides (:attr:`SphericalPattern.southern`). For a crystal without an
        inversion centre, such as GaN, calculate 'both'. The southern
        directions are ``(x, y, -z)`` below the northern ones, as in EMsoft.
    centrosymmetric : bool, optional
        Whether the crystal has an inversion centre. Recorded in the pattern;
        it decides whether the inversion image stands in for the hemisphere not
        calculated without a warning. If None (default), it is determined with
        spglib when first needed, which can be slow or fail for a large cell
        without lattice symmetry.
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energies of the backscattered electrons [eV]; several add a leading
        energy axis. See :class:`~abtem.ebsd.reciprocity.EBSD`.
    depth_weight, device :
        Passed to :class:`~abtem.ebsd.reciprocity.EBSD`.

    Notes
    -----
    Every atom emits by its backscattering cross-section and is lit evenly, as
    in EMsoft (see :mod:`abtem.ebsd.emission`).

    A direction ``d`` is where the backscattered electrons go. By reciprocity
    it is calculated with a plane wave travelling along ``-d``, so each patch
    is calculated in a slab cut along the opposite of its centre.

    The intensity leaving a crystal along ``-d`` equals the intensity leaving
    its inversion image along ``d``. The southern hemisphere is therefore
    calculated as the northern hemisphere of the crystal inverted about the
    slab origin. For a crystal with an inversion centre at the origin the two
    calculations are identical.
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
        """Whether the crystal has an inversion centre, so ``I(-k) = I(k)``."""
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

        The southern centres are the negated northern ones; see Notes.
        """
        northern = fibonacci_hemisphere(self._n_patches)
        return self._in_hemisphere(northern, -northern)

    @property
    def directions(self) -> np.ndarray:
        """Every sampled direction in the crystal frame, of shape ``(N, 3)``.

        The southern ones are ``(x, y, -z)`` of the northern ones, with -0.0 on
        the equator.
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

        Rounded up from :attr:`sampling` by :func:`fft_friendly_gpts`, so the
        actual sampling is slightly finer.
        """
        return fft_friendly_gpts(self._slab_cell[:2], self._sampling)

    def _assign_directions(self) -> list[np.ndarray]:
        """Indices of the directions each patch calculates: its nearest ones."""
        zone_axes = self.zone_axes
        nearest = np.argmax(self.directions @ zone_axes.T, axis=1)
        return [np.where(nearest == j)[0] for j in range(len(zone_axes))]

    def build(
        self,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = True,
    ) -> SphericalPattern:
        """Calculate every patch and join the results into one pattern.

        Parameters
        ----------
        max_batch_directions : int or str, optional
            Directions propagated at once within each patch. Passed to
            :meth:`~abtem.ebsd.reciprocity.EBSD.build`.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph with one task per
            patch. Defaults to the abTEM configuration.
        pbar : bool, optional
            If True (default), show a progress bar over the patches. Ignored
            when lazy.

        Returns
        -------
        pattern : SphericalPattern
            Backscattered intensity at every sampled direction, in the crystal
            frame.
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

        origin = (
            None
            if self._origin is None
            else central_origin(self._atoms, repetitions, self._origin)
        )

        # the southern patches are calculated in the inverted crystal (see Notes)
        blocks = {False: block}
        if self._hemisphere != "north":
            centre_of_inversion = (
                np.mean(block.positions, axis=0) if origin is None else origin
            )
            blocks[True] = block.copy()
            blocks[True].positions = 2.0 * centre_of_inversion - block.positions

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

            blocks = {south: dask.delayed(b) for south, b in blocks.items()}

        progress = TqdmWrapper(
            total=len(zone_axes), enabled=pbar and not lazy, leave=False
        )
        try:
            for centre, indices in zip(zone_axes, assignment):
                south = bool(centre[2] < 0.0)
                sign = -1.0 if south else 1.0

                # electrons leaving along d: reciprocity waves along -d
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
                    # one task per patch; the patches already fill the workers
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
