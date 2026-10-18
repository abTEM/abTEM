"""An EBSD pattern on a detector, calculated directly for one orientation.

:class:`~abtem.ebsd.reference.EBSDReferencePattern` calculates the whole
hemisphere, and :meth:`~abtem.ebsd.measurements.SphericalPattern.project` then
reads any detector out of it. When one orientation is all that is wanted -- a
defect, a strained region, a disordered specimen -- that is most of the
hemisphere calculated for nothing, and the detector's pixels are interpolated
from a grid usually coarser than they are. This goes the other way round: it
works out which directions the pixels look along in the crystal, and calculates
exactly those.

An orientation also fixes where the beam comes from, which a reference pattern
cannot know. So here the atoms can be lit as the beam really lights them.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture
from abtem.array import validate_lazy
from abtem.core.axes import EnergyAxis
from abtem.core.backend import get_array_module
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.ebsd.detectors import BackscatterDetector
from abtem.ebsd.emission import configuration_atoms
from abtem.ebsd.geometry import EBSDGeometry, bunge_rotation
from abtem.ebsd.measurements import EBSDPatternImages
from abtem.ebsd.orientations import (
    _crystal_and_displacements,
    _displaced,
    bulk_block,
    central_origin,
    estimate_repetitions,
    fibonacci_hemisphere,
    rotated_slab,
    zone_axis_rotation,
)
from abtem.ebsd.reciprocity import (
    EBSD,
    DepthWeight,
    _energy_metadata,
    _potential_configurations,
    _validate_backscatter_energy,
)
from abtem.ebsd.reference import fft_friendly_gpts
from abtem.ebsd.sampling import (
    AntialiasLossWarning,
    _warn_if_undersampled,
    potential_sampling,
)
from abtem.inelastic.phonons import FrozenPhonons
from abtem.multislice import FresnelPropagator, conventional_multislice_step
from abtem.potentials.iam import Potential
from abtem.scan import BaseScan, CustomScan
from abtem.slicing import SliceIndexedAtoms
from abtem.waves import PlaneWave, Probe

__all__ = ["EBSDDetectorPattern"]

# Kept between the incident beam's slab and every atom the patches use, so that
# the seams of that non-periodic slab stay clear of them [Å].
_INCIDENT_MARGIN = 10.0


def _detector_patch(
    builder: "EBSDDetectorPattern",
    block: Atoms,
    slab_axis: np.ndarray,
    local: np.ndarray,
    origin: Optional[np.ndarray],
    gpts: tuple[int, int],
    max_batch_directions: int | str,
    illumination: Optional[np.ndarray],
) -> np.ndarray:
    """Cut one patch's slab and calculate its pixels, eagerly.

    Returns one row per source, whether or not the source had an axis of its
    own, of one value per direction of `local`.
    """
    slab, _ = rotated_slab(
        block, slab_axis, builder.slab_cell, repetitions=(1, 1, 1), origin=origin
    )

    if illumination is not None:
        illumination = illumination[:, slab.arrays["block_index"]]

    # A coarse sampling is warned about once, by the builder.
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
            illumination=illumination,
            depth_weight=builder._depth_weight,
            backscatter_energy=builder._backscatter_energy,
            device=builder._device,
        ).build(max_batch_directions=max_batch_directions, lazy=False)

    return np.asarray(result.array).reshape(-1, len(local))


class EBSDDetectorPattern(CopyMixin, EqualityMixin):
    """One orientation's EBSD pattern on a detector, calculated by reciprocity.

    The single-orientation counterpart of
    :class:`~abtem.ebsd.reference.EBSDReferencePattern`: it takes the same
    detector geometry and Euler angles as
    :meth:`~abtem.ebsd.measurements.SphericalPattern.project`, and returns the
    same kind of image, but calculates each pixel's direction itself.

    The multislice is only accurate near a slab's axis, so the pixels are tiled
    into patches as a reference pattern tiles the hemisphere: each belongs to
    its nearest of ``n_patches`` zone axes, and is calculated in a slab cut
    along it. With the same `n_patches` and slab, and the uniform source, the
    result *is* the reference pattern at those pixels. Pixels looking into the
    crystal's southern hemisphere get patches of their own, along the negated
    axes, rather than an assumption that the crystal is centrosymmetric.

    Parameters
    ----------
    atoms : ase.Atoms or FrozenPhonons
        The crystal's unit cell, or an atomic model of the specimen, in its own
        frame. The Euler angles take the sample frame to this one. Give
        :class:`~abtem.FrozenPhonons` of either for thermal vibrations: every
        slab -- the incident beam's too -- is displaced configuration by
        configuration, and the pattern averaged over them, as for
        :class:`~abtem.ebsd.reference.EBSDReferencePattern`.
    geometry : EBSDGeometry
        Where the detector is, in EMsoft's conventions -- and where the beam
        comes in: along the microscope axis, at `geometry.sample_tilt` to the
        sample normal.
    euler : sequence of three float
        Bunge Euler angles ``(phi1, Phi, phi2)`` of the crystal.
    slab_cell : tuple of three float
        Dimensions of each slab [Å]; the third entry is the thickness along the
        beam. As for a reference pattern there is no default -- see
        :class:`~abtem.ebsd.reference.EBSDReferencePattern`.
    energy : float, optional
        Energy of the incident beam [eV]. Taken from the source when that is a
        wave, and required otherwise.
    source : 'uniform' or PlaneWave or Probe, optional
        How the incident beam lights the emitting atoms.

        ``'uniform'`` (default): evenly, as in a reference pattern and EMsoft.

        A :class:`~abtem.PlaneWave` or :class:`~abtem.Probe`: sent in along the
        beam's true direction, through a slab of its own cut from the same
        crystal about the same `origin`, recording how brightly it lights every
        atom the patches use. Channelling of the incident beam then picks out
        some atoms over others -- one sublattice over another in silicon, say.
        The depth at which an atom sits below that slab's entrance face is not
        its depth below the real surface, so the illumination is normalized
        within each of its slices: the beam decides which atoms are lit, and
        `depth_weight` how deep they count. A plane wave is what an SEM beam is
        on this scale; a probe can be scanned (see :meth:`build`).

        Both waves travel into the crystal as they do in the experiment: the
        beam down the microscope axis, and each pixel's reciprocity wave from
        the detector's side, against the electrons it collects -- in a slab
        cut along the opposite of the pixel's direction. So a crystal without
        an inversion centre shows each polar face where it really is.
    n_patches : int or None, optional
        Zone axes tiling each hemisphere, as for a reference pattern (default
        400, a patch radius of 132 mrad). Only the patches the pixels fall in
        are calculated. None calculates everything in one slab along the mean
        pixel direction -- cheaper, and fine for a narrow detector, but the
        edges of a wide one come out dim.
    degrees : bool, optional
        If True (default), `euler` is in degrees.
    sampling : float, optional
        Real-space sampling of the potentials [Å]. Defaults to
        :func:`~abtem.ebsd.reference.potential_sampling` for `atoms`; the
        angles of the pixels set no condition of their own.
    slice_thickness : float, optional
        Multislice slice thickness [Å] (default 1.0).
    repetitions : tuple of three int, optional
        Repetitions of `atoms` building the block every slab is cut from; see
        :func:`~abtem.ebsd.orientations.rotated_slab`. Pass ``(1, 1, 1)`` for
        an atomic model that already holds the feature of interest.
    origin : np.ndarray, optional
        Point of `atoms` at the centre of every slab -- the point every slab
        turns about, and that the incident beam passes through.
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energies of the backscattered electrons [eV]; several add a leading
        energy axis, one detector image per energy. The beam keeps `energy`.
        See :class:`~abtem.ebsd.reciprocity.EBSD`.
    depth_weight, device :
        Passed to :class:`~abtem.ebsd.reciprocity.EBSD`.
    """

    def __init__(
        self,
        atoms: Atoms | FrozenPhonons,
        geometry: EBSDGeometry,
        euler: Sequence[float],
        slab_cell: tuple[float, float, float],
        energy: Optional[float] = None,
        source: str | PlaneWave | Probe = "uniform",
        n_patches: Optional[int] = 400,
        degrees: bool = True,
        sampling: Optional[float] = None,
        slice_thickness: float = 1.0,
        repetitions: Optional[tuple[int, int, int]] = None,
        origin: Optional[np.ndarray] = None,
        backscatter_energy=None,
        depth_weight: Optional[DepthWeight] = None,
        device: Optional[str] = None,
    ):
        slab_cell = tuple(float(x) for x in slab_cell)
        if len(slab_cell) != 3 or min(slab_cell) <= 0.0:
            raise ValueError(
                f"slab_cell must be three positive lengths [Å], got {slab_cell}"
            )

        euler = np.asarray(euler, dtype=float)
        if euler.shape != (3,):
            raise ValueError(
                f"euler must be one orientation, (phi1, Phi, phi2), got shape "
                f"{euler.shape}; for several, build one pattern per orientation"
            )

        if n_patches is not None and int(n_patches) < 1:
            raise ValueError(f"n_patches must be at least 1, got {n_patches}")

        if isinstance(source, (PlaneWave, Probe)):
            if energy is not None and float(energy) != float(source.energy):
                raise ValueError(
                    f"energy {energy} eV contradicts the source's {source.energy} "
                    f"eV; leave energy out"
                )
            energy = source.energy
        elif source != "uniform":
            raise ValueError(
                f"source must be 'uniform', a PlaneWave or a Probe, got {source!r}"
            )

        if energy is None:
            raise ValueError("give the beam energy, or a wave to take it from")

        self._atoms, self._frozen_phonons = _crystal_and_displacements(atoms)
        self._geometry = geometry
        self._euler = euler
        self._degrees = bool(degrees)
        self._slab_cell = slab_cell
        self._energy = float(energy)
        self._source = source
        self._n_patches = None if n_patches is None else int(n_patches)
        self._slice_thickness = float(slice_thickness)
        self._repetitions = repetitions
        self._origin = origin
        self._backscatter_energy = backscatter_energy
        self._depth_weight = depth_weight
        self._device = device

        if sampling is None:
            self._sampling = potential_sampling(self._atoms)
        else:
            self._sampling = float(sampling)
            _warn_if_undersampled(self._atoms, self._sampling, "a")

    @property
    def atoms(self) -> Atoms:
        """The crystal or atomic model, undisplaced."""
        return self._atoms

    @property
    def frozen_phonons(self) -> Optional[FrozenPhonons]:
        """Its thermal displacements, if any."""
        return self._frozen_phonons

    @property
    def geometry(self) -> EBSDGeometry:
        """Where the detector is."""
        return self._geometry

    @property
    def euler(self) -> np.ndarray:
        """Bunge Euler angles of the crystal."""
        return self._euler

    @property
    def slab_cell(self) -> tuple[float, float, float]:
        """Dimensions of each slab [Å]."""
        return self._slab_cell

    @property
    def energy(self) -> float:
        """Energy of the incident beam [eV]."""
        return self._energy

    @property
    def source(self):
        """How the atoms are lit: 'uniform', or the wave sent in to light them."""
        return self._source

    @property
    def n_patches(self) -> Optional[int]:
        """Zone axes tiling each hemisphere, or None for a single slab."""
        return self._n_patches

    @property
    def sampling(self) -> float:
        """Real-space sampling of the potentials [Å]."""
        return self._sampling

    @property
    def directions(self) -> np.ndarray:
        """The direction each pixel looks along, in the crystal frame.

        Of shape ``(rows, columns, 3)``.
        """
        return self._geometry.rotated_directions(self._euler, degrees=self._degrees)

    @property
    def beam_direction(self) -> np.ndarray:
        """The direction the incident beam travels, in the crystal frame.

        Down the microscope axis, which in the sample frame is
        ``(sin σ, 0, -cos σ)`` for a sample tilted by ``σ``: into the surface,
        towards the detector's side.
        """
        tilt = np.radians(self._geometry.sample_tilt)
        beam = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
        return bunge_rotation(self._euler, degrees=self._degrees) @ beam

    def _assignment(self) -> tuple[np.ndarray, np.ndarray]:
        """The zone axes in use, and which of them each pixel belongs to."""
        directions = self.directions.reshape(-1, 3)

        if self._n_patches is None:
            mean = directions.mean(axis=0)
            return (mean / np.linalg.norm(mean))[None], np.zeros(len(directions), int)

        northern = fibonacci_hemisphere(self._n_patches)
        candidates = np.concatenate([northern, -northern])
        nearest = np.argmax(directions @ candidates.T, axis=1)

        used, owner = np.unique(nearest, return_inverse=True)
        return candidates[used], owner.reshape(-1)

    @property
    def zone_axes(self) -> np.ndarray:
        """The axis of each slab calculated, in the crystal frame, ``(M, 3)``."""
        return self._assignment()[0]

    @property
    def max_angle(self) -> float:
        """Largest angle between any pixel and its own slab's axis [mrad]."""
        axes, owner = self._assignment()
        cosines = np.sum(self.directions.reshape(-1, 3) * axes[owner], axis=1)
        return float(np.arccos(np.clip(cosines.min(), -1.0, 1.0)) * 1e3)

    @property
    def potential_gpts(self) -> tuple[int, int]:
        """Grid the patch potentials are built on, at a size the FFT likes."""
        return fft_friendly_gpts(self._slab_cell[:2], self._sampling)

    @property
    def incident_cell(self) -> tuple[float, float, float]:
        """The slab the incident beam is sent through [Å].

        Wide and deep enough to hold every atom of every patch: the sphere the
        patches' slabs sweep out as they turn about `origin`, plus a margin
        that keeps the seams of this slab's own non-periodic faces clear of it.
        """
        diagonal = float(np.linalg.norm(self._slab_cell))
        width = diagonal + 2.0 * _INCIDENT_MARGIN
        return (width, width, diagonal)

    def _block_and_origin(self):
        """The block every slab is cut from, its atoms tagged, and the anchor."""
        cell = self._slab_cell if isinstance(self._source, str) else self.incident_cell
        repetitions = (
            estimate_repetitions(self._atoms, cell)
            if self._repetitions is None
            else self._repetitions
        )
        block = bulk_block(self._atoms, cell, repetitions)
        block.set_array("block_index", np.arange(len(block)))

        origin = (
            None
            if self._origin is None
            else central_origin(self._atoms, repetitions, self._origin)
        )
        return block, origin

    def _illumination(self, block: Atoms, origin, scan) -> np.ndarray:
        """How brightly the incident beam lights every atom of the block.

        Returns an array of shape ``(sources, block atoms)``, one row per probe
        position, normalized to a mean of one over the atoms in each of the
        incident slab's slices -- with the axes and shape of the positions.
        """
        xp = get_array_module(self._device)
        # The beam travels along its own direction into this slab.
        slab, _ = rotated_slab(
            block,
            self.beam_direction,
            self.incident_cell,
            repetitions=(1, 1, 1),
            origin=origin,
        )

        potential = Potential(
            _displaced(slab, self._frozen_phonons),
            gpts=fft_friendly_gpts(self.incident_cell[:2], self._sampling),
            slice_thickness=self._slice_thickness,
            projection="finite",
            device=self._device,
        )

        wave = self._source.copy()
        wave.grid.match(potential)
        if isinstance(wave, Probe):
            incident = wave.build(scan=scan, lazy=False)
        else:
            incident = wave.build(lazy=False)
        incident = incident.copy_to_device(self._device)
        shape = incident.shape[:-2]
        axes = list(incident.ensemble_axes_metadata)
        n_sources = int(np.prod(shape)) if shape else 1

        sampling = np.array(potential.sampling)
        propagator = FresnelPropagator()
        aperture = AntialiasAperture()

        from scipy.ndimage import map_coordinates  # type: ignore[import-untyped]

        # Each thermal configuration lights its own displaced atoms, and an
        # atom's illumination is the average over them.
        lit = np.zeros((n_sources, len(block)))
        for configuration, weight in _potential_configurations(potential):
            sliced = SliceIndexedAtoms(
                configuration_atoms(configuration), configuration.slice_thickness
            )
            beam = incident.copy()

            for index, potential_slice in enumerate(configuration.generate_slices()):
                potential_slice = potential_slice.copy_to_device(self._device)

                # As in EBSD: the atoms are lit by the beam arriving at their slice.
                atoms = sliced.get_atoms_in_slices(index)
                if len(atoms) > 0:
                    intensity = xp.abs(beam.array.reshape((-1,) + beam.shape[-2:])) ** 2
                    intensity = np.asarray(
                        intensity.get() if hasattr(intensity, "get") else intensity
                    )
                    coordinates = (atoms.positions[:, :2] / sampling).T

                    for i, image in enumerate(intensity):
                        values = map_coordinates(
                            image, coordinates, order=1, mode="grid-wrap"
                        )
                        lit[i, atoms.arrays["block_index"]] += (
                            weight * values / values.mean()
                        )

                transmission = aperture.bandlimit(
                    potential_slice.transmission_function(energy=self._energy),
                    in_place=True,
                )
                beam = conventional_multislice_step(
                    beam,
                    transmission,
                    propagator=propagator,
                    antialias_aperture=aperture,
                )

        # Atoms of the block outside the incident slab are in no patch either.
        illumination = np.ones((n_sources, len(block)))
        inside = slab.arrays["block_index"]
        illumination[:, inside] = lit[:, inside]

        return illumination, axes, shape

    def build(
        self,
        scan: Optional[BaseScan | Sequence] = None,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = False,
    ) -> EBSDPatternImages:
        """Calculate the detector's pixels.

        Parameters
        ----------
        scan : BaseScan or array of xy-positions, optional
            Positions of a probe source [Å], across the incident beam and
            measured from `origin`: one pattern each, as leading axes of the
            result.
        max_batch_directions : int or str, optional
            Directions propagated at once; passed to
            :meth:`~abtem.ebsd.reciprocity.EBSD.build`.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph, with one task per
            patch. The incident beam, which every patch shares, is calculated
            first. Defaults to the abTEM configuration.
        pbar : bool, optional
            If True, show a progress bar over the patches.

        Returns
        -------
        pattern : EBSDPatternImages
            Of shape ``geometry.shape``, with leading axes for any probe
            positions -- the same kind of image
            :meth:`~abtem.ebsd.measurements.SphericalPattern.project` returns.
        """
        lazy = validate_lazy(lazy)

        if scan is not None and not isinstance(self._source, Probe):
            raise ValueError(
                "only a probe source depends on where the probe is, so every "
                "position would give the same pattern; give a Probe as the source"
            )

        rows, columns = self._geometry.shape
        directions = self.directions.reshape(-1, 3)
        axes, owner = self._assignment()
        block, origin = self._block_and_origin()
        gpts = self.potential_gpts

        illumination, ensemble_axes, output_shape = None, [], ()
        if not isinstance(self._source, str):
            shifted = None
            if scan is not None:
                if not isinstance(scan, BaseScan):
                    scan = CustomScan(scan)
                # measured from origin, which is the incident slab's centre
                centre = np.array(self.incident_cell[:2]) / 2.0
                positions = np.asarray(scan.get_positions()).reshape(-1, 2)
                shifted = CustomScan(positions + centre)

            illumination, ensemble_axes, output_shape = self._illumination(
                block, origin, shifted
            )

            if scan is not None:
                # shaped and labelled as the scan was given, not as it was run
                ensemble_axes = list(scan.ensemble_axes_metadata)
                output_shape = tuple(scan.ensemble_shape)

        energies, energy_weights = _validate_backscatter_energy(
            self._backscatter_energy, self._energy
        )
        if len(energies) > 1:
            output_shape = (len(energies),) + tuple(output_shape)
            ensemble_axes = [
                EnergyAxis(values=tuple(float(e) for e in energies))
            ] + list(ensemble_axes)

        n_sources = int(np.prod(output_shape)) if output_shape else 1

        if lazy:
            import dask

            shared_block = dask.delayed(block)

        pieces, order = [], []
        progress = TqdmWrapper(total=len(axes), enabled=pbar and not lazy, leave=False)
        try:
            for j, axis in enumerate(axes):
                indices = np.where(owner == j)[0]
                # the reciprocity waves come in against the electrons, along
                # -d, in a slab cut along -axis
                local = -directions[indices] @ zone_axis_rotation(-axis).T
                arguments = dict(
                    slab_axis=-axis,
                    local=local,
                    origin=origin,
                    gpts=gpts,
                    max_batch_directions=max_batch_directions,
                    illumination=illumination,
                )
                if lazy:
                    task = dask.delayed(_detector_patch, pure=True)(
                        self, shared_block, **arguments
                    )
                    pieces.append(
                        da.from_delayed(
                            task,
                            shape=(n_sources, len(indices)),
                            dtype=get_dtype(complex=False),
                        )
                    )
                else:
                    pieces.append(_detector_patch(self, block, **arguments))
                order.append(indices)
                progress.update_if_exists(1)
        finally:
            progress.close_if_exists()

        # back from patch order to pixel order
        inverse = np.argsort(np.concatenate(order))
        array = (
            da.concatenate(pieces, axis=-1) if lazy else np.concatenate(pieces, axis=-1)
        )
        array = array[..., inverse]

        if isinstance(array, da.core.Array):
            # The pixels become the two base axes, which have to be one chunk.
            array = array.rechunk(array.chunks[:-1] + (-1,))
        array = array.reshape(output_shape + (rows, columns))

        metadata = {
            **_energy_metadata(self._energy, energies, energy_weights),
            "label": "backscattered intensity",
            "source": (
                self._source
                if isinstance(self._source, str)
                else type(self._source).__name__
            ),
            "euler": [float(a) for a in self._euler],
            "n_patches": self._n_patches,
            "num_configs": (
                1 if self._frozen_phonons is None else self._frozen_phonons.num_configs
            ),
            "max_angle": self.max_angle,
            "sampling": self._sampling,
        }

        return EBSDPatternImages(
            array,
            sampling=self._geometry.pixel_size,
            ensemble_axes_metadata=ensemble_axes,
            metadata=metadata,
        )
