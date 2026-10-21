"""An EBSD pattern on a detector, calculated directly for one orientation.

Instead of interpolating a reference pattern, the directions the detector
pixels look along are calculated directly. An orientation also fixes the
direction of the incident beam, so the atoms can be lit by it.
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
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.distributions import BaseDistribution
from abtem.ebsd.detectors import BackscatterDetector
from abtem.ebsd.emission import configuration_atoms
from abtem.ebsd.geometry import EBSDGeometry, bunge_rotation
from abtem.ebsd.measurements import EBSDPatternImages, _detector_to_image
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

# Margin between the incident beam's slab faces and the atoms the patches use [Å]
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

    Returns an array of shape ``(sources, len(local))``.
    """
    slab, rotation = rotated_slab(
        block, slab_axis, builder.slab_cell, repetitions=(1, 1, 1), origin=origin
    )

    if illumination is not None:
        illumination = illumination[:, slab.arrays["block_index"]]

    # A coarse sampling is warned about once, by the builder.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntialiasLossWarning)
        result = EBSD(
            Potential(
                _displaced(slab, rotation, builder._frozen_phonons),
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

    Takes the same detector geometry and Euler angles as
    :meth:`~abtem.ebsd.measurements.SphericalPattern.project`, and returns the
    same kind of image, but calculates each pixel's direction directly.

    The pixels are tiled into patches as a reference pattern tiles the
    hemisphere: each pixel belongs to its nearest of ``n_patches`` zone axes
    and is calculated in a slab cut along it. Pixels looking into the southern
    hemisphere are calculated in patches of their own.

    :meth:`build` lights every atom alike, which gives the reference pattern at
    those pixels. :meth:`scan` lights them with an incident
    :class:`~abtem.PlaneWave` or :class:`~abtem.Probe`, which can be scanned.

    Parameters
    ----------
    atoms : ase.Atoms or FrozenPhonons
        The crystal's unit cell, or an atomic model of the specimen, in its own
        frame. Give :class:`~abtem.FrozenPhonons` for thermal vibrations; every
        slab, including the incident beam's, is then averaged over the
        configurations.
    geometry : EBSDGeometry
        The detector geometry, in EMsoft's conventions. The incident beam comes
        in along the microscope axis, at `geometry.sample_tilt` to the sample
        normal.
    euler : sequence of three float
        Bunge Euler angles ``(phi1, Phi, phi2)`` of the crystal.
    slab_cell : tuple of three float
        Dimensions of each slab [Å]; the third entry is the thickness along the
        slab's axis. See :class:`~abtem.ebsd.reference.EBSDReferencePattern`.
    energy : float
        Energy of the incident beam [eV].
    n_patches : int or None, optional
        Zone axes tiling each hemisphere, as for a reference pattern (default
        400). Only the patches the pixels fall in are calculated. If None,
        every pixel is calculated in one slab along the mean pixel direction,
        which is cheaper but less accurate at the edges of a wide detector.
    degrees : bool, optional
        If True (default), `euler` is in degrees.
    sampling : float, optional
        Real-space sampling of the potentials [Å]. Defaults to
        :func:`~abtem.ebsd.sampling.potential_sampling` for `atoms`.
    slice_thickness : float, optional
        Multislice slice thickness [Å] (default 1.0).
    repetitions : tuple of three int, optional
        Repetitions of `atoms` in the block every slab is cut from; see
        :func:`~abtem.ebsd.orientations.rotated_slab`.
    origin : np.ndarray, optional
        Point of `atoms` at the centre of every slab, including the incident
        beam's.
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energies of the backscattered electrons [eV]; several add a leading
        energy axis. See :class:`~abtem.ebsd.reciprocity.EBSD`.
    depth_weight, device :
        Passed to :class:`~abtem.ebsd.reciprocity.EBSD`.
    """

    def __init__(
        self,
        atoms: Atoms | FrozenPhonons,
        geometry: EBSDGeometry,
        euler: Sequence[float],
        slab_cell: tuple[float, float, float],
        energy: float,
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

        if energy is None:
            raise ValueError("give the energy of the incident beam [eV]")

        self._atoms, self._frozen_phonons = _crystal_and_displacements(atoms)
        self._geometry = geometry
        self._euler = euler
        self._degrees = bool(degrees)
        self._slab_cell = slab_cell
        self._energy = float(energy)
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

        In the sample frame it is ``(sin σ, 0, -cos σ)`` for a sample tilt σ.
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

        Large enough to hold every patch's slab, at any orientation, with a
        margin to its own faces.
        """
        diagonal = float(np.linalg.norm(self._slab_cell))
        width = diagonal + 2.0 * _INCIDENT_MARGIN
        return (width, width, diagonal)

    def _block_and_origin(self, lit: bool = False):
        """The block every slab is cut from, its atoms tagged, and the anchor.

        A block lit by an incident beam also holds the beam's slab.
        """
        cell = self.incident_cell if lit else self._slab_cell
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

    def _illumination(
        self, source: PlaneWave | Probe, block: Atoms, origin, scan
    ) -> np.ndarray:
        """How brightly `source` lights every atom of the block.

        Returns an array of shape ``(sources, block atoms)``, one row per member
        of the source's ensemble and probe position, normalized to a mean of one
        within each slice, and the ensemble axes and shape of the sources.
        """
        xp = get_array_module(self._device)
        slab, rotation = rotated_slab(
            block,
            self.beam_direction,
            self.incident_cell,
            repetitions=(1, 1, 1),
            origin=origin,
        )

        potential = Potential(
            _displaced(slab, rotation, self._frozen_phonons),
            gpts=fft_friendly_gpts(self.incident_cell[:2], self._sampling),
            slice_thickness=self._slice_thickness,
            projection="finite",
            device=self._device,
        )

        wave = source.copy()
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

        # averaged over the configurations, each lighting its displaced atoms
        lit = np.zeros((n_sources, len(block)))
        for configuration, weight in _potential_configurations(potential):
            sliced = SliceIndexedAtoms(
                configuration_atoms(configuration), configuration.slice_thickness
            )
            beam = incident.copy()

            for index, potential_slice in enumerate(configuration.generate_slices()):
                potential_slice = potential_slice.copy_to_device(self._device)

                # the beam arriving at the slice, as in EBSD
                atoms = sliced.get_atoms_in_slices(index)
                if len(atoms) > 0:
                    intensity = asnumpy(
                        xp.abs(beam.array.reshape((-1,) + beam.shape[-2:])) ** 2
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

        # the atoms outside the incident slab are in no patch
        illumination = np.ones((n_sources, len(block)))
        inside = slab.arrays["block_index"]
        illumination[:, inside] = lit[:, inside]

        return illumination, axes, shape

    def _validate_source(self, source) -> PlaneWave | Probe:
        """A copy of `source` at the beam energy."""
        if not isinstance(source, (PlaneWave, Probe)):
            raise ValueError(
                f"source must be a PlaneWave or a Probe, got {type(source).__name__}"
            )
        source = source.copy()
        if source.energy is None:
            source.energy = self._energy
        elif isinstance(source.energy, BaseDistribution):
            raise ValueError(
                "the incident beam has one energy, but the source has an energy "
                "ensemble; give backscatter_energy for the backscattered electrons"
            )
        elif not np.isclose(float(source.energy), self._energy):
            raise ValueError(
                f"the source's energy ({float(source.energy):g} eV) differs from the "
                f"beam energy ({self._energy:g} eV)"
            )
        return source

    def build(
        self,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = False,
    ) -> EBSDPatternImages:
        """Calculate the detector's pixels with every atom lit alike.

        Parameters
        ----------
        max_batch_directions : int or str, optional
            Directions propagated at once; passed to
            :meth:`~abtem.ebsd.reciprocity.EBSD.build`.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph with one task per
            patch. Defaults to the abTEM configuration.
        pbar : bool, optional
            If True, show a progress bar over the patches.

        Returns
        -------
        pattern : EBSDPatternImages
            Of shape ``(columns, rows)`` of ``geometry.shape`` (see
            :class:`~abtem.ebsd.measurements.EBSDPatternImages`), after an
            energy axis if there are several backscattered energies.
        """
        return self._calculate(None, None, max_batch_directions, lazy, pbar)

    def scan(
        self,
        source: PlaneWave | Probe,
        scan: Optional[BaseScan | Sequence] = None,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = False,
    ) -> EBSDPatternImages:
        """Calculate the detector's pixels with the atoms lit by an incident beam.

        The source is propagated along the beam direction through a slab of its
        own, cut about the same `origin`, and lights each atom by its intensity
        there, normalized within each slice; `depth_weight` sets how much each
        depth counts.

        Parameters
        ----------
        source : PlaneWave or Probe
            The incident beam. Its energy, if given, must be the beam energy;
            its grid is matched to the incident slab. Ensemble axes of the
            source, such as a defocus series, are kept.
        scan : BaseScan or array of xy-positions, optional
            Positions of a probe [Å], measured from `origin`. Each gives a
            pattern.
        max_batch_directions : int or str, optional
            Directions propagated at once; passed to
            :meth:`~abtem.ebsd.reciprocity.EBSD.build`.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph with one task per
            patch; the incident beam is calculated first. Defaults to the abTEM
            configuration.
        pbar : bool, optional
            If True, show a progress bar over the patches.

        Returns
        -------
        pattern : EBSDPatternImages
            Of shape ``(columns, rows)`` of ``geometry.shape`` (see
            :class:`~abtem.ebsd.measurements.EBSDPatternImages`), after an
            energy axis if there are several backscattered energies, then the
            source's ensemble axes and the scan's.
        """
        source = self._validate_source(source)
        if scan is not None and not isinstance(source, Probe):
            raise ValueError(
                "only a probe source depends on where the probe is, so every "
                "position would give the same pattern; give a Probe as the source"
            )
        return self._calculate(source, scan, max_batch_directions, lazy, pbar)

    def _calculate(
        self,
        source: Optional[PlaneWave | Probe],
        scan: Optional[BaseScan | Sequence],
        max_batch_directions: int | str,
        lazy: Optional[bool],
        pbar: bool,
    ) -> EBSDPatternImages:
        lazy = validate_lazy(lazy)

        rows, columns = self._geometry.shape
        directions = self.directions.reshape(-1, 3)
        axes, owner = self._assignment()
        block, origin = self._block_and_origin(lit=source is not None)
        gpts = self.potential_gpts

        illumination, ensemble_axes, output_shape = None, [], ()
        if source is not None:
            shifted = None
            if scan is not None:
                if not isinstance(scan, BaseScan):
                    scan = CustomScan(scan)
                # measured from origin, which is the incident slab's centre
                centre = np.array(self.incident_cell[:2]) / 2.0
                positions = np.asarray(scan.get_positions()).reshape(-1, 2)
                shifted = CustomScan(positions + centre)

            illumination, ensemble_axes, output_shape = self._illumination(
                source, block, origin, shifted
            )

            if scan is not None:
                # the source's own axes, then the scan's, shaped and labelled as
                # the scan was given rather than as it was run
                ensemble_axes = list(source.ensemble_axes_metadata) + list(
                    scan.ensemble_axes_metadata
                )
                output_shape = tuple(source.ensemble_shape) + tuple(scan.ensemble_shape)

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

            # one node each in the graph, shared by every patch
            block_argument = dask.delayed(block)
            if illumination is not None:
                illumination = dask.delayed(illumination)
        else:
            block_argument = block

        pieces, order = [], []
        progress = TqdmWrapper(total=len(axes), enabled=pbar and not lazy, leave=False)
        try:
            for j, axis in enumerate(axes):
                indices = np.where(owner == j)[0]
                # electrons leaving along d: reciprocity waves along -d
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
                        self, block_argument, **arguments
                    )
                    pieces.append(
                        da.from_delayed(
                            task,
                            shape=(n_sources, len(indices)),
                            dtype=get_dtype(complex=False),
                        )
                    )
                else:
                    pieces.append(_detector_patch(self, block_argument, **arguments))
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
            # the base axes have to be one chunk
            array = array.rechunk(array.chunks[:-1] + (-1,))
        array = _detector_to_image(array.reshape(output_shape + (rows, columns)))

        metadata = {
            **_energy_metadata(self._energy, energies, energy_weights),
            "label": "backscattered intensity",
            "source": "uniform" if source is None else type(source).__name__,
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
