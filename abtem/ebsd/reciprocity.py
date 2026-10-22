"""EBSD patterns from multislice reciprocity.

By reciprocity, the amplitude for an electron backscattered at
:math:`(\\mathbf{r}, z)` to leave in direction :math:`\\mathbf{k}` equals the
amplitude at :math:`(\\mathbf{r}, z)` of a plane wave travelling along
:math:`-\\mathbf{k}` that entered at the surface. One multislice calculation
per direction therefore covers every point where backscattering happens.

The collected intensity for direction :math:`\\mathbf{k}` is

.. math::
    I(\\mathbf{k}) = \\frac{\\sum_z w_z \\sum_{\\mathbf{r}}
    |\\psi_{\\mathbf{k}}(\\mathbf{r}, z)|^2 \\, s(\\mathbf{r}, z)}
    {\\sum_z w_z \\sum_{\\mathbf{r}} s(\\mathbf{r}, z)}

where :math:`\\psi_{\\mathbf{k}}` is the reciprocity wave, :math:`s` the emission
of the atoms (see :mod:`abtem.ebsd.emission`), and :math:`w_z` the weight of
depth :math:`z`. A specimen that does not scatter gives :math:`I = 1` in every
direction, so the values are yields relative to a featureless specimen.

The reciprocity waves are carried as periodic envelopes, in Bloch's form
:math:`e^{2\\pi i \\mathbf{k} \\cdot \\mathbf{r}} u(\\mathbf{r})`, with
:math:`u` periodic in the cell for any direction. The envelope is propagated
with the exact propagator evaluated at :math:`\\mathbf{q} + \\mathbf{k}`. For a
specimen periodic in the cell, every direction is then exact, and a wave's
direction uses none of the antialias aperture.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture, antialias_aperture
from abtem.array import validate_lazy
from abtem.core.axes import (
    AxisMetadata,
    BackscatterEnergyAxis,
    NonLinearAxis,
    OrdinalAxis,
    _normalized_axis_weights,
)
from abtem.core.backend import (
    asnumpy,
    device_name_from_array_module,
    get_array_module,
    validate_device,
)
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.complex import complex_exponential
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2wavelength
from abtem.core.grid import spatial_frequencies
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.distributions import BaseDistribution
from abtem.ebsd.detectors import BackscatterDetector
from abtem.ebsd.emission import EmissionSlices, configuration_atoms
from abtem.ebsd.measurements import SphericalPattern
from abtem.ebsd.sampling import AntialiasLossWarning, _warn_if_undersampled
from abtem.measurements import DiffractionPatterns
from abtem.multislice import FresnelPropagator, conventional_multislice_step
from abtem.potentials.iam import BasePotential, validate_potential
from abtem.waves import Waves

__all__ = ["EBSD", "AntialiasLossWarning"]


DepthWeight = float | Sequence[float] | np.ndarray | Callable[[np.ndarray], np.ndarray]


def _validate_depth_weight(
    depth_weight: Optional[DepthWeight], depths: np.ndarray
) -> np.ndarray:
    """The weight of each slice, normalized to sum to one."""
    if depth_weight is None:
        weights = np.ones(len(depths))
    elif callable(depth_weight):
        weights = np.asarray(depth_weight(depths), dtype=float)
    elif np.isscalar(depth_weight):
        escape_depth = float(depth_weight)  # type: ignore[arg-type]
        if escape_depth <= 0.0:
            raise ValueError(
                f"an escape depth must be positive, got {escape_depth}; give an "
                f"array to use arbitrary weights"
            )
        weights = np.exp(-depths / escape_depth)
    else:
        weights = np.asarray(depth_weight, dtype=float)

    if weights.shape != depths.shape:
        raise ValueError(
            f"depth_weight resolved to shape {weights.shape}, expected "
            f"{depths.shape} (one weight per slice)"
        )

    if np.any(weights < 0.0):
        raise ValueError("depth weights must not be negative")

    total = weights.sum()
    if total <= 0.0:
        raise ValueError("depth weights must not be all zero")

    return weights / total


def _takes_energy(function) -> bool:
    """Whether a depth-weight callable takes the backscattered energy too."""
    import inspect

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    required = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    return len(required) >= 2


def _depth_weights(
    depth_weight: Optional[DepthWeight], depths: np.ndarray, energies: np.ndarray
) -> np.ndarray:
    """The depth weights at each backscattered energy, ``(energies, slices)``."""
    if callable(depth_weight) and _takes_energy(depth_weight):
        return np.stack(
            [
                _validate_depth_weight(lambda z, e=e: depth_weight(z, e), depths)
                for e in energies
            ]
        )

    if not (
        depth_weight is None or callable(depth_weight) or np.isscalar(depth_weight)
    ):
        rows = np.asarray(depth_weight, dtype=float)
        if rows.ndim == 2:
            if len(rows) != len(energies):
                raise ValueError(
                    f"depth_weight has {len(rows)} rows but there are "
                    f"{len(energies)} backscattered energies"
                )
            return np.stack([_validate_depth_weight(row, depths) for row in rows])

    row = _validate_depth_weight(depth_weight, depths)
    return np.stack([row] * len(energies))


def _slice_centres(slice_thickness) -> np.ndarray:
    """The depth of each slice's centre [Å], the mean depth of its atoms.

    The depth weights and the depth bins both take it as the slice's depth.
    """
    thickness = np.asarray(slice_thickness, dtype=float)
    return np.cumsum(thickness) - thickness / 2.0


def _depth_bins(depth_bins, slice_thickness) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the depth bins [Å], and the bin of each slice (-1 if none)."""
    thickness = np.asarray(slice_thickness, dtype=float)
    centres = _slice_centres(thickness)

    if np.isscalar(depth_bins):
        n = int(depth_bins)
        if n < 1:
            raise ValueError(f"depth_bins must be at least 1, got {depth_bins}")
        edges = np.linspace(0.0, thickness.sum(), n + 1)
    else:
        edges = np.asarray(depth_bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError(
                "depth_bins must be a number of bins, or increasing bin edges [Å]"
            )

    index = np.searchsorted(edges, centres, side="right") - 1
    index[(centres < edges[0]) | (centres >= edges[-1])] = -1

    counts = np.bincount(index[index >= 0], minlength=len(edges) - 1)
    if np.any(counts == 0):
        raise ValueError(
            "every depth bin must hold at least one slice; use fewer or wider bins"
        )
    return edges, index


def _sum_dtype(xp) -> np.dtype:
    """The dtype of the sums over every atom and slice.

    Double precision, except on Metal, which supports single precision only.
    """
    if device_name_from_array_module(xp) == "mps":
        return np.dtype(get_dtype(complex=False))
    return np.dtype(np.float64)


def _energy_metadata(
    beam_energy: float,
    backscatter_energies: np.ndarray,
    energy_axis: Optional[BackscatterEnergyAxis],
) -> dict:
    """The energy metadata of a pattern.

    ``energy`` is the incident beam's, as everywhere in abTEM. A single
    backscattered energy is recorded as ``backscatter_energy``, and so is the
    weighted mean of energies averaged over; energies kept apart are the values
    of a :class:`~abtem.core.axes.BackscatterEnergyAxis` instead.
    """
    metadata: dict = {"energy": float(beam_energy)}
    if len(backscatter_energies) == 1:
        metadata["backscatter_energy"] = float(backscatter_energies[0])
    elif _averaged(energy_axis):
        weights = _energy_weights(energy_axis)
        metadata["backscatter_energy"] = float(weights @ backscatter_energies)
    return metadata


def _validate_backscatter_energy(
    backscatter_energy, beam_energy: float
) -> tuple[np.ndarray, Optional[BackscatterEnergyAxis]]:
    """Resolve the backscattered energies, at most `beam_energy`, and their axis.

    Several energies have an axis; one from a distribution keeps its weights
    and its ``ensemble_mean`` flag.
    """
    if backscatter_energy is None:
        return np.array([beam_energy]), None

    distribution = isinstance(backscatter_energy, BaseDistribution)
    if distribution:
        energies = np.asarray(backscatter_energy.values, dtype=float)
    else:
        energies = np.atleast_1d(np.asarray(backscatter_energy, dtype=float))

    if energies.ndim != 1 or len(energies) == 0:
        raise ValueError(
            f"backscatter_energy must be a scalar or a 1d sequence, got shape "
            f"{energies.shape}"
        )

    if np.any(energies <= 0.0):
        raise ValueError("backscattered energies must be positive")

    if np.any(energies > beam_energy):
        raise ValueError(
            f"a backscattered electron cannot carry more than the beam energy "
            f"of {beam_energy:.0f} eV; got up to {energies.max():.0f} eV"
        )

    if len(energies) == 1:
        return energies, None

    values = tuple(float(e) for e in energies)
    if not distribution:
        return energies, BackscatterEnergyAxis(values=values)

    axis = BackscatterEnergyAxis.from_distribution(backscatter_energy, values=values)
    if axis.weights is not None and sum(axis.weights) <= 0.0:
        raise ValueError("the energy weights must not sum to zero")
    return energies, axis


def _averaged(axis: Optional[BackscatterEnergyAxis]) -> bool:
    """Whether the backscattered energies are averaged over, as a distribution
    with ``ensemble_mean`` asks."""
    return axis is not None and axis._ensemble_mean


def _energy_weights(axis: BackscatterEnergyAxis) -> np.ndarray:
    """The normalized weights of the energies, as ``reduce_ensemble`` takes them."""
    weights = _normalized_axis_weights(axis, len(axis))
    return np.full(len(axis), 1.0 / len(axis)) if weights is None else weights


def _energy_mean(array, axis: BackscatterEnergyAxis):
    """The probability-weighted mean over the leading backscatter-energy axis.

    As ``reduce_ensemble``, which a :class:`SphericalPattern` does not have; the
    weights take the array's precision and device.
    """
    xp = get_array_module(array)
    weights = xp.asarray(_energy_weights(axis), dtype=np.finfo(array.dtype).dtype)
    return xp.sum(array * weights.reshape((-1,) + (1,) * (array.ndim - 1)), axis=0)


def _resolve_energy_axis(
    array, axis: Optional[BackscatterEnergyAxis], ensemble_axes_metadata: list
) -> tuple:
    """The leading energy axis of `array`: dropped for one energy, averaged over
    when its distribution asks for it, and kept otherwise."""
    if axis is None:
        return array[0], ensemble_axes_metadata
    if _averaged(axis):
        return _energy_mean(array, axis), ensemble_axes_metadata
    return array, [axis] + list(ensemble_axes_metadata)


def _worker_count() -> int:
    """Threads dask will run tasks on, as a target for the number of blocks."""
    from dask.system import CPU_COUNT

    return max(1, int(CPU_COUNT))


def _prepare(
    configuration: BasePotential,
    energy: float,
    illumination: Optional[np.ndarray],
    device: str,
) -> tuple[BasePotential, EmissionSlices]:
    """One configuration's built potential and its emitting atoms."""
    emission = EmissionSlices(
        configuration_atoms(configuration),
        configuration.slice_thickness,
        configuration.gpts,
        configuration.extent,
        energy,
        illumination=illumination,
        device=device,
    )
    return configuration.build(lazy=False), emission


def _potential_configurations(
    potential: BasePotential,
) -> list[tuple[BasePotential, float]]:
    """Split a potential into its frozen-phonon configurations and their weights.

    The configurations are averaged with equal weight, since the yield is an
    incoherent sum over them; an ensemble asking to keep them separate is
    warned about.

    Returns
    -------
    configurations : list of (BasePotential, float)
        Single-configuration potentials and the weight each carries. A
        potential with no ensemble axis gives the single pair
        ``[(potential, 1.0)]``.
    """
    ensemble_shape = potential.ensemble_shape

    if len(ensemble_shape) == 0:
        return [(potential, 1.0)]

    if any(
        not getattr(axis, "_ensemble_mean", True)
        for axis in potential.ensemble_axes_metadata
    ):
        warnings.warn(
            "the potential asks to keep its configurations separate, but a "
            "backscattered yield is an incoherent sum over them; they are "
            "averaged",
            UserWarning,
        )

    configurations = [block.ravel()[0] for _, _, block in potential.generate_blocks(1)]
    weight = 1.0 / len(configurations)

    return [(configuration, weight) for configuration in configurations]


def _propagate_block_intensities(
    ebsd: "EBSD", prepared: tuple, arguments: dict
) -> np.ndarray:
    """One block of directions, as a dask task."""
    return asnumpy(ebsd._propagate_block(prepared=prepared, **arguments))


def _envelope_propagator_array(
    wave_vectors, gpts: tuple[int, int], sampling, energy: float, thickness: float, xp
):
    """Exact free-space propagators for periodic envelopes, one per wave.

    The envelope ``u`` of ``exp(2 pi i k.r) u`` holds at frequency ``q`` what the
    wave holds at ``q + k``. It is propagated with the exact propagator there,
    relative to the carrier's phase, and band-limited by the antialias aperture.
    """
    wavelength = energy2wavelength(energy)
    qx, qy = spatial_frequencies(gpts, sampling, xp=xp)
    aperture = antialias_aperture(gpts, sampling, xp)

    kernel = xp.empty((len(wave_vectors),) + tuple(gpts), dtype=get_dtype(complex=True))
    # a few waves at a time, to keep the temporaries small
    for start in range(0, len(wave_vectors), 64):
        k = wave_vectors[start : start + 64]
        kx = k[:, 0, None, None] + qx[None, :, None]
        ky = k[:, 1, None, None] + qy[None, None, :]
        x = wavelength**2 * (kx**2 + ky**2)
        x_carrier = wavelength**2 * (k[:, 0] ** 2 + k[:, 1] ** 2)[:, None, None]

        # evanescent components are dropped; the aperture removes them anyway
        propagating = x < 1.0
        root = xp.sqrt(xp.where(propagating, 1.0 - x, 0.0))
        # sqrt(1 - x) - sqrt(1 - x_carrier), without the cancellation between them
        phase = (2.0 * np.pi * thickness / wavelength) * (
            (x_carrier - x) / (root + xp.sqrt(1.0 - x_carrier))
        )
        kernel[start : start + 64] = complex_exponential(phase) * (
            propagating * aperture[None]
        )

    return kernel


class _EnvelopePropagator(FresnelPropagator):
    """abTEM's Fresnel propagator with the kernel for periodic envelopes."""

    def __init__(self, wave_vectors):
        super().__init__()
        self._wave_vectors = wave_vectors
        self._thickness: Optional[float] = None
        self._kernel = None

    def get_array(self, waves: Waves, thickness: float, order="exact"):
        # One kernel at a time, as _direction_blocks budgets: a thinner last
        # slice replaces it rather than adding a second.
        if self._kernel is None or thickness != self._thickness:
            self._kernel = None
            self._kernel = _envelope_propagator_array(
                self._wave_vectors,
                waves._valid_gpts,
                waves._valid_sampling,
                waves._valid_energy,
                thickness,
                get_array_module(waves.device),
            )
            self._thickness = thickness
        return self._kernel


class EBSD(CopyMixin, EqualityMixin):
    """Backscatter diffraction patterns from one slab, by reciprocity.

    The engine the pattern builders run in each slab. It takes a slab and
    directions in the slab's frame; for a pattern on a detector, see
    :class:`~abtem.ebsd.detector_pattern.EBSDDetectorPattern`.

    Parameters
    ----------
    potential : BasePotential or Atoms
        The specimen: ``Atoms``, :class:`~abtem.FrozenPhonons`, or a
        :class:`~abtem.Potential` of either. A ``PotentialArray`` does not
        carry the atoms, which are the emitters. The reciprocity waves travel
        along ``z`` (see :func:`abtem.ebsd.rotated_slab`). Frozen phonons smear
        the emitters as well as the scattering.
    detector : BackscatterDetector
        The directions of the reciprocity waves in the slab's frame, ``+z``
        pointing into the slab. Each gives the electrons leaving in the
        opposite direction, so to collect electrons leaving a crystal along
        ``d``, cut the slab along ``-d``.
    energy : float
        Energy of the incident beam [eV]. Sets the cross-sections, and the
        reciprocity waves' energy unless `backscatter_energy` is given.
    illumination : np.ndarray, optional
        How brightly the incident beam lights each atom of the potential, of
        shape ``(atoms,)``, or ``(sources, atoms)`` for several, which adds a
        leading axis. The values are for the atoms as given; a potential that
        transforms them, for example by orthogonalizing their cell, lights each
        copy as the atom it came from. Only relative values matter. By default
        every atom is lit alike, as in EMsoft.
    depth_weight : float or array or callable, optional
        How much the electrons generated at each depth count. The depth of a
        slice is that of its centre, the mean depth of its atoms, as for
        `depth_bins`. A float weights depth ``z`` by ``exp(-z / escape_depth)``
        [Å]; an array gives one weight per slice, and a callable maps the
        slices' depths to weights. With several
        `backscatter_energy`, the weights may depend on the energy: a callable
        of ``(depths, energy)``, or one row per energy, as from a Monte Carlo
        simulation. Normalized to sum to one. By default every depth counts
        equally.
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energies of the backscattered electrons [eV], at most `energy`, at
        which the reciprocity waves travel. Defaults to `energy`. Several add a
        leading :class:`~abtem.core.axes.BackscatterEnergyAxis`. A distribution
        keeps its weights on the axis, and one with ``ensemble_mean=True``, as
        :func:`abtem.distributions.gaussian` makes by default, is averaged over
        with them instead; :meth:`~abtem.measurements.BaseMeasurements.reduce_ensemble`
        averages a kept axis afterwards.

        ``metadata["energy"]`` is the beam's energy, as everywhere in abTEM, and
        ``metadata["backscatter_energy"]`` that of the backscattered electrons
        when there is one, or their weighted mean when they are averaged over.
        A :class:`.DiffractionPatterns` is calibrated with the beam energy,
        which gives the detector grid's angles at every backscattered energy.
    depth_tolerance : float, optional
        Stop propagating once the depth weight still to be collected falls
        below this fraction of the total (default 1e-4); the error in a
        direction can be a few times larger. Set to 0 to disable.
    depth_bins : int or array, optional
        Resolve the pattern by depth: a number of equal bins, or bin edges
        [Å]. Adds a ``depth`` axis after any backscatter-energy axis. The
        emission of each bin is recorded as ``metadata["depth_emission"]``, so
        a depth weighting ``W`` can be applied afterwards::

            E = np.array(patterns.metadata["depth_emission"])
            pattern = np.tensordot(W * E, patterns.array, (0, 0)) / (W * E).sum()

        Slices outside the edges are left out.
    device : str, optional
        'cpu', 'gpu' or 'mps'. Defaults to the user configuration.
    """

    def __init__(
        self,
        potential: BasePotential | Atoms,
        detector: BackscatterDetector,
        energy: float,
        illumination: Optional[np.ndarray] = None,
        depth_weight: Optional[DepthWeight] = None,
        backscatter_energy: Optional[
            float | Sequence[float] | np.ndarray | BaseDistribution
        ] = None,
        depth_tolerance: float = 1e-4,
        depth_bins: Optional[int | Sequence[float] | np.ndarray] = None,
        device: Optional[str] = None,
    ):
        potential = validate_potential(potential)
        frozen_phonons = getattr(potential, "frozen_phonons", None)
        if frozen_phonons is None:
            raise ValueError(
                f"the potential has to be built from atoms (Atoms, FrozenPhonons "
                f"or a Potential of either), since they emit the backscattered "
                f"electrons; a {type(potential).__name__} does not carry its atoms"
            )

        if illumination is not None:
            illumination = np.atleast_1d(np.asarray(illumination, dtype=float))
            if illumination.ndim > 2:
                raise ValueError(
                    f"an illumination must have shape (atoms,) or (sources, atoms), "
                    f"got {illumination.shape}"
                )
            if illumination.shape[-1] != len(frozen_phonons.atoms):
                raise ValueError(
                    f"an illumination needs one value per atom of the potential "
                    f"({len(frozen_phonons.atoms)}), got {illumination.shape[-1]}"
                )

        if potential.sampling is not None:
            _warn_if_undersampled(
                frozen_phonons.atoms, max(potential.sampling), "the potential's"
            )

        self._potential = potential
        self._detector = detector
        self._energy = float(energy)
        self._illumination = illumination
        self._depth_weight = depth_weight
        self._backscatter_energy = backscatter_energy
        self._depth_tolerance = float(depth_tolerance)
        self._depth_bins = depth_bins
        self._device = validate_device(device)

    @property
    def potential(self) -> BasePotential:
        """The specimen potential."""
        return self._potential

    @property
    def energy(self) -> float:
        """Energy of the incident beam [eV]."""
        return self._energy

    @property
    def detector(self) -> BackscatterDetector:
        """The collected directions."""
        return self._detector

    @property
    def illumination(self) -> Optional[np.ndarray]:
        """How brightly each atom is lit, or None for all alike."""
        return self._illumination

    @property
    def device(self) -> str:
        """Device the calculation runs on."""
        return self._device

    def _direction_blocks(
        self, max_batch: int | str, lazy: bool = False
    ) -> list[tuple[int, int]]:
        """Split the directions into batches that fit in memory.

        When lazy, the batches are split further to give every worker a task.
        """
        gpts = self._potential.gpts
        if gpts is None:
            raise RuntimeError("the potential has no grid")

        # each direction holds its wave and its own propagator kernel
        if isinstance(max_batch, int):
            max_batch = max_batch * 2 * int(np.prod(gpts))

        chunks = validate_chunks(
            shape=(len(self._detector), 2) + tuple(gpts),
            chunks=("auto", -1, -1, -1),
            max_elements=max_batch,
            dtype=np.dtype(get_dtype(complex=True)),
            device=self._device,
        )
        blocks = list(chunk_ranges(chunks)[0])

        if lazy and len(blocks) < _worker_count():
            size = int(np.ceil(len(self._detector) / _worker_count()))
            size = min(size, max(stop - start for start, stop in blocks))
            blocks = [
                (start, min(start + size, len(self._detector)))
                for start in range(0, len(self._detector), size)
            ]

        return blocks

    def build(
        self,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = False,
    ) -> DiffractionPatterns | SphericalPattern:
        """Calculate the backscattered intensity in every collected direction.

        Parameters
        ----------
        max_batch_directions : int or str, optional
            Number of directions propagated at once. 'auto' (default) picks a
            batch from the abTEM chunk-size configuration. Larger batches are
            faster but use more memory.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph. Defaults to the
            abTEM configuration.
        pbar : bool, optional
            If True, show a progress bar over the slices.

        Returns
        -------
        patterns : DiffractionPatterns or SphericalPattern
            :class:`~abtem.measurements.DiffractionPatterns` if the detector is
            a grid, otherwise a
            :class:`~abtem.ebsd.measurements.SphericalPattern` holding one
            value per direction. Leading axes are backscattered energy, depth
            and illumination, where present.
        """
        lazy = validate_lazy(lazy)

        xp = get_array_module(self._device)

        # the configurations are built where they are used
        num_slices = self._potential.num_slices
        configurations = _potential_configurations(self._potential)

        energy = self._energy
        backscatter_energies, energy_axis = _validate_backscatter_energy(
            self._backscatter_energy, energy
        )
        n_energies = len(backscatter_energies)

        depths = _slice_centres(self._potential.slice_thickness)
        weights = _depth_weights(self._depth_weight, depths, backscatter_energies)
        if (
            self._depth_bins is not None
            and _averaged(energy_axis)
            and not np.allclose(weights, weights[0])
        ):
            raise ValueError(
                "the depth bins of energies averaged over can only be recombined "
                "with one depth weighting for all of them; give the energies "
                "with ensemble_mean=False to keep them apart"
            )

        if self._depth_bins is None:
            edges, slice_bins, depth_shape = None, np.zeros(num_slices, int), ()
        else:
            edges, slice_bins = _depth_bins(
                self._depth_bins, self._potential.slice_thickness
            )
            depth_shape = (len(edges) - 1,)

        illumination = self._illumination
        if illumination is not None and illumination.ndim == 2:
            output_shape: tuple[int, ...] = (len(illumination),)
            ensemble_axes_metadata: list[AxisMetadata] = [
                OrdinalAxis(
                    label="illumination",
                    values=tuple(range(len(illumination))),
                )
            ]
        else:
            output_shape = ()
            ensemble_axes_metadata = []

        blocks = self._direction_blocks(max_batch_directions, lazy=lazy)

        # on the host, once per energy; each block moves its own slice to the
        # device, so that a lazy graph holds only NumPy arrays
        wave_vectors = [
            np.asarray(
                self._detector.transverse_wave_vectors(backscatter_energy),
                dtype=get_dtype(complex=False),
            )
            for backscatter_energy in backscatter_energies
        ]

        def block_arguments(i, start, stop):
            return dict(
                wave_vectors=wave_vectors[i][start:stop],
                weights=weights[i],
                slice_bins=slice_bins,
                depth_shape=depth_shape,
                backscatter_energy=float(backscatter_energies[i]),
                n_directions=stop - start,
                output_shape=output_shape,
            )

        # axes: backscattered energy, depth, then the illuminations
        if depth_shape:
            centres = (edges[1:] + edges[:-1]) / 2.0
            ensemble_axes_metadata = [
                NonLinearAxis(
                    label="depth", values=tuple(float(z) for z in centres), units="Å"
                )
            ] + ensemble_axes_metadata
        extra_metadata = (
            {}
            if edges is None
            else self._depth_metadata(
                configurations,
                weights[:1] if _averaged(energy_axis) else weights,
                edges,
                slice_bins,
                output_shape,
                energy,
            )
        )

        if lazy:
            return self._lazy_measurement(
                configurations=configurations,
                block_arguments=block_arguments,
                blocks=blocks,
                backscatter_energies=backscatter_energies,
                energy_axis=energy_axis,
                output_shape=output_shape,
                ensemble_axes_metadata=ensemble_axes_metadata,
                energy=energy,
                block_shape=depth_shape + output_shape,
                extra_metadata=extra_metadata,
            )

        intensities = xp.zeros(
            (n_energies,) + depth_shape + output_shape + (len(self._detector),),
            dtype=get_dtype(complex=False),
        )

        progress = TqdmWrapper(
            total=len(configurations) * n_energies * len(blocks) * num_slices,
            enabled=pbar,
            leave=False,
        )
        try:
            for configuration, configuration_weight in configurations:
                prepared = _prepare(configuration, energy, illumination, self._device)
                for i in range(n_energies):
                    for start, stop in blocks:
                        block = self._propagate_block(
                            prepared=prepared,
                            progress=progress,
                            **block_arguments(i, start, stop),
                        )
                        intensities[i][..., start:stop] += configuration_weight * block
        finally:
            progress.close_if_exists()

        intensities, ensemble_axes_metadata = _resolve_energy_axis(
            intensities, energy_axis, ensemble_axes_metadata
        )
        return self._to_measurement(
            intensities,
            ensemble_axes_metadata=ensemble_axes_metadata,
            backscatter_energies=backscatter_energies,
            energy_axis=energy_axis,
            extra_metadata=extra_metadata,
        )

    def _depth_metadata(
        self, configurations, weights, edges, slice_bins, output_shape, energy
    ) -> dict:
        """The bin edges and each bin's emission, for recombining the bins."""
        n_bins = len(edges) - 1
        emission = np.zeros((len(weights), n_bins) + output_shape)
        for configuration, configuration_weight in configurations:
            totals = EmissionSlices(
                configuration_atoms(configuration),
                configuration.slice_thickness,
                configuration.gpts,
                configuration.extent,
                energy,
                illumination=self._illumination,
            ).totals()  # (slices, sources)
            for i, row in enumerate(weights):
                for b in range(n_bins):
                    inside = slice_bins == b
                    emission[i, b] += configuration_weight * (
                        row[inside] @ totals[inside]
                    ).reshape(output_shape)

        if len(weights) == 1:
            emission = emission[0]
        return {
            "depth_bins": [float(z) for z in edges],
            "depth_emission": emission.tolist(),
        }

    def _lazy_measurement(
        self,
        configurations,
        block_arguments,
        blocks,
        backscatter_energies,
        energy_axis,
        output_shape,
        ensemble_axes_metadata,
        energy: float,
        block_shape: tuple[int, ...] = (),
        extra_metadata: Optional[dict] = None,
    ) -> DiffractionPatterns | SphericalPattern:
        """The same computation as a dask graph.

        One task per block of directions, energy and configuration; the blocks
        of a configuration share its potential and emitters.
        """
        import dask

        n_energies = len(backscatter_energies)

        array = None
        for configuration, configuration_weight in configurations:
            prepared = dask.delayed(_prepare, pure=True, nout=2)(
                configuration, energy, self._illumination, self._device
            )

            rows = []
            for i in range(n_energies):
                columns = []
                for start, stop in blocks:
                    block = dask.delayed(_propagate_block_intensities, pure=True)(
                        self, prepared, block_arguments(i, start, stop)
                    )
                    columns.append(
                        da.from_delayed(
                            block,
                            shape=block_shape + (stop - start,),
                            dtype=get_dtype(complex=False),
                        )
                    )
                rows.append(da.concatenate(columns, axis=-1))

            stacked = configuration_weight * da.stack(rows, axis=0)
            array = stacked if array is None else array + stacked

        array, ensemble_axes_metadata = _resolve_energy_axis(
            array, energy_axis, ensemble_axes_metadata
        )

        if self._detector.is_grid:
            # the base axes have to be one chunk
            array = array.rechunk(array.chunks[:-1] + (-1,))

        return self._to_measurement(
            array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            backscatter_energies=backscatter_energies,
            energy_axis=energy_axis,
            extra_metadata=extra_metadata,
        )

    def _propagate_block(
        self,
        prepared: tuple,
        wave_vectors,
        weights: np.ndarray,
        slice_bins: np.ndarray,
        depth_shape: tuple[int, ...],
        backscatter_energy: float,
        n_directions: int,
        output_shape: tuple[int, ...],
        progress: Optional[TqdmWrapper] = None,
    ) -> np.ndarray:
        """Propagate one batch of reciprocity waves and collect the emission.

        At every slice, the emission of its atoms is weighted by the waves'
        intensity and the depth weight. Numerator and denominator are summed
        over the slab, per depth bin, and divided once.
        """
        potential, emission = prepared
        xp = get_array_module(self._device)
        wave_vectors = xp.asarray(wave_vectors, dtype=get_dtype(complex=False))

        # the periodic envelope of a plane wave is one (see the module docstring)
        reciprocity = Waves(
            xp.ones(
                (n_directions,) + tuple(potential.gpts), dtype=get_dtype(complex=True)
            ),
            energy=backscatter_energy,
            extent=potential.extent,
            ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_directions)))],
        )

        n_sources = int(np.prod(output_shape)) if output_shape else 1
        n_bins = depth_shape[0] if depth_shape else 1
        sum_dtype = _sum_dtype(xp)
        numerator = xp.zeros((n_bins, n_sources, n_directions), dtype=sum_dtype)
        denominator = xp.zeros((n_bins, n_sources), dtype=sum_dtype)

        # weight still to be collected at and below each slice
        collected = weights * (slice_bins >= 0)
        remaining = np.cumsum(collected[::-1])[::-1]
        tolerance = self._depth_tolerance * collected.sum()

        propagator = _EnvelopePropagator(wave_vectors)
        antialias_aperture = AntialiasAperture()

        for index, potential_slice in enumerate(potential.generate_slices()):
            if remaining[index] < tolerance:
                break

            potential_slice = potential_slice.copy_to_device(self._device)
            transmission = antialias_aperture.bandlimit(
                potential_slice.transmission_function(energy=backscatter_energy),
                in_place=True,
            )

            # read before the step, at the plane the slice's atoms are placed on
            weight = float(weights[index])
            b = int(slice_bins[index])
            emitted = emission[index] if weight > 0.0 and b >= 0 else None

            if emitted is not None:
                overlap = self._overlap(reciprocity, emitted, n_directions)
                numerator[b] += weight * overlap.T
                denominator[b] += weight * xp.sum(
                    emitted, axis=(-2, -1), dtype=sum_dtype
                )

            reciprocity = conventional_multislice_step(
                reciprocity,
                transmission,
                propagator=propagator,
                antialias_aperture=antialias_aperture,
            )

            if progress is not None:
                progress.update_if_exists(1)

        # no emission gives zero rather than 0/0
        safe = xp.where(denominator > 0.0, denominator, 1.0)
        intensities = xp.where(
            denominator[..., None] > 0.0, numerator / safe[..., None], 0.0
        ).astype(get_dtype(complex=False))

        return intensities.reshape(depth_shape + output_shape + (n_directions,))

    def _overlap(self, reciprocity: Waves, source, n_directions: int):
        """Sum of |reciprocity|^2 * source over the pixels, per direction and source.

        Returns an array of shape ``(directions, sources)``.
        """
        xp = get_array_module(self._device)

        # (directions, pixels) @ (pixels, sources) -> (directions, sources)
        if device_name_from_array_module(xp) == "mps":
            # torch tensors cannot be viewed through a NumPy dtype
            array = reciprocity.array.reshape(n_directions, -1)
            intensity = (array.real * array.real) + (array.imag * array.imag)
            return intensity @ source.reshape(-1, intensity.shape[1]).T

        # |psi|^2 without abs(): square the interleaved real and imaginary parts
        # and contract them against the source repeated twice
        real_view = reciprocity.array.view(get_dtype(complex=False))
        real_view = real_view.reshape(n_directions, -1)

        # a one-dimensional repeat; cupy's repeat along an axis is very slow
        source_flat = source.reshape(-1, real_view.shape[1] // 2)
        source_interleaved = xp.repeat(source_flat.ravel(), 2).reshape(
            source_flat.shape[0], -1
        )

        return (real_view * real_view) @ source_interleaved.T

    def _to_measurement(
        self,
        intensities,
        ensemble_axes_metadata: list[AxisMetadata],
        backscatter_energies: np.ndarray,
        energy_axis: Optional[BackscatterEnergyAxis],
        extra_metadata: Optional[dict] = None,
    ) -> DiffractionPatterns | SphericalPattern:
        """Wrap the collected intensities in the matching measurement type."""
        array: np.ndarray | da.core.Array
        if isinstance(intensities, da.core.Array):
            array = intensities
        else:
            array = asnumpy(intensities)
        metadata = {
            **_energy_metadata(self._energy, backscatter_energies, energy_axis),
            "label": "backscattered intensity",
            "source": "uniform" if self._illumination is None else "illumination",
            **(extra_metadata or {}),
        }

        if self._detector.is_grid:
            # The grid is the same in angle at every backscattered energy, so
            # calibrating it with the beam energy, metadata["energy"], gives the
            # right angles for all of them.
            gpts = self._detector.gpts
            return DiffractionPatterns(
                array.reshape(array.shape[:-1] + (gpts, gpts)),
                sampling=self._detector.reciprocal_sampling(self._energy),
                fftshift=True,
                ensemble_axes_metadata=ensemble_axes_metadata,
                metadata=metadata,
            )

        return SphericalPattern(
            array,
            directions=self._detector.directions,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )
