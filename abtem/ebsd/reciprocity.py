"""EBSD patterns from multislice reciprocity.

The physical process is an electron entering the specimen, scattering
inelastically somewhere inside it, and leaving through the surface in some
direction. Simulating that directly would mean a separate calculation for every
scattering site. Reciprocity turns it around: the amplitude for an electron
generated at :math:`(\\mathbf{r}, z)` to leave in direction :math:`\\mathbf{k}`
equals the amplitude at :math:`(\\mathbf{r}, z)` of a plane wave travelling in
:math:`-\\mathbf{k}` that entered at the surface. One multislice run per
collected direction therefore covers every generation site at once.

The collected intensity for direction :math:`\\mathbf{k}` is

.. math::
    I(\\mathbf{k}) = \\frac{\\sum_z w_z \\sum_{\\mathbf{r}}
    |\\psi_{\\mathbf{k}}(\\mathbf{r}, z)|^2 \\, s(\\mathbf{r}, z)}
    {\\sum_z w_z \\sum_{\\mathbf{r}} s(\\mathbf{r}, z)}

where :math:`\\psi_{\\mathbf{k}}` is the reciprocity plane wave, :math:`s` is
where backscatter is generated, and :math:`w_z` is the probability that an
electron generated at depth :math:`z` escapes. The sums are incoherent: the
generation events are independent, so intensities add rather than amplitudes.

The source :math:`s` is the atoms (see :mod:`abtem.ebsd.emission`): each emits
in proportion to its cross-section for scattering straight back, as lit by the
incident beam. The ratio is the emission-weighted mean of the escape
probability, normalized once for the whole slab -- so a specimen that does not
scatter the plane waves gives :math:`I = 1` in every direction, the values are
yields relative to a featureless specimen, and nothing depends on how the atoms
fall into slices.

The reciprocity waves are carried as periodic envelopes. The multislice cell is
periodic, and a plane wave :math:`e^{2\\pi i \\mathbf{k} \\cdot \\mathbf{r}}`
fits it only when :math:`\\mathbf{k}` falls on the cell's reciprocal grid; in any
other direction it jumps in phase at the cell edge, and diffracts off the jump.
Written as :math:`e^{2\\pi i \\mathbf{k} \\cdot \\mathbf{r}} u(\\mathbf{r})`
-- Bloch's form -- the wave is carried by :math:`u`, which for a plane wave is
one everywhere, periodic whatever the direction. The transmission function
multiplies :math:`u` as it multiplies the wave, the propagator is the exact one
evaluated at :math:`\\mathbf{q} + \\mathbf{k}`, and the carrier has unit
modulus, so :math:`|u|^2` is the intensity. For a specimen that is periodic in
the cell -- a crystal, or a supercell holding a defect -- every direction is
then exact, and not only those on the grid; and a wave's direction no longer
uses up any of the antialias aperture, which limits only the scattering about it.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture, antialias_aperture
from abtem.array import validate_lazy
from abtem.core.axes import AxisMetadata, EnergyAxis, NonLinearAxis, OrdinalAxis
from abtem.core.backend import get_array_module, validate_device
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
    """Resolve the escape-probability weight of each slice.

    The weights are normalized to sum to one, so the result is always a
    weighted average over depth and does not change scale with the number of
    slices or the thickness.
    """
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
    """The depth weights at each backscattered energy, ``(energies, slices)``.

    An electron that lost more energy has, on the whole, come from deeper, so
    the weights may depend on the energy: a callable of ``(depths, energy)``,
    or one row of weights per energy.
    """
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


def _depth_bins(depth_bins, slice_thickness) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the depth bins [Å], and the bin of each slice (-1 if none)."""
    thickness = np.asarray(slice_thickness, dtype=float)
    centres = np.cumsum(thickness) - thickness / 2.0

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


def _energy_metadata(
    beam_energy: float, backscatter_energies: np.ndarray, energy_weights: np.ndarray
) -> dict:
    """The energies a pattern records, the same way for every builder.

    ``beam_energy`` is the incident beam's. ``energy`` is abTEM's: that of the
    electrons the pattern counts, the backscattered ones. With several of them
    the pattern's :class:`.EnergyAxis` carries them and ``energy`` is left
    unset, as abTEM leaves it for every energy ensemble; their weights are
    recorded rather than applied, since summing the axis is the consumer's
    business and the weights usually come from a Monte Carlo spectrum.
    """
    metadata: dict = {"beam_energy": float(beam_energy)}
    if len(backscatter_energies) == 1:
        metadata["energy"] = float(backscatter_energies[0])
    else:
        metadata["backscatter_energies"] = [float(e) for e in backscatter_energies]
        metadata["energy_weights"] = [float(w) for w in energy_weights]
    return metadata


def _validate_backscatter_energy(
    backscatter_energy, beam_energy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the backscattered energies and their weights.

    The beam travels at `beam_energy`; a backscattered electron has lost some
    of that, so the reciprocity waves belong at or below it.
    """
    if backscatter_energy is None:
        return np.array([beam_energy]), np.array([1.0])

    if isinstance(backscatter_energy, BaseDistribution):
        energies = np.asarray(backscatter_energy.values, dtype=float)
        weights = np.asarray(backscatter_energy.weights, dtype=float)
    else:
        energies = np.atleast_1d(np.asarray(backscatter_energy, dtype=float))
        weights = np.ones(len(energies))

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

    total = weights.sum()
    if total <= 0.0:
        raise ValueError("the energy weights must not sum to zero")

    return energies, weights / total


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
    """One configuration's potential and its emitting atoms, as one task.

    The blocks of directions of a configuration all share them, and dask frees
    them once the blocks are done.
    """
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

    A potential built on :class:`~abtem.FrozenPhonons` or
    :class:`~abtem.AtomsEnsemble` carries an ensemble axis, and the propagation
    below runs on one configuration at a time -- so the axis has to be taken
    apart here rather than left for ``generate_slices``, which would silently
    walk the first configuration only.

    The configurations are averaged with equal weight, which is what a
    backscattered yield is: an incoherent sum over the thermal displacements
    the specimen passes through, not a coherent one. A configuration axis is
    therefore never carried through to the measurement, and an ensemble asked
    to keep one says so.

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
    intensities = ebsd._propagate_block(prepared=prepared, **arguments)
    return np.asarray(intensities.get() if hasattr(intensities, "get") else intensities)


def _envelope_propagator_array(
    wave_vectors, gpts: tuple[int, int], sampling, energy: float, thickness: float, xp
):
    """Exact free-space propagators for periodic envelopes, one per wave.

    The envelope ``u`` of ``exp(2 pi i k.r) u`` holds at frequency ``q`` what the
    wave holds at ``q + k``, so it propagates with the exact propagator there --
    relative to the carrier's own phase, so that ``u = 1`` stays one in vacuum.
    Band-limited in ``q`` by the antialias aperture, as abTEM's propagator is.
    """
    wavelength = energy2wavelength(energy)
    qx, qy = spatial_frequencies(gpts, sampling, xp=xp)
    aperture = antialias_aperture(gpts, sampling, xp)

    kernel = xp.empty((len(wave_vectors),) + tuple(gpts), dtype=get_dtype(complex=True))
    # A few waves at a time, to keep the temporaries small.
    for start in range(0, len(wave_vectors), 64):
        k = wave_vectors[start : start + 64]
        kx = k[:, 0, None, None] + qx[None, :, None]
        ky = k[:, 1, None, None] + qy[None, None, :]
        x = wavelength**2 * (kx**2 + ky**2)
        x_carrier = wavelength**2 * (k[:, 0] ** 2 + k[:, 1] ** 2)[:, None, None]

        # Evanescent components are dropped, where abTEM's would decay; the
        # aperture keeps them far out of reach at any sampling in use.
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
    """abTEM's Fresnel propagator, for waves carried as periodic envelopes.

    Only the kernel differs, so the propagation itself -- and the choice of FFT
    behind it -- is abTEM's.
    """

    def __init__(self, wave_vectors):
        super().__init__()
        self._wave_vectors = wave_vectors
        self._kernels: dict = {}

    def get_array(self, waves: Waves, thickness: float, order="exact"):
        kernel = self._kernels.get(thickness)
        if kernel is None:
            kernel = _envelope_propagator_array(
                self._wave_vectors,
                waves._valid_gpts,
                waves._valid_sampling,
                waves._valid_energy,
                thickness,
                get_array_module(waves.device),
            )
            self._kernels[thickness] = kernel
        return kernel


class EBSD(CopyMixin, EqualityMixin):
    """Backscatter diffraction patterns from one slab, by reciprocity.

    The engine the pattern builders run in each of their slabs. It takes a slab
    already cut, and directions measured from its axis, and knows nothing of a
    sample, a beam or a detector: for a pattern on a real detector, from Euler
    angles and with the beam coming in from its true direction, see
    :class:`~abtem.ebsd.detector_pattern.EBSDDetectorPattern`.

    Parameters
    ----------
    potential : BasePotential or Atoms
        The specimen, built from atoms -- ``Atoms``,
        :class:`~abtem.FrozenPhonons`, or a :class:`~abtem.Potential` of either.
        The atoms are what emits, so a prebuilt ``PotentialArray``, which no
        longer carries them, will not do. The reciprocity waves travel along
        ``z``, so a slab should already be cut with the zone axis of interest
        along it (see :func:`abtem.ebsd.rotated_slab`).

        For the thermal cloud of the emitting atoms -- EMsoft's Debye-Waller
        factor -- build it from :class:`~abtem.FrozenPhonons`: the average over
        its configurations smears the emitters as it smears the scattering.
    detector : BackscatterDetector
        The directions to collect, in the slab's frame: the directions the
        reciprocity waves travel into it, ``+z`` being into the slab. The
        electrons they stand for leave the other way, so to collect the
        electrons leaving a crystal along ``d``, cut the slab along ``-d``;
        :class:`~abtem.ebsd.reference.EBSDReferencePattern` and
        :class:`~abtem.ebsd.detector_pattern.EBSDDetectorPattern` do.
    energy : float
        Energy of the incident beam [eV]. It sets the atoms' cross-sections,
        and the reciprocity waves' energy unless `backscatter_energy` does.
    illumination : np.ndarray, optional
        How brightly the incident beam lights each atom of the potential, of
        shape ``(atoms,)``, or ``(sources, atoms)`` for several at once, which
        become a leading axis of the result. Only the relative values matter.
        The atoms always emit, each by its cross-section for scattering
        straight back (see :mod:`abtem.ebsd.emission`); this says only how
        they are lit.

        By default all alike, as though the specimen were evenly illuminated --
        EMsoft's picture. Backscattered electrons come from tens of nanometres
        down, where the incident beam has long lost its direction, so the
        pattern is then a property of the specimen alone. Every atom in the cell
        emits, so for a specimen that is not a uniform crystal the pattern is
        the average over the cell.
        :class:`~abtem.ebsd.detector_pattern.EBSDDetectorPattern` passes the
        lighting of a beam from its true direction here.
    depth_weight : float or array or callable, optional
        How much the electrons generated at each depth count: the depth
        distribution of the backscattering events that end in the pattern.

        The elastic multislice algorithm is unitary, so the reciprocity plane
        waves carry no attenuation with depth on their own: without a weight,
        an event 200 Å deep counts as much as one just below the surface. Give
        a float to weight the depths by ``exp(-z / escape_depth)`` [Å], an array
        of one weight per slice, or a callable mapping an array of depths to
        weights. The weights are normalized to sum to one.

        An electron that lost more energy has, on the whole, come from deeper,
        so with several `backscatter_energy` the weights may depend on the
        energy: a callable of ``(depths, energy)``, or an array of one row per
        energy -- the depth distribution per energy bin of a Monte Carlo
        simulation, as EMsoft uses.

        The default, None, weights every depth equally. Thermal diffuse
        scattering along the way is the specimen's own physics: build it from
        :class:`~abtem.FrozenPhonons`. It damps the contrast but keeps the
        electrons; the smooth background of electrons whose paths ran far
        deeper than any slab is a Monte Carlo quantity this does not produce.
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energy of the backscattered electrons [eV]. A backscattered electron
        has lost some of the beam's `energy`, so these must not exceed it.
        Giving more than one adds a leading :class:`.EnergyAxis` and costs
        proportionally more: the reciprocity waves travel at this energy, so
        each one needs its own wavelength, propagator and transmission
        function. Defaults to `energy`, the elastic case.

        The weights of a distribution are recorded in the metadata rather than
        applied, since summing the energy axis is the consumer's business and
        the weights usually come from a Monte Carlo spectrum.

        ``metadata["energy"]`` is that of the backscattered electrons, unset
        when there are several, and ``metadata["beam_energy"]`` the beam's. A
        grid detector's directions are the same at every energy, while abTEM
        calibrates a :class:`.DiffractionPatterns` of several energies in
        reciprocal space, by the highest; its angles are then those of the
        grid, but indexing one lower energy out rescales them by its
        wavelength.
    depth_tolerance : float, optional
        Stop propagating once the depth weight still to be collected falls
        below this fraction of the total (default 1e-4). Slices past a short
        escape depth carry no weight, and propagating them is the dominant
        cost. Has no effect under the default uniform weighting, where the
        remaining weight only vanishes at the last slice.

        The bound is on the *discarded weight*; the error in any one direction
        can be a few times that, because the slices dropped are not average
        ones. Set to 0 to disable.
    depth_bins : int or array, optional
        Resolve the pattern by depth: a number of equal bins over the slab, or
        bin edges [Å]. Adds a ``depth`` axis, after any energy axis, holding
        the pattern of the electrons generated in each bin alone -- each a
        yield of its own, one for a featureless specimen. The emission in each
        bin is recorded as ``metadata["depth_emission"]``, shaped like the
        leading axes, so that any other depth weighting ``W`` can be applied
        afterwards::

            E = np.array(patterns.metadata["depth_emission"])   # (bins,)
            pattern = np.tensordot(W * E, patterns.array, (0, 0)) / (W * E).sum()

        which with ``W = 1`` is the pattern without bins -- exactly, for a
        specimen without frozen-phonon configurations, and to within their
        spread otherwise. Slices outside the edges are left out.
    device : str, optional
        'cpu' or 'gpu'. Defaults to the user configuration.
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
                f"backscatter is generated at the atoms, so the potential has to "
                f"be built from them -- Atoms, FrozenPhonons, or a Potential of "
                f"either; a {type(potential).__name__} does not carry its atoms"
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
        """Split the collected directions into batches that fit in memory.

        Blocks are the unit of work, so when they are going to become dask
        tasks there have to be enough of them to occupy the workers. The
        memory-derived size alone can leave three tasks on a twelve core
        machine; this only ever splits further, never coarser.
        """
        gpts = self._potential.gpts
        if gpts is None:
            raise RuntimeError("the potential has no grid")

        # Every wave carries a propagator kernel of its own as large as itself,
        # so each direction holds two complex arrays of the grid's size.
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
            If True, build a dask graph instead of computing, with one task per
            block of directions per backscattered energy.

            Defaults to abTEM's ``dask.lazy`` configuration, which ships as
            True, so ask for ``lazy=False`` to get an array back directly.
        pbar : bool, optional
            If True, show a progress bar over the slices.

        Returns
        -------
        patterns : DiffractionPatterns or SphericalPattern
            :class:`~abtem.measurements.DiffractionPatterns` if the detector is
            a grid, otherwise a
            :class:`~abtem.ebsd.measurements.SphericalPattern` holding one
            value per collected direction. Several illuminations appear as a
            leading ensemble axis, and several backscattered energies as one
            before that.
        """
        lazy = validate_lazy(lazy)

        xp = get_array_module(self._device)

        # Only the geometry is needed to lay out the work, and every frozen
        # -phonon configuration shares it. Building each one is deferred to the
        # point of use, so that 400 of them are neither built serially while
        # the graph is assembled nor held in it at once.
        num_slices = self._potential.num_slices
        configurations = _potential_configurations(self._potential)

        energy = self._energy
        backscatter_energies, energy_weights = _validate_backscatter_energy(
            self._backscatter_energy, energy
        )
        n_energies = len(backscatter_energies)

        depths = np.cumsum(np.asarray(self._potential.slice_thickness, dtype=float))
        weights = _depth_weights(self._depth_weight, depths, backscatter_energies)

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

        def block_arguments(i, start, stop):
            wave_vectors = xp.asarray(
                self._detector.transverse_wave_vectors(backscatter_energies[i]),
                dtype=get_dtype(complex=False),
            )[start:stop]
            return dict(
                wave_vectors=wave_vectors,
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
                configurations, weights, edges, slice_bins, output_shape, energy
            )
        )

        if lazy:
            return self._lazy_measurement(
                configurations=configurations,
                block_arguments=block_arguments,
                blocks=blocks,
                backscatter_energies=backscatter_energies,
                energy_weights=energy_weights,
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

        if n_energies == 1:
            intensities = intensities[0]
        else:
            ensemble_axes_metadata = [
                EnergyAxis(values=tuple(float(e) for e in backscatter_energies))
            ] + ensemble_axes_metadata

        return self._to_measurement(
            intensities,
            ensemble_axes_metadata=ensemble_axes_metadata,
            backscatter_energies=backscatter_energies,
            energy_weights=energy_weights,
            extra_metadata=extra_metadata,
        )

    def _depth_metadata(
        self, configurations, weights, edges, slice_bins, output_shape, energy
    ) -> dict:
        """The bins, and the emission in each: what recombining them needs.

        The pattern with any other depth weighting is the emission-weighted
        mean of the bins' patterns, with the new weights applied to the
        emission recorded here.
        """
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
        energy_weights,
        output_shape,
        ensemble_axes_metadata,
        energy: float,
        block_shape: tuple[int, ...] = (),
        extra_metadata: Optional[dict] = None,
    ) -> DiffractionPatterns | SphericalPattern:
        """Assemble the same per-block computation into a dask graph.

        Each block of directions, at each backscattered energy and each frozen
        -phonon configuration, is one task. The blocks of a configuration share
        its built potential and emitters, which the threaded scheduler passes
        by reference rather than copying. The wavefields a task propagates are
        its own, so only one block of them is resident at a time however many
        directions were asked for.
        """
        import dask

        n_energies = len(backscatter_energies)

        array = None
        for configuration, configuration_weight in configurations:
            # One task per configuration, so its blocks share the built
            # potential and its emitters, and dask frees them once done.
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

        if n_energies == 1:
            array = array[0]
        else:
            ensemble_axes_metadata = [
                EnergyAxis(values=tuple(float(e) for e in backscatter_energies))
            ] + ensemble_axes_metadata

        if self._detector.is_grid:
            # The two base axes of a diffraction pattern have to be one chunk.
            array = array.rechunk(array.chunks[:-1] + (-1,))

        return self._to_measurement(
            array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            backscatter_energies=backscatter_energies,
            energy_weights=energy_weights,
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

        Returns the block's own intensities rather than writing into a shared
        buffer, so the same routine serves the eager
        assembly and the lazy one, where each block is a separate task.

        At every slice the emission of its atoms is laid on the grid and
        weighted by the escape probability -- the reciprocity waves' intensity
        -- and by the depth weight. Numerator and denominator are summed over
        the whole slab and divided once, so the result is the emission-weighted
        mean escape probability -- per depth bin, when there are bins.
        """
        potential, emission = prepared
        xp = get_array_module(self._device)

        # Each wave is carried as its periodic envelope (see the module notes):
        # a plane wave of unit modulus, whose escape probability is one, is an
        # envelope of ones whatever its direction.
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
        # Accumulated in double precision whatever the configured one: these
        # sum the emission of every atom over every slice.
        numerator = xp.zeros((n_bins, n_sources, n_directions), dtype=xp.float64)
        denominator = xp.zeros((n_bins, n_sources), dtype=xp.float64)

        # Weight still to be collected at and below each slice. Once it is
        # negligible there is nothing left to gather and the remaining slices
        # are wasted propagation -- which is most of the specimen when the
        # escape depth is short compared to its thickness.
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

            # The emission of a slice's atoms is collected from the waves as
            # they arrive at the slice -- the plane the multislice lumps those
            # atoms onto -- before the step through it. Collected after the
            # step, it would be read a whole slice downstream of them, where
            # the waves have been focused by them and, travelling at an angle,
            # drifted sideways off them: the yield would then depend on the
            # slice thickness, the more so the steeper the waves.
            weight = float(weights[index])
            b = int(slice_bins[index])
            emitted = emission[index] if weight > 0.0 and b >= 0 else None

            if emitted is not None:
                overlap = self._overlap(reciprocity, emitted, n_directions)
                numerator[b] += weight * overlap.T
                denominator[b] += weight * xp.sum(
                    emitted, axis=(-2, -1), dtype=xp.float64
                )

            reciprocity = conventional_multislice_step(
                reciprocity,
                transmission,
                propagator=propagator,
                antialias_aperture=antialias_aperture,
            )

            if progress is not None:
                progress.update_if_exists(1)

        # Nothing emitted -- a slab with no atoms -- is no backscatter, not 0/0.
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
        #
        # The squared magnitude of the reciprocity waves is the largest array in
        # the loop, and taking it with abs()**2 costs more than the contraction
        # it feeds. Viewing the complex array as real numbers gives the real and
        # imaginary parts interleaved and contiguous, so squaring every
        # component and contracting against the source with each of its values
        # repeated twice computes the same sum in one pass over contiguous
        # memory -- about seven times faster than abs()**2 on the strided
        # halves, and exactly equal.
        real_view = reciprocity.array.view(get_dtype(complex=False))
        real_view = real_view.reshape(n_directions, -1)

        # The interleave is built from a one-dimensional repeat rather than
        # `repeat(source_flat, 2, axis=-1)`. The two are exactly equal -- row
        # -major order puts each position's pixels contiguously, so duplicating
        # pairwise reproduces the interleave -- but cupy's axis-ed path is
        # pathologically slow: 482 ms against 0.0 ms for half a megabyte, which
        # was the whole of a 169x slowdown on the GPU.
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
        energy_weights: np.ndarray,
        extra_metadata: Optional[dict] = None,
    ) -> DiffractionPatterns | SphericalPattern:
        """Wrap the collected intensities in the matching measurement type."""
        array: np.ndarray | da.core.Array
        if isinstance(intensities, da.core.Array):
            array = intensities
        else:
            array = np.asarray(
                intensities.get() if hasattr(intensities, "get") else intensities
            )
        metadata = {
            **_energy_metadata(self._energy, backscatter_energies, energy_weights),
            "label": "backscattered intensity",
            "source": "uniform" if self._illumination is None else "illumination",
            **(extra_metadata or {}),
        }

        if self._detector.is_grid:
            # Calibrated by the highest energy, as abTEM reads the angles of a
            # diffraction pattern of several: see `backscatter_energy`.
            gpts = self._detector.gpts
            return DiffractionPatterns(
                array.reshape(array.shape[:-1] + (gpts, gpts)),
                sampling=self._detector.reciprocal_sampling(
                    float(np.max(backscatter_energies))
                ),
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
