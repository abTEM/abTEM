"""Module for running the multislice algorithm."""

from __future__ import annotations

import copy
from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeGuard, cast

import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture, antialias_aperture
from abtem.core import config
from abtem.core.axes import AxisMetadata, OrdinalAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks, ValidatedChunks, validate_chunks
from abtem.core.complex import complex_exponential
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2wavelength
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args
from abtem.core.fft import CachedFFTWConvolution, fft2_convolve
from abtem.core.grid import spatial_frequencies
from abtem.core.utils import expand_dims_to_broadcast
from abtem.detectors import BaseDetector, WavesDetector, validate_detectors
from abtem.finite_difference import LaplaceOperator
from abtem.finite_difference import multislice_step as realspace_multislice_step
from abtem.inelastic.core_loss import TransitionPotential, TransitionPotentialArray
from abtem.inelastic.plasmons import _update_plasmon_axes
from abtem.measurements import BaseMeasurements
from abtem.potentials.iam import (
    BasePotential,
    PotentialArray,
    TransmissionFunction,
    validate_potential,
)
from abtem.slicing import SliceIndexedAtoms
from abtem.tilt import _get_tilt_axes
from abtem.transform import WavesTransform

if TYPE_CHECKING:
    from abtem.inelastic.plasmons import PhaseScramblePlasmons
    from abtem.waves import Waves


def _fresnel_propagator_array(
    thickness: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    energy: float,
    device: str,
    order: int = 1,
):
    if order > 2:
        raise ValueError(
            """
            Only orders 1 and 2 are supported in Fourier space.
            For higher orders, use the realspace multislice instead.
            """
        )

    xp = get_array_module(device)
    wavelength = energy2wavelength(energy)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    kx, ky = kx[:, None], ky[None]

    f = complex_exponential(
        -(kx**2) * np.pi * thickness * wavelength
    ) * complex_exponential(-(ky**2) * np.pi * thickness * wavelength)

    # Propagator corrected in Fourier-space, only valid for order=2
    # Eq. (4) from Microscopy and Microanalysis (2020), 26, 1147-1157
    if order == 2:
        f = f * complex_exponential(
            (-np.pi * thickness * wavelength**3) / 4.0 * (kx**4 + ky**4)
        )
    return f


def _apply_tilt_to_fresnel_propagator_array(
    array: np.ndarray,
    sampling: tuple[float, float],
    thickness: float,
    tilt: tuple[float, float] | tuple[tuple[float, float], ...] | np.ndarray,
):
    xp = get_array_module(array)
    tilt = cast(np.ndarray, xp.array(tilt))

    squeeze = False
    if tilt.shape == (2,):
        squeeze = True
        tilt = tilt[None]

    kx, ky = spatial_frequencies(array.shape[-2:], sampling, xp=xp)
    kx, ky = kx[None, :, None], ky[None, None]

    tilt = complex_exponential(
        -kx * xp.tan(tilt[:, 0, None, None] / 1e3) * thickness * 2 * np.pi
    ) * complex_exponential(
        -ky * xp.tan(tilt[:, 1, None, None] / 1e3) * thickness * 2 * np.pi
    )

    tilt, array = expand_dims_to_broadcast(tilt, array, match_dims=((-2, -1), (-2, -1)))

    array = tilt * array

    if squeeze:
        array = array[0]

    return array


class FresnelPropagator:
    """
    The Fresnel propagator is used for propagating wave functions using the near-field
    approximation (Fresnel diffraction).
    """

    def __init__(self):
        self._array = None
        self._key = None
        self._cached_fftw_convolution = CachedFFTWConvolution()

    def get_array(self, waves: Waves, thickness: float, order: int = 1) -> np.ndarray:
        """
        Get the Fresnel propagator as an array for the given wave functions and
        thickness.

        Parameters
        ----------
        waves : Waves
            The wave functions to propagate.
        thickness : float
            Distance in free space to propagate [Å].

        Returns
        -------
        array : np.ndarray
            The Fresnel propagator as an array.
        """
        key: tuple[Any, ...] = (
            waves._valid_gpts,
            waves._valid_sampling,
            thickness,
            waves.base_tilt,
            waves._valid_energy,
            waves.device,
        )

        tilt_axes_metadata = _get_tilt_axes(waves)
        if len(tilt_axes_metadata) > 0:
            key = key + copy.deepcopy(tilt_axes_metadata)

        if key == self._key:
            return self._array

        self._array = self._calculate_array(waves, thickness, order=order)
        self._key = key

        return self._array

    @staticmethod
    def _calculate_array(waves: Waves, thickness: float, order: int = 1) -> np.ndarray:
        array = _fresnel_propagator_array(
            thickness=thickness,
            gpts=waves._valid_gpts,
            sampling=waves._valid_sampling,
            energy=waves._valid_energy,
            device=waves.device,
            order=order,
        )

        array *= antialias_aperture(
            waves._valid_gpts,
            waves._valid_sampling,
            get_array_module(waves.device),
        )

        if waves.base_tilt != (0.0, 0.0):
            array = _apply_tilt_to_fresnel_propagator_array(
                array,
                sampling=waves._valid_sampling,
                thickness=thickness,
                tilt=waves.base_tilt,
            )

        xp = get_array_module(waves.device)

        tilt_axes = _get_tilt_axes(waves)
        if not tilt_axes:
            return array

        for axis in reversed(waves.ensemble_axes_metadata):
            if hasattr(axis, "tilt"):
                tilt = xp.asarray(axis.tilt)
                array = _apply_tilt_to_fresnel_propagator_array(
                    array,
                    sampling=waves._valid_sampling,
                    tilt=tilt,
                    thickness=thickness,
                )

            else:
                array = array[..., None, :, :]

        return array

    def propagate(
        self, waves: Waves, thickness: float, in_place: bool = False, order: int = 1
    ) -> Waves:
        """
        Propagate wave functions through free space.

        Parameters
        ----------
        waves : Waves
            The wave functions to propagate.
        thickness : float
            Distance in free space to propagate.
        in_place : bool
            If True, the waves are overwritten.

        Returns
        -------
        propagated_wave_functions : Waves
            Propagated wave functions.
        """
        kernel = self.get_array(waves, thickness, order=order)
        if (config.get("fft") == "fftw") and isinstance(waves._array, np.ndarray):
            array = self._cached_fftw_convolution(
                waves._array, kernel, overwrite_x=in_place
            )
        else:
            array = fft2_convolve(waves._array, kernel, overwrite_x=in_place)

        if in_place:
            waves._array = array
        else:
            kwargs = waves._copy_kwargs(exclude=("array",))
            waves = waves.__class__(array, **kwargs)

        return waves


def allocate_measurement(
    waves: Waves,
    detector: BaseDetector,
    extra_ensemble_axes_shape: tuple[int, ...],
    extra_ensemble_axes_metadata: list[AxisMetadata],
) -> BaseMeasurements | Waves:
    """
    Allocate a measurement matching the given wave functions and detector.

    Parameters
    ----------
    waves : BaseWaves
        The wave functions to derive the allocated measurement from.
    detector : BaseDetector
        The detector to derive the allocated measurement from.
    extra_ensemble_axes_shape : tuple of int, optional
        The shape of additional ensemble axes not in the waves.
    extra_ensemble_axes_metadata : list of AxisMetadata
        The axes metadata of additional ensemble axes not in the waves.

    Returns
    -------
    allocated_measurement : BaseMeasurements or Waves
        The allocated measurement
    """
    xp = get_array_module(detector._out_meta(waves)[0])

    measurement_type = detector._out_type(waves)[0]

    axes_metadata = detector._out_axes_metadata(waves)[0]

    shape = detector._out_shape(waves)[0]
    #
    if extra_ensemble_axes_shape is not None:
        if len(extra_ensemble_axes_shape) != len(extra_ensemble_axes_metadata):
            raise ValueError(
                f"extra_ensemble_axes_shape length ({len(extra_ensemble_axes_shape)}) "
                f"!= extra_ensemble_axes_metadata length ({len(extra_ensemble_axes_metadata)})"
            )
        shape = extra_ensemble_axes_shape + shape
        axes_metadata = extra_ensemble_axes_metadata + axes_metadata

    metadata = detector._out_metadata(waves)[0]

    array = xp.zeros(shape, dtype=detector._out_dtype(waves)[0])

    out_measurement = measurement_type.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=metadata
    )

    return out_measurement


def _potential_ensemble_shape_and_metadata(
    potential: BasePotential,
) -> tuple[tuple[int, ...], list[AxisMetadata]]:
    extra_ensemble_axes_shape = potential.ensemble_shape
    extra_ensemble_axes_metadata = potential.ensemble_axes_metadata

    if len(potential.exit_planes) > 1:
        extra_ensemble_axes_shape = (
            *extra_ensemble_axes_shape,
            len(potential.exit_planes),
        )
        extra_ensemble_axes_metadata = [
            *extra_ensemble_axes_metadata,
            potential._get_exit_planes_axes_metadata(),
        ]

    return extra_ensemble_axes_shape, extra_ensemble_axes_metadata


def allocate_multislice_measurements(
    waves: Waves,
    detectors: list[BaseDetector],
    extra_ensemble_axes_shape: tuple[int, ...],
    extra_ensemble_axes_metadata: list[AxisMetadata],
) -> list[BaseMeasurements | Waves]:
    """
    Allocate the multislice measurements that would be produced by detecting the given
    set of wave functions with the given set of detectors.

    Parameters
    ----------
    waves : Waves
        The waves to derive the allocated measurement from.
    detectors : list of BaseDetector
        The detectors to derive the allocated measurement from.
    extra_ensemble_axes_shape : tuple of int, optional
        The shape of additional ensemble axes not in the waves.
    extra_ensemble_axes_metadata : list of AxisMetadata
        The axes metadata of additional ensemble axes not in the waves.

    Returns
    -------
    allocated_measurements : list
        List of allocated to measurements.
    """

    measurements = []
    for detector in detectors:
        measurements.append(
            allocate_measurement(
                waves, detector, extra_ensemble_axes_shape, extra_ensemble_axes_metadata
            )
        )

    return measurements


def conventional_multislice_step(
    waves: Waves,
    potential_slice: PotentialArray | TransmissionFunction,
    propagator: FresnelPropagator,
    antialias_aperture: AntialiasAperture,
    conjugate: bool = False,
    transpose: bool = False,
    order: int = 1,
) -> Waves:
    """
    Calculate one step of the multislice algorithm for the given batch of wave functions
    through a given potential slice.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential_slice : PotentialArray or TransmissionFunction
        A potential slice as a :class:`.PotentialArray` or
        :class:`.TransmissionFunction`.
    propagator : FresnelPropagator, optional
        A Fresnel propagator type matching the wave functions. The main reason for using
        this argument is to reuse a previously calculated propagator. If not provided a
        new propagator is created.
    antialias_aperture : AntialiasAperture, optional
        An antialias aperture type matching the wave functions. The main reason for
        using this argument is to reuse a previously calculated antialias aperture.
        If not provided a new antialias aperture is created.
    conjugate : bool, optional
        If True, use the conjugate of the transmission function (default is False).
    transpose : bool, optional
        If True, reverse the order of propagation and transmission (default is False).

    Returns
    -------
    forward_stepped_waves : Waves
        Wave functions propagated and transmitted through the potential slice.
    """
    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    if isinstance(potential_slice, TransmissionFunction):
        transmission_function = potential_slice

    else:
        transmission_function = potential_slice.transmission_function(
            energy=waves._valid_energy
        )
        transmission_function = antialias_aperture.bandlimit(
            transmission_function, in_place=False
        )

    thickness = transmission_function.slice_thickness[0]

    if conjugate:
        thickness = -thickness

    if transpose:
        waves = propagator.propagate(
            waves, thickness=thickness, in_place=True, order=order
        )
        waves = transmission_function.transmit(waves, conjugate=conjugate)
    else:
        waves = transmission_function.transmit(waves, conjugate=conjugate)
        waves = propagator.propagate(
            waves, thickness=thickness, in_place=True, order=order
        )

    return waves


def _configuration_seed(potential_configuration) -> Optional[int]:
    """Per-configuration frozen-phonon seed, if available.

    Used to give the plasmon phase-scramble a globally-unique, reproducible,
    partition-safe per-configuration random stream. Returns ``None`` for potentials
    without frozen-phonon seeds (e.g. a static potential).
    """
    frozen_phonons = getattr(potential_configuration, "frozen_phonons", None)
    seed = getattr(frozen_phonons, "seed", None)
    if seed is None:
        return None
    if hasattr(seed, "__len__"):
        return int(seed[0]) if len(seed) else None
    return int(seed)


def _plasmon_total_intensity(waves: Waves):
    """Total wave-function intensity per ensemble member (summed over the grid axes)."""
    xp = get_array_module(waves.device)
    arr = xp.abs(waves._array).astype(xp.float64)
    return xp.sum(arr**2, axis=(-2, -1), keepdims=True)


def _renormalize_total_intensity(waves: Waves, target_norm) -> Waves:
    """Return a copy of ``waves`` rescaled to the given per-member total intensity.

    The phase-scramble plasmon operator is non-unitary, so the exit wave is renormalized
    to conserve the incident electron count before detection (following the reference
    implementation, which normalizes the wave before forming the diffraction pattern).
    """
    xp = get_array_module(waves.device)
    arr = xp.abs(waves._array).astype(xp.float64)
    current = xp.sum(arr**2, axis=(-2, -1), keepdims=True)
    scale = xp.sqrt(target_norm / current).astype(waves._array.dtype)
    kwargs = waves._copy_kwargs(exclude=("array",))
    kwargs["array"] = waves._array * scale
    return waves.__class__(**kwargs)


def _renormalize_order_waves(order_waves: list, target_norm) -> None:
    """Renormalize order-resolved waves so their total intensity matches the
    incident beam.  The same scale factor is applied to every order so that
    relative intensities are preserved."""
    xp = get_array_module(order_waves[0].device)
    total = sum(
        xp.sum(
            xp.abs(w._array).astype(xp.float64) ** 2,
            axis=(-2, -1),
            keepdims=True,
        )
        for w in order_waves
    )
    scale = xp.sqrt(target_norm / total).astype(order_waves[0]._array.dtype)
    for w in order_waves:
        w._array = w._array * scale


def _detect_order_resolved(
    order_waves: list,
    target_norm,
    detectors: list,
    measurements: list,
    config_meas_index: tuple[int, ...],
) -> None:
    """Renormalize and detect each order, writing into the pre-allocated
    measurements at ``(order_index,) + config_meas_index``."""
    _renormalize_order_waves(order_waves, target_norm)
    for n, w in enumerate(order_waves):
        idx = (n,) + config_meas_index
        _update_measurements(w, detectors, measurements, idx)


def _stack_order_detections(
    order_waves: list,
    detector,
    potential_ensemble_shape: tuple[int, ...],
    order_axis: "OrdinalAxis",
):
    """Detect each order and stack with a leading order axis (single-config
    path where ``measurements is None``)."""
    xp = get_array_module(order_waves[0].device)
    detected = [detector.detect(w) for w in order_waves]
    arrays = xp.stack([d.array for d in detected], axis=0)
    pad = (None,) * len(potential_ensemble_shape)
    arrays = arrays[pad]

    kwargs = detected[0]._copy_kwargs(exclude=("array",))
    kwargs["array"] = arrays
    kwargs["ensemble_axes_metadata"] = (
        [order_axis] + kwargs["ensemble_axes_metadata"]
    )
    return detected[0].__class__(**kwargs)


def _update_measurements(
    waves: Waves,
    detectors: list[BaseDetector],
    measurements: list[BaseMeasurements | Waves],
    measurement_index: tuple[int, ...] = (0,),
    additive: bool = False,
) -> None:
    assert len(detectors) == len(measurements)

    for i, detector in enumerate(detectors):
        new_measurement = detector.detect(waves)

        if additive:
            measurements[i].array[measurement_index] += new_measurement.array
        else:
            measurements[i].array[measurement_index] = new_measurement.array
    return


def _validate_potential_ensemble_indices(
    potential_index: int | tuple[int, ...],
    exit_plane_index: int | tuple[int, ...],
    potential: BasePotential,
) -> tuple[int, ...]:
    if not potential.ensemble_shape:
        potential_index = ()
    elif not isinstance(potential_index, tuple):
        potential_index = (potential_index,)

    if len(potential.exit_planes) == 1:
        exit_plane_index = ()
    elif not isinstance(exit_plane_index, tuple):
        exit_plane_index = (exit_plane_index,)

    measurement_indices = potential_index + exit_plane_index

    return measurement_indices


def _generate_potential_configurations(potential):
    for potential_index, _, potential_configuration in potential.generate_blocks():
        potential_configuration = potential_configuration.item()

        if len(potential.ensemble_shape):
            potential_index = np.unravel_index(
                potential_index, potential.ensemble_shape
            )

        yield potential_index, potential_configuration


def lookahead(iterable):
    """
    Generator that yields (current, next) items from an iterable.
    The last item is yielded as (last, None).
    """
    it = iter(iterable)
    try:
        current_item = next(it)
    except StopIteration:
        return

    for next_item in it:
        yield current_item, next_item
        current_item = next_item

    yield current_item, None


@dataclass(frozen=True)
class FourierMultislice:
    """
    Multislice algorithm computed fast in Fourier space.

    Parameters
    ----------
    order : int, optional
        Propagator order, one of 1 or 2 (default 1)
    expansion_scope: str
        Specified for compatibility. Must be "propagator" (default "propagator")
    conjugate : bool, optional
        If True, use the conjugate of the transmission function (default is False)
    transpose : bool, optional
        If True, reverse the order of propagation and transmission (default is False)
    """

    order: Literal[1, 2] = 1
    expansion_scope: Literal["propagator"] = "propagator"
    conjugate: bool = False
    transpose: bool = False


@dataclass(frozen=True)
class RealSpaceMultislice:
    """
    Multislice algorithm computed in real-space.

    Parameters
    ----------
    order : int, optional
        Propagator and/or transmission operator order (default 1)
    expansion_scope: str
        If "propagator" (default) only the propagator operator is expanded to order
        If "full" both the propagator and transmission operators are expanded to order
    derivative_accuracy : int, optional
        Finite-difference accuracy for Laplace operator (default 6)
    max_terms: int, optional
        Max terms in exponent Taylor series expansion (default 80)
    """

    order: int = 1
    expansion_scope: Literal["propagator", "full"] = "propagator"
    derivative_accuracy: int = 6
    max_terms: int = 80


def multislice_and_detect(
    waves: Waves,
    potential: BasePotential,
    detectors: Optional[list[BaseDetector]] = None,
    algorithm: FourierMultislice | RealSpaceMultislice = FourierMultislice(),
    return_backscattered: bool = False,
    plasmons: Optional["PhaseScramblePlasmons"] = None,
    pbar: bool = False,
) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
    """
    Calculate the full multislice algorithm for the given batch of wave functions
    through a given potential, detecting at each of the exit planes specified in the
    potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be
        converted to measurements after running the multislice algorithm.
    algorithm: FourierMultislice or RealSpaceMultislice, optional
        Algorithm used for multislice operator (default is FourierMultislice())
    return_backscattered: bool, optional
        If algorithm.expansion_scope="full" and return_backscatter is True, then the
        backscattered components are also returned. Requires potential exit_planes
    plasmons : PhaseScramblePlasmons, optional
        If given, plasmon energy-loss scattering is applied inline at every slice using
        the fast phase-scramble method. Phase-scramble repetitions are realised through
        the potential's frozen-phonon configuration ensemble, so use a potential with
        ``num_configs`` (frozen phonons) set to the desired number of repetitions and
        ``ensemble_mean=True`` to obtain the incoherent average. The operator is a
        real-space multiplication at each slice boundary, so it composes with both the
        ``FourierMultislice`` and ``RealSpaceMultislice`` algorithms (the latter only
        with ``expansion_scope='propagator'``; backscattering is not supported).

    """
    waves = waves.ensure_real_space()
    detectors = validate_detectors(detectors)
    waves = waves.copy()

    if return_backscattered:
        if algorithm.expansion_scope != "full":
            raise ValueError(
                "Backscattering contributions require expansion_scope='full'."
            )
        if potential.num_exit_planes == 1:
            raise ValueError(
                "Backscattering contributions require potential.exit_planes."
            )

        # moved to MultisliceTransform
        # detectors = list(detectors) + [WavesDetector()]

    is_fourier = isinstance(algorithm, FourierMultislice)

    if is_fourier:
        antialias_aperture = AntialiasAperture()
        propagator = FresnelPropagator()

        def multislice_step(waves, potential_slice, next_slice=None):
            return conventional_multislice_step(
                waves,
                potential_slice=potential_slice,
                antialias_aperture=antialias_aperture,
                propagator=propagator,
                conjugate=algorithm.conjugate,
                transpose=algorithm.transpose,
                order=algorithm.order,
            )

    else:
        laplace_operator = LaplaceOperator(algorithm.derivative_accuracy)

        def multislice_step(waves, potential_slice, next_slice=None):
            return realspace_multislice_step(
                waves,
                potential_slice=potential_slice,
                next_slice=next_slice,
                laplace=laplace_operator,
                max_terms=algorithm.max_terms,
                order=algorithm.order,
                fully_corrected=algorithm.expansion_scope == "full",
            )

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    # Order-resolved plasmon scattering: prepend a loss-order axis.
    max_loss_order = (
        plasmons.max_loss_order if plasmons is not None else None
    )
    order_resolved = max_loss_order is not None
    if order_resolved and not is_fourier and algorithm.expansion_scope == "full":
        raise NotImplementedError(
            "Order-resolved plasmon scattering is not compatible with "
            "expansion_scope='full' (backscattering is not defined per loss "
            "order). Use expansion_scope='propagator' (the default)."
        )
    if order_resolved:
        n_orders = max_loss_order + 1
        order_labels = ("Zero loss",) + tuple(
            f"{n}-plasmon" for n in range(1, n_orders)
        )
        order_axis = OrdinalAxis(
            label="Plasmon order", values=order_labels
        )
        extra_ensemble_axes_shape = (n_orders,) + extra_ensemble_axes_shape
        extra_ensemble_axes_metadata = [order_axis] + extra_ensemble_axes_metadata

    if sum(extra_ensemble_axes_shape) == 1:
        measurements = None
    else:
        measurements = allocate_multislice_measurements(
            waves,
            detectors,
            extra_ensemble_axes_shape,
            extra_ensemble_axes_metadata,
        )

    n_waves = np.prod(waves.shape[:-2])
    n_slices = n_waves * potential.num_slices * potential.num_configurations

    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=int(n_slices), leave=False, desc="multislice"
    )

    waves_input = waves.copy()

    for potential_index, potential_configuration in _generate_potential_configurations(
        potential
    ):
        waves = waves_input.copy()
        exit_plane_index = 0

        if plasmons is not None:
            plasmon_operator = plasmons._build_operator(
                waves,
                potential_index,
                config_seed=_configuration_seed(potential_configuration),
            )
            plasmon_target_norm = _plasmon_total_intensity(waves)
        else:
            plasmon_operator = None
            plasmon_target_norm = None

        # Initialise per-order wave channels for order-resolved mode.
        if order_resolved:
            xp = get_array_module(waves.device)
            order_waves = [waves.copy() for _ in range(n_orders)]
            for n in range(1, n_orders):
                order_waves[n]._array = xp.zeros_like(waves._array)

        # Handle entrance plane detection (before first slice)
        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )

            if measurements is not None:
                _update_measurements(waves, detectors, measurements, measurement_index)

            exit_plane_index += 1

        depth = 0.0

        for potential_slice, next_slice in lookahead(
            potential_configuration.generate_slices()
        ):
            if order_resolved:
                # Build a single slice operand shared across all loss-order
                # channels, then step each channel through the same
                # (algorithm-agnostic) ``multislice_step`` closure.
                if is_fourier:
                    # Pre-compute the transmission function once per slice so
                    # ``exp(i·sigma·V)`` is not recomputed for every order.
                    if isinstance(potential_slice, TransmissionFunction):
                        shared_slice = potential_slice
                    else:
                        shared_slice = potential_slice.transmission_function(
                            energy=order_waves[0]._valid_energy
                        )
                        shared_slice = antialias_aperture.bandlimit(
                            shared_slice, in_place=False
                        )
                else:
                    # The real-space (finite-difference) step consumes the
                    # potential slice directly.
                    shared_slice = potential_slice
                for n in range(n_orders):
                    order_waves[n] = multislice_step(
                        order_waves[n], shared_slice, next_slice=None
                    )
            elif algorithm.expansion_scope == "full":
                waves, backscatter_waves = multislice_step(
                    waves, potential_slice, next_slice=next_slice
                )
            else:
                waves = multislice_step(waves, potential_slice, next_slice=None)
            tqdm_pbar.update_if_exists(int(n_waves))

            slice_thickness = potential_slice.axes_metadata[0].values[0]
            depth += slice_thickness

            if order_resolved:
                for n in range(n_orders):
                    _update_plasmon_axes(order_waves[n], depth)
                plasmon_operator.scatter_by_order(
                    order_waves, depth, slice_thickness
                )
            else:
                _update_plasmon_axes(waves, depth)
                if plasmon_operator is not None:
                    plasmon_operator.scatter(waves, depth, slice_thickness)

            if potential_slice.exit_planes:
                config_meas_index = _validate_potential_ensemble_indices(
                    potential_index, exit_plane_index, potential
                )

                if measurements is not None:
                    if order_resolved:
                        _detect_order_resolved(
                            order_waves,
                            plasmon_target_norm,
                            detectors,
                            measurements,
                            config_meas_index,
                        )
                    elif algorithm.expansion_scope == "full" and return_backscattered:
                        _update_measurements(
                            waves, detectors[:-1], measurements[:-1], config_meas_index
                        )
                        _update_measurements(
                            backscatter_waves,
                            detectors[-1:],
                            measurements[-1:],
                            config_meas_index,
                        )
                    else:
                        detect_waves = waves
                        if plasmon_operator is not None:
                            detect_waves = _renormalize_total_intensity(
                                waves, plasmon_target_norm
                            )
                        _update_measurements(
                            detect_waves, detectors, measurements, config_meas_index
                        )
                exit_plane_index += 1

    # Handle final output if not using intermediate measurements
    if measurements is None:
        if order_resolved:
            _renormalize_order_waves(order_waves, plasmon_target_norm)
            measurements = [
                _stack_order_detections(
                    order_waves, detector, potential.ensemble_shape, order_axis
                )
                for detector in detectors
            ]
        else:
            if plasmon_operator is not None:
                waves = _renormalize_total_intensity(waves, plasmon_target_norm)
            measurements = [
                detector.detect(waves)[(None,) * len(potential.ensemble_shape)]
                for detector in detectors
            ]

    elif return_backscattered:
        _back_propagate_backscattered_waves(
            measurements[-1],  # type: ignore
            potential,
            multislice_step,
        )

    tqdm_pbar.close_if_exists()

    return measurements


def _aggregate_slices_by_exit_planes(potential_slices, exit_planes):
    """
    Group potential slices between exit_planes, summing their thicknesses.

    Parameters
    ----------
    potential_slices : list of PotentialSlice
        Original slices along the beam direction.
    exit_planes : list of int
        Indices of exit planes (first can be -1 for entrance plane).

    Returns
    -------
    effective_slices : list of PotentialSlice
        Aggregated slices with summed potential arrays and summed thicknesses.
    """

    effective_slices = []

    for i in range(0, len(exit_planes) - 1):
        idx_start = exit_planes[i] + 1  # slice after previous exit plane
        idx_end = exit_planes[i + 1] + 1  # include this exit plane

        # Aggregate slices in this block
        combined_slice = potential_slices[idx_start].copy()
        thickness = combined_slice.slice_thickness[0]
        # Add remaining slices in the block
        for in_bw_slice in potential_slices[idx_start + 1 : idx_end]:
            combined_slice += in_bw_slice
            thickness += in_bw_slice.slice_thickness[0]
            combined_slice._slice_thickness = (thickness,)
            combined_slice._slice_limits = [(0, thickness)]

        effective_slices.append(combined_slice)

    return effective_slices


def _back_propagate_backscattered_waves(
    backscattered_waves: Waves,
    potential: BasePotential,
    multislice_step: Callable,
) -> Waves:
    """
    For each slice in the multislice step, a small part of the wave get backscattered.
    This function runs the multislice in reverse for each backscattered wave summing
    them for a final backscattered wave result.
    """

    xp = get_array_module(backscattered_waves.device)
    potential_slices = [
        slice
        for _, config in _generate_potential_configurations(potential)
        for slice in config.generate_slices()
    ]

    effective_slices = _aggregate_slices_by_exit_planes(
        potential_slices, potential.exit_planes
    )

    num_slices = len(effective_slices)
    if len(backscattered_waves) != num_slices + 1:
        raise ValueError("Wrong shapes")

    # zero intensity in incoming wave
    backscattered_waves[0]._array[:] = 0

    # Go through potential in reverse
    for i in range(num_slices - 2, -1, -1):
        contribution_at_slice = backscattered_waves[i + 1].copy()
        contribution_at_slice.array = xp.conj(contribution_at_slice.array)
        contribution_at_slice, _ = multislice_step(
            contribution_at_slice, effective_slices[i + 1], next_slice=None
        )
        backscattered_waves[i].array += xp.conj(contribution_at_slice.array)

    return backscattered_waves


def transition_potential_multislice_and_detect(
    waves: Waves,
    potential: BasePotential,
    transition_potential: TransitionPotential | TransitionPotentialArray,
    detectors: Optional[list[BaseDetector]] = None,
    detectors_elastic: Optional[list[BaseDetector]] = None,
    double_channel: bool = True,
    threshold: float = 1.0,
    sites: Optional[SliceIndexedAtoms | Atoms] = None,
    algorithm: FourierMultislice | RealSpaceMultislice = FourierMultislice(),
    scatter_max_batch: int | str = 1,
    pbar: bool = False,
) -> list[BaseMeasurements | Waves] | BaseMeasurements | Waves:
    """
    Calculate the full multislice algorithm for the given batch of wave functions
    through a given potential, detecting at each of the exit planes specified in the
    potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be
        converted to measurements after running the multislice algorithm.
    algorithm: FourierMultislice or RealSpaceMultislice, optional
        Algorithm used for multislice operator (default is FourierMultislice())

    Returns
    -------
    measurements : Waves or tuple of :class:`.BaseMeasurement`
        Exit waves or detected measurements or lists of measurements.
    """

    def _update_loss_measurements(
        measurements, waves, detectors, potential, slice_index, potential_index
    ):
        if slice_index in potential.exit_planes:
            exit_plane_index = potential.exit_planes.index(slice_index)

            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )

            for i, detector in enumerate(detectors):
                new_measurement = detector.detect(waves)
                new_measurement = new_measurement.sum((0,))
                measurements[i].array[measurement_index] += new_measurement.array

    waves = waves.ensure_real_space()

    if isinstance(algorithm, FourierMultislice):
        antialias_aperture = AntialiasAperture()
        propagator = FresnelPropagator()

        def multislice_step(waves, potential_slice):
            return conventional_multislice_step(
                waves,
                potential_slice=potential_slice,
                antialias_aperture=antialias_aperture,
                propagator=propagator,
                conjugate=algorithm.conjugate,
                transpose=algorithm.transpose,
                order=algorithm.order,
            )

    else:
        laplace_operator = LaplaceOperator(algorithm.derivative_accuracy)

        def multislice_step(waves, potential_slice):
            return realspace_multislice_step(
                waves,
                potential_slice=potential_slice,
                next_slice=None,
                laplace=laplace_operator,
                max_terms=algorithm.max_terms,
                order=algorithm.order,
                fully_corrected=algorithm.expansion_scope == "full",
            )

    if detectors is None:
        detectors = [WavesDetector()]

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    measurements = allocate_multislice_measurements(
        waves,
        detectors,
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    )

    transition_potential.grid.match(waves)
    transition_potential.accelerator.match(waves)

    if isinstance(transition_potential, TransitionPotential):
        transition_potential = transition_potential.build()

    transition_potential = transition_potential.copy_to_device(waves.device)

    # CrystalPotential implements get_sliced_atoms by tiling its unit, so the
    # first branch covers repeating-unit potentials too (see
    # CrystalPotential.get_sliced_atoms).
    if sites is None and hasattr(potential, "get_sliced_atoms"):
        sites = potential.get_sliced_atoms()
    elif sites is None and hasattr(potential, "atoms"):
        sites = potential.atoms

    if isinstance(sites, Atoms):
        sites = SliceIndexedAtoms(sites, slice_thickness=potential.slice_thickness)
    elif not isinstance(sites, SliceIndexedAtoms):
        raise ValueError(
            "Could not derive scattering sites from the potential "
            f"({type(potential).__name__}). Pass ``sites=`` explicitly as an "
            "ase.Atoms or SliceIndexedAtoms covering the full simulation cell."
        )

    n_sites = np.sum(sites.atoms.numbers == transition_potential.Z)

    if n_sites == 0:
        raise RuntimeError(
            "No scattering sites matching transition potential for element"
            f"{transition_potential.Z}"
        )

    absolute_threshold = transition_potential.absolute_threshold(
        waves, threshold=threshold
    )

    n_waves = np.prod(waves.shape[:-2])
    n_slices = n_waves * potential.num_slices * potential.num_configurations

    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=int(n_slices), leave=False, desc="multislice"
    )

    waves_input = waves.copy()

    for (
        potential_index,
        potential_configuration,
    ) in _generate_potential_configurations(potential):
        waves = waves_input.copy()
        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, 0, potential
            )
            _update_measurements(waves, detectors, measurements, measurement_index)

        # The double-channel inner multislice re-visits slices [scatter_index+1 …]
        # once per site batch; pre-building (and bandlimiting) the transmission
        # functions saves N_sites rebuilds per outer step in that case (for
        # FourierMultislice the cache short-circuits the rebuild inside
        # conventional_multislice_step, see iam.py:1300-1302). Single-channel
        # visits each slice exactly once, so caching is pure memory overhead and
        # we stream slices instead.
        if double_channel:
            if isinstance(algorithm, FourierMultislice):
                # Dedup transmissions across z-repetitions. CrystalPotential's
                # tile cache (iam.py CrystalPotential.generate_slices) yields the
                # *same* PotentialArray object for every z-repetition of a unit
                # slice in the no-frozen-phonon case, so id(slice_obj) collapses
                # to one entry per unique unit slice. The bandlimit FFT then
                # runs O(n_unique) times instead of O(n_outer), and the
                # transmission cache footprint drops by repetitions[2].
                # For SrTiO3 reps=(4,4,25): 50 transmissions -> 2 unique
                # (-24 MB cache, -48 bandlimit FFTs per configuration).
                # The EELS driver reads exit_planes off ``potential`` globally,
                # never off the slice (compare standard_multislice_and_detect
                # at multislice.py:672), so sharing TransmissionFunction
                # instances across slice indices is safe here.
                tx_dedup: dict[int, TransmissionFunction] = {}
                slice_cache = []
                for slice_obj in potential_configuration.generate_slices():
                    key = id(slice_obj)
                    cached = tx_dedup.get(key)
                    if cached is None:
                        cached = antialias_aperture.bandlimit(
                            slice_obj.transmission_function(energy=waves._valid_energy),
                            in_place=False,
                        )
                        tx_dedup[key] = cached
                    slice_cache.append(cached)
            else:
                slice_cache = list(potential_configuration.generate_slices())
            n_outer = len(slice_cache)
            outer_iter = enumerate(slice_cache)
        else:
            slice_cache = None
            n_outer = None
            outer_iter = enumerate(potential_configuration.generate_slices())

        depth = 0.0
        for scatter_index, potential_slice in outer_iter:
            waves = multislice_step(
                waves,
                potential_slice,
            )
            depth += potential_slice.axes_metadata[0].values[0]

            _update_plasmon_axes(waves, depth)

            sites_slice = sites.get_atoms_in_slices(
                scatter_index, atomic_number=transition_potential.Z
            )

            tqdm_pbar.update_if_exists(int(n_waves))

            if len(sites_slice) == 0:
                continue

            for (
                included_sites,
                scattered_waves,
            ) in transition_potential.generate_scattered_waves(
                waves,
                sites_slice,
                max_batch=scatter_max_batch,
                threshold=absolute_threshold,
            ):
                if len(scattered_waves) == 0:
                    continue

                if double_channel:
                    _update_loss_measurements(
                        measurements,
                        scattered_waves,
                        detectors,
                        potential,
                        scatter_index,
                        potential_index,
                    )

                    # Nothing left to propagate through on the final outer slice.
                    if scatter_index + 1 == n_outer:
                        continue

                    for inner_offset, inner_potential_slice in enumerate(
                        slice_cache[scatter_index + 1 :]
                    ):
                        scattered_waves = multislice_step(
                            scattered_waves,
                            inner_potential_slice,
                        )

                        _update_plasmon_axes(waves, depth)

                        _update_loss_measurements(
                            measurements,
                            scattered_waves,
                            detectors,
                            potential,
                            scatter_index + 1 + inner_offset,
                            potential_index,
                        )

                else:
                    exit_plane_index = bisect_left(potential.exit_planes, scatter_index)

                    measurement_plane_indices: tuple[slice] | tuple = ()
                    if len(potential.exit_planes) > 1:
                        exit_planes = slice(
                            exit_plane_index, len(potential.exit_planes)
                        )
                        measurement_plane_indices = (exit_planes,)

                    for i, detector in enumerate(detectors):
                        new_measurement = detector.detect(scattered_waves).sum((0,))
                        measurements[i].array[
                            measurement_plane_indices
                        ] += new_measurement.array[
                            (None,) * len(measurement_plane_indices)
                        ]

    tqdm_pbar.close_if_exists()

    return measurements


def is_waves_base_measurements_or_list(
    value: Any,
) -> TypeGuard["Waves | BaseMeasurements | list[Waves | BaseMeasurements]"]:
    waves_class_name = "Waves"
    base_measurements_class_name = "BaseMeasurements"
    waves_module_name = "abtem.waves"
    base_measurements_module_name = "abtem.measurements"

    def is_instance_of_waves_or_base_measurements(obj: Any) -> bool:
        return (
            obj.__class__.__name__ == waves_class_name
            and obj.__class__.__module__ == waves_module_name
        ) or (
            obj.__class__.__name__ == base_measurements_class_name
            and obj.__class__.__module__ == base_measurements_module_name
        )

    if is_instance_of_waves_or_base_measurements(value):
        return True
    if isinstance(value, list) and all(
        is_instance_of_waves_or_base_measurements(item) for item in value
    ):
        return True
    return False


class MultisliceTransform(WavesTransform[BaseMeasurements]):
    """
    Transformation applying the multislice algorithm to wave functions, producing new
    wave functions or measurements.

    Parameters
    ----------
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be
        converted to measurements after running the multislice algorithm.
    multislice_func : callable, optional
        The multislice function defining the multislice algorithm used
        (default is :func:`.multislice_and_detect`).
    **multislice_func_kwargs
        Additional keyword arguments passed to the multislice function.
    """

    def __init__(
        self,
        potential: BasePotential,
        detectors: Optional[BaseDetector | list[BaseDetector]] = None,
        multislice_func: Optional[Callable] = None,
        **multislice_func_kwargs,
    ):
        if multislice_func is None:
            multislice_func = multislice_and_detect

        potential = validate_potential(potential)

        self._potential = potential

        detectors = validate_detectors(detectors)
        self._user_detectors = detectors

        if multislice_func_kwargs.get("return_backscattered", False):
            detectors = detectors + [WavesDetector()]

        if "pbar" not in multislice_func_kwargs:
            multislice_func_kwargs["pbar"] = config.get(
                "diagnostics.task_progress", False
            )

        self._detectors = detectors
        self._multislice_func = multislice_func
        self._multislice_func_kwargs = multislice_func_kwargs

    @property
    def multislice_func(self) -> Callable:
        """The multislice function defining the multislice algorithm used."""
        return self._multislice_func

    @property
    def _num_outputs(self):
        return len(self._detectors)

    @property
    def potential(self) -> BasePotential:
        """Electrostatic potential for each multislice slice."""
        return self._potential

    @property
    def detectors(self) -> list[BaseDetector]:
        """List of detectors defining how the wave functions should be converted to
        measurements."""
        return self._detectors

    @property
    def _plasmon_order_axis(self):
        plasmons = self._multislice_func_kwargs.get("plasmons")
        if plasmons is not None and plasmons.max_loss_order is not None:
            n = plasmons.max_loss_order + 1
            labels = ("Zero loss",) + tuple(
                f"{i}-plasmon" for i in range(1, n)
            )
            return OrdinalAxis(label="Plasmon order", values=labels)
        return None

    @property
    def ensemble_axes_metadata(self):
        order_axis = self._plasmon_order_axis
        order_meta = [order_axis] if order_axis is not None else []

        ensemble_axes_metadata = self.potential.ensemble_axes_metadata

        if len(self.potential.exit_planes) > 1:
            exit_planes_metadata = [self.potential._get_exit_planes_axes_metadata()]
        else:
            exit_planes_metadata = []

        ensemble_axes_metadata = [
            *order_meta,
            *ensemble_axes_metadata,
            *exit_planes_metadata,
        ]
        return ensemble_axes_metadata

    @property
    def ensemble_shape(self):
        order_axis = self._plasmon_order_axis
        order_shape = (len(order_axis.values),) if order_axis is not None else ()

        ensemble_shape = self._potential.ensemble_shape
        if len(self._potential.exit_planes) > 1:
            ensemble_shape = (*ensemble_shape, len(self._potential.exit_planes))
        return order_shape + ensemble_shape

    def _out_metadata(self, waves: Waves) -> tuple[dict, ...]:
        return tuple(detector._out_metadata(waves)[0] for detector in self.detectors)

    def _out_dtype(self, waves: Waves) -> tuple[np.dtype, ...]:
        return tuple(detector._out_dtype(waves)[0] for detector in self.detectors)

    def _out_meta(self, waves: Waves) -> tuple[np.ndarray, ...]:
        return tuple(detector._out_meta(waves)[0] for detector in self.detectors)

    def _out_type(self, waves: Waves) -> tuple[type, ...]:
        return tuple(detector._out_type(waves)[0] for detector in self.detectors)

    def _out_ensemble_shape(self, waves: Waves) -> tuple[tuple[int, ...], ...]:
        shape = tuple(
            self.ensemble_shape + detector._out_ensemble_shape(waves)[0]
            for detector in self.detectors
        )
        return shape

    def _out_base_shape(self, waves: Waves) -> tuple[tuple[int, ...], ...]:
        base_shape = tuple(
            detector._out_base_shape(waves)[0] for detector in self.detectors
        )
        return base_shape

    def _out_base_axes_metadata(self, waves: Waves) -> tuple[list[AxisMetadata], ...]:
        return tuple(
            detector._out_base_axes_metadata(waves)[0] for detector in self.detectors
        )

    def _out_ensemble_axes_metadata(
        self, waves: Waves
    ) -> tuple[list[AxisMetadata], ...]:
        order_axis = self._plasmon_order_axis
        order_meta = [order_axis] if order_axis is not None else []

        if len(self.potential.exit_planes) > 1:
            potential_axes_metadata = self.potential.ensemble_axes_metadata + [
                self.potential._get_exit_planes_axes_metadata()
            ]
        else:
            potential_axes_metadata = self.potential.ensemble_axes_metadata

        ensemble_axes_metadata = tuple(
            order_meta + potential_axes_metadata
            + detector._out_ensemble_axes_metadata(waves)[0]
            for detector in self.detectors
        )

        return ensemble_axes_metadata

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        chunks: tuple[int, ...] = ()

        order_axis = self._plasmon_order_axis
        if order_axis is not None:
            chunks = chunks + (len(order_axis.values),)

        if len(self.potential.ensemble_shape) > 0:
            chunks = chunks + (1,)

        if len(self.potential.exit_planes) > 1:
            chunks = chunks + (len(self.potential.exit_planes),)

        return chunks

    def _validate_ensemble_chunks(
        self, chunks: Optional[Chunks] = None, limit: str | int = "auto"
    ) -> ValidatedChunks:
        if chunks is None:
            chunks = self._default_ensemble_chunks

        if (
            isinstance(chunks, int)
            and len(self.ensemble_shape) > 1
            and self.potential.num_exit_planes > 1
        ):
            chunks = (chunks, self.potential.num_exit_planes)

        chunks = validate_chunks(self.ensemble_shape, chunks, max_elements=limit)

        if self.potential.num_exit_planes > 1:
            chunks = chunks[:-1] + ((self.potential.num_exit_planes,),)

        return chunks

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        chunks = self._validate_ensemble_chunks(chunks)

        # Strip the order-axis chunk (not a potential dimension).
        order_axis = self._plasmon_order_axis
        pot_chunks = chunks
        if order_axis is not None:
            pot_chunks = pot_chunks[1:]

        if self.potential.num_exit_planes > 1:
            pot_chunks = pot_chunks[:-1]

        args = self._potential._partition_args(chunks=pot_chunks, lazy=lazy)

        if len(self._potential.exit_planes) > 1:
            args = (args[0][..., None],)

        # Prepend a trivial dimension for the order axis so blockwise
        # broadcasting works (the order dimension is produced entirely
        # inside multislice_and_detect, not partitioned here).
        if order_axis is not None:
            args = (args[0][None],)

        return args

    @staticmethod
    def _multislice_transform_member(*args, potential_partial: Callable, **kwargs):
        args = unpack_blockwise_args(args)

        potential = potential_partial(*args)
        potential = potential.item()
        transform = MultisliceTransform(potential, **kwargs)

        ndims = len(transform.ensemble_shape)
        wrapped_transform = _wrap_with_array(transform, ndims)
        return wrapped_transform

    def _from_partitioned_args(self) -> Callable:
        potential_partial = self._potential._from_partitioned_args()
        return partial(
            self._multislice_transform_member,
            potential_partial=potential_partial,
            detectors=self._user_detectors,
            multislice_func=self.multislice_func,
            **self._multislice_func_kwargs,
        )

    def _calculate_new_array(self, waves: Waves):
        measurements = self.multislice_func(
            waves=waves,
            potential=self.potential,
            detectors=self.detectors,
            **self._multislice_func_kwargs,
        )

        if len(measurements) != len(self.detectors):
            raise RuntimeError(
                f"Expected {len(self.detectors)} outputs, got {len(measurements)}"
            )

        arrays = tuple(measurement.array for measurement in measurements)
        if len(arrays) == 1:
            arrays = arrays[0]

        return arrays

    def apply(
        self, waves: Waves, max_batch: int | str = "auto"
    ) -> Waves | BaseMeasurements | list[Waves | BaseMeasurements]:
        """
        Run the multislice algorithm on the given wave functions. An output is returned
        for each detector.

        Parameters
        ----------
        waves : Waves
            The wave functions to run the multislice algorithm on.
        max_batch : int or str, optional
            The maximum batch size to use for the multislice algorithm. If 'auto' the
            batch size is chosen automatically based on the available memory.

        Returns
        -------
        waves : tuple of Waves and BaseMeasurements
            The wave functions after running the multislice algorithm.
        """
        output = waves.apply_transform(self, max_batch=max_batch)
        # assert is_waves_base_measurements_or_list(output)

        return output
