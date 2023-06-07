"""Module for running the multislice algorithm."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.antialias import antialias_aperture
from abtem.core.axes import AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.config import config
from abtem.core.energy import energy2wavelength
from abtem.core.fft import fft2_convolve, CachedFFTWConvolution
from abtem.core.grid import spatial_frequencies
from abtem.core.utils import expand_dims_to_broadcast
from abtem.detectors import BaseDetector
from abtem.inelastic.plasmons import _update_plasmon_axes
from abtem.measurements import BaseMeasurements
from abtem.potentials.iam import (
    BasePotential,
    TransmissionFunction,
    PotentialArray,
)
from abtem.tilt import _get_tilt_axes

if TYPE_CHECKING:
    from abtem.waves import Waves


def _fresnel_propagator_array(
    thickness: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    energy: float,
    device: str,
):
    xp = get_array_module(device)
    wavelength = energy2wavelength(energy)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    kx, ky = kx[:, None], ky[None]

    f = complex_exponential(
        -(kx**2) * np.pi * thickness * wavelength
    ) * complex_exponential(-(ky**2) * np.pi * thickness * wavelength)
    return f


def _apply_tilt_to_fresnel_propagator_array(
    array: np.ndarray,
    sampling: tuple[float, float],
    thickness: float,
    tilt: tuple[float, float] | tuple[tuple[float, float], ...],
):
    xp = get_array_module(array)
    tilt = xp.array(tilt)

    remove_first_dim = False
    if tilt.shape == (2,):
        remove_first_dim = True

    kx, ky = spatial_frequencies(array.shape[-2:], sampling, xp=xp)
    kx, ky = kx[None, :, None], ky[None, None]

    tilt = complex_exponential(
        -kx * xp.tan(tilt[:, 0, None, None] / 1e3) * thickness * 2 * np.pi
    ) * complex_exponential(
        -ky * xp.tan(tilt[:, 1, None, None] / 1e3) * thickness * 2 * np.pi
    )

    tilt, array = expand_dims_to_broadcast(tilt, array, match_dims=[(-2, -1), (-2, -1)])

    array = tilt * array

    if remove_first_dim:
        array = array[0]

    return array


class FresnelPropagator:
    """
    The Fresnel propagator is used for propagating wave functions using the near-field approximation
    (Fresnel diffraction).
    """

    def __init__(self):
        self._array = None
        self._key = None
        self._cached_fftw_convolution = CachedFFTWConvolution()

    def get_array(self, waves: Waves, thickness: float) -> np.ndarray:

        key = (
            waves.gpts,
            waves.sampling,
            thickness,
            waves.base_tilt,
            waves.energy,
            waves.device,
        )

        tilt_axes = _get_tilt_axes(waves)
        tilt_axes_metadata = [waves.ensemble_axes_metadata[i] for i in tilt_axes]
        if tilt_axes:
            key += (copy.deepcopy(tilt_axes_metadata),)

        if key == self._key:
            return self._array

        self._array = self._calculate_array(waves, thickness)
        self._key = key

        return self._array

    def _calculate_array(self, waves: "Waves", thickness: float) -> np.ndarray:

        array = _fresnel_propagator_array(
            thickness=thickness,
            gpts=waves.gpts,
            sampling=waves.sampling,
            energy=waves.energy,
            device=waves.device,
        )

        array *= antialias_aperture(
            waves.gpts,
            waves.sampling,
            get_array_module(waves.device),
        )

        if waves.base_tilt != (0.0, 0.0):
            array = _apply_tilt_to_fresnel_propagator_array(
                array,
                sampling=waves.sampling,
                thickness=thickness,
                tilt=waves.base_tilt,
            )

        xp = get_array_module(waves.device)

        tilt_axes = _get_tilt_axes(waves)

        if not tilt_axes:
            return array

        for axis in waves.ensemble_axes_metadata:

            if hasattr(axis, "tilt"):
                tilt = xp.array(axis.tilt)
                array = _apply_tilt_to_fresnel_propagator_array(
                    array, sampling=waves.sampling, tilt=tilt, thickness=thickness
                )

            else:
                array = array[..., None, :, :]

        return array

    def propagate(
        self, waves: Waves, thickness: float, in_place: bool = False
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
        kernel = self.get_array(waves, thickness)

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


def _allocate_measurement(
    waves: Waves,
    detector: BaseDetector,
    extra_ensemble_axes_shape: tuple[int, ...],
    extra_ensemble_axes_metadata: list[AxisMetadata],
) -> BaseMeasurements:
    xp = get_array_module(detector._out_meta(waves))

    measurement_type = detector._out_type(waves)

    axes_metadata = detector._out_axes_metadata(waves)

    shape = detector._out_shape(waves)

    if extra_ensemble_axes_shape is not None:
        assert len(extra_ensemble_axes_shape) == len(extra_ensemble_axes_shape)
        shape = extra_ensemble_axes_shape + shape
        axes_metadata = extra_ensemble_axes_metadata + axes_metadata

    metadata = detector._out_metadata(waves)

    array = xp.zeros(shape, dtype=detector._out_dtype(waves))
    return measurement_type.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=metadata
    )


def _potential_ensemble_shape_and_metadata(potential):
    if potential is None:
        return ()

    extra_ensemble_axes_shape = potential.ensemble_shape
    extra_ensemble_axes_metadata = potential.ensemble_axes_metadata

    if len(potential.exit_planes) > 1:
        extra_ensemble_axes_shape = extra_ensemble_axes_shape + (
            len(potential.exit_planes),
        )
        extra_ensemble_axes_metadata = extra_ensemble_axes_metadata + [
            potential._get_exit_planes_axes_metadata()
        ]

    return extra_ensemble_axes_shape, extra_ensemble_axes_metadata


def allocate_multislice_measurements(
    waves: Waves,
    detectors: list[BaseDetector],
    extra_ensemble_axes_shape: tuple[int, ...],
    extra_ensemble_axes_metadata: list[AxisMetadata],
) -> dict[BaseDetector, BaseMeasurements]:
    """
    Allocate the multislice measurements that would be produced by detecting the given set of wave functions with the
    given set of detectors.

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
    allocated_measurements : dict
        Mapping of detectors to measurements.
    """

    measurements = {}
    for detector in detectors:
        measurements[detector] = _allocate_measurement(
            waves, detector, extra_ensemble_axes_shape, extra_ensemble_axes_metadata
        )

    return measurements


def multislice_step(
    waves: Waves,
    potential_slice: PotentialArray | TransmissionFunction,
    propagator: FresnelPropagator,
    antialias_aperture: AntialiasAperture,
    conjugate: bool = False,
    transpose: bool = False,
) -> Waves:
    """
    Calculate one step of the multislice algorithm for the given batch of wave functions through a given potential slice.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential_slice : PotentialArray or TransmissionFunction
        A potential slice as a :class:`.PotentialArray` or :class:`.TransmissionFunction`.
    propagator : FresnelPropagator, optional
        A Fresnel propagator type matching the wave functions. The main reason for using this argument is to reuse
        a previously calculated propagator. If not provided a new propagator is created.
    antialias_aperture : AntialiasAperture, optional
        An antialias aperture type matching the wave functions. The main reason for using this argument is to reuse
        a previously calculated antialias aperture. If not provided a new antialias aperture is created.
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
            energy=waves.energy
        )
        transmission_function = antialias_aperture.bandlimit(
            transmission_function, overwrite_x=True
        )

    thickness = transmission_function.slice_thickness[0]

    if conjugate:
        thickness = -thickness

    if transpose:
        waves = propagator.propagate(waves, thickness=thickness, in_place=True)
        waves = transmission_function.transmit(waves, conjugate=conjugate)
    else:
        waves = transmission_function.transmit(waves, conjugate=conjugate)
        waves = propagator.propagate(waves, thickness=thickness, in_place=True)

    return waves


def _update_measurements(
    waves: Waves,
    detectors: list[BaseDetector],
    measurements: dict[BaseDetector, BaseMeasurements],
    measurement_index: tuple[int, ...],
    additive: bool = False,
):
    if measurements is None:
        return

    for detector in detectors:
        new_measurement = detector.detect(waves)

        if detector in measurements.keys():
            if additive:
                measurements[detector].array[measurement_index] += new_measurement.array
            else:
                measurements[detector].array[measurement_index] = new_measurement.array
        else:
            measurements[detector] = new_measurement

    return measurements


def _validate_potential_ensemble_indices(
    potential_index: int, exit_plane_index: int, potential: BasePotential
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


def multislice_and_detect(
    waves: Waves,
    potential: BasePotential,
    detectors: list[BaseDetector] = None,
    conjugate: bool = False,
    transpose: bool = False,
) -> tuple[BaseMeasurements | Waves, ...] | BaseMeasurements | Waves:
    """
    Calculate the full multislice algorithm for the given batch of wave functions through a given potential, detecting
    at each of the exit planes specified in the potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be converted to measurements after
        running the multislice algorithm.
    conjugate : bool, optional
        If True, use the complex conjugate of the transmission function (default is False).
    transpose : bool, optional
        If True, reverse the order of propagation and transmission (default is False).

    Returns
    -------
    measurements : Waves or tuple of :class:`.BaseMeasurement`
        Exit waves or detected measurements or lists of measurements.
    """
    antialias_aperture = AntialiasAperture()

    propagator = FresnelPropagator()
    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    if sum(extra_ensemble_axes_shape) == 1:
        measurements = None
    else:
        measurements = allocate_multislice_measurements(
            waves,
            detectors,
            extra_ensemble_axes_shape,
            extra_ensemble_axes_metadata,
        )

    for potential_index, _, potential_configuration in potential.generate_blocks():

        exit_plane_index = 0
        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential_configuration
            )

            measurements = _update_measurements(
                waves, detectors, measurements, measurement_index
            )
            exit_plane_index += 1

        depth = 0
        for potential_slice in potential_configuration.generate_slices():

            waves = multislice_step(
                waves,
                potential_slice,
                propagator,
                antialias_aperture,
                conjugate=conjugate,
                transpose=transpose,
            )

            depth += potential_slice.axes_metadata[0].values[0]

            _update_plasmon_axes(waves, depth)

            if potential_slice.exit_planes:
                measurement_index = _validate_potential_ensemble_indices(
                    potential_index, exit_plane_index, potential_configuration
                )

                if measurements is None:
                    measurements = {
                        detector: detector.detect(waves)[
                            (None,) * len(potential.ensemble_shape)
                        ]
                        for detector in detectors
                    }
                else:
                    measurements = _update_measurements(
                        waves, detectors, measurements, measurement_index
                    )

                exit_plane_index += 1

    measurements = tuple(measurements.values())

    return measurements


def transition_potential_multislice_and_detect(
    waves: Waves,
    potential: BasePotential,
    detectors: list[BaseDetector],
    sites,
    transition_potentials,
    conjugate: bool = False,
    transpose: bool = False,
) -> tuple[BaseMeasurements | Waves, ...] | BaseMeasurements | Waves:
    """
    Calculate the full multislice algorithm for the given batch of wave functions through a given potential, detecting
    at each of the exit planes specified in the potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential : BasePotential
        A potential as :class:`BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be converted to measurements after
        running the multislice algorithm.
    conjugate : bool, optional
        If True, use the conjugate of the transmission function (default is False).
    transpose : bool, optional
        If True, reverse the order of propagation and transmission (default is False).

    Returns
    -------
    measurements : Waves or (list of) :class:`.BaseMeasurement`
        Exit waves or detected measurements or lists of measurements.
    """

    antialias_aperture = AntialiasAperture()
    propagator = FresnelPropagator()

    measurements = allocate_multislice_measurements(
        waves,
        detectors,
        potential,
    )

    for potential_index, _, potential_configuration in potential.generate_blocks():
        slice_generator = potential_configuration.generate_slices()

        for scatter_index, (scatter_slice, sites_slice) in enumerate(
            zip(slice_generator, sites)
        ):

            waves = multislice_step(
                waves,
                scatter_slice,
                propagator,
                antialias_aperture,
                conjugate=conjugate,
                transpose=transpose,
            )

            sites_slice = transition_potentials.validate_sites(sites_slice)

            for _, scattered_waves in transition_potentials.generate_scattered_waves(
                waves, sites_slice
            ):

                for depth, potential_slice in potential_configuration.generate_slices(
                    first_slice=scatter_index, return_depth=True
                ):

                    scattered_waves = multislice_step(
                        scattered_waves,
                        potential_slice,
                        propagator,
                        antialias_aperture,
                        conjugate=conjugate,
                        transpose=transpose,
                    )

                    if potential_slice.exit_planes:
                        exit_plane_index = np.searchsorted(
                            potential.exit_thicknesses, depth - 1e-6
                        )

                        measurement_index = _validate_potential_ensemble_indices(
                            potential_index, exit_plane_index, potential_configuration
                        )

                        for detector in detectors:
                            new_measurement = detector.detect(scattered_waves).mean(0)
                            measurements[detector].array[
                                measurement_index
                            ] += new_measurement.array

    measurements = tuple(measurements.values())

    return measurements
