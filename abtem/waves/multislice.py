from typing import TYPE_CHECKING, Union, Tuple, List, Dict

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import AxisMetadata, AxisAlignedTiltAxis
from abtem.core.backend import get_array_module, HasDevice
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.events import Events, watch, HasEventsMixin
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.core.utils import expand_dims_to_match
from abtem.measure.detect import AbstractDetector
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import (
    AbstractPotential,
    TransmissionFunction,
    PotentialArray,
)
from abtem.structures.transform import plane_to_axes
from abtem.waves.base import WavesLikeMixin

if TYPE_CHECKING:
    from abtem.waves.waves import Waves


def fresnel_propagator(
    thickness: float,
    tilt: Tuple[float, float],
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    energy: float,
    device: str,
):
    xp = get_array_module(device)
    wavelength = energy2wavelength(energy)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    kx, ky = kx[:, None], ky[None]

    f = complex_exponential(
        -(kx ** 2) * np.pi * thickness * wavelength
    ) * complex_exponential(-(ky ** 2) * np.pi * thickness * wavelength)

    if tilt != (0.0, 0.0):
        f *= complex_exponential(
            -kx * xp.tan(tilt[0] / 1e3) * thickness * 2 * np.pi
        ) * complex_exponential(-ky * xp.tan(tilt[1] / 1e3) * thickness * 2 * np.pi)

    return f


def tilt_shift(k, tilt, thickness):
    xp = get_array_module(k)
    return complex_exponential(-k * xp.tan(tilt / 1e3) * thickness * 2 * np.pi)


class FresnelPropagator:
    def __init__(self):
        """
        The fresnel propagtor is used for propagating wave functions using the near-field approximation
        (Fresnel diffraction).
        """
        self._array = None
        self._key = None

    def _get_array(self, waves: "Waves", thickness: float) -> np.ndarray:

        key = (
            waves.gpts,
            waves.sampling,
            thickness,
            waves.base_tilt,
            waves.energy,
            waves.device,
        )

        if waves.tilt_axes:
            key += (waves.tilt_axes_metadata,)

        if key == self._key:
            return self._array

        self._array = self._calculate_array(waves, thickness)
        self._key = key

        return self._array

    def _calculate_array(self, waves: "Waves", thickness: float) -> np.ndarray:

        antialias_aperture = AntialiasAperture(device=waves.device)
        antialias_aperture.grid.match(waves)
        array = antialias_aperture.array.astype(np.complex64)

        array *= fresnel_propagator(
            gpts=waves.gpts,
            sampling=waves.sampling,
            thickness=thickness,
            tilt=waves.base_tilt,
            energy=waves.energy,
            device=waves.device,
        )

        xp = get_array_module(waves.device)

        if not waves.tilt_axes:
            return array

        for i in range(len(waves.ensemble_shape)):
            axis = waves.ensemble_axes_metadata[i]
            if isinstance(axis, AxisAlignedTiltAxis):
                j = plane_to_axes(axis.direction)[0]
                k = spatial_frequencies((waves.gpts[j],), (waves.sampling[j],), xp=xp)
                tilts = xp.array(axis.values)[:, None]
                shift = tilt_shift(k[0][None], tilts, thickness)
                shift, array = expand_dims_to_match(
                    shift, array, match_dims=[(-1,), (-2 + j,)]
                )
                array = array * shift
            else:
                array = array[..., None, :, :]

        return array

    def propagate(
        self, waves: "Waves", thickness: float, overwrite_x: bool = False
    ) -> "Waves":
        """
        Propagate wave functions.

        Parameters
        ----------
        waves : Waves
            The wave functions to propagate.
        overwrite_x : bool
            If True, the wave functions may be overwritten.

        Returns
        -------
        propagated_wave_functions : Waves
        """
        array = self._get_array(waves, thickness)

        waves._array = fft2_convolve(waves.array, array, overwrite_x=overwrite_x)

        return waves


def allocate_multislice_measurements(
    waves: "WavesLikeMixin",
    detectors: List[AbstractDetector],
    extra_ensemble_axes_shape: Tuple[int, ...] = None,
    extra_ensemble_axes_metadata: List[AxisMetadata] = None,
) -> Dict[AbstractDetector, AbstractMeasurement]:
    measurements = {}
    for detector in detectors:
        xp = get_array_module(detector.measurement_meta(waves))

        measurement_type = detector.measurement_type(waves)

        axes_metadata = (
            waves.ensemble_axes_metadata + detector.measurement_axes_metadata(waves)
        )

        shape = waves.ensemble_shape + detector.measurement_shape(waves)

        if extra_ensemble_axes_shape is not None:
            assert len(extra_ensemble_axes_shape) == len(extra_ensemble_axes_metadata)
            shape = extra_ensemble_axes_shape + shape
            axes_metadata = extra_ensemble_axes_metadata + axes_metadata

        array = xp.zeros(shape, dtype=detector.measurement_dtype)
        measurements[detector] = measurement_type.from_array_and_metadata(
            array, axes_metadata=axes_metadata, metadata=waves.metadata
        )

    return measurements


def multislice_step(
    waves: "Waves",
    potential_slice: Union[PotentialArray, TransmissionFunction],
    propagator: FresnelPropagator,
    antialias_aperture: AntialiasAperture,
    conjugate: bool = False,
    transpose: bool = False,
) -> "Waves":
    """


    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a `abtem.Waves` type object.
    potential_slice : PotentialArray or TransmissionFunction
        A potential slice as a `abtem.potentials.PotentialArray` or `abtem.potentials.TransmissionFunction`.
    propagator : FresnelPropagator, optional
        A fresnel propagator type matching the wave functions. The main reason for using this argument is to reuse
        a previously calculated propagator. If not provided a new propagator is created.
    antialias_aperture : AntialiasAperture, optional
        An antialias aperture type matching the wave functions. The main reason for using this argument is to reuse
        a previously calculated antialias aperture. If not provided a new antialias aperture is created.
    conjugate : bool, optional
        If True, use the conjugate of the transmission function. Default is False.
    transpose : bool, optional
        If True, reverse the order of propagation and transmission. Default is False.

    Returns
    -------
    forward_stepped_waves : Waves
    """

    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    if isinstance(potential_slice, TransmissionFunction):
        transmission_function = potential_slice

    else:
        transmission_function = potential_slice.transmission_function(
            energy=waves.energy
        )
        transmission_function = antialias_aperture.bandlimit(transmission_function)

    thickness = transmission_function.slice_thickness[0]

    if conjugate:
        thickness = -thickness

    if transpose:
        waves = propagator.propagate(waves, thickness=thickness)
        waves = transmission_function.transmit(waves, conjugate=conjugate)
    else:
        waves = transmission_function.transmit(waves, conjugate=conjugate)
        waves = propagator.propagate(waves, thickness=thickness)

    return waves


def multislice_and_detect(
    waves: "Waves",
    potential: AbstractPotential,
    detectors: List[AbstractDetector],
    conjugate: bool = False,
    transpose: bool = False,
) -> Union[
    Tuple[Union[AbstractMeasurement, "Waves"], ...], AbstractMeasurement, "Waves"
]:
    """
    Run the multislice algorithm given a batch of wave functions and a potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a `abtem.Waves` type object.
    potential : AbstractPotential
        A potential as `abtem.potentials.AbstractPotential`.
    detectors : detector, list of detectors, optional
        A detector or a list of detectors defining how the wave functions should be converted to measurements after
        running the multislice algorithm. See abtem.measure.detect for a list of implemented detectors.
    conjugate : bool, optional
        If True, use the conjugate of the transmission function. Default is False.
    transpose : bool, optional
        If True, reverse the order of propagation and transmission. Default is False.

    Returns
    -------
    exit_waves : Waves
    """
    antialias_aperture = AntialiasAperture(device=get_array_module(waves.array))
    antialias_aperture.match_grid(waves)


    propagator = FresnelPropagator()

    if len(potential.exit_planes) > 1 or potential.ensemble_shape:
        extra_ensemble_axes_shape = potential.ensemble_shape
        extra_ensemble_axes_metadata = potential.ensemble_axes_metadata

        if len(potential.exit_planes) > 1:
            extra_ensemble_axes_shape += (len(potential.exit_planes),)
            extra_ensemble_axes_metadata += [potential.exit_planes_axes_metadata]

        measurements = allocate_multislice_measurements(
            waves,
            detectors,
            extra_ensemble_axes_shape=extra_ensemble_axes_shape,
            extra_ensemble_axes_metadata=extra_ensemble_axes_metadata,
        )

    else:
        measurements = {}

    if potential.ensemble_shape:
        potential_generator = potential.generate_blocks()
    else:
        potential_generator = ((0, (0, 1), potential) for _ in range(1))

    for potential_index, _, potential_configuration in potential_generator:

        slice_generator = potential_configuration.generate_slices()

        current_slice_index = 0
        for exit_plane_index, exit_plane in enumerate(potential.exit_planes):

            while exit_plane != current_slice_index:
                potential_slice = next(slice_generator)
                waves = multislice_step(
                    waves,
                    potential_slice,
                    propagator,
                    antialias_aperture,
                    conjugate=conjugate,
                    transpose=transpose,
                )
                current_slice_index += 1

            for detector in detectors:
                new_measurement = detector.detect(waves)

                if detector in measurements.keys():
                    index = ()

                    if potential_configuration.ensemble_shape:
                        index += (potential_index,)

                    if len(potential_configuration.exit_planes) > 1:
                        index += (exit_plane_index,)


                    measurements[detector].array[index] = new_measurement.array
                else:

                    measurements[detector] = new_measurement

    measurements = tuple(measurements.values())

    return measurements
