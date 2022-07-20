from typing import TYPE_CHECKING, Union, Tuple, List, Dict

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import TiltAxis
from abtem.core.backend import get_array_module, HasDevice
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.events import Events, watch, HasEventsMixin
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.measure.detect import AbstractDetector
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import AbstractPotential, TransmissionFunction, PotentialArray
from abtem.waves.base import WavesLikeMixin

if TYPE_CHECKING:
    from abtem.waves.waves import Waves


def fresnel_propagator(thickness: float, gpts, sampling, energy, device):
    xp = get_array_module(device)
    wavelength = energy2wavelength(energy)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)

    f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * thickness * wavelength) *
         complex_exponential(-(ky ** 2)[None] * np.pi * thickness * wavelength))
    return f


def tilt_shift(thickness, tilt, gpts, sampling, device):
    xp = get_array_module(device)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    return (complex_exponential(-kx[:, None] * xp.tan(tilt[0] / 1e3) * thickness * 2 * np.pi) *
            complex_exponential(-ky[None] * xp.tan(tilt[1] / 1e3) * thickness * 2 * np.pi))


def axis_tilt_shifts(thickness, tilt_axes, gpts, sampling, device):
    xp = get_array_module(device)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)

    kx = kx[(None,) * len(tilt_axes) + (slice(None),) + (None,)]
    ky = ky[(None,) * len(tilt_axes) + (None,) + (slice(None),)]

    shifts = ()
    for i, axis in enumerate(tilt_axes):
        if axis.direction == 'x':
            ki = kx
        elif axis.direction == 'y':
            ki = ky
        else:
            raise RuntimeError()

        slic = tuple(slice(None) if i == j else None for j in range(len(tilt_axes) + 2))
        shift = complex_exponential(-ki * xp.tan(np.array(axis.values)[slic] / 1e3) * thickness * 2 * np.pi)
        shifts += (shift,)

    return xp.prod(list(shifts))


class FresnelPropagator(HasGridMixin, HasAcceleratorMixin, HasEventsMixin, HasDevice):

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None,
                 thickness: float = .5,
                 tilt: Tuple[float, float] = (0., 0.),
                 tilt_axes: List[TiltAxis] = None,
                 device: str = 'cpu'):
        """
        The fresnel propagtor is used for propagating wave functions using the near-field approximation
        (Fresnel diffraction).

        Parameters
        ----------
        extent : one or two float
            Extent of Fresnel Propagator in x and y [Å]. Should match the propagated Waves.
        gpts : one or two int, optional
            Number of grid points in x and y describing the Fresnel Propagator. Should match the propagated Waves.
        sampling : one or two float
            Sampling of Fresnel Propagator in x and y [1 / Å]. Should match the propagated Waves.
        thickness : float, optional
            Propagation distance [Å]. Default is 0.5 Å.
        energy : float, optional
            Electron energy [eV].
        base_tilt : two float, optional
            Small angle beam tilt [mrad]. Implemented by shifting the wave function at every slice. Default is (0., 0.).
        device : str, optional
            The fresnel propagator data is stored on this device. The default is determined by the user configuration.
        """

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

        self._thickness = thickness
        self._device = device
        self._tilt = tilt
        self._tilt_axes = tilt_axes

        self._array = None

        def clear_data(*args):
            self._array = None

        self._events = Events()
        # self.grid.observe(clear_data, ('sampling', 'gpts', 'extent'))
        # self.accelerator.observe(clear_data, ('energy',))
        self.observe(clear_data, ('thickness',))

    def match_waves(self, waves):
        self.grid.match(waves)
        self.accelerator.match(waves)
        # self._tilt = waves.tilt
        self._tilt_axes = [axis for axis in waves.ensemble_axes_metadata if isinstance(axis, TiltAxis)]

    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    @watch
    def thickness(self, value: float):
        self._thickness = value

    @property
    def array(self):
        if self._array is None:
            self._array = self._calculate_array()

        return self._array

    def _calculate_array(self) -> np.ndarray:
        antialias_aperture = AntialiasAperture(device=self._device)
        antialias_aperture.grid.match(self)

        array = fresnel_propagator(gpts=self.gpts,
                                   sampling=self.sampling,
                                   thickness=self.thickness,
                                   energy=self.energy,
                                   device=self.device)

        # array = array * axis_tilt_shifts(thickness=self.thickness,
        #                                  tilt_axes=self._tilt_axes,
        #                                  gpts=self.gpts,
        #                                  sampling=self.sampling,
        #                                  device=self.device)

        array *= antialias_aperture.array
        return array

    def __call__(self, waves: 'Waves'):
        return self.propagate(waves)

    def propagate(self, waves: 'Waves', overwrite_x: bool = False) -> 'Waves':
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
        waves._array = fft2_convolve(waves.array, self.array, overwrite_x=overwrite_x)
        return waves


# class FresnelPropagator(HasGridMixin, HasAcceleratorMixin, HasEventsMixin, HasDeviceMixin):
#
#     def __init__(self,
#                  extent: Union[float, Tuple[float, float]] = None,
#                  gpts: Union[int, Tuple[int, int]] = None,
#                  sampling: Union[float, Tuple[float, float]] = None,
#                  thickness: float = .5,
#                  energy: float = None,
#                  tilt=(0., 0.),
#                  device: str = 'cpu'):

#         self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
#         self._accelerator = Accelerator(energy=energy)
#         self._device = device
#         self._thickness = thickness
#         self._tilt = tilt
#         self._array = None
#
#     @property
#     def array(self) -> np.ndarray:
#         if self._array is None:
#             self._array = self._calculate_array()
#
#         return self._array
#
#     @property
#     def tilt(self):
#         return self._tilt
#
#     @property
#     def thickness(self) -> float:
#         return self._thickness
#
#     @thickness.setter
#     def thickness(self, value: float):
#         if self._thickness != value:
#             self._array = None
#
#         self._thickness = value
#
#     def _calculate_array(self) -> np.ndarray:
#         antialias_aperture = AntialiasAperture(device=self._device)
#         antialias_aperture.grid.match(self)
#
#         array = fresnel_propagator(self.gpts,
#                                    self.sampling,
#                                    self.thickness,
#                                    self.energy,
#                                    tilt=self.tilt,
#                                    device=self.device
#                                    )
#
#         array *= antialias_aperture.array
#         return array
#
#     def propagate(self, waves: 'Waves', overwrite_x: bool = False) -> 'Waves':
#         """
#         Propagate wave functions.
#
#         Parameters
#         ----------
#         waves : Waves
#             The wave functions to propagate.
#         overwrite_x : bool
#             If True, the wave functions may be overwritten.
#
#         Returns
#         -------
#         propagated_wave_functions : Waves
#         """
#
#         waves._array = fft2_convolve(waves.array, self.array, overwrite_x=overwrite_x)
#         return waves


def allocate_multislice_measurements(waves: 'WavesLikeMixin',
                                     detectors: List[AbstractDetector],
                                     extra_ensemble_axes_shape=None,
                                     extra_ensemble_axes_metadata=None) \
        -> Dict[AbstractDetector, AbstractMeasurement]:
    measurements = {}
    for detector in detectors:
        xp = get_array_module(detector.measurement_meta(waves))

        measurement_type = detector.measurement_type(waves)

        axes_metadata = waves.ensemble_axes_metadata + detector.measurement_axes_metadata(waves)

        shape = waves.ensemble_shape + detector.measurement_shape(waves)

        if extra_ensemble_axes_shape is not None:
            assert len(extra_ensemble_axes_shape) == len(extra_ensemble_axes_metadata)
            shape = extra_ensemble_axes_shape + shape
            axes_metadata = extra_ensemble_axes_metadata + axes_metadata

        array = xp.zeros(shape, dtype=detector.measurement_dtype)
        measurements[detector] = measurement_type.from_array_and_metadata(array,
                                                                          axes_metadata=axes_metadata,
                                                                          metadata=waves.metadata)

    return measurements


def multislice_step(waves: 'Waves',
                    potential_slice: Union[PotentialArray, TransmissionFunction],
                    propagator: FresnelPropagator,
                    antialias_aperture: AntialiasAperture,
                    conjugate: bool = False,
                    transpose: bool = False) -> 'Waves':
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
        transmission_function = potential_slice.transmission_function(energy=waves.energy)
        transmission_function = antialias_aperture.bandlimit(transmission_function)

    if conjugate:
        propagator.thickness = -transmission_function.slice_thickness[0]
    else:
        propagator.thickness = transmission_function.slice_thickness[0]

    if transpose:
        waves = propagator.propagate(waves)
        waves = transmission_function.transmit(waves, conjugate=conjugate)
    else:
        waves = transmission_function.transmit(waves, conjugate=conjugate)
        waves = propagator.propagate(waves)

    return waves


def multislice_and_detect(waves: 'Waves',
                          potential: AbstractPotential,
                          detectors: List[AbstractDetector],
                          conjugate: bool = False,
                          transpose: bool = False,
                          ) \
        -> Union[Tuple[Union[AbstractMeasurement, 'Waves'], ...], AbstractMeasurement, 'Waves']:
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

    propagator = FresnelPropagator(device=get_array_module(waves.array), tilt=waves.tilt)
    propagator.match_waves(waves)

    if len(potential.exit_planes) > 1 or potential.ensemble_shape:
        extra_ensemble_axes_shape = potential.ensemble_shape
        extra_ensemble_axes_metadata = potential.ensemble_axes_metadata

        if len(potential.exit_planes) > 1:
            extra_ensemble_axes_shape += (len(potential.exit_planes),)
            extra_ensemble_axes_metadata += [potential.exit_planes_axes_metadata]

        measurements = allocate_multislice_measurements(waves,
                                                        detectors,
                                                        extra_ensemble_axes_shape=extra_ensemble_axes_shape,
                                                        extra_ensemble_axes_metadata=extra_ensemble_axes_metadata)

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
                waves = multislice_step(waves,
                                        potential_slice,
                                        propagator,
                                        antialias_aperture,
                                        conjugate=conjugate,
                                        transpose=transpose)
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
