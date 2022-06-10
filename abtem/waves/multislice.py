from typing import TYPE_CHECKING, Union, Tuple, List, Dict

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.events import Events, watch, HasEventsMixin
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.measure.detect import AbstractDetector
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import AbstractPotential, TransmissionFunction, PotentialArray
from abtem.waves.base import WavesLikeMixin
from abtem.waves.tilt import HasBeamTiltMixin, BeamTilt

if TYPE_CHECKING:
    from abtem.waves.waves import Waves


def fresnel_propagator(gpts, sampling, dz, energy, xp, tilt=(0., 0.)):
    wavelength = energy2wavelength(energy)

    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)

    f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * dz * wavelength) *
         complex_exponential(-(ky ** 2)[None] * np.pi * dz * wavelength))

    if tilt != (0., 0.):
        f *= (complex_exponential(-kx[:, None] * xp.tan(tilt[0] / 1e3) * dz * 2 * np.pi) *
              complex_exponential(-ky[None] * xp.tan(tilt[1] / 1e3) * dz * 2 * np.pi))

    return f


class FresnelPropagator(HasGridMixin, HasAcceleratorMixin, HasBeamTiltMixin, HasEventsMixin):

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 thickness: float = .5,
                 energy: float = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 device: str = 'cpu'):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._device = device
        self._thickness = thickness
        self._array = None

        def clear_data(*args):
            self._array = None

        self._events = Events()
        self.grid.observe(clear_data, ('sampling', 'gpts', 'extent'))
        self.accelerator.observe(clear_data, ('energy',))
        self.beam_tilt.observe(clear_data, ('tilt',))
        self.observe(clear_data, ('thickness',))

    def match_waves(self, waves):
        self.grid.match(waves)
        self.accelerator.match(waves)
        self.beam_tilt.match(waves)
        return self

    @property
    def array(self):
        if self._array is None:
            self._array = self._calculate_array()

        return self._array

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    @watch
    def thickness(self, value):
        self._thickness = value

    def _calculate_array(self):
        antialias_aperture = AntialiasAperture(device=self._device)
        antialias_aperture.grid.match(self)

        array = fresnel_propagator(self.gpts,
                                   self.sampling,
                                   self.thickness,
                                   self.energy,
                                   get_array_module(self._device),
                                   tilt=self.tilt)

        array *= antialias_aperture.array
        return array

    def propagate(self, waves: Union['Waves'], overwrite_x: bool = False, **kwargs):
        waves._array = fft2_convolve(waves.array, self.array, overwrite_x=overwrite_x, **kwargs)
        return waves


def multislice_step(waves: 'Waves',
                    potential_slice: Union[PotentialArray, TransmissionFunction],
                    propagator: FresnelPropagator,
                    antialias_aperture: AntialiasAperture,
                    conjugate: bool = False,
                    transpose: bool = False):
    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy(device=waves.device)

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

    waves.antialias_aperture = 2. / 3.

    return waves


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


def multislice(waves, potential, start=0, stop=None):
    if potential.num_frozen_phonons > 1:
        raise NotImplementedError

    antialias_aperture = AntialiasAperture(device=get_array_module(waves.array))
    antialias_aperture.match_grid(waves)

    propagator = FresnelPropagator(device=get_array_module(waves.array), tilt=waves.tilt)
    propagator.match_waves(waves)

    slice_generator = potential.generate_slices(first_slice=start, last_slice=stop)

    for i, potential_slice in enumerate(slice_generator):
        waves = multislice_step(waves, potential_slice, propagator, antialias_aperture)

    return waves


def multislice_and_detect(waves: 'Waves',
                          potential: AbstractPotential,
                          detectors: List[AbstractDetector],
                          start: int = 0,
                          stop: int = None,
                          keep_ensemble_dims: bool = True) -> Tuple[AbstractMeasurement]:
    #     """
    #     Run the multislice algorithm given a batch of wave functions and a potential.
    #
    #     Parameters
    #     ----------
    #     waves : Waves
    #         A batch of wave functions as a Waves type object.
    #     potential : AbstractPotential
    #         A potential as an AbstractPotential type object.
    #     propagator : FresnelPropagator, optional
    #         A fresnel propapgator type matching the wave functions. The main reason for using this argument is to reuse
    #         a previously calculated propagator. If not provided a new propagator is created.
    #     antialias_aperture : AntialiasAperture, optional
    #         An antialias aperture type matching the wave functions. The main reason for using this argument is to reuse
    #         a previously calculated antialias aperture. If not provided a new antialias aperture is created.
    #     start : int
    #         First slice index for running the multislice algorithm. Default is first slice of the potential.
    #     stop : int
    #         Last slice for running the multislice algorithm. If smaller than start the multislice algorithm will run
    #         in the reverse direction. Default is last slice of the potential.
    #     conjugate : bool
    #         I True, run the multislice algorithm using the conjugate of the transmission function
    #
    #     Returns
    #     -------
    #     exit_waves : Waves
    #     """

    if potential.num_frozen_phonons > 1:
        raise NotImplementedError

    antialias_aperture = AntialiasAperture(device=get_array_module(waves.array))
    antialias_aperture.match_grid(waves)

    propagator = FresnelPropagator(device=get_array_module(waves.array), tilt=waves.tilt)
    propagator.match_waves(waves)

    slice_generator = potential.generate_slices(first_slice=start)

    if detectors is None and len(potential.exit_planes) == 1:
        for i, potential_slice in enumerate(slice_generator):
            waves = multislice_step(waves, potential_slice, propagator, antialias_aperture)

            if i == potential.exit_planes[-1]:
                break

        return waves

    measurements = allocate_multislice_measurements(waves,
                                                    detectors,
                                                    extra_ensemble_axes_shape=potential.ensemble_shape,
                                                    extra_ensemble_axes_metadata=potential.ensemble_axes_metadata)

    current_slice_index = start
    for detect_index, exit_slice in enumerate(potential.exit_planes):

        while exit_slice != current_slice_index:
            potential_slice = next(slice_generator)
            waves = multislice_step(waves, potential_slice, propagator, antialias_aperture)
            current_slice_index += 1

        for detector in detectors:
            indices = (0,) * (len(potential.ensemble_shape) - 1) + (detect_index,)

            new_measurement = detector.detect(waves)
            measurements[detector].array[indices] = new_measurement.array

    measurements = tuple(measurements.values())

    if not keep_ensemble_dims:
        measurements = tuple(measurement[0, 0] for measurement in measurements)

    return measurements
