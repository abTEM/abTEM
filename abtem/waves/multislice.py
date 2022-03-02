import itertools
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Union, Tuple, List
import dask
import dask.array as da

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import FrozenPhononsAxis, OrdinalAxis
from abtem.core.backend import get_array_module, copy_to_device
from abtem.core.complex import complex_exponential
from abtem.core.dask import ComputableList
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.events import Events, watch, HasEventsMixin
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.measure.detect import stack_waves, AbstractDetector, validate_detectors
from abtem.measure.thickness import thickness_series_precursor, detectors_at_stop_slice
from abtem.potentials.potentials import AbstractPotential, TransmissionFunction
from abtem.waves.fresnel import FresnelPropagator
from abtem.waves.tilt import HasBeamTiltMixin, BeamTilt

if TYPE_CHECKING:
    from abtem.waves.waves import Waves


def fresnel_propagator(gpts, sampling, dz, energy, xp):
    wavelength = energy2wavelength(energy)

    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)

    f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * dz * wavelength) *
         complex_exponential(-(ky ** 2)[None] * np.pi * dz * wavelength))

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
                                   get_array_module(self._device))

        array *= antialias_aperture.array
        return array

    def propagate(self, waves: Union['Waves']):
        waves._array = fft2_convolve(waves.array, self.array, overwrite_x=False)
        return waves


def stack_measurements(measurements, axes_metadata):
    array = np.stack([measurement.array for measurement in measurements])
    cls = measurements[0].__class__
    d = measurements[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + d['extra_axes_metadata']
    return cls(**d)


def multislice_and_detect_with_frozen_phonons(waves, potential, detectors, lazy, func, func_kwargs=None):
    detectors = validate_detectors(detectors)

    if func_kwargs is None:
        func_kwargs = {}

    measurements = []
    for potential in potential.get_distribution(lazy=lazy):

        new_measurements = func(waves, potential, detectors, **func_kwargs)
        measurements.append(new_measurements)

    measurements = list(map(list, zip(*measurements)))

    for i, (detector, output) in enumerate(zip(detectors, measurements)):
        if not detector.ensemble_mean or isinstance(output, list):
            measurements[i] = stack_measurements(output, axes_metadata=FrozenPhononsAxis())

            if detector.ensemble_mean:
                measurements[i] = measurements[i].mean(0)

        measurements[i] = measurements[i].squeeze()

    if len(measurements) == 1:
        return measurements[0]
    else:
        return ComputableList(measurements)


def multislice_and_detect(waves, potential, detectors, func, func_kwargs=None):
    if func is None:
        func_kwargs = {}

    antialias_aperture = AntialiasAperture(device=get_array_module(waves.array))
    antialias_aperture.match_grid(waves)

    propagator = FresnelPropagator(device=get_array_module(waves.array))
    propagator.match_waves(waves)

    measurements = {detector: None for detector in detectors}

    multislice_start_stop, detect_every, _ = thickness_series_precursor(detectors, potential)

    for i, (start, stop) in enumerate(multislice_start_stop):

        waves = multislice(waves,
                           potential=potential,
                           start=start,
                           stop=stop,
                           propagator=propagator,
                           antialias_aperture=antialias_aperture)

        waves.antialias_aperture = 2. / 3.

        detectors_at = detectors_at_stop_slice(detect_every, stop)

        new_measurements = func(waves, detectors_at, **func_kwargs)

        for detector, new_measurement in zip(detectors_at, new_measurements):

            if measurements[detector] is None:
                xp = get_array_module(new_measurement.array)

                extra_axes_shape = (detector.num_detections(potential),)
                array = xp.zeros(extra_axes_shape + new_measurement.shape, dtype=new_measurement.array.dtype)

                d = new_measurement._copy_as_dict(copy_array=False)
                d['array'] = array
                d['extra_axes_metadata'] = [OrdinalAxis()] + d['extra_axes_metadata']

                measurements[detector] = new_measurement.__class__(**d)

            j = -1 if stop == len(potential) else stop // detector.detect_every - 1

            measurements[detector].array[j] = new_measurement.array

    return measurements.values()


def multislice(waves: 'Waves',
               potential: AbstractPotential,
               propagator: FresnelPropagator = None,
               antialias_aperture: AntialiasAperture = None,
               start: int = 0,
               stop: int = None,
               conjugate: bool = False):
    if stop is None:
        stop = len(potential)

    if potential.num_configurations > 1:
        raise RuntimeError()

    if antialias_aperture is None:
        antialias_aperture = AntialiasAperture(device=get_array_module(waves.array))

    antialias_aperture.match_grid(waves)

    if propagator is None:
        propagator = FresnelPropagator(device=get_array_module(waves.array))

    propagator.match_waves(waves)

    for i, potential_slices in enumerate(potential.generate_slices(start=start, stop=stop)):

        if isinstance(potential_slices, TransmissionFunction):
            transmission_functions = potential_slices

        else:
            transmission_functions = potential_slices.transmission_function(energy=waves.energy)
            transmission_functions = antialias_aperture.bandlimit(transmission_functions)

        for transmission_function in transmission_functions:
            if conjugate:
                propagator.thickness = -transmission_function.slice_thickness[0]
            else:
                propagator.thickness = transmission_function.slice_thickness[0]

            if start > stop:
                waves = propagator.propagate(waves)
                waves = transmission_function.transmit(waves, conjugate=conjugate)
            else:
                waves = transmission_function.transmit(waves, conjugate=conjugate)
                waves = propagator.propagate(waves)

    return waves
