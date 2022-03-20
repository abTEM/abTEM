import itertools
from typing import TYPE_CHECKING, Union, Tuple, List

import dask
import dask.array as da
import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import FrozenPhononsAxis, OrdinalAxis
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.dask import ComputableList
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.events import Events, watch, HasEventsMixin
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.measure.detect import AbstractDetector, validate_detectors
from abtem.measure.measure import AbstractMeasurement
from abtem.measure.thickness import thickness_series_precursor, detectors_at_stop_slice
from abtem.potentials.potentials import AbstractPotential, TransmissionFunction
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


def multislice_and_detect_with_frozen_phonons(waves: 'Waves',
                                              potential: AbstractPotential,
                                              detectors: List[AbstractDetector]) \
        -> Union[AbstractMeasurement, 'Waves', List[Union[AbstractMeasurement, 'Waves']]]:
    detectors = validate_detectors(detectors)

    measurements = []
    for frozen_phonon_potential in potential.get_frozen_phonon_potentials(lazy=True):
        cloned_waves = waves.clone()
        new_measurements = _lazy_multislice_and_detect(cloned_waves, frozen_phonon_potential, detectors)
        measurements.append(new_measurements)

    measurements = list(map(list, zip(*measurements)))

    for i, (detector, output) in enumerate(zip(detectors, measurements)):
        if not potential.ensemble_mean or isinstance(output, list):
            measurements[i] = stack_measurements(output, axes_metadata=potential.frozen_phonons.axes_metadata)

        measurements[i] = measurements[i].squeeze()

        if hasattr(measurements[i], '_reduce_ensemble'):
            measurements[i] = measurements[i]._reduce_ensemble()

    if len(measurements) == 1:
        return measurements[0]
    else:
        return ComputableList(measurements)


def concatenate_last_axis_recursively(arrays, axis=-1):
    new_shape = list(arrays.shape)
    del new_shape[-1]

    new_arrays = np.empty(new_shape, dtype=object)
    for item in np.ndindex(*new_shape):
        new_arrays[item] = da.concatenate(arrays[item], axis=axis)

    if len(new_shape) > 0:
        return concatenate_last_axis_recursively(new_arrays, axis=axis - 1)
    else:
        return new_arrays


def _lazy_multislice_and_detect(waves, potential, detectors):
    chunks = waves.array.chunks[:-2]
    delayed_waves = waves.to_delayed()

    def wrapped_multislice_detect(waves, potential, detectors):
        return [measurement.array for measurement in multislice_and_detect(waves, potential, detectors)]

    dwrapped = dask.delayed(wrapped_multislice_detect, nout=len(detectors))
    delayed_potential = potential.to_delayed()

    collections = np.empty_like(delayed_waves, dtype=object)
    for index, waves_block in np.ndenumerate(delayed_waves):
        collections[index] = dwrapped(waves_block, delayed_potential, detectors)

    _, _, thickness_axes_metadata = thickness_series_precursor(detectors, potential)

    measurements = []
    for i, (detector, axes_metadata) in enumerate(zip(detectors, thickness_axes_metadata)):
        arrays = np.empty_like(collections, dtype=object)

        thicknesses = detector.num_detections(potential)
        measurement_shape = detector.measurement_shape(waves)[waves.num_extra_axes:]

        for (index, collection), chunk in zip(np.ndenumerate(collections), itertools.product(*chunks)):
            shape = (thicknesses,) + chunk + measurement_shape
            arrays[index] = da.from_delayed(collection[i], shape=shape, meta=np.array((), dtype=np.float32))

        if len(arrays.shape) > 0:
            arrays = concatenate_last_axis_recursively(arrays, axis=-max(len(measurement_shape) + 1, 1))

        arrays = arrays.item()

        d = detector.measurement_kwargs(waves)
        d['extra_axes_metadata'] = [axes_metadata] + d['extra_axes_metadata']

        measurement = detector.measurement_type(waves)(arrays, **d)

        measurement = measurement.squeeze()
        measurements.append(measurement)

    return ComputableList(measurements)


def multislice_and_detect(waves, potential, detectors):
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

        new_measurements = [detector.detect(waves) for detector in detectors_at]

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
    """
    Run the multislice algorithm given a batch of wave functions and a potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a Waves type object.
    potential : AbstractPotential
        A potential as an AbstractPotential type object.
    propagator : FresnelPropagator, optional
        A fresnel propapgator type matching the wave functions. The main reason for using this argument is to reuse
        a previously calculated propagator. If not provided a new propagator is created.
    antialias_aperture : AntialiasAperture, optional
        An antialias aperture type matching the wave functions. The main reason for using this argument is to reuse
        a previously calculated antialias aperture. If not provided a new antialias aperture is created.
    start : int
        First slice index for running the multislice algorithm. Default is first slice of the potential.
    stop : int
        Last slice for running the multislice algorithm. If smaller than start the multislice algorithm will run
        in the reverse direction. Default is last slice of the potential.
    conjugate : bool
        I True, run the multislice algorithm using the conjugate of the transmission function

    Returns
    -------
    exit_waves : Waves
    """

    if stop is None:
        stop = len(potential)

    if potential.num_frozen_phonons > 1:
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
