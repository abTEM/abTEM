from typing import TYPE_CHECKING, Union, Tuple

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import FrozenPhononsAxis
from abtem.core.backend import get_array_module, copy_to_device
from abtem.core.complex import complex_exponential
from abtem.core.dask import ComputableList
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.events import Events, watch, HasEventsMixin
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.measure.detect import stack_waves
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





# def multislice_and_detect(exit_waves_func, potentials, detectors, **kwargs):
#     measurements = [] #[[] for i in range(len(potentials))]
#
#     for i, potential in enumerate(potentials):
#         exit_waves = exit_waves_func(potential, **kwargs)
#
#         #print(detectors[0].detect(exit_waves))
#         #print(measurements[i])
#         measurements.append([detector.detect(exit_waves) for detector in detectors])
#         #print(measurements[i])
#         #for j, detector in enumerate(detectors):
#         #    #if exit_waves.is_lazy:
#
#
#
#
#             # if not exit_waves.is_lazy:
#             #     if detector.ensemble_mean and i == 0:
#             #         outputs.append(detector.detect(exit_waves))
#             #     elif detector.ensemble_mean:
#             #         outputs[j] += detector.detect(exit_waves)
#             #     elif i == 0:
#             #         outputs.append([detector.detect(exit_waves)])
#             #     else:
#             #         outputs[j].append(detector.detect(exit_waves))
#             #
#             # elif i == 0:
#                 #outputs.append([detector.detect(exit_waves)])
#             #else:
#             #    outputs[j].append(detector.detect(exit_waves))
#
#     measurements = list(map(list, zip(*measurements)))
#
#     return stack_measurement_ensembles(detectors, measurements)


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

    streaming = False
    for i, potential_slices in enumerate(potential.generate_slices(start=start, stop=stop)):

        # if i == 0:
        #     if get_array_module(potential_slices.array) != get_array_module(waves.array):
        #         streaming = True
        #
        # if streaming:
        #     potential_slices._array = copy_to_device(potential_slices._array, waves.array)

        if not isinstance(potential_slices, TransmissionFunction):
            transmission_functions = potential_slices.transmission_function(energy=waves.energy)
            transmission_functions = antialias_aperture.bandlimit(transmission_functions)

        else:
            transmission_functions = potential_slices

        for transmission_function in transmission_functions:
            if conjugate:
                propagator.thickness = -transmission_function.slice_thickness[0]
            else:
                propagator.thickness = transmission_function.slice_thickness[0]

            if start > stop:
                waves = propagator.propagate(waves)
                waves = transmission_function.transmit(waves, conjugate=conjugate)
            else:
                # propagator.thickness = transmission_function.slice_thickness[0]
                waves = transmission_function.transmit(waves, conjugate=conjugate)
                waves = propagator.propagate(waves)

    return waves
