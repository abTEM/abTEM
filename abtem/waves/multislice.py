"""Module to describe electron waves and their propagation."""
from typing import TYPE_CHECKING, Union, Tuple

import numpy as np

from abtem.core.antialias import AntialiasAperture
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength, HasAcceleratorMixin, Accelerator
from abtem.core.fft import fft2_convolve
from abtem.core.grid import spatial_frequencies, HasGridMixin, Grid
from abtem.potentials.potentials import AbstractPotential

from abtem.waves.tilt import HasBeamTiltMixin, BeamTilt

if TYPE_CHECKING:
    from abtem.waves.waves import Waves
    from abtem.waves.prism import SMatrixArray


def fresnel_propagator(gpts, sampling, dz, energy, xp):
    wavelength = energy2wavelength(energy)

    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)

    f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * dz * wavelength) *
         complex_exponential(-(ky ** 2)[None] * np.pi * dz * wavelength))

    return f


class FresnelPropagator(HasGridMixin, HasAcceleratorMixin, HasBeamTiltMixin):

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 antialias_aperture: float = 2 / 3.,
                 device: str = 'cpu'):

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(cutoff=antialias_aperture, device=device)
        self._antialias_aperture.grid.match(self)

        self._device = device
        self._array = None
        self._dz = None

    @property
    def array(self) -> np.ndarray:
        return self._array

    def build(self, dz: float):
        self._antialias_aperture.grid.match(self)

        if dz == self._dz:
            return self._array

        self._array = fresnel_propagator(self.gpts,
                                         self.sampling,
                                         dz,
                                         self.energy,
                                         get_array_module(self._device))

        self._array *= self._antialias_aperture.build()
        return self._array

    def propagate(self, waves: Union[np.ndarray, 'Waves'], dz: float):
        if self.array is None:
            self.build(dz)

        if hasattr(waves, 'array'):
            waves._array = fft2_convolve(waves.array, self.array, overwrite_x=False)
        else:
            waves = fft2_convolve(waves, self.array, overwrite_x=False)

        return waves


def _multislice(waves: np.ndarray,
                sampling: Tuple[float, float],
                energy: float,
                potential: AbstractPotential,
                transpose: bool = False) -> np.ndarray:
    xp = get_array_module(waves)

    propagator = FresnelPropagator(gpts=waves.shape[-2:], sampling=sampling, energy=energy, device=xp)
    antialias_aperture = AntialiasAperture(gpts=waves.shape[-2:], sampling=sampling)

    for potential_slices in potential.generate_chunks():
        transmission_functions = potential_slices.transmission_function(energy=energy)
        transmission_functions = antialias_aperture.apply(transmission_functions)

        for transmission_function in transmission_functions:
            if transpose:
                waves = propagator.propagate(waves, transmission_function.slice_thickness[0])
                waves *= transmission_function.array[0]
            else:
                waves *= transmission_function.array[0]
                waves = propagator.propagate(waves, transmission_function.slice_thickness[0])

    return waves


def multislice(waves: Union['Waves', 'SMatrixArray'], potential: AbstractPotential) -> Union['Waves', 'SMatrixArray']:
    waves.antialias_aperture = 2 / 3.
    if waves.is_lazy:
        xp = get_array_module(waves.array)
        waves._array = waves.array.map_blocks(_multislice,
                                              sampling=waves.sampling,
                                              energy=waves.energy,
                                              potential=potential,
                                              meta=xp.array((), dtype=xp.complex64))
    else:

        waves._array = _multislice(waves.array,
                                   sampling=waves.sampling,
                                   energy=waves.energy,
                                   potential=potential)

    return waves
