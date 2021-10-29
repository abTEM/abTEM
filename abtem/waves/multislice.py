"""Module to describe electron waves and their propagation."""
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

from abtem.basic.antialias import antialias_kernel
from abtem.basic.backend import get_array_module, xp_to_str
from abtem.basic.complex import complex_exponential
from abtem.basic.energy import energy2wavelength, energy2sigma
from abtem.basic.fft import fft2_convolve
from abtem.basic.grid import spatial_frequencies
from abtem.potentials.potentials import PotentialArray

if TYPE_CHECKING:
    from abtem.waves.waves import Waves
    from abtem.waves.prism import SMatrixArray


def fresnel_propagator(gpts, sampling, dz, energy, xp):
    xp = get_array_module(xp)

    wavelength = energy2wavelength(energy)

    kx, ky = spatial_frequencies(gpts, sampling, delayed=False, xp=xp)

    f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * dz * wavelength) *
         complex_exponential(-(ky ** 2)[None] * np.pi * dz * wavelength))

    return f


def _multislice(waves_array,
                gpts,
                sampling,
                energy,
                potential):
    xp = get_array_module(waves_array)

    slice_thickness = potential.slice_thickness

    antialias_kernel_array = antialias_kernel(gpts, sampling, xp, delay=False)

    initial_fresnel_propagator = fresnel_propagator(gpts,
                                                    sampling,
                                                    slice_thickness[0],
                                                    energy,
                                                    xp_to_str(xp))

    initial_fresnel_propagator *= antialias_kernel_array

    def _transmission_function(array, energy):
        array = complex_exponential(xp.float32(energy2sigma(energy)) * array)
        return array

    chunks = potential._chunks

    for start, end in chunks:
        potential_slice = potential._get_chunk(start, end)

        transmission_function = _transmission_function(potential_slice, energy=energy)
        # transmission_function = fft2_convolve(transmission_function, antialias_kernel_array, overwrite_x=False)

        if len(transmission_function.shape) == 2:
            transmission_function = transmission_function[None]

        for transmission_function_slice in transmission_function:
            waves_array = waves_array * transmission_function_slice
            #     #print(waves_array.dtype, transmission_function_slice.dtype)
            #     #waves_array *= copy_to_device(transmission_function_slice, xp)
            waves_array = fft2_convolve(waves_array, initial_fresnel_propagator, overwrite_x=False)

    return waves_array


def multislice(waves: Union['Waves', 'SMatrixArray'],
               potential: Union[PotentialArray]
               ) -> Union['Waves', 'SMatrixArray']:
    if waves.is_lazy:
        xp = get_array_module(waves.array)
        waves._array = waves.array.map_blocks(_multislice,
                                              gpts=waves.gpts,
                                              sampling=waves.sampling,
                                              energy=waves.energy,
                                              potential=potential,
                                              meta=xp.array((), dtype=xp.complex64))
    else:
        waves._array = _multislice(waves._array,
                                   gpts=waves.gpts,
                                   sampling=waves.sampling,
                                   energy=waves.energy,
                                   potential=potential)

    waves.antialias_aperture = 2 / 3.
    return waves
