"""Module to describe electron waves and their propagation."""
from typing import TYPE_CHECKING
from typing import Union

import dask
import numpy as np

from abtem.basic.antialias import antialias_kernel
from abtem.basic.backend import get_array_module, xp_to_str, copy_to_device
from abtem.basic.complex import complex_exponential
from abtem.basic.energy import energy2wavelength
from abtem.basic.fft import fft2_convolve
from abtem.basic.grid import spatial_frequencies
from abtem.basic.utils import generate_chunks
from abtem.potentials.potentials import AbstractPotential

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


def _run_multislice_precalculated(waves_array,
                                  transmission_functions,
                                  antialias_kernel_array,
                                  initial_fresnel_propagator):
    xp = get_array_module(waves_array)

    for transmission_function in transmission_functions:
        transmission_function = fft2_convolve(transmission_function, antialias_kernel_array, overwrite_x=True)
        waves_array *= copy_to_device(transmission_function, xp)
        waves_array = fft2_convolve(waves_array, initial_fresnel_propagator, overwrite_x=False)

    return waves_array


def _run_multislice_generated(waves_array,
                              energy,
                              potential,
                              antialias_kernel_array,
                              initial_fresnel_propagator):
    xp = get_array_module(waves_array)

    for potential_slice in potential.generate():
        transmission_function = potential_slice.transmission_function(energy=energy).array
        transmission_function = fft2_convolve(transmission_function, antialias_kernel_array, overwrite_x=False)

        for transmission_function_slice in transmission_function:
            waves_array *= copy_to_device(transmission_function_slice, xp)
            waves_array = fft2_convolve(waves_array, initial_fresnel_propagator, overwrite_x=False)

    return waves_array


def multislice(waves: Union['Waves', 'SMatrixArray'],
               potential: AbstractPotential,
               chunks=1,
               ) -> Union['Waves', 'SMatrixArray']:
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()

    if hasattr(waves, 'array'):
        waves_array = waves.array
    else:
        waves = waves.build()
        waves_array = waves.array

    xp = get_array_module(waves_array)

    antialias_kernel_array = antialias_kernel(waves.gpts, waves.sampling, xp)

    initial_fresnel_propagator = dask.delayed(fresnel_propagator, pure=True)(waves.gpts,
                                                                             waves.sampling,
                                                                             potential.slice_thickness[0],
                                                                             waves.energy,
                                                                             xp_to_str(xp))

    initial_fresnel_propagator *= antialias_kernel_array

    if potential.precalculate:
        if hasattr(potential, 'array'):
            potential_array = potential
        else:
            potential_array = potential.build()

        for start, end in generate_chunks(len(potential_array), chunks=chunks):
            potential_slics = potential_array[start:end].transmission_function(energy=waves.energy, antialias=True)

            waves_array = waves_array.map_blocks(_run_multislice_precalculated,
                                                 transmission_functions=potential_slics.array,
                                                 antialias_kernel_array=antialias_kernel_array,
                                                 initial_fresnel_propagator=initial_fresnel_propagator,
                                                 dtype=np.complex64)

    else:
        waves_array = waves_array.map_blocks(_run_multislice_generated,
                                             energy=waves.energy,
                                             potential=potential,
                                             antialias_kernel_array=antialias_kernel_array,
                                             initial_fresnel_propagator=initial_fresnel_propagator,
                                             dtype=np.complex64)

    waves._array = waves_array

    waves.antialias_aperture = 2 / 3.
    return waves
