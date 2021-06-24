"""Module to describe electron waves and their propagation."""
from typing import TYPE_CHECKING
from typing import Union

import dask
import dask.array as da
import numpy as np

from abtem.potentials import AbstractPotential, AbstractPotentialBuilder
from abtem.basic.antialias import AntialiasFilter
from abtem.basic.backend import get_array_module
from abtem.basic.complex import complex_exponential
from abtem.basic.grid import spatial_frequencies
from abtem.basic.fft import fft2_convolve

if TYPE_CHECKING:
    from abtem.waves.waves import Waves
    from abtem.waves.prism import SMatrixArray


def fresnel(gpts, sampling, dz):
    kx, ky = spatial_frequencies(gpts, sampling, delayed=False)

    f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * dz) *
         complex_exponential(-(ky ** 2)[None] * np.pi * dz))

    return f


class FresnelPropagator:
    """
    Fresnel propagator object.

    This class is used for propagating a wave function object using the near-field approximation (Fresnel diffraction).
    The array representing the Fresnel propagator function is cached.
    """

    def __init__(self, waves):
        self._gpts = waves.gpts
        self._sampling = waves.sampling
        self._wavelength = waves.wavelength
        self._tilt = waves.tilt
        self._xp = get_array_module(waves.array)

        self._array = None
        self._dz = None

    def build(self, dz: float) -> np.ndarray:
        if dz != self._dz:
            self._array = None

        if self._array is not None:
            return self._array

        f = dask.delayed(fresnel, pure=True)(self._gpts, self._sampling, self._wavelength * dz)
        f = da.from_delayed(f, shape=self._gpts, dtype=np.complex64)

        # if self._tilt is not None:
        #     f *= (complex_exponential(-kx[:, None] * self._xp.tan(self._tilt[0] / 1e3) * dz * 2 * np.pi) *
        #           complex_exponential(-ky[None] * self._xp.tan(self._tilt[1] / 1e3) * dz * 2 * np.pi))

        self._array = f * AntialiasFilter().build(self._gpts, self._sampling, self._xp)

        self._dz = dz
        return self._array

    def propagate(self, waves: Union['Waves', 'SMatrixArray'], dz: float) -> Union['Waves', 'SMatrixArray']:
        """
        Propagate wave functions or scattering matrix.

        Parameters
        ----------
        dz : float
            Propagation distance [Å].

        Returns
        -------
        Waves or SMatrixArray object
            The propagated wave functions.
        """
        waves._array = fft2_convolve(waves.array, self.build(dz))
        waves.antialias_aperture = (2 / 3.,) * 2
        return waves


def chunk_sequence(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# dask.delayed
def _multislice(waves, transmission_function, slice_thicknesses, propagator):
    for t, d in zip(transmission_function, slice_thicknesses):
        waves *= t
        waves = fft2_convolve(waves, propagator, overwrite_x=False)
    return waves


def multislice(waves: Union['Waves', 'SMatrixArray'],
               potential: Union[AbstractPotential, AbstractPotentialBuilder],
               splits=1,
               ) -> Union['Waves', 'SMatrixArray']:
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()

    try:
        potential_array = potential.build()
    except:
        potential_array = potential

    try:
        waves_array = waves.array
    except AttributeError:
        waves = waves.build()
        waves_array = waves.array

    propagator = FresnelPropagator(waves)
    propagator = propagator.build(.5)

    chunks = potential_array.array.chunks[0]

    chunks = tuple(sum(chunk) for chunk in chunk_sequence(chunks, min(len(chunks), splits)))

    slics = [slice(start, start + length) for start, length in zip(np.cumsum((0,) + chunks), chunks)]

    for slic in slics:
        potential_chunks = potential_array[slic]

        try:
            transmission_function = potential_chunks.transmission_function(energy=waves.energy)
        except AttributeError:
            transmission_function = potential_chunks.build().transmission_function(energy=waves.energy)

        waves_array = waves_array.map_blocks(_multislice,
                                             transmission_function=transmission_function.array,
                                             slice_thicknesses=transmission_function.slice_thicknesses,
                                             propagator=propagator,
                                             dtype=np.complex64)

    waves._array = waves_array
    waves.antialias_aperture = 2 / 3.
    return waves
