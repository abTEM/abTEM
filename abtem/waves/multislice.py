"""Module to describe electron waves and their propagation."""
from typing import TYPE_CHECKING
from typing import Union, Tuple

import dask
import dask.array as da
import numpy as np

from abtem.potentials import AbstractPotential, AbstractPotentialBuilder
from abtem.utils.antialias import AntialiasFilter
from abtem.utils.backend import get_array_module
from abtem.utils.complex import complex_exponential
from abtem.utils.coordinates import spatial_frequencies
from abtem.utils.fft import fft2_convolve

if TYPE_CHECKING:
    from abtem.waves.waves import Waves
    from abtem.waves.prism import SMatrixArray


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

        kx, ky = spatial_frequencies(self._gpts, self._sampling)

        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * self._wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * self._wavelength * dz))

        if self._tilt is not None:
            f *= (complex_exponential(-kx[:, None] * self._xp.tan(self._tilt[0] / 1e3) * dz * 2 * np.pi) *
                  complex_exponential(-ky[None] * self._xp.tan(self._tilt[1] / 1e3) * dz * 2 * np.pi))

        self._array = (f * AntialiasFilter().build(self._gpts, self._sampling, self._xp)).compute()
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


@dask.delayed
def _multislice(waves, transmission_function, slice_thicknesses, propagator):
    for t, d in zip(transmission_function, slice_thicknesses):
        waves *= t
        waves = fft2_convolve(waves, propagator.build(d), overwrite_x=False)
    return waves


def multislice(waves: Union['Waves', 'SMatrixArray'],
               potential: Union[AbstractPotential, AbstractPotentialBuilder]
               ) -> Union['Waves', 'SMatrixArray']:
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()

    propagator = FresnelPropagator(waves)

    try:
        transmission_function = potential.transmission_function(energy=waves.energy)
    except AttributeError:
        transmission_function = potential.build().transmission_function(energy=waves.energy)

    try:
        waves_array = waves.array
    except AttributeError:
        waves_array = waves.build().array

    waves._array = da.from_delayed(
        _multislice(waves_array, transmission_function.array, transmission_function.slice_thicknesses, propagator),
        shape=waves.array.shape, dtype=np.complex64)

    return waves
