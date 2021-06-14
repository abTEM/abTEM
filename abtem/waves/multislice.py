"""Module to describe electron waves and their propagation."""
from typing import Union, Tuple

import numpy as np

from abtem.potentials import AbstractPotential, AbstractPotentialBuilder
from abtem.utils.antialias import AntialiasFilter
from abtem.utils.backend import get_array_module
from abtem.utils.complex import complex_exponential
from abtem.utils.fft import fft2_convolve
from abtem.utils.coordinates import spatial_frequencies

class FresnelPropagator:
    """
    Fresnel propagator object.

    This class is used for propagating a wave function object using the near-field approximation (Fresnel diffraction).
    The array representing the Fresnel propagator function is cached.
    """

    def _evaluate_propagator_array(self,
                                   gpts: Tuple[int, int],
                                   sampling: Tuple[float, float],
                                   wavelength: float,
                                   dz: float,
                                   tilt: Tuple[float, float],
                                   xp) -> np.ndarray:

        kx = xp.fft.fftfreq(gpts[0], sampling[0]).astype(np.float32)
        ky = xp.fft.fftfreq(gpts[1], sampling[1]).astype(np.float32)
        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * wavelength * dz))

        if tilt is not None:
            f *= (complex_exponential(-kx[:, None] * xp.tan(tilt[0] / 1e3) * dz * 2 * np.pi) *
                  complex_exponential(-ky[None] * xp.tan(tilt[1] / 1e3) * dz * 2 * np.pi))

        self.f = f

        return f * AntialiasFilter().get_mask(gpts, sampling, xp)

    def propagate(self,
                  waves: Union['Waves', 'SMatrixArray'],
                  dz: float) -> Union['Waves', 'SMatrixArray']:
        """
        Propagate wave functions or scattering matrix.

        Parameters
        ----------
        waves : Waves or SMatrixArray object
            Wave function or scattering matrix to propagate.
        dz : float
            Propagation distance [Å].
        in_place : bool, optional
            If True the wavefunction array will be modified in place. Default is True.

        Returns
        -------
        Waves or SMatrixArray object
            The propagated wave functions.
        """

        try:
            xp = get_array_module(waves.array)
        except:
            return self.propagate(waves.build(), dz=dz)

        kx,ky = spatial_frequencies(waves.gpts, waves.sampling)

        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * waves.wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * waves.wavelength * dz))

        # if tilt is not None:
        #     f *= (complex_exponential(-kx[:, None] * xp.tan(tilt[0] / 1e3) * dz * 2 * np.pi) *
        #           complex_exponential(-ky[None] * xp.tan(tilt[1] / 1e3) * dz * 2 * np.pi))

        propagator_array = f * AntialiasFilter().build(waves.gpts, waves.sampling, xp)

        waves._array = fft2_convolve(waves.array, propagator_array)
        waves.antialias_aperture = (2 / 3.,) * 2
        return waves


def _multislice(waves: Union['Waves', 'SMatrixArray'],
                potential: AbstractPotential,
                propagator: FresnelPropagator = None,
                ) -> Union['Waves', 'SMatrixArray']:
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()

    if propagator is None:
        propagator = FresnelPropagator()

    if isinstance(potential, AbstractPotentialBuilder):
        potential = potential.build()

    # transmission_function = potential.transmission_function(energy=300e3)

    for t in potential.transmission_function(energy=waves.energy):
        waves = t.transmit(waves)
        waves = propagator.propagate(waves, t.thickness)

    return waves
