from typing import Union, Tuple

import numpy as np

from abtem.core.backend import get_array_module, xp_to_str
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.grid import polar_spatial_frequencies
from abtem.core.fft import fft2_convolve

TAPER = 0.01
CUTOFF = 2 / 3.


def antialias_aperture(cutoff, taper, gpts, sampling, xp):
    cutoff = cutoff / max(sampling) / 2
    taper = taper / max(sampling)

    r, _ = polar_spatial_frequencies(gpts, sampling, xp=xp)

    if taper > 0.:
        array = .5 * (1 + xp.cos(np.pi * (r - cutoff + taper) / taper))
        array[r > cutoff] = 0.
        array = xp.where(r > cutoff - taper, array, xp.ones_like(r, dtype=np.float32))
    else:
        array = xp.array(r < cutoff).astype(np.float32)

    return array


class AntialiasAperture(HasGridMixin):

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 cutoff: float = CUTOFF, taper: float = TAPER,
                 device: str = 'cpu'):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._cutoff = cutoff
        self._taper = taper
        self._device = device
        self._array = None

    @property
    def array(self):
        return self._array

    def build(self):
        return antialias_aperture(self.cutoff,
                                  self.taper,
                                  self.gpts,
                                  self.sampling,
                                  get_array_module(self._device))

    def apply(self, x):
        if self._array is None:
            self._array = self.build()

        x._array = fft2_convolve(x.array, self._array, overwrite_x=False)

        return x

    @property
    def taper(self) -> float:
        return self._taper

    @property
    def cutoff(self) -> float:
        """Anti-aliasing aperture as a fraction of the Nyquist frequency."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float):
        self._cutoff = value


class HasAntialiasApertureMixin:
    _antialias_aperture: AntialiasAperture

    def _valid_rectangle(self, gpts, sampling):
        if self.antialias_aperture is None:
            return gpts
        extent = (gpts[0] * sampling[0], gpts[1] * sampling[1])
        kcut = ((self.antialias_aperture - 2 * self.antialias_taper) / (max(sampling) * 2)) / np.sqrt(2)
        return (int(np.floor(kcut * extent[0] * 2)), int(np.floor(kcut * extent[1] * 2)))

    def _cutoff_rectangle(self, gpts, sampling):
        if self.antialias_aperture is None:
            return gpts
        kcut = 1 / max(sampling) / 2 * self.antialias_aperture
        extent = (gpts[0] * sampling[0], gpts[1] * sampling[1])
        return (int(np.floor(kcut * extent[0] * 2)), int(np.floor(kcut * extent[1] * 2)))

    @property
    def antialias_taper(self):
        return self._antialias_aperture._taper

    @property
    def antialias_aperture(self) -> float:
        return self._antialias_aperture.cutoff

    @antialias_aperture.setter
    def antialias_aperture(self, value: float):
        self._antialias_aperture.cutoff = value
