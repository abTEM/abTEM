from typing import Union, Tuple

import numpy as np

from abtem.core.backend import get_array_module, xp_to_str
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.grid import polar_spatial_frequencies
from abtem.core.fft import fft2_convolve
from abtem.core.events import HasEventsMixin, Events

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
                 cutoff: float = 2. / 3.,
                 taper: float = TAPER,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 device: str = 'cpu'):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._cutoff = cutoff
        self._taper = taper
        self._device = device
        self._array = None

        def clear_data(*args):
            self._array = None

        self.grid.events.observe(clear_data, ('sampling', 'gpts', 'extent'))

    @property
    def array(self):
        if self._array is None:
            self._array = self._calculate_array()

        return self._array

    def _calculate_array(self):
        self.grid.check_is_defined()
        array = antialias_aperture(self.cutoff,
                                   self.taper,
                                   self.gpts,
                                   self.sampling,
                                   get_array_module(self._device))
        return array

    def bandlimit(self, x):
        x._array = fft2_convolve(x.array, self.array, overwrite_x=False)

        if hasattr(x, '_antialias_cutoff_gpts'):
            x._antialias_cutoff_gpts = x.antialias_cutoff_gpts
        return x

    @property
    def taper(self) -> float:
        return self._taper

    @property
    def cutoff(self) -> float:
        """Anti-aliasing aperture as a fraction of the Nyquist frequency."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value




class HasAntialiasApertureMixin:
    _antialias_cutoff: float
    _antialias_taper = 0.01

    def _calculate_cutoff_angles(self, gpts, sampling):
        kcut = self.antialias_aperture / max(sampling)

        extent = gpts[0] * sampling[0], gpts[1] * sampling[1]
        return safe_floor_int(kcut * extent[0]), safe_floor_int(kcut * extent[1])

    def _valid_rectangle(self, gpts, sampling):
        if self.antialias_aperture is None:
            return gpts
        extent = (gpts[0] * sampling[0], gpts[1] * sampling[1])

        kcut = self.antialias_aperture / (max(sampling) * np.sqrt(2))

        return safe_floor_int(kcut * extent[0]), safe_floor_int(kcut * extent[1])

    def _cutoff_rectangle(self, gpts, sampling):
        if self.antialias_aperture is None:
            return gpts
        kcut = self.antialias_aperture / max(sampling)
        extent = gpts[0] * sampling[0], gpts[1] * sampling[1]
        return safe_floor_int(kcut * extent[0]), safe_floor_int(kcut * extent[1])

    @property
    def antialias_taper(self):
        return self._antialias_taper

    @property
    def antialias_aperture(self) -> float:
        return self._antialias_cutoff



    #@antialias_aperture.setter
    #def antialias_aperture(self, value):
    #    self._antialias_aperture.cutoff = value
