from typing import Union, Tuple

import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.fft import fft2_convolve
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.grid import polar_spatial_frequencies
from abtem.core.utils import EqualityMixin, CopyMixin

TAPER = 0.01
CUTOFF = 2 / 3.0


def antialias_aperture(cutoff, taper, gpts, sampling, xp):
    cutoff = cutoff / max(sampling) / 2
    taper = taper / max(sampling)

    r, _ = polar_spatial_frequencies(gpts, sampling, xp=xp)

    if taper > 0.0:
        array = 0.5 * (1 + xp.cos(np.pi * (r - cutoff + taper) / taper))
        array[r > cutoff] = 0.0
        array = xp.where(r > cutoff - taper, array, xp.ones_like(r, dtype=np.float32))
    else:
        array = xp.array(r < cutoff).astype(np.float32)

    return array


class AntialiasAperture(HasGridMixin, CopyMixin, EqualityMixin):
    def __init__(
        self,
        cutoff: float = 2.0 / 3.0,
        taper: float = TAPER,
    ):
        self._cutoff = cutoff
        self._taper = taper
        self._key = None
        self._array = None

    def get_array(self, x):

        key = (
            self._cutoff,
            self._taper,
            x.gpts,
            x.sampling,
            x.energy,
            x.device,
        )

        if key == self._key:
            return self._array

        self._array = antialias_aperture(
            self._cutoff,
            self._taper,
            x.gpts,
            x.sampling,
            get_array_module(x.device),
        )
        self._key = key

        return self._array

    def bandlimit(self, x):
        array = self.get_array(x)

        x._array = fft2_convolve(x.array, array, overwrite_x=False)

        if hasattr(x, "_antialias_cutoff_gpts"):
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
