import dask.array as da
import numpy as np

from abtem.utils.backend import get_array_module
from abtem.utils.coordinates import spatial_frequencies
from abtem.utils.fft import fft2_convolve


class AntialiasFilter:
    """
    Antialias filter object.
    """
    _cutoff = 2 / 3.
    _taper = .05

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def taper(self):
        return self._taper

    def _build_antialias_filter(self, gpts, sampling, xp):
        k = spatial_frequencies(gpts, sampling, return_radial=True)[2]

        kcut = 1 / max(sampling) / 2 * self.cutoff
        taper = self.taper / max(sampling)

        if self.taper > 0.:
            array = .5 * (1 + xp.cos(np.pi * (k - kcut + taper) / taper))
            array[k > kcut] = 0.
            array = xp.where(k > kcut - taper, array, xp.ones_like(k, dtype=np.float32))
        else:
            array = xp.array(k < kcut).astype(np.float32)

        return array

    def build(self, gpts, sampling, xp=np):
        return self._build_antialias_filter(gpts, sampling, xp)

    def __call__(self, array, sampling, overwrite_x=True):
        xp = get_array_module(array)

        array = fft2_convolve(array, self.build(array.shape[-2:], sampling, xp), overwrite_x=overwrite_x)
        return array
