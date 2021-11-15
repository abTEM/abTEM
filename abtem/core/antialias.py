import dask
import dask.array as da
import numpy as np

from abtem.core.backend import get_array_module, xp_to_str
from abtem.core.fft import fft2_convolve
from abtem.core.grid import polar_spatial_frequencies

TAPER = 0.01
CUTOFF = 2 / 3.


def antialias_kernel(gpts, sampling, xp, delay=True):
    def _antialias_kernel(gpts, sampling, cutoff, taper, xp):
        xp = get_array_module(xp)

        r, _ = polar_spatial_frequencies(gpts, sampling, delayed=False, xp=xp)
        if taper > 0.:
            array = .5 * (1 + xp.cos(np.pi * (r - cutoff + taper) / taper))
            array[r > cutoff] = 0.
            array = xp.where(r > cutoff - taper, array, xp.ones_like(r, dtype=np.float32))
        else:
            array = xp.array(r < cutoff).astype(np.float32)
        return array

    cutoff = CUTOFF / max(sampling) / 2
    taper = TAPER / max(sampling)

    if delay:
        kernel = dask.delayed(_antialias_kernel)(gpts, sampling, cutoff, taper, xp_to_str(xp))
        return da.from_delayed(kernel, shape=gpts, meta=xp.array((), dtype=xp.float32))
    else:
        return _antialias_kernel(gpts, sampling, cutoff, taper, xp_to_str(xp))


class AntialiasFilter:
    """
    Antialias filter object.
    """

    _taper = 0.01
    _cutoff = 2 / 3.

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def taper(self):
        return self._taper

    def _build_antialias_filter(self, gpts, sampling, xp):
        return

    def build(self, gpts, sampling, xp=np):
        return self._build_antialias_filter(gpts, sampling, xp)

    def __call__(self, array, sampling, overwrite_x=True):
        xp = get_array_module(array)
        kernel = self.build(array.shape[-2:], sampling, xp)
        array = fft2_convolve(array, kernel, overwrite_x=False)
        return array


class AntialiasAperture:

    def __init__(self, cutoff=2 / 3.):
        self._cutoff = cutoff

    @property
    def cutoff(self) -> float:
        """Anti-aliasing aperture as a fraction of the Nyquist frequency."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float):
        self._cutoff = value


class HasAntialiasApertureMixin:
    _antialias_aperture: AntialiasAperture
    _antialias_taper = AntialiasFilter._taper

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
        return self._antialias_taper

    @property
    def antialias_aperture(self) -> float:
        return self._antialias_aperture.cutoff

    @antialias_aperture.setter
    def antialias_aperture(self, value: float):
        self._antialias_aperture.cutoff = value
