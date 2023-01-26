import numpy as np

from abtem.core import config
from abtem.core.backend import get_array_module
from abtem.core.fft import fft2, ifft2, fft2_convolve
from abtem.core.grid import HasGridMixin, spatial_frequencies
from abtem.core.utils import EqualityMixin, CopyMixin


def antialias_aperture(gpts, sampling, xp):
    cutoff = config.get("antialias.cutoff") / max(sampling) / 2
    taper = config.get("antialias.taper") / max(sampling)

    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    r = xp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)

    if taper > 0.0:
        array = 0.5 * (1 + xp.cos(np.pi * (r - cutoff + taper) / taper))
        array[r > cutoff] = 0.0
        array = xp.where(r > cutoff - taper, array, 1.0)
    else:
        array = xp.array(r < cutoff)

    return array


class AntialiasAperture(HasGridMixin, CopyMixin, EqualityMixin):
    def __init__(
        self,
    ):
        self._key = None
        self._array = None

    def get_array(self, x):
        key = (
            x.gpts,
            x.sampling,
            x.energy,
            x.device,
        )

        if key == self._key:
            return self._array

        self._array = antialias_aperture(
            x.gpts,
            x.sampling,
            get_array_module(x.device),
        )
        self._key = key

        return self._array

    def bandlimit(self, x, overwrite_x: bool = False):
        kernel = self.get_array(x)
        kernel = kernel[(None,) * (len(x.shape) - 2)]

        x._array = fft2_convolve(x._array, kernel, overwrite_x=overwrite_x)
        return x
