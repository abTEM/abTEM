"""Module for describing antialiasing objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from abtem.core import config
from abtem.core.backend import get_array_module
from abtem.core.fft import fft2_convolve
from abtem.core.grid import HasGrid2DMixin, spatial_frequencies
from abtem.core.utils import CopyMixin, EqualityMixin

if TYPE_CHECKING:
    from abtem.potentials.iam import TransmissionFunction
    from abtem.waves import Waves


def antialias_aperture(
    gpts: tuple[int, int], sampling: tuple[float, float], xp=None
) -> np.ndarray:
    """
    Array defining a Fourier-space antialiasing aperture.

    Parameters
    ----------
    gpts : two int, optional
        Number of grid points in `x` and `y` describing the antialiasing aperture.
    sampling : two float, optional
        Reciprocal-space sampling in `x` and `y` of the antialiasing aperture. Units are arbitrary.

    Returns
    -------
    antialias_aperture_array : np.ndarray
    """
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


class AntialiasAperture(HasGrid2DMixin, CopyMixin, EqualityMixin):
    def __init__(
        self,
    ):
        self._key = None
        self._array = None

    def get_array(self, x: Waves | TransmissionFunction):
        key = (
            x.gpts,
            x.sampling,
            x.energy,
            x.device,
        )

        if key == self._key:
            return self._array

        self._array = antialias_aperture(
            x._valid_gpts,
            x._valid_sampling,
            get_array_module(x.device),
        )
        self._key = key

        return self._array

    def bandlimit(
        self, x: Waves | TransmissionFunction, in_place: bool = False
    ) -> Waves | TransmissionFunction:
        kernel = self.get_array(x)
        kernel = kernel[(None,) * (len(x.shape) - 2)]

        x._array = fft2_convolve(x._array, kernel, overwrite_x=in_place)
        return x
