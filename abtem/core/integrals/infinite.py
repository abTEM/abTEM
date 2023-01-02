from typing import Tuple

import numpy as np

from abtem.core.antialias import antialias_aperture
from abtem.core.backend import get_array_module, device_name_from_array_module
from abtem.core.fft import fft2, ifft2, fft2_convolve
from abtem.core.grid import spatial_frequencies, polar_spatial_frequencies
from abtem.core.integrals.base import ProjectionIntegrator, ProjectionIntegratorPlan
from abtem.core.parametrizations import validate_parametrization

from abtem.core.backend import cp

if cp is not None:
    import cupyx


def sinc(gpts: Tuple[int, int], sampling: Tuple[float, float], xp):
    xp = get_array_module(xp)
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp)
    k = xp.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2)
    dk2 = sampling[0] * sampling[1]
    k[0, 0] = 1
    sinc = xp.sin(k) / k * dk2
    sinc[0, 0] = dk2
    return sinc


def superpose_deltas(positions: np.ndarray, array: np.ndarray, slice_index=None, weights=None) -> np.ndarray:
    xp = get_array_module(array)
    shape = array.shape

    positions = xp.array(positions)

    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    x = positions[:, 0] - rows
    y = positions[:, 1] - cols
    xy = x * y

    if slice_index is None:
        i = xp.array([rows % shape[0], (rows + 1) % shape[0]] * 2)
        j = xp.array([cols % shape[1]] * 2 + [(cols + 1) % shape[1]] * 2)
        v = xp.array([1 + xy - y - x, x - xy, y - xy, xy])

        if weights is not None:
            v = v * weights[None]

        if device_name_from_array_module(xp) == 'cpu':
            xp.add.at(array, (i, j), v)
        else:
            cupyx.scatter_add(array, (i, j), v)
    else:
        raise NotImplementedError

    return array


class ProjectedScatteringFactors(ProjectionIntegrator):

    def __init__(self, scattering_factor):
        self._scattering_factor = scattering_factor

    @property
    def gpts(self):
        return self._scattering_factor.shape[-2:]

    @property
    def scattering_factor(self):
        return self._scattering_factor

    def integrate_on_grid(self,
                          positions: np.ndarray,
                          a: np.ndarray,
                          b: np.ndarray,
                          gpts: Tuple[int, int],
                          sampling: Tuple[float, float],
                          device: str = 'cpu',
                          fourier_space: bool = False, ):
        xp = get_array_module(device)

        if len(positions) == 0:
            return xp.zeros(gpts, dtype=xp.float32)

        positions = (positions[:, :2] / sampling).astype(xp.float32)

        array = xp.zeros(gpts, dtype=xp.float32)

        array = superpose_deltas(positions, array).astype(xp.complex64)

        array = fft2(array, overwrite_x=True)

        array *= self._scattering_factor / sinc(self.gpts, sampling, device)

        if fourier_space:
            return array.real
        else:
            array = ifft2(array, overwrite_x=True).real
            return array


class InfinitePotentialProjections(ProjectionIntegratorPlan):

    def __init__(self, parametrization='lobato'):
        self._parametrization = validate_parametrization(parametrization)
        super().__init__(periodic=True, finite=False)

    def cutoff(self, symbol: str):
        return 0.

    def calculate_scattering_factor(self, symbol, gpts, sampling, device):
        xp = get_array_module(device)
        kx, ky = spatial_frequencies(gpts, sampling, xp=np)
        k2 = kx[:, None] ** 2 + ky[None] ** 2
        f = self._parametrization.projected_scattering_factor(symbol)(k2)
        f = xp.asarray(f, dtype=xp.float32)
        return ProjectedScatteringFactors(f)

    def build(self, symbol: str, gpts: Tuple[int, int], sampling: Tuple[float, float], device: str = 'cpu'):
        scattering_factor = self.calculate_scattering_factor(symbol, gpts, sampling, device)
        return scattering_factor
