from __future__ import annotations

import numpy as np

from abtem.core.backend import cp
from abtem.core.backend import get_array_module, device_name_from_array_module
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import spatial_frequencies
from abtem.core.integrals.base import ProjectionIntegrator, ProjectionIntegratorPlan
from abtem.core.parametrizations import validate_parametrization
from abtem.core.parametrizations.base import Parametrization

if cp is not None:
    import cupyx


def sinc(gpts: tuple[int, int], sampling: tuple[float, float], xp):
    xp = get_array_module(xp)
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp)
    k = xp.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2)
    dk2 = sampling[0] * sampling[1]
    k[0, 0] = 1
    sinc = xp.sin(k) / k * dk2
    sinc[0, 0] = dk2
    return sinc


def superpose_deltas(
    positions: np.ndarray,
    array: np.ndarray,
    weights: np.ndarray = None,
    round_positions: bool = True,
) -> np.ndarray:
    xp = get_array_module(array)
    shape = array.shape

    positions = xp.array(positions)

    if round_positions:
        rounded = xp.round(positions).astype(xp.int32)
        i, j = rounded[:, 0][None] % shape[0], rounded[:, 1][None] % shape[1]
        v = xp.array([1.0], dtype=xp.float32)[:, None]
    else:
        rounded = xp.floor(positions).astype(xp.int32)
        rows, cols = rounded[:, 0], rounded[:, 1]
        x = positions[:, 0] - rows
        y = positions[:, 1] - cols
        xy = x * y
        i = xp.array([rows % shape[0], (rows + 1) % shape[0]] * 2)
        j = xp.array([cols % shape[1]] * 2 + [(cols + 1) % shape[1]] * 2)
        v = xp.array([1 + xy - y - x, x - xy, y - xy, xy], dtype=xp.float32)

    if weights is not None:
        v = v * weights[None]

    if device_name_from_array_module(xp) == "cpu":
        xp.add.at(array, (i, j), v)
    else:
        cupyx.scatter_add(array, (i, j), v)
    # else:
    #    raise NotImplementedError

    return array


class ProjectedScatteringFactors(ProjectionIntegrator):
    def __init__(self, scattering_factor: np.ndarray):
        self._scattering_factor = scattering_factor

    @property
    def gpts(self) -> tuple[int, int]:
        return self._scattering_factor.shape[-2:]

    @property
    def scattering_factor(self) -> np.ndarray:
        return self._scattering_factor

    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        fourier_space: bool = False,
    ):
        xp = get_array_module(device)

        if len(positions) == 0:
            return xp.zeros(gpts, dtype=xp.float32)

        positions = (positions[:, :2] / sampling).astype(xp.float32)

        array = xp.zeros(gpts, dtype=xp.float32)

        array = superpose_deltas(positions, array).astype(xp.complex64)

        array = fft2(array, overwrite_x=True)

        f = self.scattering_factor / sinc(self.gpts, sampling, device)

        array *= f

        if fourier_space:
            return array.real
        else:
            array = ifft2(array, overwrite_x=True).real
            return array


class InfinitePotentialProjections(ProjectionIntegratorPlan):
    def __init__(self, parametrization: str | Parametrization = "lobato"):
        self._parametrization = validate_parametrization(parametrization)
        super().__init__(periodic=True, finite=False)

    @property
    def parametrization(self) -> Parametrization:
        return self._parametrization

    def cutoff(self, symbol: str) -> float:
        return np.inf

    def build(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> ProjectedScatteringFactors:

        xp = get_array_module(device)
        kx, ky = spatial_frequencies(gpts, sampling, xp=np)

        k2 = kx[:, None] ** 2 + ky[None] ** 2
        f = self.parametrization.projected_scattering_factor(symbol)(k2)
        f = xp.asarray(f, dtype=xp.float32)

        if symbol in self.parametrization.sigmas.keys():
            sigma = self.parametrization.sigmas[symbol]
            f = f * xp.exp(-xp.asarray(k2, dtype=xp.float32) * (xp.pi * sigma / xp.sqrt(3 / 2)) ** 2)
            print(f.dtype)

        return ProjectedScatteringFactors(f)
