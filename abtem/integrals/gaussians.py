from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

from abtem.core.backend import get_array_module
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import polar_spatial_frequencies
from abtem.integrals.base import ProjectionIntegratorPlan, ProjectionIntegrator
from abtem.integrals.infinite import superpose_deltas, sinc
from abtem.integrals.quadrature import cutoff
from abtem.parametrizations import validate_parametrization


if TYPE_CHECKING:
    from abtem.parametrizations.base import Parametrization


class GaussianScatteringFactors(ProjectionIntegrator):
    def __init__(
        self,
        gaussian_scattering_factors,
        error_function_scales,
        correction_scattering_factors,
    ):
        self._gaussian_scattering_factors = gaussian_scattering_factors
        self._error_function_scales = error_function_scales
        self._correction_scattering_factors = correction_scattering_factors

    @property
    def gpts(self):
        return self._gaussian_scattering_factors.shape[-2:]

    def _integrate_gaussian_scattering_factors(self, positions, a, b, sampling, device):
        xp = get_array_module(device)

        a = a - positions[:, 2]
        b = b - positions[:, 2]

        positions = (positions[:, :2] / sampling).astype(xp.float32)

        weights = (
            np.abs(
                erf(self._error_function_scales[:, None] * b[None])
                - erf(self._error_function_scales[:, None] * a[None])
            )
            / 2
        )

        array = xp.zeros(self.gpts, dtype=xp.complex64)
        for i in range(5):
            temp = xp.zeros_like(array, dtype=xp.complex64)

            superpose_deltas(positions, temp, weights=weights[i])

            array += fft2(temp, overwrite_x=False) * self._gaussian_scattering_factors[
                i
            ].astype(xp.complex64)

        return array

    def _integrate_correction_factors(self, positions, a, b, sampling, device):
        xp = get_array_module(device)

        temp = xp.zeros(self.gpts, dtype=xp.complex64)

        positions = positions[(positions[:, 2] >= a) * (positions[:, 2] < b)]

        positions = (positions[:, :2] / sampling).astype(xp.float32)

        superpose_deltas(positions, temp)

        return fft2(
            temp, overwrite_x=False
        ) * self._correction_scattering_factors.astype(xp.complex64)

    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        fourier_space: bool = False,
    ) -> np.ndarray:

        assert gpts == self.gpts

        array = self._integrate_gaussian_scattering_factors(
            positions, a, b, sampling, device
        )

        if self._correction_scattering_factors is not None:
            array += self._integrate_correction_factors(
                positions, a, b, sampling, device
            )

        if fourier_space:
            return array
        else:
            return ifft2(array / sinc(self.gpts, sampling, device)).real



class GaussianProjectionIntegrals(ProjectionIntegratorPlan):
    def __init__(
        self,
        gaussian_parametrization: str | Parametrization = "peng",
        correction_parametrization: str | Parametrization = "lobato",
        cutoff_tolerance: float = 1e-3,
    ):

        self._gaussian_parametrization = validate_parametrization(
            gaussian_parametrization
        )

        if correction_parametrization is not None:
            self._correction_parametrization = validate_parametrization(
                correction_parametrization
            )
        else:
            self._correction_parametrization = correction_parametrization

        self._cutoff_tolerance = cutoff_tolerance

        super().__init__(periodic=True, finite=True)

    @property
    def cutoff_tolerance(self):
        return self._cutoff_tolerance

    @property
    def gaussian_parametrization(self):
        return self._gaussian_parametrization

    @property
    def correction_parametrization(self):
        return self._correction_parametrization

    def cutoff(self, symbol: str) -> float:
        return cutoff(
            self.gaussian_parametrization.potential(symbol),
            self.cutoff_tolerance,
            a=1e-3,
            b=1e3,
        )  # noqa

    def gaussian_scattering_factors(self, symbol, gpts, sampling, device="cpu"):
        xp = get_array_module(device)
        k, _ = polar_spatial_frequencies(gpts, sampling, xp=xp)

        parameters = xp.array(
            self.gaussian_parametrization.scaled_parameters(symbol)[
                "projected_scattering_factor"
            ]
        )

        gaussian_scattering_factors = parameters[0, :, None, None] * np.exp(
            -parameters[1, :, None, None] * k[None] ** 2.0
        )

        if self.correction_parametrization:
            infinite_gaussian = (
                self.gaussian_parametrization.projected_scattering_factor(symbol)
            )
            infinite_correction = (
                self.correction_parametrization.projected_scattering_factor(symbol)
            )
            correction_scattering_factors = infinite_correction(
                k**2
            ) - infinite_gaussian(k**2)
        else:
            correction_scattering_factors = None

        error_function_scales = np.pi / np.sqrt(parameters[1])

        return GaussianScatteringFactors(
            gaussian_scattering_factors,
            error_function_scales,
            correction_scattering_factors,
        )

    def build(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ):
        return self.gaussian_scattering_factors(symbol, gpts, sampling, device=device)
