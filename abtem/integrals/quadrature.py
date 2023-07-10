"""Module to describe projection integrals using numerical quadrature rules."""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from abtem.core.backend import cp, get_array_module
from abtem.core.grid import disc_meshgrid
from abtem.integrals.base import ProjectionIntegrator, ProjectionIntegratorPlan
from abtem.parametrizations import validate_parametrization

if cp is not None:
    from abtem.core._cuda import (
        interpolate_radial_functions as interpolate_radial_functions_cuda,
    )
else:
    interpolate_radial_functions_cuda = None

if TYPE_CHECKING:
    from abtem.parametrizations.base import Parametrization


@jit(nopython=True, nogil=True)
def interpolate_radial_functions(
    array: np.ndarray,
    positions: np.ndarray,
    disk_indices: np.ndarray,
    sampling: tuple[float, float],
    radial_gpts: np.ndarray,
    radial_functions: np.ndarray,
    radial_derivative: np.ndarray,
):
    n = radial_gpts.shape[0]
    dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)

    for i in range(positions.shape[0]):

        px = int(round(positions[i, 0] / sampling[0]))
        py = int(round(positions[i, 1] / sampling[1]))

        for j in range(disk_indices.shape[0]):
            k = px + disk_indices[j, 0]
            m = py + disk_indices[j, 1]

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                r_interp = np.sqrt(
                    (k * sampling[0] - positions[i, 0]) ** 2
                    + (m * sampling[1] - positions[i, 1]) ** 2
                )

                idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))

                if idx < 0:
                    array[k, m] += radial_functions[i, 0]
                elif idx < n - 1:
                    slope = radial_derivative[i, idx]
                    array[k, m] += (
                        radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope
                    )


class ProjectionIntegralTable(ProjectionIntegrator):
    def __init__(self, radial_gpts: np.ndarray, limits: np.ndarray, values: np.ndarray):
        assert values.shape[0] == len(limits)
        assert values.shape[1] == len(radial_gpts)

        self._radial_gpts = radial_gpts
        self._limits = limits
        self._values = values

    @property
    def radial_gpts(self) -> np.ndarray:
        return self._radial_gpts

    @property
    def limits(self) -> np.ndarray:
        return self._limits

    @property
    def values(self) -> np.ndarray:
        return self._values

    def integrate(self, a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
        f = interp1d(
            self.limits, self.values, axis=0, kind="linear", fill_value="extrapolate"
        )
        return f(b) - f(a)

    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: float,
        b: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> np.ndarray:

        # assert len(a) == len(b) == len(positions)
        # assert len(a.shape) == 1
        # assert len(b.shape) == 1
        if np.isscalar(a):
            a = np.array([a] * len(positions))

        if np.isscalar(b):
            b = np.array([b] * len(positions))

        assert len(sampling) == 2

        xp = get_array_module(device)

        array = xp.zeros(gpts, dtype=xp.float32)

        a = a - positions[:, 2]
        b = b - positions[:, 2]

        disk_indices = xp.asarray(
            disc_meshgrid(int(np.ceil(self._radial_gpts[-1] / np.min(sampling))))
        )
        radial_potential = xp.asarray(self.integrate(a, b))

        positions = xp.asarray(positions, dtype=np.float32)

        # sampling = xp.array(sampling, dtype=xp.float32)
        # positions[:, :2] = xp.round(positions[:, :2] / sampling) * sampling

        radial_potential_derivative = xp.zeros_like(radial_potential)
        radial_potential_derivative[:, :-1] = (
            xp.diff(radial_potential, axis=1) / xp.diff(self.radial_gpts)[None]
        )

        if xp is cp:
            interpolate_radial_functions_cuda(
                array=array,
                positions=positions,
                disk_indices=disk_indices,
                sampling=sampling,
                radial_gpts=xp.asarray(self.radial_gpts),
                radial_functions=radial_potential,
                radial_derivative=radial_potential_derivative,
            )
        else:
            interpolate_radial_functions(
                array=array,
                positions=positions,
                disk_indices=disk_indices,
                sampling=sampling,
                radial_gpts=self.radial_gpts,
                radial_functions=radial_potential,
                radial_derivative=radial_potential_derivative,
            )

        return array


def cutoff(func: callable, tolerance: float, a: float, b: float) -> float:
    return brentq(f=lambda r: func(r) - tolerance, a=a, b=b)


class ProjectionQuadratureRule(ProjectionIntegratorPlan):
    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        cutoff_tolerance: float = 1e-3,
        taper: float = 0.85,
        integration_step: float = 0.02,
        quad_order: int = 8,
    ):
        """
        Integration plan for calculating projection integrals

        Parameters
        ----------
        parametrization : str, optional
            The potential parametrization describing the radial dependence of the potential. Default is `lobato`.
        cutoff_tolerance : float, optional
            The error tolerance used for deciding the radial cutoff distance of the potential [eV / e].
            The cutoff is only relevant for potentials using the 'finite' projection scheme. Default is 1e-3.
        taper : float, optional
            The fraction from the cutoff of the radial distance from the core where the atomic potential starts tapering
            to zero. Default is 0.85.
        integration_step : float, optional
            The step size between integration limits used for calculating the integral table. Default is 0.05.
        quad_order : int, optional
            Order of quadrature integration passed to scipy.integrate.fixed_quad. Default is 8.
        """
        self._parametrization = validate_parametrization(parametrization)
        self._taper = taper
        self._quad_order = quad_order
        self._cutoff_tolerance = cutoff_tolerance
        self._integration_step = integration_step
        super().__init__(periodic=False, finite=True)

    @property
    def parametrization(self):
        """The potential parametrization describing the radial dependence of the potential."""
        return self._parametrization

    @property
    def quad_order(self):
        """Order of quadrature integration."""
        return self._quad_order

    @property
    def cutoff_tolerance(self) -> float:
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def integration_step(self) -> float:
        """The step size between integration limits used for calculating the integral table."""
        return self._integration_step

    def cutoff(self, symbol: str) -> float:
        return cutoff(
            self.parametrization.potential(symbol), self.cutoff_tolerance, a=1e-3, b=1e3
        )

    @staticmethod
    def _radial_gpts(inner_cutoff: float, cutoff: float) -> np.ndarray:
        num_points = int(np.ceil(cutoff / inner_cutoff))
        return np.geomspace(inner_cutoff, cutoff, num_points)

    @staticmethod
    def _taper_values(radial_gpts: np.ndarray, cutoff: float, taper: float):
        taper_start = taper * cutoff
        taper_mask = radial_gpts > taper_start
        taper_values = np.ones_like(radial_gpts)
        taper_values[taper_mask] = (
            np.cos(
                np.pi * (radial_gpts[taper_mask] - taper_start) / (cutoff - taper_start)
            )
            + 1.0
        ) / 2
        return taper_values

    def _integral_limits(self, cutoff: float):
        limits = np.linspace(-cutoff, 0, int(np.ceil(cutoff / self._integration_step)))
        return np.concatenate((limits, -limits[::-1][1:]))

    def build_integral_table(
        self, symbol: str, inner_limit: float
    ) -> ProjectionIntegralTable:
        """
        Build table of projection integrals of the radial atomic potential.

        Parameters
        ----------
        inner_limit : float, optional
            Smallest radius from the core at which to calculate the projection integral [Å].

        Returns
        -------
        projection_integral_table :
            ProjectionIntegralTable
        """
        potential = self.parametrization.potential(symbol)
        cutoff = self.cutoff(symbol)

        radial_gpts = self._radial_gpts(inner_limit, cutoff)
        limits = self._integral_limits(cutoff)

        projection = lambda z: potential(
            np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2)
        )

        table = np.zeros((len(limits) - 1, len(radial_gpts)))
        table[0, :] = integrate.fixed_quad(
            projection, -limits[0] * 2, limits[0], n=self._quad_order
        )[0]

        for j, (a, b) in enumerate(zip(limits[1:-1], limits[2:])):
            table[j + 1] = (
                table[j] + integrate.fixed_quad(projection, a, b, n=self._quad_order)[0]
            )

        table = table * self._taper_values(radial_gpts, cutoff, self._taper)[None]

        return ProjectionIntegralTable(radial_gpts, limits[1:], table)

    def build(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> ProjectionIntegralTable:
        inner_limit = min(sampling) / 2
        return self.build_integral_table(symbol, inner_limit)
