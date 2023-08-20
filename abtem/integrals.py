"""Module to describe projection integrals of radial potential parametrizations."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import brentq, fsolve
from scipy.special import erf

from abtem.core.backend import cp, get_array_module
from abtem.core.backend import cupyx
from abtem.core.backend import device_name_from_array_module
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import disc_meshgrid
from abtem.core.grid import polar_spatial_frequencies
from abtem.core.grid import spatial_frequencies
from abtem.core.utils import EqualityMixin, CopyMixin
from abtem.parametrizations import validate_parametrization

if cp is not None:
    from abtem.core._cuda import (
        interpolate_radial_functions as interpolate_radial_functions_cuda,
    )
else:
    interpolate_radial_functions_cuda = None

if TYPE_CHECKING:
    from abtem.parametrizations import Parametrization


class ProjectionIntegrator:
    """Base class for projection integrator object used for calculating projection integrals of radial potentials."""

    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Integrate radial potential between two limits at the given 2D positions on a grid. The integration limits are
        only used when the integration method is finite.

        Parameters
        ----------
        positions : np.ndarray
            2D array of xy-positions of the centers of each radial function [Å].
        a : np.ndarray
            Lower integration limit of the projection integrals along z for each position [Å]. The limit is given
            relative to the center of the radial function.
        b : np.ndarray
            Upper integration limit of the projection integrals along z for each position [Å]. The limit is given
            relative to the center of the radial function.
        gpts : two int
            Number of grid points in `x` and `y` describing each slice of the potential.
        sampling : two float
            Sampling of the potential in `x` and `y` [1 / Å].
        device : str, optional
            The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
            configuration file.
        """
        pass


class ProjectionIntegratorPlan(EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    The ProjectionIntegratorPlan facilitates the creation of :class:`.ProjectionIntegrator` objects using the ``.build``
    method given a grid and a chemical symbol.

    Parameters
    ----------
    periodic : bool
        True indicates that the projection integrals are periodic perpendicular to the projection direction.
    finite : bool
        True indicates that the projection integrals are finite along the projection direction.
    """

    def __init__(self, periodic: bool, finite: bool):
        self._periodic = periodic
        self._finite = finite

    @property
    def periodic(self) -> bool:
        """True indicates that the created projection integrators are implemented only for periodic potentials."""
        return self._periodic

    @property
    def finite(self) -> bool:
        """True indicates that the created projection integrators are implemented only for infinite potential
        projections."""
        return self._finite

    @abstractmethod
    def build(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str,
    ) -> ProjectionIntegrator:
        """
        Build projection integrator for given chemical symbol, grid and device.

        Parameters
        ----------
        symbol : str
            Chemical symbol to build the projection integrator for.
        gpts : two int
            Number of grid points in `x` and `y` describing each slice of the potential.
        sampling : two float
            Sampling of the potential in `x` and `y` [1 / Å].
        device : str, optional
            The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
            configuration file.

        Returns
        -------
        projection_integrator : ProjectionIntegrator
            The projection integrator for the specified chemical symbol.
        """

        pass

    @abstractmethod
    def cutoff(self, symbol: str) -> float:
        """Radial cutoff of the potential for the given chemical symbol."""
        pass


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

        array = xp.zeros(
            self._gaussian_scattering_factors.shape[-2:], dtype=xp.complex64
        )

        for i in range(5):
            temp = xp.zeros_like(array, dtype=xp.complex64)

            superpose_deltas(positions, temp, weights=weights[i])

            array += fft2(temp, overwrite_x=False) * self._gaussian_scattering_factors[
                i
            ].astype(xp.complex64)

        return array

    def _integrate_correction_factors(self, positions, a, b, sampling, device):
        xp = get_array_module(device)

        temp = xp.zeros(
            self._gaussian_scattering_factors.shape[-2:], dtype=xp.complex64
        )

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

        shape = self._gaussian_scattering_factors.shape[-2:]

        assert gpts == shape

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
            return ifft2(array / sinc(shape, sampling, device)).real


class GaussianProjectionIntegrals(ProjectionIntegratorPlan):
    """
    Parameters
    ----------
    gaussian_parametrization : str or Parametrization, optional
        The Gaussian radial potential parametrization to integrate. Must be parametrization described by a superposition
        of Gaussians. Default is the Peng parametrization.
    correction_parametrization : str or Parametrization, optional
        The correction radial potential parametrization to integrate. Used for correcting the dependence of the
        potential close to the nuclear core. Default is the Lobato parametrization.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]. Default is 1e-3.
    """

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
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def gaussian_parametrization(self):
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
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

    def build(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ):
        xp = get_array_module(device)
        k, _ = polar_spatial_frequencies(gpts, sampling, xp=xp)

        parameters = xp.array(
            self.gaussian_parametrization.scaled_parameters(
                symbol, "projected_scattering_factor"
            )
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


def sinc(
    gpts: tuple[int, int], sampling: tuple[float, float], device: str = "cpu"
) -> np.ndarray:
    """
    Returns an array representing a 2D sinc function centered at [0, 0]. The result is used to
    compensate for the finite size of single pixels used for representing delta functions.

    Parameters
    ----------
    gpts : two int
        Number of grid points in the first and second dimension to evaluate the sinc over.
    sampling : two float
        Size of the pixels of the grid determining the scale of the sinc.
    device : str
        The array is created on this device ('cpu' or 'gpu').
    """
    xp = get_array_module(device)
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
    round_positions: bool = False,
) -> np.ndarray:
    """
    Add superposition of delta functions at given positions to a 2D array.

    Parameters
    ----------
    positions : np.ndarray
        Array of 2D positions as an nx2 array. The positions are given in units of pixels.
    array : np.ndarray
        The delta functions are added to this 2D array.
    weights : np.ndarray, optional
        If given each delta function is weighted by the given factor. Must match the length of `positions`.
    round_positions : bool, optional
        If True, the delta function positions are rounded to the center of the nearest pixel, otherwise subpixel
        precision is used.
    """

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
    elif device_name_from_array_module(xp) == "gpu":
        cupyx.scatter_add(array, (i, j), v)
    else:
        raise RuntimeError()

    return array


class ProjectedScatteringFactorIntegrator(ProjectionIntegrator):
    """
    A ProjectionIntegrator calculating infinite projections of radial potential parametrizations. The hybrid real and
    reciprocal space method by Wouter Van den Broek et al. is used.

    Parameters
    ----------
    scattering_factor : 2d array
        Array representing a projected scattering factor. The zero-frequency is assumed to be at [0,0].

    References
    ----------
    W. Van den Broek et al. Ultramicroscopy, 158:89–97, 2015. doi:10.1016/j.ultramic.2015.07.005.
    """

    def __init__(self, scattering_factor: np.ndarray):
        self._scattering_factor = scattering_factor

    @property
    def scattering_factor(self) -> np.ndarray:
        """Projected scattering factor array on a 2D grid."""
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

        f = self.scattering_factor / sinc(
            self._scattering_factor.shape[-2:], sampling, device
        )

        array *= f

        if fourier_space:
            return array.real
        else:
            array = ifft2(array, overwrite_x=True).real
            return array


class ProjectedScatteringFactors(ProjectionIntegratorPlan):
    """
    The :class:`.ProjectedScatteringFactors` facilitates the creation of :class:`.ProjectedScatteringFactorIntegrator`
    objects using the ``.build`` method given a grid and a chemical symbol.

    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The radial potential parametrization to integrate. Default is the Lobato parametrization.
    """

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
    ) -> ProjectedScatteringFactorIntegrator:

        xp = get_array_module(device)
        kx, ky = spatial_frequencies(gpts, sampling, xp=np)

        k2 = kx[:, None] ** 2 + ky[None] ** 2
        f = self.parametrization.projected_scattering_factor(symbol)(k2)
        f = xp.asarray(f, dtype=xp.float32)

        if symbol in self.parametrization.sigmas.keys():
            sigma = self.parametrization.sigmas[symbol]
            f = f * xp.exp(
                -xp.asarray(k2, dtype=xp.float32)
                * (xp.pi * sigma / xp.sqrt(3 / 2)) ** 2
            )

        return ProjectedScatteringFactorIntegrator(f)


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
    """
    A ProjectionIntegrator calculating finite projections of radial potential parametrizations. An integral table
    for each

     used to evaluate the projection integrals for each atom in a slice given p integral limits.
    The projected potential evaluated along the

    Parameters
    ----------
    radial_gpts : array
        The points along a radial in the `xy`-plane where the projection integrals of the integral table are evaluated.
    limits : array
        The points along the projection direction where the projection integrals are evaluated.
    values : array
    """

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
    """
    Calculate the point where a function becomes lower than a given tolerance within a given bracketing interval.

    Parameters
    ----------
    func : callable
        The function to calculate the cutoff for.
    tolerance : float
        The tolerance to calculate the cutoff for.
    a : float
        One end of the bracketing interval.
    b : float
        The other end of the bracketing interval.

    Returns
    -------
    cutoff : float
    """
    f = brentq(f=lambda r: np.abs(func(r)) - tolerance, a=a, b=b)  # noqa
    return f


def cutoff_taper(radial_gpts, cutoff, taper):
    taper_start = taper * cutoff
    taper_mask = radial_gpts > taper_start
    taper_values = np.ones_like(radial_gpts)
    taper_values[taper_mask] = (
        np.cos(np.pi * (radial_gpts[taper_mask] - taper_start) / (cutoff - taper_start))
        + 1.0
    ) / 2
    return taper_values


class ProjectionQuadratureRule(ProjectionIntegratorPlan):
    """
    Projection integration plan for calculating finite projection integrals based on Gaussian quadrature rule.

    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The potential parametrization describing the radial dependence of the potential. Default is 'lobato'.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]. Default is 1e-3.
    taper : float, optional
        The fraction from the cutoff of the radial distance from the core where the atomic potential starts tapering
        to zero. Default is 0.85.
    integration_step : float, optional
        The step size between integration limits used for calculating the integral table. Default is 0.02.
    quad_order : int, optional
        Order of quadrature integration passed to scipy.integrate.fixed_quad. Default is 8.
    """

    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        cutoff_tolerance: float = 1e-3,
        taper: float = 0.85,
        integration_step: float = 0.02,
        quad_order: int = 8,
    ):

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
        symbol : str
            Chemical symbol to build the integral table.
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
