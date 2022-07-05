# from typing import Tuple, Union
#
# import numpy as np
# from ase.data import chemical_symbols
# from numba import jit, prange
# from scipy import integrate
# from scipy.interpolate import interp1d
# from scipy.optimize import brentq
# from scipy.special import erf
#
# from abtem import LineProfiles
# from abtem.core.axes import OrdinalAxis
# from abtem.core.backend import cp, get_array_module
# from abtem.core.fft import fft2, ifft2
# from abtem.core.grid import disc_meshgrid, polar_spatial_frequencies
# from abtem.core.utils import CopyMixin, EqualityMixin
# from abtem.measure.measure import FourierSpaceLineProfiles
# from abtem.potentials.infinite import superpose_deltas, _sinc
# from abtem.potentials.parametrizations.base import Parametrization
# from abtem.potentials.parametrizations import named_parametrizations
#
# if cp is not None:
#     from abtem.core.cuda import interpolate_radial_functions as interpolate_radial_functions_cuda
# else:
#     interpolate_radial_functions_cuda = None
#
#
# @jit(nopython=True, nogil=True, parallel=True)
# def interpolate_radial_functions(array: np.ndarray,
#                                  positions: np.ndarray,
#                                  disk_indices: np.ndarray,
#                                  sampling: Tuple[float, float],
#                                  radial_gpts: np.ndarray,
#                                  radial_functions: np.ndarray,
#                                  radial_derivative: np.ndarray):
#     n = radial_gpts.shape[0]
#     dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)
#
#     for i in range(positions.shape[0]):
#
#         px = int(round(positions[i, 0] / sampling[0]))
#         py = int(round(positions[i, 1] / sampling[1]))
#
#         for j in prange(disk_indices.shape[0]):
#             k = px + disk_indices[j, 0]
#             m = py + disk_indices[j, 1]
#
#             if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
#                 r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
#                                    (m * sampling[1] - positions[i, 1]) ** 2)
#
#                 idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))
#
#                 if idx < 0:
#                     array[k, m] += radial_functions[i, 0]
#                 elif idx < n - 1:
#                     slope = radial_derivative[i, idx]
#                     array[k, m] += radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope
#
#
# def validate_parametrization(parametrization):
#     if isinstance(parametrization, str):
#         parametrization = named_parametrizations[parametrization]()
#
#     return parametrization
#
#
# def cutoff_distance(symbol, parametrization, tolerance: float = 1e-3):
#     parametrization = validate_parametrization(parametrization)
#     potential = parametrization.potential(symbol)
#     return brentq(f=lambda r: potential(r) - tolerance, a=1e-3, b=1e3)
#
#
# class ProjectionIntegralTable:
#
#     def __init__(self, radial_gpts, limits, values):
#         assert values.shape[0] == len(limits)
#         assert values.shape[1] == len(radial_gpts)
#
#         self._radial_gpts = radial_gpts
#         self._limits = limits
#         self._values = values
#
#     @property
#     def radial_gpts(self):
#         return self._radial_gpts
#
#     @property
#     def limits(self):
#         return self._limits
#
#     @property
#     def values(self):
#         return self._values
#
#     def project(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> np.ndarray:
#         f = interp1d(self._limits, self._values, axis=0, kind='linear', fill_value='extrapolate')
#         return f(b) - f(a)
#
#     def project_on_grid(self,
#                         gpts: np.ndarray,
#                         sampling: Tuple[float, float],
#                         positions: np.ndarray,
#                         a: np.ndarray,
#                         b: np.ndarray,
#                         device='cpu') -> np.ndarray:
#
#         # if len(positions) == 0:
#         #    return array
#
#         # assert len(a) == len(b) == len(positions)
#         # assert len(array.shape) == 2
#         # assert len(a.shape) == 1
#         # assert len(b.shape) == 1
#         # assert len(sampling) == 2
#         #
#         xp = get_array_module(device)
#
#         array = xp.zeros(gpts, dtype=xp.float32)
#
#         disk_indices = xp.asarray(disc_meshgrid(int(np.ceil(self._radial_gpts[-1] / np.min(sampling)))))
#         radial_potential = xp.asarray(self.project(a, b))
#         print(radial_potential.shape)
#
#         positions = xp.asarray(positions)
#         radial_potential_derivative = xp.zeros_like(radial_potential)
#         radial_potential_derivative[:, :-1] = xp.diff(radial_potential, axis=1) / xp.diff(self.radial_gpts)[None]
#
#         if xp is cp:
#             interpolate_radial_functions_cuda(array=array,
#                                               positions=positions,
#                                               disk_indices=disk_indices,
#                                               sampling=sampling,
#                                               radial_gpts=xp.asarray(self.radial_gpts),
#                                               radial_functions=radial_potential,
#                                               radial_derivative=radial_potential_derivative)
#         else:
#             interpolate_radial_functions(array=array,
#                                          positions=positions,
#                                          disk_indices=disk_indices,
#                                          sampling=sampling,
#                                          radial_gpts=self.radial_gpts,
#                                          radial_functions=radial_potential,
#                                          radial_derivative=radial_potential_derivative)
#
#         return array
#
#
#
#
#
# def cutoff(func, tolerance, a, b) -> float:
#     return brentq(f=lambda r: func(r) - tolerance, a=a, b=b)  # noqa
#
#
# class AtomicPotential(CopyMixin, EqualityMixin):
#
#     def __init__(self,
#                  symbol: Union[int, str],
#                  parametrization: Union[str, Parametrization] = 'lobato',
#                  cutoff_tolerance: float = 1e-3,
#                  taper: float = 0.85,
#                  integration_step: float = 0.02,
#                  quad_order: int = 8):
#         """
#         The atomic
#
#         Parameters
#         ----------
#         symbol : int or str
#             The chemical symbol or atomic number.
#         parametrization : str, optional
#             The potential parametrization describing the radial dependence of the potential. Default is `lobato`.
#         cutoff_tolerance : float, optional
#             The error tolerance used for deciding the radial cutoff distance of the potential [eV / e].
#             The cutoff is only relevant for potentials using the 'finite' projection scheme. Default is 1e-3.
#         taper : float, optional
#             The fraction from the cutoff of the radial distance from the core where the atomic potential starts tapering
#             to zero. Default is 0.85.
#         integration_step : float, optional
#             The step size between integration limits used for calculating the integral table. Default is 0.05.
#         quad_order : int, optional
#             Order of quadrature integration passed to scipy.integrate.fixed_quad. Default is 8.
#         """
#
#         if isinstance(symbol, (int, np.int32, np.int64)):
#             symbol = chemical_symbols[symbol]
#
#         self._symbol = symbol
#         self._parametrization = validate_parametrization(parametrization)
#         self._taper = taper
#         self._quad_order = quad_order
#         self._cutoff_tolerance = cutoff_tolerance
#         self._integration_step = integration_step
#
#     @property
#     def symbol(self):
#         return self._symbol
#
#     @property
#     def parametrization(self):
#         return self._parametrization
#
#     @property
#     def quad_order(self):
#         return self._quad_order
#
#     @property
#     def cutoff_tolerance(self) -> float:
#         return self._cutoff_tolerance
#
#     @property
#     def integration_step(self) -> float:
#         return self._integration_step
#
#     def radial_gpts(self, inner_cutoff) -> np.ndarray:
#         num_points = int(np.ceil(self.cutoff / inner_cutoff))
#         return np.geomspace(inner_cutoff, self.cutoff, num_points)
#
#     @property
#     def cutoff(self) -> float:
#         return cutoff(self.potential, self.cutoff_tolerance, a=1e-3, b=1e3)  # noqa
#
#     @property
#     def parameters(self):
#         return self.parametrization.parameters
#
#     def potential(self, r) -> np.ndarray:
#         return self.parametrization.potential(self.symbol)(r)
#
#     def scattering_factor(self, k) -> np.ndarray:
#         return self.parametrization.scattering_factor(self.symbol)(k)
#
#     def charge(self, r) -> np.ndarray:
#         return self.parametrization.charge(self.symbol)(r)
#
#     def projected_potential(self, r) -> np.ndarray:
#         return self.parametrization.projected_potential(self.symbol)(r)
#
#     def finite_projected_potential(self, r, a, b) -> np.ndarray:
#         return self.parametrization.finite_projected_potential(self.symbol)(r, a, b)
#
#     def projected_scattering_factor(self, k) -> np.ndarray:
#         return self.parametrization.projected_scattering_factor(self.symbol)(k)
#
#     def finite_projected_scattering_factor(self, k, a, b) -> np.ndarray:
#         return self.parametrization.finite_projected_scattering_factor(self.symbol)(k, a, b)
#
#     def _taper_values(self, radial_gpts):
#         taper_start = self._taper * self.cutoff
#         taper_mask = radial_gpts > taper_start
#         taper_values = np.ones_like(radial_gpts)
#         taper_values[taper_mask] = (np.cos(
#             np.pi * (radial_gpts[taper_mask] - taper_start) / (self.cutoff - taper_start)) + 1.) / 2
#         return taper_values
#
#     def _integral_limits(self):
#         limits = np.linspace(-self.cutoff, 0, int(np.ceil(self.cutoff / self._integration_step)))
#         return np.concatenate((limits, -limits[::-1][1:]))
#
#     def build_integral_table(self, inner_limit) -> ProjectionIntegralTable:
#         """
#         Build table of projection integrals of the radial atomic potential.
#
#         Parameters
#         ----------
#         inner_limit : float, optional
#             Smallest radius from the core at which to calculate the projection integral [Å].
#
#         Returns
#         -------
#         radial_integral_table :
#             RadialIntegralTable
#         """
#
#         radial_gpts = self.radial_gpts(inner_limit)
#         limits = self._integral_limits()
#
#         projection = lambda z: self.potential(np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2))
#
#         table = np.zeros((len(limits) - 1, len(radial_gpts)))
#         table[0, :] = integrate.fixed_quad(projection, -limits[0] * 2, limits[0], n=self._quad_order)[0]
#
#         for j, (a, b) in enumerate(zip(limits[1:-1], limits[2:])):
#             table[j + 1] = table[j] + integrate.fixed_quad(projection, a, b, n=self._quad_order)[0]
#
#         table = table * self._taper_values(radial_gpts)[None]
#
#         return ProjectionIntegralTable(radial_gpts, limits[1:], table)
#
#     def line_profiles(self, sampling: float = 0.001, cutoff=None, name: str = 'potential'):
#
#         func = self.parametrization.get_function(name, self.symbol)
#
#         ensemble_axes_metadata = [OrdinalAxis(values=(self.symbol,))]
#
#         if cutoff is None:
#             cutoff = self.cutoff
#
#         # if name in real_space_funcs:
#         #     r = np.arange(sampling, cutoff, sampling)
#         #     return LineProfiles(func(r)[None], sampling=sampling, ensemble_axes_metadata=ensemble_axes_metadata)
#         #
#         # elif name in fourier_space_funcs:
#         #     k = np.arange(0., cutoff, sampling)
#         #     return FourierSpaceLineProfiles(func(k)[None], sampling=sampling,
#         #                                     ensemble_axes_metadata=ensemble_axes_metadata)
#
#     def show(self, name: str = 'potential', ax=None):
#         return self.line_profiles(name=name).show(ax=ax)
#
#     # def show(self, ax=None):
#     #     if ax is None:
#     #         ax = plt.subplot()
#     #
#     #     ax.plot(self.radial_gpts, self.evaluate(self.radial_gpts), label=self._symbol)
#     #     ax.set_ylabel('V [V]')
#     #     ax.set_xlabel('r [Å]')
