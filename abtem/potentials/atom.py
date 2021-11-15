from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from numba import jit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from abtem.core.backend import cp, get_array_module
from abtem.core.grid import disc_meshgrid
from abtem.potentials.parametrizations import names as parametrization_names

if cp is not None:
    from abtem.core.cuda import interpolate_radial_functions as interpolate_radial_functions_cuda
else:
    interpolate_radial_functions_cuda = None


@jit(nopython=True, nogil=True)
def interpolate_radial_functions(array: np.ndarray,
                                 positions: np.ndarray,
                                 disk_indices: np.ndarray,
                                 sampling: Tuple[float, float],
                                 radial_gpts: np.ndarray,
                                 radial_functions: np.ndarray,
                                 radial_derivative: np.ndarray):
    n = radial_gpts.shape[0]
    dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)

    for i in range(positions.shape[0]):

        px = int(round(positions[i, 0] / sampling[0]))
        py = int(round(positions[i, 1] / sampling[1]))

        for j in range(disk_indices.shape[0]):
            k = px + disk_indices[j, 0]
            m = py + disk_indices[j, 1]

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                   (m * sampling[1] - positions[i, 1]) ** 2)

                idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))

                if idx < 0:
                    array[k, m] += radial_functions[i, 0]
                elif idx < n - 1:
                    slope = radial_derivative[i, idx]
                    array[k, m] += radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope


class AtomicPotential:

    def __init__(self, symbol: Union[int, str], parametrization: str = 'lobato', core_size: float = .01,
                 cutoff_tolerance: float = 1e-3):

        if not isinstance(symbol, str):
            symbol = chemical_symbols[symbol]

        self._symbol = symbol
        parametrization = parametrization_names[parametrization]

        self._parameters = parametrization.load_parameters()[symbol]
        self._potential = parametrization.potential

        self._cutoff_tolerance = cutoff_tolerance
        self._core_size = core_size
        self._integral_table = None
        self._cutoff = None

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def cutoff_tolerance(self) -> float:
        return self._cutoff_tolerance

    @property
    def radial_gpts(self) -> np.ndarray:
        num_points = int(np.ceil(self.cutoff / self._core_size))
        return np.geomspace(self._core_size, self.cutoff, num_points)

    @property
    def cutoff(self) -> float:
        if self._cutoff is None:
            self._cutoff = brentq(f=lambda r: self.evaluate(r) - self.cutoff_tolerance, a=1e-3, b=1e3)

        return self._cutoff

    def evaluate(self, r) -> np.ndarray:
        return self._potential(r, self._parameters)

    def build_integral_table(self, taper: float = .85) -> Tuple[np.ndarray, np.ndarray]:
        limits = np.linspace(-self.cutoff, 0, 50)
        limits = np.concatenate((limits, -limits[::-1][1:]))
        table = np.zeros((len(limits) - 1, len(self.radial_gpts)))

        for i, Ri in enumerate(self.radial_gpts):
            v = lambda z: self.evaluate(np.sqrt(Ri ** 2 + z ** 2))

            table[0, i] = quad(v, -np.inf, limits[0])[0]
            for j, limit in enumerate(limits[1:-1]):
                table[j + 1, i] = table[j, i] + quad(v, limit, limits[j + 2])[0]

        taper_start = taper * self.cutoff
        taper_mask = self.radial_gpts > taper_start
        taper_values = np.ones_like(self.radial_gpts)
        taper_values[taper_mask] = (np.cos(
            np.pi * (self.radial_gpts[taper_mask] - taper_start) / (self.cutoff - taper_start)) + 1.) / 2
        table = table * taper_values[None]

        self._integral_table = limits[1:], table
        return self._integral_table

    def project(self, a: float, b: float) -> np.ndarray:
        if self._integral_table is None:
            self.build_integral_table()

        f = interp1d(*self._integral_table, axis=0, kind='linear', fill_value='extrapolate')
        return f(b) - f(a)

    def project_on_grid(self,
                        array: np.ndarray,
                        sampling: Tuple[float, float],
                        positions: np.ndarray,
                        a: float,
                        b: float) -> np.ndarray:
        if len(positions) == 0:
            return array

        xp = get_array_module(array)

        disk_indices = xp.asarray(disc_meshgrid(int(np.ceil(self.cutoff / np.min(sampling)))))
        radial_potential = xp.asarray(self.project(a, b))

        radial_potential_derivative = xp.zeros_like(radial_potential)
        radial_potential_derivative[:, :-1] = xp.diff(radial_potential, axis=1) / xp.diff(self.radial_gpts)[None]

        positions = xp.asarray(positions)

        if xp is cp:
            interpolate_radial_functions_cuda(array=array,
                                              positions=positions,
                                              disk_indices=disk_indices,
                                              sampling=sampling,
                                              radial_gpts=xp.asarray(self.radial_gpts),
                                              radial_functions=radial_potential,
                                              radial_derivative=radial_potential_derivative)
        else:
            interpolate_radial_functions(array=array,
                                         positions=positions,
                                         disk_indices=disk_indices,
                                         sampling=sampling,

                                         radial_gpts=self.radial_gpts,
                                         radial_functions=radial_potential,
                                         radial_derivative=radial_potential_derivative)
        return array

    def show(self, ax=None):
        if ax is None:
            ax = plt.subplot()

        ax.plot(self.radial_gpts, self.evaluate(self.radial_gpts), label=self._symbol)
        ax.set_ylabel('V [V]')
        ax.set_xlabel('r [Å]')
