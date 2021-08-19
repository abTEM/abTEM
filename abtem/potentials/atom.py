import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.interpolate import interp1d
from typing import Tuple
from abtem.potentials.parametrizations import load_lobato_parameters, load_kirkland_parameters
from abtem.potentials.parametrizations import lobato, kirkland
from abtem.basic.grid import disc_meshgrid
from numba import jit, prange


@jit(nopython=True, nogil=True, parallel=True)
def interpolate_radial_functions(array: np.ndarray,
                                 positions: np.ndarray,
                                 disk_indices: np.ndarray,
                                 sampling: Tuple[float, float],
                                 radial_gpts: np.ndarray,
                                 radial_potential: np.ndarray,
                                 radial_potential_derivative: np.ndarray):
    n = radial_gpts.shape[0]
    dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)

    for i in range(positions.shape[0]):

        px = int(round(positions[i, 0] / sampling[0]))
        py = int(round(positions[i, 1] / sampling[1]))

        for j in prange(disk_indices.shape[0]):  # Thread safe loop
            k = px + disk_indices[j, 0]
            m = py + disk_indices[j, 1]

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                   (m * sampling[1] - positions[i, 1]) ** 2)

                idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))

                if idx < 0:
                    array[k, m] += radial_potential[i, 0]
                elif idx < n - 1:
                    slope = radial_potential_derivative[i, idx]
                    array[k, m] += radial_potential[i, idx] + (r_interp - radial_gpts[idx]) * slope


class AtomicPotential:

    def __init__(self, atomic_number, parametrization, radial_gpts, tolerance=1e-3):
        self._atomic_number = atomic_number

        if parametrization == 'kirkland':
            self._parameters = load_kirkland_parameters()[atomic_number]
            self._potential = kirkland
        elif parametrization == 'lobato':
            self._parameters = load_lobato_parameters()[atomic_number]
            self._potential = lobato

        self._tolerance = tolerance
        self._radial_gpts = radial_gpts
        self._integral_table = None
        self._cutoff = None

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def cutoff(self) -> float:
        if self._cutoff is None:
            self._calculate_cutoff()

        return self._cutoff

    def evaluate(self, r) -> np.ndarray:
        return self._potential(r, self._parameters)

    def _calculate_cutoff(self):
        self._cutoff = brentq(f=lambda r: self._potential(r, self.parameters) - self.tolerance, a=1e-3, b=1e3)

    def _build_integral_table(self):
        limits = np.linspace(-self.cutoff, 0, 100)
        limits = np.concatenate((limits, -limits[::-1][1:]))
        table = np.zeros((len(limits) - 1, len(self._radial_gpts)))

        for i, Ri in enumerate(self._radial_gpts):
            R2 = Ri ** 2
            v = lambda z: kirkland(np.sqrt(R2 + z ** 2), self.parameters)

            table[0, i] = quad(v, -np.inf, limits[0])[0]
            for j, limit in enumerate(limits[1:-1]):
                table[j + 1, i] = table[j, i] + quad(v, limit, limits[j + 2])[0]

        self._integral_table = limits[1:], table

    def project(self, a, b) -> np.ndarray:
        if self._integral_table is None:
            self._build_integral_table()

        f = interp1d(*self._integral_table, axis=0, kind='linear', fill_value='extrapolate')
        return f(b) - f(a)

    def project_on_grid(self, array, sampling, positions, a, b):
        disk_indices = np.array(disc_meshgrid(int(np.ceil(self.cutoff / np.min(sampling))))).T
        radial_potential = self.project(a, b)

        radial_potential_derivative = np.zeros_like(radial_potential)
        radial_potential_derivative[:, :-1] = np.diff(radial_potential, axis=1) / np.diff(self._radial_gpts)[None]

        interpolate_radial_functions(array=array,
                                     positions=positions,
                                     disk_indices=disk_indices,
                                     sampling=sampling,
                                     radial_potential=radial_potential,
                                     radial_gpts=self._radial_gpts,
                                     radial_potential_derivative=radial_potential_derivative)
        return array
