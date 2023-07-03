import json
import os

import numpy as np
from ase import units
from numba import jit
from scipy.special import kn

from abtem.core.parametrizations.base import Parametrization, get_data_path
from abtem.core.constants import kappa


@jit(nopython=True, nogil=True)
def scattering_factor(k2, p):
    return ((p[0, 0] * (2. + p[1, 0] * k2) / (1. + p[1, 0] * k2) ** 2) +
            (p[0, 1] * (2. + p[1, 1] * k2) / (1. + p[1, 1] * k2) ** 2) +
            (p[0, 2] * (2. + p[1, 2] * k2) / (1. + p[1, 2] * k2) ** 2) +
            (p[0, 3] * (2. + p[1, 3] * k2) / (1. + p[1, 3] * k2) ** 2) +
            (p[0, 4] * (2. + p[1, 4] * k2) / (1. + p[1, 4] * k2) ** 2))


@jit(nopython=True, nogil=True)
def potential(r, p):
    return (p[0, 0] * (2. / (p[1, 0] * r) + 1.) * np.exp(-p[1, 0] * r) +
            p[0, 1] * (2. / (p[1, 1] * r) + 1.) * np.exp(-p[1, 1] * r) +
            p[0, 2] * (2. / (p[1, 2] * r) + 1.) * np.exp(-p[1, 2] * r) +
            p[0, 3] * (2. / (p[1, 3] * r) + 1.) * np.exp(-p[1, 3] * r) +
            p[0, 4] * (2. / (p[1, 4] * r) + 1.) * np.exp(-p[1, 4] * r))


@jit(nopython=True, nogil=True)
def potential_derivative(r, p):
    dvdr = - (p[0, 0] * (2. / (p[1, 0] * r ** 2) + 2. / r + p[1, 0]) * np.exp(-p[1, 0] * r) +
              p[0, 1] * (2. / (p[1, 1] * r ** 2) + 2. / r + p[1, 1]) * np.exp(-p[1, 1] * r) +
              p[0, 2] * (2. / (p[1, 2] * r ** 2) + 2. / r + p[1, 2]) * np.exp(-p[1, 2] * r) +
              p[0, 3] * (2. / (p[1, 3] * r ** 2) + 2. / r + p[1, 3]) * np.exp(-p[1, 3] * r) +
              p[0, 4] * (2. / (p[1, 4] * r ** 2) + 2. / r + p[1, 4]) * np.exp(-p[1, 4] * r))
    return dvdr


@jit(nopython=True, nogil=True)
def charge(r, p):
    n = (2 * np.pi ** 4 * units.Bohr * p[0, 0] / p[1, 0] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 0])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 1] / p[1, 1] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 1])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 2] / p[1, 2] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 2])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 3] / p[1, 3] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 3])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 4] / p[1, 4] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 4])))
    return n


@jit(nopython=True, nogil=True)
def x_ray_scattering_factor(k, p):
    n = (2 * np.pi ** 2 * units.Bohr * p[0, 0] / (p[1, 0] * (1 + p[1, 0] * k ** 2) ** 2) +
         2 * np.pi ** 2 * units.Bohr * p[0, 1] / (p[1, 1] * (1 + p[1, 1] * k ** 2) ** 2) +
         2 * np.pi ** 2 * units.Bohr * p[0, 2] / (p[1, 2] * (1 + p[1, 2] * k ** 2) ** 2) +
         2 * np.pi ** 2 * units.Bohr * p[0, 3] / (p[1, 3] * (1 + p[1, 3] * k ** 2) ** 2) +
         2 * np.pi ** 2 * units.Bohr * p[0, 4] / (p[1, 4] * (1 + p[1, 4] * k ** 2) ** 2))
    return n


def projected_potential(r, p):
    v = 2 * (2 * p[0][:, None] / p[1][:, None] * kn(0, r[None] * p[1][:, None]) +
             p[0][:, None] * r[None] * kn(1, r[None] * p[1][:, None])).sum(0)
    return v.astype(np.float32)


@jit(nopython=True, nogil=True)
def projected_scattering_factor(k2, p):
    pi = np.array(np.pi, dtype=np.float32)
    pi2 = np.array(np.pi ** 2, dtype=np.float32)
    k2 = 4 * pi2 * k2
    f = 8 * pi * ((p[0, 0] / p[1, 0] / (k2 + p[1, 0] ** 2) +
                   p[0, 0] * p[1, 0] / (k2 + p[1, 0] ** 2) ** 2) +
                  (p[0, 1] / p[1, 1] / (k2 + p[1, 1] ** 2) +
                   p[0, 1] * p[1, 1] / (k2 + p[1, 1] ** 2) ** 2) +
                  (p[0, 2] / p[1, 2] / (k2 + p[1, 2] ** 2) +
                   p[0, 2] * p[1, 2] / (k2 + p[1, 2] ** 2) ** 2) +
                  (p[0, 3] / p[1, 3] / (k2 + p[1, 3] ** 2) +
                   p[0, 3] * p[1, 3] / (k2 + p[1, 3] ** 2) ** 2) +
                  (p[0, 4] / p[1, 4] / (k2 + p[1, 4] ** 2) +
                   p[0, 4] * p[1, 4] / (k2 + p[1, 4] ** 2) ** 2)
                  )
    return f


class LobatoParametrization(Parametrization):
    _functions = {'potential': potential,
                  'scattering_factor': scattering_factor,
                  'projected_potential': projected_potential,
                  'projected_scattering_factor': projected_scattering_factor,
                  'x_ray_scattering_factor': x_ray_scattering_factor,
                  'charge': charge,
                  }

    def __init__(self, sigmas: dict[str, float] = None):
        path = os.path.join(get_data_path(), "lobato.json")

        with open(path, 'r') as f:
            parameters = json.load(f)

        super().__init__(parameters=parameters, sigmas=sigmas)

    def scaled_parameters(self, symbol):
        parameters = np.array(self.parameters[symbol])

        a = np.pi ** 2 * parameters[0] / parameters[1] ** (3 / 2.) / kappa
        b = 2 * np.pi / np.sqrt(parameters[1])
        scaled_parameters = np.vstack((a, b))

        return {'potential': scaled_parameters.astype(np.float32),
                'scattering_factor': parameters,
                'projected_potential': scaled_parameters.astype(np.float32),
                'projected_scattering_factor': scaled_parameters,
                'x_ray_scattering_factor': parameters,
                'charge': parameters
                }

#
