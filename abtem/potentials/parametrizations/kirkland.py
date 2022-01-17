import json
import os

import numpy as np
from numba import jit
from scipy.special import kn
from abtem.potentials.utils import kappa


def load_parameters(scale_parameters=True):
    """Function to load the Kirkland parameters (doi:10.1007/978-1-4419-6533-2)."""
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kirkland_parameters.json'), 'r') as f:
        parameters = json.load(f)

    for key, value in parameters.items():
        value = np.array(value)

        if scale_parameters:
            a = value[0]
            b = value[1]
            c = value[2]
            d = value[3]

            a = np.pi * a
            b = 2. * np.pi * np.sqrt(b)
            c = np.pi ** (3. / 2.) * c / d ** (3. / 2.)
            d = np.pi ** 2 / d
            value = np.vstack((a, b, c, d))

        parameters[key] = value

    return parameters


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    return (p[0, 0] / (p[1, 0] + k ** 2) + p[2, 0] * np.exp(-p[3, 0] * k ** 2) +
            p[0, 1] / (p[1, 1] + k ** 2) + p[2, 1] * np.exp(-p[3, 1] * k ** 2) +
            p[0, 2] / (p[1, 2] + k ** 2) + p[2, 2] * np.exp(-p[3, 2] * k ** 2)) / kappa


@jit(nopython=True, nogil=True)
def potential(r, p):
    return (p[0, 0] * np.exp(-p[1, 0] * r) / r + p[2, 0] * np.exp(-p[3, 0] * r ** 2.) +
            p[0, 1] * np.exp(-p[1, 1] * r) / r + p[2, 1] * np.exp(-p[3, 1] * r ** 2.) +
            p[0, 2] * np.exp(-p[1, 2] * r) / r + p[2, 2] * np.exp(-p[3, 2] * r ** 2.)) / kappa


@jit(nopython=True, nogil=True)
def potential_derivative(r, p):
    dvdr = (- p[0, 0] * (1 / r + p[1, 0]) * np.exp(-p[1, 0] * r) / r -
            2 * p[2, 0] * p[3, 0] * r * np.exp(-p[3, 0] * r ** 2)
            - p[0, 1] * (1 / r + p[1, 1]) * np.exp(-p[1, 1] * r) / r -
            2 * p[2, 1] * p[3, 1] * r * np.exp(-p[3, 1] * r ** 2)
            - p[0, 2] * (1 / r + p[1, 2]) * np.exp(-p[1, 2] * r) / r -
            2 * p[2, 2] * p[3, 2] * r * np.exp(-p[3, 2] * r ** 2)) / kappa
    return dvdr


def projected_potential(r, p):
    v = (2 * p[0, 0] * kn(0, p[1, 0] * r) + np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.exp(-p[3, 0] * r ** 2.) +
         2 * p[0, 1] * kn(0, p[1, 1] * r) + np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.exp(-p[3, 1] * r ** 2.) +
         2 * p[0, 2] * kn(0, p[1, 2] * r) + np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.exp(-p[3, 2] * r ** 2.))
    return v / kappa


def projected_scattering_factor(k, p):
    f = (4 * np.pi * p[0, 0] / (4 * np.pi ** 2 * k ** 2 + p[1, 0] ** 2) +
         np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.pi / p[3, 0] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 0]) +
         4 * np.pi * p[0, 1] / (4 * np.pi ** 2 * k ** 2 + p[1, 1] ** 2) +
         np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.pi / p[3, 1] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 1]) +
         4 * np.pi * p[0, 2] / (4 * np.pi ** 2 * k ** 2 + p[1, 2] ** 2) +
         np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.pi / p[3, 2] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 2]))
    return f / kappa
