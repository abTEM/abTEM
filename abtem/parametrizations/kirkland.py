from __future__ import annotations
import json
import os

import numpy as np
from numba import jit
from scipy.special import kn

from abtem.parametrizations.base import Parametrization, get_data_path
from abtem.core.constants import kappa


@jit(nopython=True, nogil=True)
def scattering_factor(k2, p):
    return (
        p[0, 0] / (p[1, 0] + k2)
        + p[2, 0] * np.exp(-p[3, 0] * k2)
        + p[0, 1] / (p[1, 1] + k2)
        + p[2, 1] * np.exp(-p[3, 1] * k2)
        + p[0, 2] / (p[1, 2] + k2)
        + p[2, 2] * np.exp(-p[3, 2] * k2)
    )


@jit(nopython=True, nogil=True)
def potential(r, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * r) / r
        + p[2, 0] * np.exp(-p[3, 0] * r**2.0)
        + p[0, 1] * np.exp(-p[1, 1] * r) / r
        + p[2, 1] * np.exp(-p[3, 1] * r**2.0)
        + p[0, 2] * np.exp(-p[1, 2] * r) / r
        + p[2, 2] * np.exp(-p[3, 2] * r**2.0)
    )


@jit(nopython=True, nogil=True)
def potential_derivative(r, p):
    dvdr = (
        -p[0, 0] * (1 / r + p[1, 0]) * np.exp(-p[1, 0] * r) / r
        - 2 * p[2, 0] * p[3, 0] * r * np.exp(-p[3, 0] * r**2)
        - p[0, 1] * (1 / r + p[1, 1]) * np.exp(-p[1, 1] * r) / r
        - 2 * p[2, 1] * p[3, 1] * r * np.exp(-p[3, 1] * r**2)
        - p[0, 2] * (1 / r + p[1, 2]) * np.exp(-p[1, 2] * r) / r
        - 2 * p[2, 2] * p[3, 2] * r * np.exp(-p[3, 2] * r**2)
    )
    return dvdr


def projected_potential(r, p):
    v = (
        2 * p[0, 0] * kn(0, p[1, 0] * r)
        + np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.exp(-p[3, 0] * r**2.0)
        + 2 * p[0, 1] * kn(0, p[1, 1] * r)
        + np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.exp(-p[3, 1] * r**2.0)
        + 2 * p[0, 2] * kn(0, p[1, 2] * r)
        + np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.exp(-p[3, 2] * r**2.0)
    )
    return v


@jit(nopython=True, nogil=True)
def projected_scattering_factor(k2, p):
    pi = np.array(np.pi, dtype=np.float32)
    f = (
        4 * np.pi * p[0, 0] / (4 * pi**2 * k2 + p[1, 0] ** 2)
        + np.sqrt(np.pi / p[3, 0])
        * p[2, 0]
        * np.pi
        / p[3, 0]
        * np.exp(-(pi**2) * k2 / p[3, 0])
        + 4 * np.pi * p[0, 1] / (4 * pi**2 * k2 + p[1, 1] ** 2)
        + np.sqrt(np.pi / p[3, 1])
        * p[2, 1]
        * np.pi
        / p[3, 1]
        * np.exp(-(pi**2) * k2 / p[3, 1])
        + 4 * np.pi * p[0, 2] / (4 * pi**2 * k2 + p[1, 2] ** 2)
        + np.sqrt(np.pi / p[3, 2])
        * p[2, 2]
        * np.pi
        / p[3, 2]
        * np.exp(-(pi**2) * k2 / p[3, 2])
    )
    return f


class KirklandParametrization(Parametrization):
    _functions = {
        "potential": potential,
        "scattering_factor": scattering_factor,
        "projected_potential": projected_potential,
        "projected_scattering_factor": projected_scattering_factor,
    }

    def __init__(self, parameters: str = None, sigmas: dict[str, float] = None):
        if parameters is None:
            path = os.path.join(get_data_path(), "lobato.json")

            with open(path, "r") as f:
                parameters = json.load(f)

        super().__init__(parameters=parameters, sigmas=sigmas)

    def scaled_parameters(self, symbol):
        parameters = np.array(self.parameters[symbol])

        a = np.pi * parameters[0] / kappa
        b = 2.0 * np.pi * np.sqrt(parameters[1])
        c = np.pi ** (3.0 / 2.0) * parameters[2] / parameters[3] ** (3.0 / 2.0) / kappa
        d = np.pi**2 / parameters[3]

        scaled_parameters = np.vstack([a, b, c, d])

        return {
            "potential": scaled_parameters,
            "scattering_factor": parameters,
            "projected_potential": scaled_parameters,
            "projected_scattering_factor": scaled_parameters,
        }
