from __future__ import annotations

import json
import os

import numpy as np
from numba import jit
from scipy.special import erf

from abtem.parametrizations.base import Parametrization, get_data_path
from abtem.core.constants import kappa


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * k**2.0)
        + p[0, 1] * np.exp(-p[1, 1] * k**2.0)
        + p[0, 2] * np.exp(-p[1, 2] * k**2.0)
        + p[0, 3] * np.exp(-p[1, 3] * k**2.0)
        + p[0, 4] * np.exp(-p[1, 4] * k**2.0)
    )


@jit(nopython=True, nogil=True)
def scattering_factor_k2(k2, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * k2)
        + p[0, 1] * np.exp(-p[1, 1] * k2)
        + p[0, 2] * np.exp(-p[1, 2] * k2)
        + p[0, 3] * np.exp(-p[1, 3] * k2)
        + p[0, 4] * np.exp(-p[1, 4] * k2)
    )


def finite_projected_scattering_factor(r, p, a, b):
    p = np.expand_dims(p, tuple(range(2, 2 + len(r.shape))))
    return (
        np.abs(erf(p[2] * b) - erf(p[2] * a))
        * p[0]
        * np.exp(-p[1] * r[None, ...] ** 2.0)
    ).sum(0) / 2


class PengParametrization(Parametrization):
    _functions = {
        "potential": scattering_factor,
        "scattering_factor": scattering_factor_k2,
        "projected_potential": scattering_factor,
        "projected_scattering_factor": scattering_factor_k2,
        "finite_projected_potential": finite_projected_scattering_factor,
        "finite_projected_scattering_factor": finite_projected_scattering_factor,
    }

    def __init__(self, sigmas: dict[str, float] = None):
        path = os.path.join(get_data_path(), "peng_high.json")

        with open(path, 'r') as f:
            parameters = json.load(f)

        super().__init__(parameters=parameters, sigmas=sigmas)

    def scaled_parameters(self, symbol):
        scattering_factor = np.array(self.parameters[symbol])
        scattering_factor[1] /= 2**2  # convert scattering factor units

        potential = np.vstack(
            (
                np.pi ** (3.0 / 2.0)
                * scattering_factor[0]
                / scattering_factor[1] ** (3 / 2.0)
                / kappa,
                np.pi**2 / scattering_factor[1],
            )
        )

        projected_potential = np.vstack(
            [
                1 / kappa * np.pi * scattering_factor[0] / scattering_factor[1],
                np.pi**2 / scattering_factor[1],
            ]
        )

        projected_scattering_factor = np.vstack(
            [scattering_factor[0] / kappa, scattering_factor[1]]
        )

        finite_projected_scattering_factor = np.concatenate(
            (
                projected_scattering_factor,
                [np.pi / np.sqrt(projected_scattering_factor[1])],
            )
        )

        finite_projected_potential = np.concatenate(
            (projected_potential, [np.sqrt(projected_potential[1])])
        )

        return {
            "potential": potential,
            "scattering_factor": scattering_factor,
            "finite_projected_potential": finite_projected_potential,
            "finite_projected_scattering_factor": finite_projected_scattering_factor,
            "projected_potential": projected_potential,
            "projected_scattering_factor": projected_scattering_factor,
        }
