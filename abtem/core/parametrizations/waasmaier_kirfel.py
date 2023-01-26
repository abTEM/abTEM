import json
import os

import numpy as np
from ase.data import atomic_numbers
from numba import jit
from scipy.special import erf

from abtem.core.parametrizations.base import Parametrization
from abtem.core.constants import kappa
from ase import units


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    return p[5] + (
        p[0] * np.exp(-p[6] * k**2.0)
        + p[1] * np.exp(-p[7] * k**2.0)
        + p[2] * np.exp(-p[8] * k**2.0)
        + p[3] * np.exp(-p[9] * k**2.0)
        + p[4] * np.exp(-p[10] * k**2.0)
    )


# @jit(nopython=True, nogil=True)
def xray_scattering_factor_k2(k2, p):
    return p[5] + (
        p[0] * np.exp(-p[6] * k2)
        + p[1] * np.exp(-p[7] * k2)
        + p[2] * np.exp(-p[8] * k2)
        + p[3] * np.exp(-p[9] * k2)
        + p[4] * np.exp(-p[10] * k2)
    )


def scattering_factor_k2(k2, p):
    return (p[0] - xray_scattering_factor_k2(k2 * 0.25, p[1:])) / k2  # / .25


def finite_projected_scattering_factor(r, p, a, b):
    p = np.expand_dims(p, tuple(range(2, 2 + len(r.shape))))
    return (
        np.abs(erf(p[2] * b) - erf(p[2] * a))
        * p[0]
        * np.exp(-p[1] * r[None, ...] ** 2.0)
    ).sum(0) / 2


class WaasmaierKirfelParametrization(Parametrization):
    _functions = {
        "potential": scattering_factor,
        "scattering_factor": scattering_factor_k2,
        "projected_potential": scattering_factor,
        "projected_scattering_factor": scattering_factor_k2,
        #'finite_projected_potential': finite_projected_scattering_factor,
        #'finite_projected_scattering_factor': finite_projected_scattering_factor,
    }

    def __init__(self):
        with open(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "data/waasmaier_kirfel.json"
            ),
            "r",
        ) as f:
            parameters = json.load(f)

        super().__init__(parameters)

    def scaled_parameters(self, symbol):
        # scattering_factor = np.array(self.parameters[symbol])

        scattering_factor = np.zeros(12)
        scattering_factor[0] = (
            atomic_numbers[symbol.replace("+", "").replace("-", "")]
        )
        scattering_factor[1:] = np.array(self.parameters[symbol])
        scattering_factor[:7] /= 2 * np.pi**2 * units.Bohr

        # potential = np.vstack((np.pi ** (3. / 2.) * scattering_factor[0] / scattering_factor[1] ** (3 / 2.) / kappa,
        #                        np.pi ** 2 / scattering_factor[1]))
        #
        # projected_potential = np.vstack([1 / kappa * np.pi * scattering_factor[0] / scattering_factor[1],
        #                                  np.pi ** 2 / scattering_factor[1]])
        #
        # projected_scattering_factor = np.vstack([scattering_factor[0] / kappa, scattering_factor[1]])

        # finite_projected_scattering_factor = np.concatenate((projected_scattering_factor,
        #                                                      [np.pi / np.sqrt(projected_scattering_factor[1])]))
        #
        # finite_projected_potential = np.concatenate((projected_potential,
        #                                              [np.sqrt(projected_potential[1])]))

        return {  #'potential': potential,
            "scattering_factor": scattering_factor,
            #'finite_projected_potential': finite_projected_potential,
            #'finite_projected_scattering_factor': finite_projected_scattering_factor,
            #'projected_potential': projected_potential,
            #'projected_scattering_factor': projected_scattering_factor,
        }
