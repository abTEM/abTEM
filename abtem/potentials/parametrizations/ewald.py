import numpy as np
from scipy.special import erf

from abtem.potentials.parametrizations.base import Parametrization
from abtem.potentials.utils import eps0
from ase.data import chemical_symbols


def gaussian_potential(r, Z, width):
    return Z / (4 * np.pi * eps0) * erf(r / (np.sqrt(2) * width)) / r


def gaussian_charge(r, Z, width):
    return Z / (width ** 3 * np.sqrt(2 * np.pi) ** 3) * np.exp(-r ** 2 / (2 * width ** 2))


def point_charge_potential(r, Z):
    return Z / (4 * np.pi * eps0) / r


def potential(r, p):
    return point_charge_potential(r, p[1]) - gaussian_potential(r, p[1], p[0])


class EwaldParametrization(Parametrization):
    _functions = {'potential': potential}

    def __init__(self, width: float = 1.):
        parameters = {symbol: [width, Z] for Z, symbol in enumerate(chemical_symbols[1:], 1)}
        super().__init__(parameters=parameters)

    @property
    def width(self):
        return self.parameters['H'][0]

    def scaled_parameters(self, symbol):
        return {'potential': self.parameters[symbol]}
