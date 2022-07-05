import numpy as np
from scipy.special import erf

from abtem.potentials.parametrizations.base import Parametrization
from abtem.potentials.utils import eps0


def gaussian_potential(r, Z, width):
    return Z / (4 * np.pi * eps0) * erf(r / (np.sqrt(2) * width)) / r


def gaussian_charge(r, Z, width):
    return Z / (width ** 3 * np.sqrt(2 * np.pi) ** 3) * np.exp(-r ** 2 / (2 * width ** 2))


def point_charge_potential(r, Z):
    return Z / (4 * np.pi * eps0) / r


def potential(r, Z, width):
    return point_charge_potential(r, Z) - gaussian_potential(r, Z, width)


class EwaldParametrization(Parametrization):
    pass
    # _functions = {'potential': ewald.potential}
    #
    # def __init__(self, width=1.):
    #     self._width = width
    #
    # def load_parameters(self, symbol):
    #     return self._width
    #
    # def get_function(self, name, symbol, charge=0.):
    #     if charge > 0.:
    #         raise RuntimeError('charge not implemented for parametrization "ewald"')
    #
    #     try:
    #         func = self._functions[name]
    #         return lambda r: func(r, atomic_numbers[symbol], self._width)
    #     except KeyError:
    #         raise RuntimeError(f'parametrized function "{name}" does not exist for element {symbol}')
