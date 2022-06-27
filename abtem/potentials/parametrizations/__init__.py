from abc import ABCMeta, abstractmethod

from ase.data import chemical_symbols, atomic_numbers

from abtem.core.utils import EqualityMixin
from abtem.potentials.parametrizations import lobato, kirkland, ewald, peng
import numpy as np

import os, json
from ase import units
from ase.data import chemical_symbols

# Vacuum permitivity in ASE units
eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

# from abtem.potentials.utils import kappa
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

real_space_funcs = 'potential', 'projected_potential', 'charge'
fourier_space_funcs = 'scattering_factor', 'projected_scattering_factor'


class Parametrization(EqualityMixin, metaclass=ABCMeta):

    @abstractmethod
    def get_function(self, name, symbol, charge):
        pass

    def potential(self, symbol, charge=0.):
        return self.get_function('potential', symbol, charge)

    def scattering_factor(self, symbol, charge=0.):
        return self.get_function('scattering_factor', symbol, charge)

    def projected_potential(self, symbol, charge=0.):
        return self.get_function('projected_potential', symbol, charge)

    def projected_scattering_factor(self, symbol, charge=0.):
        return self.get_function('projected_scattering_factor', symbol, charge)

    def charge(self, symbol, charge=0.):
        return self.get_function('charge', symbol, charge)


#
# class ParametrizationFromFunction(Parametrization):
#
#     def __init__(self,
#                  parameters: dict,
#                  functions: dict):
#
#         self._parameters = parameters
#
#         self._functions = functions
#
#         if ionic_functions is None:
#             ionic_functions = {}
#
#         self._ionic_functions = ionic_functions
#
#     def _get_function_ionic(self):
#         pass
#
#     def get_function(self, name, symbol, charge=None):
#
#         if abs(charge) == 0.:
#
#
#         try:
#             parameters = self._parameters[name][symbol]
#             func = self._functions[name][symbol]
#         except KeyError:
#             raise RuntimeError(
#                 f'parameters for function "{name}" not implemented for element {symbol}')
#
#         return lambda r: func(r, parameters)
#
#         # # try:
#         # if abs(charge) > 0.:
#         #     func = self._ionic_parametrizations[func_name]
#         #     return func(r, parameters, charge)
#         # else:
#         #     func = self._parametrizations[func_name]
#         #     return func(r, parameters)
#         # # except KeyError:
#         # #    raise RuntimeError()
#
#     def potential(self, symbol, charge):
#         return self._get_parameterized_function('potential', symbol, charge=None)

# def scattering_factor(self, k, symbol, charge=0.):
#     return self._calculate(k, symbol, charge, 'scattering_factor')
#
# def projected_potential(self, r, symbol, charge=0.):
#     return self._calculate(r, symbol, charge, 'projected_potential')
#
# def projected_scattering_factor(self, k, symbol, charge=0.):
#     return self._calculate(k, symbol, charge, 'projected_scattering_factor')

class KirklandParametrization(Parametrization):
    _functions = {'potential': kirkland.potential,
                  'scattering_factor': kirkland.scattering_factor,
                  'projected_potential': kirkland.projected_potential,
                  'projected_scattering_factor': kirkland.projected_scattering_factor,
                  }

    def load_parameters(self, symbol):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/kirkland.json'), 'r') as f:
            parameters = np.array(json.load(f)[symbol])

        a = np.pi * parameters[0] / kappa
        b = 2. * np.pi * np.sqrt(parameters[1])
        c = np.pi ** (3. / 2.) * parameters[2] / parameters[3] ** (3. / 2.) / kappa
        d = np.pi ** 2 / parameters[3]

        scaled_parameters = np.vstack([a, b, c, d])

        return {'potential': scaled_parameters,
                'scattering_factor': parameters,
                'projected_potential': scaled_parameters,
                'projected_scattering_factor': scaled_parameters,
                }

    def get_function(self, name, symbol, charge=0.):
        if charge > 0.:
            raise RuntimeError('charge not implemented for parametrization "kirkland"')

        if isinstance(symbol, int):
            symbol = chemical_symbols[symbol]

        try:
            func = self._functions[name]
            parameters = self.load_parameters(symbol)[name]
            return lambda r: func(r, parameters)
        except KeyError:
            raise RuntimeError(f'parametrized function "{name}" does not exist for element {symbol}')


class LobatoParametrization(Parametrization):
    _functions = {'potential': lobato.potential,
                  'scattering_factor': lobato.scattering_factor,
                  'projected_potential': lobato.projected_potential,
                  'projected_scattering_factor': lobato.projected_scattering_factor,
                  'charge': lobato.charge,
                  }

    def load_parameters(self, symbol):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/lobato.json'), 'r') as f:
            parameters = np.array(json.load(f)[symbol])

        a = np.pi ** 2 * parameters[0] / parameters[1] ** (3 / 2.) / kappa
        b = 2 * np.pi / np.sqrt(parameters[1])
        scaled_parameters = np.vstack((a, b))

        return {'potential': scaled_parameters,
                'scattering_factor': parameters,
                'projected_potential': scaled_parameters,
                'projected_scattering_factor': scaled_parameters,
                'charge': parameters
                }

    def get_function(self, name, symbol, charge=0.):
        if isinstance(symbol, (int, np.int32, np.int64)):
            symbol = chemical_symbols[symbol]

        if charge > 0.:
            raise RuntimeError('charge not implemented for parametrization "lobato"')

        try:
            func = self._functions[name]
            parameters = self.load_parameters(symbol)[name]

            return lambda r: func(r, parameters)
        except KeyError:
            raise RuntimeError(f'parametrized function "{name}" does not exist for element {symbol}')


class EwaldParametrization(Parametrization):

    def __init__(self, width=1.):
        self._width = width
        self._functions = {'potential': ewald.potential}

    def get_function(self, name, symbol, charge=0.):
        if charge > 0.:
            raise RuntimeError('charge not implemented for parametrization "ewald"')

        try:
            func = self._functions[name]
            return lambda r: func(r, atomic_numbers[symbol], self._width)
        except KeyError:
            raise RuntimeError(f'parametrized function "{name}" does not exist for element {symbol}')


parametrizations = {'ewald': EwaldParametrization,
                    'lobato': LobatoParametrization,
                    'kirkland': KirklandParametrization}

# class PengParametrization(DataParametrization):
#
#     def __init__(self):
#         with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/peng.json'), 'r') as f:
#             data = json.load(f)
#
#         with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/peng_ionic.json'), 'r') as f:
#             data_ionic = json.load(f)
#
#         data.update(data_ionic)
#
#         parameters = {}
#         scaled_parameters = {}
#         for key, value in data.items():
#             value = np.array(value)
#
#             charge = key.count('+') - key.count('-')
#             symbol = key.replace('+', '').replace('-', '')
#
#             # if not charge:
#             #    value[1] /= 4
#
#             a = np.pi ** (3. / 2.) * value[0] / value[1] ** (3. / 2.) / kappa
#             b = np.pi ** 2 / value[1]
#
#             parameters[(symbol, charge)] = value
#             scaled_parameters[(symbol, charge)] = np.vstack((a, b))
#
#         parameters = {'potential': scaled_parameters,
#                       'scattering_factor': parameters,
#                       # 'projected_potential': scaled_parameters,
#                       # 'projected_scattering_factor': scaled_parameters,
#                       }
#
#         parametrizations = {'potential': peng.potential,
#                             'scattering_factor': peng.scattering_factor,
#                             # 'projected_potential': lobato.projected_potential,
#                             # 'projected_scattering_factor': lobato.projected_scattering_factor,
#                             }
#
#         ionic_parametrizations = {'scattering_factor': peng.scattering_factor_ionic,
#                                   'potential': peng.potential_ionic
#                                   }
#
#         super().__init__(parameters, parametrizations, ionic_parametrizations=ionic_parametrizations)
