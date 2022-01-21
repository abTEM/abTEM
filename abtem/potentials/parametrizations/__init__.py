from abc import ABCMeta, abstractmethod

from abtem.potentials.parametrizations import lobato, kirkland, ewald, peng
import numpy as np

import os, json

from abtem.potentials.utils import kappa


class Parametrization(metaclass=ABCMeta):

    @abstractmethod
    def potential(self, r, symbol, charge=None):
        pass

    @abstractmethod
    def scattering_factor(self, k, symbol, charge=None):
        pass

    @abstractmethod
    def projected_potential(self, r, symbol, charge=None):
        pass

    @abstractmethod
    def projected_scattering_factor(self, k, symbol, charge=None):
        pass


class DataParametrization(Parametrization):

    def __init__(self, parameters, parametrizations, ionic_parametrizations=None):
        self._parameters = parameters
        self._parametrizations = parametrizations

        if ionic_parametrizations is None:
            ionic_parametrizations = {}

        self._ionic_parametrizations = ionic_parametrizations

    def _calculate(self, r, symbol, charge, func_name):
        try:
            parameters = self._parameters[func_name]
        except KeyError:
            raise RuntimeError(f'parameter for {func_name} not implemented')

        try:
            parameters = parameters[(symbol, charge)]
        except KeyError:
            raise RuntimeError(f'no parametrization for {symbol} with charge {charge}')

        # try:
        if abs(charge) > 0.:
            func = self._ionic_parametrizations[func_name]
            return func(r, parameters, charge)
        else:
            func = self._parametrizations[func_name]
            return func(r, parameters)
        # except KeyError:
        #    raise RuntimeError()

    def potential(self, r, symbol, charge=0.):
        return self._calculate(r, symbol, charge, 'potential')

    def scattering_factor(self, k, symbol, charge=0.):
        return self._calculate(k, symbol, charge, 'scattering_factor')

    def projected_potential(self, r, symbol, charge=0.):
        return self._calculate(r, symbol, charge, 'projected_potential')

    def projected_scattering_factor(self, k, symbol, charge=0.):
        return self._calculate(k, symbol, charge, 'projected_scattering_factor')


class KirklandParametrization(DataParametrization):

    def __init__(self):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/kirkland.json'), 'r') as f:
            data = json.load(f)

        parameters = {}
        scaled_parameters = {}
        for key, value in data.items():
            value = np.array(value)
            a = np.pi * value[0] / kappa
            b = 2. * np.pi * np.sqrt(value[1])
            c = np.pi ** (3. / 2.) * value[2] / value[3] ** (3. / 2.) / kappa
            d = np.pi ** 2 / value[3]

            parameters[(key, 0.)] = value
            scaled_parameters[(key, 0.)] = np.vstack([a, b, c, d])

        parameters = {'potential': scaled_parameters,
                      'scattering_factor': parameters,
                      'projected_potential': scaled_parameters,
                      'projected_scattering_factor': scaled_parameters,
                      }

        parametrizations = {'potential': kirkland.potential,
                            'scattering_factor': kirkland.scattering_factor,
                            'projected_potential': kirkland.projected_potential,
                            'projected_scattering_factor': kirkland.projected_scattering_factor,
                            }

        super().__init__(parameters, parametrizations)


class LobatoParametrization(DataParametrization):

    def __init__(self):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/lobato.json'), 'r') as f:
            data = json.load(f)

        parameters = {}
        scaled_parameters = {}
        for key, value in data.items():
            value = np.array(value)
            a = np.pi ** 2 * value[0] / value[1] ** (3 / 2.) / kappa
            b = 2 * np.pi / np.sqrt(value[1])
            parameters[(key, 0.)] = value
            scaled_parameters[(key, 0.)] = np.vstack((a, b))

        parameters = {'potential': scaled_parameters,
                      'scattering_factor': parameters,
                      'projected_potential': scaled_parameters,
                      'projected_scattering_factor': scaled_parameters,
                      }

        parametrizations = {'potential': lobato.potential,
                            'scattering_factor': lobato.scattering_factor,
                            'projected_potential': lobato.projected_potential,
                            'projected_scattering_factor': lobato.projected_scattering_factor,
                            }

        super().__init__(parameters, parametrizations)


class PengParametrization(DataParametrization):

    def __init__(self):
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/peng.json'), 'r') as f:
            data = json.load(f)

        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/peng_ionic.json'), 'r') as f:
            data_ionic = json.load(f)

        data.update(data_ionic)

        parameters = {}
        scaled_parameters = {}
        for key, value in data.items():
            value = np.array(value)

            charge = key.count('+') - key.count('-')
            symbol = key.replace('+', '').replace('-', '')

            #if not charge:
            #    value[1] /= 4

            a = np.pi ** (3. / 2.) * value[0] / value[1] ** (3. / 2.) / kappa
            b = np.pi ** 2 / value[1]

            parameters[(symbol, charge)] = value
            scaled_parameters[(symbol, charge)] = np.vstack((a, b))

        parameters = {'potential': scaled_parameters,
                      'scattering_factor': parameters,
                      # 'projected_potential': scaled_parameters,
                      # 'projected_scattering_factor': scaled_parameters,
                      }

        parametrizations = {'potential': peng.potential,
                            'scattering_factor': peng.scattering_factor,
                            # 'projected_potential': lobato.projected_potential,
                            # 'projected_scattering_factor': lobato.projected_scattering_factor,
                            }

        ionic_parametrizations = {'scattering_factor': peng.scattering_factor_ionic,
                                  'potential': peng.potential_ionic
                                  }

        super().__init__(parameters, parametrizations, ionic_parametrizations=ionic_parametrizations)
