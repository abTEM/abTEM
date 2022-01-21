from abc import ABCMeta, abstractmethod


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


