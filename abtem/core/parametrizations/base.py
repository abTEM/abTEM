from abc import ABCMeta, abstractmethod

import numpy as np
from ase.data import chemical_symbols

from abtem.core.axes import OrdinalAxis
from abtem.core.utils import EqualityMixin
from abtem.measurements import ReciprocalSpaceLineProfiles, RealSpaceLineProfiles

real_space_funcs = "potential", "projected_potential", "charge"
fourier_space_funcs = "scattering_factor", "projected_scattering_factor"


class Parametrization(EqualityMixin, metaclass=ABCMeta):
    _functions: dict

    def __init__(self, parameters):
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def scaled_parameters(self, symbol):
        pass

    def potential(self, symbol, charge=0.0):
        return self.get_function("potential", symbol, charge)

    def scattering_factor(self, symbol, charge=0.0):
        return self.get_function("scattering_factor", symbol, charge)

    def projected_potential(self, symbol, charge=0.0):
        return self.get_function("projected_potential", symbol, charge)

    def projected_scattering_factor(self, symbol, charge=0.0):
        return self.get_function("projected_scattering_factor", symbol, charge)

    def charge(self, symbol, charge=0.0):
        return self.get_function("charge", symbol, charge)

    def x_ray_scattering_factor(self, symbol, charge=0.0):
        return self.get_function("x_ray_scattering_factor", symbol, charge)

    def finite_projected_potential(self, symbol, charge=0.0):
        return self.get_function("finite_projected_potential", symbol, charge)

    def finite_projected_scattering_factor(self, symbol, charge=0.0):
        return self.get_function("finite_projected_scattering_factor", symbol, charge)

    def get_function(self, name, symbol, charge=0.0):
        if isinstance(symbol, (int, np.int32, np.int64)):
            symbol = chemical_symbols[symbol]

        if charge > 0.0:
            raise RuntimeError(
                f"charge not implemented for parametrization {self.__class__.__name__}"
            )

        try:
            func = self._functions[name]
            parameters = self.scaled_parameters(symbol)[name]

            return lambda r, *args, **kwargs: func(r, parameters, *args, **kwargs)
        except KeyError:
            raise RuntimeError(
                f'parametrized function "{name}" does not exist for element {symbol}'
            )

    def line_profiles(
        self,
        symbol: str,
        cutoff: float,
        sampling: float = 0.001,
        name: str = "potential",
    ):

        func = self.get_function(name, symbol)

        ensemble_axes_metadata = [OrdinalAxis(values=(symbol,))]

        if name in real_space_funcs:
            r = np.arange(sampling, cutoff, sampling)
            return RealSpaceLineProfiles(
                func(r)[None],
                sampling=sampling,
                ensemble_axes_metadata=ensemble_axes_metadata,
            )

        elif name in fourier_space_funcs:
            k = np.arange(0.0, cutoff, sampling)
            return ReciprocalSpaceLineProfiles(
                func(k)[None],
                sampling=sampling,
                ensemble_axes_metadata=ensemble_axes_metadata,
            )
