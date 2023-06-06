from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
from ase.data import chemical_symbols

from abtem.core.axes import OrdinalAxis
from abtem.core.utils import EqualityMixin
from abtem.measurements import ReciprocalSpaceLineProfiles, RealSpaceLineProfiles
from abtem.array import concatenate

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

    def potential(self, symbol, charge:float=0.0):
        return self.get_function("potential", symbol, charge)

    def scattering_factor(self, symbol, charge:float=0.0):
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

        # if charge > 0.0:
        #     raise RuntimeError(
        #         f"charge not implemented for parametrization {self.__class__.__name__}"
        #     )

        if charge == 0.0:
            charge_symbol = ""
        elif charge > 0.0:
            charge_symbol = "+" * int(abs(charge))
        else:
            charge_symbol = "-" * int(abs(charge))

        try:
            func = self._functions[name]
            parameters = np.array(
                self.scaled_parameters(symbol + charge_symbol)[name], dtype=np.float32
            )
            return lambda r, *args, **kwargs: func(r, parameters, *args, **kwargs)
        except KeyError:
            raise RuntimeError(
                f'parametrized function "{name}" does not exist for element {symbol} with charge {charge}'
            )

    def line_profiles(
        self,
        symbol: str | Sequence[str],
        cutoff: float,
        sampling: float = 0.001,
        name: str = "potential",
    ):

        if not isinstance(symbol, str):
            return concatenate(
                [
                    self.line_profiles(
                        s,
                        cutoff=cutoff,
                        sampling=sampling,
                        name=name,
                    )
                    for s in symbol
                ]
            )

        func = self.get_function(name, symbol)

        ensemble_axes_metadata = [OrdinalAxis(label="", values=(symbol,), _default_type="overlay")]

        if name in real_space_funcs:
            r = np.arange(sampling, cutoff, sampling)
            metadata = {"label": "potential", "units": "eV/e"}
            return RealSpaceLineProfiles(
                func(r)[None],
                sampling=sampling,
                ensemble_axes_metadata=ensemble_axes_metadata,
                metadata=metadata,
            )

        elif name in fourier_space_funcs:
            k2 = np.arange(0.0, cutoff, sampling) ** 2
            metadata = {"label": "scattering factor", "units": "Å"}
            return ReciprocalSpaceLineProfiles(
                func(k2)[None],
                sampling=sampling,
                ensemble_axes_metadata=ensemble_axes_metadata,
                metadata=metadata
            )
