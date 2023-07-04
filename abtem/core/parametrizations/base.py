from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
from ase.data import chemical_symbols

from abtem.core.axes import OrdinalAxis
from abtem.core.utils import EqualityMixin
from abtem.measurements import ReciprocalSpaceLineProfiles, RealSpaceLineProfiles
from abtem.array import concatenate
import os

def get_data_path():
    this_file = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(this_file, "data")


class Parametrization(EqualityMixin, metaclass=ABCMeta):
    _functions: dict
    _real_space_funcs = (
        "potential",
        "projected_potential",
        "charge",
        "finite_projected_potential",
    )
    _fourier_space_funcs = (
        "scattering_factor",
        "projected_scattering_factor",
        "x_ray_scattering_factor",
        "finite_projected_scattering_factor",
    )

    def __init__(
        self, parameters: dict[str, np.ndarray], sigmas: dict[str, float] = None
    ):
        self._parameters = parameters

        if sigmas is None:
            sigmas = {}

        self._sigmas = sigmas

    @property
    def sigmas(self):
        return self._sigmas

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return self._parameters

    @abstractmethod
    def scaled_parameters(self, symbol) -> np.ndarray:
        pass

    def potential(self, symbol: str, charge: float = 0.0):
        return self.get_function("potential", symbol, charge)

    def scattering_factor(self, symbol: str, charge: float = 0.0):
        return self.get_function("scattering_factor", symbol, charge)

    def projected_potential(self, symbol: str, charge: float = 0.0):
        return self.get_function("projected_potential", symbol, charge)

    def projected_scattering_factor(self, symbol: str, charge: float = 0.0):
        return self.get_function("projected_scattering_factor", symbol, charge)

    def charge(self, symbol: str, charge: float = 0.0):
        return self.get_function("charge", symbol, charge)

    def x_ray_scattering_factor(self, symbol: str, charge: float = 0.0):
        return self.get_function("x_ray_scattering_factor", symbol, charge)

    def finite_projected_potential(self, symbol: str, charge: float = 0.0):
        return self.get_function("finite_projected_potential", symbol, charge)

    def finite_projected_scattering_factor(self, symbol: str, charge: float = 0.0):
        return self.get_function("finite_projected_scattering_factor", symbol, charge)

    def get_function(self, name: str, symbol: str, charge: float = 0.0):
        """
        Returns the line profiles for a parameterized function for one or more element.

        Parameters
        ----------
        name : {'potential', 'projected_potential', 'charge', 'finite_projected_potential', 'scattering_factor',
                'projected_scattering_factor', 'x_ray_scattering_factor', 'finite_projected_scattering_factor'}
            Name of the function to return.



        """
        if isinstance(symbol, (int, np.int32, np.int64)):
            symbol = chemical_symbols[symbol]

        if charge > 0.0:
            raise NotImplementedError

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
    ) -> RealSpaceLineProfiles | ReciprocalSpaceLineProfiles:
        """
        Returns the line profiles for a parameterized function for one or more element.

        Parameters
        ----------
        symbol : str or list of str
            Chemical symbol(s) of atom(s).
        cutoff : float
            The outer radial cutoff distance of the line profiles in units of Ångstrom for potentials and units of
            reciprocal Ångstrom for scattering factors.
        sampling : float, optional
            The radial sampling of the line profiles in units of Ångstrom for potentials and units of reciprocal
            Ångstrom for scattering factors. Default is 0.001.
        name : str
            Name of the line profile to return.
        """

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

        ensemble_axes_metadata = [
            OrdinalAxis(label="", values=(symbol,), _default_type="overlay")
        ]

        if name in self._real_space_funcs:
            r = np.arange(sampling, cutoff, sampling)
            metadata = {"label": "potential", "units": "eV/e"}
            return RealSpaceLineProfiles(
                func(r)[None],
                sampling=sampling,
                ensemble_axes_metadata=ensemble_axes_metadata,
                metadata=metadata,
            )

        elif name in self._fourier_space_funcs:
            k2 = np.arange(0.0, cutoff, sampling) ** 2
            metadata = {"label": "scattering factor", "units": "Å"}
            return ReciprocalSpaceLineProfiles(
                func(k2)[None],
                sampling=sampling,
                ensemble_axes_metadata=ensemble_axes_metadata,
                metadata=metadata,
            )
        else:
            raise RuntimeError(f"function name {name} not recognized")
