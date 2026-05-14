"""Module for describing analytical potential parametrizations."""

from __future__ import annotations

import json
import os
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Callable, Sequence

import numpy as np
from ase.data import chemical_symbols
from scipy.optimize import least_squares

from abtem.array import concatenate
from abtem.core.axes import OrdinalAxis
from abtem.core.constants import kappa
from abtem.core.utils import EqualityMixin, get_dtype
from abtem.measurements import RealSpaceLineProfiles, ReciprocalSpaceLineProfiles
from abtem.parametrizations.functions import ewald, kirkland, lobato, peng


def _get_data_path():
    this_file = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(this_file, "data")


def validate_sigmas(sigmas: float | dict | None) -> dict:
    if sigmas is None:
        sigmas = {}
    elif isinstance(sigmas, Number):
        sigmas = {chemical_symbol: sigmas for chemical_symbol in chemical_symbols}

    if not isinstance(sigmas, dict):
        raise ValueError()
    return sigmas


class Parametrization(EqualityMixin, metaclass=ABCMeta):
    """
    Base class for potential parametrizations.

    Parameters
    ----------
    parameters : str or dict
        A given string must be either the full path to a valid ``.json`` file or the
        name of a valid ``.json`` file in the ``abtem/parametrizations/data/`` folder.
        A given dictionary must be a mapping from chemical symbols to anything that can
        converted to a NumPy array with the correct shape for the given parametrization.
    sigmas : dict or float
        The standard deviation of isotropic displacements for each element as a mapping
        from chemical symbols to a number. If given as a float the standard deviation of
        the displacements is assumed to be identical for all atoms.
    """

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
        self, parameters: dict[str, np.ndarray] | str, sigmas: dict[str, float] | None = None
    ):
        self._parameters = validate_parameters(parameters)
        self._sigmas = validate_sigmas(sigmas)

    def to_json(self, file: str):
        with open(file, "w") as fp:
            data = {
                symbol: parameters.tolist()
                for symbol, parameters in self.parameters.items()
            }
            json.dump(data, fp)

    def from_json(self, file: str):
        with open(file, "r") as fp:
            self._parameters = {
                symbol: np.array(parameters)
                for symbol, parameters in json.load(fp).items()
            }

    @property
    def sigmas(self):
        """The standard deviation of isotropic displacements."""
        return self._sigmas

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return self._parameters

    @abstractmethod
    def scaled_parameters(self, symbol: str, name: str) -> np.ndarray:
        """
        The parameters of the parametrization scaled to the abTEM units for a given
        function.

        Parameters
        ----------
        symbol : str
            Chemical symbol to get the scaled parameters for.
        name : str
            Name of the function to get the scaled parameters for.

        Returns
        -------
        scaled_parameters : np.ndarray
        """

        pass

    def potential(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        Radial electrostatic potential for given chemical symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        electrostatic_potential : callable
            Function describing electrostatic potential parameterized by the radial
            distance to the core [Å].
        """
        return self.get_function("potential", symbol, charge)

    def scattering_factor(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        Radial scattering factor for given chemical symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        scattering_factor : callable
            Function describing scattering parameterized by the squared reciprocal
            radial distance to the core [1/Å^2].
        """
        return self.get_function("scattering_factor", symbol, charge)

    def projected_potential(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        Analytical infinite projection of radial electrostatic potential for given
        chemical symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        projected_potential : callable
            Function describing projected electrostatic potential parameterized by the
            radial distance to the core [Å].
        """
        return self.get_function("projected_potential", symbol, charge)

    def projected_scattering_factor(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        Analytical infinite projection of radial scattering factor for given chemical
        symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        projected_scattering_factor : callable
            Function describing projected scattering parameterized by the squared
            reciprocal radial distance to the core [1/Å^2].
        """
        return self.get_function("projected_scattering_factor", symbol, charge)

    def charge(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        Radial charge distribution for given chemical symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        charge : callable
            Function describing charge parameterized by the radial distance to the core
            [Å].
        """
        return self.get_function("charge", symbol, charge)

    def x_ray_scattering_factor(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        X-ray scattering factor for given chemical symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        x_ray_scattering_factor : callable
        """
        return self.get_function("x_ray_scattering_factor", symbol, charge)

    def finite_projected_potential(self, symbol: str, charge: float = 0.0) -> Callable:
        """
        X-ray scattering factor for given chemical symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        x_ray_scattering_factor : callable
        """
        return self.get_function("finite_projected_potential", symbol, charge)

    def finite_projected_scattering_factor(
        self, symbol: str, charge: float = 0.0
    ) -> Callable:
        """
        Analytical infinite projection of radial scattering factor for given chemical
        symbol and charge.

        Parameters
        ----------
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.

        Returns
        -------
        projected_scattering_factor : callable
            Function describing projected scattering parameterized by the squared
            reciprocal radial distance to the core [1/Å^2].
        """
        return self.get_function("finite_projected_scattering_factor", symbol, charge)

    def get_function(
        self, name: str, symbol: str, charge: float = 0.0, screening: float = 0.0
    ) -> Callable:
        """
        Returns a callable for a parameterized function for one element.

        Parameters
        ----------
        name : str
            Name of the function to return.
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.
        screening : float
            Screening wavevector κ [1/Å] for the ionic Coulomb correction
            ΔZ·M/(κ²+q²). Only used by PengParametrization for charged species.
        """
        if isinstance(symbol, (int, np.int32, np.int64)):
            symbol = chemical_symbols[symbol]

        if charge == 0.0:
            charge_symbol = ""
        elif charge > 0.0:
            charge_symbol = "+" * int(abs(charge))
        else:
            charge_symbol = "-" * int(abs(charge))

        try:
            func = self._functions[name]
            parameters = self.scaled_parameters(symbol + charge_symbol, name)
            dtype = get_dtype(complex=False)
            parameters = np.array(parameters, dtype=dtype)
            return lambda r, *args, **kwargs: func(r, parameters, *args, **kwargs)
        except KeyError:
            raise RuntimeError(
                f"parametrized function '{name}' does not exist for element {symbol}"
                f" with charge {charge}"
            )

    def line_profiles(
        self,
        symbol: str | Sequence[str],
        cutoff: float,
        sampling: float = 0.001,
        name: str = "potential",
        charge: float = 0.0,
        screening: float = 0.0,
    ) -> RealSpaceLineProfiles | ReciprocalSpaceLineProfiles:
        """
        Returns the line profiles for a parameterized function for one or more element.

        Parameters
        ----------
        symbol : str or list of str
            Chemical symbol(s) of atom(s).
        cutoff : float
            The outer radial cutoff distance of the line profiles in units of Ångstrom
            for potentials and units of reciprocal Ångstrom for scattering factors.
        sampling : float, optional
            The radial sampling of the line profiles in units of Ångstrom for potentials
            and units of reciprocal Ångstrom for scattering factors. Default is 0.001.
        name : str
            Name of the line profile to return.
        charge : float, optional
            Charge of the element in elementary units. Default is 0.0.
        screening : float, optional
            Screening wavevector κ [1/Å] for the ionic Coulomb correction. Default is 0.0.
        """

        if not isinstance(symbol, str):
            return concatenate(
                [
                    self.line_profiles(
                        s,
                        cutoff=cutoff,
                        sampling=sampling,
                        name=name,
                        charge=charge,
                        screening=screening,
                    )
                    for s in symbol
                ]
            )

        func = self.get_function(name, symbol, charge, screening)

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


class KirklandParametrization(Parametrization):
    """
    Potential parametrization by Earl J Kirkland.

    Parameters
    ----------
    parameters : str or dict
        A given string must be either the full path to a valid ``.json`` file or the
        name of a valid ``.json`` file in the ``abtem/parametrizations/data/`` folder.
        A given dictionary must be a mapping from chemical symbols to anything that can
        converted to a NumPy array of shape (4, 3).
    sigmas : dict or float
        The standard deviation of isotropic displacements for each element as a mapping
        from chemical symbols to a number. If given as a float the standard deviation of
        the displacements is assumed to be identical for all atoms.

    References
    ----------
    E.J. Kirkland. Advanced computing in electron microscopy. Springer, 2. edition, 2010
    """

    _functions = {
        "potential": kirkland.potential,
        "scattering_factor": kirkland.scattering_factor,
        "projected_potential": kirkland.projected_potential,
        "projected_scattering_factor": kirkland.projected_scattering_factor,
    }

    def __init__(
        self, parameters: str | dict = "kirkland.json", sigmas: dict[str, float] = None
    ):
        super().__init__(parameters=parameters, sigmas=sigmas)

    def fit(self, Z, k, f, guess=None):
        def reshape_parameters(p):
            p = p.reshape((4, 3))
            return p

        def apply_constraint(p):
            p = p.copy()
            p = np.abs(p)
            return p

        def make_residuals_func(k2, target, func):
            def residuals_func(p):
                p = reshape_parameters(p)
                p = apply_constraint(p)
                return target - func(k2, p)

            return residuals_func

        if guess is None:
            if chemical_symbols[Z] in self.parameters:
                guess = self.scaled_parameters(chemical_symbols[Z], "scattering_factor")
            else:
                guess = np.array(
                    validate_parameters("kirkland.json")[chemical_symbols[Z]]
                )

        func = self._functions["scattering_factor"]

        residuals_func = make_residuals_func(k**2, f, func)

        result = least_squares(residuals_func, guess.ravel())
        p_optimal = reshape_parameters(result.x)
        p_optimal = apply_constraint(p_optimal)
        self.parameters[chemical_symbols[Z]] = p_optimal
        return p_optimal

    def scaled_parameters(self, symbol: str, name: str) -> np.ndarray:
        parameters = np.array(self.parameters[symbol])

        a = np.pi * parameters[0] / kappa
        b = 2.0 * np.pi * np.sqrt(parameters[1])
        c = np.pi ** (3.0 / 2.0) * parameters[2] / parameters[3] ** (3.0 / 2.0) / kappa
        d = np.pi**2 / parameters[3]

        scaled_parameters = np.vstack([a, b, c, d])

        scaled_parameters = {
            "potential": scaled_parameters,
            "scattering_factor": parameters,
            "projected_potential": scaled_parameters,
            "projected_scattering_factor": scaled_parameters,
        }

        return scaled_parameters[name]


class LobatoParametrization(Parametrization):
    """
    Potential parametrization by Ivan Lobato and Dirk Van Dyck.

    Parameters
    ----------
    parameters : str or dict
        A given string must be either the full path to a valid ``.json`` file or the
        name of a valid ``.json`` file in the ``abtem/parametrizations/data/`` folder.
        A given dictionary must be a mapping from chemical symbols to anything that can
        converted to a NumPy array of shape (2, 5).
    sigmas : dict or float
        The standard deviation of isotropic displacements for each element as a mapping
        from chemical symbols to a number. If given as a float the standard deviation of
        the displacements is assumed to be identical for all atoms.

    References
    ----------
    Ivan Lobato and Dirk Van Dyck. Acta Crystallographica Section A, 70:636-649, 2014.
    """

    _functions = {
        "potential": lobato.potential,
        "scattering_factor": lobato.scattering_factor,
        "projected_potential": lobato.projected_potential,
        "projected_scattering_factor": lobato.projected_scattering_factor,
        "x_ray_scattering_factor": lobato.x_ray_scattering_factor,
        "charge": lobato.charge,
    }

    def __init__(
        self, parameters: str | dict = "lobato.json", sigmas: dict[str, float] = None
    ):
        super().__init__(parameters=parameters, sigmas=sigmas)

    def fit(self, Z, k, f, guess=None):
        def reshape_parameters(p):
            p = p.reshape((2, 5))
            return p

        def apply_constraint(p):
            p = p.copy()
            p[1, :] = np.abs(p[1, :])
            return p

        def make_residuals_func(k2, target, func):
            def residuals_func(p):
                p = reshape_parameters(p)
                p = apply_constraint(p)
                return target - func(k2, p)

            return residuals_func

        if guess is None:
            if chemical_symbols[Z] in self.parameters:
                guess = self.scaled_parameters(chemical_symbols[Z], "scattering_factor")
            else:
                guess = np.array(
                    validate_parameters("lobato.json")[chemical_symbols[Z]]
                )

        func = self._functions["scattering_factor"]

        residuals_func = make_residuals_func(k**2, f, func)

        result = least_squares(residuals_func, guess.ravel())
        p_optimal = reshape_parameters(result.x)
        p_optimal = apply_constraint(p_optimal)
        self.parameters[chemical_symbols[Z]] = p_optimal
        return p_optimal

    def scaled_parameters(self, symbol: str, name: str) -> np.ndarray:
        parameters = np.array(self.parameters[symbol])

        a = np.pi**2 * parameters[0] / parameters[1] ** (3 / 2.0) / kappa
        b = 2 * np.pi / np.sqrt(parameters[1])
        scaled_parameters = np.vstack((a, b))

        scaled_parameters = {
            "potential": scaled_parameters,
            "scattering_factor": parameters,
            "projected_potential": scaled_parameters,
            "projected_scattering_factor": scaled_parameters,
            "x_ray_scattering_factor": parameters,
            "charge": parameters,
        }

        return scaled_parameters[name]


class PengParametrization(Parametrization):
    """
    Potential parametrization by Lian-Mao Peng.

    Parameters
    ----------
    parameters : str or dict
        A given string must be either the full path to a valid ``.json`` file or the
        name of a valid ``.json`` file in the ``abtem/parametrizations/data/`` folder.
        A given dictionary must be a mapping from chemical symbols to anything that can
        converted to a NumPy array of shape (2, 5).
    sigmas : dict or float
        The standard deviation of isotropic displacements for each element as a mapping
        from chemical symbols to a number. If given as a float the standard deviation of
        the displacements is assumed to be identical for all atoms.

    References
    ----------
    L. Peng. Micron, 30(6):625–648, 1999.
    """

    _functions = {
        "potential": peng.scattering_factor,
        "scattering_factor": peng.scattering_factor_k2,
        "projected_potential": peng.scattering_factor,
        "projected_scattering_factor": peng.scattering_factor_k2,
        "finite_projected_potential": peng.finite_projected_scattering_factor,
        "finite_projected_scattering_factor": peng.finite_projected_scattering_factor,
    }

    # Mott formula constant (m₀e²/8π²ε₀ℏ²) in units of Å, from Peng 1999 eq. (3).
    # Converts ionic charge ΔZ to an additive 1/s² contribution to f_el(s).
    _mott_constant = 0.02393366

    def __init__(
        self,
        parameters: str | dict = "peng_high.json",
        sigmas: dict[str, float] | None = None,
        screening: float = 0.0,
    ):
        super().__init__(parameters=parameters, sigmas=sigmas)
        self._screening = screening

    # k²-domain functions that need the ΔZ / s² Coulomb correction for ionic species.
    # projected_scattering_factor: scaled_parameters divides a_i by kappa → include kappa.
    # scattering_factor:           scaled_parameters keeps raw a_i         → no kappa.
    _ionic_k2_names = frozenset({"scattering_factor", "projected_scattering_factor"})

    @property
    def screening(self) -> float:
        """Screening wavevector κ [1/Å] for the ionic Coulomb correction ΔZ·M/(κ²+q²)."""
        return self._screening

    @screening.setter
    def screening(self, value: float):
        self._screening = value

    def get_function(
        self,
        name: str,
        symbol: str,
        charge: float = 0.0,
        screening: float | None = None,
    ) -> Callable:
        """
        Returns a callable for a parameterized function for one element.

        Parameters
        ----------
        name : str
            Name of the function to return.
        symbol : str
            Chemical symbol of element.
        charge : float
            Charge of element. Given as elementary charges.
        screening : float or None
            Screening wavevector κ [1/Å] for the ionic Coulomb correction
            ΔZ·M/(κ²+q²). Overrides the instance-level screening when provided;
            falls back to self.screening otherwise.
        """
        if screening is None:
            screening = self._screening

        if charge == 0.0 or name not in self._ionic_k2_names:
            return super().get_function(name, symbol, charge)

        # Use super() for validation (raises RuntimeError if symbol+charge not in JSON).
        super().get_function(name, symbol, charge)

        if isinstance(symbol, (int, np.int32, np.int64)):
            symbol = chemical_symbols[symbol]
        charge_symbol = "+" * int(abs(charge)) if charge > 0.0 else "-" * int(abs(charge))
        parameters = np.array(
            self.scaled_parameters(symbol + charge_symbol, name),
            dtype=get_dtype(complex=False),
        )
        # ΔZ / s² correction from Peng 1999 eq. (3); abTEM uses k = 2s, so 1/s² = 4/k².
        # projected_scattering_factor parameters have a_i / kappa, so the Coulomb
        # coefficient must match; scattering_factor keeps raw a_i, so no kappa here.
        base_coeff = charge * 4.0 * self._mott_constant
        coulomb_coeff = base_coeff / kappa if name == "projected_scattering_factor" else base_coeff
        return lambda k2: peng.ionic_scattering_factor_k2(k2, parameters, coulomb_coeff, screening)

    def scattering_factor(
        self, symbol: str, charge: float = 0.0, screening: float | None = None
    ) -> Callable:
        return self.get_function("scattering_factor", symbol, charge, screening)

    def projected_scattering_factor(
        self, symbol: str, charge: float = 0.0, screening: float | None = None
    ) -> Callable:
        return self.get_function("projected_scattering_factor", symbol, charge, screening)

    def scaled_parameters(self, symbol: str, name: str) -> np.ndarray:
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

        scaled_parameters = {
            "potential": potential,
            "scattering_factor": scattering_factor,
            "finite_projected_potential": finite_projected_potential,
            "finite_projected_scattering_factor": finite_projected_scattering_factor,
            "projected_potential": projected_potential,
            "projected_scattering_factor": projected_scattering_factor,
        }

        return scaled_parameters[name]


class EwaldParametrization(Parametrization):
    _functions = {"potential": ewald.potential}

    def __init__(self, width: float = 1.0):
        parameters = {
            symbol: [width, Z] for Z, symbol in enumerate(chemical_symbols[1:], 1)
        }
        super().__init__(parameters=parameters)

    @property
    def width(self):
        return self.parameters["H"][0]

    def scaled_parameters(self, symbol: str, name: str) -> np.ndarray:
        return {"potential": self.parameters[symbol]}[name]


def validate_parametrization(parametrization: str | Parametrization) -> Parametrization:
    named_parametrizations = {
        "ewald": EwaldParametrization,
        "lobato": LobatoParametrization,
        "peng": PengParametrization,
        "kirkland": KirklandParametrization,
    }

    if isinstance(parametrization, str):
        parametrization = named_parametrizations[parametrization]()

    return parametrization


def validate_parameters(parameters: str | dict) -> dict:
    if isinstance(parameters, str):
        if os.path.isabs(parameters):
            path = parameters
        else:
            path = os.path.join(_get_data_path(), parameters)

        with open(path, "r") as f:
            parameters = json.load(f)

    elif not isinstance(parameters, dict):
        raise ValueError()

    return parameters
