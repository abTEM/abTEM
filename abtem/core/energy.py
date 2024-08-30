"""Module for handling electron energy."""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import units

from abtem.core.utils import CopyMixin, EqualityMixin


def relativistic_mass_correction(energy: float) -> float:
    """
    Calculate relativistic mass correction from energy.

    Parameters
    ----------
    energy: float
        Electron energy [eV].

    Returns
    -------
    float
        Relativistic mass correction.
    """
    return 1 + units._e * energy / (units._me * units._c**2)


def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    enerElectron energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]̄
    """
    return relativistic_mass_correction(energy) * units._me


def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """

    return (
        units._hplanck
        * units._c
        / np.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [ev].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """
    return (
        2
        * np.pi
        * energy2mass(energy)
        * units.kg
        * units._e
        * units.C
        * energy2wavelength(energy)
        / (units._hplanck * units.s * units.J) ** 2
    )


def reciprocal_space_sampling_to_angular_sampling(
    reciprocal_space_sampling: tuple[float, float], energy: float
) -> tuple[float, float]:
    """
    Convert reciprocal space sampling in 1/Å to angular sampling in mrad.

    Parameters
    ----------
    reciprocal_space_sampling: tuple of floats
        Reciprocal space sampling [1/Å].
    energy: float
        Electron energy in [eV].

    Returns
    -------
    tuple of floats
        Angular sampling [mrad].
    """
    wavelength = energy2wavelength(energy)
    return (
        reciprocal_space_sampling[0] * wavelength * 1e3,
        reciprocal_space_sampling[1] * wavelength * 1e3,
    )


class EnergyUndefinedError(Exception):
    """
    Error raised when energy is not defined.
    """


class Accelerator(EqualityMixin, CopyMixin):
    """
    Accelerator object describes the energy of wave functions and transfer functions.

    Parameters
    ----------
    energy: float
        Acceleration energy [eV].
    """

    def __init__(self, energy: Optional[float] = None, lock_energy: bool = False):
        if energy is not None:
            energy = float(energy)

        self._energy = energy
        self._lock_energy = lock_energy

    @property
    def energy(self) -> float | None:
        """
        Electron acceleration energy [eV].
        """
        return self._energy

    @energy.setter
    def energy(self, value: float):
        if self._lock_energy:
            raise RuntimeError("Energy cannot be modified")

        if value is not None:
            value = float(value)
        self._energy = value

    @property
    def wavelength(self) -> float:
        """
        Relativistic wavelength [Å].
        """
        self.check_is_defined()
        assert self.energy is not None
        return energy2wavelength(self.energy)

    @property
    def sigma(self) -> float:
        """Interaction parameter."""
        self.check_is_defined()
        assert self.energy is not None
        return energy2sigma(self.energy)

    def check_is_defined(self):
        """
        Raise error if the energy is not defined.
        """
        if self.energy is None:
            raise EnergyUndefinedError("Energy is not defined")

    def check_match(self, other: Accelerator | HasAcceleratorMixin):
        """
        Raise error if the accelerator of another object is different from this object.

        Parameters
        ----------
        other: Accelerator object
            The accelerator that should be checked.
        """
        if (
            (self.energy is not None)
            & (other.energy is not None)
            & (self.energy != other.energy)
        ):
            raise RuntimeError("Inconsistent energies")

    def match(
        self, other: Accelerator | HasAcceleratorMixin, check_match: bool = False
    ):
        """
        Set the parameters of this accelerator to match another accelerator.

        Parameters
        ----------
        other: Accelerator object
            The accelerator that should be matched.
        check_match: bool
            If true check whether accelerators can match without overriding an already defined energy.
        """

        if check_match:
            self.check_match(other)

        if other.energy is None:
            other.energy = self.energy

        elif self.energy != other.energy:
            self.energy = other.energy


class HasAcceleratorMixin:
    """
    Mixin class for objects that have an electron energy.
    """

    _accelerator: Accelerator

    @property
    def accelerator(self) -> Accelerator:
        """
        Accelerator object describing the acceleration energy.
        """
        return self._accelerator

    @accelerator.setter
    def accelerator(self, new: Accelerator):
        self._accelerator = new

    @property
    def energy(self) -> float | None:
        """
        Electron acceleration energy in electron volts.
        """
        return self.accelerator.energy

    @energy.setter
    def energy(self, energy: float):
        self.accelerator.energy = energy

    @property
    def _valid_energy(self) -> float:
        """
        Electron acceleration energy in electron volts.

        Raises
        ------
        EnergyUndefinedError
            If the energy is not defined.
        """
        if self.energy is None:
            raise EnergyUndefinedError("Energy is not defined")
        return self.energy

    @property
    def wavelength(self) -> float:
        """
        Relativistic wavelength in Ångstrom.
        """
        return self.accelerator.wavelength
