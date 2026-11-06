"""Module for handling electron energy."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, TypeVar, cast

import numpy as np
from ase import units  # type: ignore

from abtem.core import config
from abtem.core.backend import get_array_module
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

    if energy <= 0:
        raise ValueError(f"Electron energy must be positive, got {energy} eV")

    return float(
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
    return float(
        2
        * np.pi
        * energy2mass(energy)
        * units.kg
        * units._e
        * units.C
        * energy2wavelength(energy)
        / (units._hplanck * units.s * units.J) ** 2
    )


T = TypeVar("T", bound=tuple[float, ...])


def reciprocal_space_sampling_to_angular_sampling(
    reciprocal_space_sampling: T, energy: float
) -> T:
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
    angular_sampling = tuple(d * 1e3 * wavelength for d in reciprocal_space_sampling)
    angular_sampling = cast(T, angular_sampling)
    return angular_sampling


def angle_convention() -> str:
    """The configured relation between angles and spatial frequencies,
    ``"paraxial"`` or ``"exact"`` (the ``angles`` key of the configuration)."""
    convention = config.get("angles")
    if convention not in ("paraxial", "exact"):
        raise ValueError(
            f"config 'angles' must be 'paraxial' or 'exact', not {convention!r}"
        )
    return convention


_LARGE_ANGLE = 250.0


def warn_if_paraxial_angle_is_large(*angles) -> None:
    """
    Warn about a large angle that you give, which is a ``wavelength * k`` with the
    paraxial convention. Values that are not numbers (None, strings) and infinite
    ones are ignored.
    """
    if angle_convention() != "paraxial":
        return

    values = []
    for angle in angles:
        if isinstance(angle, (int, float, np.number, np.ndarray, tuple, list)):
            values.extend(np.abs(np.asarray(angle, dtype=float)).ravel())

    values = np.array(values)
    values = values[np.isfinite(values)]
    if values.size and values.max() > _LARGE_ANGLE:
        warnings.warn(
            f"An angle above {_LARGE_ANGLE:g} mrad is taken as wavelength * k, which "
            "differs from the scattering angle by more than 1 % there. Set the "
            "'angles' configuration to 'exact' to give scattering angles instead.",
            UserWarning,
            stacklevel=3,
        )


def to_grid_angle(angle, convention: Optional[str] = None):
    """
    The coordinate on the angular grid of a given angle.

    The angular grid is uniform in ``1e3 * wavelength * k`` [mrad]. With the
    ``paraxial`` convention an angle is that coordinate. With the ``exact``
    convention an angle is a scattering angle, and the coordinate is
    ``1e3 * sin(angle / 1e3)``, and angles beyond 90 degrees (or infinite) give the
    coordinate of 90 degrees.

    Parameters
    ----------
    angle : float or array
        Angle [mrad] in the configured convention.
    convention : {"paraxial", "exact"}, optional
        The convention, if not the configured one.

    Returns
    -------
    float or array
        Coordinate on the angular grid [mrad].
    """
    if (convention or angle_convention()) == "paraxial":
        return angle
    xp = get_array_module(angle)
    # beyond 90 degrees every propagating beam is within the angle
    return 1e3 * xp.sin(xp.minimum(angle / 1e3, np.pi / 2))


def to_scattering_angle(grid_angle, convention: Optional[str] = None):
    """
    The angle in the configured convention of a coordinate on the angular grid, the
    inverse of :func:`to_grid_angle`.

    With the ``exact`` convention the result is NaN beyond
    ``1e3 * wavelength * k = 1e3``, where the beam is evanescent and has no scattering
    angle.

    Parameters
    ----------
    grid_angle : float or array
        Coordinate on the angular grid [mrad].
    convention : {"paraxial", "exact"}, optional
        The convention, if not the configured one.

    Returns
    -------
    float or array
        Angle [mrad] in the configured convention.
    """
    if (convention or angle_convention()) == "paraxial":
        return grid_angle
    xp = get_array_module(grid_angle)
    ratio = grid_angle / 1e3
    with np.errstate(invalid="ignore"):
        return 1e3 * xp.arcsin(xp.where(xp.abs(ratio) <= 1.0, ratio, xp.nan))


def require_paraxial_angles(feature: str) -> None:
    """Raise if the ``angles`` configuration is ``"exact"``, for a feature whose
    angles are not converted yet."""
    if angle_convention() != "paraxial":
        raise NotImplementedError(
            f"{feature} does not support the 'exact' angle convention yet; set the "
            "'angles' configuration to 'paraxial'."
        )


def to_scattering_angle_limit(grid_angle, convention: Optional[str] = None):
    """
    The limit of the angular grid, as an angle in the configured convention.

    The same as :func:`to_scattering_angle`, but beyond the largest scattering angle
    (``1e3 * wavelength * k > 1e3``) the result is 90 degrees, so that a limit which
    reaches into the evanescent region stays a limit that holds every propagating
    beam.

    Parameters
    ----------
    grid_angle : float
        Coordinate on the angular grid [mrad].
    convention : {"paraxial", "exact"}, optional
        The convention, if not the configured one.

    Returns
    -------
    float
        Angle [mrad] in the configured convention.
    """
    if (convention or angle_convention()) == "paraxial":
        return grid_angle
    return float(1e3 * np.arcsin(min(grid_angle / 1e3, 1.0)))


class EnergyUndefinedError(Exception):
    """
    Error raised when energy is not defined.
    """


def resolve_energy(
    energy: Optional[float],
    metadata: Optional[dict],
    ensemble_axes_metadata: Sequence,
) -> Optional[float]:
    """Resolve a scalar energy [eV] shared by ``Waves``, ``_energy_from_waves``
    and ``BaseMeasurements._get_energy``.

    Resolution order:
    1. ``energy`` if not None (e.g. ``accelerator.energy`` for a ``Waves``-like
       object).
    2. ``metadata["energy"]`` -- populated by ``EnergyAxis.item_metadata`` when
       the object was produced by indexing an energy-ensemble.
    3. A single-element ``EnergyAxis`` in ``ensemble_axes_metadata`` -- the
       residual case during per-member partitioned computation.

    An array object refuses a scalar energy that contradicts its
    ``EnergyAxis`` at construction, so the stores never disagree.

    Returns
    -------
    float or None
        None if no scalar energy can be resolved this way -- notably for a
        full, un-indexed multi-energy ensemble. Callers decide what to do
        about that case (raise, return None, or fall back to an approximate
        value).
    """
    from abtem.core.axes import EnergyAxis

    if energy is not None:
        return float(energy)
    energy = metadata.get("energy") if metadata else None
    if energy is not None:
        return float(energy)
    for axis in ensemble_axes_metadata:
        if isinstance(axis, EnergyAxis) and len(axis.values) == 1:
            return float(axis.values[0])
    return None


def _resolve_other_energy(other: "Accelerator | HasAcceleratorMixin") -> Optional[float]:
    """Best-effort scalar energy for ``other``, seeing through an indexed
    energy-ensemble member.

    ``other.accelerator.energy`` (or ``other.energy`` for a bare
    ``Accelerator``) is ``None`` both when no energy has ever been set *and*
    when ``other`` is a single, already-indexed member of an energy ensemble
    -- its real energy then lives in ``other.metadata["energy"]`` or a
    single-valued ``EnergyAxis`` in ``other.ensemble_axes_metadata`` instead
    (the same resolution :func:`resolve_energy` already performs for
    ``Waves._valid_energy``). Reading only the raw accelerator energy
    conflates the two cases: ``Accelerator.match``/``check_match`` would then
    treat a perfectly well-defined ensemble member as "no energy defined"
    and let their own energy silently override it (``match``), or never
    notice a genuine mismatch against it (``check_match``).

    ``other`` need not be an ``ArrayObject`` -- ``metadata`` and
    ``ensemble_axes_metadata`` are read with ``getattr`` and default to
    "absent", so a plain ``Accelerator`` or any other
    ``HasAcceleratorMixin`` without those attributes falls back to exactly
    the previous, raw-energy behaviour.
    """
    energy = (
        other.energy if isinstance(other, Accelerator) else other.accelerator.energy
    )
    if energy is not None:
        return float(energy)
    metadata = getattr(other, "metadata", None)
    ensemble_axes_metadata = getattr(other, "ensemble_axes_metadata", None) or ()
    return resolve_energy(None, metadata, ensemble_axes_metadata)


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
        other_energy = _resolve_other_energy(other)
        if (
            (self.energy is not None)
            and (other_energy is not None)
            and (self.energy != other_energy)
        ):
            raise RuntimeError(
                f"Inconsistent energies: {self.energy} eV vs {other_energy} eV. "
                "A fixed energy (e.g. a built TransitionPotentialArray, or a CTF "
                "with an explicit energy) cannot be applied to wave functions at "
                "a different energy -- this also covers one member of a "
                "multi-energy ensemble, whose own energy is resolved from its "
                "metadata/ensemble axes even though its accelerator energy reads "
                "None. Build a separate object per energy, or leave the energy "
                "unset so it is taken from the wave functions."
            )

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
            If true check whether accelerators can match without overriding an already
            defined energy.
        """

        if check_match:
            self.check_match(other)

        other_energy = _resolve_other_energy(other)

        if other_energy is None:
            if isinstance(other, Accelerator):
                other.energy = self.energy
            else:
                other.accelerator.energy = self.energy

        elif self.energy != other_energy:
            self.energy = other_energy


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
