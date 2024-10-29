"""Module for handling units and unit conversion."""

from typing import Optional

import numpy as np

from abtem.core import config
from abtem.core.energy import energy2wavelength

_unit_categories = {
    "real_space": ("Å", "Angstrom", "nm", "um", "mm", "m"),
    "reciprocal_space": ("1/Å", "1/Angstrom", "1/nm", "1/um", "1/mm", "1/m"),
    "angular": ["rad", "mrad", "deg"],
    "energy": ["eV", "keV"],
}

# A mapping from unit to unit category
units_type = {
    unit: category for category, units in _unit_categories.items() for unit in units
}

_conversion_factors = {
    "Å": 1,
    "nm": 1e-1,
    "um": 1e-4,
    "mm": 1e-7,
    "m": 1e-10,
    "1/Å": 1,
    "1/nm": 10,
    "1/um": 1e4,
    "1/mm": 1e7,
    "1/m": 1e10,
    "mrad": 1,
    "rad": 1e3,
    "deg": 1e3 / np.pi * 180.0,
}

_tex_units = {
    "Å": r"\mathrm{\AA}",
    "nm": r"\mathrm{nm}",
    "um": r"\mathrm{\mu m}",
    "mm": r"\mathrm{mm}",
    "m": r"\mathrm{mm}",
    "1/Å": r"\mathrm{\AA}^{-1}",
    "1/nm": r"\mathrm{nm}^{-1}",
    "1/um": r"\mathrm{\mu m}^{-1}",
    "1/mm": r"\mathrm{mm}^{-1}",
    "1/m": r"\mathrm{m}^{-1}",
    "mrad": r"\mathrm{mrad}",
    "deg": r"\mathrm{deg}",
    "e/Å^2": r"\mathrm{e}^-/\mathrm{\AA}^2",
}


def format_units(units: Optional[str], use_tex: Optional[bool] = None) -> str:
    """
    Format units as a string.

    Parameters
    ----------
    units : str
        The units to format.

    Returns
    -------
    str
        The formatted units.
    """
    if units is None:
        return ""

    use_tex = config.get("visualize.use_tex", False) if use_tex is None else use_tex

    if use_tex:
        try:
            units = _tex_units[units]
        except KeyError:
            if units == "%":
                units = r"\mathrm{\%}"
            else:
                units = r"\mathrm{" + f"{units}" + r"}"

        return f"${units}$"
    else:
        return units


def validate_units(
    units: Optional[str] = None, old_units: Optional[str] = None
) -> Optional[str]:
    """
    Validate units and convert to a standard format.

    If `old_units` is provided, the function will check if the conversion is
    possible and raise an error if not.

    Parameters
    ----------
    units : str
        The units to validate.
    old_units : str, optional
        The optional units to check whether conversion from is possible.

    Returns
    -------
    str
        The validated units

    Raises
    ------
    ValueError
        If the units are invalid or if conversion from `old_units` is not possible.
    """

    if old_units is None and units is None:
        return None
    elif units is None:
        units = old_units
    elif units is not None and old_units is not None:
        if units_type[units] != units_type[old_units]:
            raise RuntimeError(f"cannot convert units {old_units} to {units}")

    if units not in units_type:
        return units

    if units_type[units] == "real_space":
        if units == "Angstrom":
            units = "Å"

        return units
    elif units_type[units] == "reciprocal_space":
        if units == "Angstrom":
            units = "1/Å"

        return units
    elif units_type[units] == "angular":
        return units
    else:
        raise ValueError(f"Invalid units: {units}")


def get_conversion_factor(
    units: Optional[str] = None,
    old_units: Optional[str] = None,
    energy: Optional[float] = None,
) -> float:
    """
    Get the conversion factor between two units.

    Parameters
    ----------
    units : str, optional
        The units to convert to.
    old_units : str, optional
        The units to convert from.
    energy : float, optional
        The energy to use for conversion from reciprocal space to angular units [eV].

    Returns
    -------
    float
        The conversion factor.
    """
    if units is None:
        return 1.0

    if old_units is None and units is not None:
        raise RuntimeError("old_units must be provided if units is provided")

    if units_type[old_units] == "reciprocal_space" and units_type[units] == "angular":
        if energy is None:
            raise RuntimeError(
                "energy must be provided to convert from reciprocal space to angular"
                " units"
            )

        wavelength = energy2wavelength(energy)

        units = validate_units(units, "mrad")
        assert units is not None

        conversion = wavelength * 1e3 * _conversion_factors[units]
        return conversion

    validated_units = validate_units(units, old_units)
    assert validated_units is not None

    return _conversion_factors[validated_units]
