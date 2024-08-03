import numpy as np

from abtem.core import config
from abtem.core.energy import energy2wavelength

categories = {
    "real_space": ["Å", "Angstrom", "nm", "um", "mm", "m"],
    "reciprocal_space": ["1/Å", "1/Angstrom", "1/nm", "1/um", "1/mm", "1/m"],
    "angular": ["rad", "mrad", "deg"],
    "energy": ["eV", "keV"],
}

# Use a dictionary comprehension to create the final mapping
units_type = {
    unit: category for category, units in categories.items() for unit in units
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


def _format_units(units):
    if config.get("visualize.use_tex", False) is True:
        try:
            units = _tex_units[units]
        except KeyError:
            if units == "%":
                units = r"\mathrm{\%}"
                # TODO: temporary fix for the percent sign
            else:
                units = r"\mathrm{" + f"{units}" + r"}"

        return f"${units}$"
    else:
        return units


def _validate_units(units, old_units=None):
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
        if units is None:
            units = config.get("visualize.real_space_units", "Angstrom")

        if units == "Angstrom":
            units = "Å"

        return units
    elif units_type[units] == "reciprocal_space":
        if units is None:
            units = config.get("visualize.reciprocal_space_units", "Angstrom")

        if units == "Angstrom":
            units = "1/Å"

        return units
    elif units_type[units] == "angular":
        return units
    else:
        raise NotImplementedError


def _get_conversion_factor(units: str, old_units: str, energy: float = None):
    if units is None:
        return 1.0

    if units_type[old_units] == "reciprocal_space" and units_type[units] == "angular":
        if energy is None:
            raise RuntimeError("")

        wavelength = energy2wavelength(energy)
        conversion = (
                wavelength * 1e3 * _conversion_factors[_validate_units(units, "mrad")]
        )
        return conversion

    return _conversion_factors[_validate_units(units, old_units)]
