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
    "Å": "\mathrm{\AA}",
    "nm": "\mathrm{nm}",
    "um": "\mathrm{\mu m}",
    "mm": "\mathrm{mm}",
    "m": "\mathrm{mm}",
    "1/Å": "\AA^{-1}",
    "1/nm": "\mathrm{nm}^{-1}",
    "1/um": "\mathrm{\mu m}^{-1}",
    "1/mm": "\mathrm{mm}^{-1}",
    "1/m": "\mathrm{m}^{-1}",
    "mrad": "\mathrm{mrad}",
    "deg": "\mathrm{deg}",
    "e/Å^2": "\mathrm{e}^-/\mathrm{\AA}^2",
}


def _format_units(units):
    if config.get("visualize.use_tex", False):
        units = _tex_units.get(units, f"\mathrm{{{units}}}")
        return f"${units}$"
    else:
        return units


def _validate_units(units, old_units):
    if old_units is None and units is None:
        return None
    elif units is None:
        units = old_units
    elif units is not None and old_units is not None:
        if units_type[units] != units_type[old_units]:
            raise RuntimeError(f"cannot convert units {old_units} to {units}")

    if not units in units_type:
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
    if units_type[old_units] == "reciprocal_space" and units_type[units] == "angular":
        if energy is None:
            raise RuntimeError("")

        wavelength = energy2wavelength(energy)
        conversion = (
            wavelength * 1e3 * _conversion_factors[_validate_units(units, "mrad")]
        )
        return conversion

    return _conversion_factors[_validate_units(units, old_units)]
