from abtem.core import config

units_type = {
    **dict(zip(("Å", "Angstrom", "nm", "um", "mm", "m"), ("real_space",) * 6)),
    **dict(
        zip(
            ("1/Å", "1/Angstrom", "1/nm", "1/um", "1/mm", "1/m"),
            ("reciprocal_space",) * 6,
        )
    ),
    **dict(zip(("mrad",), ("angular",) * 1)),
    **dict(
        zip(
            (
                "eV",
                "keV",
            ),
            ("energy",) * 2,
        )
    ),
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
}
_tex_units = {
    "Å": "\AA",
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
}


def _format_units(units):
    if config.get("visualize.use_tex", False):
        return _tex_units[units]
    else:
        return units


def _validate_units(units, old_units):

    if old_units is None and units is None:
        raise RuntimeError()
    elif units is None:
        units = old_units
    elif units is not None and old_units is not None:
        if units_type[units] != units_type[old_units]:
            raise RuntimeError(f"cannot convert units {old_units} to {units}")

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


def _get_conversion_factor(units, old_units):
    return _conversion_factors[_validate_units(units, old_units)]
