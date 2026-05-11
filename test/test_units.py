"""Tests for abtem/core/units.py"""

import pytest

from abtem.core.units import (
    format_units,
    get_conversion_factor,
    validate_units,
)


class TestFormatUnits:
    def test_none_returns_empty_string(self):
        assert format_units(None) == ""

    def test_plain_angstrom(self):
        assert format_units("Å", use_tex=False) == "Å"

    def test_plain_nm(self):
        assert format_units("nm", use_tex=False) == "nm"

    def test_plain_mrad(self):
        assert format_units("mrad", use_tex=False) == "mrad"

    def test_tex_known_unit(self):
        result = format_units("Å", use_tex=True)
        assert result.startswith("$") and result.endswith("$")
        assert r"\AA" in result

    def test_tex_reciprocal_unit(self):
        result = format_units("1/Å", use_tex=True)
        assert result.startswith("$") and result.endswith("$")
        assert "AA" in result or "mathrm" in result

    def test_tex_percent(self):
        result = format_units("%", use_tex=True)
        assert r"\mathrm{\%}" in result

    def test_tex_unknown_unit_wrapped(self):
        result = format_units("arb.u.", use_tex=True)
        assert r"\mathrm{" in result

    def test_plain_unrecognised_passthrough(self):
        assert format_units("arb.u.", use_tex=False) == "arb.u."


class TestValidateUnits:
    def test_both_none(self):
        assert validate_units(None, None) is None

    def test_units_none_returns_old(self):
        assert validate_units(None, "Å") == "Å"

    def test_old_none_returns_units(self):
        assert validate_units("nm") == "nm"

    def test_same_category_ok(self):
        assert validate_units("nm", "Å") == "nm"

    def test_cross_category_raises(self):
        with pytest.raises(RuntimeError, match="cannot convert"):
            validate_units("mrad", "Å")

    def test_angstrom_alias_real(self):
        assert validate_units("Angstrom") == "Å"

    def test_angstrom_alias_reciprocal(self):
        # "1/Angstrom" is not in the mapping, so it passes through unchanged
        result = validate_units("1/Angstrom")
        # Not in units_type dict — returned as-is
        assert result == "1/Angstrom"

    def test_reciprocal_space_unit(self):
        assert validate_units("1/nm") == "1/nm"

    def test_angular_unit(self):
        assert validate_units("deg") == "deg"

    def test_energy_unit(self):
        assert validate_units("keV") == "keV"


class TestGetConversionFactor:
    def test_units_none_returns_one(self):
        assert get_conversion_factor(None) == 1.0

    def test_no_old_units_raises(self):
        with pytest.raises(RuntimeError, match="old_units must be provided"):
            get_conversion_factor("nm")

    def test_angstrom_to_nm(self):
        factor = get_conversion_factor("nm", "Å")
        assert abs(factor - 1e-1) < 1e-15

    def test_angstrom_to_m(self):
        factor = get_conversion_factor("m", "Å")
        assert abs(factor - 1e-10) < 1e-20

    def test_reciprocal_to_reciprocal(self):
        factor = get_conversion_factor("1/nm", "1/Å")
        assert abs(factor - 10) < 1e-10

    def test_mrad_to_rad(self):
        factor = get_conversion_factor("rad", "mrad")
        assert abs(factor - 1e3) < 1e-10

    def test_reciprocal_to_angular_requires_energy(self):
        with pytest.raises(RuntimeError, match="energy must be provided"):
            get_conversion_factor("mrad", "1/Å")

    def test_reciprocal_to_angular_with_energy(self):
        factor = get_conversion_factor("mrad", "1/Å", energy=100e3)
        assert factor > 0

    def test_reciprocal_to_angular_deg_with_energy(self):
        factor_mrad = get_conversion_factor("mrad", "1/Å", energy=100e3)
        factor_deg = get_conversion_factor("deg", "1/Å", energy=100e3)
        # deg factor should be larger than mrad factor
        assert factor_deg > factor_mrad
