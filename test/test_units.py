"""Tests for abtem/core/units.py"""

import numpy as np
import pytest
import scipy.constants as const

from abtem.core.axes import ReciprocalSpaceAxis
from abtem.core.units import (
    format_units,
    get_conversion_factor,
    validate_units,
)


def _relativistic_wavelength(energy):
    # Independent oracle: lambda = h c / sqrt(E (E + 2 m_e c^2)), E = e * V, in Å.
    E = energy * const.e
    return const.h * const.c / np.sqrt(E * (E + 2 * const.m_e * const.c**2)) * 1e10


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
        # 1/Å is Å to the power -1
        assert result == r"$\mathrm{\AA}^{-1}$"

    def test_tex_metre(self):
        # SI symbol for metre is "m", not "mm"
        assert format_units("m", use_tex=True) == r"$\mathrm{m}$"

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
        # "1/Angstrom" is listed as a reciprocal-space unit and is an alias of 1/Å
        assert validate_units("1/Angstrom") == "1/Å"

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
        # 1 mrad = 1e-3 rad
        factor = get_conversion_factor("rad", "mrad")
        assert factor == pytest.approx(1e-3, rel=1e-12)

    def test_mrad_to_deg(self):
        # 1 mrad = 1e-3 rad = 1e-3 * 180 / pi deg
        factor = get_conversion_factor("deg", "mrad")
        assert factor == pytest.approx(1e-3 * 180 / np.pi, rel=1e-12)

    def test_rad_to_deg(self):
        # 1 rad = 180 / pi deg
        factor = get_conversion_factor("deg", "rad")
        assert factor == pytest.approx(180 / np.pi, rel=1e-12)

    def test_nm_to_angstrom(self):
        # 1 nm = 10 Å
        factor = get_conversion_factor("Å", "nm")
        assert factor == pytest.approx(10, rel=1e-12)

    def test_angular_round_trip(self):
        # mrad -> rad -> deg -> mrad composes to the identity
        factor = (
            get_conversion_factor("rad", "mrad")
            * get_conversion_factor("deg", "rad")
            * get_conversion_factor("mrad", "deg")
        )
        assert factor == pytest.approx(1.0, rel=1e-12)

    def test_reciprocal_angstrom_alias(self):
        # "1/Angstrom" and "1/Å" are the same unit
        assert get_conversion_factor("1/Å", "1/Angstrom") == pytest.approx(1.0)
        assert get_conversion_factor("1/Angstrom", "1/Å") == pytest.approx(1.0)

    def test_reciprocal_to_angular_requires_energy(self):
        with pytest.raises(RuntimeError, match="energy must be provided"):
            get_conversion_factor("mrad", "1/Å")

    @pytest.mark.parametrize(
        "units, expected_per_wavelength",
        [
            # abTEM uses the linear (small-angle) relation alpha = lambda * k [rad]
            ("rad", 1.0),
            ("mrad", 1e3),
            ("deg", 180 / np.pi),
        ],
    )
    @pytest.mark.parametrize("energy", [80e3, 100e3, 300e3])
    def test_reciprocal_to_angular_with_energy(
        self, units, expected_per_wavelength, energy
    ):
        factor = get_conversion_factor(units, "1/Å", energy=energy)
        expected = _relativistic_wavelength(energy) * expected_per_wavelength
        # CODATA constants in scipy vs. ase differ at the ~5e-9 level
        assert factor == pytest.approx(expected, rel=1e-6)

    def test_reciprocal_nm_to_mrad(self):
        # k [1/Å] = k [1/nm] / 10, so alpha [mrad] = 1e3 * lambda * k [1/nm] / 10
        factor = get_conversion_factor("mrad", "1/nm", energy=100e3)
        expected = 1e3 * _relativistic_wavelength(100e3) / 10
        assert factor == pytest.approx(expected, rel=1e-6)

    def test_reciprocal_axis_convert_units_to_rad(self):
        # sampling 0.1 1/Å at 100 keV corresponds to lambda * 0.1 rad
        axis = ReciprocalSpaceAxis(label="k", sampling=0.1, units="1/Å")
        converted = axis.convert_units("rad", energy=100e3)
        assert converted.units == "rad"
        assert converted.sampling == pytest.approx(
            0.1 * _relativistic_wavelength(100e3), rel=1e-6
        )
