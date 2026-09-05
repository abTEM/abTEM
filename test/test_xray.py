"""Tests for the X-ray emission data adapter.

Pure lookup and arithmetic over the tabulation: no multislice, no device, no
GPAW.
"""

from __future__ import annotations

import numpy as np
import pytest

from abtem.inelastic.xray_data import (
    absorption_edge,
    emission_lines,
    fluorescence_yield,
    line_families,
    natural_width,
    statistical_weights,
    subshell_levels,
    vacancy_distribution,
)

xraydb = pytest.importorskip("xraydb", reason="EDX data requires xraydb")


# Fluorescence yields from the Elam/Krause tabulation, i.e. the values the
# adapter must reproduce exactly for a K edge.
K_SHELL_YIELDS = {
    "C": 0.0014,
    "O": 0.0058,
    "Si": 0.0429108,
    "Ti": 0.218425,
    "Fe": 0.350985,
    "Cu": 0.441091,
    "Ag": 0.821892,
    "Au": 0.980803,
}


class TestSubshellMapping:
    @pytest.mark.parametrize(
        "n, l, expected",
        [
            (1, 0, ("K",)),
            (2, 0, ("L1",)),
            (2, 1, ("L2", "L3")),
            (3, 0, ("M1",)),
            (3, 1, ("M2", "M3")),
            (3, 2, ("M4", "M5")),
            (4, 3, ("N6", "N7")),
        ],
    )
    def test_iupac_level_names(self, n, l, expected):
        assert subshell_levels(n, l) == expected

    @pytest.mark.parametrize("n, l", [(1, 1), (2, 2), (0, 0), (3, 3)])
    def test_invalid_quantum_numbers_raise(self, n, l):
        with pytest.raises(ValueError):
            subshell_levels(n, l)

    @pytest.mark.parametrize(
        "l, expected",
        [(0, (1.0,)), (1, (1 / 3, 2 / 3)), (2, (0.4, 0.6)), (3, (3 / 7, 4 / 7))],
    )
    def test_statistical_weights_are_2j_plus_1(self, l, expected):
        assert statistical_weights(l) == pytest.approx(expected)

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_statistical_weights_sum_to_one(self, l):
        assert sum(statistical_weights(l)) == pytest.approx(1.0)


class TestVacancyDistribution:
    def test_k_shell_is_a_single_level(self):
        assert vacancy_distribution("Cu", 1, 0) == {"K": 1.0}

    def test_without_coster_kronig_is_statistical(self):
        distribution = vacancy_distribution("Fe", 2, 1, coster_kronig=False)
        assert distribution == pytest.approx({"L2": 1 / 3, "L3": 2 / 3})

    def test_coster_kronig_moves_vacancies_to_the_higher_level(self):
        without = vacancy_distribution("Fe", 2, 1, coster_kronig=False)
        with_ck = vacancy_distribution("Fe", 2, 1, coster_kronig=True)

        assert with_ck["L2"] < without["L2"]
        assert with_ck["L3"] > without["L3"]

        # The transferred fraction is the tabulated L2 -> L3 probability.
        transferred = without["L2"] - with_ck["L2"]
        expected = without["L2"] * xraydb.ck_probability("Fe", "L2", "L3")
        assert transferred == pytest.approx(expected)

    @pytest.mark.parametrize("element", ["Fe", "Ag", "Au"])
    @pytest.mark.parametrize("coster_kronig", [True, False])
    def test_vacancies_are_conserved(self, element, coster_kronig):
        distribution = vacancy_distribution(element, 2, 1, coster_kronig)
        assert sum(distribution.values()) == pytest.approx(1.0)


class TestFluorescenceYield:
    @pytest.mark.parametrize("element, expected", K_SHELL_YIELDS.items())
    def test_k_shell_matches_tabulated_values(self, element, expected):
        assert fluorescence_yield(element, 1, 0) == pytest.approx(expected, rel=1e-5)

    def test_accepts_atomic_number_and_symbol(self):
        assert fluorescence_yield(29, 1, 0) == fluorescence_yield("Cu", 1, 0)

    def test_yield_increases_with_atomic_number(self):
        elements = ["C", "Si", "Ti", "Cu", "Ag", "Au"]
        yields = [fluorescence_yield(el, 1, 0) for el in elements]
        assert np.all(np.diff(yields) > 0)

    def test_l_shell_is_between_the_two_level_yields(self):
        edges = xraydb.xray_edges("Ag")
        combined = fluorescence_yield("Ag", 2, 1)
        assert (
            min(edges["L2"].fyield, edges["L3"].fyield)
            <= combined
            <= max(edges["L2"].fyield, edges["L3"].fyield)
        )

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError):
            fluorescence_yield("Xx", 1, 0)


class TestEmissionLines:
    @pytest.mark.parametrize("element", ["Si", "Ti", "Fe", "Cu", "Ag"])
    def test_line_intensities_sum_to_the_fluorescence_yield(self, element):
        lines = emission_lines(element, 1, 0)
        total = sum(line.intensity for line in lines.values())
        assert total == pytest.approx(fluorescence_yield(element, 1, 0), rel=1e-10)

    @pytest.mark.parametrize("element", ["Fe", "Ag", "Au"])
    def test_l_shell_intensities_sum_to_the_fluorescence_yield(self, element):
        lines = emission_lines(element, 2, 1)
        total = sum(line.intensity for line in lines.values())
        assert total == pytest.approx(fluorescence_yield(element, 2, 1), rel=1e-10)

    def test_cu_ka1_energy(self):
        assert emission_lines("Cu", 1, 0)["Ka1"].energy == pytest.approx(8046.3, abs=0.5)

    def test_ka1_to_ka2_ratio_is_about_two(self):
        # The 2:1 ratio follows from the 2j+1 degeneracy of L3 versus L2.
        lines = emission_lines("Cu", 1, 0)
        ratio = lines["Ka1"].intensity / lines["Ka2"].intensity
        assert ratio == pytest.approx(2.0, rel=0.05)

    def test_lines_are_sorted_by_decreasing_intensity(self):
        intensities = [line.intensity for line in emission_lines("Cu", 1, 0).values()]
        assert intensities == sorted(intensities, reverse=True)

    def test_min_intensity_filters_weak_lines(self):
        all_lines = emission_lines("Cu", 1, 0)
        strong = emission_lines("Cu", 1, 0, min_intensity=0.01)
        assert 0 < len(strong) < len(all_lines)
        assert all(line.intensity >= 0.01 for line in strong.values())

    def test_l_lines_come_from_both_spin_orbit_levels(self):
        lines = emission_lines("Ag", 2, 1)
        levels = {line.initial_level for line in lines.values()}
        assert levels == {"L2", "L3"}

    @pytest.mark.parametrize(
        "name, family",
        [("Ka1", "Ka"), ("Ka2", "Ka"), ("Kb1", "Kb"), ("Lb2,15", "Lb"), ("Ll", "Ll")],
    )
    def test_family_parsing(self, name, family):
        from abtem.inelastic.xray_data import EmissionLine

        line = EmissionLine(name, 1000.0, 1.0, "K", "L2")
        assert line.family == family

    def test_line_families_partition_the_lines(self):
        lines = emission_lines("Ag", 2, 1)
        families = line_families(lines)

        grouped = [line.name for members in families.values() for line in members]
        assert sorted(grouped) == sorted(lines)

        total = sum(
            line.intensity for members in families.values() for line in members
        )
        assert total == pytest.approx(sum(l.intensity for l in lines.values()))


class TestAtomicConstants:
    def test_natural_width_of_cu_k(self):
        assert natural_width("Cu", 1, 0) == pytest.approx(1.55, rel=1e-6)

    def test_absorption_edge_of_cu_k(self):
        assert absorption_edge("Cu", 1, 0) == pytest.approx(8979.0, rel=1e-6)

    def test_l_edge_is_between_the_two_levels(self):
        edges = xraydb.xray_edges("Ag")
        combined = absorption_edge("Ag", 2, 1)
        assert edges["L3"].energy < combined < edges["L2"].energy


class TestWithoutXraydb:
    """abTEM must import and fail helpfully when the optional extra is absent."""

    @staticmethod
    def _hide_xraydb(monkeypatch):
        # Importing a module bound to None in sys.modules raises ImportError.
        import sys

        monkeypatch.setitem(sys.modules, "xraydb", None)

    def test_data_layer_points_at_the_extra(self, monkeypatch):
        self._hide_xraydb(monkeypatch)
        with pytest.raises(ImportError, match=r"abtem\[gpaw\]"):
            fluorescence_yield("Cu", 1, 0, coster_kronig=False)



class TestCrossShellCosterKronig:
    """Vacancies migrate across the whole shell, not only within a subshell.

    Ionising 2s puts a vacancy in L1; most of it moves to L2 and L3 and
    radiates from there. Treating the subshell alone puts an L1 edge's
    fluorescence yield several times too low.
    """

    def test_shell_levels(self):
        from abtem.inelastic.xray_data import shell_levels

        assert shell_levels(1) == ("K",)
        assert shell_levels(2) == ("L1", "L2", "L3")
        assert shell_levels(3) == ("M1", "M2", "M3", "M4", "M5")

    def test_l1_vacancies_migrate(self):
        distribution = vacancy_distribution("Fe", 2, 0)
        assert set(distribution) == {"L1", "L2", "L3"}
        assert distribution["L1"] < 0.2  # most of it leaves
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_migration_matches_the_tabulated_direct_rates(self):
        f12 = xraydb.ck_probability("Fe", "L1", "L2", total=False)
        f13 = xraydb.ck_probability("Fe", "L1", "L3", total=False)
        f23 = xraydb.ck_probability("Fe", "L2", "L3", total=False)

        distribution = vacancy_distribution("Fe", 2, 0)
        assert distribution["L1"] == pytest.approx(1 - f12 - f13)
        assert distribution["L2"] == pytest.approx(f12 * (1 - f23))
        assert distribution["L3"] == pytest.approx(f13 + f12 * f23)

    def test_cascade_reproduces_the_tabulated_total_rate(self):
        # xraydb's total=True folds the cascade in; the cascade of direct rates
        # must land on the same number, or one of the two is being double
        # counted somewhere.
        for element in ["Ti", "Fe", "Cu", "Ag", "Au"]:
            distribution = vacancy_distribution(element, 2, 0)
            assert distribution["L3"] == pytest.approx(
                xraydb.ck_probability(element, "L1", "L3", total=True)
            )

    @pytest.mark.parametrize("element", ["Ti", "Fe", "Cu", "Ag"])
    def test_l1_yield_is_much_larger_than_its_own_level(self, element):
        edges = xraydb.xray_edges(element)
        combined = fluorescence_yield(element, 2, 0)
        assert combined > 2 * edges["L1"].fyield

    def test_2p_is_unchanged_by_the_wider_cascade(self):
        # L2 and L3 cannot feed L1, so a 2p edge sees only the L2 -> L3 step.
        distribution = vacancy_distribution("Fe", 2, 1)
        assert set(distribution) == {"L2", "L3"}
        f23 = xraydb.ck_probability("Fe", "L2", "L3", total=False)
        assert distribution["L2"] == pytest.approx((1 / 3) * (1 - f23))

    def test_k_shell_has_no_cascade(self):
        assert vacancy_distribution("Cu", 1, 0) == {"K": 1.0}

    def test_disabling_coster_kronig_leaves_the_subshell_alone(self):
        distribution = vacancy_distribution("Fe", 2, 0, coster_kronig=False)
        assert distribution == pytest.approx({"L1": 1.0})

    @pytest.mark.parametrize("n, l", [(2, 0), (2, 1), (3, 0), (3, 2)])
    def test_vacancies_conserved_for_every_subshell(self, n, l):
        distribution = vacancy_distribution("Au", n, l)
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_lines_come_from_levels_the_vacancy_reached(self):
        # An L1 ionisation must produce L3 lines after the cascade.
        lines = emission_lines("Ag", 2, 0)
        assert "L3" in {line.initial_level for line in lines.values()}


