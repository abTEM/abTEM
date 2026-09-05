"""Tests for the X-ray emission data adapter and the X-ray detector.

These cover the atomic-data layer and the radiometric chain only. They are pure
lookup and arithmetic: no multislice, no device, no GPAW.
"""

from __future__ import annotations

import numpy as np
import pytest

from abtem.inelastic.xray import (
    SDDEfficiency,
    TabulatedEfficiency,
    XrayDetector,
)
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


class TestEfficiencyModels:
    def test_tabulated_efficiency_interpolates(self):
        efficiency = TabulatedEfficiency([1000.0, 10000.0], [0.5, 1.0])
        assert efficiency(5500.0) == pytest.approx(0.75)

    def test_tabulated_efficiency_clamps_outside_the_range(self):
        efficiency = TabulatedEfficiency([1000.0, 10000.0], [0.5, 1.0])
        assert efficiency(10.0) == pytest.approx(0.5)
        assert efficiency(1e6) == pytest.approx(1.0)

    def test_tabulated_efficiency_requires_increasing_energies(self):
        with pytest.raises(ValueError):
            TabulatedEfficiency([10000.0, 1000.0], [0.5, 1.0])

    def test_tabulated_efficiency_requires_matching_shapes(self):
        with pytest.raises(ValueError):
            TabulatedEfficiency([1000.0, 10000.0], [0.5])

    def test_sdd_efficiency_is_between_zero_and_one(self):
        energies = np.logspace(np.log10(200.0), np.log10(30000.0), 50)
        values = SDDEfficiency(window=("Be", 8.0))(energies)
        assert np.all(values >= 0.0) and np.all(values <= 1.0)

    def test_thick_sensor_absorbs_everything_at_low_energy(self):
        # A 450 um Si sensor is opaque to a few-keV photon.
        assert SDDEfficiency(contact=None, dead_layer=None)(2000.0) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_sensor_becomes_transparent_at_high_energy(self):
        efficiency = SDDEfficiency()
        assert efficiency(30000.0)[0] < efficiency(8000.0)[0]

    def test_beryllium_window_blocks_soft_x_rays(self):
        windowless = SDDEfficiency()
        windowed = SDDEfficiency(window=("Be", 8.0))

        # C Ka at 277 eV is absorbed by the window but not by a windowless
        # detector; Cu Ka at 8 keV passes both.
        assert windowless(277.0)[0] > 0.4
        assert windowed(277.0)[0] < 1e-6
        assert windowed(8040.0)[0] == pytest.approx(windowless(8040.0)[0], rel=5e-3)

    def test_dead_layer_absorption_edge_is_visible(self):
        # The Si K edge of the dead layer at 1839 eV dips the efficiency.
        efficiency = SDDEfficiency(dead_layer=("Si", 0.5))
        assert efficiency(1850.0)[0] < efficiency(1830.0)[0]

    def test_thicker_window_transmits_less(self):
        thin = SDDEfficiency(window=("Be", 4.0))(1500.0)[0]
        thick = SDDEfficiency(window=("Be", 25.0))(1500.0)[0]
        assert thick < thin

    def test_invalid_layer_specification_raises(self):
        with pytest.raises(ValueError):
            SDDEfficiency(window="Be")

        with pytest.raises(ValueError):
            SDDEfficiency(window=("Be", -1.0))


class TestXrayDetector:
    def test_collection_fraction_is_the_isotropic_solid_angle_fraction(self):
        detector = XrayDetector(solid_angle=0.7)
        assert detector.collection_fraction == pytest.approx(0.7 / (4 * np.pi))

    def test_full_sphere_collects_everything(self):
        detector = XrayDetector(solid_angle=4 * np.pi)
        assert detector.collection_fraction == pytest.approx(1.0)
        assert detector.total_yield("Cu", 1, 0) == pytest.approx(
            fluorescence_yield("Cu", 1, 0)
        )

    @pytest.mark.parametrize("solid_angle", [0.0, -1.0, 4 * np.pi + 0.1])
    def test_invalid_solid_angle_raises(self, solid_angle):
        with pytest.raises(ValueError):
            XrayDetector(solid_angle=solid_angle)

    @pytest.mark.parametrize("efficiency", [-0.1, 1.5])
    def test_invalid_constant_efficiency_raises(self, efficiency):
        with pytest.raises(ValueError):
            XrayDetector(solid_angle=1.0, efficiency=efficiency)

    def test_total_yield_is_the_full_radiometric_chain(self):
        detector = XrayDetector(solid_angle=0.7, efficiency=0.9)
        expected = fluorescence_yield("Cu", 1, 0) * 0.7 / (4 * np.pi) * 0.9
        assert detector.total_yield("Cu", 1, 0) == pytest.approx(expected)

    def test_yield_scales_linearly_with_solid_angle(self):
        one = XrayDetector(solid_angle=0.1).total_yield("Fe", 1, 0)
        two = XrayDetector(solid_angle=0.2).total_yield("Fe", 1, 0)
        assert two == pytest.approx(2 * one)

    def test_line_selection_by_family(self):
        detector = XrayDetector(solid_angle=1.0, lines="Ka")
        names = set(detector.detected_lines("Cu", 1, 0))
        assert names == {"Ka1", "Ka2", "Ka3"}

    def test_line_selection_by_name(self):
        detector = XrayDetector(solid_angle=1.0, lines=["Ka1"])
        assert set(detector.detected_lines("Cu", 1, 0)) == {"Ka1"}

    def test_selected_lines_sum_to_less_than_all_lines(self):
        everything = XrayDetector(solid_angle=1.0).total_yield("Cu", 1, 0)
        alpha = XrayDetector(solid_angle=1.0, lines="Ka").total_yield("Cu", 1, 0)
        beta = XrayDetector(solid_angle=1.0, lines="Kb").total_yield("Cu", 1, 0)
        assert alpha + beta == pytest.approx(everything)

    def test_unavailable_line_raises_with_available_families(self):
        detector = XrayDetector(solid_angle=1.0, lines="La")
        with pytest.raises(ValueError, match="available families"):
            detector.detected_lines("Cu", 1, 0)

    def test_empty_line_selection_raises(self):
        with pytest.raises(ValueError):
            XrayDetector(solid_angle=1.0, lines=[])

    def test_efficiency_callable_is_applied_per_line(self):
        # An efficiency that is zero below 8.5 keV keeps only the Kb lines.
        detector = XrayDetector(
            solid_angle=1.0,
            efficiency=TabulatedEfficiency([8500.0, 8501.0], [0.0, 1.0]),
        )
        lines = detector.detected_lines("Cu", 1, 0)
        assert lines["Ka1"].intensity == pytest.approx(0.0)
        assert lines["Kb1"].intensity > 0.0

    def test_from_sdd_matches_an_explicit_efficiency_model(self):
        efficiency = SDDEfficiency(window=("Be", 8.0))
        expected = XrayDetector(solid_angle=0.7, efficiency=efficiency)
        actual = XrayDetector.from_sdd(0.7, window=("Be", 8.0))
        assert actual.total_yield("Cu", 1, 0) == pytest.approx(
            expected.total_yield("Cu", 1, 0)
        )

    def test_windowed_detector_is_blind_to_carbon(self):
        windowed = XrayDetector.from_sdd(0.7, window=("Be", 8.0))
        windowless = XrayDetector.from_sdd(0.7)
        assert windowed.total_yield("C", 1, 0) < 1e-12
        assert windowless.total_yield("C", 1, 0) > 1e-6

    def test_detected_families_group_the_detected_lines(self):
        detector = XrayDetector(solid_angle=1.0)
        families = detector.detected_families("Cu", 1, 0)
        assert set(families) == {"Ka", "Kb"}

        total = sum(
            line.intensity for members in families.values() for line in members
        )
        assert total == pytest.approx(detector.total_yield("Cu", 1, 0))

    def test_coster_kronig_changes_the_l_shell_yield(self):
        with_ck = XrayDetector(solid_angle=1.0, coster_kronig=True)
        without = XrayDetector(solid_angle=1.0, coster_kronig=False)
        assert with_ck.total_yield("Ag", 2, 1) != pytest.approx(
            without.total_yield("Ag", 2, 1)
        )

    def test_copy_preserves_parameters(self):
        detector = XrayDetector.from_sdd(0.7, window=("Be", 8.0), lines="Ka")
        clone = detector.copy()
        assert clone.solid_angle == detector.solid_angle
        assert clone.lines == detector.lines
        assert clone.total_yield("Cu", 1, 0) == pytest.approx(
            detector.total_yield("Cu", 1, 0)
        )


class TestToCounts:
    @staticmethod
    def _ionization():
        from abtem.measurements import Images

        return Images(
            np.ones((4, 4)) * 1e-3,
            sampling=(0.1, 0.1),
            metadata={"label": "ionisation", "units": "arb. unit"},
        )

    def test_to_counts_scales_by_the_total_yield(self):
        detector = XrayDetector(solid_angle=0.7)
        counts = detector.to_counts(self._ionization(), "Cu", 1, 0)

        expected = 1e-3 * detector.total_yield("Cu", 1, 0)
        assert np.allclose(counts.array, expected)
        assert counts.metadata["units"] == "photons / electron"

    def test_to_counts_per_family_stacks_the_families(self):
        detector = XrayDetector(solid_angle=0.7)
        counts = detector.to_counts(self._ionization(), "Cu", 1, 0, per_family=True)

        assert counts.shape[0] == 2
        assert counts.ensemble_axes_metadata[0].values == ("Ka", "Kb")
        assert counts.array.sum() == pytest.approx(
            detector.to_counts(self._ionization(), "Cu", 1, 0).array.sum()
        )

    def test_to_counts_per_family_with_one_family_is_not_stacked(self):
        detector = XrayDetector(solid_angle=0.7, lines="Ka")
        counts = detector.to_counts(self._ionization(), "Cu", 1, 0, per_family=True)
        assert counts.shape == (4, 4)


def test_xray_detector_is_a_valid_detector():
    """It models the experiment, so it can be passed straight to a simulation."""
    from abtem.detectors import IonizationDetector, validate_detectors

    detector = XrayDetector(solid_angle=0.7)
    assert validate_detectors([detector]) == [detector]
    assert isinstance(detector, IonizationDetector)


def test_xray_detector_needs_the_edge_identity():
    detector = XrayDetector(solid_angle=0.7)
    with pytest.raises(RuntimeError, match="which edge was ionised"):
        detector._photon_yield({"energy": 100e3})


def test_photon_yield_is_the_total_yield():
    detector = XrayDetector(solid_angle=0.7)
    assert detector._photon_yield(
        {"Z": 29, "n": 1, "l": 0}
    ) == pytest.approx(detector.total_yield("Cu", 1, 0))


def test_ionization_detector_yield_is_one():
    from abtem.detectors import IonizationDetector

    assert IonizationDetector()._photon_yield({}) == 1.0


def test_exported_from_the_top_level_namespace():
    import abtem

    assert abtem.XrayDetector is XrayDetector
    assert "XrayDetector" in abtem.__all__


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

    def test_efficiency_model_points_at_the_extra(self, monkeypatch):
        self._hide_xraydb(monkeypatch)
        with pytest.raises(ImportError, match=r"abtem\[gpaw\]"):
            SDDEfficiency()(8000.0)

    def test_constant_efficiency_needs_no_xraydb(self, monkeypatch):
        self._hide_xraydb(monkeypatch)
        # Only the layered efficiency model needs the tabulated coefficients.
        assert XrayDetector(solid_angle=0.7, efficiency=0.5).collection_fraction > 0


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


class TestCrossShellCosterKronigMShell:
    """The same cross-shell migration as TestCrossShellCosterKronig, but for
    the 5-level M shell (M1..M5), which exercises transfers that skip levels
    (e.g. M1 -> M4 directly, not only M1 -> M2 -> ... -> M4) and elements
    where some channels are simply untabulated (rate 0).

    M-shell Coster-Kronig data is only tabulated for the heavier elements
    where the sub-levels are resolved, so these tests use Au, Pb, W and U.
    """

    HEAVY_ELEMENTS = ["W", "Au", "Pb", "U"]

    def test_m1_vacancies_migrate_across_the_whole_shell(self):
        distribution = vacancy_distribution("Au", 3, 0)
        assert set(distribution) == {"M1", "M2", "M3", "M4", "M5"}
        assert distribution["M1"] < 0.2  # most of it leaves
        assert sum(distribution.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("element", HEAVY_ELEMENTS)
    def test_migration_matches_a_manual_cascade_of_the_direct_rates(self, element):
        # Redo the cascade from raw ck_probability(..., total=False) calls,
        # independently of _coster_kronig_cascade, following the same
        # decreasing-binding-energy processing order the module documents.
        levels = ["M1", "M2", "M3", "M4", "M5"]
        vacancies = {level: 0.0 for level in levels}
        vacancies["M1"] = 1.0

        for i, upper in enumerate(levels):
            if vacancies[upper] == 0.0:
                continue
            transfers = {
                lower: vacancies[upper]
                * xraydb.ck_probability(element, upper, lower, total=False)
                for lower in levels[i + 1 :]
            }
            vacancies[upper] -= sum(transfers.values())
            for lower, amount in transfers.items():
                vacancies[lower] += amount

        distribution = vacancy_distribution(element, 3, 0)
        for level in levels:
            assert distribution.get(level, 0.0) == pytest.approx(
                vacancies[level], abs=1e-9
            ), f"{element} {level}"

    @pytest.mark.parametrize("element", HEAVY_ELEMENTS)
    def test_cascade_reproduces_the_tabulated_total_rate(self, element):
        # xraydb's total=True is the cumulative probability a vacancy ever
        # passes through "lower", not the final resting population -- for an
        # intermediate level (M2, M3, M4) part of that keeps moving on down
        # the shell, so total > the final distribution fraction there. Only
        # M5, the last level in the shell, has nowhere further to go, so
        # cumulative arrivals there equals the final population -- the same
        # reason the L-shell version of this test only checks L3.
        distribution = vacancy_distribution(element, 3, 0)
        assert distribution["M5"] == pytest.approx(
            xraydb.ck_probability(element, "M1", "M5", total=True)
        )

    @pytest.mark.parametrize("element", HEAVY_ELEMENTS)
    def test_intermediate_levels_leak_less_than_their_cumulative_total(self, element):
        # For a non-terminal level, the tabulated "total" (cumulative
        # pass-through) must be >= the final resting population, since some
        # of what arrives at M2/M3/M4 continues cascading onward to M5.
        distribution = vacancy_distribution(element, 3, 0)
        for lower in ["M2", "M3", "M4"]:
            total = xraydb.ck_probability(element, "M1", lower, total=True)
            assert distribution[lower] <= total + 1e-9, f"{element} {lower}"

    def test_m2_ionisation_does_not_feed_back_into_m1(self):
        # M1 has higher binding energy than M2, so an M2 (3p1/2) vacancy can
        # only cascade forward to M3, M4, M5.
        distribution = vacancy_distribution("Au", 3, 1)
        assert "M1" not in distribution
        assert set(distribution) <= {"M2", "M3", "M4", "M5"}
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_m4_ionisation_only_cascades_to_m5(self):
        # The (n=3, l=2) subshell is M4 and M5; M4 can only feed M5.
        distribution = vacancy_distribution("Au", 3, 2)
        assert set(distribution) == {"M4", "M5"}
        f45 = xraydb.ck_probability("Au", "M4", "M5", total=False)
        m4_weight, m5_weight = statistical_weights(2)
        assert distribution["M4"] == pytest.approx(m4_weight * (1 - f45))
        assert distribution["M5"] == pytest.approx(
            m5_weight + m4_weight * f45
        )

    def test_missing_ck_channels_are_treated_as_zero(self):
        # M4 -> M5 is untabulated (0.0) for lighter elements with otherwise
        # complete M-shell CK data; the cascade must not crash or lose
        # vacancies over it, it just carries nothing through that channel.
        assert xraydb.ck_probability("Ag", "M4", "M5", total=False) == 0.0
        distribution = vacancy_distribution("Ag", 3, 2)
        m4_weight, m5_weight = statistical_weights(2)
        assert distribution["M4"] == pytest.approx(m4_weight)
        assert distribution["M5"] == pytest.approx(m5_weight)
        assert sum(distribution.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("element", HEAVY_ELEMENTS)
    @pytest.mark.parametrize("n, l", [(3, 0), (3, 1), (3, 2)])
    def test_vacancies_conserved_for_every_m_subshell(self, element, n, l):
        distribution = vacancy_distribution(element, n, l)
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_lines_come_from_levels_the_vacancy_reached(self):
        # An M1 ionisation of a heavy element must produce M3/M4/M5 lines
        # after the cascade -- not just whatever M1 itself radiates.
        lines = emission_lines("Au", 3, 0)
        initial_levels = {line.initial_level for line in lines.values()}
        assert initial_levels & {"M3", "M4", "M5"}
        assert "M1" not in initial_levels  # too little M1 population left

    @pytest.mark.parametrize("element", HEAVY_ELEMENTS)
    def test_m1_yield_exceeds_its_own_level(self, element):
        # Most of an M1 vacancy ends up at M4/M5, which fluoresce far more
        # strongly than M1 itself, so the cascaded yield should be well
        # above the bare tabulated M1 omega.
        edges = xraydb.xray_edges(element)
        combined = fluorescence_yield(element, 3, 0)
        assert combined > 5 * edges["M1"].fyield


class TestCombineSubshells:
    @staticmethod
    def _image(value):
        from abtem.measurements import Images

        return Images(np.full((2, 2), value), sampling=(0.1, 0.1))

    def test_combines_weighted_by_each_subshell_yield(self):
        detector = XrayDetector(solid_angle=0.7)
        got = detector.to_counts_from_subshells(
            {(2, 0): self._image(1e-3), (2, 1): self._image(2e-3)}, "Ag"
        )
        expected = 1e-3 * detector.total_yield("Ag", 2, 0) + 2e-3 * detector.total_yield(
            "Ag", 2, 1
        )
        assert float(np.asarray(got.array)[0, 0]) == pytest.approx(expected)
        assert got.metadata["units"] == "photons / electron"

    def test_single_entry_matches_to_counts(self):
        detector = XrayDetector(solid_angle=0.7)
        image = self._image(1e-3)
        combined = detector.to_counts_from_subshells({(2, 1): image}, "Ag")
        direct = detector.to_counts(image, "Ag", 2, 1)
        np.testing.assert_allclose(
            np.asarray(combined.array), np.asarray(direct.array), rtol=1e-12
        )

    def test_mixing_shells_is_refused(self):
        detector = XrayDetector(solid_angle=0.7)
        with pytest.raises(ValueError, match="one shell"):
            detector.to_counts_from_subshells(
                {(1, 0): self._image(1.0), (2, 1): self._image(1.0)}, "Ag"
            )

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="at least one subshell"):
            XrayDetector(solid_angle=0.7).to_counts_from_subshells({}, "Ag")


class TestSpecimenAbsorption:
    """Photons generated at depth z traverse z / sin(takeoff) of specimen."""

    def test_zero_depth_is_transparent(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        assert SpecimenAbsorption("Si").transmission(1740.0, 0.0)[0] == 1.0

    def test_transmission_falls_with_depth(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        absorption = SpecimenAbsorption("Si")
        depths = [0.0, 100.0, 1000.0, 10000.0]
        values = [float(absorption.transmission(1740.0, d)[0]) for d in depths]
        assert np.all(np.diff(values) < 0)
        assert values[-1] > 0.0

    def test_soft_lines_are_absorbed_more(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        absorption = SpecimenAbsorption("Si")
        soft = float(absorption.transmission(1740.0, 5000.0)[0])   # Si Ka
        hard = float(absorption.transmission(8040.0, 5000.0)[0])   # Cu Ka
        assert soft < hard

    def test_shallower_takeoff_absorbs_more(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        steep = SpecimenAbsorption("Si", takeoff_angle=60.0)
        shallow = SpecimenAbsorption("Si", takeoff_angle=10.0)
        assert float(shallow.transmission(1740.0, 5000.0)[0]) < float(
            steep.transmission(1740.0, 5000.0)[0]
        )

    def test_ninety_degrees_is_the_shortest_path(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        import xraydb as _xdb

        absorption = SpecimenAbsorption("Si", takeoff_angle=90.0)
        depth = 5000.0
        mu = _xdb.material_mu("Si", np.array([1740.0]))[0]
        expected = np.exp(-mu * depth * 1e-8)
        assert float(absorption.transmission(1740.0, depth)[0]) == pytest.approx(
            expected, rel=1e-5
        )

    @pytest.mark.parametrize("angle", [0.0, -5.0, 91.0])
    def test_invalid_takeoff_angle_raises(self, angle):
        from abtem.inelastic.xray import SpecimenAbsorption

        with pytest.raises(ValueError, match="takeoff_angle"):
            SpecimenAbsorption("Si", takeoff_angle=angle)

    def test_invalid_density_raises(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        with pytest.raises(ValueError, match="density"):
            SpecimenAbsorption("Si", density=-1.0)

    def test_detector_yield_falls_with_depth(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        detector = XrayDetector(0.7, absorption=SpecimenAbsorption("Si"))
        edge = {"Z": 14, "n": 1, "l": 0}
        shallow = detector._photon_yield({**edge, "depth": 0.0})
        deep = detector._photon_yield({**edge, "depth": 5000.0})
        assert deep < shallow
        assert shallow == pytest.approx(
            XrayDetector(0.7)._photon_yield(edge), rel=1e-12
        )

    def test_absorption_needs_a_depth(self):
        from abtem.inelastic.xray import SpecimenAbsorption

        detector = XrayDetector(0.7, absorption=SpecimenAbsorption("Si"))
        with pytest.raises(RuntimeError, match="depth at which the photon"):
            detector._photon_yield({"Z": 14, "n": 1, "l": 0})

    def test_absorption_is_off_by_default(self):
        assert XrayDetector(0.7).absorption is None
