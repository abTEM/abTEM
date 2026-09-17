import numpy as np
import pytest

from abtem.potentials.core_densities import (
    conventional_core_electrons,
    core_configuration,
    slater_core_form_factor,
)


@pytest.mark.parametrize(
    "symbol, expected",
    [("H", 0), ("B", 2), ("N", 2), ("O", 2), ("S", 10), ("Ti", 18), ("Mo", 36)],
)
def test_conventional_core_is_the_largest_smaller_noble_gas(symbol, expected):
    assert conventional_core_electrons(symbol) == expected


def test_cores_are_filled_by_shell_not_by_aufbau_energy():
    """A frozen core is filled by shell, so strontium's 28-electron core is
    [Ar]3d10. Aufbau energy ordering puts 4s before 3d and would give [Ar]4s2 3d8 --
    an error that puts Sr's core form factor ~98% off."""
    configuration = core_configuration(28)
    assert configuration[-1] == (3, 2, 10), configuration
    assert sum(occupancy for _, _, occupancy in configuration) == 28
    assert not any(n == 4 for n, _, _ in configuration)


@pytest.mark.parametrize("symbol, core", [("B", 2), ("S", 10), ("Sr", 28), ("Mo", 36)])
def test_form_factor_equals_the_core_electron_count_at_zero(symbol, core):
    G = np.linspace(0.0, 50.0, 200)
    assert slater_core_form_factor(symbol, core, G)[0] == pytest.approx(core, rel=1e-6)


@pytest.mark.parametrize("symbol, core", [("B", 2), ("S", 10), ("Mo", 36)])
def test_form_factor_decays_monotonically(symbol, core):
    G = np.linspace(0.0, 60.0, 300)
    f = slater_core_form_factor(symbol, core, G)
    assert np.all(np.diff(f) < 1e-9)
    assert f[-1] < 0.5 * core


def test_no_core_gives_no_correction():
    G = np.linspace(0.0, 50.0, 50)
    assert np.all(slater_core_form_factor("H", 0, G) == 0.0)


def test_heavier_cores_are_more_compact_in_real_space():
    """A heavier element's core is bound more tightly, so its form factor falls off
    more slowly in reciprocal space."""
    G = np.linspace(0.0, 60.0, 300)
    boron = slater_core_form_factor("B", 2, G) / 2
    sulphur = slater_core_form_factor("S", 10, G) / 10
    assert sulphur[-1] > boron[-1]
