"""CrystalPotential's gpts and sampling setters keep the unit and the crystal grid
consistent: the crystal's gpts are the unit's gpts times the x and y repetitions."""

import numpy as np
import pytest
from ase import Atoms

import abtem


def _unit(cell=(4, 3, 5), gpts=(40, 30)):
    atoms = Atoms("C", positions=[(1, 1, 1)], cell=cell, pbc=True)
    return abtem.Potential(atoms, gpts=gpts)


def test_gpts_setter_checks_each_axis_against_its_own_repetitions():
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    # 100 is divisible by the 2 repetitions along x but not by the 3 along y.
    with pytest.raises(ValueError, match="divisible"):
        crystal.gpts = (100, 100)

    assert crystal.gpts == (80, 90)
    assert crystal.potential_unit.gpts == (40, 30)


@pytest.mark.parametrize("gpts", [(101, 99), (100, 98), (101, 100)])
def test_gpts_setter_rejects_a_count_that_is_not_a_multiple(gpts):
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    with pytest.raises(ValueError, match="divisible"):
        crystal.gpts = gpts

    assert crystal.gpts == (80, 90)
    assert crystal.potential_unit.gpts == (40, 30)


def test_gpts_setter_accepts_a_count_divisible_along_each_axis():
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    crystal.gpts = (100, 99)

    assert crystal.gpts == (100, 99)
    assert crystal.potential_unit.gpts == (50, 33)
    assert crystal.build(lazy=False).array.shape[-2:] == (100, 99)


def test_gpts_setter_accepts_what_only_the_y_repetitions_divide():
    # 99 / 3 along x and 100 / 2 along y: 100 is not divisible by the 3 repetitions
    # along x, which the setter must not test the y count against.
    crystal = abtem.CrystalPotential(_unit(), repetitions=(3, 2, 1))

    crystal.gpts = (99, 100)

    assert crystal.gpts == (99, 100)
    assert crystal.potential_unit.gpts == (33, 50)
    assert crystal.build(lazy=False).array.shape[-2:] == (99, 100)


def test_sampling_setter_sets_the_crystal_and_the_unit():
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    crystal.sampling = 0.05

    assert crystal.sampling == pytest.approx((0.05, 0.05))
    assert crystal.potential_unit.sampling == pytest.approx((0.05, 0.05))
    assert crystal.gpts == (160, 180)
    assert crystal.potential_unit.gpts == (80, 60)
    assert crystal.build(lazy=False).array.shape[-2:] == (160, 180)


def test_sampling_setter_takes_a_sampling_for_each_axis():
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    crystal.sampling = (0.05, 0.1)

    assert crystal.sampling == pytest.approx((0.05, 0.1))
    assert crystal.potential_unit.sampling == pytest.approx((0.05, 0.1))
    assert crystal.gpts == (160, 90)
    assert crystal.build(lazy=False).array.shape[-2:] == (160, 90)


def test_sampling_setter_keeps_the_crystal_gpts_a_multiple_of_the_unit_gpts():
    # 4.1 / 0.15 = 27.3 and 8.2 / 0.15 = 54.7: the sampling does not divide the
    # crystal into whole unit cells, so the crystal's gpts follow the unit's.
    crystal = abtem.CrystalPotential(
        _unit(cell=(4.1, 3.3, 5), gpts=(41, 33)), repetitions=(2, 3, 1)
    )

    crystal.sampling = 0.15

    unit_gpts = crystal.potential_unit.gpts
    assert unit_gpts == (28, 22)
    assert crystal.gpts == (2 * unit_gpts[0], 3 * unit_gpts[1])
    assert crystal.build(lazy=False).array.shape[-2:] == crystal.gpts
    assert crystal.sampling == pytest.approx(
        (4.1 * 2 / crystal.gpts[0], 3.3 * 3 / crystal.gpts[1])
    )


def test_sampling_setter_with_an_invalid_value_leaves_the_crystal_unchanged():
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    with pytest.raises(RuntimeError, match="Invalid grid property"):
        crystal.sampling = "fine"

    assert crystal.gpts == (80, 90)
    assert crystal.potential_unit.gpts == (40, 30)


def test_construction_and_getters_are_unchanged():
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    assert crystal.gpts == (80, 90)
    assert crystal.sampling == pytest.approx((0.1, 0.1))
    assert crystal.potential_unit.gpts == (40, 30)
    assert crystal.build(lazy=False).array.shape[-2:] == (80, 90)


def _built_unit_crystal(repetitions=(2, 1, 1)):
    unit = _unit().build(lazy=False)
    return abtem.CrystalPotential(unit, repetitions=repetitions), unit


def test_sampling_setter_rejects_a_built_unit_and_leaves_it_unchanged():
    # The grid of a PotentialArray is that of its data.
    crystal, unit = _built_unit_crystal()

    with pytest.raises(RuntimeError, match="PotentialArray"):
        crystal.sampling = 0.05

    assert unit.gpts == (40, 30)
    assert unit.sampling == pytest.approx((0.1, 0.1))
    assert crystal.gpts == (80, 30)
    assert crystal.build(lazy=False).array.shape[-2:] == (80, 30)


def test_gpts_setter_rejects_a_built_unit_and_leaves_it_unchanged():
    crystal, unit = _built_unit_crystal()

    with pytest.raises(RuntimeError, match="PotentialArray"):
        crystal.gpts = (100, 30)

    assert unit.gpts == (40, 30)
    assert crystal.gpts == (80, 30)
    assert crystal.build(lazy=False).array.shape[-2:] == (80, 30)


def test_setting_the_grid_a_built_unit_already_has_is_accepted():
    crystal, unit = _built_unit_crystal()

    crystal.gpts = (80, 30)
    crystal.sampling = (0.1, 0.1)

    assert crystal.gpts == (80, 30)
    assert unit.gpts == (40, 30)


def test_gpts_setter_checks_the_divisibility_of_a_built_unit_first():
    crystal, _ = _built_unit_crystal()

    with pytest.raises(ValueError, match="divisible"):
        crystal.gpts = (81, 30)


@pytest.mark.parametrize(
    "sampling", [-0.1, 0, 0.0, (0.05, -0.1), (0, 0.05), np.nan, np.inf, None]
)
def test_sampling_setter_rejects_a_sampling_that_is_not_positive(sampling):
    crystal = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))

    with pytest.raises(ValueError, match="positive"):
        crystal.sampling = sampling

    assert crystal.gpts == (80, 90)
    assert crystal.potential_unit.gpts == (40, 30)
    assert crystal.sampling == pytest.approx((0.1, 0.1))


def test_a_sampling_that_equals_the_grid_of_a_built_unit_leaves_it_untouched():
    # 0.0999999 rounds the unit's gpts up to (41, 31): the unit is not asked.
    crystal, unit = _built_unit_crystal()

    crystal.sampling = 0.0999999

    assert unit.gpts == (40, 30)
    assert crystal.gpts == (80, 30)
    assert crystal.build(lazy=False).array.shape[-2:] == (80, 30)


def test_the_exact_sampling_of_a_built_unit_is_not_rounded_to_a_fast_size():
    unit = _unit(cell=(4.1, 3.3, 5), gpts=(41, 33)).build(lazy=False)
    crystal = abtem.CrystalPotential(unit, repetitions=(2, 1, 1))

    with abtem.config.set({"grid.round-to-fast-fft": True}):
        crystal.sampling = unit.sampling

    assert unit.gpts == (41, 33)
    assert crystal.gpts == (82, 33)


@pytest.mark.parametrize("gpts", [(243, 180), (240, 182)])
def test_a_unit_crystal_that_rejects_its_gpts_leaves_the_crystal_unchanged(gpts):
    # The outer repetitions divide both counts; the inner ones (2, 3) do not
    # divide 243 / 3 = 81 along x or 182 / 2 = 91 along y.
    inner = abtem.CrystalPotential(_unit(), repetitions=(2, 3, 1))
    crystal = abtem.CrystalPotential(inner, repetitions=(3, 2, 1))

    with pytest.raises(ValueError, match="divisible"):
        crystal.gpts = gpts

    assert crystal.gpts == (240, 180)
    assert inner.gpts == (80, 90)
    assert crystal.build(lazy=False).array.shape[-2:] == (240, 180)


def test_a_unit_crystal_of_a_built_unit_leaves_the_crystal_unchanged():
    inner, unit = _built_unit_crystal()
    crystal = abtem.CrystalPotential(inner, repetitions=(1, 3, 1))

    with pytest.raises(RuntimeError, match="PotentialArray"):
        crystal.gpts = (160, 90)

    assert crystal.gpts == (80, 90)
    assert inner.gpts == (80, 30)
    assert crystal.build(lazy=False).array.shape[-2:] == (80, 90)
