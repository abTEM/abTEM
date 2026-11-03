"""Vacuum at the entrance and exit surfaces of a `CrystalPotential`."""

import warnings

import numpy as np
import pytest
from ase.build import bulk

from abtem import CrystalPotential, FrozenPhonons, Potential, Probe

A = 5.43
SAMPLING = 0.2


@pytest.fixture
def silicon():
    return bulk("Si", cubic=True, a=A)


def _slab_potential(atoms, repetitions, vacuum, slice_thickness, projection, gpts):
    """The potential of the slab the crystal stands for: its atoms with the vacuum in
    the cell, calculated as one `Potential`."""
    slab = atoms * (1, 1, repetitions)
    slab.positions[:, 2] += vacuum
    cell = np.array(slab.cell)
    cell[2, 2] += 2 * vacuum
    slab.set_cell(cell)
    return Potential(
        slab,
        gpts=gpts,
        slice_thickness=slice_thickness,
        projection=projection,
    ).build(lazy=False)


def _unit(atoms, projection, slice_thickness=A / 4):
    return Potential(
        atoms, sampling=SAMPLING, slice_thickness=slice_thickness, projection=projection
    )


def test_default_follows_the_projection(silicon):
    finite = CrystalPotential(_unit(silicon, "finite"), (1, 1, 2))
    infinite = CrystalPotential(_unit(silicon, "infinite"), (1, 1, 2))
    assert finite.add_vacuum and not infinite.add_vacuum
    assert infinite.num_slices == 8
    assert infinite.box[2] == pytest.approx(2 * A)
    assert finite.num_slices == 16
    assert finite.box[2] == pytest.approx(4 * A)
    assert finite.thickness == pytest.approx(4 * A)


def test_explicit_choice(silicon):
    finite = CrystalPotential(_unit(silicon, "finite"), (1, 1, 2), add_vacuum=False)
    assert not finite.add_vacuum and finite.num_slices == 8
    infinite = CrystalPotential(_unit(silicon, "infinite"), (1, 1, 2), add_vacuum=True)
    assert infinite.add_vacuum and infinite.num_slices == 16
    assert infinite.copy() == infinite
    assert infinite.copy().add_vacuum


def test_a_unit_that_is_not_a_potential_has_no_vacuum_by_default(silicon):
    built = _unit(silicon, "finite").build(lazy=False)
    assert not CrystalPotential(built, (1, 1, 2)).add_vacuum


def test_vacuum_has_the_slicing_of_the_unit():
    zno = bulk("ZnO", "wurtzite", a=3.25, c=5.2, orthorhombic=True)
    unit = _unit(zno, "finite", slice_thickness="auto")
    unit_slices = np.array(unit.slice_thickness)
    assert not np.allclose(unit_slices, unit_slices[0]), "needs a nonuniform pattern"

    crystal = CrystalPotential(unit, (1, 1, 2))
    n = len(unit_slices)
    # the atoms are sliced as without vacuum, between a unit cell of it on each side
    np.testing.assert_allclose(crystal.slice_thickness, np.tile(unit_slices, 4))
    assert crystal.num_slices == 4 * n
    assert crystal.thickness == pytest.approx(4 * unit.thickness)


@pytest.mark.parametrize("repetitions", [1, 2, 6])
def test_infinite_projection_vacuum_is_empty(silicon, repetitions):
    unit = _unit(silicon, "infinite")
    bare = CrystalPotential(unit, (1, 1, repetitions)).build(lazy=False).array
    array = (
        CrystalPotential(unit, (1, 1, repetitions), add_vacuum=True)
        .build(lazy=False)
        .array
    )

    n_vacuum = 4
    assert array.shape[0] == bare.shape[0] + 2 * n_vacuum
    assert not array[:n_vacuum].any()
    assert not array[-n_vacuum:].any()
    # an atom is in one slice: the others are those of the periodic stack
    np.testing.assert_array_equal(array[n_vacuum:-n_vacuum], bare)


@pytest.mark.parametrize("repetitions", [1, 2, 3, 6])
def test_finite_projection_matches_a_slab(silicon, repetitions):
    crystal = CrystalPotential(_unit(silicon, "finite"), (1, 1, repetitions))
    array = crystal.build(lazy=False).array
    expected = _slab_potential(
        silicon,
        repetitions,
        A,
        crystal.slice_thickness,
        "finite",
        crystal.gpts,
    ).array

    assert array.shape == expected.shape
    np.testing.assert_allclose(array, expected, atol=1e-4 * expected.max())
    # the vacuum holds the tails of the potential of the surface atoms
    n_vacuum = 4
    assert array[n_vacuum - 1].any() and array[-n_vacuum].any()
    # and they fall off with the distance from the surface (outermost first)
    assert array[0].max() < 1e-3 * array[n_vacuum - 1].max()
    assert array[-1].max() < 1e-3 * array[-n_vacuum].max()


def test_periodic_stack_does_not_match_a_slab(silicon):
    """The concern the option addresses: the surface slices of the periodic stack
    hold the potential of the images of the unit beyond the surfaces."""
    unit = _unit(silicon, "finite")
    stack = CrystalPotential(unit, (1, 1, 3), add_vacuum=False).build(lazy=False).array
    # a slab with vacuum, from which the vacuum slices are removed
    expected = _slab_potential(
        silicon, 3, 2 * A, (A / 4,) * 28, "finite", stack.shape[1:]
    ).array[8:-8]
    # the slab has no layer at its far end, where the stack has an image's
    assert np.abs(stack[-1] - expected[-1]).max() > 0.1 * expected.max()
    assert np.abs(stack[0] - expected[0]).max() > 1e-3 * expected.max()
    # the interior is unaffected
    np.testing.assert_allclose(stack[3:-3], expected[3:-3], atol=1e-4 * expected.max())


def test_nonuniform_auto_slicing_matches_a_slab():
    zno = bulk("ZnO", "wurtzite", a=3.25, c=5.2, orthorhombic=True)
    unit = _unit(zno, "finite", slice_thickness="auto")
    crystal = CrystalPotential(unit, (2, 1, 2))
    array = crystal.build(lazy=False).array
    expected = _slab_potential(
        zno * (2, 1, 1),
        2,
        unit.thickness,
        crystal.slice_thickness,
        "finite",
        array.shape[1:],
    ).array
    np.testing.assert_allclose(array, expected, atol=1e-4 * expected.max())


def test_builds_agree(silicon):
    unit = _unit(silicon, "finite", slice_thickness=A / 3)
    crystal = CrystalPotential(unit, (1, 1, 2), exit_planes=5)
    eager = crystal.build(lazy=False)
    np.testing.assert_array_equal(crystal.build(lazy=True).array.compute(), eager.array)

    window = crystal.build(first_slice=3, last_slice=9, lazy=False).array
    np.testing.assert_array_equal(window, eager.array[3:9])

    chunked = np.concatenate(
        [chunk.array for chunk in crystal.generate_chunked_slices(chunk_size=4)]
    )
    np.testing.assert_array_equal(chunked, eager.array)

    # the exit planes count the vacuum slices, and the last slice is the exit
    assert crystal.exit_planes[-1] == crystal.num_slices - 1
    assert len(eager.exit_planes) == len(crystal.exit_planes)


def test_lateral_repetitions_with_vacuum(silicon):
    crystal = CrystalPotential(_unit(silicon, "finite"), (2, 3, 2))
    array = crystal.build(lazy=False).array
    assert array.shape[1:] == tuple(crystal.gpts)
    assert array[0].any() or array[3].any()
    tiles = array.reshape(array.shape[0], 2, -1, 3, array.shape[2] // 3)
    np.testing.assert_array_equal(tiles[:, 0, :, 0], tiles[:, 1, :, 2])


def test_sliced_atoms_sit_between_the_vacuum(silicon):
    crystal = CrystalPotential(_unit(silicon, "infinite"), (1, 1, 2), add_vacuum=True)
    atoms = crystal.get_sliced_atoms().atoms
    assert atoms.cell[2, 2] == pytest.approx(4 * A)
    assert atoms.positions[:, 2].min() == pytest.approx(A)
    assert atoms.positions[:, 2].max() < A + 2 * A


def test_frozen_phonon_pool_with_vacuum(silicon):
    phonons = FrozenPhonons(silicon, num_configs=4, sigmas=0.08, seed=1)
    unit = Potential(phonons, sampling=SAMPLING, slice_thickness=1.5, projection="finite")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal = CrystalPotential(unit, (2, 2, 2), seeds=1)
        bare = CrystalPotential(unit, (2, 2, 2), seeds=1, add_vacuum=False)
        eager = crystal.build(lazy=False).array
        lazy = crystal.build(lazy=True).array.compute()
        n_bare = bare.build(lazy=False).array.shape[1]

    # a unit cell of vacuum on each side: as many slices as the unit has
    assert crystal.add_vacuum
    assert eager.shape[1] == n_bare + 2 * len(unit.slice_thickness)
    np.testing.assert_array_equal(eager, lazy)
    # an outer vacuum slice holds little, one at the surface much more
    n_vacuum = len(unit.slice_thickness)
    profile = eager[0].sum(axis=(1, 2))
    assert profile[0] < 0.01 * profile.max()
    assert profile[n_vacuum - 1] > 10 * profile[0]
    assert profile[-n_vacuum] > 10 * profile[-1]


def test_multislice_through_the_vacuum(silicon):
    unit = _unit(silicon, "finite", slice_thickness=A / 2)
    crystal = CrystalPotential(unit, (2, 2, 2))
    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(crystal)
    exit_wave = probe.multislice(crystal)
    assert np.isfinite(exit_wave.intensity().array).all()
    # no absorption: the vacuum only propagates
    bare = CrystalPotential(unit, (2, 2, 2), add_vacuum=False)
    exit_bare = probe.multislice(bare)
    np.testing.assert_allclose(
        exit_wave.intensity().array.sum(), exit_bare.intensity().array.sum(), rtol=1e-4
    )
