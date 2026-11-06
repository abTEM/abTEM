"""The slices at the surfaces of a `CrystalPotential` with a finite projection."""

import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from utils import devices

from abtem import CrystalPotential, FrozenPhonons, PlaneWave, Potential, PotentialArray
from abtem.core.backend import asnumpy

A = 5.43
SAMPLING = 0.2

# Most of the tests use atoms on the entrance plane, the sensitive case, and the
# warning about them is tested on its own.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*atom.* of the potential unit are within:UserWarning"
)


@pytest.fixture
def silicon():
    return bulk("Si", cubic=True, a=A)


@pytest.fixture
def zinc_oxide():
    return bulk("ZnO", "wurtzite", a=3.25, c=5.2, orthorhombic=True)


def _unit(atoms, projection="finite", slice_thickness=A / 4, device="cpu"):
    return Potential(
        atoms,
        sampling=SAMPLING,
        slice_thickness=slice_thickness,
        projection=projection,
        device=device,
    )


def _slab(atoms, repetitions, vacuum, slice_thickness, gpts):
    """The potential of the slab the crystal stands for: its atoms between `vacuum`
    units of vacuum in the cell, calculated as one `Potential`, as an array with the
    slices of the vacuum and of the crystal apart."""
    height = atoms.cell[2, 2]
    slab = atoms * (1, 1, repetitions)
    slab.positions[:, 2] += vacuum * height
    cell = np.array(slab.cell)
    cell[2, 2] += 2 * vacuum * height
    slab.set_cell(cell)

    n_unit = len(slice_thickness) // repetitions
    thickness = tuple(slice_thickness[:n_unit]) * (repetitions + 2 * vacuum)
    array = (
        Potential(slab, gpts=gpts, slice_thickness=thickness, projection="finite")
        .build(lazy=False)
        .array
    )
    n_vacuum = vacuum * n_unit
    return array[:n_vacuum], array[n_vacuum:-n_vacuum], array[-n_vacuum:]


def _folded_slab(atoms, repetitions, vacuum, slice_thickness, gpts):
    """The slab with the potential in its vacuum added to the slices next to it."""
    entrance, crystal, exit_ = _slab(atoms, repetitions, vacuum, slice_thickness, gpts)
    crystal[0] += entrance.sum(0)
    crystal[-1] += exit_.sum(0)
    return crystal


def _assert_same(actual, expected, device):
    """Equal on the cpu. A pool built twice on an accelerator can differ in the last
    bit of float32, as the order of the sums of its kernels is not fixed."""
    if device == "cpu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max()
        )


def test_default_follows_the_projection(silicon):
    assert CrystalPotential(_unit(silicon), (1, 1, 2)).correct_surfaces
    assert not CrystalPotential(
        _unit(silicon, projection="infinite"), (1, 1, 2)
    ).correct_surfaces


def test_explicit_choice(silicon):
    unit = _unit(silicon)
    assert not CrystalPotential(
        unit, (1, 1, 2), correct_surfaces=False
    ).correct_surfaces
    assert CrystalPotential(
        _unit(silicon, projection="infinite"), (1, 1, 2), correct_surfaces=True
    ).correct_surfaces

    crystal = CrystalPotential(unit, (1, 1, 2), correct_surfaces=False)
    assert crystal.copy() == crystal
    assert not crystal.copy().correct_surfaces


def test_a_unit_that_is_not_a_potential_is_left_as_it_is(silicon):
    built = _unit(silicon).build(lazy=False)
    assert not CrystalPotential(built, (1, 1, 2)).correct_surfaces
    forced = CrystalPotential(built, (1, 1, 2), correct_surfaces=True)
    plain = CrystalPotential(built, (1, 1, 2), correct_surfaces=False)
    np.testing.assert_array_equal(
        forced.build(lazy=False).array, plain.build(lazy=False).array
    )


@pytest.mark.parametrize("projection", ["finite", "infinite"])
def test_the_crystal_is_unchanged(silicon, projection):
    """The surfaces are corrected in the slices: not in the extent of the crystal."""
    unit = _unit(silicon, projection=projection, slice_thickness=A / 3)
    corrected = CrystalPotential(unit, (1, 1, 3), exit_planes=4, correct_surfaces=True)
    plain = CrystalPotential(unit, (1, 1, 3), exit_planes=4, correct_surfaces=False)

    assert corrected.num_slices == plain.num_slices == 3 * len(unit.slice_thickness)
    assert corrected.box == plain.box
    assert corrected.thickness == plain.thickness == pytest.approx(3 * A)
    assert corrected.slice_thickness == plain.slice_thickness
    assert corrected.exit_planes == plain.exit_planes
    assert (
        corrected.build(lazy=False).slice_thickness
        == plain.build(lazy=False).slice_thickness
    )


def test_infinite_projection_has_no_surfaces_to_correct(silicon):
    """An atom is in one slice: nothing beyond a surface reaches into it."""
    unit = _unit(silicon, projection="infinite")
    corrected = CrystalPotential(unit, (1, 1, 3), correct_surfaces=True)
    plain = CrystalPotential(unit, (1, 1, 3), correct_surfaces=False)
    np.testing.assert_array_equal(
        corrected.build(lazy=False).array, plain.build(lazy=False).array
    )


@devices
@pytest.mark.parametrize("repetitions", [1, 2, 3, 6])
def test_finite_projection_matches_a_slab(silicon, repetitions, device):
    crystal = CrystalPotential(_unit(silicon, device=device), (1, 1, repetitions))
    array = asnumpy(crystal.build(lazy=False).array)
    expected = _folded_slab(
        silicon, repetitions, 1, crystal.slice_thickness, crystal.gpts
    )

    assert array.shape == expected.shape
    np.testing.assert_allclose(array, expected, atol=1e-4 * expected.max())


@devices
def test_nonuniform_auto_slicing_matches_a_slab(zinc_oxide, device):
    unit = _unit(zinc_oxide, slice_thickness="auto", device=device)
    unit_slices = np.array(unit.slice_thickness)
    assert not np.allclose(unit_slices, unit_slices[0]), "needs a nonuniform pattern"

    crystal = CrystalPotential(unit, (2, 1, 3))
    array = asnumpy(crystal.build(lazy=False).array)
    # the atoms are sliced as they are without the correction
    np.testing.assert_allclose(crystal.slice_thickness, np.tile(unit_slices, 3))
    expected = _folded_slab(
        zinc_oxide * (2, 1, 1), 3, 1, crystal.slice_thickness, crystal.gpts
    )
    np.testing.assert_allclose(array, expected, atol=1e-4 * expected.max())


@pytest.mark.parametrize("repetitions", [2, 3, 7])
def test_a_cutoff_longer_than_the_unit_matches_a_slab(repetitions):
    """The surface slices reach several units into the crystal, and the vacuum
    that holds the tails is several units thick."""
    thin = bulk("Cu", cubic=True, a=1.9)
    unit = _unit(thin, slice_thickness=0.95)
    reach = int(np.ceil(unit._margins()[2] / unit.box[2]))
    assert reach > 1

    crystal = CrystalPotential(unit, (1, 1, repetitions))
    array = crystal.build(lazy=False).array
    expected = _folded_slab(
        thin, repetitions, reach, crystal.slice_thickness, crystal.gpts
    )
    np.testing.assert_allclose(array, expected, atol=1e-4 * expected.max())


def test_only_the_slices_at_the_surfaces_change(silicon):
    unit = _unit(silicon)
    reach = int(np.ceil(unit._margins()[2] / unit.box[2]))
    n_unit = len(unit.slice_thickness)
    corrected = CrystalPotential(unit, (1, 1, 8)).build(lazy=False).array
    plain = CrystalPotential(unit, (1, 1, 8), correct_surfaces=False)
    plain = plain.build(lazy=False).array

    n_surface = reach * n_unit
    assert np.abs(corrected[0] - plain[0]).max() > 0.1 * plain.max()
    assert np.abs(corrected[-1] - plain[-1]).max() > 0.1 * plain.max()
    np.testing.assert_array_equal(
        corrected[n_surface:-n_surface], plain[n_surface:-n_surface]
    )


def test_periodic_stack_is_not_a_slab(silicon):
    """The concern the option addresses: the periodic images of the unit beyond the
    surfaces are in the potential of the first and last slices of a stack, and the
    tails of the potential of the surface atoms beyond them are not."""
    unit = _unit(silicon)
    plain = CrystalPotential(unit, (1, 1, 3), correct_surfaces=False)
    plain = plain.build(lazy=False).array
    expected = _folded_slab(silicon, 3, 1, (A / 4,) * 12, plain.shape[1:])
    assert np.abs(plain[0] - expected[0]).max() > 0.01 * expected.max()
    assert np.abs(plain[-1] - expected[-1]).max() > 0.01 * expected.max()
    np.testing.assert_allclose(plain[3:-3], expected[3:-3], atol=1e-4 * expected.max())


@devices
@pytest.mark.parametrize("repetitions", [1, 4])
def test_builds_agree(silicon, repetitions, device):
    unit = _unit(silicon, slice_thickness=A / 3, device=device)
    crystal = CrystalPotential(unit, (2, 1, repetitions), exit_planes=3)
    eager = crystal.build(lazy=False)
    expected = asnumpy(eager.array)

    _assert_same(asnumpy(crystal.build(lazy=True).array.compute()), expected, device)
    last = crystal.num_slices - 1
    window = crystal.build(first_slice=1, last_slice=last, lazy=False).array
    _assert_same(asnumpy(window), expected[1:last], device)
    _assert_same(
        np.concatenate(
            [asnumpy(c.array) for c in crystal.generate_chunked_slices(chunk_size=4)]
        ),
        expected,
        device,
    )
    _assert_same(
        np.stack([asnumpy(s.array[0]) for s in crystal.generate_slices()]),
        expected,
        device,
    )
    assert len(eager.exit_planes) == len(crystal.exit_planes)


def test_lateral_repetitions(silicon):
    crystal = CrystalPotential(_unit(silicon), (2, 3, 2))
    array = crystal.build(lazy=False).array
    assert array.shape[1:] == tuple(crystal.gpts)
    tiles = array.reshape(array.shape[0], 2, -1, 3, array.shape[2] // 3)
    np.testing.assert_array_equal(tiles[:, 0, :, 0], tiles[:, 1, :, 2])


def test_sliced_atoms_are_those_of_the_crystal(silicon):
    corrected = CrystalPotential(_unit(silicon), (1, 1, 2))
    plain = CrystalPotential(_unit(silicon), (1, 1, 2), correct_surfaces=False)
    assert corrected.get_sliced_atoms().atoms == plain.get_sliced_atoms().atoms


def _frozen_phonon_unit(atoms, sigmas, device="cpu"):
    phonons = FrozenPhonons(atoms, num_configs=4, sigmas=sigmas, seed=1)
    return Potential(
        phonons,
        sampling=SAMPLING,
        slice_thickness=1.5,
        projection="finite",
        device=device,
    )


def test_frozen_phonon_pool_without_displacements_matches_a_slab(silicon):
    """With no displacement every configuration of the pool is the unit."""
    unit = _frozen_phonon_unit(silicon, sigmas=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal = CrystalPotential(unit, (2, 2, 2), seeds=1)
        array = crystal.build(lazy=False).array
    expected = _folded_slab(
        silicon * (2, 2, 1), 2, 1, crystal.slice_thickness, crystal.gpts
    )
    np.testing.assert_allclose(array[0], expected, atol=1e-4 * expected.max())


@devices
def test_frozen_phonon_pool_at_the_surfaces(silicon, device):
    unit = _frozen_phonon_unit(silicon, sigmas=0.08, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal = CrystalPotential(unit, (2, 2, 6), seeds=1)
        plain = CrystalPotential(unit, (2, 2, 6), seeds=1, correct_surfaces=False)
        eager = asnumpy(crystal.build(lazy=False).array)
        lazy = asnumpy(crystal.build(lazy=True).array.compute())
        plain = asnumpy(plain.build(lazy=False).array)
        chunked = np.concatenate(
            [asnumpy(c.array) for c in crystal.generate_chunked_slices(chunk_size=5)]
        )

    _assert_same(lazy, eager, device)
    _assert_same(chunked, eager[0], device)
    # the slices at the surfaces are the ones that differ
    assert not np.allclose(eager[:, 0], plain[:, 0], rtol=0.01)
    assert not np.allclose(eager[:, -1], plain[:, -1], rtol=0.01)
    _assert_same(eager[:, 6:-6], plain[:, 6:-6], device)


def test_surface_correction_moves_the_exit_wave_towards_the_slab(silicon):
    """The diffraction intensity of the crystal, with the correction, is closer to
    that of the slab with vacuum than that of the periodic stack."""
    unit = _unit(silicon)
    corrected = CrystalPotential(unit, (1, 1, 8))
    plain = CrystalPotential(unit, (1, 1, 8), correct_surfaces=False)
    entrance, crystal, exit_ = _slab(
        silicon, 8, 1, corrected.slice_thickness, corrected.gpts
    )
    thickness = (A / 4,) * (len(crystal) + len(entrance) + len(exit_))
    slab = PotentialArray(
        np.concatenate([entrance, crystal, exit_]),
        slice_thickness=thickness,
        extent=corrected.extent,
    )

    def intensity(potential):
        wave = PlaneWave(energy=100e3, sampling=SAMPLING, extent=potential.extent)
        wave = wave.multislice(potential, lazy=False)
        return np.abs(np.fft.fft2(wave.array)) ** 2

    reference = intensity(slab)

    def error(potential):
        return np.abs(intensity(potential) - reference).sum() / reference.sum()

    assert error(corrected.build(lazy=False)) < 0.5 * error(plain.build(lazy=False))


def test_a_crystal_stacked_from_a_correcting_crystal_along_z_raises(silicon):
    """Every junction between its repetitions would have the surfaces of a slab."""
    inner = CrystalPotential(_unit(silicon), (2, 2, 1))
    assert inner.correct_surfaces
    with pytest.raises(ValueError, match="correct_surfaces=False"):
        CrystalPotential(inner, (1, 1, 3))


def test_a_nested_crystal_without_inner_correction_is_the_stack(silicon):
    unit = _unit(silicon)
    inner = CrystalPotential(unit, (2, 2, 1), correct_surfaces=False)
    nested = CrystalPotential(inner, (1, 1, 6), correct_surfaces=False)
    flat = CrystalPotential(unit, (2, 2, 6), correct_surfaces=False)
    np.testing.assert_array_equal(
        nested.build(lazy=False).array, flat.build(lazy=False).array
    )


def test_a_nested_crystal_tiled_laterally_keeps_the_correction(silicon):
    """Tiling a crystal that is tall along z does not put surfaces inside it."""
    unit = _unit(silicon)
    inner = CrystalPotential(unit, (1, 1, 3))
    nested = CrystalPotential(inner, (2, 2, 1))
    flat = CrystalPotential(unit, (2, 2, 3))
    np.testing.assert_allclose(
        nested.build(lazy=False).array,
        flat.build(lazy=False).array,
        atol=1e-6 * flat.build(lazy=False).array.max(),
    )


def test_a_unit_without_atoms_has_no_surfaces_to_correct():
    empty = Atoms(cell=[4, 4, 4], pbc=True)
    unit = Potential(empty, sampling=SAMPLING, slice_thickness=2.0, projection="finite")
    array = CrystalPotential(unit, (1, 1, 3)).build(lazy=False).array
    assert array.shape[0] == 6
    assert not array.any()


def _near_surfaces(atoms, shift):
    atoms = atoms.copy()
    atoms.positions[:, 2] += shift
    atoms.wrap()
    return atoms


@pytest.mark.filterwarnings("error:.*atom.* of the potential unit are within")
def test_atoms_centered_in_the_cell_are_not_warned_about(silicon):
    atoms = silicon.copy()
    atoms.center(axis=2)
    CrystalPotential(_unit(atoms), (1, 1, 2))


@pytest.mark.parametrize("shift", [0.0, 0.1, -0.1, 0.29])
def test_atoms_near_the_entrance_or_exit_plane_are_warned_about(silicon, shift):
    atoms = _near_surfaces(silicon, shift)
    with pytest.warns(UserWarning, match=r"atoms.center\(axis=2\)"):
        CrystalPotential(_unit(atoms), (1, 1, 2))


@pytest.mark.filterwarnings("error:.*atom.* of the potential unit are within")
@pytest.mark.parametrize("shift", [0.31, 0.5, -0.31])
def test_atoms_farther_than_the_distance_are_not_warned_about(silicon, shift):
    CrystalPotential(_unit(_near_surfaces(silicon, shift)), (1, 1, 2))


@pytest.mark.filterwarnings("error:.*atom.* of the potential unit are within")
def test_no_warning_when_there_is_no_correction_to_be_poor(silicon):
    CrystalPotential(_unit(silicon), (1, 1, 2), correct_surfaces=False)
    CrystalPotential(_unit(silicon, projection="infinite"), (1, 1, 2))
    CrystalPotential(_unit(silicon).build(lazy=False), (1, 1, 2))


def test_the_warning_is_given_once_not_by_every_lazy_block(silicon):
    with pytest.warns(UserWarning, match="within"):
        crystal = CrystalPotential(_unit(silicon), (1, 1, 2), seeds=(1, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crystal.build(lazy=True).array.compute()
