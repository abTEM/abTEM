"""A CrystalPotential without `seeds` is one crystal per simulation: every lazy
scan block, plane-wave batch and potential chunk of one computation, and every
build of one object, draws the same mosaic of frozen-phonon configurations."""

import copy
import pickle

import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem import CrystalPotential, PlaneWave, Potential


def _frozen_phonon_crystal(num_configs, repetitions, unit_seed=1, **kwargs):
    atoms = bulk("Si", cubic=True)
    unit = Potential(
        abtem.FrozenPhonons(atoms, num_configs, sigmas=0.1, seed=unit_seed),
        gpts=(16, 16),
        slice_thickness=atoms.cell[2, 2] / 4,  # 4 slices per unit
    )
    return CrystalPotential(unit, repetitions, **kwargs)


def _crystal_of_root(root):
    crystal = _frozen_phonon_crystal(
        8, (2, 3, 3), seeds=(5,) if root == "seeds" else None
    )
    if root == "built ensemble":
        crystal = CrystalPotential(
            crystal.potential_unit.build(lazy=False), crystal.repetitions
        )
    elif root == "lazy built ensemble":
        crystal = CrystalPotential(
            crystal.potential_unit.build(lazy=True), crystal.repetitions
        )
    return crystal


ROOTS = ["seeds", "frozen phonons", "built ensemble", "lazy built ensemble"]


@pytest.mark.parametrize("root", ROOTS)
def test_a_crystal_is_one_crystal_in_every_scan_block(root):
    """Identical probe positions in different lazy blocks see one crystal,
    whichever of the crystal's seeds, the unit's frozen phonons, or a seed
    drawn at construction sets its mosaic."""
    crystal = _crystal_of_root(root)
    scan = abtem.CustomScan(np.array([[1.0, 1.0]] * 4))

    measurement = abtem.Probe(energy=100e3, semiangle_cutoff=20).scan(
        crystal,
        scan=scan,
        detectors=abtem.AnnularDetector(10, 30),
        lazy=True,
        max_batch=1,
    )
    assert measurement.array.numblocks[-1] == 4
    values = np.asarray(measurement.compute(scheduler="synchronous").array)

    assert values.shape == (4,)
    np.testing.assert_allclose(
        values, values[0], rtol=0, atol=1e-6 * np.abs(values).max()
    )


@pytest.mark.parametrize("root", ROOTS)
def test_every_build_of_a_crystal_is_the_same_crystal(root):
    crystal = _crystal_of_root(root)
    expected = crystal.build(lazy=False).array

    np.testing.assert_array_equal(crystal.build(lazy=False).array, expected)
    np.testing.assert_array_equal(
        crystal.build(lazy=True).compute(scheduler="synchronous").array, expected
    )


@pytest.mark.parametrize("root", ROOTS)
def test_chunks_of_the_slices_are_chunks_of_one_crystal(root):
    """Slices generated in two windows are those of the whole crystal."""
    crystal = _crystal_of_root(root)
    whole = [np.asarray(s.array) for s in crystal.generate_slices()]
    split = [np.asarray(s.array) for s in crystal.generate_slices(0, 5)] + [
        np.asarray(s.array) for s in crystal.generate_slices(5)
    ]

    assert len(split) == len(whole) == crystal.num_slices
    for a, b in zip(split, whole):
        np.testing.assert_array_equal(a, b)


def test_the_plane_wave_batches_of_an_s_matrix_see_one_crystal():
    """A lazy S-matrix split into several plane-wave batches simulates the
    crystal that one batch does."""
    crystal = _frozen_phonon_crystal(2, (1, 2, 2))
    s_matrix = abtem.SMatrix(
        potential=crystal, energy=100e3, semiangle_cutoff=10, interpolation=1
    )
    assert len(s_matrix) > 4

    def reduce(**kwargs):
        measurement = s_matrix.reduce(
            scan=abtem.GridScan(gpts=(2, 3)),
            detectors=abtem.AnnularDetector(0, 20),
            lazy=True,
            **kwargs,
        )
        return np.asarray(measurement.compute(scheduler="synchronous").array)

    expected = reduce()
    batched = reduce(max_batch_multislice=4)

    np.testing.assert_allclose(
        batched, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_an_unseeded_crystal_backscatters_as_it_builds(lazy):
    """The backward pass of full-expansion backscattering, which calls
    generate_slices again, sees the crystal of the forward pass."""
    from abtem.multislice import RealSpaceMultislice

    def crystal():
        return _frozen_phonon_crystal(6, (2, 3, 2), exit_planes=1)

    def run(potential):
        result = PlaneWave(energy=100e3).multislice(
            potential,
            lazy=lazy,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
            return_backscattered=True,
        )
        if lazy:
            result = [r.compute(scheduler="synchronous") for r in result]
        return [np.asarray(r.array) for r in result]

    expected = run(crystal().build(lazy=False))
    result = run(crystal())

    assert len(result) == len(expected) == 2
    for r, e in zip(result, expected):
        assert r.shape == e.shape
        np.testing.assert_allclose(r, e, rtol=0, atol=1e-6 * np.abs(e).max())


@pytest.mark.parametrize(
    "num_configs, repetitions, kwargs, expected",
    [
        (
            6,
            (2, 3, 2),
            dict(seeds=(5, 6)),
            [
                [[[1, 4, 2], [3, 5, 0]], [[4, 0, 1], [3, 2, 5]]],
                [[[2, 3, 0], [5, 4, 1]], [[4, 0, 2], [1, 5, 3]]],
            ],
        ),
        (
            12,
            (2, 3, 2),
            dict(num_frozen_phonons=2, seeds=11),
            [
                [[[7, 11, 5], [8, 2, 3]], [[4, 10, 1], [9, 6, 0]]],
                [[[0, 10, 2], [6, 3, 9]], [[7, 11, 5], [8, 4, 1]]],
            ],
        ),
        (
            4,
            (3, 1, 3),
            dict(seeds=(7,)),
            [[[[0], [3], [1]], [[0], [2], [3]], [[1], [2], [0]]]],
        ),
    ],
    ids=["seeds", "master seed", "one seed"],
)
def test_explicit_seeds_draw_the_same_mosaic(
    num_configs, repetitions, kwargs, expected
):
    """The pool configuration of every tile, pinned for explicit seeds (the
    values of 1.0.11)."""
    crystal = _frozen_phonon_crystal(num_configs, repetitions, **kwargs)
    np.testing.assert_array_equal(_drawn_tiles(crystal), expected)


def test_a_seeded_unit_draws_a_pinned_mosaic():
    """Without seeds on the crystal, the seed of the unit's frozen phonons
    fixes the pool configuration of every tile, in every session."""
    crystal = _frozen_phonon_crystal(6, (2, 3, 2))
    np.testing.assert_array_equal(
        _drawn_tiles(crystal), [[[[5, 4, 2], [0, 1, 3]], [[5, 2, 0], [3, 1, 4]]]]
    )


def test_two_crystals_of_one_built_ensemble_are_different_crystals():
    """A crystal of a unit without frozen phonons draws its seed when it is
    created: a copy is the same crystal, another construction is not."""
    from dask.base import tokenize

    unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit.build(lazy=False)
    a = CrystalPotential(unit, (2, 3, 2))
    b = CrystalPotential(unit, (2, 3, 2))

    assert a == a.copy()
    assert tokenize(a) == tokenize(a.copy())
    assert a != b
    assert tokenize(a) != tokenize(b)
    assert not np.array_equal(a.build(lazy=False).array, b.build(lazy=False).array)


def test_two_crystals_of_one_frozen_phonon_unit_are_one_crystal():
    unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit
    a = CrystalPotential(unit, (2, 3, 2))
    b = CrystalPotential(unit, (2, 3, 2))

    assert a == b
    np.testing.assert_array_equal(a.build(lazy=False).array, b.build(lazy=False).array)


@pytest.mark.parametrize("seeds", [None, (5, 6)], ids=["unseeded", "seeded"])
def test_a_crystal_whose_pickle_lacks_the_root_seed_equals_a_fresh_one(seeds):
    """A crystal pickled before it kept a root seed."""
    fresh = _frozen_phonon_crystal(4, (2, 2, 1), seeds=seeds)
    stripped = copy.copy(fresh)
    del stripped.__dict__["_root_seed"]
    loaded = pickle.loads(pickle.dumps(stripped))

    eager = loaded.build(lazy=False).array

    assert loaded == fresh
    assert fresh == loaded
    np.testing.assert_array_equal(loaded.build(lazy=False).array, eager)
    np.testing.assert_array_equal(loaded.build(lazy=True).compute().array, eager)
    np.testing.assert_array_equal(fresh.build(lazy=False).array, eager)


def _drawn_tiles(crystal):
    """The pool configuration of every lateral tile of every z-repetition of
    the built crystal, found by exact match against the member's pool; axes:
    member, z-repetition, repetitions[0], repetitions[1]."""
    repetitions = crystal.repetitions
    built = crystal.build(lazy=False).array
    if crystal.seeds is None:
        built, members = built[None], [None]
    else:
        members = [int(seed) for seed in crystal.seeds]

    drawn = np.empty((len(members), repetitions[2]) + repetitions[:2], dtype=int)
    for m, seed in enumerate(members):
        pool = crystal._pool_unit_for_member(seed).build(lazy=False).array
        n_sub, uy, ux = pool.shape[1:]
        for i in range(repetitions[2]):
            layer = built[m, n_sub * i : n_sub * (i + 1)].reshape(
                n_sub, repetitions[0], uy, repetitions[1], ux
            )
            for a in range(repetitions[0]):
                for b in range(repetitions[1]):
                    d = np.abs(pool - layer[None, :, a, :, b, :]).max(axis=(1, 2, 3))
                    assert d.min() == 0
                    drawn[m, i, a, b] = np.argmin(d)
    return drawn
