"""Sheared cells whose diagonal equals their best orthogonal box are not noise."""

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene

import abtem
from abtem.atoms import best_orthogonal_cell, orthogonalize_cell

GRID = dict(sampling=0.1, slice_thickness=1.0)


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64", "fft": "numpy"}):
        yield


def _bn():
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    return atoms


def _two_atoms(cell):
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=cell,
        pbc=True,
    )


def _noisy_orthorhombic():
    return _two_atoms([[4, 0, 0], [1e-9, 3, 0], [0, 0, 5]])


# Rows: name, atoms, whether the diagonal of the cell equals its best orthogonal
# box (the cells the shortcut of orthogonalize_cell took for orthogonal ones).
def _cases():
    return [
        ("BN x (1, 2, 1)", _bn() * (1, 2, 1), True),
        ("BN x (1, 4, 1)", _bn() * (1, 4, 1), True),
        ("BN x (2, 4, 1)", _bn() * (2, 4, 1), True),
        ("graphene x (1, 2, 1)", graphene(vacuum=2.0) * (1, 2, 1), True),
        ("hcp Mg x (1, 2, 1)", bulk("Mg") * (1, 2, 1), True),
        (
            "monoclinic c = (4, 0, 5)",
            _two_atoms([[4, 0, 0], [0, 3, 0], [4, 0, 5]]),
            True,
        ),
        (
            "monoclinic c = (-8, 0, 5)",
            _two_atoms([[4, 0, 0], [0, 3, 0], [-8, 0, 5]]),
            True,
        ),
        ("sheared b = (4, 3, 0)", _two_atoms([[4, 0, 0], [4, 3, 0], [0, 0, 5]]), True),
        # Guards: the diagonal is not the best orthogonal box.
        ("BN", _bn(), False),
        ("BN x (2, 3, 2)", _bn() * (2, 3, 2), False),
        (
            "monoclinic c = (2, 0, 5)",
            _two_atoms([[4, 0, 0], [0, 3, 0], [2, 0, 5]]),
            False,
        ),
    ]


def _by_hand(atoms, box, repetitions=6):
    """The atoms written into the orthogonal box without an abTEM transform: the
    crystal repeated, shifted by a lattice translation, wrapped, and each atom
    kept once."""
    nz = 1 if atoms.cell[2, 0] == 0 and atoms.cell[2, 1] == 0 else repetitions
    big = atoms * (repetitions, repetitions, nz)
    big.positions -= (repetitions // 2) * (atoms.cell[0] + atoms.cell[1]) + (
        nz // 2
    ) * atoms.cell[2]
    out = Atoms(big.numbers, positions=big.positions, cell=np.diag(box), pbc=True)
    out.wrap()
    scaled = np.round(out.get_scaled_positions() % 1.0, 8) % 1.0
    _, keep = np.unique(np.c_[scaled, out.numbers], axis=0, return_index=True)
    return out[np.sort(keep)]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_cell_with_its_diagonal_as_the_orthogonal_box(case):
    name, atoms, diagonal_is_the_box = case
    box = tuple(best_orthogonal_cell(atoms.cell))
    assert (tuple(np.diag(atoms.cell)) == box) == diagonal_is_the_box

    orthogonal = orthogonalize_cell(atoms)

    assert orthogonal.cell.lengths() == pytest.approx(box)
    assert np.allclose(np.array(orthogonal.cell), np.diag(box))
    expected = len(atoms) * abs(np.prod(box) / np.linalg.det(atoms.cell))
    assert len(orthogonal) == round(expected)
    assert expected == pytest.approx(round(expected))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_potential_of_a_cell_with_its_diagonal_as_the_orthogonal_box(case):
    name, atoms, _ = case
    potential = abtem.Potential(atoms, **GRID)
    oracle = _by_hand(atoms, potential.box)
    assert len(potential.get_transformed_atoms()) == len(oracle)

    actual = potential.build(lazy=False).array
    expected = abtem.Potential(oracle, **GRID).build(lazy=False).array
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    "atoms",
    [_bn() * (1, 2, 1), _two_atoms([[4, 0, 0], [0, 3, 0], [4, 0, 5]])],
    ids=["BN x (1, 2, 1)", "monoclinic c = (4, 0, 5)"],
)
def test_automatic_sampling_of_a_sheared_cell(atoms):
    potential = abtem.Potential(atoms, sampling="auto")
    oracle = _by_hand(atoms, potential.box)
    # The grid that fits the atoms as the orthogonal cell holds them.
    assert potential.gpts == abtem.Potential(oracle, sampling="auto").gpts


def test_noise_in_the_off_diagonal_components_is_still_removed():
    atoms = _noisy_orthorhombic()
    assert tuple(np.diag(atoms.cell)) == tuple(best_orthogonal_cell(atoms.cell))

    orthogonal = orthogonalize_cell(atoms)

    assert np.array(orthogonal.cell).tolist() == np.diag([4.0, 3.0, 5.0]).tolist()
    assert len(orthogonal) == 2


def _sheared(base, k):
    """`base` with its lattice vectors sheared by the integer matrix [[1, 0, 0],
    [k0, 1, 0], [k1, k2, 1]]: the same crystal in a skewed cell."""
    shear = np.array([[1, 0, 0], [k[0], 1, 0], [k[1], k[2], 1]])
    atoms = Atoms(
        base.numbers,
        positions=base.positions,
        cell=shear @ np.array(base.cell),
        pbc=True,
    )
    atoms.wrap()
    return atoms


def _rectangular_bn():
    return orthogonalize_cell(_bn())


def test_atom_on_a_cell_boundary_by_round_off_is_not_dropped():
    # The atom at the origin of the rectangular BN cell sits at a scaled
    # coordinate of -3.6e-16 of this sheared cell, which `ase.build.cut` wraps to
    # just below 1 and then misses.
    atoms = _sheared(_rectangular_bn(), (-3, -3, 1))
    box = tuple(best_orthogonal_cell(atoms.cell))
    assert tuple(np.diag(atoms.cell)) == box
    assert len(orthogonalize_cell(atoms)) == len(atoms)

    potential = abtem.Potential(atoms, **GRID)
    oracle = _by_hand(atoms, potential.box)
    actual = potential.build(lazy=False).array
    expected = abtem.Potential(oracle, **GRID).build(lazy=False).array
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    "base",
    [
        _rectangular_bn(),
        _two_atoms([[4, 0, 0], [0, 3, 0], [0, 0, 5]]),
    ],
    ids=["rectangular BN", "orthorhombic CO"],
)
def test_no_atom_is_dropped_from_any_sheared_cell_of_a_crystal(base):
    # Every sheared cell of the crystal, skewed by small integer matrices, whose
    # diagonal is its best orthogonal box: each atom of the input is in the
    # orthogonal cell once.
    checked = 0
    for k in itertools.product(range(-3, 4), repeat=3):
        if not any(k):
            continue
        atoms = _sheared(base, k)
        if tuple(np.diag(atoms.cell)) != tuple(best_orthogonal_cell(atoms.cell)):
            continue
        assert len(orthogonalize_cell(atoms)) == len(atoms), k
        checked += 1
    assert checked > 200
