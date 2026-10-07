"""The supercell that orthogonalize_cell cuts holds every atom exactly once."""

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, cut, graphene
from scipy.spatial import cKDTree

import abtem
import abtem.atoms
from abtem.atoms import best_orthogonal_cell, orthogonalize_cell


def _two_atoms(cell=(4.0, 3.0, 5.0)):
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=cell,
        pbc=True,
    )


def _rattled(base, seed, sigma=0.05):
    """An MD-like snapshot of a crystal: Gaussian displacements, wrapped."""
    atoms = base.copy()
    atoms.pbc = True
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, sigma, atoms.positions.shape)
    atoms.wrap()
    return atoms


def _expected_number_of_atoms(atoms, box):
    return round(len(atoms) * np.prod(box) / abs(np.linalg.det(np.array(atoms.cell))))


def _images_by_enumeration(atoms, vectors):
    """The images of every atom inside the supercell, from the enumeration of
    all lattice translations: for each atom, its scaled positions in the
    supercell."""
    cell = np.array(atoms.cell)
    newcell = vectors @ cell
    inverse = np.linalg.inv(newcell)
    reach = int(np.abs(vectors).sum(axis=0).max()) + 2
    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)))
    images = []
    for position in atoms.positions:
        scaled = (position + shifts @ cell) @ inverse
        inside = np.all((scaled > -1e-7) & (scaled < 1.0 - 1e-7), axis=1)
        images.append(scaled[inside])
    return images


def _images_of(supercell, num_atoms):
    """The same, read from a supercell whose atoms carry their source index."""
    scaled = supercell.positions @ np.linalg.inv(np.array(supercell.cell))
    source = supercell.get_array("source")
    return [scaled[source == i] for i in range(num_atoms)]


def _assert_same_images(actual, expected):
    """Equal as sets of points in the periodic supercell, atom by atom."""
    assert len(actual) == len(expected)
    for points, reference in zip(actual, expected):
        assert len(points) == len(reference)
        tree = cKDTree(np.mod(reference, 1.0) % 1.0, boxsize=1.0 + 1e-12)
        distance, nearest = tree.query(np.mod(points, 1.0) % 1.0)
        assert distance.max() < 1e-6
        assert len(set(nearest)) == len(reference)


def _with_source(atoms):
    atoms = atoms.copy()
    atoms.set_array("source", np.arange(len(atoms)))
    return atoms


@pytest.mark.parametrize(
    "name, base, seed",
    [
        ("graphene x (12, 12, 1)", graphene(vacuum=2.0) * (12, 12, 1), 1),
        ("graphene x (12, 12, 1)", graphene(vacuum=2.0) * (12, 12, 1), 2),
        ("graphene x (30, 30, 1)", graphene(vacuum=2.0) * (30, 30, 1), 1),
        ("hcp Mg x (6, 6, 4)", bulk("Mg") * (6, 6, 4), 1),
        ("hcp Mg x (6, 6, 4)", bulk("Mg") * (6, 6, 4), 2),
        ("hcp Mg x (10, 10, 6)", bulk("Mg") * (10, 10, 6), 1),
        (
            "BN x (10, 10, 1)",
            graphene(formula="BN", a=2.5, vacuum=2.0) * (10, 10, 1),
            1,
        ),
        (
            "BN x (20, 20, 1)",
            graphene(formula="BN", a=2.5, vacuum=2.0) * (20, 20, 1),
            1,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_rattled_hexagonal_supercell_keeps_every_atom(name, base, seed):
    # Atoms within 0.1 % of the cell of an upper face of the orthogonal cell are
    # many in a cell of a few nanometres, and ase.build.cut loses them.
    atoms = _rattled(base, seed)
    box = tuple(best_orthogonal_cell(np.array(atoms.cell)))

    orthogonal = orthogonalize_cell(atoms)

    assert len(orthogonal) == _expected_number_of_atoms(atoms, box)
    assert np.allclose(np.array(orthogonal.cell), np.diag(box))


def test_potential_of_a_rattled_hexagonal_supercell_keeps_every_atom():
    atoms = _rattled(graphene(vacuum=2.0) * (12, 12, 1), 1)
    potential = abtem.Potential(atoms, sampling=0.2)

    transformed = potential.get_transformed_atoms()

    assert len(transformed) == _expected_number_of_atoms(atoms, potential.box)


@pytest.mark.parametrize(
    "num_atoms, length, repetitions",
    [(200, 15.0, (2, 2, 1)), (500, 15.0, (3, 3, 1)), (1000, 20.0, (3, 3, 1))],
)
def test_large_random_cell_with_a_box_keeps_every_atom(num_atoms, length, repetitions):
    rng = np.random.default_rng(0)
    atoms = Atoms("C" * num_atoms, cell=[length] * 3, pbc=True)
    box = tuple(length * np.array(repetitions, dtype=float))
    for _ in range(3):
        atoms.positions = rng.uniform(0.0, length, (num_atoms, 3))
        orthogonal = orthogonalize_cell(atoms, box=box)
        assert len(orthogonal) == num_atoms * int(np.prod(repetitions))


@pytest.mark.parametrize("fraction", [0.9960, 0.9972, 0.9990, 0.0005])
@pytest.mark.parametrize("periods", [2, 3, 10])
def test_atom_close_to_an_upper_face_is_kept(fraction, periods):
    # 0.9972 of a 4 A axis is 0.0112 A below the face: outside the 0.01 A that
    # orthogonalize_cell snaps, inside the 0.1 % band of the supercell for a box
    # of 12 A.
    atoms = Atoms("C", positions=[(4.0 * fraction, 1.0, 1.0)], cell=[4.0, 3.0, 5.0])
    atoms.pbc = True
    orthogonal = orthogonalize_cell(atoms, box=(4.0 * periods, 3.0, 5.0))
    assert len(orthogonal) == periods


def test_plane_box_and_origin_together_keep_every_atom():
    atoms = _two_atoms()
    orthogonal = orthogonalize_cell(
        atoms, plane="xz", box=(8.0, 10.0, 6.0), origin=(1.0, 0.5, 0.25)
    )
    # The rotated cell is 4 x 5 x 3 A: the box holds 2 x 2 x 2 cells of 2 atoms.
    assert len(orthogonal) == 16


def _random_sheared_cases(count=40, seed=9):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(count):
        n = int(rng.integers(5, 40))
        lengths = rng.uniform(3.0, 12.0, 3)
        cell = np.diag(lengths)
        cell[1, 0] = rng.uniform(-lengths[0], lengths[0]) * rng.integers(0, 2)
        cell[2, :2] = rng.uniform(-2.0, 2.0, 2) * rng.integers(0, 2)
        atoms = Atoms("C" * n, cell=cell, pbc=True)
        atoms.set_scaled_positions(rng.uniform(0.0, 1.0, (n, 3)))
        vectors = np.diag(rng.integers(1, 4, 3)).astype(float)
        vectors[1, 0] = rng.integers(-2, 3)
        vectors[2, 1] = rng.integers(-1, 2)
        cases.append((atoms, vectors))
    return cases


@pytest.mark.parametrize("case", range(40))
def test_supercell_equals_the_enumeration_of_its_images(case):
    atoms, vectors = _random_sheared_cases()[case]
    supercell = abtem.atoms._cut_supercell(_with_source(atoms), vectors, tolerance=0.01)

    assert len(supercell) == len(atoms) * round(abs(np.linalg.det(vectors)))
    assert np.allclose(np.array(supercell.cell), vectors @ np.array(atoms.cell))
    _assert_same_images(
        _images_of(supercell, len(atoms)), _images_by_enumeration(atoms, vectors)
    )


def test_supercell_of_a_rattled_crystal_equals_the_enumeration_of_its_images():
    atoms = _with_source(_rattled(graphene(vacuum=2.0) * (6, 6, 1), 1))
    box = best_orthogonal_cell(np.array(atoms.cell))
    vectors = np.round(np.diag(box) @ np.linalg.inv(np.array(atoms.cell)))

    supercell = abtem.atoms._cut_supercell(atoms, vectors, tolerance=0.01)

    _assert_same_images(
        _images_of(supercell, len(atoms)), _images_by_enumeration(atoms, vectors)
    )


def test_cut_that_was_right_is_returned_unchanged():
    atoms = _two_atoms()
    vectors = np.diag([2.0, 3.0, 2.0])
    reference = cut(atoms, a=vectors[0], b=vectors[1], c=vectors[2], tolerance=0.01)

    supercell = abtem.atoms._cut_supercell(atoms, vectors, tolerance=0.01)

    assert np.array_equal(supercell.positions, reference.positions)
    assert np.array_equal(supercell.numbers, reference.numbers)


def test_per_atom_arrays_survive_the_rebuilt_supercell():
    # The atom at 0.9972 of the axis is lost by ase.build.cut, so the supercell
    # is rebuilt; the arrays of the atoms are carried to every image.
    atoms = Atoms(
        "CO",
        positions=[(4.0 * 0.9972, 1.0, 1.0), (1.0, 2.0, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )
    atoms.set_array("source", np.array([7, 8]))
    atoms.set_array("moment", np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]))
    vectors = np.diag([3.0, 1.0, 1.0])
    assert len(cut(atoms, a=vectors[0], b=vectors[1], c=vectors[2])) != 6

    supercell = abtem.atoms._cut_supercell(atoms, vectors, tolerance=0.01)

    assert len(supercell) == 6
    assert sorted(supercell.get_array("source")) == [7, 7, 7, 8, 8, 8]
    assert np.array_equal(
        supercell.get_array("moment")[supercell.get_array("source") == 8][:, 2],
        [2.0, 2.0, 2.0],
    )
    assert list(supercell.numbers).count(6) == 3


# A hexagonal cell and a supercell of twice its volume, sheared in the xy plane.
_HEX_VECTORS = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]])


def _single_atom(scaled, cell):
    return Atoms("C", scaled_positions=[scaled], cell=cell, pbc=True)


def _assert_images_of_one_atom(atoms, vectors):
    supercell = abtem.atoms._wrapped_supercell(atoms, vectors)
    expected = int(round(abs(np.linalg.det(vectors))))

    assert len(supercell) == expected
    scaled = supercell.positions @ np.linalg.inv(np.array(supercell.cell))
    _assert_same_images([scaled], _images_by_enumeration(atoms, vectors))


@pytest.mark.parametrize("distance", [1e-9, 1.0000001e-9, 5e-10])
@pytest.mark.parametrize("scaled_xy", [(0.3, 0.2), (0.5, 0.5), (0.0, 0.0)])
def test_atom_within_round_off_of_a_face_keeps_all_its_images(distance, scaled_xy):
    # 1 - 1e-9 is on the threshold at which a fractional coordinate counts as 1,
    # so the copies of one image, which differ by round-off, fall on both sides.
    atoms = _single_atom((*scaled_xy, 1.0 - distance), graphene(vacuum=2).cell)

    _assert_images_of_one_atom(atoms, _HEX_VECTORS)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_images_at_a_rounding_tie_of_the_coordinates_are_not_counted_twice(seed):
    # The scaled coordinates in the supercell are n * 1e-8 + 5e-9, which is a tie
    # of the rounding to 8 decimals, plus 3e-16 of round-off.
    cell = graphene(vacuum=2).cell
    newcell = _HEX_VECTORS @ np.array(cell)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        scaled = np.round(rng.random(3), 8) + 5e-9 + rng.uniform(-3e-16, 3e-16, 3)
        atoms = _single_atom((0.0, 0.0, 0.5), cell)
        atoms.positions[:] = scaled @ newcell

        _assert_images_of_one_atom(atoms, _HEX_VECTORS)


@pytest.mark.parametrize("length", [4.0, 40.0, 400.0])
@pytest.mark.parametrize("distance", [2e-9, 5e-9])
def test_atom_a_few_1e_9_of_a_period_from_a_face_keeps_all_its_images(length, distance):
    cell = np.diag([length] * 3)
    for y in np.linspace(0.1, 0.9, 50):
        atoms = _single_atom((1.0 - distance, y, 0.37), cell)

        _assert_images_of_one_atom(atoms, _HEX_VECTORS)
