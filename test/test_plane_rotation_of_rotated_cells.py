"""Cells whose lattice is rotated about the axis that becomes the beam direction.

A rectangular 4.0 x 4.6 x 5.3 A cell, rotated by an angle about the axis that
becomes the beam direction of the plane, simulated with plane "xz", "yz" or "zx":
standardize_cell undoes the rotation, so the exact result is the unrotated atoms
with their axes permuted to the plane, in the cell with the lengths permuted the
same way. The three lengths differ, so the box of each plane differs from the
box of every other permutation of the axes. That holds for any number of atoms
and any angle at which the first lattice vector is the in-plane vector most
aligned with x (|cos| > |sin|).
"""

import warnings

import ase
import numpy as np
import pytest

import abtem
from abtem.atoms import best_orthogonal_cell, cut_cell, rotate_atoms_to_plane

CELL = (4.0, 4.6, 5.3)
# plane: the axis that becomes the beam direction, and the permutation of the axes
PLANES = {"xz": ("y", (0, 2, 1)), "yz": ("x", (1, 2, 0)), "zx": ("y", (2, 0, 1))}
ANGLES = [30.0, 150.0]  # 150: the diagonal of the permuted cell is (-, -, +)
NUM_ATOMS = [2, 3, 5]  # 3 is the only count dev does not raise for


def _unrotated(num_atoms):
    rng = np.random.default_rng(num_atoms)
    return ase.Atoms(
        "Si" * num_atoms,
        scaled_positions=rng.uniform(0.05, 0.95, (num_atoms, 3)),
        cell=CELL,
        pbc=True,
    )


def _rotated(num_atoms, angle, plane):
    atoms = _unrotated(num_atoms)
    atoms.rotate(angle, PLANES[plane][0], rotate_cell=True)
    return atoms


def _lengths(plane):
    return np.array(CELL)[list(PLANES[plane][1])]


def _assert_is_the_unrotated_atoms_in_plane(transformed, num_atoms, plane):
    lengths = _lengths(plane)
    np.testing.assert_allclose(transformed.cell[:], np.diag(lengths), rtol=0, atol=1e-12)
    assert len(transformed) == num_atoms
    expected = _unrotated(num_atoms).positions[:, list(PLANES[plane][1])]
    difference = transformed.positions - expected
    difference -= lengths * np.round(difference / lengths)
    # positions are a few A; a rotation by round-off moves them by ~1e-15 A
    np.testing.assert_allclose(difference, 0.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("plane", list(PLANES))
@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("num_atoms", NUM_ATOMS)
def test_rotate_atoms_to_plane_undoes_a_rotation_about_the_beam(
    num_atoms, angle, plane
):
    rotated = rotate_atoms_to_plane(_rotated(num_atoms, angle, plane), plane)
    _assert_is_the_unrotated_atoms_in_plane(rotated, num_atoms, plane)


@pytest.mark.parametrize("num_atoms", [2, 3])
def test_standardize_cell_keeps_atoms_of_a_lattice_vector_along_minus_y(num_atoms):
    """plane="xy": the second lattice vector points along -y. Making it positive
    is a change of lattice basis, not a move of any atom."""
    rng = np.random.default_rng(7)
    unrotated = ase.Atoms(
        "Si" * num_atoms,
        positions=rng.uniform(0.2, 3.8, (num_atoms, 3)) * (1, -1, 1),
        cell=[[CELL[0], 0.0, 0.0], [0.0, -CELL[1], 0.0], [0.0, 0.0, CELL[2]]],
        pbc=True,
    )
    atoms = unrotated.copy()
    atoms.rotate(30, "z", rotate_cell=True)

    standardized = abtem.standardize_cell(atoms)

    lengths = np.array(CELL)
    np.testing.assert_allclose(standardized.cell[:], np.diag(lengths), rtol=0, atol=1e-12)
    difference = standardized.positions - unrotated.positions
    difference -= lengths * np.round(difference / lengths)
    np.testing.assert_allclose(difference, 0.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("plane", list(PLANES))
@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("num_atoms", NUM_ATOMS)
@pytest.mark.parametrize("box", [None, "lengths"], ids=["default_box", "box"])
def test_potential_of_a_cell_rotated_about_the_beam(num_atoms, angle, plane, box):
    """No strain is needed, so the construction reports none."""
    box = None if box is None else tuple(_lengths(plane))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        potential = abtem.Potential(
            _rotated(num_atoms, angle, plane), sampling=0.1, plane=plane, box=box
        )
    assert potential.box == pytest.approx(tuple(_lengths(plane)), rel=0, abs=1e-12)
    _assert_is_the_unrotated_atoms_in_plane(
        potential.get_transformed_atoms(), num_atoms, plane
    )


def _rotated_about_x(num_atoms, angle):
    atoms = _unrotated(num_atoms)
    atoms.rotate(angle, "x", rotate_cell=True)
    return atoms


CELLS_IN_ANOTHER_PLANE = {
    "cubic_xz": (ase.build.bulk("Si", cubic=True), "xz"),
    "cubic_yz": (ase.build.bulk("Si", cubic=True), "yz"),
    "orthorhombic_xz": (_unrotated(2), "xz"),
    "orthorhombic_yz": (_unrotated(2), "yz"),
    "orthorhombic_zx": (_unrotated(2), "zx"),
    "rotated_about_y_xz": (_rotated(3, 30.0, "xz"), "xz"),
    "rotated_about_x_yz": (_rotated_about_x(3, -20.0), "yz"),
    "rotated_about_y_zx": (_rotated(3, 30.0, "zx"), "zx"),
}


@pytest.mark.parametrize("name", list(CELLS_IN_ANOTHER_PLANE))
def test_default_box_is_orthogonalize_cells_own(name):
    """Oracle: orthogonalize_cell with no box rotates the atoms to the plane, then
    chooses its box from the rotated cell."""
    atoms, plane = CELLS_IN_ANOTHER_PLANE[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = tuple(np.diag(abtem.orthogonalize_cell(atoms, plane=plane).cell))
    assert abtem.Potential(atoms, sampling=0.1, plane=plane).box == pytest.approx(
        expected, rel=0, abs=1e-12
    )


@pytest.mark.parametrize("plane", list(PLANES))
def test_default_box_of_a_rectangle_is_its_lengths_permuted_to_the_plane(plane):
    """The three lengths differ, so a box with two axes exchanged is not this one."""
    box = abtem.Potential(_unrotated(2), sampling=0.1, plane=plane).box
    assert box == pytest.approx(tuple(_lengths(plane)), rel=0, abs=1e-12)


@pytest.mark.parametrize(
    "atoms",
    [
        ase.build.mx2("WSe2", vacuum=2),
        ase.Atoms("Au", cell=[[4.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 5.0]], pbc=True),
    ],
    ids=["hexagonal", "sheared"],
)
def test_default_box_in_plane_xy_is_unchanged(atoms):
    """Guard: plane="xy" never rotates the cell, so its box is best_orthogonal_cell's."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        box = abtem.Potential(atoms, sampling=0.1).box
    assert box == tuple(best_orthogonal_cell(np.array(atoms.cell)))


@pytest.mark.parametrize("plane", list(PLANES))
@pytest.mark.parametrize("angle", [0.0, 30.0])
@pytest.mark.parametrize("num_atoms", [2, 5])
def test_cut_cell_default_cell_is_the_cell_rotated_to_the_plane(
    num_atoms, angle, plane
):
    """cut_cell without a cell fits the atoms into the best orthogonal cell of the
    cell in the frame of the plane, the cell the atoms are cut from."""
    atoms = _rotated(num_atoms, angle, plane)

    cut = cut_cell(atoms, plane=plane)

    lengths = _lengths(plane)
    np.testing.assert_allclose(cut.cell[:], np.diag(lengths), rtol=0, atol=1e-12)
    assert len(cut) == num_atoms
    expected = _unrotated(num_atoms).positions[:, list(PLANES[plane][1])]
    # the atoms of the cut are in any order: each expected atom has one of them
    difference = cut.positions[None, :, :] - expected[:, None, :]
    difference -= lengths * np.round(difference / lengths)
    nearest = np.linalg.norm(difference, axis=-1).min(axis=1)
    np.testing.assert_allclose(nearest, 0.0, rtol=0, atol=1e-10)
