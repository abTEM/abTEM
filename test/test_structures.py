import numpy as np
import pytest
from ase import build
from ase.build import bulk

from abtem.atoms import (
    best_orthogonal_cell,
    cut_cell,
    decompose_affine_transform,
    euler_sequence,
    euler_to_rotation,
    flip_atoms,
    is_cell_hexagonal,
    is_cell_orthogonal,
    is_cell_valid,
    merge_close_atoms,
    orthogonalize_cell,
    plane_to_axes,
    rotate_atoms,
    rotate_atoms_to_plane,
    rotation_matrix_to_euler,
    shrink_cell,
    standardize_cell,
    wrap_with_tolerance,
)


def fcc(orthogonal=False):
    if orthogonal:
        return bulk("Au", cubic=True)
    else:
        return bulk("Au")


def fcc110(orthogonal=False):
    if orthogonal:
        atoms = build.fcc110("Au", size=(1, 1, 2), periodic=True)
        atoms.positions[:] -= atoms.positions[-1]
        atoms.wrap()
        return atoms
    else:
        atoms = bulk("Au")
        atoms.rotate(45, "x", rotate_cell=True)
        return atoms


def fcc111(orthogonal=False):
    # x_vector = [ 0.81649658, 0.        , 0.57735027]
    # y_vector = [-0.40824829, 0.70710678, 0.57735027]
    if orthogonal:
        atoms = build.fcc111("Au", size=(1, 2, 3), periodic=True, orthogonal=True)
        atoms.positions[:] -= atoms.positions[-1]
        atoms.wrap()
        return atoms
    else:
        atoms = bulk("Au")
        atoms.rotate(45, "x", rotate_cell=True)
        atoms.rotate(np.arctan(np.sqrt(2) / 2) / np.pi * 180, "y", rotate_cell=True)
        atoms.rotate(-90, "z", rotate_cell=True)
        return atoms


def bcc(orthogonal=False):
    if orthogonal:
        return bulk("Fe", cubic=True)
    else:
        return bulk("Fe")


def diamond(orthogonal=False):
    if orthogonal:
        return bulk("C", cubic=True)
    else:
        return bulk("C")


def hcp(orthogonal=False):
    if orthogonal:
        atoms = bulk("Be", orthorhombic=True)
        atoms.positions[:] -= atoms.positions[2]
        atoms.wrap()
        return atoms
    else:
        return bulk("Be")


def assert_atoms_close(atoms1, atoms2):
    merged = merge_close_atoms(atoms1 + atoms2)

    assert len(atoms1) == len(atoms2)
    assert len(atoms1) == len(merged)

    cell1 = atoms1.cell[np.lexsort(np.rot90(atoms1.cell))]
    cell2 = atoms2.cell[np.lexsort(np.rot90(atoms2.cell))]
    assert np.allclose(cell1, cell2)


@pytest.mark.parametrize("structure", [fcc, fcc110, fcc111, bcc, diamond, hcp])
def test_orthogonalize_atoms(structure):
    atoms = structure()
    orthogonal_atoms = structure(orthogonal=True)
    orthogonalized_atoms = orthogonalize_cell(atoms)
    assert_atoms_close(orthogonal_atoms, orthogonalized_atoms)


@pytest.mark.parametrize("structure", [fcc, bcc, diamond, hcp])
@pytest.mark.parametrize("n", [2, 3])
def test_shrink_cell(structure, n):
    atoms = structure()
    repeated_atoms = atoms * (n, n, n)
    shrinked_atoms = shrink_cell(repeated_atoms)
    assert_atoms_close(atoms, shrinked_atoms)


@pytest.mark.parametrize("structure", [fcc, fcc110, fcc111, bcc, diamond, hcp])
def test_cut(structure):
    atoms = structure()

    orthogonalized_atoms = orthogonalize_cell(atoms)
    cut_atoms = cut_cell(atoms, cell=np.diag(orthogonalized_atoms.cell) - 1e-12)

    assert_atoms_close(orthogonalized_atoms, cut_atoms)


# ---------------------------------------------------------------------------
# euler_sequence
# ---------------------------------------------------------------------------

def test_euler_sequence():
    assert euler_sequence("xyz", "intrinsic") == (0, 0, 0, 0)
    assert euler_sequence("zyx", "extrinsic") == (0, 0, 0, 1)
    assert euler_sequence("zyz", "static") == euler_sequence("zyz", "intrinsic")
    assert euler_sequence("xyz", "rotating") == euler_sequence("xyz", "extrinsic")
    with pytest.raises(ValueError):
        euler_sequence("xyz", "bad_convention")


# ---------------------------------------------------------------------------
# plane_to_axes
# ---------------------------------------------------------------------------

def test_plane_to_axes():
    assert plane_to_axes("xy") == (0, 1, 2)
    assert plane_to_axes("xz") == (0, 2, 1)
    assert plane_to_axes("yz") == (1, 2, 0)
    for plane in ("xy", "xz", "yx", "yz", "zx", "zy"):
        assert sorted(plane_to_axes(plane)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# is_cell_hexagonal / is_cell_orthogonal / is_cell_valid / standardize_cell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("atoms,is_hex,is_ortho", [
    (bulk("Be"), True, False),
    (bulk("Al", cubic=True), False, True),
])
def test_cell_predicates(atoms, is_hex, is_ortho):
    assert is_cell_hexagonal(atoms) == is_hex
    assert is_cell_orthogonal(atoms) == is_ortho


def test_is_cell_valid():
    assert is_cell_valid(bulk("Al", cubic=True))
    assert is_cell_valid(standardize_cell(bulk("Al", cubic=True)))


def test_cell_predicate_alternate_inputs():
    assert is_cell_hexagonal(bulk("Be").cell)           # accepts Cell object
    assert is_cell_orthogonal(np.diag([3.0, 3.0, 3.0]))  # accepts ndarray


def test_standardize_cell():
    assert is_cell_valid(standardize_cell(bulk("Al", cubic=True)))
    with pytest.raises(RuntimeError):
        standardize_cell(bulk("Be"))  # hexagonal — not standardizable to orthogonal


# ---------------------------------------------------------------------------
# euler_to_rotation / rotation_matrix_to_euler round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axes", ["xyz", "zxz", "zyz"])
def test_euler_round_trip(axes):
    angles = (0.1, 0.2, 0.3)
    R = euler_to_rotation(*angles, axes=axes)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    R2 = euler_to_rotation(*rotation_matrix_to_euler(R, axes=axes), axes=axes)
    assert np.allclose(R, R2, atol=1e-10)


def test_euler_zero_angles_and_extrinsic():
    assert np.allclose(euler_to_rotation(0.0, 0.0, 0.0), np.eye(3))
    R = euler_to_rotation(0.1, 0.2, 0.3, axes="zyx", convention="extrinsic")
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# decompose_affine_transform
# ---------------------------------------------------------------------------

def test_decompose_affine_transform():
    R, scale, shear = decompose_affine_transform(np.eye(3))
    assert np.allclose(scale, [1.0, 1.0, 1.0]) and np.allclose(shear, [0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3))
    _, scale, _ = decompose_affine_transform(np.diag([2.0, 3.0, 4.0]))
    assert np.allclose(scale, [2.0, 3.0, 4.0], atol=1e-10)


# ---------------------------------------------------------------------------
# wrap_with_tolerance
# ---------------------------------------------------------------------------

def test_wrap_with_tolerance():
    from ase import Atoms
    atoms = bulk("Al", cubic=True)
    original_pos = atoms.positions.copy()
    result = wrap_with_tolerance(atoms)
    assert isinstance(result, Atoms)
    scaled = result.get_scaled_positions()
    assert np.all(scaled >= -1e-6) and np.all(scaled < 1.0 + 1e-6)
    assert np.allclose(atoms.positions, original_pos)  # original not modified


# ---------------------------------------------------------------------------
# flip_atoms
# ---------------------------------------------------------------------------

def test_flip_atoms():
    atoms = bulk("Al", cubic=True)
    original_pos = atoms.positions.copy()
    flipped = flip_atoms(atoms, axis=2)
    assert np.allclose(flipped.positions[:, 2], atoms.cell[2, 2] - atoms.positions[:, 2])
    assert np.allclose(flipped.positions[:, :2], atoms.positions[:, :2])
    assert np.allclose(atoms.positions, original_pos)  # original not modified
    assert np.allclose(flip_atoms(flipped, axis=2).positions, atoms.positions)  # double flip


# ---------------------------------------------------------------------------
# rotate_atoms
# ---------------------------------------------------------------------------

def test_rotate_atoms():
    atoms = bulk("Al", cubic=True)
    original_pos = atoms.positions.copy()
    assert np.allclose(rotate_atoms(atoms, angles=(0.0, 0.0, 0.0)).positions, atoms.positions)
    rotate_atoms(atoms, angles=(0.1, 0.2, 0.3))
    assert np.allclose(atoms.positions, original_pos)  # original not modified


def test_rotate_atoms_preserves_distances():
    atoms = bulk("Al", cubic=True)
    rotated = rotate_atoms(atoms, angles=(0.3, 0.5, 0.1))
    pairwise = lambda pos: np.sort(np.linalg.norm(
        pos[:, None] - pos[None, :], axis=-1
    ).ravel())
    assert np.allclose(pairwise(atoms.positions), pairwise(rotated.positions), atol=1e-10)


# ---------------------------------------------------------------------------
# rotate_atoms_to_plane / best_orthogonal_cell
# ---------------------------------------------------------------------------

def test_rotate_atoms_to_plane():
    atoms = bulk("Al", cubic=True)
    assert rotate_atoms_to_plane(atoms, plane="xy") is atoms
    assert is_cell_valid(rotate_atoms_to_plane(atoms, plane="xz"))


def test_best_orthogonal_cell():
    atoms = bulk("Al", cubic=True)
    result = best_orthogonal_cell(np.array(atoms.cell))
    assert result.shape == (3,) and np.all(result > 0)
    with pytest.raises(RuntimeError):
        # Two zero-norm columns trigger the RuntimeError
        best_orthogonal_cell(np.array([[0., 0., 3.], [0., 0., 4.], [0., 0., 5.]]))
