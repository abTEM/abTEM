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

class TestEulerSequence:
    def test_intrinsic_xyz(self):
        assert euler_sequence("xyz", "intrinsic") == (0, 0, 0, 0)

    def test_static_same_as_intrinsic(self):
        assert euler_sequence("zyz", "static") == euler_sequence("zyz", "intrinsic")

    def test_extrinsic_zyx(self):
        assert euler_sequence("zyx", "extrinsic") == (0, 0, 0, 1)

    def test_rotating_same_as_extrinsic(self):
        assert euler_sequence("xyz", "rotating") == euler_sequence("xyz", "extrinsic")

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError):
            euler_sequence("xyz", "bad_convention")


# ---------------------------------------------------------------------------
# plane_to_axes
# ---------------------------------------------------------------------------

class TestPlaneToAxes:
    def test_xy_gives_0_1_2(self):
        assert plane_to_axes("xy") == (0, 1, 2)

    def test_xz_gives_0_2_1(self):
        assert plane_to_axes("xz") == (0, 2, 1)

    def test_yz_gives_1_2_0(self):
        assert plane_to_axes("yz") == (1, 2, 0)

    def test_result_is_permutation_of_012(self):
        for plane in ("xy", "xz", "yx", "yz", "zx", "zy"):
            axes = plane_to_axes(plane)
            assert sorted(axes) == [0, 1, 2]


# ---------------------------------------------------------------------------
# is_cell_hexagonal
# ---------------------------------------------------------------------------

class TestIsCellHexagonal:
    def test_hcp_is_hexagonal(self):
        atoms = bulk("Be")
        assert is_cell_hexagonal(atoms)

    def test_cubic_is_not_hexagonal(self):
        atoms = bulk("Al", cubic=True)
        assert not is_cell_hexagonal(atoms)

    def test_accepts_cell_directly(self):
        atoms = bulk("Be")
        assert is_cell_hexagonal(atoms.cell)


# ---------------------------------------------------------------------------
# is_cell_orthogonal
# ---------------------------------------------------------------------------

class TestIsCellOrthogonal:
    def test_cubic_is_orthogonal(self):
        atoms = bulk("Al", cubic=True)
        assert is_cell_orthogonal(atoms)

    def test_hcp_is_not_orthogonal(self):
        atoms = bulk("Be")
        assert not is_cell_orthogonal(atoms)

    def test_accepts_ndarray(self):
        cell = np.diag([3.0, 3.0, 3.0])
        assert is_cell_orthogonal(cell)


# ---------------------------------------------------------------------------
# is_cell_valid
# ---------------------------------------------------------------------------

class TestIsCellValid:
    def test_orthogonal_cubic_is_valid(self):
        atoms = bulk("Al", cubic=True)
        assert is_cell_valid(atoms)

    def test_standardized_structure_is_valid(self):
        atoms = bulk("Al", cubic=True)
        standardized = standardize_cell(atoms)
        assert is_cell_valid(standardized)


# ---------------------------------------------------------------------------
# standardize_cell
# ---------------------------------------------------------------------------

class TestStandardizeCell:
    def test_cubic_unchanged(self):
        atoms = bulk("Al", cubic=True)
        result = standardize_cell(atoms)
        assert is_cell_valid(result)

    def test_invalid_cell_raises(self):
        atoms = bulk("Be")  # hexagonal — not directly standardizable to orthogonal
        with pytest.raises(RuntimeError):
            standardize_cell(atoms)


# ---------------------------------------------------------------------------
# euler_to_rotation / rotation_matrix_to_euler round-trip
# ---------------------------------------------------------------------------

class TestEulerRoundTrip:
    @pytest.mark.parametrize("axes", ["xyz", "zxz", "zyz"])
    def test_round_trip_intrinsic(self, axes):
        angles = (0.1, 0.2, 0.3)
        R = euler_to_rotation(*angles, axes=axes)
        assert R.shape == (3, 3)
        # Rotation matrix should be orthogonal
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
        recovered = rotation_matrix_to_euler(R, axes=axes)
        # Re-compute R from recovered angles and compare
        R2 = euler_to_rotation(*recovered, axes=axes)
        assert np.allclose(R, R2, atol=1e-10)

    def test_zero_angles_gives_identity(self):
        R = euler_to_rotation(0.0, 0.0, 0.0)
        assert np.allclose(R, np.eye(3))

    def test_extrinsic_convention(self):
        R = euler_to_rotation(0.1, 0.2, 0.3, axes="zyx", convention="extrinsic")
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# decompose_affine_transform
# ---------------------------------------------------------------------------

class TestDecomposeAffineTransform:
    def test_pure_scale(self):
        A = np.diag([2.0, 3.0, 4.0])
        rotation, scale, shear = decompose_affine_transform(A)
        assert np.allclose(scale, [2.0, 3.0, 4.0], atol=1e-10)

    def test_identity(self):
        rotation, scale, shear = decompose_affine_transform(np.eye(3))
        assert np.allclose(scale, [1.0, 1.0, 1.0], atol=1e-10)
        assert np.allclose(shear, [0.0, 0.0, 0.0], atol=1e-10)
        assert np.allclose(rotation, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# wrap_with_tolerance
# ---------------------------------------------------------------------------

class TestWrapWithTolerance:
    def test_returns_atoms(self):
        from ase import Atoms
        atoms = bulk("Al", cubic=True)
        result = wrap_with_tolerance(atoms)
        assert isinstance(result, Atoms)

    def test_positions_within_cell(self):
        atoms = bulk("Al", cubic=True)
        result = wrap_with_tolerance(atoms)
        scaled = result.get_scaled_positions()
        assert np.all(scaled >= -1e-6)
        assert np.all(scaled < 1.0 + 1e-6)

    def test_does_not_modify_original(self):
        atoms = bulk("Al", cubic=True)
        original_pos = atoms.positions.copy()
        wrap_with_tolerance(atoms)
        assert np.allclose(atoms.positions, original_pos)


# ---------------------------------------------------------------------------
# flip_atoms
# ---------------------------------------------------------------------------

class TestFlipAtoms:
    def test_z_flip(self):
        atoms = bulk("Al", cubic=True)
        flipped = flip_atoms(atoms, axis=2)
        # z positions should be mirrored about cell[2,2]/2
        expected_z = atoms.cell[2, 2] - atoms.positions[:, 2]
        assert np.allclose(flipped.positions[:, 2], expected_z)

    def test_does_not_modify_original(self):
        atoms = bulk("Al", cubic=True)
        original_pos = atoms.positions.copy()
        flip_atoms(atoms, axis=2)
        assert np.allclose(atoms.positions, original_pos)

    def test_x_and_y_unchanged(self):
        atoms = bulk("Al", cubic=True)
        flipped = flip_atoms(atoms, axis=2)
        assert np.allclose(flipped.positions[:, :2], atoms.positions[:, :2])

    def test_double_flip_is_identity(self):
        atoms = bulk("Al", cubic=True)
        double_flipped = flip_atoms(flip_atoms(atoms, axis=2), axis=2)
        assert np.allclose(double_flipped.positions, atoms.positions)


# ---------------------------------------------------------------------------
# rotate_atoms
# ---------------------------------------------------------------------------

class TestRotateAtoms:
    def test_zero_rotation_is_identity(self):
        atoms = bulk("Al", cubic=True)
        rotated = rotate_atoms(atoms, angles=(0.0, 0.0, 0.0))
        assert np.allclose(rotated.positions, atoms.positions)

    def test_does_not_modify_original(self):
        atoms = bulk("Al", cubic=True)
        original_pos = atoms.positions.copy()
        rotate_atoms(atoms, angles=(0.1, 0.2, 0.3))
        assert np.allclose(atoms.positions, original_pos)

    def test_preserves_interatomic_distances(self):
        atoms = bulk("Al", cubic=True)
        rotated = rotate_atoms(atoms, angles=(0.3, 0.5, 0.1))
        orig_dists = np.sort(np.linalg.norm(
            atoms.positions[:, None] - atoms.positions[None, :], axis=-1
        ).ravel())
        rot_dists = np.sort(np.linalg.norm(
            rotated.positions[:, None] - rotated.positions[None, :], axis=-1
        ).ravel())
        assert np.allclose(orig_dists, rot_dists, atol=1e-10)

    def test_scalar_angle(self):
        atoms = bulk("Al", cubic=True)
        rotated = rotate_atoms(atoms, angles=0.1)
        assert rotated.positions.shape == atoms.positions.shape


# ---------------------------------------------------------------------------
# rotate_atoms_to_plane
# ---------------------------------------------------------------------------

class TestRotateAtomsToPlane:
    def test_xy_plane_no_change(self):
        atoms = bulk("Al", cubic=True)
        result = rotate_atoms_to_plane(atoms, plane="xy")
        assert result is atoms  # returns original object unchanged

    def test_xz_plane_returns_valid(self):
        atoms = bulk("Al", cubic=True)
        result = rotate_atoms_to_plane(atoms, plane="xz")
        assert is_cell_valid(result)


# ---------------------------------------------------------------------------
# best_orthogonal_cell
# ---------------------------------------------------------------------------

class TestBestOrthogonalCell:
    def test_cubic_cell_returns_lengths(self):
        atoms = bulk("Al", cubic=True)
        result = best_orthogonal_cell(np.array(atoms.cell))
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_result_is_positive(self):
        cell = np.diag([3.0, 4.0, 5.0])
        result = best_orthogonal_cell(cell)
        assert np.all(result > 0)

    def test_zero_vector_raises(self):
        # Two columns with zero norm trigger the RuntimeError
        cell = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
        with pytest.raises(RuntimeError):
            best_orthogonal_cell(cell)
