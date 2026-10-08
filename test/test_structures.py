import itertools
import warnings

import ase
import numpy as np
import pytest
from ase import Atoms, build
from ase.build import bulk, cut, graphene
from scipy.spatial import cKDTree
from utils import cpu_float64, ignore_strain_warning

import abtem
from abtem.atoms import (
    _wrap_far_atoms,
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
    pad_atoms,
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


def mos2():
    atoms = build.mx2(
        formula="MoS2", kind="2H", a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=6
    )
    atoms.pbc = True
    return atoms


@pytest.mark.parametrize("perturbation", [-1e-9, -1e-6, -1e-4, 1e-9, 1e-6, 1e-4])
def test_orthogonalize_cell_does_not_drop_boundary_atom(perturbation):
    # Regression test: the Mo atom in this hexagonal cell sits exactly on the
    # cell origin, a corner shared by two faces of the resulting orthogonal
    # supercell. Relaxed DFT/MLIP structures routinely leave such
    # high-symmetry atoms with a tiny numerical residual instead of exactly
    # zero (e.g. -1e-9 instead of 0.0). Depending on its sign, that residual
    # used to determine whether orthogonalize_cell silently dropped the atom
    # (via ase.build.tools.cut's boundary mask), instead of correctly
    # duplicating it across both faces.
    atoms = mos2()
    reference = orthogonalize_cell(atoms.copy())

    perturbed = atoms.copy()
    perturbed.positions[0, :2] += perturbation
    orthogonalized = orthogonalize_cell(perturbed)

    assert len(orthogonalized) == len(reference)


def _near_orthorhombic_cell(noise):
    # `orthogonalize_cell` zeroes any cell component below 1e-6 A before doing
    # anything else, so the off-diagonal noise here has to survive that (i.e.
    # be >= 1e-6) while still being small enough, relative to the cell size,
    # that `best_orthogonal_cell`'s float64 norm computation rounds it away
    # and reports box lengths exactly equal to the (still not-quite-zero)
    # diagonal. That combination only occurs for large cells, hence the
    # unusually large lattice constants below.
    cell = np.diag([5000.0, 6000.0, 7000.0]) + noise
    return Atoms("H", positions=[[1.0, 1.0, 1.0]], cell=cell, pbc=True)


@pytest.mark.parametrize(
    "noise",
    [
        np.array([[0.0, 5e-5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # in v1
        np.array([[0.0, 0.0, 0.0], [5e-5, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # in v2
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5e-5, 5e-5, 0.0]]),  # in v3
    ],
)
def test_orthogonalize_cell_near_orthorhombic_fallback_removes_all_noise(noise):
    # Regression test: `orthogonalize_cell` takes a fallback path when
    # `best_orthogonal_cell`'s box lengths round (in float64) to exactly the
    # cell's diagonal, which can happen for near-orthorhombic cells with tiny
    # off-diagonal noise. The fallback used to only correct the second
    # lattice vector against the first (via a partial Gram-Schmidt), silently
    # leaving the cell non-orthogonal if the noise instead sat in the first
    # or third vector.
    atoms = _near_orthorhombic_cell(noise)
    orthogonalized = orthogonalize_cell(atoms)

    assert is_cell_orthogonal(orthogonalized.cell)
    assert np.allclose(np.diag(orthogonalized.cell), (5000.0, 6000.0, 7000.0))


def test_orthogonalize_cell_near_orthorhombic_fallback_maps_a_real_shear_affinely(
    monkeypatch,
):
    # If a genuinely large (non-noise) shear ever coincides with `diag(cell)
    # == box`, it must not be discarded as noise (that would return a structure
    # with the wrong periodicity): the cell goes through the general
    # repeat-and-cut path, which maps it affinely onto the box, so the shear is
    # strained away and the fractional positions are kept.
    import abtem.atoms as atoms_module

    monkeypatch.setattr(
        atoms_module,
        "best_orthogonal_cell",
        lambda cell, max_repetitions=5: np.array([5.0, 6.0, 7.0]),
    )
    cell = np.array([[5.0, 0.0, 0.0], [0.5, 6.0, 0.0], [0.0, 0.0, 7.0]])
    atoms = Atoms("H", positions=[[1.0, 1.0, 1.0]], cell=cell, pbc=True)

    orthogonalized = orthogonalize_cell(atoms)

    assert np.allclose(np.array(orthogonalized.cell), np.diag([5.0, 6.0, 7.0]))
    assert np.allclose(
        orthogonalized.get_scaled_positions(), atoms.get_scaled_positions()
    )


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


# The box that `orthogonalize_cell` is given.


@ignore_strain_warning
@pytest.mark.parametrize(
    "box", [(1.5, 3.0, 5.0), (4.0, 1.0, 5.0), (4.0, 3.0, 2.0)], ids=["x", "y", "z"]
)
def test_orthogonalize_cell_box_with_no_whole_period_raises(box):
    with pytest.raises(ValueError, match="no whole repetition"):
        orthogonalize_cell(_two_atoms(), box=box)


@ignore_strain_warning
def test_orthogonalize_cell_box_with_one_period_strains_it():
    # A box of 2.1 A holds one period of the 4 A axis, compressed by 47.5 %.
    box = (2.1, 3.0, 5.0)
    atoms = orthogonalize_cell(_two_atoms(), box=box)
    assert atoms.cell.lengths() == pytest.approx(box)
    assert len(atoms) == 2


# Sheared cells whose diagonal equals their best orthogonal box are not noise.


GRID = dict(sampling=0.1, slice_thickness=1.0)


def _bn():
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    return atoms


def _two_atoms(cell=(4.0, 3.0, 5.0)):
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


@cpu_float64
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


@cpu_float64
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


@cpu_float64
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


@cpu_float64
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


@cpu_float64
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


@cpu_float64
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


# The supercell that orthogonalize_cell cuts holds every atom exactly once.


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


# Cells whose lattice is rotated about the axis that becomes the beam direction.


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


def _sorted_positions(atoms):
    return atoms.positions[np.lexsort(np.round(atoms.positions, 6).T)]


@pytest.mark.parametrize("margin", [0.0, 1.25, 3.0])
@pytest.mark.parametrize(
    "pbc, origin",
    [(True, (0.0, 0.0, 0.0)), (False, (0.0, 0.0, 0.0)), (False, (0.7, 0.3, 0.0))],
)
@pytest.mark.parametrize("shift", [(-3, 0, 0), (0, 2, 0), (0, 0, 3), (2, -2, -2)])
def test_cut_cell_keeps_atoms_given_outside_the_cell(shift, pbc, origin, margin):
    """Moving an atom by whole lattice vectors leaves the repeated structure
    unchanged, so it must leave the cut unchanged. Hexagonal, so the lattice
    vectors are not the box axes; the cut has a margin, a nonzero origin, and
    pbc False, where ``wrap`` does nothing."""
    atoms = graphene(a=2.46, vacuum=2.0)
    atoms.pbc = pbc
    box = (4.26, 4.92, 4.0)
    reference = cut_cell(atoms, cell=box, margin=margin, origin=origin)

    moved = atoms.copy()
    moved.positions[0] += np.dot(shift, atoms.cell)
    cut = cut_cell(moved, cell=box, margin=margin, origin=origin)

    assert len(cut) == len(reference)
    np.testing.assert_allclose(
        _sorted_positions(cut), _sorted_positions(reference), rtol=0, atol=1e-9
    )


def test_cut_cell_keeps_an_atom_at_the_upper_face_within_rounding():
    """An atom 1e-15 short of the upper face of the cell is kept by the cut at the
    position the cut keeps for the same atom at the lower face, where it is one
    cell length further out than the cell it is given in."""
    cell = np.diag([4.0, 5.0, 4.0])
    reference = Atoms("B", positions=[[0.0, 2.5, 2.0]], cell=cell, pbc=True)
    atoms = Atoms("B", positions=[[4.0 - 1e-15, 2.5, 2.0]], cell=cell, pbc=True)

    cut = cut_cell(atoms, cell=(8.0, 5.0, 4.0))

    assert len(cut) == len(cut_cell(reference, cell=(8.0, 5.0, 4.0))) == 2
    np.testing.assert_allclose(
        np.sort(cut.positions[:, 0]), [0.0, 4.0], rtol=0, atol=1e-12
    )


def test_cut_cell_repeats_the_atoms_as_far_as_the_box_needs_and_no_further(
    monkeypatch,
):
    """Atoms given many cells outside the cell are repeated as often as atoms
    inside it, as the repeated structure holds every atom of the cell."""
    atoms = bulk("Si", cubic=True)
    built = []
    multiply = Atoms.__mul__

    def spy(self, repetitions):
        repeated = multiply(self, repetitions)
        built.append(len(repeated))
        return repeated

    monkeypatch.setattr(Atoms, "__mul__", spy)
    reference = cut_cell(atoms, cell=(6.0, 6.0, 5.0), margin=2.0)
    reference_built = max(built)

    far = atoms.copy()
    far.positions[0] -= 20 * far.cell.sum(0)
    far.positions[1] += 20 * far.cell.sum(0)
    built.clear()
    cut = cut_cell(far, cell=(6.0, 6.0, 5.0), margin=2.0)

    assert len(cut) == len(reference)
    assert max(built) <= reference_built
    np.testing.assert_allclose(
        _sorted_positions(cut), _sorted_positions(reference), rtol=0, atol=1e-9
    )


def test_cut_cell_leaves_atoms_within_reach_of_the_repetitions_in_place():
    """An atom given just outside the cell, on either side, has all its images in
    the repetitions of the cell the box gives, so the cut holds the atoms of those
    repetitions in order: the repetitions in turn, the atoms of the cell in each."""
    cell = np.diag([4.0, 5.0, 4.0])
    atoms = Atoms(
        "B3",
        positions=[[2.0, 2.5, 2.0], [-0.3, 1.0, -0.2], [1.0, -2.0, 2.7]],
        cell=cell,
        pbc=True,
    )

    cut = cut_cell(atoms, cell=(8.0, 10.0, 4.0))

    shifts = itertools.product(range(3), range(3), range(2))
    repeated = np.concatenate(
        [atoms.positions + np.dot(shift, cell) for shift in shifts]
    )
    box = np.array([8.0, 10.0, 4.0])
    kept = np.all((repeated >= -1e-12 * box) & (repeated < box - 1e-12 * box), axis=1)
    np.testing.assert_array_equal(cut.positions, repeated[kept])


# The atoms that cut_cell keeps from the cell and its margin, in order, for the
# inputs of the tests below.
SI_PRIMITIVE_MARGIN_POSITIONS = [
    [5.43, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.3575, 1.3575, 1.3575],
    [2.715, 2.715, 0.0],
    [4.0725, 4.0725, 1.3575],
    [2.715, 0.0, 2.715],
    [4.0725, 1.3575, 4.0725],
    [5.43, 2.715, 2.715],
    [0.0, 5.43, 0.0],
    [0.0, 2.715, 2.715],
    [1.3575, 4.0725, 4.0725],
    [2.715, 5.43, 2.715],
    [0.0, 0.0, 5.43],
    [2.715, 2.715, 5.43],
    [5.43, 5.43, 5.43],
]
MOS2_PAST_THE_FACE_POSITIONS = [
    [0.0, 0.0, 4.595],
    [0.4823, 2.8366, 6.19],
    [1.59, 0.918, 3.0],
    [0.0, 3.6719, 3.0],
    [1.59, 2.754, 4.595],
]


def test_cut_cell_keeps_the_order_of_the_atoms_the_repetitions_reach():
    """FrozenPhonons draws the displacements of a cut by the index of the atom, so
    the atoms the repetitions reach come in the same order wherever the images of
    others are added: the margin images of a non-orthogonal cell, and the images
    of an atom 0.08 A past the second lattice vector's face of a hexagonal cell,
    follow them. The positions are those the cut had before it added images."""
    primitive = cut_cell(bulk("Si"), margin=0.8)
    assert len(primitive) == 18
    np.testing.assert_allclose(
        primitive.positions[:15], SI_PRIMITIVE_MARGIN_POSITIONS, rtol=0, atol=5e-5
    )

    atoms = build.mx2("MoS2", vacuum=3.0)
    scaled = atoms.get_scaled_positions(wrap=False)
    scaled[1, 1] = 1.03
    atoms.set_scaled_positions(scaled)
    hexagonal = cut_cell(atoms)
    assert len(hexagonal) == 6
    np.testing.assert_allclose(
        hexagonal.positions[:5], MOS2_PAST_THE_FACE_POSITIONS, rtol=0, atol=5e-5
    )


@pytest.mark.parametrize("length, margin", [(4.0, 4.29), (20.0, 4.29), (5.0, 1.05)])
def test_wrap_far_atoms_leaves_pad_atoms_every_image_of_every_atom(length, margin):
    """Wherever an atom is given along x, the padding of the wrapped atom holds
    all the atoms of the periodic structure within the margin, and the wrapped atom
    is itself within the margin. An atom is moved only when it needs to be."""
    margins = (margin, margin, margin)
    given = np.arange(-7.5 * length, 7.5 * length, 0.37)
    atoms = Atoms(
        "B" * len(given),
        positions=np.column_stack([given, np.ones_like(given), np.ones_like(given)]),
        cell=np.diag([length, 5.0, 5.0]),
        pbc=True,
    )
    wrapped = atoms.copy()

    _wrap_far_atoms(wrapped, margins, "x")

    moved = wrapped.positions[:, 0] != given
    within = (given >= -margin) & (given < length + margin)
    reached = (given >= margin - np.ceil(margin / length) * length) & (
        given < (np.ceil(margin / length) + 1) * length - margin
    )
    np.testing.assert_array_equal(moved, ~(within & reached))
    for x, new_x in zip(given, wrapped.positions[:, 0]):
        single = Atoms("B", positions=[[new_x, 1.0, 1.0]], cell=atoms.cell, pbc=True)
        padded = pad_atoms(single, margins, "x").positions[:, 0]
        expected = x + length * np.arange(-12, 13)
        expected = expected[
            (expected >= -margin - 1e-12) & (expected < length + margin - 1e-12)
        ]
        np.testing.assert_allclose(np.sort(padded), expected, rtol=0, atol=1e-9)
        assert -margin <= new_x < length + margin
