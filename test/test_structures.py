import itertools

import numpy as np
import pytest
from ase import Atoms, build
from ase.build import bulk

from abtem.atoms import (
    _wrap_far_atoms,
    cut_cell,
    is_cell_orthogonal,
    merge_close_atoms,
    orthogonalize_cell,
    pad_atoms,
    shrink_cell,
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


def test_orthogonalize_cell_near_orthorhombic_fallback_raises_for_real_shear(
    monkeypatch,
):
    # If a genuinely large (non-noise) shear ever coincides with `diag(cell)
    # == box`, the fallback must refuse to silently discard it rather than
    # returning a structure with the wrong periodicity.
    import abtem.atoms as atoms_module

    monkeypatch.setattr(
        atoms_module,
        "best_orthogonal_cell",
        lambda cell, max_repetitions=5: np.array([5.0, 6.0, 7.0]),
    )
    cell = np.array([[5.0, 0.0, 0.0], [0.5, 6.0, 0.0], [0.0, 0.0, 7.0]])
    atoms = Atoms("H", positions=[[1.0, 1.0, 1.0]], cell=cell, pbc=True)

    with pytest.raises(RuntimeError):
        orthogonalize_cell(atoms)


def _sorted_positions(atoms):
    return atoms.positions[np.lexsort(np.round(atoms.positions, 6).T)]


@pytest.mark.parametrize(
    "pbc, origin",
    [(True, (0.0, 0.0, 0.0)), (False, (0.0, 0.0, 0.0)), (False, (0.7, 0.3, 0.0))],
)
def test_cut_cell_keeps_atoms_given_outside_the_cell(pbc, origin):
    """Moving an atom by whole lattice vectors leaves the repeated structure
    unchanged, so it must leave the cut unchanged. Hexagonal, so the lattice
    vectors are not the box axes; the cut has a margin, a nonzero origin, and
    pbc False, where ``wrap`` does nothing."""
    atoms = build.graphene(a=2.46, vacuum=2.0)
    atoms.pbc = pbc
    box = (4.26, 4.92, 4.0)
    for margin in (0.0, 1.25, 3.0):
        reference = cut_cell(atoms, cell=box, margin=margin, origin=origin)
        for shift in [(-3, 0, 0), (0, 2, 0), (0, 0, 3), (2, -2, -2)]:
            moved = atoms.copy()
            moved.positions[0] += np.dot(shift, atoms.cell)
            cut = cut_cell(moved, cell=box, margin=margin, origin=origin)

            assert len(cut) == len(reference), (margin, shift)
            np.testing.assert_allclose(
                _sorted_positions(cut),
                _sorted_positions(reference),
                rtol=0,
                atol=1e-9,
                err_msg=f"margin {margin}, shift {shift}",
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


@pytest.mark.parametrize(
    "short, length, expected", [(6e-12, 8.0, 2), (1e-11, 40.0, 10), (1e-15, 40.0, 10)]
)
def test_cut_cell_keeps_every_image_the_crop_keeps_at_a_face_of_a_wide_box(
    short, length, expected
):
    """The crop keeps an atom up to 1e-12 of the box length below the lower face of
    the box, so the cut holds every image of an atom short of the upper face of its
    cell by less than that, 1e-11 A short included when the box is 40 A wide:
    one per cell along the box, as for an atom at the face."""
    cell = np.diag([4.0, 5.0, 4.0])
    atoms = Atoms("B", positions=[[4.0 - short, 2.5, 2.0]], cell=cell, pbc=True)

    cut = cut_cell(atoms, cell=(length, 5.0, 4.0))

    assert len(cut) == expected


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
