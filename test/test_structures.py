import numpy as np
import pytest
from ase import Atoms, build
from ase.build import bulk

from abtem.atoms import (
    cut_cell,
    is_cell_orthogonal,
    merge_close_atoms,
    orthogonalize_cell,
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


def test_orthogonalize_cell_near_orthorhombic_fallback_raises_for_real_shear(monkeypatch):
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
