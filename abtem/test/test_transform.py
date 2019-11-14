from ase import build
from ase.io import read
from ..transform import orthogonalize_atoms


def test_orthogonalize():
    atoms = read('data/SrTiO.traj')
    atoms = build.surface(atoms, (1, 1, 1), 9)
    atoms.center(axis=2, vacuum=0)

    orthogonal_atoms = orthogonalize_atoms(atoms, 1, 2)

    assert len(orthogonal_atoms) == 2 * len(atoms)


