import numpy as np
import pytest
from ase import Atoms

from abtem.potentials import Potential


def test_create_potential():
    atoms = Atoms('C', positions=[(2, 3, 2)], cell=(4, 6, 4.3))

    potential = Potential(atoms=atoms, sampling=.1)

    assert np.all(potential.extent == [4, 6])
    assert potential.thickness == 4.3
    assert potential.num_slices == 9
    assert potential.slice_thickness(0) == 4.3 / 9


def test_potential_raises():
    with pytest.raises(RuntimeError) as e:
        Potential(Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 0)))

    assert str(e.value) == 'atoms has no thickness'

    with pytest.raises(RuntimeError) as e:
        Potential(Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 0)))


def test_cutoff():
    potential = Potential(Atoms(cell=(1, 1, 1)))

    cutoff = potential.get_cutoff(6)
    assert np.isclose(potential.evaluate_potential(cutoff, 6), potential.cutoff_tolerance)

    cutoff = potential.get_cutoff(47)
    assert np.isclose(potential.evaluate_potential(cutoff, 47), potential.cutoff_tolerance)


def test_padded_atoms():
    atoms = Atoms('C', positions=[(1, 1, 1)], cell=(2, 2, 2))
    potential = Potential(atoms, cutoff_tolerance=1e-3)
    padded_atoms = potential.get_padded_atoms()
    assert len(padded_atoms) == 9

    potential = Potential(atoms, cutoff_tolerance=1e-5)
    padded_atoms = potential.get_padded_atoms()
    assert len(padded_atoms) == 25


def test_potential_centered():
    atoms = Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 4))
    potential = Potential(atoms=atoms, gpts=41, num_slices=1).precalculate()

    proj = potential.array.sum(0)
    center = np.where(np.isclose(proj, np.max(proj)))

    assert np.all(center[0] == [20, 20, 21, 21]) & np.all(center[1] == [20, 21, 20, 21])

    atoms = Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 4))
    potential = Potential(atoms=atoms, gpts=40, num_slices=1).precalculate()

    proj = potential.array.sum(0)
    center = np.where(np.isclose(proj, np.max(proj)))

    assert np.all(center[0] == [20]) & np.all(center[1] == [20])


def test_potential_symmetric():
    atoms = Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 4))
    potential = Potential(atoms=atoms, gpts=32, num_slices=5).precalculate().array.sum(0)

    a = potential[:, 1:17]
    b = np.fliplr(potential[:, 16:])

    assert np.all(np.isclose(a, b))

    a = potential[1:17]
    b = np.flipud(potential[16:])

    assert np.all(np.isclose(a, b))
