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
    assert potential.get_slice_thickness(0) == 4.3 / 9


def test_potential_raises():
    with pytest.raises(RuntimeError) as e:
        Potential(Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 0)))

    assert str(e.value) == 'atoms has no thickness'


def test_padded_atoms():
    atoms = Atoms('C', positions=[(1, 1, 1)], cell=(2, 2, 2))
    potential = Potential(atoms, cutoff_tolerance=1e-3, sampling=.1)
    padded_atoms = potential.get_padded_atoms()
    assert len(padded_atoms) == 9

    potential = Potential(atoms, cutoff_tolerance=1e-5, sampling=.1)
    padded_atoms = potential.get_padded_atoms()
    assert len(padded_atoms) == 25


def test_potential_centered():
    Lx = 5
    Ly = 5
    gpts_x = 40
    gpts_y = 60

    atoms1 = Atoms('C', positions=[(0, Ly / 2, 2)], cell=[Lx, Ly, 4])
    atoms2 = Atoms('C', positions=[(Lx / 2, Ly / 2, 2)], cell=[Lx, Ly, 4])
    potential1 = Potential(atoms1, gpts=(gpts_x, gpts_y), slice_thickness=4, cutoff_tolerance=1e-2)
    potential2 = Potential(atoms2, gpts=(gpts_x, gpts_y), slice_thickness=4, cutoff_tolerance=1e-2)

    assert np.allclose(potential1[0].array[:, gpts_y // 2], np.roll(potential2[0].array[:, gpts_y // 2], gpts_x // 2))
    assert np.allclose(potential2[0].array[:, gpts_y // 2][1:], (potential2[0].array[:, gpts_y // 2])[::-1][:-1])
    assert np.allclose(potential2[0].array[gpts_x // 2][1:], potential2[0].array[gpts_x // 2][::-1][:-1])
