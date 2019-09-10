import numpy as np
import pytest
from ase import Atoms

from abtem.potentials import Potential


@pytest.fixture
def potential():
    atoms = Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 4))

    potential = Potential(atoms=atoms, gpts=32, num_slices=5)

    potential.current_slice = 2

    return potential._evaluate_interpolation(2)


def test_potential_centered(potential):
    center = np.where(potential == np.max(potential))

    assert (center[0] == 16) & (center[1] == 16)


def test_potential_symmetric(potential):
    a = potential[:, 1:17]
    b = np.fliplr(potential[:, 16:])

    assert np.all(np.isclose(a, b))

    a = potential[1:17]
    b = np.flipud(potential[16:])

    assert np.all(np.isclose(a, b))
