import numpy as np
import pytest
from ase.io import read

from abtem.potentials import Potential


@pytest.mark.gpaw
def test_dft():
    from gpaw import GPAW, PW
    from abtem.dft import GPAWPotential

    atoms = read('data/graphene.traj')

    calc = GPAW(mode=PW(400), h=.1, txt=None, kpts=(4, 2, 2))
    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    potential_dft = GPAWPotential(calc, sampling=.05).build()
    potential_iam = Potential(atoms, sampling=.05).build()

    projected_dft = potential_dft.array.sum(0)
    projected_dft -= projected_dft.min()
    projected_iam = potential_iam.array.sum(0)
    projected_iam -= projected_iam.min()

    rel_diff = (projected_iam - projected_dft) / (projected_iam + 1e-16) * 100
    rel_diff[projected_iam < 10] = np.nan

    assert np.round(np.nanmax(rel_diff) / 10, 0) * 10 == 40
