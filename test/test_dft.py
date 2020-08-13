import os

import numpy as np
import pytest
from ase.io import read
from gpaw import GPAW

from abtem.dft import GPAWPotential
from abtem.potentials import Potential
from abtem.structures import orthogonalize_cell


@pytest.mark.gpaw
def test_dft():
    atoms = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/hexagonal_graphene.cif'))

    gpaw = GPAW(h=.1, txt=None, kpts=(3, 3, 1))
    atoms.set_calculator(gpaw)
    atoms.get_potential_energy()

    dft_pot = GPAWPotential(gpaw, sampling=.02)

    dft_array = dft_pot.build()

    dft_potential = dft_array.tile((3, 2))

    atoms = orthogonalize_cell(gpaw.atoms) * (3, 2, 1)

    iam_potential = Potential(atoms, gpts=dft_potential.gpts, cutoff_tolerance=1e-4, device='cpu').build()

    projected_iam = iam_potential.array.sum(0)
    projected_iam -= projected_iam.min()

    projected_dft = dft_potential.array.sum(0)
    projected_dft -= projected_dft.min()

    absolute_difference = projected_iam - projected_dft

    valid = np.abs(projected_iam) > 1
    relative_difference = np.zeros_like(projected_iam)
    relative_difference[:] = np.nan
    relative_difference[valid] = 100 * (projected_iam[valid] - projected_dft[valid]) / projected_iam[valid]

    assert np.isclose(9.553661, absolute_difference.max(), atol=.1)
    assert np.isclose(44.327312, relative_difference[valid].max(), atol=.1)
