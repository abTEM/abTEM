import os

import numpy as np
import pytest
from ase.io import read

from abtem.potentials import Potential
from abtem.structures import orthogonalize_cell
from ase import Atoms
from ase import units


@pytest.mark.gpaw
def test_dft():
    from gpaw import GPAW
    from abtem.dft import GPAWPotential

    atoms = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/hexagonal_graphene.cif'))

    gpaw = GPAW(h=.1, txt=None, kpts=(3, 3, 1))
    atoms.calc = gpaw
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
    assert np.isclose(43.573837, relative_difference[valid].max(), atol=.1)


@pytest.mark.gpaw
def test_compare_abtem_to_gpaw():
    from gpaw import GPAW
    from gpaw.utilities.ps2ae import PS2AE
    from abtem.dft import GPAWPotential

    atoms = Atoms('C', positions=[(0, 1, 2)], cell=(2, 2, 4), pbc=True)

    calc = GPAW(h=.2, txt=None, kpts=(2, 2, 1))
    atoms.calc = calc
    atoms.get_potential_energy()

    h = 0.01
    t = PS2AE(calc, h=h)
    ae = (-t.get_electrostatic_potential(rcgauss=.02 * units.Bohr) * h).sum(-1)
    ae -= ae.min()

    dft_pot = GPAWPotential(calc, gpts=ae.shape, core_size=.02, slice_thickness=4)
    dft_array = dft_pot.build(pbar=False)
    abtem_ae = dft_array.array.sum(0)
    abtem_ae -= abtem_ae.min()

    valid = abtem_ae > 1

    assert np.all(((abtem_ae[valid] - ae[valid]) / ae[valid]).max() < .001)
