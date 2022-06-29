import os
import sys

import numpy as np
import pytest
from ase import Atoms, units

from abtem.potentials.potentials import Potential

try:
    from gpaw import GPAW
    from gpaw.utilities.ps2ae import PS2AE
    from abtem.potentials.gpaw import GPAWPotential
except ImportError:
    pass


@pytest.fixture
def single_atom_gpaw_calculator(request):
    atoms = Atoms(request.param['symbol'], positions=[(0, 0, 0)], cell=(request.param['side_length'],) * 3, pbc=True)
    gpaw = GPAW(h=.2, txt=None, kpts=(3, 3, 3))
    atoms.calc = gpaw
    atoms.get_potential_energy()
    return gpaw


@pytest.mark.parametrize("single_atom_gpaw_calculator",
                         [{'side_length': 5., 'symbol': 'C'},
                          {'side_length': 2., 'symbol': 'C'}],
                         indirect=True)
@pytest.mark.skipif('gpaw' not in sys.modules, reason="requires gpaw")
def test_ps2ae_vs_abtem(single_atom_gpaw_calculator):
    ps2ae_potential = PS2AE(single_atom_gpaw_calculator, h=0.02)
    ps2ae_potential = ps2ae_potential.get_electrostatic_potential(rcgauss=.01 * units.Bohr, ae=True)
    ps2ae_potential = - ps2ae_potential * single_atom_gpaw_calculator.atoms.cell[2, 2] / ps2ae_potential.shape[-1]
    ps2ae_potential = ps2ae_potential.sum(0)
    ps2ae_potential -= ps2ae_potential.min()

    gpaw_potential = GPAWPotential(single_atom_gpaw_calculator,
                                   gpts=ps2ae_potential.shape).build().project().compute().array
    gpaw_potential -= gpaw_potential.min()

    assert np.allclose(ps2ae_potential[1:], gpaw_potential[1:], rtol=1e-2, atol=1)


@pytest.mark.parametrize("single_atom_gpaw_calculator",
                         [{'side_length': 5., 'symbol': 'C'}],
                         indirect=True)
@pytest.mark.skipif('gpaw' not in sys.modules, reason="requires gpaw")
def test_gpaw_vs_iam(single_atom_gpaw_calculator):
    gpaw_potential = GPAWPotential(single_atom_gpaw_calculator, gpts=128).build().project().array
    gpaw_potential -= gpaw_potential.min()

    iam_potential = Potential(single_atom_gpaw_calculator.atoms, gpts=gpaw_potential.shape,
                              projection='finite').build().project().array

    iam_potential -= iam_potential.min()
    assert np.allclose(iam_potential, gpaw_potential, rtol=1e-3, atol=5)


@pytest.mark.parametrize("single_atom_gpaw_calculator",
                         [{'side_length': 2., 'symbol': 'C'}],
                         indirect=True)
def test_gpaw_potential_from_disk(single_atom_gpaw_calculator, tmpdir):
    path = os.path.join(str(tmpdir), 'test.gpw')
    single_atom_gpaw_calculator.write(path)

    gpaw_potential = GPAWPotential(single_atom_gpaw_calculator, gpts=(32, 32))
    gpaw_potential = gpaw_potential.build().compute()

    gpaw_potential_from_disk = GPAWPotential(path, gpts=(32, 32))
    gpaw_potential_from_disk = gpaw_potential_from_disk.build().compute()

    assert gpaw_potential_from_disk == gpaw_potential


@pytest.mark.parametrize("single_atom_gpaw_calculator",
                         [{'side_length': 2., 'symbol': 'C'}],
                         indirect=True)
def test_gpaw_potential_with_frozen_phonons(single_atom_gpaw_calculator, tmpdir):
    gpaw_potential = GPAWPotential([single_atom_gpaw_calculator] * 2, gpts=(32, 32))
    gpaw_potential = gpaw_potential.build().compute()
