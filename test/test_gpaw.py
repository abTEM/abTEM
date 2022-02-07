from ase import Atoms
from gpaw import GPAW

from abtem import Potential
from abtem.potentials.gpaw import GPAWPotential
from utils import array_is_close
import pytest
from gpaw.utilities.ps2ae import PS2AE


@pytest.fixture
def single_atom_gpaw_calculator(symbol='C'):
    atoms = Atoms(symbol, positions=[(0, 0, 0)], cell=(5, 5, 5), pbc=True)
    gpaw = GPAW(h=.2, txt=None, kpts=(3, 3, 3))
    atoms.calc = gpaw
    atoms.get_potential_energy()
    return gpaw


def test_ps2ae_vs_abtem(single_atom_gpaw_calculator):
    ps2ae_potential = PS2AE(single_atom_gpaw_calculator, grid_spacing=0.02).get_electrostatic_potential()
    ps2ae_potential = - ps2ae_potential * single_atom_gpaw_calculator.atoms.cell[2, 2] / ps2ae_potential.shape[-1]
    ps2ae_potential = ps2ae_potential.sum(0)
    ps2ae_potential -= ps2ae_potential.min()

    gpaw_potential = GPAWPotential(single_atom_gpaw_calculator, gpts=ps2ae_potential.shape).build().project().array
    gpaw_potential -= gpaw_potential.min()

    array_is_close(ps2ae_potential[0, 2:-1], gpaw_potential[0, 2:-1], rel_tol=0.01, check_above_rel=.01)


def test_gpaw_vs_iam(single_atom_gpaw_calculator):
    gpaw_potential = GPAWPotential(single_atom_gpaw_calculator, gpts=256).build().project().array
    gpaw_potential -= gpaw_potential.min()

    iam_potential = Potential(single_atom_gpaw_calculator.atoms, gpts=256, parametrization='lobato',
                              projection='finite', cutoff_tolerance=1e-7).build().project().array
    iam_potential -= iam_potential.min()
    array_is_close(iam_potential, gpaw_potential, rel_tol=0.05, check_above_rel=.05)
