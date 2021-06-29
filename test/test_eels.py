from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

from abtem.ionization import SubshellTransitions, TransitionPotential, EELSDetector
from abtem import SMatrix, Potential, GridScan, show_atoms, AnnularDetector, FrozenPhonons

from ase import units
from ase.io import read


class DummyProjectedAtomicTransition:

    def build(self, x):
        return 1


class DummyCarbonTransitions:
    Z = 6

    def get_transition_potentials(self, *arg, **kwargs):
        return [DummyProjectedAtomicTransition()]

    def __len__(self):
        return 1


def test_dummy_transition():
    atoms = Atoms('C', positions=[(2.5, 2.5, 1)], cell=(5, 5, 2))

    transitions = [DummyCarbonTransitions()]

    potential = Potential(atoms[[False]], slice_thickness=1, projection='infinite', parametrization='kirkland')

    S = SMatrix(energy=100e3, sampling=0.02, semiangle_cutoff=20, rolloff=0.0)

    S.match_grid(potential)

    detector_eels = EELSDetector(collection_angle=20, interpolation=1)

    transition_potential = TransitionPotential(transitions, atoms=atoms)

    scan = GridScan((0, 0), potential.extent, sampling=0.9 * S.ctf.nyquist_sampling)

    measurement_eels = S.coreloss_scan(scan,
                                       detector_eels,
                                       potential,
                                       transition_potential)

    assert np.allclose(measurement_eels.array, 1)
