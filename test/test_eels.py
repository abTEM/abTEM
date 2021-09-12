import numpy as np
from ase import Atoms

from abtem import SMatrix, Potential, GridScan
from abtem.ionization import TransitionPotential, EELSDetector


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
                                       transition_potential, pbar=False)

    assert np.allclose(measurement_eels.array, 1)


def test_tranpose():
    pass
    # from abtem.waves import FresnelPropagator, PlaneWave
    #
    # atoms = Atoms('O', positions=[(2, 2, 2)], cell=(4, 4, 4))
    #
    # potential = Potential(atoms, gpts=512, slice_thickness=4, projection='infinite', parametrization='kirkland')
    # potential = potential.build(pbar=False).as_transmission_function(100e3)
    #
    # wave = PlaneWave(energy=100e3, device='cpu')
    # wave.match_grid(potential)
    # wave = wave.build()
    #
    # # plt.imshow((potential.array * np.conjugate(potential.array))[0].real)
    # # plt.colorbar()
    #
    # wave = potential.transmit(wave)
    # wave = propagator.propagate(wave, potential_slice.thickness)
    # wave = potential.transmit(wave)
    # wave = propagator.propagate(wave, potential_slice.thickness)
    #
    # plt.imshow(np.abs(wave.array[0]) ** 2)
    # plt.colorbar()
    # plt.show()
    #
    # wave = propagator.propagate(wave, -potential_slice.thickness)
    # wave = potential.transmit(wave, conjugate=True)
    # wave = propagator.propagate(wave, -potential_slice.thickness)
    # wave = potential.transmit(wave, conjugate=True)

