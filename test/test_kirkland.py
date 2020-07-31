import numpy as np
from ase import Atoms
from abtem.waves import PlaneWave
from abtem.potentials import Potential


def test_fig_5_12():
    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 4) for x in np.linspace(5, 45, 5)], cell=(50, 50, 8))

    potential = Potential(atoms=atoms, gpts=512, parametrization='kirkland', cutoff_tolerance=1e-4)

    waves = PlaneWave(energy=200e3)

    waves = waves.multislice(potential, pbar=False)
    waves = waves.apply_ctf(defocus=700, Cs=1.3e7, semiangle_cutoff=.01037)

    intensity = np.abs(waves.array) ** 2

    assert np.round(intensity.min(), 2) == np.float32(.72)
    assert np.round(intensity.max(), 2) == np.float32(1.03)
