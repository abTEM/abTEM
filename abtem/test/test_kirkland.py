import numpy as np
from ase import Atoms
from abtem.waves import PlaneWaves
from abtem.potentials import Potential


def test_fig_5_12():
    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 4) for x in np.linspace(5, 45, 5)], cell=(50, 50, 8))

    potential = Potential(atoms=atoms, gpts=512, parametrization='kirkland', num_slices=1)

    waves = PlaneWaves(energy=200e3)

    waves = waves.multislice(potential, show_progress=False)
    waves = waves.apply_ctf(defocus=700, Cs=1.3e7, cutoff=.01037)

    intensity = np.abs(waves.array[0]) ** 2

    assert np.round(intensity.min(), 2) == .72
    assert np.round(intensity.max(), 2) == 1.03


