import numpy as np
from ase import Atoms

from abtem.potentials import Potential
from abtem.waves.waves import PlaneWave
import pytest


@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('projection', ['infinite'])
def test_fig_5_12(projection, lazy):
    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 4) for x in np.linspace(5, 45, 5)], cell=(50, 50, 8))

    potential = Potential(atoms=atoms, gpts=512, parametrization='kirkland', projection=projection)
    waves = PlaneWave(energy=200e3, normalize=False)

    waves = waves.multislice(potential, lazy=lazy)

    waves = waves.apply_ctf(defocus=700, Cs=1.3e7, semiangle_cutoff=10.37, taper=0.)

    intensity = waves.intensity().compute().array

    assert np.round(intensity.min(), 2) == np.float32(.72)
    assert np.round(intensity.max(), 2) == np.float32(1.03)
