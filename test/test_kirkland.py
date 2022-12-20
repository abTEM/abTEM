import numpy as np
from ase import Atoms

from abtem.potentials import Potential
from abtem.waves import PlaneWave
import pytest


@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('projection', ['infinite', 'finite'])
@pytest.mark.parametrize('integral_space', ['real', 'fourier'])
def test_fig_5_12(projection, lazy, integral_space):
    if integral_space == 'real' and projection == 'infinite':
        pytest.skip("invalid parameter combination")

    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 4) for x in np.linspace(5, 45, 5)], cell=(50, 50, 8))

    potential = Potential(atoms=atoms, gpts=512, parametrization='kirkland', projection=projection)
    waves = PlaneWave(energy=200e3)

    exit_wave = waves.multislice(potential, lazy=lazy)

    exit_wave = exit_wave.apply_ctf(defocus=700, Cs=1.3e7, semiangle_cutoff=10.37)

    intensity = exit_wave.intensity().compute().array

    assert np.round(intensity.min(), 2) == np.float32(.72)
    assert np.round(intensity.max(), 2) == np.float32(1.03)
