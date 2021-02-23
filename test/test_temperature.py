import numpy as np
import pytest
from ase import Atoms

from abtem import FrozenPhonons
from abtem.detect import AnnularDetector
from abtem.potentials import Potential
from abtem.scan import LineScan
from abtem.waves import Probe


def test_frozen_phonons():
    atoms = Atoms('CO', positions=[(2.5, 2.5, 2), (2.5, 2.5, 3)], cell=(5, 5, 4))
    fp1 = FrozenPhonons(atoms, 10, sigmas={'C': .1, 'O': .1})
    fp2 = FrozenPhonons(atoms, 10, sigmas=.1)
    fp3 = FrozenPhonons(atoms, 10, sigmas=[.1, .1])

    assert np.all(fp1._sigmas == fp2._sigmas) & np.all(fp2._sigmas == fp3._sigmas)


def test_frozen_phonons_raise():
    atoms = Atoms('CO', positions=[(2.5, 2.5, 2), (2.5, 2.5, 3)], cell=(5, 5, 4))

    with pytest.raises(RuntimeError) as e:
        FrozenPhonons(atoms, 10, sigmas={'C': .1})

    assert str(e.value) == 'Displacement standard deviation must be provided for all atomic species.'

    with pytest.raises(RuntimeError) as e:
        FrozenPhonons(atoms, 10, sigmas=[.1])

    assert str(e.value) == 'Displacement standard deviation must be provided for all atoms.'


def test_probe_line_scan():
    atoms = Atoms('CO', positions=[(2.5, 2.5, 2), (2.5, 2.5, 3)], cell=(5, 5, 4))
    frozen_phonons = FrozenPhonons(atoms, 2, sigmas={'C': 0, 'O': 0.})

    potential = Potential(atoms, sampling=.05)
    tds_potential = Potential(frozen_phonons, sampling=.05)

    linescan = LineScan(start=[0, 0], end=[2.5, 2.5], gpts=10)
    detector = AnnularDetector(inner=80, outer=200)

    probe = Probe(semiangle_cutoff=30, energy=80e3, gpts=500)

    measurement = probe.scan(linescan, detector, potential, max_batch=50, pbar=False)
    tds_measurement = probe.scan(linescan, detector, tds_potential, max_batch=50, pbar=False)
    assert np.allclose(measurement.array, tds_measurement.array, atol=1e-6)

    frozen_phonons = FrozenPhonons(atoms, 2, sigmas={'C': 0, 'O': 0.1})
    tds_potential = Potential(frozen_phonons, sampling=.05)
    tds_measurement = probe.scan(linescan, detector, tds_potential, max_batch=50, pbar=False)
    assert not np.allclose(measurement.array, tds_measurement.array, atol=1e-6)
