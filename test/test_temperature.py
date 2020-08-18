import numpy as np
import pytest
from ase import Atoms

from abtem.detect import AnnularDetector
from abtem.potentials import Potential
from abtem.scan import LineScan
from abtem.waves import Probe, SMatrix
from abtem.device import asnumpy, cp
from abtem import FrozenPhonons


def test_frozen_phonons_raise():
    atoms = Atoms('CO', positions=[(2.5, 2.5, 2), (2.5, 2.5, 3)], cell=(5, 5, 4))

    with pytest.raises(RuntimeError) as e:
        frozen_phonons = FrozenPhonons(atoms, 10, sigmas={'C': .1})

    assert str(e.value) == 'Displacement standard deviation not provided for all atomic species.'


def test_probe_waves_line_scan():
    atoms = Atoms('CO', positions=[(2.5, 2.5, 2), (2.5, 2.5, 3)], cell=(5, 5, 4))
    frozen_phonons = FrozenPhonons(atoms, 2, sigmas={'C': 0, 'O': 0.})

    potential = Potential(atoms, sampling=.05)
    tds_potential = Potential(frozen_phonons, sampling=.05)

    linescan = LineScan(start=[0, 0], end=[2.5, 2.5], gpts=10)
    detector = AnnularDetector(inner=80, outer=200)

    probe = Probe(semiangle_cutoff=30, energy=80e3, gpts=500)

    measurements = probe.scan(linescan, [detector], potential, max_batch=50, pbar=False)
    tds_measurements = probe.scan(linescan, [detector], tds_potential, max_batch=50, pbar=False)
    assert np.allclose(measurements[detector].array, tds_measurements[detector].array, atol=1e-6)

    frozen_phonons = FrozenPhonons(atoms, 2, sigmas={'C': 0, 'O': 0.1})
    tds_potential = Potential(frozen_phonons, sampling=.05)
    tds_measurements = probe.scan(linescan, [detector], tds_potential, max_batch=50, pbar=False)
    assert not np.allclose(measurements[detector].array, tds_measurements[detector].array, atol=1e-6)