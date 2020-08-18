import numpy as np
import pytest
from ase import Atoms

from abtem.detect import AnnularDetector
from abtem.potentials import Potential
from abtem.scan import LineScan
from abtem.waves import Probe, SMatrix
from abtem.device import asnumpy, cp


def test_prism_raises():
    with pytest.raises(ValueError) as e:
        SMatrix(.01, .5)

    assert str(e.value) == 'Interpolation factor must be int'

    with pytest.raises(RuntimeError) as e:
        prism = SMatrix(10, 1)
        prism.build()

    assert str(e.value) == 'Grid extent is not defined'

    with pytest.raises(RuntimeError) as e:
        prism = SMatrix(10, 1, extent=10, gpts=100)
        prism.build()

    assert str(e.value) == 'Energy is not defined'


def test_prism_match_probe():
    S_builder = SMatrix(30., 1, extent=5, gpts=101, energy=60e3)
    probe = Probe(extent=5, gpts=101, energy=60e3, semiangle_cutoff=30., rolloff=0.)
    assert np.allclose(probe.build([(0., 0.)]).array, S_builder.build().collapse([(0., 0.)]).array, atol=2e-5)


def test_prism_translate():
    S_builder = SMatrix(30, 1, extent=5, gpts=50, energy=60e3)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=30, rolloff=0)
    assert np.allclose(probe.build(np.array([(2.5, 2.5)])).array,
                       S_builder.build().collapse([(2.5, 2.5)]).array, atol=1e-5)


def test_prism_interpolation():
    S_builder = SMatrix(30, 2, extent=10, gpts=100, energy=60e3)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=30, rolloff=0)
    assert np.allclose(probe.build(np.array([(2.5, 2.5)])).array,
                       S_builder.build().collapse([(2.5, 2.5)]).array, atol=1e-5)


def test_prism_multislice():
    potential = Potential(Atoms('C', positions=[(0, 0, 2)], cell=(5, 5, 4)))
    S = SMatrix(30, 1, extent=5, gpts=500, energy=60e3)
    probe = Probe(extent=5, gpts=500, energy=60e3, semiangle_cutoff=30, rolloff=0.)
    assert np.allclose(probe.build(np.array([[2.5, 2.5]])).multislice(potential, pbar=False).array,
                       S.multislice(potential, pbar=False).collapse([(2.5, 2.5)]).array, atol=2e-5)


def test_probe_waves_line_scan():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))
    linescan = LineScan(start=[0, 0], end=[2.5, 2.5], gpts=10)
    detector = AnnularDetector(inner=80, outer=200)

    S_builder = SMatrix(30, 1, energy=80e3, gpts=500)
    S = S_builder.multislice(potential, pbar=False)
    probe = Probe(semiangle_cutoff=30, energy=80e3, gpts=500)

    prism_measurements = S.scan(linescan, [detector], max_batch_probes=10, pbar=False)
    measurements = probe.scan(linescan, [detector], potential, max_batch=50, pbar=False)

    assert np.allclose(measurements[detector].array, prism_measurements[detector].array, atol=1e-6)


def test_prism_batch():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))

    S_builder = SMatrix(30, 1, energy=80e3, gpts=500)
    S1 = S_builder.multislice(potential, pbar=False)
    S2 = S_builder.multislice(potential, max_batch=5, pbar=False)

    assert np.allclose(S1.array, S2.array)


@pytest.mark.gpu
def test_prism_gpu():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))

    S_builder = SMatrix(30, 1, energy=80e3, gpts=500, device='cpu')
    S_cpu = S_builder.multislice(potential, pbar=False)

    assert type(S_cpu.array) is np.ndarray

    S_builder = SMatrix(30, 1, energy=80e3, gpts=500, device='gpu')
    S_gpu = S_builder.multislice(potential, pbar=False)

    assert type(S_gpu.array) is cp.ndarray
    assert np.allclose(S_cpu.array, asnumpy(S_gpu.array))


@pytest.mark.gpu
def test_prism_storage():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))

    S_builder = SMatrix(30, 1, energy=80e3, gpts=500, device='gpu', storage='cpu')
    S_gpu = S_builder.multislice(potential, pbar=False)

    assert type(S_gpu.array) is np.ndarray
