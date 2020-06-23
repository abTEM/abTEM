import numpy as np
import pytest
from ase import Atoms

from abtem.detect import AnnularDetector
from abtem.potentials import Potential
from abtem.scan import LineScan
from abtem.waves import Probe, SMatrixBuilder


def test_prism_raises():
    with pytest.raises(ValueError) as e:
        SMatrixBuilder(.01, .5)

    assert str(e.value) == 'interpolation factor must be int'

    with pytest.raises(RuntimeError) as e:
        prism = SMatrixBuilder(.01, 1)
        prism.build()

    assert str(e.value) == 'grid extent is not defined'

    with pytest.raises(RuntimeError) as e:
        prism = SMatrixBuilder(.01, 1, extent=10, gpts=100)
        prism.build()

    assert str(e.value) == 'energy is not defined'


def test_prism_match_probe():
    S_builder = SMatrixBuilder(.01, 1, extent=5, gpts=101, energy=60e3)
    probe = Probe(extent=5, gpts=101, energy=60e3, semiangle_cutoff=.01)
    assert np.allclose(np.fft.fftshift(probe.build([(0., 0.)]).array),
                       S_builder.build().collapse([(0., 0.)]).array, atol=1e-6)


def test_prism_translate():
    S_builder = SMatrixBuilder(.01, 1, extent=5, gpts=50, energy=60e3)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=.01)
    assert np.allclose(probe.build(np.array([(2.5, 2.5)])).array,
                       S_builder.build().collapse([(2.5, 2.5)]).array, atol=1e-6)


def test_prism_interpolation():
    S_builder = SMatrixBuilder(.01, 2, extent=10, gpts=100, energy=60e3)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=.01)
    assert np.allclose(probe.build(np.array([(2.5, 2.5)])).array,
                       S_builder.build().collapse([(2.5, 2.5)]).array, atol=1e-6)


def test_prism_multislice():
    potential = Potential(Atoms('C', positions=[(0, 0, 2)], cell=(5, 5, 4)))
    S = SMatrixBuilder(.03, 1, extent=5, gpts=500, energy=60e3).build()
    probe = Probe(extent=5, gpts=500, energy=60e3, semiangle_cutoff=.03)
    assert np.allclose(probe.build(np.array([[2.5, 2.5]])).multislice(potential).array,
                       S.multislice(potential).collapse([(2.5, 2.5)]).array, atol=2e-5)


def test_probe_waves_line_scan():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))
    linescan = LineScan(start=[0, 0], end=[2.5, 2.5], gpts=10)
    detector = AnnularDetector(inner=.0, outer=.02)

    S_builder = SMatrixBuilder(.03, 1, energy=80e3, gpts=500)
    S = S_builder.multislice(potential)
    probe = Probe(semiangle_cutoff=.03, energy=80e3, gpts=500)

    prism_measurements = S.scan(linescan, [detector], max_batch=10, show_progress=False)
    measurements = probe.scan(linescan, [detector], potential, max_batch=50, show_progress=False)

    assert np.allclose(measurements[detector].array, prism_measurements[detector].array, atol=1e-6)
