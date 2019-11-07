import pytest
import numpy as np
from ..waves import Waves, PlaneWaves, ProbeWaves, PrismWaves
from ..bases import Grid
from ..detect import DetectorBase
import mock


class DummyPotential(Grid):

    def __init__(self, extent=None, gpts=None, sampling=None, num_slices=10):
        self._num_slices = num_slices
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=2)

    @property
    def num_slices(self):
        return 10

    def slice_thickness(self, i):
        return .5

    def get_slice(self, i):
        array = np.zeros(self.gpts, dtype=np.float32)
        array[:self.gpts[0] // 2] = 1
        return array

    def copy(self, copy_atoms=False):
        return self


def test_prism_raises():
    with pytest.raises(ValueError) as e:
        PrismWaves(.01, .5)

    assert str(e.value) == 'interpolation factor must be int'

    with pytest.raises(RuntimeError) as e:
        prism = PrismWaves(.01, 1)
        prism.build()

    assert str(e.value) == 'extent is not defined'

    with pytest.raises(RuntimeError) as e:
        prism = PrismWaves(.01, 1, extent=10, gpts=100)
        prism.build()

    assert str(e.value) == 'energy is not defined'


def test_prism():
    prism = PrismWaves(.01, 1, extent=5, gpts=50, energy=60e3)
    probe = ProbeWaves(extent=5, gpts=50, energy=60e3, cutoff=.01)
    assert np.allclose(prism.build().build().array, probe.build().array)


def test_prism_translate():
    S = PrismWaves(.01, 1, extent=5, gpts=50, energy=60e3).build()
    probe = ProbeWaves(extent=5, gpts=50, energy=60e3, cutoff=.01)

    probe_waves = probe.build_at(np.array([(0, 0)]))
    prism_waves = S.build_at(np.array([(0, 0)]))

    assert np.allclose(probe_waves.array, prism_waves.array)

    probe_waves = probe.build_at(np.array([(2.5, 2.5)]))
    prism_waves = S.build_at(np.array([(2.5, 2.5)]))

    assert np.allclose(probe_waves.array, prism_waves.array)


def test_prism_interpolation():
    S = PrismWaves(.01, 2, extent=10, gpts=100, energy=60e3).build()
    probe = ProbeWaves(extent=5, gpts=50, energy=60e3, cutoff=.01)

    probe_waves = probe.build_at(np.array([(2.5, 2.5)]))
    prism_waves = S.build_at(np.array([(0, 0)]))

    assert np.allclose(probe_waves.array, prism_waves.array)


def test_prism_multislice():
    S = PrismWaves(.01, 1, extent=5, gpts=100, energy=60e3).build()
    probe = ProbeWaves(extent=5, gpts=100, energy=60e3, cutoff=.01)

    potential = DummyPotential(extent=5)

    S = S.multislice(potential)

    prism_waves = S.build_at(np.array([[2.5, 2.5]]))
    probe_waves = probe.build_at(np.array([[2.5, 2.5]])).multislice(potential)

    assert np.allclose(probe_waves.array, prism_waves.array)
