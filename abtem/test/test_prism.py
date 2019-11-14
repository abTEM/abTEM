import mock
import numpy as np
import pytest

from ..bases import Grid
from ..detect import DetectorBase
from ..waves import ProbeWaves, PrismWaves


@pytest.fixture
def mocked_detector_base():
    with mock.patch.object(DetectorBase, 'detect') as detect:
        with mock.patch.object(DetectorBase, 'out_shape', (1,)):
            detect.side_effect = lambda x: np.array([1])
            yield DetectorBase


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


def test_prism_custom_scan(mocked_detector_base):
    S = PrismWaves(.01, energy=60e3, sampling=.05).multislice(DummyPotential(extent=5))
    detectors = mocked_detector_base()
    positions = np.array([[1.25, 1.25], [3.75, 3.75]])
    scan = S.custom_scan(detectors, positions=positions, show_progress=False)
    assert mocked_detector_base.detect.call_count == 2
    assert np.all(scan.measurements[detectors] == [1., 1.])


def test_probe_waves_line_scan(mocked_detector_base):
    S = PrismWaves(.01, energy=60e3, sampling=.05).multislice(DummyPotential(extent=5))
    detectors = mocked_detector_base()

    start = [0, 0]
    end = [1, 1]
    gpts = 2

    scan = S.linescan(detectors, start=start, end=end, gpts=gpts, show_progress=False)
    assert mocked_detector_base.detect.call_count == 2
    assert np.all(scan.measurements[detectors] == [1., 1.])


def test_probe_waves_grid_scan(mocked_detector_base):
    S = PrismWaves(.01, energy=60e3, sampling=.05).multislice(DummyPotential(extent=5))
    detectors = mocked_detector_base()

    start = [0, 0]
    end = [1, 1]
    gpts = 2

    scan = S.gridscan(detectors, start=start, end=end, gpts=gpts, show_progress=False)
    assert mocked_detector_base.detect.call_count == 4
    assert np.all(scan.measurements[detectors] == [[1., 1.], [1., 1.]])
