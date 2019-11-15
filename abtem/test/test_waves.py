import pytest
import numpy as np
from ..waves import Waves, PlaneWaves, ProbeWaves
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


@pytest.fixture
def potential():
    return DummyPotential(extent=5, gpts=50)


@pytest.fixture
def array():
    return np.ones((1, 25, 25), dtype=np.complex64)


def test_create_waves(array):
    waves = Waves(array)
    waves.extent = 10

    assert np.all(waves.sampling == 10 / array.shape[1])

    with pytest.raises(AttributeError):
        waves.gpts = 200

    waves = Waves(array, sampling=.2)
    assert np.all(waves.extent == array.shape[1] * .2)


def test_waves_raises(array):
    waves = Waves(array)

    with pytest.raises(RuntimeError):
        waves.check_is_grid_defined()

    waves = Waves(array, extent=5)

    with pytest.raises(RuntimeError):
        waves.check_is_energy_defined()

    waves = Waves(array, extent=5, energy=60e3)

    waves.check_is_grid_defined()
    waves.check_is_energy_defined()


def test_multislice(array):
    waves = Waves(array, energy=60e3)

    potential = DummyPotential(extent=5)

    new_waves = waves.multislice(potential)

    assert waves.extent is None
    assert np.any(new_waves.array != waves.array)

    new_waves = waves.multislice(potential, in_place=True)

    assert new_waves.extent is None
    assert np.all(new_waves.array == waves.array)


def test_multislice_raises(array):
    waves = Waves(array, extent=4, energy=60e3)
    potential = DummyPotential(extent=5)

    with pytest.raises(RuntimeError) as e:
        waves.multislice(potential)

    assert str(e.value) == 'inconsistent extent'

    waves = Waves(array, extent=5)
    with pytest.raises(RuntimeError) as e:
        waves.multislice(potential)

    assert str(e.value) == 'energy is not defined'

    waves.energy = 60e3
    waves.multislice(potential)


def test_create_plane_waves():
    plane_waves = PlaneWaves(2, gpts=10, energy=60e3)

    waves = plane_waves.build()

    assert np.all(waves.array == np.ones((2, 10, 10), dtype=np.complex))


def test_plane_waves_raises():
    plane_waves = PlaneWaves(2, energy=60e3)

    with pytest.raises(RuntimeError) as e:
        plane_waves.multislice(DummyPotential(extent=5))

    assert str(e.value) == 'gpts not defined'


def test_plane_waves_multislice():
    plane_waves = PlaneWaves(2, gpts=50, energy=60e3)
    plane_waves.multislice(DummyPotential(extent=5))

    plane_waves = PlaneWaves(2, sampling=.1, energy=60e3)
    plane_waves.multislice(DummyPotential(extent=5))


def test_create_probe_waves():
    probe_waves = ProbeWaves(extent=10, gpts=10, energy=60e3, defocus=10, C32=30, parameters={'C10': 100},
                             normalize=True)

    waves = probe_waves.build()

    assert waves.array.shape == (1, 10, 10)

    waves = probe_waves.build_at([[0, 0], [1, 1]])

    assert waves.array.shape == (2, 10, 10)


def test_probe_waves_raises():
    with pytest.raises(ValueError) as e:
        ProbeWaves(not_a_parameter=10)

    assert str(e.value) == 'not_a_parameter not a recognized parameter'

    probe_waves = ProbeWaves()
    with pytest.raises(RuntimeError) as e:
        probe_waves.build()

    assert str(e.value) == 'extent is not defined'

    probe_waves.extent = 10
    probe_waves.gpts = 100
    with pytest.raises(RuntimeError) as e:
        probe_waves.build()

    assert str(e.value) == 'energy is not defined'

    probe_waves.energy = 60e3
    probe_waves.build()


@pytest.fixture
def mocked_detector_base():
    with mock.patch.object(DetectorBase, 'detect') as detect:
        with mock.patch.object(DetectorBase, 'out_shape', (1,)):
            detect.side_effect = lambda x: np.array([1])
            yield DetectorBase


def test_probe_waves_custom_scan(potential, mocked_detector_base):
    probe_waves = ProbeWaves(energy=60e3)
    detectors = mocked_detector_base()
    positions = np.array([[1.25, 1.25], [3.75, 3.75]])
    scan = probe_waves.custom_scan(potential, detectors, positions=positions, show_progress=False)
    assert mocked_detector_base.detect.call_count == 2
    assert np.all(scan.measurements[detectors] == [1., 1.])


def test_probe_waves_line_scan(potential, mocked_detector_base):
    probe_waves = ProbeWaves(energy=60e3)
    detectors = mocked_detector_base()

    start = [0, 0]
    end = [1, 1]
    gpts = 2

    scan = probe_waves.line_scan(potential, detectors, start=start, end=end, gpts=gpts, show_progress=False)
    assert mocked_detector_base.detect.call_count == 2
    assert np.all(scan.measurements[detectors] == [1., 1.])


def test_probe_waves_grid_scan(potential, mocked_detector_base):
    probe_waves = ProbeWaves(energy=60e3)
    detectors = mocked_detector_base()

    start = [0, 0]
    end = [1, 1]
    gpts = 2

    scan = probe_waves.grid_scan(potential, detectors, start=start, end=end, gpts=gpts, show_progress=False)
    assert mocked_detector_base.detect.call_count == 4
    assert np.all(scan.measurements[detectors] == [[1., 1.], [1., 1.]])
