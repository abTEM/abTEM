from copy import copy

import numpy as np
import pytest

from abtem.base_classes import Grid, AntialiasFilter
from abtem.detect import AbstractDetector
from abtem.measure import Measurement
from abtem.potentials import AbstractPotential, PotentialArray
from abtem.scan import LineScan, GridScan
from abtem.waves import Waves, PlaneWave, Probe


class DummyPotential(AbstractPotential):

    def __init__(self, extent=None, gpts=None, sampling=None, num_slices=10):
        self._num_slices = num_slices
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        super().__init__(precalculate=False)

    @property
    def num_frozen_phonon_configs(self):
        return 1

    def generate_frozen_phonon_potentials(self, pbar=False):
        for i in range(self.num_frozen_phonon_configs):
            yield self

    @property
    def num_slices(self):
        return 10

    def get_slice_thickness(self, i):
        return .5

    def generate_slices(self, start=0, end=None, max_batch=1):
        for i in range(self.num_slices):
            array = np.zeros(self.gpts, dtype=np.float32)
            array[:self.gpts[0] // 2] = 1
            yield start, end, PotentialArray(array[None],
                                             slice_thicknesses=np.array([self.get_slice_thickness(i)]),
                                             extent=self.extent)


@pytest.fixture
def potential():
    return DummyPotential(extent=5, gpts=50)


def test_create_waves():
    array = np.ones((1, 25, 25), dtype=np.complex64)
    waves = Waves(array)
    waves.extent = 10

    assert (waves.sampling[0] == 10 / array.shape[1]) & (waves.sampling[1] == 10 / array.shape[2])

    with pytest.raises(RuntimeError):
        waves.gpts = 200

    waves = Waves(array, sampling=.2)
    assert (waves.extent[0] == array.shape[1] * .2) & (waves.extent[1] == array.shape[2] * .2)


def test_waves_raises():
    array = np.ones((1, 25, 25), dtype=np.complex64)
    waves = Waves(array)

    with pytest.raises(RuntimeError):
        waves.grid.check_is_defined()

    waves = Waves(array, extent=5)

    with pytest.raises(RuntimeError):
        waves.accelerator.check_is_defined()

    waves = Waves(array, extent=5, energy=60e3)

    waves.grid.check_is_defined()
    waves.accelerator.check_is_defined()


def test_multislice():
    array = np.ones((1, 25, 25), dtype=np.complex64)
    waves = Waves(array, energy=60e3)

    potential = DummyPotential(extent=5)

    new_waves = waves.multislice(potential, pbar=False)

    assert potential.gpts is not None
    assert waves.extent is not None
    assert new_waves is waves

    new_waves = copy(new_waves)
    new_waves = new_waves.multislice(potential, pbar=False)

    assert potential.gpts is not None
    assert waves.extent is not None
    assert not np.all(np.isclose(new_waves.array, waves.array))


def test_multislice_raises():
    array = np.ones((1, 25, 25), dtype=np.complex64)
    potential = DummyPotential(extent=5)

    waves = Waves(array, extent=5)
    with pytest.raises(RuntimeError) as e:
        waves.multislice(potential, pbar=False)

    assert str(e.value) == 'Energy is not defined'

    waves.energy = 60e3
    waves.multislice(potential, pbar=False)


def test_create_plane_waves():
    plane_wave = PlaneWave(extent=2, gpts=10, energy=60e3)
    waves = plane_wave.build()
    assert np.all(waves.array == np.ones((10, 10), dtype=np.complex64))


def test_plane_waves_raises():
    plane_waves = PlaneWave(energy=60e3)

    with pytest.raises(RuntimeError) as e:
        plane_waves.multislice(DummyPotential(extent=5), pbar=False)

    assert str(e.value) == 'Grid gpts cannot be inferred'


def test_plane_waves_multislice():
    plane_waves = PlaneWave(gpts=50, energy=60e3)
    plane_waves.multislice(DummyPotential(extent=5))

    plane_waves = PlaneWave(sampling=.1, energy=60e3)
    plane_waves.multislice(DummyPotential(extent=5))


def test_create_probe_waves():
    probe_waves = Probe(extent=10, gpts=10, energy=60e3, defocus=10, C32=30, **{'C10': 100})

    waves = probe_waves.build()

    assert waves.array.shape == (1, 10, 10)

    waves = probe_waves.build([[0, 0], [1, 1]])

    assert waves.array.shape == (2, 10, 10)


def test_probe_waves_raises():
    with pytest.raises(ValueError) as e:
        Probe(not_a_parameter=10)

    assert str(e.value) == 'not_a_parameter not a recognized parameter'

    probe = Probe()
    with pytest.raises(RuntimeError) as e:
        probe.build()

    assert str(e.value) == 'Grid extent is not defined'

    probe.extent = 10
    probe.gpts = 100
    with pytest.raises(RuntimeError) as e:
        probe.build()

    assert str(e.value) == 'Energy is not defined'

    probe.energy = 60e3
    probe.build()


def test_downsample():
    f = AntialiasFilter()

    sampling = (.1, .1)
    gpts = (228, 229)

    mask = f.get_mask(gpts, sampling, np)
    n = np.sum(mask > 0.)

    array = np.fft.ifft2(mask)

    waves = Waves(array, sampling=sampling, energy=80e3, antialias_aperture=(2 / 3.,) * 2)

    assert np.allclose(waves.downsample('valid', return_fourier_space=True).array.real, 1.)
    #assert not np.allclose(waves.downsample('limit', return_fourier_space=True).array.real, 1.)
    #assert np.sum(waves.downsample('limit', return_fourier_space=True).array.real > 1e-6) == n


class DummyDetector(AbstractDetector):

    def __init__(self):
        self._detect_count = 0
        super().__init__()

    def detect(self, waves):
        self._detect_count += 1
        return np.array([1.])

    def allocate_measurement(self, waves, scan):
        array = np.zeros(scan.shape, dtype=np.float32)
        return Measurement(array, calibrations=scan.calibrations)


def test_probe_waves_line_scan():
    probe = Probe(energy=60e3)
    detector = DummyDetector()
    potential = DummyPotential(extent=5, sampling=.1)

    scan = LineScan((0, 0), (1, 1), gpts=10)
    measurement = probe.scan(scan, detector, potential, max_batch=1, pbar=False)

    assert detector._detect_count == 10
    assert np.all(measurement.array == 1.)

    measurement = probe.scan(scan, [detector], potential, pbar=False)
    assert detector._detect_count == 11
    assert np.all(measurement.array == 1.)

    measurement = probe.scan(scan, detector, potential, max_batch=3, pbar=False)
    assert detector._detect_count == 15
    assert np.all(measurement.array == 1.)


def test_probe_waves_grid_scan():
    probe = Probe(energy=60e3)
    detector = DummyDetector()
    potential = DummyPotential(extent=5, sampling=.1)

    scan = GridScan((0, 0), (1, 1), gpts=10)
    measurement = probe.scan(scan, detector, potential, max_batch=1, pbar=False)

    assert detector._detect_count == 100
    assert np.all(measurement.array == 1.)
