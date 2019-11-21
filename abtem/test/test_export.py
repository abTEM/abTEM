import os

from abtem.detect import FourierSpaceDetector, RingDetector
from abtem.scan import GridScan, LineScan, load_measurements


def test_export_fourier_space_grid_scan(tmpdir):
    path = tmpdir.join('test.hdf5')

    scan = GridScan(start=(0, 0), end=(10, 10), gpts=(9, 10))
    det = FourierSpaceDetector(extent=10, gpts=(11, 12), export=str(path))
    scan.open_measurements(det)

    assert len(os.listdir(tmpdir)) == 1
    measurements = load_measurements(str(path))
    assert measurements.shape == (9, 10, 11, 12)


def test_export_ring_detector_grid_scan(tmpdir):
    path = tmpdir.join('test.hdf5')

    scan = GridScan(start=(0, 0), end=(10, 10), gpts=(9, 10))
    det = RingDetector(inner=.1, outer=.2, gpts=(11, 12), export=str(path))
    scan.open_measurements(det)

    assert len(os.listdir(tmpdir)) == 1
    measurements = load_measurements(str(path))
    assert measurements.shape == (9, 10)


def test_export_fourier_space_line_scan(tmpdir):
    path = tmpdir.join('test.hdf5')

    scan = LineScan(start=(0, 0), end=(10, 10), gpts=10)
    det = FourierSpaceDetector(extent=10, gpts=(11, 12), export=str(path))
    scan.open_measurements(det)

    assert len(os.listdir(tmpdir)) == 1
    measurements = load_measurements(str(path))
    assert measurements.shape == (10, 11, 12)


def test_export_ring_detector_line_scan(tmpdir):
    path = tmpdir.join('test.hdf5')

    scan = LineScan(start=(0, 0), end=(10, 10), gpts=10)
    det = RingDetector(inner=.1, outer=.2, gpts=(11, 12), export=str(path))
    scan.open_measurements(det)

    assert len(os.listdir(tmpdir)) == 1
    measurements = load_measurements(str(path))
    assert measurements.shape == (10,)
