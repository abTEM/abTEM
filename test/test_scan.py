import os

import numpy as np
import pytest
from abtem.detect import AnnularDetector, PixelatedDetector
from abtem.potentials import Potential
from abtem.scan import PositionScan, LineScan, GridScan
from abtem.waves import Probe
from ase.io import read


def test_custom_scan():
    positions = np.array([(2, 2), (1, 1)])
    scan = PositionScan(positions=positions)

    assert np.all(next(scan.generate_positions(1))[1] == positions[0])
    assert np.all(next(scan.generate_positions(2))[1] == positions)


def test_line_scan():
    start = (0, 0)
    end = (1, 1)
    scan = LineScan(start, end, gpts=5)

    positions = scan.get_positions()
    assert np.allclose(positions[0], start)
    assert np.allclose(positions[-1], end)
    assert np.allclose(positions[2], np.mean([start, end], axis=0))
    assert np.allclose(np.linalg.norm(np.diff(positions, axis=0), axis=1), scan.sampling[0])

    scan = LineScan(start, end, gpts=5, endpoint=False)
    positions = scan.get_positions()
    assert np.allclose(positions[0], start)
    assert np.allclose(positions[-1], (end[0] - (end[0] - start[0]) / 5, end[1] - (end[1] - start[1]) / 5))


def test_gridscan_raises():
    with pytest.raises(ValueError) as e:
        GridScan(start=0, end=1)

    assert str(e.value) == 'Scan start/end has incorrect shape'


def test_grid_scan():
    start = (0, 0)
    end = (1, 2)
    scan = GridScan(start, end, gpts=5, endpoint=True)

    positions = scan.get_positions()
    assert np.all(positions[0] == start)
    assert np.all(positions[-1] == end)
    assert np.allclose(positions[4], [start[1], end[1]])
    assert np.allclose(np.linalg.norm(np.diff(positions[:4], axis=0), axis=1), scan.sampling[1])

    scan = GridScan(start, end, gpts=5, endpoint=False)
    positions = scan.get_positions()
    assert np.all(positions[0] == start)
    assert np.allclose(positions[-1], (end[0] - (end[0] - start[0]) / 5, end[1] - (end[1] - start[1]) / 5))


def test_partition():
    gridscan = GridScan(start=(0, 0), end=(2, 2), sampling=.5, endpoint=False)
    scans = gridscan.partition_scan((2, 2))
    positions = []
    for scan in scans:
        positions.append(scan.get_positions())

    assert np.allclose(((np.vstack(positions)[None] - gridscan.get_positions()[:, None]) ** 2).min(0), 0)
    assert np.allclose(((np.vstack(positions)[None] - gridscan.get_positions()[:, None]) ** 2).min(1), 0)

    gridscan = GridScan(start=(0, 0), end=(2, 2), sampling=.5, endpoint=True)
    scans = gridscan.partition_scan((2, 2))
    positions = []
    for scan in scans:
        positions.append(scan.get_positions())

    assert np.allclose(((np.vstack(positions)[None] - gridscan.get_positions()[:, None]) ** 2).min(0), 0)
    assert np.allclose(((np.vstack(positions)[None] - gridscan.get_positions()[:, None]) ** 2).min(1), 0)


def test_partition_measurement():
    atoms = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/amorphous_carbon.cif'))
    potential = Potential(atoms, gpts=256, slice_thickness=1, projection='infinite',
                          parametrization='kirkland').build(pbar=False)

    detector = AnnularDetector(inner=70, outer=100)
    gridscan = GridScan(start=[0, 0], end=potential.extent, gpts=4)

    probe = Probe(semiangle_cutoff=15, energy=300e3)

    measurements = probe.scan(gridscan, detector, potential, pbar=False)

    scans = gridscan.partition_scan((2, 2))
    partitioned_measurements = detector.allocate_measurement(probe, gridscan)

    for scan in scans:
        probe.scan(scan, detector, potential, measurements=partitioned_measurements, pbar=False)

    assert np.allclose(partitioned_measurements.array, measurements.array)


def test_preallocated_measurement():
    atoms = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/amorphous_carbon.cif'))
    potential = Potential(atoms, gpts=256, slice_thickness=1, projection='infinite',
                          parametrization='kirkland').build(pbar=False)
    scan = GridScan(start=[0, 0], end=potential.extent, gpts=4)

    detector1 = AnnularDetector(inner=70, outer=100)
    probe = Probe(semiangle_cutoff=15, energy=300e3, extent=potential.extent, gpts=512)

    measurement = detector1.allocate_measurement(probe, scan)
    probe.scan(scan, detector1, potential, measurement, pbar=False)

    assert np.any(measurement.array > 0)

    detector2 = PixelatedDetector()

    measurement1 = detector1.allocate_measurement(probe, scan)
    measurement2 = detector2.allocate_measurement(probe, scan)

    with pytest.raises(ValueError) as e:
        probe.scan(scan, [detector1, detector2], potential, measurement1, pbar=False)

    probe.scan(scan, [detector1, detector2], potential, {detector1: measurement1, detector2: measurement2}, pbar=False)
