import numpy as np
import pytest

from abtem.scan import PositionScan, LineScan, GridScan


def test_custom_scan():
    positions = np.array([(2, 2), (1, 1)])
    scan = PositionScan(positions=positions)

    assert np.all(next(scan.generate_positions(1))[2] == positions[0])
    assert np.all(next(scan.generate_positions(2))[2] == positions)


def test_linescan_raises():
    with pytest.raises(ValueError) as e:
        LineScan(start=0, end=1)

    assert str(e.value) == 'scan start/end has wrong shape'


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

    assert str(e.value) == 'scan start/end has wrong shape'


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
