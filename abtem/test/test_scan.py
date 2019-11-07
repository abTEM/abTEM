import mock
import numpy as np
import pytest

from ..scan import ScanBase, CustomScan, LineScan, GridScan, assemble_partitions_2d


def test_scan_base():
    scan = ScanBase()
    positions = np.array([(2, 2), (1, 1)])
    scan.get_positions = mock.Mock(return_value=positions)

    for i, (start, size, position) in enumerate(scan.generate_positions(1)):
        assert size == 1
        assert np.all(positions[i] == position[0])

    start, size, gen_positions = next(scan.generate_positions(2))
    assert size == 2
    assert np.all(gen_positions == positions)

    gen_positions = next(scan.generate_positions(3))[2]
    assert np.all(gen_positions == positions)


def test_custom_scan():
    positions = np.array([(2, 2), (1, 1)])
    scan = CustomScan(positions=positions)

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
    assert np.all(positions[0] == start)
    assert np.all(positions[-1] == end)
    assert np.all(positions[2] == np.mean([start, end], axis=0))
    assert np.allclose(np.linalg.norm(np.diff(positions, axis=0), axis=1), scan.sampling[0])

    scan = LineScan(start, end, gpts=5, endpoint=False)
    positions = scan.get_positions()
    assert np.all(positions[0] == start)
    assert np.allclose(positions[-1], (end[0] - (end[0] - start[0]) / 5, end[1] - (end[1] - start[1]) / 5))


def test_partition_line_scan():
    start = (0, 0)
    end = (1, 1)
    for endpoint in [False, True]:
        scan = LineScan(start, end, gpts=5, endpoint=endpoint)
        scans = scan.partition(2)
        assert np.allclose(scan.get_positions(), np.vstack((scans[0].get_positions(), scans[1].get_positions())))


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


def test_partition_grid_scan():
    start = (0, 0)
    end = (1, 2)
    scan = GridScan(start=start, end=end, gpts=(4, 5))

    positions = scan.get_positions()

    partitions = {}
    for key, partition_scan in scan.partition((2, 2)).items():
        partitions[key] = partition_scan.get_positions().reshape(tuple(partition_scan.gpts) + (2,))

    assembled = assemble_partitions_2d(partitions)
    assert np.allclose(positions.reshape(tuple(scan.gpts) + (2,)), assembled)
