"""Tests for abtem.core.axes, focused on the offset of LinearAxis under indexing
and dask ensemble chunk partitioning."""

import numpy as np
import pytest

from abtem import GridScan, Probe
from abtem.core.axes import LinearAxis, OrdinalAxis, RealSpaceAxis, ScanAxis
from abtem.core.chunks import iterate_chunk_ranges, validate_chunks


@pytest.mark.parametrize("axis_cls", [LinearAxis, RealSpaceAxis, ScanAxis])
def test_linear_axis_getitem_shifts_offset(axis_cls):
    axis = axis_cls(offset=10.0, sampling=0.5)

    sliced = axis[3:6]

    assert sliced.offset == pytest.approx(10.0 + 3 * 0.5)
    assert sliced.sampling == axis.sampling


def test_linear_axis_getitem_with_int():
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    assert axis[4].offset == pytest.approx(10.0 + 4 * 0.5)


def test_linear_axis_getitem_with_length_one_tuple():
    """`iterate_chunk_ranges` yields a tuple of slices even when partitioning a
    single axis, which must be handled the same as the bare slice."""
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    assert axis[(slice(3, 6),)].offset == axis[3:6].offset


@pytest.mark.parametrize(
    "bad_item",
    [slice(None, None, -1), slice(None, None, 2), slice(-3, None), slice(5, 2), -1],
)
def test_linear_axis_getitem_rejects_unsupported_slices(bad_item):
    """Strided, negative or empty slices cannot be represented by shifting the
    offset. This must raise TypeError, which `_get_ensemble_axes_metadata_items`
    catches to fall back to a plain copy of the axis."""
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    with pytest.raises(TypeError):
        axis[bad_item]


def test_linear_axis_getitem_rejects_unsupported_types():
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    with pytest.raises(TypeError):
        axis[np.array([0, 2, 3])]


def test_linear_axis_to_blocks_partitions_offset_per_chunk():
    """Each chunk must get its own, chunk-local offset instead of an identical
    copy of the global one."""
    axis = RealSpaceAxis(offset=100.0, sampling=0.25)
    chunks = validate_chunks(shape=(8,), chunks=((3, 3, 2),))

    blocks = axis._to_blocks(chunks).compute()

    assert len(blocks) == 3
    for block, start in zip(blocks, [0, 3, 6]):
        assert block.offset == pytest.approx(100.0 + start * 0.25)
        assert block.sampling == axis.sampling


def test_linear_axis_to_blocks_matches_direct_coordinates():
    """Each chunk's axis must reproduce the matching slice of the full axis'
    coordinates."""
    n = 11
    axis = ScanAxis(offset=-2.5, sampling=0.4)
    full_coordinates = np.array(axis.coordinates(n))

    chunks = validate_chunks(shape=(n,), chunks=((3, 4, 4),))
    blocks = axis._to_blocks(chunks).compute()

    for (_, (slic,)), block in zip(iterate_chunk_ranges(chunks), blocks):
        block_coordinates = np.array(block.coordinates(slic.stop - slic.start))
        np.testing.assert_allclose(block_coordinates, full_coordinates[slic])


def test_ordinal_axis_still_partitions_correctly():
    values = tuple(range(10, 20))
    axis = OrdinalAxis(values=values)
    chunks = validate_chunks(shape=(len(values),), chunks=((4, 4, 2),))

    blocks = axis._to_blocks(chunks).compute()

    assert [b.values for b in blocks] == [values[0:4], values[4:8], values[8:10]]


def test_indexing_scan_ensemble_axis_shifts_offset():
    """Slicing a scan ensemble axis of a measurement must keep the offset of
    the first retained position."""
    scan = GridScan((1.0, 2.0), (9.0, 10.0), gpts=(8, 8), endpoint=False)
    probe = Probe(energy=100e3, semiangle_cutoff=20, sampling=0.2, extent=10)
    waves = probe.build(scan=scan, lazy=False)

    sliced = waves[2:5, 3:7]

    assert sliced.ensemble_axes_metadata[0].offset == pytest.approx(1.0 + 2 * 1.0)
    assert sliced.ensemble_axes_metadata[1].offset == pytest.approx(2.0 + 3 * 1.0)
