"""Tests for abtem.core.axes, focused on LinearAxis (offset + sampling)
metadata under dask ensemble chunk partitioning.

Background: `ArrayObject._partition_ensemble_axes_metadata` and
`AxisMetadata._to_blocks` both partition an axis's metadata per dask chunk
via `axis[slic] if hasattr(axis, "__getitem__") else axis.copy()`.
`OrdinalAxis` (an explicit `values` tuple) has always supported this
correctly. `LinearAxis` and its subclasses (`RealSpaceAxis`, `ScanAxis`,
`ReciprocalSpaceAxis`) previously had no `__getitem__` at all, so every
chunk silently received an identical copy of the GLOBAL axis, including its
global `offset` -- any code that read a chunk's own axis metadata to recover
where that chunk sits (rather than just its size) got the wrong answer for
every chunk but the first, without any error.
"""
import dataclasses

import numpy as np
import pytest

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

    sliced = axis[4]

    assert sliced.offset == pytest.approx(10.0 + 4 * 0.5)


def test_linear_axis_getitem_with_length_one_tuple():
    """`iterate_chunk_ranges` always yields a tuple of slices, even when
    partitioning a single axis on its own -- this is what `_to_blocks`
    passes through, so it must be handled the same as the bare slice."""
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    assert axis[(slice(3, 6),)].offset == axis[3:6].offset


@pytest.mark.parametrize(
    "bad_item",
    [slice(None, None, -1), slice(None, None, 2), slice(-3, None), slice(5, 2)],
)
def test_linear_axis_getitem_rejects_unsupported_slices(bad_item):
    """Strided, negative, or empty slices cannot be represented by shifting
    `offset` alone. This must raise TypeError specifically (not e.g.
    NotImplementedError): `ArrayObject._get_ensemble_axes_metadata_items`
    catches TypeError to fall back to a plain copy for whatever an axis
    can't represent exactly -- the same fallback this axis relied on before
    it had a `__getitem__` at all. Any other exception type would break
    general indexing instead of falling back gracefully."""
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    with pytest.raises(TypeError):
        axis[bad_item]


def test_linear_axis_getitem_rejects_unsupported_types():
    axis = RealSpaceAxis(offset=10.0, sampling=0.5)

    with pytest.raises(TypeError):
        axis[np.array([0, 2, 3])]


def test_linear_axis_to_blocks_partitions_offset_per_chunk():
    """The regression this file exists for: partitioning a LinearAxis into
    multiple dask chunks must give each chunk its own, chunk-local offset --
    not an identical copy of the global one."""
    n = 8
    axis = RealSpaceAxis(offset=100.0, sampling=0.25)
    chunks = validate_chunks(shape=(n,), chunks=((3, 3, 2),))

    blocks = axis._to_blocks(chunks).compute()

    assert len(blocks) == 3
    starts = [0, 3, 6]
    for block, start in zip(blocks, starts):
        assert block.offset == pytest.approx(100.0 + start * 0.25)
        assert block.sampling == axis.sampling


def test_linear_axis_to_blocks_matches_direct_coordinates():
    """Cross-check against the axis's own `coordinates()`: each chunk's
    partitioned axis, re-expanded, must reproduce exactly the corresponding
    slice of the full axis's coordinates."""
    n = 11
    axis = ScanAxis(offset=-2.5, sampling=0.4)
    full_coordinates = np.array(axis.coordinates(n))

    chunks = validate_chunks(shape=(n,), chunks=((3, 4, 4),))
    blocks = axis._to_blocks(chunks).compute()

    for (_, (slic,)), block in zip(iterate_chunk_ranges(chunks), blocks):
        chunk_len = slic.stop - slic.start
        block_coordinates = np.array(block.coordinates(chunk_len))
        np.testing.assert_allclose(block_coordinates, full_coordinates[slic])


def test_ordinal_axis_still_partitions_correctly():
    """OrdinalAxis relied on `__getitem__` (via numpy indexing an explicit
    `values` array) before this fix and must be unaffected by it."""
    values = tuple(range(10, 20))
    axis = OrdinalAxis(values=values)
    chunks = validate_chunks(shape=(len(values),), chunks=((4, 4, 2),))

    blocks = axis._to_blocks(chunks).compute()

    assert [b.values for b in blocks] == [
        values[0:4],
        values[4:8],
        values[8:10],
    ]


def test_linear_axis_copy_still_works():
    """`.copy()` (used by the general fallback and elsewhere) must remain
    unaffected by adding `__getitem__`."""
    axis = RealSpaceAxis(offset=10.0, sampling=0.5, label="x")
    copied = axis.copy()

    assert copied is not axis
    assert dataclasses.asdict(copied) == dataclasses.asdict(axis)
