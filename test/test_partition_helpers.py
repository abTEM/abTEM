"""Tests of the ensemble partition helpers in abtem.core.ensemble: a lazy partition
must deliver the same entries to its tasks as an eager one delivers to its blocks."""

import dask
import dask.array as da
import numpy as np
import pytest

from abtem.core.chunks import validate_chunks
from abtem.core.ensemble import (
    partition_blocks,
    partition_entries,
    unpack_blockwise_args,
)


def _unpack(block):
    # what a task's reconstruction function does with its argument
    return unpack_blockwise_args((block,))[0]


def _eager_entries(partition):
    return {
        index: dask.compute(_unpack(partition[index]))[0]
        for index in np.ndindex(partition.shape)
    }


def _lazy_entries(partition):
    assert isinstance(partition, da.Array)
    # one chunk, and so one task, per entry
    assert all(c == (1,) * len(c) for c in partition.chunks)
    blocks = partition.to_delayed()
    return {
        index: _unpack(blocks[index].compute()) for index in np.ndindex(partition.shape)
    }


def _assert_same_entries(eager, lazy):
    eager_entries = _eager_entries(eager)
    lazy_entries = _lazy_entries(lazy)
    assert eager_entries.keys() == lazy_entries.keys()
    for index, entry in eager_entries.items():
        assert repr(entry) == repr(lazy_entries[index]), index


SHARED = dask.delayed(lambda: "shared")()

CASES = [
    ("one axis", (10,), 3, lambda s: {"members": np.arange(10)[s[0]]}),
    ("one axis, shared node", (10,), 4, lambda s: (SHARED, s[0].start, s[0].stop)),
    ("one axis, dask array", (10,), 3, lambda s: da.arange(10)[s[0]]),
    ("two axes", (4, 6), (2, 3), lambda s: (s[0].start, s[1].start)),
    ("two axes, shared node", (4, 6), (3, 2), lambda s: (SHARED, s[1].stop)),
    ("no axes", (), (), lambda s: {"value": 1.0}),
    ("no axes, shared node", (), (), lambda s: (SHARED,)),
]


@pytest.mark.parametrize(
    "shape, chunks, entry", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_partition_blocks_lazy_matches_eager(shape, chunks, entry):
    chunks = validate_chunks(shape, chunks)
    eager = partition_blocks(chunks, entry, lazy=False)
    lazy = partition_blocks(chunks, entry, lazy=True)

    assert isinstance(eager, np.ndarray)
    assert eager.shape == tuple(len(c) for c in chunks)
    assert lazy.shape == eager.shape
    _assert_same_entries(eager, lazy)


def test_partition_blocks_passes_member_slices():
    chunks = validate_chunks((7, 5), ((3, 4), (2, 3)))
    seen = []
    partition_blocks(chunks, lambda s: seen.append(s), lazy=False)
    assert seen == [
        (slice(0, 3), slice(0, 2)),
        (slice(0, 3), slice(2, 5)),
        (slice(3, 7), slice(0, 2)),
        (slice(3, 7), slice(2, 5)),
    ]


@pytest.mark.parametrize("lazy", [False, True])
def test_partition_entries_keeps_tuples_and_arrays_whole(lazy):
    # an entry may be a tuple or an array: it must not be spread over the
    # object array that holds the entries
    entries = [(1, 2), np.arange(3), {"a": 1}]
    partition = partition_entries(entries, lazy=lazy)
    unpacked = _lazy_entries(partition) if lazy else _eager_entries(partition)
    assert unpacked[(0,)] == (1, 2)
    np.testing.assert_array_equal(unpacked[(1,)], np.arange(3))
    assert unpacked[(2,)] == {"a": 1}


def test_partition_entries_shared_node_is_one_graph_node():
    calls = []

    def build():
        calls.append(1)
        return "shared"

    shared = dask.delayed(build)()
    partition = partition_entries([(shared, i) for i in range(5)], lazy=True)
    entries = dask.compute(*partition.to_delayed().ravel())
    assert [_unpack(e) for e in entries] == [("shared", i) for i in range(5)]
    assert len(calls) == 1
