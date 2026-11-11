from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from itertools import accumulate
from typing import Any, Callable, Generator, Optional, Sequence, Union

import dask
import dask.array as da
import numpy as np

from abtem.core.axes import AxesMetadataList, AxisMetadata
from abtem.core.chunks import (
    Chunks,
    ValidatedChunks,
    chunk_ranges,
    iterate_chunk_ranges,
    validate_chunks,
)
from abtem.core.utils import interleave, itemset


def _wrap_with_array(x: Any, ndims: int | None = None) -> np.ndarray:
    if ndims is None:
        ndims = len(x.ensemble_shape)

    wrapped = np.zeros((1,) * ndims, dtype=object)
    itemset(wrapped, 0, x)
    return wrapped


def unpack_blockwise_args(args) -> tuple:
    unpacked = tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)
    return unpacked


def _holds_dask_collections(x: Any) -> bool:
    if dask.is_dask_collection(x):
        return True
    if isinstance(x, (list, tuple)):
        return any(_holds_dask_collections(item) for item in x)
    if isinstance(x, dict):
        return any(_holds_dask_collections(item) for item in x.values())
    return False


def partition_entries(
    entries: np.ndarray | Sequence[Any], lazy: bool
) -> np.ndarray | da.core.Array:
    """Package one entry per block of the ensemble axes as a partitioned argument.

    `entries` is an object array with one entry per block, e.g. the positions or
    the parameters of the members of the block, or a sequence of them for one
    ensemble axis. Every entry reaches the task (or
    the eager block) as an object array of shape ``(1,) * entries.ndim`` holding
    it, which :func:`unpack_blockwise_args` unpacks.

    For a lazy partition, an entry holding dask collections -- a `dask.delayed`
    object, e.g. a node shared by every block, or a dask array -- is evaluated in
    the graph, as a dependency of its task. Any other entry is stored in the graph
    as it is.

    Parameters
    ----------
    entries : numpy.ndarray of objects or sequence
        One entry per block, with one axis per partitioned ensemble axis.
    lazy : bool
        If True, return a dask array with one chunk per entry.
    """
    # Assigned by index tuple throughout, which stores any object as it is (an
    # entry may itself be a tuple or an array); `[...] =` would broadcast it.
    if not isinstance(entries, np.ndarray):
        sequence = entries
        entries = np.empty(len(sequence), dtype=object)
        for i, entry in enumerate(sequence):
            entries[(i,)] = entry
    ndim = entries.ndim

    if not lazy:
        wrapped = np.empty(entries.shape, dtype=object)
        for index in np.ndindex(entries.shape):
            wrapped[index] = _wrap_with_array(entries[index], ndim)
        return wrapped

    if not any(_holds_dask_collections(entry) for entry in entries.flat):
        return da.from_array(entries, chunks=1)

    blocks = np.empty(entries.shape, dtype=object)
    for index in np.ndindex(entries.shape):
        blocks[index] = da.from_delayed(
            dask.delayed(_wrap_with_array)(entries[index], ndim),
            shape=(1,) * ndim,
            dtype=object,
        )
    if ndim == 0:
        return blocks[()]
    return da.block(blocks.tolist())


def partition_blocks(
    chunks: ValidatedChunks, entry: Callable[[tuple[slice, ...]], Any], lazy: bool
) -> np.ndarray | da.core.Array:
    """Partition ensemble axes into blocks, with one entry per block.

    `entry` is called with the slices that index the members of a block along
    each axis, and returns what the block's task needs to rebuild its part of the
    ensemble, e.g. ``lambda slices: {"positions": positions[slices[0]]}``. See
    :func:`partition_entries` for how the entries reach the tasks.

    Parameters
    ----------
    chunks : tuple of tuple of int
        Validated chunks of the partitioned ensemble axes.
    entry : callable
        Returns the entry of the block indexed by the given slices.
    lazy : bool
        If True, return a dask array with one chunk per block.
    """
    entries = np.empty(tuple(len(c) for c in chunks), dtype=object)
    for index, slices in iterate_chunk_ranges(chunks):
        entries[index] = entry(slices)
    return partition_entries(entries, lazy)


def shared_constant_arg(x: Any, lazy: bool = True) -> np.ndarray | da.core.Array:
    """Package a large constant as a single node of the task graph.

    An ensemble member's keyword arguments are baked into the function of
    every task with ``functools.partial``, and dask cannot look inside a
    partial: a multi-megabyte object baked there is copied into every task,
    so the serialized graph, the scheduler traffic and the worker memory all
    grow with the task count. Returned as a partitioned argument instead,
    the object becomes one scheduler-managed graph node -- one copy in the
    graph, one materialized copy per worker, released with the graph.

    The materialized object is **shared** by every task on a worker, and by
    every thread of the local scheduler, so task code must treat it as
    read-only and privatize any state it intends to mutate.
    """
    if not lazy:
        return _wrap_with_array(x, ndims=0)

    return da.from_delayed(
        dask.delayed(_wrap_with_array)(x, ndims=0), shape=(), dtype=object
    )


class Ensemble:
    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the ensemble axes."""
        return ()

    @property
    def base_shape(self) -> tuple[int, ...]:
        """Shape of the base axes."""
        return ()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the ensemble."""
        return self.ensemble_shape + self.base_shape

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the base axes."""
        return []

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the ensemble axes."""
        return []

    @property
    def axes_metadata(self) -> AxesMetadataList:
        """List of AxisMetadata."""
        return AxesMetadataList(
            self.ensemble_axes_metadata + self.base_axes_metadata, self.shape
        )

    @property
    @abstractmethod
    def _default_ensemble_chunks(self) -> Chunks:
        pass

    def _validate_ensemble_chunks(
        self, chunks: Optional[Chunks] = None, limit: Union[str, int] = "auto"
    ) -> ValidatedChunks:
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, max_elements=limit)
        return chunks

    @abstractmethod
    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple:
        pass

    @abstractmethod
    def _from_partitioned_args(self) -> Callable[..., np.ndarray]:
        pass

    def ensemble_blocks(self, chunks: Optional[Chunks] = None) -> da.core.Array:
        """
        Split the ensemble into an array of smaller ensembles.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """

        chunks = self._validate_ensemble_chunks(chunks)

        args = self._partition_args(chunks, lazy=True)
        arg_dims = tuple(len(arg.shape) for arg in args)
        arg_starts = accumulate((0,) + arg_dims[:-1])
        arg_ends = accumulate(arg_dims)
        arg_ind = tuple(
            tuple(range(start, end)) for start, end in zip(arg_starts, arg_ends)
        )

        out_ind = tuple(range(sum(arg_dims)))
        adjust_chunks = {i: axes_chunks for i, axes_chunks in enumerate(chunks)}

        func = self._from_partitioned_args()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Increasing number of chunks")
            blocks = da.blockwise(
                func,
                out_ind,
                *interleave(args, arg_ind),
                adjust_chunks=adjust_chunks,
                concatenate=True,
                meta=np.array((), dtype=object),
            )
            return blocks

    def generate_blocks(
        self, chunks: Chunks = 1
    ) -> Generator[tuple[tuple[int, ...], tuple[slice, ...], np.ndarray], None, None]:
        """
        Generate chunks of the ensemble.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """
        chunks = self._validate_ensemble_chunks(chunks)
        blocks = self._partition_args(chunks=chunks, lazy=False)

        shape = sum((block.shape for block in blocks), ())

        start_stops = chunk_ranges(chunks)

        # print(start_stops)
        # print(shape)
        assert tuple(len(cr) for cr in start_stops) == shape

        for indices, start_stop in zip(
            np.ndindex(shape), itertools.product(*start_stops)
        ):
            block_indices: tuple[tuple[int, ...], ...] = ()
            j = 0
            for block in blocks:
                n = len(block.shape)
                block_indices += (tuple(indices[index] for index in range(j, j + n)),)
                j += n

            args = tuple(block[i] for i, block in zip(block_indices, blocks))
            slics = tuple(slice(start, stop) for start, stop in start_stop)

            yield indices, slics, self._from_partitioned_args()(*args)

        # print(blocks)

        # for block in blocks:
        # if len(block.shape) > 1:
        #    print(block)
        #         raise NotImplementedError
        # axis_indices = tuple(
        #     tuple(range(block.shape[0])) if len(block.shape) else () for block in blocks
        # )

        # if not any(len(indices) for indices in axis_indices):
        #     yield (), (), self._from_partitioned_args()(*blocks)

        # print(len(tuple(itertools.product(*chunk_ranges(chunks)))))
        # for block_indices, start_stop in zip(
        #     itertools.product(*axis_indices),
        #     itertools.product(*chunk_ranges(chunks)),
        # ):

        #     block = tuple(block[i] for i, block in zip(block_indices, blocks))
        #     slics = tuple(slice(start, stop) for start, stop in start_stop)
        #     print(slics)

        #     yield block_indices, slics, self._from_partitioned_args()(*block)


class EmptyEnsemble(Ensemble):
    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple:
        return ()

    def _from_partitioned_args(self) -> type:
        return self.__class__

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()


def concatenate_array_blocks(blocks: np.ndarray) -> np.ndarray:
    for i in range(len(blocks.shape)):
        new_blocks = np.empty(blocks.shape[:-1], dtype=object)

        for indices in np.ndindex(blocks.shape):
            concat_index = len(indices) - 1
            indices = indices[:-1]
            new_blocks[indices] = np.concatenate(blocks[indices], axis=concat_index)

        blocks = new_blocks

    return blocks.item()
