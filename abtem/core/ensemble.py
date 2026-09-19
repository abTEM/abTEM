from __future__ import annotations

import itertools
import warnings
import weakref
from abc import abstractmethod
from itertools import accumulate
from typing import Any, Callable, Generator, Optional, Union

import dask
import dask.array as da
import numpy as np

from abtem.core.axes import AxesMetadataList, AxisMetadata
from abtem.core.chunks import Chunks, ValidatedChunks, chunk_ranges, validate_chunks
from abtem.core.utils import interleave, itemset


class _ScatterCandidate:
    """Marks a value embedded by ``shared_constant_arg`` as eligible to be
    scattered into a ``distributed.Client``, once one is known, instead of
    staying a literal value in the task graph.

    ``shared_constant_arg``'s caller builds its lazy graph well before any
    ``.compute()`` call resolves which client, if any, will run it -- e.g.
    ``transition_potential_scan(lazy=True)`` builds the whole graph
    immediately, while abTEM's own multi-GPU cluster (``ensure_cuda_cluster``)
    only starts inside ``_resolve_gpu_scheduler``, at ``.compute()`` time. This
    marker keeps the payload identifiable in the already-built graph so
    ``_prescatter_shared_constants`` (``abtem/array.py``) can find and replace
    it there, right before submission. ``_wrap_with_array`` unwraps it
    transparently if it is still wrapped by the time the task actually runs
    (no suitable client was ever found).
    """

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


def _wrap_with_array(x: Any, ndims: int | None = None) -> np.ndarray:
    if isinstance(x, _ScatterCandidate):
        x = x.value

    if ndims is None:
        ndims = len(x.ensemble_shape)

    wrapped = np.zeros((1,) * ndims, dtype=object)
    itemset(wrapped, 0, x)
    return wrapped


def unpack_blockwise_args(args) -> tuple:
    unpacked = tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)
    return unpacked


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

    The lazy path wraps ``x`` in ``_ScatterCandidate`` rather than embedding it
    directly -- see that class and ``_prescatter_shared_constants`` for why
    that's necessary rather than deciding here whether to scatter.
    """
    if not lazy:
        return _wrap_with_array(x, ndims=0)

    return da.from_delayed(
        dask.delayed(_wrap_with_array)(_ScatterCandidate(x), ndims=0),
        shape=(),
        dtype=object,
    )


def _prescatter_shared_constants(arrays: list, client: Any) -> list:
    """Replace any ``_ScatterCandidate`` payload in each of ``arrays``' graphs
    with a ``distributed.Future``, scattered into ``client`` -- in a *copy* of
    the graph used only for the immediate ``dask.compute()`` call.

    Never mutates the original array: a wrapper's own stored ``.array`` keeps
    its self-contained, literal-embedding graph for any future recompute under
    a different (or no) client -- see
    ``test_graph_computes_on_a_distributed_cluster_and_survives_client_loss``
    (``test/test_transition_potential_transport.py``), which this is designed
    not to break.

    ``broadcast=True``: ``ensure_cuda_cluster`` (``abtem/core/backend.py``)
    pins one worker per visible GPU, so the cluster is small and fixed; any
    worker can end up running a task that needs this payload, and moving it to
    a GPU-pinned worker that doesn't already have it (the ``broadcast=False``
    default) costs a transfer later anyway.

    Scattering is memoized on ``client`` via a weak reference to each scattered
    object, so the same live object -- e.g. one potential or transition
    potential reused across many calls -- is scattered only once; the memo
    entry is dropped via the weakref's own finalizer exactly when the object is
    collected, never by comparing possibly-reused ``id()`` values. An object
    that doesn't support weak references (e.g. a bare ``tuple``/``list``) is
    still scattered but not memoized.

    Uses dask's own low-level task representation (``dask._task_spec.Task``,
    not a public API) to find and rewrite the specific argument holding a
    ``_ScatterCandidate``. Falls back to leaving an array's graph untouched --
    today's literal-embed behaviour -- for anything that doesn't match the
    expected shape, rather than raising: this is a performance optimization,
    not a correctness-critical path, so silently not applying it is always a
    safe fallback (e.g. across a future dask release that changes this
    representation).
    """
    if client is None:
        return arrays

    try:
        memo = client._abtem_scatter_memo
    except AttributeError:
        memo = client._abtem_scatter_memo = {}

    def scatter_once(x):
        key = id(x)
        cached = memo.get(key)
        if cached is not None and cached[0]() is x:
            return cached[1]

        # [x][0], not scatter(x, ...): distributed.Client.scatter fans a bare
        # list/tuple out into one Future per element instead of treating it
        # as a single object -- wrapping in a length-1 list forces the whole
        # of x, whatever its own type, to be scattered as one opaque payload.
        future = client.scatter([x], broadcast=True)[0]
        try:
            ref = weakref.ref(x, lambda _, memo=memo, key=key: memo.pop(key, None))
        except TypeError:
            return future  # scattered, but can't be memoized safely
        memo[key] = (ref, future)
        return future

    result = []
    for arr in arrays:
        if not isinstance(arr, da.core.Array):
            result.append(arr)
            continue
        try:
            result.append(_substitute_scatter_candidates(arr, scatter_once))
        except Exception:
            result.append(arr)
    return result


def _substitute_scatter_candidates(arr: da.core.Array, scatter_once: Callable) -> da.core.Array:
    dsk = dict(arr.__dask_graph__())
    changed = {}
    for key, task in dsk.items():
        args = getattr(task, "args", None)
        func = getattr(task, "func", None)
        if args is None or func is None:
            continue
        new_args = None
        for i, a in enumerate(args):
            if isinstance(a, _ScatterCandidate):
                if new_args is None:
                    new_args = list(args)
                new_args[i] = scatter_once(a.value)
        if new_args is not None:
            changed[key] = type(task)(key, func, *new_args, **task.kwargs)

    if not changed:
        return arr

    dsk.update(changed)
    return da.Array(dsk, name=arr.name, chunks=arr.chunks, dtype=arr.dtype, meta=arr._meta)


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
