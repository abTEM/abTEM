"""Module for determining chunk sizes of Dask arrays."""

from __future__ import annotations

import itertools
from functools import reduce
from itertools import accumulate
from operator import mul
from typing import Generator, Optional, TypeGuard, Union

import numpy as np
from dask.utils import parse_bytes

from abtem.core import config

Chunks = Union[int, str, tuple[Union[int, str, tuple[int, ...]], ...]]
ChunksTuple = tuple[Union[int, str, tuple[int, ...]], ...]
ValidatedChunks = tuple[tuple[int, ...], ...]


def is_tuple_of_ints(x: Chunks) -> TypeGuard[tuple[int, ...]]:
    return isinstance(x, tuple) and all(isinstance(c, int) for c in x)


def is_tuple_of_tuple_of_ints(x: Chunks) -> TypeGuard[tuple[tuple[int, ...], ...]]:
    return isinstance(x, tuple) and all(
        isinstance(x1, tuple) and all(isinstance(c, int) for c in x1) for x1 in x
    )


def is_tuple_of_ints_or_tuple_of_ints(
    x: Chunks,
) -> TypeGuard[tuple[tuple[int, ...], ...]]:
    return isinstance(x, tuple) and all(
        isinstance(x1, int)
        or (isinstance(x1, tuple) and all(isinstance(c, int) for c in x1))
        for x1 in x
    )


def is_tuple_of_ints_or_tuple_of_tuple_of_ints(
    x: Chunks,
) -> TypeGuard[tuple[int | tuple[int, ...], ...]]:
    return is_tuple_of_ints(x) or is_tuple_of_tuple_of_ints(x)


def is_validated_chunks(x: Chunks) -> TypeGuard[ValidatedChunks]:
    """
    Check if the input is are valid chunk sizes.

    Parameters
    ----------
    x : int or tuple of int or tuple of tuple of int or str
        The chunk sizes of the Dask array.

    Returns
    -------
    TypeGuard[ValidatedChunks]
        True if the input is a valid chunk size.
    """
    return is_tuple_of_tuple_of_ints(x)


def assert_chunks_match_shape(shape: tuple[int, ...], chunks: ValidatedChunks) -> None:
    if not all(sum(c) == s for s, c in zip(shape, chunks)):
        raise ValueError(f"chunks must match shape, got {chunks} for shape {shape}")


def chunk_ranges(chunks: ValidatedChunks) -> tuple[tuple[tuple[int, int], ...], ...]:
    """
    Get the start and end indices for each chunk.

    Parameters
    ----------
    chunks : tuple of tuple of int
        The chunk sizes of the Dask array.

    Returns
    -------
    tuple of tuple of tuple of two int
        The range of indices for each chunk.
    """
    return tuple(
        tuple((cumchunks - cc, cumchunks) for cc, cumchunks in zip(c, accumulate(c)))
        for c in chunks
    )


def iterate_chunk_ranges(chunks: ValidatedChunks):
    """
    Iterate over the chunk ranges.

    Parameters
    ----------
    chunks : tuple of tuple of int
        The chunk sizes of the Dask array.

    Yields
    ------
    block_indices : tuple of int
        The indices of the current block.
    slices : tuple of slice
        The slices indexing the current block.
    """
    chunk_shape = tuple(len(c) for c in chunks)
    for block_indices, chunk_range in zip(
        itertools.product(*(range(n) for n in chunk_shape)),
        itertools.product(*chunk_ranges(chunks)),
    ):
        slic = tuple(slice(*cr) for cr in chunk_range)

        yield block_indices, slic


def fill_in_chunk_sizes(
    shape: tuple[int, ...], chunks: tuple[int | tuple[int, ...], ...]
) -> ValidatedChunks:
    validated_chunks = []
    for s, c in zip(shape, chunks):
        if isinstance(c, tuple):
            validated_chunks.append(c)
        elif isinstance(c, int):
            if c == -1:
                validated_chunks.append((s,))
            else:
                chunk_size = (c,) * (s // c)
                if s % c:
                    chunk_size += (s % c,)
                validated_chunks.append(chunk_size)
        else:
            raise RuntimeError("Invalid chunk type")

    return tuple(validated_chunks)


def check_chunks_match_shape_length(shape: tuple[int, ...], chunks: Chunks) -> None:
    if isinstance(chunks, tuple) and not len(shape) == len(chunks):
        raise ValueError(f"length of shape: {shape} does not match chunks {chunks}")


def validate_chunks(
    shape: tuple[int, ...],
    chunks: Chunks,
    max_elements: int | str = "auto",
    dtype: Optional[np.dtype] = None,
    device: str = "cpu",
) -> ValidatedChunks:
    """
    Validate the chunks for a Dask array based on the shape and a maximum number of
    elements.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array.
    chunks : int or tuple of int or str
        The chunk sizes of the Dask array. If an integer, the array will be split into
        equal chunks. If a tuple, the array will be split into the specified chunks.
        If "auto", the chunks will be determined automatically based on the shape and
        the maximum number of elements.
    max_elements : int or str
        The maximum number of elements in a chunk. If "auto", the maximum number of
        elements will be determined based on the maximum number of bytes per chunk and
        the dtype.
    dtype : np.dtype
        The dtype of the array.
    device : str
        The device the array will be stored on.

    Returns
    -------
    tuple of tuple of int
        The chunk sizes of the Dask array.
    """
    check_chunks_match_shape_length(shape, chunks)

    if is_validated_chunks(chunks):
        validated_chunks = chunks

    elif chunks == -1:
        validated_chunks = validate_chunks(shape, shape)

    elif isinstance(chunks, int):
        max_elements = chunks
        chunks = ("auto",) * len(shape)
        validated_chunks = _auto_chunks(
            shape, chunks, max_elements, dtype=dtype, device=device
        )

    elif isinstance(chunks, str):
        raise NotImplementedError()

    elif any(isinstance(c, str) for c in chunks):
        validated_chunks = _auto_chunks(
            shape, chunks, max_elements, dtype=dtype, device=device
        )

    elif is_tuple_of_ints_or_tuple_of_ints(chunks):
        validated_chunks = fill_in_chunk_sizes(shape, chunks)

    else:
        raise ValueError(
            "chunks must be an integer, a tuple of integers a tuple of tuple of"
            f"integers or 'auto' got {chunks}"
        )

    assert_chunks_match_shape(shape, validated_chunks)

    return validated_chunks


def _auto_chunks(
    shape: tuple[int, ...],
    chunks: ChunksTuple,
    max_elements: str | int = "auto",
    dtype: Optional[np.dtype] = None,
    device: str = "cpu",
) -> ValidatedChunks:
    """
    Automatically determine the chunks for a Dask array based on the shape and a maximum
    number of elements.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array.
    chunks : tuple of int or str
        The chunk sizes of the Dask array. If an integer, the array will be split into
        equal chunks. If a tuple, the array will be split into the specified chunks.
        If "auto", the chunks will be determined automatically based on the shape and
        the maximum number of elements.
    max_elements : int or str
        The maximum number of elements in a chunk. If "auto", the maximum number of
        elements will be determined based on the maximum number of bytes per chunk and
        the dtype.
    dtype : np.dtype
        The dtype of the array.
    device : str
        The device the array will be stored on.

    Returns
    -------
    tuple of tuple of int
        The chunk sizes of the Dask array.
    """
    check_chunks_match_shape_length(shape, chunks)

    if max_elements == "auto":
        if device == "gpu":
            chunk_bytes = parse_bytes(config.get("dask.chunk-size-gpu"))
        elif device == "cpu":
            chunk_bytes = parse_bytes(config.get("dask.chunk-size"))
        else:
            raise RuntimeError(f"Unknown device: {device}")

        if dtype is None:
            raise ValueError("auto selecting chunk sizes requires dtype")

        max_elements = int(np.floor(chunk_bytes) / np.dtype(dtype).itemsize)

    elif isinstance(max_elements, str):
        max_elements = int(
            np.floor(parse_bytes(max_elements) / np.dtype(dtype).itemsize)
        )

    elif not isinstance(max_elements, int):
        raise ValueError("limit must be an integer or a string")

    normalized_chunks = tuple(s if c == -1 else c for s, c in zip(shape, chunks))

    # minimum_chunks = tuple(
    #     1 if c == "auto" else c for s, c in zip(shape, normalized_chunks)
    # )

    current_chunks = []
    max_chunks = []
    for n, c in zip(shape, normalized_chunks):
        if c == "auto":
            current_chunks.append(1)
            max_chunks.append(n)
        elif isinstance(c, int):
            current_chunks.append(c)
            max_chunks.append(c)
        elif isinstance(c, tuple):
            current_chunks.append(max(c))
            max_chunks.append(max(c))
        else:
            raise RuntimeError()

    autodims = [i for i, c in enumerate(normalized_chunks) if c == "auto"]

    j = 0
    while len(autodims):
        # autodims = [i for i in autodims if current_chunks[i] != maximum_chunks[i]]
        if len(autodims) == 0:
            break

        j = j % len(autodims)

        current_chunks[autodims[j]] = min(
            current_chunks[autodims[j]] + 1, shape[autodims[j]]
        )

        total = reduce(mul, current_chunks)

        if total > max_elements:
            current_chunks[autodims[j]] -= 1
            if current_chunks[autodims[j]] == 0:
                raise RuntimeError(
                    "Object cannot be automatically chunked; consider increasing chunk-size parameter!"
                )
            break

        if current_chunks == max_chunks:
            break

        j += 1

    chunks = ()
    for i, c in enumerate(normalized_chunks):
        if c == "auto":
            chunks += (current_chunks[i],)
        else:
            chunks += (c,)

    chunks = validate_chunks(shape, chunks, max_elements, dtype)
    return chunks


def equal_sized_chunks(
    num_items: int, num_chunks: Optional[int] = None, chunk_size: Optional[int] = None
) -> tuple[int, ...]:
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller
    integers equals n.

    Parameters
    ----------
    num_items: int
        The integer to split.
    num_chunks: int
        The number integers n will be split into.
    chunk_size: int
        The size of each chunk.

    Returns
    -------
    tuple of int
        The split integers.
    """
    if num_items == 0:
        return ()

    if num_chunks is not None and chunk_size is not None:
        raise RuntimeError("specify either num_chunks or chunks, not both")

    if num_chunks is None:
        if chunk_size is not None:
            num_chunks = (num_items + (-num_items % chunk_size)) // chunk_size
        else:
            raise RuntimeError("either num_chunks or chunks must be specified")

    if num_items < num_chunks:
        raise RuntimeError(
            f"num_chunks ({num_chunks}) may not be larger than num_items ({num_items})"
        )

    elif num_items % num_chunks == 0:
        chunks = tuple([num_items // num_chunks] * num_chunks)
    else:
        zp = num_chunks - (num_items % num_chunks)
        pp = num_items // num_chunks
        chunks = tuple(pp + 1 if i >= zp else pp for i in range(num_chunks))

    assert sum(chunks) == num_items
    return chunks


def generate_chunks(
    num_items: int,
    num_chunks: Optional[int] = None,
    chunks: Optional[int] = None,
    start: int = 0,
) -> Generator[tuple[int, int], None, None]:
    """
    Generate start and end indices for each chunks of equal sized chunks.

    Parameters
    ----------
    num_items: int
        The integer to split.
    num_chunks: int
        The number integers n will be split into.
    chunks: int
        The size of each chunk.
    start: int
        The starting index.

    Yields
    ------
    tuple of int
        The start and end indices of the current chunk.
    """
    for batch in equal_sized_chunks(num_items, num_chunks, chunks):
        if num_items == 0:
            break

        end = start + batch

        yield start, end
        start = end


def estimate_potential_chunk_size(
    gpts: tuple[int, int],
    device: str = "cpu",
    dtype: np.dtype = None,
) -> int:
    """
    Estimate the number of potential slices that fit in the memory budget.

    ``build()`` places the entire slice dimension into a single dask chunk, so
    the full potential must fit in memory at once. This function calculates how
    many slices can be held simultaneously when the potential is instead built
    in smaller chunks via ``generate_chunked_slices()``.

    On GPU the per-slice cost accounts for CuPy memory pool fragmentation —
    the pool may hold large contiguous blocks for live arrays (waves, probes)
    that prevent new allocations even when total free bytes suffice.  The
    effective per-slice cost under fragmentation is empirically ~5× the raw
    slice size for scan workloads at 4096² grids.

    On CPU there is no pool fragmentation and system RAM is typically
    abundant.  The budget is therefore the raw slice size (1×), meaning
    ``dask.chunk-size`` directly controls how many slices fit.  In
    practice the cap in :meth:`generate_chunked_slices` collapses this
    to a single chunk whenever the whole potential fits within the budget.

    On GPU the budget uses the CUDA-reported free memory without calling
    ``free_all_blocks()`` first. Dead pool blocks represent recent memory
    pressure from build/propagation temporaries; leaving them gives a
    conservative estimate that self-adapts as the pool fills up over
    successive scan batches. Falls back to ``dask.chunk-size-gpu`` when
    CuPy is unavailable.

    Parameters
    ----------
    gpts : tuple of int
        The number of grid points (y, x).
    device : str
        The device ('cpu' or 'gpu').
    dtype : np.dtype, optional
        The dtype of the potential array. If None, uses float32.

    Returns
    -------
    int
        The estimated number of slices that fit in memory.
    """
    from abtem.core.utils import get_dtype

    if dtype is None:
        dtype = np.dtype(get_dtype(complex=False))

    chunk_size_key = "potential.slice-chunk-size"
    chunk_size_setting = config.get(chunk_size_key, "auto")

    if chunk_size_setting != "auto":
        return int(chunk_size_setting)

    slice_bytes = gpts[0] * gpts[1] * dtype.itemsize

    if device == "gpu":
        try:
            import cupy as cp

            pool = cp.get_default_memory_pool()

            # Use CUDA-reported free memory without calling
            # free_all_blocks() first. Dead pool blocks represent recent
            # memory pressure and keep the estimate conservative, which
            # is the desired behaviour as the pool fills over successive
            # scan batches.
            free_mem, total_mem = cp.cuda.Device().mem_info

            # Cross-check against pool live usage. This guards the rare
            # case where dead blocks were released by someone else (e.g.
            # cuFFT plan cache eviction), making free_mem higher than the
            # live-data picture suggests.
            pool_used = pool.used_bytes()
            effective_free = min(free_mem, total_mem - pool_used)

            # Per-slice cost: output array (1×) + transmission function
            # (2×) + build temporaries (FFTs, Gaussian integrals) +
            # propagation FFT workspace. 5× is less conservative than the
            # original 8× now that the synchronous scheduler prevents
            # concurrent batch execution from multiplying peak VRAM.
            effective_per_slice = slice_bytes * 5
            budget_bytes = int(effective_free * 0.25)
        except (ImportError, Exception):
            effective_per_slice = slice_bytes * 5
            budget_bytes = parse_bytes(config.get("dask.chunk-size-gpu", "512 MB"))
    else:
        # On CPU, system RAM is abundant and there is no memory-pool
        # fragmentation.  Use the raw slice size as the unit cost (1×)
        # so that the dask.chunk-size budget directly controls how many
        # slices fit.  The cap in generate_chunked_slices will collapse
        # this to a single chunk whenever the whole potential fits.
        effective_per_slice = slice_bytes
        budget_bytes = parse_bytes(config.get("dask.chunk-size", "128 MB"))

    chunk_size = max(1, int(budget_bytes / effective_per_slice))

    return min(chunk_size, 4096)


def _nearest_power_of_two(n: int) -> int:
    """Round n to the nearest power of two.

    Prefers the *upper* power when it fits within 25 % of the raw estimate
    (i.e. when ``ceil_pot <= n * 1.25``).  This lets the auto-sizing snap
    to GPU-friendly batch sizes (8, 16, 32, 64 …) without meaningfully
    exceeding the VRAM budget, since the raw estimate already uses only
    50 % of free VRAM as headroom.

    Examples
    --------
    59 → 64  (int(59 * 1.25) = 73;  64 ≤ 73,  use upper)
    14 → 16  (int(14 * 1.25) = 17;  16 ≤ 17,  use upper)
    20 → 16  (int(20 * 1.25) = 25;  32 > 25,  use lower)
    """
    if n <= 1:
        return 1
    floor_pot = 1 << (n.bit_length() - 1)   # largest power of two ≤ n
    ceil_pot = floor_pot << 1                # smallest power of two ≥ n
    if n == floor_pot:
        return n                             # already a power of two
    if ceil_pot <= int(n * 1.25):
        return ceil_pot
    return floor_pot


def estimate_scan_batch_size(
    gpts: tuple[int, int],
    dtype,
    device: str,
) -> int:
    """
    Estimate the maximum number of probe wavefunctions per scan batch.

    For GPU, queries free CUDA memory at graph-construction time and
    allocates up to half of it for probe wavefunctions. This is
    intentionally generous because ``estimate_potential_chunk_size`` is
    called at *computation* time (inside ``generate_chunked_slices``) when
    the probe batch is already resident in VRAM; it therefore sees the
    reduced free memory and sizes the potential chunk to fit in what
    remains. The two estimates are thus naturally coordinated without
    requiring explicit cross-referencing.

    The raw estimate is rounded to the nearest power of two (preferring
    the upper power when within 25 % headroom) so that batch sizes snap
    to GPU-friendly values such as 8, 16, 32, 64.

    For CPU, falls back to the ``dask.chunk-size`` configuration key.

    Parameters
    ----------
    gpts : tuple of int
        Spatial grid size ``(ny, nx)`` of each probe wavefunction.
    dtype : dtype-like
        Wavefunction dtype (typically ``complex64``).
    device : str
        ``"gpu"`` or ``"cpu"``.

    Returns
    -------
    int
        Maximum number of probe wavefunctions per batch (≥ 1).
    """
    per_probe_bytes = int(np.prod(gpts)) * np.dtype(dtype).itemsize

    if device == "gpu":
        try:
            import cupy as cp

            free_mem, _ = cp.cuda.Device().mem_info

            # Allocate up to 50 % of currently-free VRAM for probe
            # wavefunctions.  A 6× overhead factor accounts for transient
            # copies during transmission-function multiply, FFT workspaces,
            # and cuFFT plan buffers.  Empirically: 4096² uses ~756 MB/probe
            # (5.6× raw wavefunction), 2048² uses ~147 MB/probe (4.4×).
            # The potential chunk (sized by estimate_potential_chunk_size at
            # computation time) plus remaining workspace fills the other 50%.
            probe_budget = int(free_mem * 0.5)
            per_probe_effective = max(1, int(per_probe_bytes * 6))
            n_probes = max(1, probe_budget // per_probe_effective)
            return _nearest_power_of_two(n_probes)
        except (ImportError, Exception):
            chunk_bytes = parse_bytes(config.get("dask.chunk-size-gpu", "512 MB"))
    else:
        chunk_bytes = parse_bytes(config.get("dask.chunk-size", "128 MB"))

    return max(1, chunk_bytes // max(1, per_probe_bytes))
