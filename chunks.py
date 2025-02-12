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
            raise RuntimeError("Object cannot be automatically chunked; consider increasing chunk-size parameter!")
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
        return 0, 0

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
