"""Module for determining chunk sizes of Dask arrays."""

from __future__ import annotations

import itertools
from functools import reduce
from itertools import accumulate
from operator import mul
from typing import Union

import numpy as np
from dask.utils import parse_bytes

from abtem.core import config

Chunks = Union[int, str, tuple[Union[int, str, tuple[int, ...]], ...]]
ValidatedChunks = tuple[tuple[int, ...], ...]


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


def validate_chunks(
    shape: tuple[int, ...],
    chunks: Chunks,
    max_elements: int | str = "auto",
    dtype: np.dtype.base = None,
    device: str = "cpu",
) -> ValidatedChunks:
    """
    Validate the chunks for a Dask array based on the shape and a maximum number of elements.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array.
    chunks : int or tuple of int or str
        The chunk sizes of the Dask array. If an integer, the array will be split into equal chunks. If a tuple, the
        array will be split into the specified chunks. If "auto", the chunks will be determined automatically based
        on the shape and the maximum number of elements.
    max_elements : int or str
        The maximum number of elements in a chunk. If "auto", the maximum number of elements will be determined based
        on the maximum number of bytes per chunk and the dtype.
    dtype : np.dtype
        The dtype of the array.
    device : str
        The device the array will be stored on.

    Returns
    -------
    tuple of tuple of int
        The chunk sizes of the Dask array.
    """
    if isinstance(chunks, tuple):
        if not len(shape) == len(chunks):

            raise ValueError(f"length of shape: {shape} does not match chunks {chunks}")

    if chunks == -1:
        return validate_chunks(shape, shape)

    if isinstance(chunks, int):
        max_elements = chunks
        chunks = ("auto",) * len(shape)
        return _auto_chunks(shape, chunks, max_elements, dtype=dtype, device=device)

    if all(isinstance(c, tuple) for c in chunks):
        assert len(shape) == len(chunks)
        return chunks

    if any(isinstance(c, str) for c in chunks):
        return _auto_chunks(shape, chunks, max_elements, dtype=dtype, device=device)

    if len(shape) == 1 and len(chunks) != len(shape):
        assert sum(chunks) == shape[0]
        return (chunks,)

    validated_chunks = ()
    for s, c in zip(shape, chunks):
        if isinstance(c, tuple):
            assert sum(c) == s
            validated_chunks += (c,)

        elif isinstance(c, int):
            if c == -1:
                validated_chunks += ((s,),)
            elif s % c:
                validated_chunks += ((c,) * (s // c) + (s - c * (s // c),),)
            else:
                validated_chunks += ((c,) * (s // c),)
        else:
            raise RuntimeError()
    
    assert tuple(sum(c) for c in validated_chunks) == shape

    return validated_chunks


def _auto_chunks(
    shape: tuple[int, ...],
    chunks: Chunks,
    max_elements: str | int = "auto",
    dtype: np.dtype.base = None,
    device: str = "cpu",
) -> ValidatedChunks:
    """
    Automatically determine the chunks for a Dask array based on the shape and a maximum number of elements.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array.
    chunks : int or tuple of int or str
        The chunk sizes of the Dask array. If an integer, the array will be split into equal chunks. If a tuple, the
        array will be split into the specified chunks. If "auto", the chunks will be determined automatically based
        on the shape and the maximum number of elements.
    max_elements : int or str
        The maximum number of elements in a chunk. If "auto", the maximum number of elements will be determined based
        on the maximum number of bytes per chunk and the dtype.
    dtype : np.dtype
        The dtype of the array.
    device : str
        The device the array will be stored on.

    Returns
    -------
    tuple of tuple of int
        The chunk sizes of the Dask array.
    """
    if not len(shape) == len(chunks):
        raise ValueError("shape and chunks must have the same length")

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


def equal_sized_chunks(num_items: int, num_chunks: int = None, chunks: int = None):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    num_items: int
        The integer to split.
    num_chunks: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if num_items == 0:
        return 0, 0

    if (num_chunks is not None) & (chunks is not None):
        raise RuntimeError()

    if (num_chunks is None) & (chunks is not None):
        num_chunks = (num_items + (-num_items % chunks)) // chunks

    if num_items < num_chunks:
        raise RuntimeError(
            f"num_chunks ({num_chunks}) may not be larger than num_items ({num_items})"
        )

    elif num_items % num_chunks == 0:
        return tuple([num_items // num_chunks] * num_chunks)
    else:
        v = []
        zp = num_chunks - (num_items % num_chunks)
        pp = num_items // num_chunks
        for i in range(num_chunks):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return tuple(v)


def generate_chunks(
    num_items: int, num_chunks: int = None, chunks: int = None, start: int = 0
):
    for batch in equal_sized_chunks(num_items, num_chunks, chunks):
        if num_items == 0:
            break

        end = start + batch

        yield start, end
        start = end
