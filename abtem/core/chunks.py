from __future__ import annotations
import itertools
from functools import reduce
from itertools import accumulate
from operator import mul
from typing import Union

import numpy as np
from dask.utils import parse_bytes

from abtem.core import config


def chunk_ranges(chunks):
    return tuple(
        tuple((cumchunks - cc, cumchunks) for cc, cumchunks in zip(c, accumulate(c)))
        for c in chunks
    )


def chunk_shape(chunks):
    return tuple(len(c) for c in chunks)


def iterate_chunk_ranges(chunks):
    for block_indices, chunk_range in zip(
        itertools.product(*(range(n) for n in chunk_shape(chunks))),
        itertools.product(*chunk_ranges(chunks)),
    ):
        slic = tuple(slice(*cr) for cr in chunk_range)

        yield block_indices, slic


def config_chunk_size(device):
    if device == "gpu":
        return parse_bytes(config.get("dask.chunk-size-gpu"))

    elif device != "cpu":
        raise RuntimeError("Unknown device")

    return parse_bytes(config.get("dask.chunk-size"))


Chunks = Union[int, str, tuple[Union[int, str, tuple[int, ...]], ...]]
ValidatedChunks = tuple[tuple[int, ...], ...]


def validate_chunks(
    shape: tuple[int, ...],
    chunks: Chunks,
    limit: int | str = "auto",
    dtype: np.dtype.base = None,
    device: str = "cpu",
) -> ValidatedChunks:
    if chunks == -1:
        return validate_chunks(shape, shape)

    if isinstance(chunks, int):
        limit = chunks
        chunks = ("auto",) * len(shape)
        return auto_chunks(shape, chunks, limit, dtype=dtype, device=device)

    if all(isinstance(c, tuple) for c in chunks):
        assert len(shape) == len(chunks)
        return chunks

    if any(isinstance(c, str) for c in chunks):
        return auto_chunks(shape, chunks, limit, dtype=dtype, device=device)

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


def auto_chunks(
    shape: tuple[int, ...],
    chunks: Chunks,
    limit: str | int = "auto",
    dtype: np.dtype.base = None,
    device: str = "cpu",
) -> ValidatedChunks:
    if limit == "auto":
        if dtype is None:
            raise ValueError("auto selecting chunk limits requires dtype")

        limit = int(np.floor(config_chunk_size(device)) / np.dtype(dtype).itemsize)

    elif isinstance(limit, str):
        limit = int(np.floor(parse_bytes(limit) / np.dtype(dtype).itemsize))

    elif not isinstance(limit, int):
        raise ValueError

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

        if total > limit:
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

    # current_chunks = tuple(current_chunks)
    chunks = validate_chunks(shape, chunks, limit, dtype)
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
