"""Module for various convenient utilities."""

import numpy as np

from abtem.basic.backend import get_array_module


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)


def subdivide_into_chunks(num_items: int, num_chunks: int = None, chunks: int = None):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if (num_chunks is not None) & (chunks is not None):
        raise RuntimeError()

    if (num_chunks is None) & (chunks is not None):
        num_chunks = (num_items + (-num_items % chunks)) // chunks

    if num_items < num_chunks:
        raise RuntimeError('num_chunks may not be larger than num_items')

    elif num_items % num_chunks == 0:
        return [num_items // num_chunks] * num_chunks
    else:
        v = []
        zp = num_chunks - (num_items % num_chunks)
        pp = num_items // num_chunks
        for i in range(num_chunks):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v


def generate_chunks(num_items: int, num_chunks: int = None, chunks: int = None, start=0):
    for batch in subdivide_into_chunks(num_items, num_chunks, chunks):
        end = start + batch
        yield start, end
        start = end


def generate_array_chunks(array, chunks):
    if len(chunks) != len(array.shape):
        raise ValueError()

    def _recursive_generate_array_chunks(array, chunks):

        i = len(chunks) - 1

        for start, end in generate_chunks(array.shape[i], chunks=chunks[-1]):
            slc = [slice(None)] * len(array.shape)
            slc[i] = slice(start, end)

            chunk = array[tuple(slc)]

            if len(chunks) > 1:
                yield from _recursive_generate_array_chunks(chunk, chunks[:-1])
            else:
                yield chunk

    return _recursive_generate_array_chunks(array, chunks)


def reassemble_chunks_along(blocks, shape, axis):
    row_blocks = []
    new_blocks = []
    row_tally = 0
    for block in blocks:
        row_blocks.append(block)
        row_tally += block.shape[axis]

        if row_tally == shape:
            new_blocks.append(np.concatenate(row_blocks, axis=axis))
            row_blocks = []
            row_tally = 0

        if row_tally > shape:
            raise RuntimeError()

    return new_blocks


def tapered_cutoff(x, cutoff, rolloff=.1):
    xp = get_array_module(x)

    rolloff = rolloff * cutoff

    if rolloff > 0.:
        array = .5 * (1 + xp.cos(np.pi * (x - cutoff + rolloff) / rolloff))
        array[x > cutoff] = 0.
        array = xp.where(x > cutoff - rolloff, array, xp.ones_like(x, dtype=xp.float32))
    else:
        array = xp.array(x < cutoff).astype(xp.float32)

    return array
