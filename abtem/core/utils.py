"""Module for various convenient utilities."""

import numpy as np

from abtem.core.backend import get_array_module
import dask.array as da


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
    if num_items == 0:
        return 0, 0

    if (num_chunks is not None) & (chunks is not None):
        raise RuntimeError()

    if (num_chunks is None) & (chunks is not None):
        num_chunks = (num_items + (-num_items % chunks)) // chunks

    if num_items < num_chunks:
        raise RuntimeError('num_chunks may not be larger than num_items')

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


def generate_chunks(num_items: int, num_chunks: int = None, chunks: int = None, start: int = 0):
    for batch in subdivide_into_chunks(num_items, num_chunks, chunks):
        end = start + batch
        yield start, end
        start = end


def label_to_index(labels, max_label=None):
    if max_label is None:
        max_label = np.max(labels)

    xp = get_array_module(labels)
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = xp.arange(0, len(labels) + 1)[labels_order]
    index = xp.arange(0, max_label + 1)
    lo = xp.searchsorted(sorted_labels, index, side='left')
    hi = xp.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


def reassemble_chunks_along(blocks, shape, axis, concatenate=True, delayed=False):
    xp = get_array_module(blocks[0])
    if delayed:
        xp = da

    row_blocks = []
    new_blocks = []
    row_tally = 0
    for block in blocks:
        row_blocks.append(block)
        row_tally += block.shape[axis]

        if row_tally == shape:
            if concatenate:
                new_blocks.append(xp.concatenate(row_blocks, axis=axis))
            else:
                new_blocks.append(row_blocks)

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
