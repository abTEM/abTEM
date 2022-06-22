"""Module for various convenient utilities."""
import copy
import inspect
from typing import Tuple

import numpy as np

from abtem.core.backend import get_array_module
import dask.array as da
from abtem.core.backend import cp


def is_array_like(x):
    if isinstance(x, np.ndarray) or (cp is not None and isinstance(x, cp.ndarray)):
        return True
    else:
        return False


class CopyMixin:

    def copy_kwargs(self, exclude: Tuple['str', ...] = ()) -> dict:
        parameters = inspect.signature(self.__class__).parameters
        keys = {key for key, value in parameters.items() if value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD)}
        keys = [key for key in keys if key not in exclude]
        kwargs = {key: copy.deepcopy(getattr(self, key)) for key in keys}
        return kwargs

    def copy(self):
        return copy.deepcopy(self)


class EqualityMixin:

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        for key, value in self.__dict__.items():

            try:
                equal = value == other.__dict__[key]
            except KeyError:
                return False

            if equal is False:
                return False

            if is_array_like(value):
                xp = get_array_module(value)
                if not xp.allclose(value, other.__dict__[key]):
                    return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)


def safe_floor_int(n: float, tol: int = 7):
    return int(np.floor(np.round(n, decimals=tol)))


def insert_empty_axis(match_axis1, match_axis2):
    for i, (a1, a2) in enumerate(zip(reversed(match_axis1), reversed(match_axis2))):
        if a1 is True and a2 is False:
            match_axis2.insert(len(match_axis2) - i, None)
            break

        if a1 is False and a2 is True:
            match_axis1.insert(len(match_axis1) - i, None)
            break

        if a1 is True and a2 is True:
            match_axis1.insert(len(match_axis1) - i, None)
            break


def normalize_axes(dims, shape):
    num_dims = len(shape)
    return tuple(dim if dim >= 0 else num_dims + dim for dim in dims)


def expand_dims_to_match(arr1, arr2, match_dims):
    assert len(match_dims) == 2
    assert len(match_dims[0]) == len(match_dims[1])

    match_dims[0] = normalize_axes(match_dims[0], arr1.shape)
    match_dims[1] = normalize_axes(match_dims[1], arr2.shape)

    match_axis1 = [not i in match_dims[0] for i in range(len(arr1.shape))]
    match_axis2 = [not i in match_dims[1] for i in range(len(arr2.shape))]

    last_length = len(match_axis1) + len(match_axis2)
    for i in range(last_length):
        insert_empty_axis(match_axis1, match_axis2)

        if len(match_axis1) + len(match_axis2) == last_length:
            break

        last_length = len(match_axis1) + len(match_axis2)

    if len(match_axis1) < len(match_axis2):
        match_axis1 = [None] * (len(match_axis2) - len(match_axis1)) + match_axis1
    elif len(match_axis1) > len(match_axis2):
        match_axis2 = [None] * (len(match_axis1) - len(match_axis2)) + match_axis2

    axis1 = tuple(i for i, a in enumerate(match_axis1) if a is None)
    axis2 = tuple(i for i, a in enumerate(match_axis2) if a is None)



    arr1 = np.expand_dims(arr1, axis=axis1)
    arr2 = np.expand_dims(arr2, axis=axis2)

    return arr1, arr2


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
