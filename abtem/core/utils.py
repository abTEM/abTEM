"""Module for various convenient utilities."""
import copy
import inspect
import warnings
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
    _exclude_from_copy: tuple = ()

    def _arg_keys(self, cls):
        parameters = inspect.signature(cls).parameters
        return tuple(
            key
            for key, value in parameters.items()
            if value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD)
        )

    def _copy_kwargs(self, exclude: Tuple["str", ...] = (), cls=None) -> dict:
        if cls is None:
            cls = self.__class__

        exclude = self._exclude_from_copy + exclude
        keys = [key for key in self._arg_keys(cls) if key not in exclude]
        kwargs = {key: copy.deepcopy(getattr(self, key)) for key in keys}
        return kwargs

    def copy(self):
        """ Make a copy. """
        return copy.deepcopy(self)


def safe_equality(a, b, exclude: Tuple[str, ...] = ()) -> bool:
    if not isinstance(b, a.__class__):
        return False

    for key, value in a.__dict__.items():

        if key in exclude:
            continue

        try:
            equal = value == b.__dict__[key]
        except (KeyError, TypeError):
            return False

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"Creating an ndarray from nested sequences"
            )
            # TODO: properly handle this warning

            try:
                equal = np.allclose(value, b.__dict__[key])
            except (ValueError, TypeError):
                pass

        if equal is False:
            return False

    return True


class EqualityMixin:
    def __eq__(self, other):
        return safe_equality(self, other)

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


def expand_dims_to_match(arr1, arr2, match_dims=None):
    if match_dims is None:
        match_dims = [(), ()]

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


def label_to_index(labels, max_label=None):
    if max_label is None:
        max_label = np.max(labels)

    xp = get_array_module(labels)
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = xp.arange(0, len(labels) + 1)[labels_order]
    index = xp.arange(0, max_label + 1)
    lo = xp.searchsorted(sorted_labels, index, side="left")
    hi = xp.searchsorted(sorted_labels, index, side="right")
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]
