"""Module for various convenient utilities."""

from __future__ import annotations

import copy
import inspect
import itertools
import os
import warnings
from typing import Any, Optional, Self, Sequence, TypeGuard, TypeVar, overload

import dask.array as da
import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.config import config

T = TypeVar("T", float, int, bool)


np.ndarray(())

def number_to_tuple(
    value: T | tuple[T, ...], dimension: Optional[int] = None
) -> tuple[T, ...]:
    if isinstance(value, (float, int, bool)):
        if dimension is None:
            return (value,)
        else:
            return (value,) * dimension
    else:
        if dimension is not None:
            assert len(value) == dimension
        return value


def itemset(arr: np.ndarray, args: int | slice | Sequence[int], item: Any) -> None:
    if arr.shape == ():
        arr[...] = item
        return

    elif isinstance(args, tuple):
        assert len(args) == len(arr.shape)
        arr[args] = item
        return

    elif isinstance(args, int) and len(arr.shape) == 1:
        arr[args] = item
        return

    elif isinstance(args, int):
        assert all(n == 1 for n in arr.shape[1:])
        args = (args,) + (0,) * (len(arr.shape) - 1)
        arr[args] = item
        return

    else:
        raise RuntimeError()


def is_broadcastable(*shapes: tuple[int, ...]) -> bool | tuple[int, ...]:
    if not shapes:
        return True

    # Start with the first shape
    result_shape = shapes[0]

    for shape in shapes[1:]:
        # Check broadcastability between result_shape and the current shape
        for a, b in zip(result_shape[::-1], shape[::-1]):
            if a != 1 and b != 1 and a != b:
                return False
        # Update result_shape to the broadcasted shape
        result_shape = tuple(
            max(a, b) for a, b in zip(result_shape[::-1], shape[::-1])
        )[::-1]

    return True


class CopyMixin:
    _exclude_from_copy: tuple = ()

    @staticmethod
    def _arg_keys(cls):
        parameters = inspect.signature(cls).parameters
        return tuple(
            key
            for key, value in parameters.items()
            if value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD)
        )

    def _copy_kwargs(self, exclude: tuple[str, ...] = (), cls=None) -> dict:
        if cls is None:
            cls = self.__class__

        exclude = self._exclude_from_copy + exclude
        keys = [key for key in self._arg_keys(cls) if key not in exclude]
        kwargs = {key: copy.deepcopy(getattr(self, key)) for key in keys}
        return kwargs

    def copy(self) -> Self:
        """Make a copy."""
        return copy.deepcopy(self)


def safe_equality(a, b, exclude: tuple[str, ...] = ()) -> bool:
    if not isinstance(b, a.__class__):
        return False

    for key, value in a.__dict__.items():
        if key in exclude:
            continue

        try:
            equal = value == b.__dict__[key]
        except (KeyError, TypeError, ValueError):
            return False

        from abtem.core.ensemble import EmptyEnsemble

        if isinstance(value, EmptyEnsemble) and isinstance(
            b.__dict__[key], EmptyEnsemble
        ):
            return True

        with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

            try:
                equal = np.allclose(value, b.__dict__[key])

            except (ValueError, TypeError):
                if isinstance(value, EqualityMixin):
                    equal = safe_equality(value, b.__dict__[key])

        if equal is False:
            return False

    return True


def _get_dims_to_broadcast(
    arr1: np.ndarray | da.core.Array,
    arr2: np.ndarray | da.core.Array,
    match_dims: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if match_dims is None:
        match_dims = ((), ())

    assert len(match_dims) == 2
    assert len(match_dims[0]) == len(match_dims[1])

    match_dims = (
        normalize_axes(match_dims[0], arr1.shape),
        normalize_axes(match_dims[1], arr2.shape),
    )

    match_axis1 = [i not in match_dims[0] for i in range(len(arr1.shape))]
    match_axis2 = [i not in match_dims[1] for i in range(len(arr2.shape))]

    last_length = len(match_axis1) + len(match_axis2)
    for _ in range(last_length):
        insert_empty_axis(match_axis1, match_axis2)

        if len(match_axis1) + len(match_axis2) == last_length:
            break

        last_length = len(match_axis1) + len(match_axis2)

    max_len = max(len(match_axis1), len(match_axis2))
    padded_match_axis1 = [None] * (max_len - len(match_axis1)) + match_axis1
    padded_match_axis2 = [None] * (max_len - len(match_axis2)) + match_axis2

    axis1 = tuple(i for i, a in enumerate(padded_match_axis1) if a is None)
    axis2 = tuple(i for i, a in enumerate(padded_match_axis2) if a is None)

    return axis1, axis2


class EqualityMixin:
    def __eq__(self, other):
        return safe_equality(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)


def safe_floor_int(n: float, tol: int = 7) -> int:
    return int(np.floor(np.round(n, decimals=tol)))


def safe_ceiling_int(n: float, tol: int = 7) -> int:
    return int(np.ceil(np.round(n, decimals=tol)))


def ensure_list(x):
    return [x] if not isinstance(x, list) else x


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


def normalize_axes(
    axes: tuple[int, ...] | int, shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Normalize the axes tuple so that all axes are non-negative.

    Parameters
    ----------
    axes : tuple
        The axes to normalize.
    shape : tuple
        The shape of the array.

    Returns
    -------
    tuple
        The normalized axes tuple.
    """
    ndim = len(shape)

    # Ensure that 'axes' is a tuple
    if not isinstance(axes, tuple):
        axes = (axes,)

    # Normalize negative indices
    normalized_axes = tuple(axis if axis >= 0 else axis + ndim for axis in axes)

    return normalized_axes


@overload
def expand_dims_to_broadcast(
    arr1: np.ndarray,
    arr2: np.ndarray,
    match_dims: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None,
    broadcast: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ...


@overload
def expand_dims_to_broadcast(
    arr1: da.core.Array,
    arr2: da.core.Array,
    match_dims: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None,
    broadcast: bool = False,
) -> tuple[da.core.Array, da.core.Array]:
    ...


def expand_dims_to_broadcast(
    arr1: np.ndarray | da.core.Array,
    arr2: np.ndarray | da.core.Array,
    match_dims: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None,
    broadcast: bool = False,
) -> tuple[np.ndarray | da.core.Array, np.ndarray | da.core.Array]:
    """
    Expand the dimensions of two arrays to make them broadcastable.

    Parameters
    ----------
    arr1 : np.ndarray
        The first array.
    arr2 : np.ndarray
        The second array.
    match_dims : list, optional
        A list of two tuples, each containing the dimensions that should match (i.e. not
        be broadcasted) between the two arrays.
    broadcast : bool, optional
        If True, broadcast the arrays to the same shape, otherwise only expand the
        dimensions. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the expanded arrays.
    """
    xp = get_array_module(arr1)

    axis1, axis2 = _get_dims_to_broadcast(arr1, arr2, match_dims)

    arr1 = xp.expand_dims(arr1, axis=axis1)
    arr2 = xp.expand_dims(arr2, axis=axis2)

    if broadcast:
        s = xp.broadcast_shapes(arr1.shape, arr2.shape)
        arr1 = xp.broadcast_to(arr1, s)
        arr2 = xp.broadcast_to(arr2, s)

    return arr1, arr2


def tuple_range(length: int, offset: int = 0) -> tuple[int, ...]:
    return tuple(range(offset, offset + length))


def interleave(l1: list | tuple, l2: list | tuple) -> list | tuple:
    """Interleave two lists or tuples."""
    return tuple(val for pair in zip(l1, l2) for val in pair)


def flatten_list_of_lists(lst: list[list]) -> list:
    """Flatten a list of lists into a single list."""
    return list(itertools.chain(*lst))


def label_to_index(
    labels: np.ndarray, max_label: Optional[int] = None, min_label: int = 0
):
    """
    Returns a generator that yields indices for each label in the labels array.

    Parameters
    ----------
    labels : np.ndarray
        An array of integers.
    max_label : int, optional
        The assumed maximum label in the array. If None, the maximum the array is used.
    min_label : int, optional
        The assumed minimum label in the array. Defaults to 0.
    """
    if max_label is None:
        max_label = np.max(labels)

    xp = get_array_module(labels)
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = xp.arange(0, len(labels) + 1)[labels_order]
    index = xp.arange(min_label, max_label + 1)
    lows = xp.searchsorted(sorted_labels, index, side="left")
    highs = xp.searchsorted(sorted_labels, index, side="right")
    for i, (low, high) in enumerate(zip(lows, highs)):
        yield indices[low:high]


def get_data_path(file: str) -> str:
    this_file = os.path.abspath(os.path.dirname(file))
    return os.path.join(this_file, "data")


def get_dtype(complex: bool = False) -> np.dtype:
    """
    Get the numpy dtype from the config precision setting.

    Parameters
    ----------
    complex : bool, optional
        If True, return a complex dtype. Defaults to False.
    """
    dtype = config.get("precision")

    if dtype == "float32" and complex:
        dtype = np.complex64
    elif dtype == "float32":
        dtype = np.float32
    elif dtype == "float64" and complex:
        dtype = np.complex128
    elif dtype == "float64":
        dtype = np.float64
    else:
        raise RuntimeError(f"Invalid dtype: {dtype}")

    return dtype


def is_scalar(value) -> TypeGuard[float | int | np.floating | np.integer]:
    """
    Check if the value is a float, int, or a NumPy scalar.

    Parameters
    ----------
    value : any
        The value to check.

    Returns
    -------
    bool
        True if the value is a float, int, or a NumPy scalar, False otherwise.
    """
    return isinstance(value, (float, int, np.floating, np.integer))
