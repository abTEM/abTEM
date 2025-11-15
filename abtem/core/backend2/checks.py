from __future__ import annotations

from typing import (
    Any,
    Callable,
    TypeGuard,
    TypeVar,
)

import numpy as np
from dask.array.core import Array as DaskArray

from abtem.core.backend2 import (
    cp,
    is_cupy_array,
    is_numpy_array,
    is_scalar_array,
    is_torch_array,
)
from abtem.core.backend2.array_lib import is_any_array
from abtem.core.typing import (
    Array,
)
from abtem.core.typing.array_object import ComplexDType, DType, Scalar

T = TypeVar("T")
U = TypeVar("U")


def is_scalar(x) -> TypeGuard[int | float | complex | Scalar[DType]]:
    if isinstance(x, (int, float, complex)):
        return True
    return is_scalar_array(x)


def or_typeguard(
    guard1: Callable[[Any], TypeGuard[T]],
    guard2: Callable[[Any], TypeGuard[U]],
) -> Callable[[Any], TypeGuard[T] | TypeGuard[U]]:
    def guard(x: Any) -> TypeGuard[T] | TypeGuard[U]:
        return guard1(x) or guard2(x)

    return guard


def and_typeguard(
    guard1: Callable[[Any], TypeGuard[T]],
    guard2: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[T]]:
    def guard(x: Any) -> TypeGuard[T]:
        return guard1(x) and guard2(x)

    return guard


def is_tuple_of(
    item_guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[tuple[T, ...]]]:
    def guard(x: Any) -> TypeGuard[tuple[T, ...]]:
        return isinstance(x, tuple) and all(item_guard(item) for item in x)

    return guard


def is_list_of(
    item_guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[list[T]]]:
    def guard(x: Any) -> TypeGuard[list[T]]:
        return isinstance(x, list) and all(item_guard(item) for item in x)

    return guard


def is_pair_of(
    item_guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[tuple[T, T]]]:
    def guard(x: Any) -> TypeGuard[tuple[T, T]]:
        return (
            isinstance(x, tuple) and len(x) == 2 and all(item_guard(item) for item in x)
        )

    return guard


def is_triple_of(
    item_guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[tuple[T, T, T]]]:
    def guard(x: Any) -> TypeGuard[tuple[T, T, T]]:
        return (
            isinstance(x, tuple) and len(x) == 3 and all(item_guard(item) for item in x)
        )

    return guard


def is_tuple_of_pair_of(
    item_guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[tuple[tuple[T, T], ...]]]:
    def guard(x: Any) -> TypeGuard[tuple[tuple[T, T], ...]]:
        return isinstance(x, tuple) and all(
            isinstance(item, tuple)
            and len(item) == 2
            and all(item_guard(subitem) for subitem in item)
            for item in x
        )

    return guard


def is_tuple_of_triple_of(
    item_guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[tuple[tuple[T, T, T], ...]]]:
    def guard(x: Any) -> TypeGuard[tuple[tuple[T, T, T], ...]]:
        return isinstance(x, tuple) and all(
            isinstance(item, tuple)
            and len(item) == 3
            and all(item_guard(subitem) for subitem in item)
            for item in x
        )

    return guard


def is_optional(
    guard: Callable[[Any], TypeGuard[T]],
) -> Callable[[Any], TypeGuard[T | None]]:
    def inner(x: Any) -> TypeGuard[T | None]:
        return x is None or guard(x)

    return inner


def is_int(x: Any) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool)


def is_float(x: Any) -> TypeGuard[float]:
    return isinstance(x, float) and not isinstance(x, bool)


def is_tuple(x: Any) -> TypeGuard[tuple[Any]]:
    return isinstance(x, tuple)


def is_str(x: Any) -> TypeGuard[str]:
    return isinstance(x, str)


def is_instance_of[T](cls: type[T]) -> Callable[[Any], TypeGuard[T]]:
    def guard(x: Any) -> TypeGuard[T]:
        return isinstance(x, cls)

    return guard


def is_none(x: Any) -> TypeGuard[None]:
    return x is None


def is_tuple_of_int(x: Any) -> TypeGuard[tuple[int, ...]]:
    return is_tuple_of(is_int)(x)


def is_tuple_of_tuple_of_int(x: Any) -> TypeGuard[tuple[tuple[int, ...], ...]]:
    return is_tuple_of(is_tuple_of_int)(x)


def is_int_or_tuple_of_int(x: Any) -> TypeGuard[int | tuple[int, ...]]:
    return or_typeguard(is_int, is_tuple_of_int)(x)


def is_complex_object(x: Array) -> TypeGuard[Array[Any, ComplexDType]]:
    if isinstance(x, DaskArray):
        return is_complex_object(x._meta)

    if is_numpy_array(x):
        return np.iscomplexobj(x)
    elif is_torch_array(x):
        return x.is_complex()  # type: ignore
    elif is_cupy_array(x):
        return cp.iscomplexobj(x)
    else:
        raise TypeError(
            f"Unsupported array type: {type(x)}. Expected NumPy, CuPy, or Torch array."
        )


def is_lazy_array(x: Any) -> bool:
    return isinstance(x, DaskArray)


def array_dtype_is[T: DType](
    array: Array[Any, Any], dtype: T | Array[Any, T]
) -> TypeGuard[Array[Any, T]]:
    if is_any_array(dtype):
        dtype = dtype.dtype
    return array.dtype is dtype
