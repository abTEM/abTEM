from __future__ import annotations

from enum import Enum
from types import EllipsisType
from typing import (
    Any,
    Literal,
    Protocol,
    TypeVar,
    TypeVarTuple,
    runtime_checkable,
)

import numpy as np

PyCapsule = Any
type Shape = tuple[int, ...]

type ComplexDType = np.dtype[np.complexfloating] | complex
type RealDType = np.dtype[np.floating] | float
type IntDType = np.dtype[np.integer] | int
type DType = ComplexDType | RealDType | IntDType
type Device = Any

ShapeT = TypeVar("ShapeT", covariant=True)
DTypeT = TypeVar("DTypeT", bound=DType, covariant=True)

@runtime_checkable
class Array(Protocol[ShapeT, DTypeT]):
    #__abtem_array_protocol__: ClassVar[bool]

    def __init__(self) -> None: ...

    @property
    def dtype(self) -> DTypeT: ...

    @property
    def device(self) -> Device: ...

    @property
    def mT(self) -> Array[Any, Any]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    @property
    def T(self) -> Array[Any, Any]: ...

    def __iter__(self) -> Array[Any, Any]: ...

    def __next__(self) -> Array[Any, Any]: ...

    def __abs__(self, /) -> Array[Any, Any]: ...

    def __add__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __and__(self, other: int | bool | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __array_namespace__(self, /, *, api_version: str | None = None) -> Any: ...

    def __bool__(self, /) -> bool: ...

    def __complex__(self, /) -> complex: ...

    def __dlpack__(
        self,
        /,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[Enum, int] | None = None,
        copy: bool | None = None,
    ) -> PyCapsule: ...

    def __dlpack_device__(self, /) -> tuple[Enum, int]: ...

    def __eq__(self, other: Any, /) -> Array[Any, Any]: ...  # type: ignore[override]

    def __float__(self, /) -> float: ...

    def __floordiv__(
        self, other: int | float | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __ge__(self, other: int | float | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __getitem__(
        self,
        key: int
        | slice
        | EllipsisType
        | None
        | tuple[int | slice | EllipsisType | Array[Any, Any] | None, ...]
        | Array[Any, Any],
        /,
    ) -> Array[Any, Any]: ...

    def __gt__(self, other: int | float | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __index__(self, /) -> int: ...

    def __int__(self, /) -> int: ...

    def __invert__(self, /) -> Array[Any, Any]: ...

    def __le__(self, other: int | float | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __lshift__(self, other: int | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __lt__(self, other: int | float | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __matmul__(self, other: Array[Any, Any], /) -> Array[Any, Any]: ...

    def __mod__(self, other: int | float | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __mul__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __ne__(self, other: Any, /) -> Array[Any, Any]: ...  # type: ignore[override]

    def __neg__(self, /) -> Array[Any, Any]: ...

    def __or__(self, other: int | bool | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __pos__(self, /) -> Array[Any, Any]: ...

    def __pow__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __rshift__(self, other: int | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __setitem__(
        self,
        key: int
        | slice
        | EllipsisType
        | tuple[int | slice | EllipsisType | Array[Any, Any], ...]
        | Array[Any, Any],
        value: int | float | complex | bool | Array[Any, Any],
        /,
    ) -> None: ...

    def __sub__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __truediv__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __xor__(self, other: int | bool | Array[Any, Any], /) -> Array[Any, Any]: ...

    def to_device(
        self, device: Device, /, *, stream: int | Any | None = None
    ) -> Array[Any, Any]: ...

    def __radd__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __rand__(self, other: int | bool | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __rfloordiv__(
        self, other: int | float | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __rlshift__(self, other: int | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __rmatmul__(self, other: Array[Any, Any], /) -> Array[Any, Any]: ...

    def __rmod__(self, other: int | float | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __rmul__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __ror__(self, other: int | bool | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __rpow__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __rrshift__(self, other: int | Array[Any, Any], /) -> Array[Any, Any]: ...

    def __rsub__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __rtruediv__(
        self, other: int | float | complex | Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def __rxor__(self, other: int | bool | Array[Any, Any], /) -> Array[Any, Any]: ...


class LazyArray(Array[ShapeT, DTypeT]):
    @property
    def chunks(self) -> tuple[tuple[int, ...], ...]: ...

    def compute(self) -> Array[ShapeT, DTypeT]: ...


N = TypeVar("N", covariant=True, bound=int)
M = TypeVar("M", covariant=True, bound=int)
DTypeT = TypeVar("DTypeT", bound=DType, covariant=True)
B = TypeVarTuple("B")
AnyArray = Array[tuple[int, ...], DTypeT]
Matrix = Array[tuple[N, M], DTypeT]
SquareMatrix = Array[tuple[N, N], DTypeT]
SquareMatrixBatch = Array[tuple[*B, N, N], DTypeT]
Vector = Array[tuple[N], DTypeT]
Vector1D = Array[tuple[N, Literal[1]], DTypeT]
Vector2D = Array[tuple[N, Literal[2]], DTypeT]
Vector3D = Array[tuple[N, Literal[3]], DTypeT]
VectorBatch = Array[tuple[*B, N], DTypeT]
Scalar = Array[tuple[()], DTypeT]
CellLike = Array[tuple[Literal[3], Literal[3]], DTypeT]
CellLikeBatch = Array[tuple[*B, Literal[3], Literal[3]], DTypeT]
