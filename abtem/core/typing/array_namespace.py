# ruff: noqa: E501
from collections.abc import Buffer as SupportsBufferProtocol
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar, overload

from .array_object import Array, Device, DType, DTypeT, Scalar
from .nested_sequence import _NestedSequence as NestedSequence

BuiltIn = TypeVar("BuiltIn", bound=int | float)


class ToplevelNamespaceGeneric(Protocol):
    e: float
    inf: float
    nan: float
    newaxis = None
    pi: float

    @overload
    def arange(
        self,
        start: int | float | Scalar,
        /,
        stop: int | float | Scalar | None = None,
        step: int | float | Scalar = 1,
        *,
        dtype: type[BuiltIn],
        device: Device | None = None,
    ) -> Array[Any, BuiltIn]: ...

    @overload
    def arange(
        self,
        start: int | float | Scalar,
        /,
        stop: int | float | Scalar | None = None,
        step: int | float | Scalar = 1,
        *,
        dtype: None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    @overload
    def arange(
        self,
        start: int | float | Scalar,
        /,
        stop: int | float | Scalar | None = None,
        step: int | float | Scalar = 1,
        *,
        dtype: DTypeT,
        device: Device | None = None,
    ) -> Array[Any, DTypeT]: ...

    def arange(
        self,
        start: int | float | Scalar,
        /,
        stop: int | float | Scalar | None = None,
        step: int | float | Scalar = 1,
        *,
        dtype: DTypeT | type[BuiltIn] | None = None,
        device: Device | None = None,
    ) -> Array[Any, DTypeT | BuiltIn]: ...

    @overload
    def asarray(
        self,
        obj: Array[Any, Any]
        | bool
        | int
        | float
        | complex
        | NestedSequence
        | SupportsBufferProtocol,
        /,
        *,
        dtype: type[BuiltIn],
        device: Device | None = None,
        copy: bool | None = None,
    ) -> Array[Any, BuiltIn]: ...

    @overload
    def asarray(
        self,
        obj: Array[Any, Any]
        | bool
        | int
        | float
        | complex
        | NestedSequence
        | SupportsBufferProtocol,
        /,
        *,
        dtype: None = None,
        device: Device | None = None,
        copy: bool | None = None,
    ) -> Array[Any, Any]: ...

    @overload
    def asarray(
        self,
        obj: Array[Any, Any]
        | bool
        | int
        | float
        | complex
        | NestedSequence
        | SupportsBufferProtocol,
        /,
        *,
        dtype: DTypeT,
        device: Device | None = None,
        copy: bool | None = None,
    ) -> Array[Any, DTypeT]: ...

    def asarray(
        self,
        obj: Array[Any, Any]
        | bool
        | int
        | float
        | complex
        | NestedSequence
        | SupportsBufferProtocol,
        /,
        *,
        dtype: DTypeT | type[BuiltIn] | None = None,
        device: Device | None = None,
        copy: bool | None = None,
    ) -> Array[Any, DTypeT | BuiltIn]: ...

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def empty_like(
        self,
        x: Array[Any, Any],
        /,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    @overload
    def eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: type[BuiltIn],
        device: Device | None = None,
    ) -> Array[Any, BuiltIn]: ...

    @overload
    def eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    @overload
    def eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: DTypeT,
        device: Device | None = None,
    ) -> Array[Any, DTypeT]: ...

    def eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: DTypeT | type[BuiltIn] | None = None,
        device: Device | None = None,
    ) -> Array[Any, DTypeT | BuiltIn]: ...

    def from_dlpack(
        self, x: object, /, *, device: Device | None = None, copy: bool | None = None
    ) -> Array[Any, Any]: ...

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: bool | int | float | complex | Array,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def full_like(
        self,
        x: Array[Any, Any],
        /,
        fill_value: bool | int | float | complex,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def linspace(
        self,
        start: int | float | complex | Scalar,
        stop: int | float | complex | Scalar,
        /,
        num: int,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        endpoint: bool = True,
    ) -> Array[Any, Any]: ...

    def meshgrid(
        self, *arrays: Array[Any, Any], indexing: Literal["xy", "ij"] = "xy"
    ) -> list[Array[Any, Any]]: ...

    @overload
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: type[BuiltIn],
        device: Device | None = None,
    ) -> Array[Any, BuiltIn]: ...

    @overload
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    @overload
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT,
        device: Device | None = None,
    ) -> Array[Any, DTypeT]: ...

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | type[BuiltIn] | None = None,
        device: Device | None = None,
    ) -> Array[Any, DTypeT | BuiltIn]: ...

    def ones_like(
        self,
        x: Array[Any, Any],
        /,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def tril(self, x: Array[Any, Any], /, *, k: int = 0) -> Array[Any, Any]: ...

    def triu(self, x: Array[Any, Any], /, *, k: int = 0) -> Array[Any, Any]: ...

    @overload
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: type[BuiltIn],
        device: Device | None = None,
    ) -> Array[Any, BuiltIn]: ...

    @overload
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    @overload
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT,
        device: Device | None = None,
    ) -> Array[Any, DTypeT]: ...

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: DTypeT | type[BuiltIn] | None = None,
        device: Device | None = None,
    ) -> Array[Any, DTypeT | BuiltIn]: ...

    def zeros_like(
        self,
        x: Array[Any, Any],
        /,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def astype(
        self,
        x: Array[Any, Any],
        dtype: DType,
        /,
        *,
        copy: bool = True,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def can_cast(self, from_: DType | Array[Any, Any], to: DType, /) -> bool: ...

    def isdtype(
        self, dtype: DType, kind: DType | str | tuple[DType | str, ...]
    ) -> bool: ...

    def result_type(
        self, *arrays_and_dtypes: Array[Any, Any] | int | float | complex | bool | DType
    ) -> DType: ...

    def abs(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def acos(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def acosh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def add(
        self,
        x1: Array[Any, Any] | int | float | complex,
        x2: Array[Any, Any] | int | float | complex,
        /,
    ) -> Array[Any, Any]: ...

    def asin(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def asinh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def atan(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def atan2(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def atanh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def bitwise_and(
        self, x1: Array[Any, Any] | int | bool, x2: Array[Any, Any] | int | bool, /
    ) -> Array[Any, Any]: ...

    def bitwise_left_shift(
        self, x1: Array[Any, Any] | int, x2: Array[Any, Any] | int, /
    ) -> Array[Any, Any]: ...

    def bitwise_invert(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def bitwise_or(
        self, x1: Array[Any, Any] | int | bool, x2: Array[Any, Any] | int | bool, /
    ) -> Array[Any, Any]: ...

    def bitwise_right_shift(
        self, x1: Array[Any, Any] | int, x2: Array[Any, Any] | int, /
    ) -> Array[Any, Any]: ...

    def bitwise_xor(
        self, x1: Array[Any, Any] | int | bool, x2: Array[Any, Any] | int | bool, /
    ) -> Array[Any, Any]: ...

    def ceil(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def clip(
        self,
        x: Array[Any, Any],
        /,
        min: int | float | Array[Any, Any] | None = None,
        max: int | float | Array[Any, Any] | None = None,
    ) -> Array[Any, Any]: ...

    def conj(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def copysign(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def cos(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def cosh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def divide(
        self,
        x1: Array[Any, Any] | int | float | complex,
        x2: Array[Any, Any] | int | float | complex,
        /,
    ) -> Array[Any, Any]: ...

    def equal(
        self,
        x1: Array[Any, Any] | int | float | complex | bool,
        x2: Array[Any, Any] | int | float | complex | bool,
        /,
    ) -> Array[Any, Any]: ...

    def exp(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def expm1(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def floor(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def floor_divide(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def greater(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def greater_equal(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def hypot(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def imag(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def isfinite(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def isinf(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def isnan(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def less(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def less_equal(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def log(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def log1p(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def log2(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def log10(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def logaddexp(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def logical_and(
        self, x1: Array[Any, Any] | bool, x2: Array[Any, Any] | bool, /
    ) -> Array[Any, Any]: ...

    def logical_not(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def logical_or(
        self, x1: Array[Any, Any] | bool, x2: Array[Any, Any] | bool, /
    ) -> Array[Any, Any]: ...

    def logical_xor(
        self, x1: Array[Any, Any] | bool, x2: Array[Any, Any] | bool, /
    ) -> Array[Any, Any]: ...

    def maximum(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def minimum(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def multiply(
        self,
        x1: Array[Any, Any] | int | float | complex,
        x2: Array[Any, Any] | int | float | complex,
        /,
    ) -> Array[Any, Any]: ...

    def negative(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def nextafter(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def not_equal(
        self,
        x1: Array[Any, Any] | int | float | complex | bool,
        x2: Array[Any, Any] | int | float | complex | bool,
        /,
    ) -> Array[Any, Any]: ...

    def positive(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def pow(
        self,
        x1: Array[Any, Any] | int | float | complex,
        x2: Array[Any, Any] | int | float | complex,
        /,
    ) -> Array[Any, Any]: ...

    def real(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def reciprocal(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def remainder(
        self, x1: Array[Any, Any] | int | float, x2: Array[Any, Any] | int | float, /
    ) -> Array[Any, Any]: ...

    def round(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def sign(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def signbit(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def sin(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def sinh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def square(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def sqrt(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def subtract(
        self,
        x1: Array[Any, Any] | int | float | complex,
        x2: Array[Any, Any] | int | float | complex,
        /,
    ) -> Array[Any, Any]: ...

    def tan(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def tanh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def trunc(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def take(
        self,
        x: Array[Any, Any],
        indices: Array[Any, Any],
        /,
        *,
        axis: int | None = None,
    ) -> Array[Any, Any]: ...

    def take_along_axis(
        self, x: Array[Any, Any], indices: Array[Any, Any], /, *, axis: int = -1
    ) -> Array[Any, Any]: ...

    def matmul(
        self, x1: Array[Any, Any], x2: Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def matrix_transpose(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def tensordot(
        self,
        x1: Array[Any, Any],
        x2: Array[Any, Any],
        /,
        *,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> Array[Any, Any]: ...

    def vecdot(
        self, x1: Array[Any, Any], x2: Array[Any, Any], /, *, axis: int = -1
    ) -> Array[Any, Any]: ...

    def broadcast_arrays(self, *arrays: Array[Any, Any]) -> list[Array[Any, Any]]: ...

    def broadcast_to(
        self, x: Array[Any, Any], /, shape: tuple[int, ...]
    ) -> Array[Any, Any]: ...

    def concat(
        self,
        arrays: tuple[Array[Any, Any], ...] | list[Array[Any, Any]],
        /,
        *,
        axis: int | None = 0,
    ) -> Array[Any, Any]: ...

    def expand_dims(
        self, x: Array[Any, Any], /, *, axis: int = 0
    ) -> Array[Any, Any]: ...

    def flip(
        self, x: Array[Any, Any], /, *, axis: int | tuple[int, ...] | None = None
    ) -> Array[Any, Any]: ...

    def moveaxis(
        self,
        x: Array[Any, Any],
        source: int | tuple[int, ...],
        destination: int | tuple[int, ...],
        /,
    ) -> Array[Any, Any]: ...

    def permute_dims(
        self, x: Array[Any, Any], /, axes: tuple[int, ...]
    ) -> Array[Any, Any]: ...

    def repeat(
        self,
        x: Array[Any, Any],
        repeats: int | Array[Any, Any],
        /,
        *,
        axis: int | None = None,
    ) -> Array[Any, Any]: ...

    def reshape(
        self, x: Array[Any, Any], /, shape: tuple[int, ...], *, copy: bool | None = None
    ) -> Array[Any, Any]: ...

    def roll(
        self,
        x: Array[Any, Any],
        /,
        shift: int | tuple[int, ...],
        *,
        axis: int | tuple[int, ...] | None = None,
    ) -> Array[Any, Any]: ...

    def squeeze(
        self, x: Array[Any, Any], /, axis: int | tuple[int, ...]
    ) -> Array[Any, Any]: ...

    def stack(
        self,
        arrays: tuple[Array[Any, Any], ...] | list[Array[Any, Any]],
        /,
        *,
        axis: int = 0,
    ) -> Array[Any, Any]: ...

    def tile(
        self, x: Array[Any, Any], repetitions: tuple[int, ...], /
    ) -> Array[Any, Any]: ...

    def unstack(
        self, x: Array[Any, Any], /, *, axis: int = 0
    ) -> tuple[Array[Any, Any], ...]: ...

    def argmax(
        self, x: Array[Any, Any], /, *, axis: int | None = None, keepdims: bool = False
    ) -> Array[Any, Any]: ...

    def argmin(
        self, x: Array[Any, Any], /, *, axis: int | None = None, keepdims: bool = False
    ) -> Array[Any, Any]: ...

    def count_nonzero(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def nonzero(self, x: Array[Any, Any], /) -> tuple[Array[Any, Any], ...]: ...

    def searchsorted(
        self,
        x1: Array[Any, Any],
        x2: Array[Any, Any],
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: Array[Any, Any] | None = None,
    ) -> Array[Any, Any]: ...

    def where(
        self,
        condition: Array[Any, Any],
        x1: Array[Any, Any] | int | float | complex | bool,
        x2: Array[Any, Any] | int | float | complex | bool,
        /,
    ) -> Array[Any, Any]: ...

    def unique_all(
        self, x: Array[Any, Any], /
    ) -> tuple[Array[Any, Any], Array[Any, Any], Array[Any, Any], Array[Any, Any]]: ...

    def unique_counts(
        self, x: Array[Any, Any], /
    ) -> tuple[Array[Any, Any], Array[Any, Any]]: ...

    def unique_inverse(
        self, x: Array[Any, Any], /
    ) -> tuple[Array[Any, Any], Array[Any, Any]]: ...

    def unique_values(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def argsort(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> Array[Any, Any]: ...

    def sort(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> Array[Any, Any]: ...

    def cumulative_prod(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> Array[Any, Any]: ...

    def cumulative_sum(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> Array[Any, Any]: ...

    def max(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def mean(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def min(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def prod(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def std(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float | Scalar = 0.0,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def sum(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def var(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float | Scalar = 0.0,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def all(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def any(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array[Any, Any]: ...

    def diff(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int = -1,
        n: int = 1,
        prepend: Array[Any, Any] | None = None,
        append: Array[Any, Any] | None = None,
    ) -> Array[Any, Any]: ...

    def allclose(
        self,
        a: Array[Any, Any],
        b: Array[Any, Any],
        /,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool: ...

    def isclose(
        self,
        a: float | Array[Any, Any],
        b: float | Array[Any, Any],
        /,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool: ...


class FFTNamespace(Protocol):
    def fft(
        self,
        x: Array[Any, Any],
        /,
        *,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def ifft(
        self,
        x: Array[Any, Any],
        /,
        *,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def fftn(
        self,
        x: Array[Any, Any],
        /,
        *,
        s: Sequence[int] | None = None,
        axes: Sequence[int] | None = None,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def ifftn(
        self,
        x: Array[Any, Any],
        /,
        *,
        s: Sequence[int] | None = None,
        axes: Sequence[int] | None = None,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def rfft(
        self,
        x: Array[Any, Any],
        /,
        *,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def irfft(
        self,
        x: Array[Any, Any],
        /,
        *,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def rfftn(
        self,
        x: Array[Any, Any],
        /,
        *,
        s: Sequence[int] | None = None,
        axes: Sequence[int] | None = None,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def irfftn(
        self,
        x: Array[Any, Any],
        /,
        *,
        s: Sequence[int] | None = None,
        axes: Sequence[int] | None = None,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def hfft(
        self,
        x: Array[Any, Any],
        /,
        *,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def ihfft(
        self,
        x: Array[Any, Any],
        /,
        *,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
    ) -> Array[Any, Any]: ...

    def fftfreq(
        self,
        n: int,
        /,
        *,
        d: float = 1.0,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def rfftfreq(
        self,
        n: int,
        /,
        *,
        d: float = 1.0,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Array[Any, Any]: ...

    def fftshift(
        self, x: Array[Any, Any], /, *, axes: int | Sequence[int] | None = None
    ) -> Array[Any, Any]: ...

    def ifftshift(
        self, x: Array[Any, Any], /, *, axes: int | Sequence[int] | None = None
    ) -> Array[Any, Any]: ...


class LinalgNamespace(Protocol):
    def cholesky(
        self, x: Array[Any, Any], /, *, upper: bool = False
    ) -> Array[Any, Any]: ...

    def cross(
        self, x1: Array[Any, Any], x2: Array[Any, Any], /, *, axis: int = -1
    ) -> Array[Any, Any]: ...

    def det(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def diagonal(
        self, x: Array[Any, Any], /, *, offset: int = 0
    ) -> Array[Any, Any]: ...

    def eigh(
        self, x: Array[Any, Any], /
    ) -> tuple[Array[Any, Any], Array[Any, Any]]: ...

    def eigvalsh(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def inv(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def matmul(
        self, x1: Array[Any, Any], x2: Array[Any, Any], /
    ) -> Array[Any, Any]: ...

    def matrix_norm(
        self,
        x: Array[Any, Any],
        /,
        *,
        keepdims: bool = False,
        ord: int | float | Literal["fro", "nuc"] | None = "fro",
    ) -> Array[Any, Any]: ...

    def matrix_power(self, x: Array[Any, Any], n: int, /) -> Array[Any, Any]: ...

    def matrix_rank(
        self, x: Array[Any, Any], /, *, rtol: float | Array[Any, Any] | None = None
    ) -> Array[Any, Any]: ...

    def matrix_transpose(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def outer(self, x1: Array[Any, Any], x2: Array[Any, Any], /) -> Array[Any, Any]: ...

    def pinv(
        self, x: Array[Any, Any], /, *, rtol: float | Array[Any, Any] | None = None
    ) -> Array[Any, Any]: ...

    def qr(
        self, x: Array[Any, Any], /, *, mode: Literal["reduced", "complete"] = "reduced"
    ) -> tuple[Array[Any, Any], Array[Any, Any]]: ...

    def slogdet(
        self, x: Array[Any, Any], /
    ) -> tuple[Array[Any, Any], Array[Any, Any]]: ...

    def solve(self, x1: Array[Any, Any], x2: Array[Any, Any], /) -> Array[Any, Any]: ...

    def svd(
        self, x: Array[Any, Any], /, *, full_matrices: bool = True
    ) -> tuple[Array[Any, Any], Array[Any, Any], Array[Any, Any]]: ...

    def svdvals(self, x: Array[Any, Any], /) -> Array[Any, Any]: ...

    def tensordot(
        self,
        x1: Array[Any, Any],
        x2: Array[Any, Any],
        /,
        *,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> Array[Any, Any]: ...

    def trace(
        self, x: Array[Any, Any], /, *, offset: int = 0, dtype: DType | None = None
    ) -> Array[Any, Any]: ...

    def vecdot(
        self, x1: Array[Any, Any], x2: Array[Any, Any], /, *, axis: int = -1
    ) -> Array[Any, Any]: ...

    def vector_norm(
        self,
        x: Array[Any, Any],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,
    ) -> Array[Any, Any]: ...


class ArrayNamespace(ToplevelNamespaceGeneric):
    fft: FFTNamespace
    linalg: LinalgNamespace
