
from __future__ import annotations

from typing import Any, cast

import dask.array as da
import numba as nb
import numpy as np
from numpy.typing import NDArray

from abtem.core.backend2 import (
    cp,
    is_cupy_array,
    is_numpy_array,
    is_torch_array,
    torch,
)
from abtem.core.backend2.checks import is_lazy_array, is_scalar
from abtem.core.backend2.dtype import to_real_dtype
from abtem.core.typing import Array, ComplexDType, RealDType
from abtem.core.typing.array_object import DType


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def _abs2_numba(x):
    return x.real**2 + x.imag**2


if cp is not None:
    _abs2_cupy_float32 = cp.ElementwiseKernel(
        in_params="float32 x, float32 y",
        out_params="float32 z",
        operation="z = x * x + y * y",
        name="abs_squared",
    )
    _abs2_cupy_float64 = cp.ElementwiseKernel(
        in_params="float64 x, float64 y",
        out_params="float64 z",
        operation="z = x * x + y * y",
        name="abs_squared",
    )
else:
    _abs2_cupy_float32 = None
    _abs2_cupy_float64 = None


def _abs2_cupy(x: Any) -> Any:
    if x.dtype == "complex64":
        assert _abs2_cupy_float32 is not None
        return _abs2_cupy_float32(x.real, x.imag)
    elif x.dtype == "complex128":
        assert _abs2_cupy_float64 is not None
        return _abs2_cupy_float64(x.real, x.imag)
    else:
        raise RuntimeError("Unsupported dtype for calculation of abs2 with cupy array.")


def _abs2_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x) ** 2


def abs2[Shape](x: Array[Shape, DType]) -> Array[Shape, RealDType]:
    if is_lazy_array(x):
        return da.map_blocks(
            abs2,
            x,
            dtype=to_real_dtype(x.dtype),
        )

    if is_cupy_array(x):
        x_cp = cast(Any, x)
        if cp.iscomplexobj(x_cp):
            return cast(Array, _abs2_cupy(x_cp))
        else:
            return cast(Array, cp.abs(x_cp) ** 2)
    elif is_numpy_array(x) or is_scalar(x):
        x_np = cast(NDArray[Any], x)
        if np.iscomplexobj(x_np):
            return cast(Array, _abs2_numba(x_np))
        else:
            return cast(Array, np.abs(x_np) ** 2)
    elif is_torch_array(x):
        x_torch = cast(torch.Tensor, x)
        return cast(Array, _abs2_torch(x_torch))
    else:
        raise TypeError("Unsupported array type")


@nb.vectorize(
    [nb.complex64(nb.float32), nb.complex128(nb.float64), nb.complex64(nb.complex64)]
)
def _complex_exponential_numba(x: Any) -> Any:
    return np.cos(x) + 1.0j * np.sin(x)


def _complex_exponential_cupy(x: Any) -> Any:
    return cp.exp(1.0j * x)


def _complex_exponential_torch(x: torch.Tensor) -> torch.Tensor:
    real = torch.cos(x)
    imag = torch.sin(x)
    return torch.complex(real, imag)


def complex_exponential[Shape](
    x: Array[Shape, RealDType | ComplexDType],
) -> Array[Shape, ComplexDType]:
    if is_cupy_array(x):
        x_cp = cast(Any, x)
        return cast(Array, _complex_exponential_cupy(x_cp))
    elif is_numpy_array(x):
        x_np = cast(NDArray[Any], x)
        return cast(Array, _complex_exponential_numba(x_np))
    elif is_torch_array(x):
        x_torch = cast(torch.Tensor, x)
        return cast(Array, _complex_exponential_torch(x_torch))
    else:
        raise TypeError("Unsupported array type")
