"""Module for handling the array backend (NumPy, CuPy, Dask, etc.) of the library."""

from __future__ import annotations

from typing import Literal, TypeVar, cast, overload

import numpy as np

from abtem.core.backend2.array_lib import (
    BackendName,
    TorchNotPresentError,
    array_namespace_name,
    torch,
)
from abtem.core.config import config
from abtem.core.typing import ComplexDType, DType, RealDType
from abtem.core.typing.array_namespace import (
    ArrayNamespace,
)

DTypeString = Literal[
    "float32",
    "float64",
    "complex64",
    "complex128",
    "int32",
    "int64",
    "uint8",
    "bool",
]

DTypeLike = DType | DTypeString

DT = TypeVar("DT", bound=DType, covariant=True)


def validate_dtype_numpy(dtype: DTypeLike) -> np.dtype:
    try:
        return np.dtype(dtype)  # type: ignore
    except Exception:
        pass

    try:
        return np.dtype(getattr(dtype, "dtype"))
    except Exception:
        pass

    try:
        return torch.tensor([], dtype=dtype).numpy().dtype  # type: ignore
    except Exception:
        pass
    raise TypeError(f"Cannot convert {dtype!r} to numpy dtype")


def get_numpy_torch_dtype_mappings() -> tuple[dict, dict]:
    if torch is None:
        raise TorchNotPresentError()
    numpy_to_torch = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool"): torch.bool,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
    }
    torch_to_numpy = {v: k for k, v in numpy_to_torch.items()}
    return numpy_to_torch, torch_to_numpy


def is_torch_dtype(dtype) -> bool:
    if torch is None:
        return False
    try:
        return isinstance(dtype, torch.dtype)
    except Exception:
        return False


def ensure_numpy_dtype(dtype: DType) -> np.dtype:
    try:
        return torch_to_numpy_dtype(dtype)
    except KeyError:
        return validate_dtype_numpy(dtype)


def numpy_to_torch_dtype(dtype: DType) -> torch.dtype:
    numpy_to_torch, _ = get_numpy_torch_dtype_mappings()
    dtype = validate_dtype_numpy(dtype)
    return numpy_to_torch[dtype]


def torch_to_numpy_dtype(dtype) -> np.dtype:
    _, torch_to_numpy = get_numpy_torch_dtype_mappings()
    return torch_to_numpy[dtype]


def validate_dtype_torch(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    numpy_dtype = validate_dtype_numpy(dtype)
    return numpy_to_torch_dtype(numpy_dtype)


@overload
def validate_dtype(
    dtype: DT, backend: BackendName | ArrayNamespace | None = ...
) -> DT: ...


@overload
def validate_dtype(
    dtype: DTypeString, backend: BackendName | ArrayNamespace | None = ...
) -> DType: ...


@overload
def validate_dtype(
    dtype: None = None, backend: BackendName | ArrayNamespace | None = ...
) -> RealDType: ...


def validate_dtype(
    dtype: DT | DTypeString | None = None,
    backend: BackendName | ArrayNamespace | None = None,
) -> DT | DType | RealDType:
    if backend is None:
        backend = config.get("backend", "numpy")

    if dtype is None:
        dtype = config.get("precision", "float32")

    if not isinstance(backend, str):
        backend = array_namespace_name(backend)

    if isinstance(dtype, np.dtype):
        if backend == "numpy":
            return dtype
        if backend == "torch":
            return numpy_to_torch_dtype(dtype)
        raise ValueError("backend must be 'numpy' or 'torch'")

    if torch is not None:
        try:
            if isinstance(dtype, torch.dtype):
                if backend == "torch":
                    return cast(DType, dtype)
                if backend == "numpy":
                    return torch_to_numpy_dtype(dtype)
                raise ValueError("backend must be 'numpy' or 'torch'")
        except Exception:
            # Fall through to string/other conversion below
            pass

    if backend == "numpy":
        return validate_dtype_numpy(dtype)  # type: ignore[arg-type]
    elif backend == "torch":
        return validate_dtype_torch(dtype)  # type: ignore[arg-type]
    else:
        raise ValueError("backend must be 'numpy' or 'torch'")


def to_real_dtype_numpy(complex_dtype: np.dtype) -> np.dtype:
    if np.issubdtype(complex_dtype, np.floating):
        return np.dtype(complex_dtype)

    if np.issubdtype(complex_dtype, np.complexfloating):
        if np.dtype(complex_dtype) == np.dtype("complex64"):
            return np.dtype("float32")
        elif np.dtype(complex_dtype) == np.dtype("complex128"):
            return np.dtype("float64")
        else:
            raise ValueError(f"unsupported complex dtype: {complex_dtype}")
    else:
        raise ValueError(
            f"input dtype is not a (complex) floating dtype: {complex_dtype}"
        )


def to_real_dtype(complex_dtype: DType) -> RealDType:
    if isinstance(complex_dtype, np.dtype):
        return to_real_dtype_numpy(complex_dtype)
    elif isinstance(complex_dtype, torch.dtype):
        return torch.dtype.to_real(complex_dtype)
    else:
        raise TypeError(
            f"Unsupported dtype: {complex_dtype}. Must be numpy or torch dtype."
        )


def to_complex_dtype_numpy(float_dtype: np.dtype) -> np.dtype:
    if np.issubdtype(float_dtype, np.complexfloating):
        return np.dtype(float_dtype)

    if np.issubdtype(float_dtype, np.floating):
        if np.dtype(float_dtype) == np.dtype("float32"):
            return np.dtype("complex64")
        elif np.dtype(float_dtype) == np.dtype("float64"):
            return np.dtype("complex128")
        else:
            raise ValueError(f"unsupported float dtype: {float_dtype}")
    else:
        raise ValueError(
            f"input dtype is not a (complex) floating dtype: {float_dtype}"
        )


def to_complex_dtype(float_dtype: DType) -> ComplexDType:
    """
    Convert a float dtype to its corresponding complex dtype.
    """
    if isinstance(float_dtype, np.dtype):
        return to_complex_dtype_numpy(float_dtype)
    elif isinstance(float_dtype, torch.dtype):
        return torch.dtype.to_complex(float_dtype)
    else:
        raise TypeError(
            f"Unsupported dtype: {float_dtype}. Must be numpy or torch dtype."
        )


def to_precission(dtype: DType, target_precission: DType | DTypeString) -> DType:
    use_torch = is_torch_dtype(dtype)
    backend = "torch" if use_torch else "numpy"
    normalized_target = validate_dtype(target_precission, backend=backend)
    input_real = to_real_dtype(dtype)
    target_real = to_real_dtype(normalized_target)
    if input_real == target_real:
        return dtype
    is_complex = dtype != input_real
    if is_complex:
        return to_complex_dtype(target_real)
    return target_real
