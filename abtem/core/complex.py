import dask.array as da
import numba as nb  # type: ignore
import numpy as np

from abtem.core.backend import check_cupy_is_installed, cp


@nb.vectorize(
    [nb.complex64(nb.float32), nb.complex128(nb.float64), nb.complex64(nb.complex64)]
)
def _complex_exponential(x):
    """
    Calculate the complex exponential.
    """
    return np.cos(x) + 1.0j * np.sin(x)


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def _abs2(x):
    """
    Calculate the absolute square of a complex number.
    """
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


def abs2(x: np.ndarray | da.core.Array) -> np.ndarray | da.core.Array:
    """
    Fast calculation of the absolute square of a complex number.
    """
    if isinstance(x, np.ndarray):
        return _abs2(x)

    if isinstance(x, da.core.Array):
        return da.map_blocks(abs2, x)

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        if x.dtype == "complex64":
            return _abs2_cupy_float32(x.real, x.imag)
        elif x.dtype == "complex128":
            return _abs2_cupy_float64(x.real, x.imag)
        else:
            raise RuntimeError("Unsupported dtype for calculation of abs2 with cupy array.")

    raise ValueError(
        "abs2 only supports numpy arrays, dask arrays and cupy arrays."
    )


def complex_exponential(x: np.ndarray | da.core.Array) -> np.ndarray | da.core.Array:
    """
    Fast calculation of the complex exponential.
    """

    if isinstance(x, np.ndarray):
        return _complex_exponential(x)

    if isinstance(x, da.core.Array):
        return da.map_blocks(complex_exponential, x)

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.exp(1.0j * x)

    raise ValueError(
        "complex_exponential only supports numpy arrays, dask arrays and cupy arrays."
    )
