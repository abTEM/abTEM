import dask.array as da
import numba as nb
import numpy as np

from abtem.core.backend import check_cupy_is_installed, cp
from abtem.core.config import config


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
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
        in_params=f"float32 x, float32 y",
        out_params=f"float32 z",
        operation="z = x * x + y * y",
        name="abs_squared",
    )
    _abs2_cupy_float64 = cp.ElementwiseKernel(
        in_params=f"float64 x, float64 y",
        out_params=f"float64 z",
        operation="z = x * x + y * y",
        name="abs_squared",
    )
else:
    _abs2_cupy = None


def abs2(x, **kwargs):
    if isinstance(x, np.ndarray):
        return _abs2(x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(abs2, **kwargs)

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        if x.dtype == "complex64":
            return _abs2_cupy_float32(x.real, x.imag)
        elif x.dtype == "complex128":
            return _abs2_cupy_float64(x.real, x.imag)
        else:
            raise RuntimeError("")

    raise ValueError()


def complex_exponential(x):
    if isinstance(x, np.ndarray):
        return _complex_exponential(x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(complex_exponential)

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.exp(1.0j * x)

    raise ValueError()
