import cupy as cp
import dask.array as da
import numba as nb
import numpy as np

from abtem.basic.backend import check_cupy_is_installed


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def _complex_exponential(x):
    """
    Calculate the complex exponential.
    """
    return np.cos(x) + 1.j * np.sin(x)


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def _abs2(x):
    """
    Calculate the absolute square of a complex number.
    """
    return x.real ** 2 + x.imag ** 2


def abs2(x, **kwargs):
    if isinstance(x, np.ndarray):
        return _abs2(x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(_abs2, **kwargs)

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.abs(x) ** 2

    raise ValueError()


def complex_exponential(x, **kwargs):
    if isinstance(x, np.ndarray):
        return _complex_exponential(x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(_complex_exponential)

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.exp(1.j * x)

    raise ValueError()
