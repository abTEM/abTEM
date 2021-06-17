import numba as nb
import numpy as np
import dask.array as da


@nb.guvectorize([(nb.float32[:, :], nb.complex64[:, :])], '(n,m)->(n,m)')
def _complex_exponential(x, out):
    """
    Calculate the complex exponential.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.cos(x[i, j]) + 1.j * np.sin(x[i, j])


@nb.guvectorize([(nb.complex64[:, :], nb.float32[:, :])], '(n,m)->(n,m)')
def _abs2(x, out):
    """
    Calculate the absolute square of a complex number.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = x[i, j].real ** 2 + x[i, j].imag ** 2


def abs2(x):
    if isinstance(x, np.ndarray):
        return _abs2(x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(_abs2)
