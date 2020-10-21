"""Module for the CPU-optimization of numerical calculations using numba."""
import numba as nb
import numpy as np
from numba import jit, prange
from typing import Sequence


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def complex_exponential(x):
    """
    Calculate the complex exponential.
    """
    return np.cos(x) + 1.j * np.sin(x)


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def abs2(x):
    """
    Calculate the absolute square of a complex number.
    """
    return x.real ** 2 + x.imag ** 2


@jit(nopython=True, nogil=True, parallel=True)
def interpolate_radial_functions(array: np.ndarray,
                                 rle_encoding: np.ndarray,
                                 disc_indices: np.ndarray,
                                 positions: np.ndarray,
                                 v: np.ndarray,
                                 r: np.ndarray,
                                 dvdr: np.ndarray,
                                 sampling: Sequence[float]):
    """
    Interpolate radial functions in 2d at specified positions. The radial functions are assumed to be spaced evenly on a
    log grid.

    Parameters
    ----------
    array : 2d array of float
        The radial functions will be interpolated in this array.
    disc_indices : 2d array of float
        The relative indices to a central index where the radial functions should be interpolated.
    positions : 2d array of float
        The interpolation positions. In units consistent with the radial distances and sampling.
    v : 2d array of float
        Values of the radial functions. The first dimension indexes the functions, the second dimension indexes the
        values along the radial from the center to the cutoff.
    r : array of float
        The radial distance of the function values. The distances should be spaced evenely on a log grid.
    dvdr : 2d array of float
        The derivative of the radial functions. The first dimension indexes the functions, the second dimension indexes
        the derivatives along the radial from the center to the cutoff.
    sampling : two float
        The sampling rate in x and y.
    """
    n = r.shape[0]
    dt = np.log(r[-1] / r[0]) / (n - 1)

    for p in prange(rle_encoding.shape[0] - 1): # Thread safe loop
        for i in range(rle_encoding[p], rle_encoding[p + 1]):
            px = int(round(positions[i, 0] / sampling[0]))
            py = int(round(positions[i, 1] / sampling[1]))

            for j in range(disc_indices.shape[0]): # Thread safe loop
                k = px + disc_indices[j, 0]
                m = py + disc_indices[j, 1]

                if (k < array.shape[1]) & (m < array.shape[2]) & (k >= 0) & (m >= 0):
                    r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                       (m * sampling[1] - positions[i, 1]) ** 2)

                    idx = int(np.floor(np.log(r_interp / r[0] + 1e-7) / dt))

                    if idx < 0:
                        array[p, k, m] += v[i, 0]
                    elif idx < n - 1:
                        array[p, k, m] += v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx]


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def sum_run_length_encoded(array, result, separators):
    for x in prange(result.shape[1]):
        for i in range(result.shape[0]):
            for j in range(separators[x], separators[x + 1]):
                result[i, x] += array[i, j]
