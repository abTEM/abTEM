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
    for i in range(positions.shape[0]):
        for j in prange(disc_indices.shape[0]):
            k = int(round(positions[i, 0] / sampling[0]) + disc_indices[j, 0])
            m = int(round(positions[i, 1] / sampling[1]) + disc_indices[j, 1])

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                   (m * sampling[1] - positions[i, 1]) ** 2)

                idx = int(np.floor(np.log(r_interp / r[0] + 1e-7) / dt))
                if idx < 0:
                    array[k, m] += v[i, 0]
                elif idx < n - 1:
                    array[k, m] += v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx]


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def windowed_scale_reduce(probes: np.ndarray, S: np.ndarray, corners: np.ndarray, coefficients: np.ndarray):
    """
    Collapse a PRISM scattering matrix into probe wave functions. The probes are cropped around their center to the size
    of the given probes array.

    Parameters
    ----------
    probes : 3d array
        The array in which the probe wave functions should be written. The first dimension indexes the probe batch,
        the last two dimensions indexes the spatial dimensions.
    S : 3d array
        The compact scattering matrix. The first dimension indexes the plane waves, the last two dimensions indexes the
        spatial dimensions.
    corners : 2d array of int
        The corners of the probe windows. The first dimension indexes the probe batch, the two components of the second
        dimension are the first and second index of the spatial dimension.
    coefficients : 2d array of complex
        The coefficients of the plane wave expansion of a probe at a specific position. The first dimension indexes the
        probe batch, the second dimension indexes the coefficients corresponding to the plane waves of the scattering
        matrix.
    """

    for k in prange(probes.shape[0]):
        for i in prange(probes.shape[1]):
            ii = (corners[k, 0] + i) % S.shape[1]
            for j in prange(probes.shape[2]):
                jj = (corners[k, 1] + j) % S.shape[2]
                probes[k, i, j] = (coefficients[k] * S[:, ii, jj]).sum()


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def scale_reduce(probes: np.ndarray, S: np.ndarray, coefficients: np.ndarray):
    """
    Collapse a PRISM scattering matrix into probe wave functions.

    Parameters
    ----------
    probes : 3d array
        The array in which the probe wave functions should be written. The first dimension indexes the probe batch,
        the last two dimensions indexes the spatial dimensions.
    S : 3d array
        The compact scattering matrix. The first dimension indexes the plane waves, the last two dimensions indexes the
        spatial dimensions.
    coefficients : 2d array of complex
        The coefficients of the plane wave expansion of a probe at a specific position. The first dimension indexes the
        probe batch, the second dimension indexes the coefficients corresponding to the plane waves of the scattering
        matrix.
    """
    for i in prange(S.shape[1]):
        for j in range(S.shape[2]):
            for m in range(S.shape[0]):
                for n in range(probes.shape[0]):
                    probes[n, i, j] += (coefficients[n, m] * S[m, i, j])


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def sum_run_length_encoded(array, result, separators):
    for x in prange(result.shape[1]):
        for i in range(result.shape[0]):
            for j in range(separators[x], separators[x + 1]):
                result[i, x] += array[i, j]
