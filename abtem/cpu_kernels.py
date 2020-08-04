import numba as nb
import numpy as np
from numba import jit, prange


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def complex_exponential(x):
    return np.cos(x) + 1.j * np.sin(x)


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


@jit(nopython=True, nogil=True, parallel=True)
def interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling):
    n = r.shape[0]
    dt = np.log(r[-1] / r[0]) / (n - 1)
    for i in range(positions.shape[0]):
        for j in prange(disc_indices.shape[0]):
            k = int(round(positions[i, 0] / sampling[0]) + disc_indices[j, 0])
            l = int(round(positions[i, 1] / sampling[1]) + disc_indices[j, 1])

            if ((k < array.shape[0]) & (l < array.shape[1]) & (k >= 0) & (l >= 0)):
                r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                   (l * sampling[1] - positions[i, 1]) ** 2)

                idx = int(np.floor(np.log(r_interp / r[0] + 1e-7) / dt))
                # idx = np.searchsorted(r, r_interp)
                if (idx < 0):
                    array[k, l] += v[i, 0]
                elif (idx < n - 1):
                    array[k, l] += v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx]


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def windowed_scale_reduce(probes: np.ndarray, S: np.ndarray, corners, coefficients):
    """
    Function for collapsing a PRISM scattering matrix into a probe wave function.

    Parameters
    ----------
    probes : 3d numpy.ndarray
        The array in which the probe wave functions should be written.
    S : 3d numpy.ndarray
        Scattering matrix.
    corners :
    coefficients :
    """

    for k in prange(probes.shape[0]):
        for i in prange(probes.shape[1]):
            ii = (corners[k, 0] + i) % S.shape[1]
            for j in prange(probes.shape[2]):
                jj = (corners[k, 1] + j) % S.shape[2]
                # for l in prange(coefficients.shape[1]):
                #    probes[k, i, j] += (coefficients[k][l] * S[l, ii, jj])
                probes[k, i, j] = (coefficients[k] * S[:, ii, jj]).sum()


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def scale_reduce(probes: np.ndarray, S: np.ndarray, coefficients):
    """
    Function for collapsing a PRISM scattering matrix into a probe wave function.

    Parameters
    ----------
    probes : 3d numpy.ndarray
        The array in which the probe wave functions should be written.
    S : 3d numpy.ndarray
        Scattering matrix.
    corners :
    coefficients :
    """
    for i in prange(S.shape[1]):
        for j in prange(S.shape[2]):
            for l in range(S.shape[0]):
                # s = S[l, i, j]
                for k in prange(probes.shape[0]):
                    probes[k, i, j] += (coefficients[k, l] * S[l, i, j])  # .sum()
