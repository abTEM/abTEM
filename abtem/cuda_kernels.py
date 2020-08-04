import math

import cupy as cp
import numpy as np
from numba import cuda, complex64


@cuda.jit
def superpose_deltas(array: np.ndarray, positions, indices):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < indices.shape[0]:
        row, col = indices[i]
        cuda.atomic.add(array, (row, col), (positions[i, 0] - row) * (positions[i, 1] - col))
        cuda.atomic.add(array, (row + 1, col), (row + 1 - positions[i, 0]) * (positions[i, 1] - col))
        cuda.atomic.add(array, (row, col + 1), (positions[i, 0] - row) * (col + 1 - positions[i, 1]))
        cuda.atomic.add(array, (row + 1, col + 1), (row + 1 - positions[i, 0]) * (col + 1 - positions[i, 1]))


def launch_superpose_deltas(positions, shape):
    array = cp.zeros((shape[0], shape[1]), dtype=cp.float32)

    if len(positions) == 0:
        return array

    if (cp.any(positions[:, 0] < 0) or cp.any(positions[:, 1] < 0) or cp.any(positions[:, 0] > shape[0]) or cp.any(
            positions[:, 1] > shape[1])):
        raise RuntimeError()

    rounded = cp.floor(positions).astype(cp.int32)

    threadsperblock = 32
    blockspergrid = (positions.shape[0] + (threadsperblock - 1)) // threadsperblock
    superpose_deltas[blockspergrid, threadsperblock](array, positions, rounded)
    return array


@cuda.jit
def _interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling, dt):
    i, j = cuda.grid(2)
    if (i < positions.shape[0]) & (j < disc_indices.shape[0]):
        k = round(positions[i, 0] / sampling[0]) + disc_indices[j, 0]
        l = round(positions[i, 1] / sampling[1]) + disc_indices[j, 1]
        if ((k < array.shape[0]) & (l < array.shape[1]) & (k >= 0) & (l >= 0)):
            r_interp = math.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                 (l * sampling[1] - positions[i, 1]) ** 2)

            idx = int(math.log(r_interp / r[0] + 1e-7) / dt)

            if idx < 0:
                cuda.atomic.add(array, (k, l), v[i, 0])
            elif idx < r.shape[0] - 1:
                cuda.atomic.add(array, (k, l), v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx])


def launch_interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling):
    threadsperblock = (1, 256)
    blockspergrid_x = math.ceil(positions.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(disc_indices.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    dt = (cp.log(r[-1] / r[0]) / (r.shape[0] - 1)).item()
    #print(type(dt))
    #sss
    _interpolate_radial_functions[blockspergrid, threadsperblock](array,
                                                                  disc_indices,
                                                                  positions,
                                                                  v,
                                                                  r,
                                                                  dvdr,
                                                                  sampling,
                                                                  dt)


@cuda.jit
def _scale_reduce(probes, S, coefficients):
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
    x, y, z = cuda.grid(3)
    if (x < S.shape[1]) & (y < S.shape[2]) & (z < probes.shape[0]):
        # for i in range(coefficients.shape[0]):
        # tmp = 0.
        for k in range(S.shape[0]):
            probes[z, x, y] += coefficients[z, k] * S[k, x, y]

        # probes[z, x, y] = tmp


def launch_scale_reduce(probes, S, coefficients):
    threadsperblock = (4, 4, 16)
    blockspergrid_x = math.ceil(S.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(S.shape[2] / threadsperblock[1])
    blockspergrid_z = math.ceil(probes.shape[0] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _scale_reduce[blockspergrid, threadsperblock](probes, S, coefficients)


@cuda.jit
def _windowed_scale_reduce(probes, S, corners, coefficients):
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
    x, y, z = cuda.grid(3)

    # sA = cuda.shared.array(shape=(8, 8, 8), dtype=complex64)

    if (x < probes.shape[1]) & (y < probes.shape[2]) & (z < probes.shape[0]):
        xx = (corners[z, 0] + x) % S.shape[1]
        yy = (corners[z, 1] + y) % S.shape[2]

        for k in range(S.shape[0]):
            probes[z, x, y] += coefficients[z, k] * S[k, xx, yy]


def launch_windowed_scale_reduce(probes, S, corners, coefficients):
    threadsperblock = (4, 4, 16)
    blockspergrid_x = math.ceil(S.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(S.shape[2] / threadsperblock[1])
    blockspergrid_z = math.ceil(probes.shape[0] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _windowed_scale_reduce[blockspergrid, threadsperblock](probes, S, corners, coefficients)
