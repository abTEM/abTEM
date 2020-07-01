import math

import cupy as cp
import numpy as np
from numba import cuda


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
def _interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling):
    n = r.shape[0]
    dt = math.log(r[-1] / r[0]) / (n - 1)

    i, j = cuda.grid(2)
    if i < positions.shape[0]:
        if j < disc_indices.shape[0]:
            k = round(positions[i, 0] / sampling[0]) + disc_indices[j, 0]
            l = round(positions[i, 1] / sampling[1]) + disc_indices[j, 1]
            if ((k < array.shape[0]) & (l < array.shape[1]) & (k >= 0) & (l >= 0)):
                r_interp = math.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                     (l * sampling[1] - positions[i, 1]) ** 2)

                idx = int(math.floor(math.log(r_interp / r[0] + 1e-7) / dt))

                if idx < 0:
                    cuda.atomic.add(array, (k, l), v[i, 0])
                elif idx < n - 1:
                    new = v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx]
                    cuda.atomic.add(array, (k, l), new)


def launch_interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling):
    threadsperblock = (16, 32)
    blockspergrid_x = math.ceil(positions.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(disc_indices.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _interpolate_radial_functions[blockspergrid, threadsperblock](array,
                                                                  disc_indices,
                                                                  positions,
                                                                  v,
                                                                  r,
                                                                  dvdr,
                                                                  sampling)


# @cuda.jit
# def _window_and_collapse(probes, S, corners, coefficients):
#     """
#     Function for collapsing a Prism scattering matrix into a probe wave function.
#
#     Parameters
#     ----------
#     probes : 3d numpy.ndarray
#         The array in which the probe wave functions should be written.
#     S : 3d numpy.ndarray
#         Scattering matrix
#     corners :
#     coefficients :
#     """
#     N, M = S.shape[1:]
#     L, n, m = probes.shape
#     # L = coefficients.shape[1]
#     i = cuda.grid(1)
#
#     if i < S.shape[0]:
#         # for k in range(probes.shape[0]):
#         for l in range(L):
#             C = coefficients[i][l]
#             for j in range(n):
#                 for k in range(m):
#                     cx, cy = corners[i]
#                     jj = (cx + j) % N
#                     kk = (cy + k) % M
#                     probes[l, j, k] += S[i, jj, kk] * C

@cuda.jit
def _window_and_collapse(probes, S, corners, coefficients):
    """
    Function for collapsing a Prism scattering matrix into a probe wave function.

    Parameters
    ----------
    probes : 3d numpy.ndarray
        The array in which the probe wave functions should be written.
    S : 3d numpy.ndarray
        Scattering matrix
    corners :
    coefficients :
    """
    x, y, z = cuda.grid(3)
    if (x < probes.shape[0]) & (y < probes.shape[1]) & (z < probes.shape[2]):
        yy = (corners[x, 0] + y) % S.shape[1]
        zz = (corners[x, 1] + z) % S.shape[2]
        for l in range(coefficients.shape[1]):
            probes[x, y, z] += coefficients[x, l] * S[l, yy, zz]


def launch_window_and_collapse(probes, S, corners, coefficients):
    threadsperblock = (2, 8, 8)
    blockspergrid_x = math.ceil(probes.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(probes.shape[1] / threadsperblock[1])
    blockspergrid_z = math.ceil(probes.shape[2] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _window_and_collapse[blockspergrid, threadsperblock](probes, S, corners, coefficients)
