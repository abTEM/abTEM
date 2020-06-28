import math

import cupy as cp
import numpy as np
from numba import cuda


@cuda.jit
def superpose_deltas_kernel(array: np.ndarray, positions, indices):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < indices.shape[0]:
        row, col = indices[i]
        cuda.atomic.add(array, (row, col), (positions[i, 0] - row) * (positions[i, 1] - col))
        cuda.atomic.add(array, (row + 1, col), (row + 1 - positions[i, 0]) * (positions[i, 1] - col))
        cuda.atomic.add(array, (row, col + 1), (positions[i, 0] - row) * (col + 1 - positions[i, 1]))
        cuda.atomic.add(array, (row + 1, col + 1), (row + 1 - positions[i, 0]) * (col + 1 - positions[i, 1]))


def superpose_deltas(positions, shape):
    array = cp.zeros((shape[0], shape[1]), dtype=cp.float32)

    if len(positions) == 0:
        return array

    if (cp.any(positions[:, 0] < 0) or cp.any(positions[:, 1] < 0) or cp.any(positions[:, 0] > shape[0]) or cp.any(
            positions[:, 1] > shape[1])):
        raise RuntimeError()

    rounded = cp.floor(positions).astype(cp.int32)

    threadsperblock = 32
    blockspergrid = (positions.shape[0] + (threadsperblock - 1)) // threadsperblock
    superpose_deltas_kernel[blockspergrid, threadsperblock](array, positions, rounded)
    return array


@cuda.jit
def _interpolate_radial_functions(array, array_rows, array_cols, indices, disc_indices, positions, v, r, dvdr,
                                  ):
    n = r.shape[0]
    dt = math.log(r[-1] / r[0]) / (n - 1)

    x, y = cuda.grid(2)
    if x < indices.shape[0]:
        if y < disc_indices.shape[0]:
            k = indices[x] + disc_indices[y]
            if k < array.shape[0]:
                r_interp = math.sqrt((array_rows[k] - positions[x, 0]) ** 2 +
                                     (array_cols[k] - positions[x, 1]) ** 2)

                idx = int(min(max(math.floor(math.log(r_interp / r[0] + 1e-7) / dt), 0), n - 1))

                if idx < dvdr.shape[1] - 1:
                    new = v[x, idx] + (r_interp - r[idx]) * dvdr[x, idx]
                    cuda.atomic.add(array, k, new)


def launch_interpolate_radial_functions(array, array_rows, array_cols, indices, disc_indices, positions, v, r, dvdr,
                                        ):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(indices.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(disc_indices.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _interpolate_radial_functions[blockspergrid, threadsperblock](array, array_rows, array_cols, indices, disc_indices,
                                                                  positions, v, r, dvdr)

# def interpolate_radial_functions(func, positions, shape, cutoff, inner_cutoff=0.):
#     xp = cp.get_array_module(positions)
#
#     n = xp.int(xp.ceil(cutoff - inner_cutoff))
#     r = xp.linspace(inner_cutoff, cutoff, 2 * n)
#     values = func(r)
#
#     margin = xp.int(xp.ceil(r[-1]))
#
#     padded_shape = (shape[0] + 2 * margin, shape[1] + 2 * margin)
#     array = xp.zeros(padded_shape[0] * padded_shape[1], dtype=cp.float32)
#
#     derivatives = xp.diff(values) / xp.diff(r)
#
#     positions = positions + margin
#     indices = xp.rint(positions).astype(xp.int)[:, 0] * padded_shape[0] + xp.rint(positions).astype(xp.int)[:, 1]
#     indices = (indices[:, None] + xp.asarray(coordinates_in_disc(margin - 1, padded_shape))[None])  # .ravel()
#
#     rows, cols = xp.indices(padded_shape)
#     rows = rows.ravel()
#     cols = cols.ravel()
#
#     if xp is cp:
#         indices = indices.ravel()
#         threadsperblock = 32
#         blockspergrid = (indices.size + (threadsperblock - 1)) // threadsperblock
#         interpolate_radial_functions_cuda_kernel[blockspergrid, threadsperblock](array, indices, positions, rows, cols,
#                                                                                  r, values, derivatives)
#     else:
#         interpolate_radial_functions_kernel(array, indices, positions, rows, cols, r, values, derivatives)
#
#     array = array.reshape(padded_shape)
#     array = array[margin:-margin, margin:-margin]
#
#     return array
