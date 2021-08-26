import cupy as cp
import numpy as np
from numba import cuda
import math


@cuda.jit
def _batch_crop_2d(new_array, array, corners):
    x, y, z = cuda.grid(3)
    if (x < new_array.shape[0]) & (y < new_array.shape[1]) & (z < new_array.shape[2]):
        new_array[x, y, z] = array[x, corners[x, 0] + y, corners[x, 1] + z]


def batch_crop_2d(array, corners, new_shape):
    threads_per_block = (1, 32, 32)

    blocks_per_grid_x = int(np.ceil(corners.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(new_shape[0] / threads_per_block[1]))
    blocks_per_grid_z = int(np.ceil(new_shape[1] / threads_per_block[2]))

    blockspergrid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    result = cp.zeros((len(array),) + new_shape, dtype=array.dtype)

    _batch_crop_2d[blockspergrid, threads_per_block](result, array, corners)
    return result


@cuda.jit
def _sum_run_length_encoded(array, result, separators):
    x = cuda.grid(1)
    if x < result.shape[1]:
        for i in range(result.shape[0]):
            for j in range(separators[x], separators[x + 1]):
                result[i, x] += array[i, j]


def sum_run_length_encoded(array, result, separators):
    assert len(array) == len(result)
    assert len(result.shape) == 2
    assert result.shape[1] == len(separators) - 1

    threadsperblock = (256,)
    blockspergrid = int(np.ceil(result.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid,)

    _sum_run_length_encoded[blockspergrid, threadsperblock](array, result, separators)


@cuda.jit
def _interpolate_radial_functions(array,
                                  positions,
                                  disk_indices,
                                  sampling,
                                  radial_gpts,
                                  radial_functions,
                                  radial_derivatives,
                                  dt):
    i, j = cuda.grid(2)

    if (i < positions.shape[0]) & (j < disk_indices.shape[0]):
        k = round(positions[i, 0] / sampling[0]) + disk_indices[j, 0]
        m = round(positions[i, 1] / sampling[1]) + disk_indices[j, 1]

        if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
            r_interp = math.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                 (m * sampling[1] - positions[i, 1]) ** 2)

            idx = int(math.log(r_interp / radial_gpts[0] + 1e-12) / dt)

            if idx < 0:
                cuda.atomic.add(array, (k, m), radial_functions[i, 0])

            elif idx < radial_gpts.shape[0] - 1:
                slope = radial_derivatives[i, idx]
                cuda.atomic.add(array, (k, m), radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope)


def interpolate_radial_functions(array, positions, disk_indices, sampling, radial_gpts, radial_functions,
                                 radial_derivative):
    threadsperblock = (1, 256)
    blockspergrid_x = int(math.ceil(positions.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(disk_indices.shape[0] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    dt = (cp.log(radial_gpts[-1] / radial_gpts[0]) / (radial_gpts.shape[0] - 1)).item()

    _interpolate_radial_functions[blockspergrid, threadsperblock](array,
                                                                  positions,
                                                                  disk_indices,
                                                                  sampling,
                                                                  radial_gpts,
                                                                  radial_functions,
                                                                  radial_derivative,
                                                                  dt)
