import cupy as cp
import numpy as np
from numba import cuda


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
