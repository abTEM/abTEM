"""Module for the GPU-optimization of numerical calculations using numba, CuPy, and CUDA."""
import math

import cupy as cp
import numpy as np
from numba import cuda


@cuda.jit
def _superpose_deltas(array: np.ndarray, positions, indices):
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
    _superpose_deltas[blockspergrid, threadsperblock](array, positions, rounded)
    return array


@cuda.jit
def _interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling, dt):
    i, j = cuda.grid(2)
    if (i < positions.shape[0]) & (j < disc_indices.shape[0]):
        k = round(positions[i, 0] / sampling[0]) + disc_indices[j, 0]
        m = round(positions[i, 1] / sampling[1]) + disc_indices[j, 1]
        if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
            r_interp = math.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                 (m * sampling[1] - positions[i, 1]) ** 2)

            idx = int(math.log(r_interp / r[0] + 1e-7) / dt)

            if idx < 0:
                cuda.atomic.add(array, (k, m), v[i, 0])
            elif idx < r.shape[0] - 1:
                cuda.atomic.add(array, (k, m), v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx])


def launch_interpolate_radial_functions(array, disc_indices, positions, v, r, dvdr, sampling):
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
        The radial distance of the function values. The distances should be spaced evenly on a log grid.
    dvdr : 2d array of float
        The derivative of the radial functions. The first dimension indexes the functions, the second dimension indexes
        the derivatives along the radial from the center to the cutoff.
    sampling : two float
        The sampling rate in x and y [1 / Å].
    """

    threadsperblock = (1, 256)
    blockspergrid_x = math.ceil(positions.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(disc_indices.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    dt = (cp.log(r[-1] / r[0]) / (r.shape[0] - 1)).item()

    _interpolate_radial_functions[blockspergrid, threadsperblock](array,
                                                                  disc_indices,
                                                                  positions,
                                                                  v,
                                                                  r,
                                                                  dvdr,
                                                                  sampling,
                                                                  dt)


@cuda.jit
def _windowed_scale_reduce(probes: cp.ndarray, S: cp.ndarray, corners: cp.ndarray, coefficients: cp.ndarray):
    # '3' because threads on a GPU are laid out in a 3D grid. Note that x, y, z here are not Cartesian directions!
    x, y, z = cuda.grid(3)

    if (x < probes.shape[1]) & (y < probes.shape[2]) & (z < probes.shape[0]):
        xx = (corners[z, 0] + x)
        if xx < 0:
            xx += S.shape[1]
        elif xx >= S.shape[1]:
            xx -= S.shape[1]

        yy = (corners[z, 1] + y)
        if yy < 0:
            yy += S.shape[2]
        elif yy >= S.shape[2]:
            yy -= S.shape[2]
        # xx = (corners[z, 0] + x) % S.shape[1]
        # yy = (corners[z, 1] + y) % S.shape[2]

        for k in range(S.shape[0]):
            probes[z, x, y] += coefficients[z, k] * S[k, xx, yy]


def launch_windowed_scale_reduce(probes: cp.ndarray, S: cp.ndarray, corners: cp.ndarray, coefficients: cp.ndarray):
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

    # TODO : improve threadsperblock

    threadsperblock = (4, 4, 16)
    blockspergrid_x = math.ceil(S.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(S.shape[2] / threadsperblock[1])
    blockspergrid_z = math.ceil(probes.shape[0] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _windowed_scale_reduce[blockspergrid, threadsperblock](probes, S, corners, coefficients)


@cuda.jit
def _scale_reduce(probes: cp.ndarray, S: cp.ndarray, coefficients: cp.ndarray):
    x, y, z = cuda.grid(3)
    if (x < S.shape[1]) & (y < S.shape[2]) & (z < probes.shape[0]):
        for k in range(S.shape[0]):
            probes[z, x, y] += coefficients[z, k] * S[k, x, y]


def launch_scale_reduce(probes: cp.ndarray, S: cp.ndarray, coefficients: cp.ndarray):
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

    # TODO : improve threadsperblock

    threadsperblock = (8, 8, 1)
    blockspergrid_x = math.ceil(S.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(S.shape[2] / threadsperblock[1])
    blockspergrid_z = math.ceil(probes.shape[0] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _scale_reduce[blockspergrid, threadsperblock](probes, S, coefficients)


@cuda.jit
def sum_run_length_encoded(array, result, separators):
    x = cuda.grid(1)
    if x < result.shape[1]:
        for i in range(result.shape[0]):
            for j in range(separators[x], separators[x + 1]):
                result[i, x] += array[i, j]


def launch_sum_run_length_encoded(array, result, separators):
    assert len(array) == len(result)
    assert len(result.shape) == 2
    assert result.shape[1] == len(separators) - 1

    threadsperblock = (256,)
    blockspergrid = math.ceil(result.shape[1] / threadsperblock[0])
    blockspergrid = (blockspergrid,)
    sum_run_length_encoded[blockspergrid, threadsperblock](array, result, separators)