"""Module for the GPU-optimization of numerical calculations using numba, CuPy, and CUDA."""
import math

import cupy as cp
from numba import cuda


@cuda.jit
def _interpolate_radial_functions(array, slice_idx, disc_indices, positions, v, r, dvdr, sampling, dt):
    i, j = cuda.grid(2)
    if (i < positions.shape[0]) & (j < disc_indices.shape[0]):
        k = round(positions[i, 0] / sampling[0]) + disc_indices[j, 0]
        m = round(positions[i, 1] / sampling[1]) + disc_indices[j, 1]

        if (k < array.shape[1]) & (m < array.shape[2]) & (k >= 0) & (m >= 0):
            r_interp = math.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                 (m * sampling[1] - positions[i, 1]) ** 2)

            idx = int(math.log(r_interp / r[0] + 1e-7) / dt)

            if idx < 0:
                cuda.atomic.add(array, (slice_idx[i], k, m), v[i, 0])
            elif idx < r.shape[0] - 1:
                cuda.atomic.add(array, (slice_idx[i], k, m), v[i, idx] + (r_interp - r[idx]) * dvdr[i, idx])


def launch_interpolate_radial_functions(array, rle_encoding, disc_indices, positions, v, r, dvdr, sampling):
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

    slice_idx = cp.zeros((rle_encoding[-1],), dtype=cp.int32)
    for i in range(len(rle_encoding) - 1):
        slice_idx[rle_encoding[i]:rle_encoding[i + 1]] = i

    _interpolate_radial_functions[blockspergrid, threadsperblock](array,
                                                                  slice_idx,
                                                                  disc_indices,
                                                                  positions,
                                                                  v,
                                                                  r,
                                                                  dvdr,
                                                                  sampling,
                                                                  dt)


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


@cuda.jit
def batch_crop(new_array, array, corners):
    x, y, z = cuda.grid(3)
    if (x < new_array.shape[0]) & (y < new_array.shape[1]) & (z < new_array.shape[2]):
        new_array[x, y, z] = array[x, corners[x, 0] + y, corners[x, 1] + z]


def launch_batch_crop(array, corners, new_shape):
    threadsperblock = (1, 32, 32)
    blockspergrid_x = math.ceil(corners.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(new_shape[0] / threadsperblock[1])
    blockspergrid_z = math.ceil(new_shape[1] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    result = cp.zeros((len(array),) + new_shape, dtype=array.dtype)
    batch_crop[blockspergrid, threadsperblock](result, array, corners)

    return result


def interpolate_bilinear_gpu(x, v, u, vw, uw):
    B, H, W = x.shape
    out_H, out_W = v.shape
    y = cp.empty((B, out_H, out_W), dtype=x.dtype)

    cp.ElementwiseKernel(
        'raw T x, S v, S u, T vw, T uw, S H, S W, S outsize', 'T y', '''
        // indices
        S v0 = v;
        S v1 = min(v + 1, (S)(H - 1));
        S u0 = u;
        S u1 = min(u + 1, (S)(W - 1));
        // weights
        T w0 = (1 - vw) * (1 - uw);
        T w1 = (1 - vw) * uw;
        T w2 = vw * (1 - uw);
        T w3 = vw * uw;
        // fetch
        S offset = i / outsize * H * W;
        T px0 = x[offset + v0 * W + u0];
        T px1 = x[offset + v0 * W + u1];
        T px2 = x[offset + v1 * W + u0];
        T px3 = x[offset + v1 * W + u1];
        // interpolate
        y = (w0 * px0 + w1 * px1) + (w2 * px2 + w3 * px3);
        ''', 'resize_images_interpolate_bilinear'
    )(x, v, u, vw, uw, H, W, out_H * out_W, y)
    return y
