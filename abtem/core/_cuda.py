import math
import warnings

import cupy as cp  # type: ignore
import numpy as np
from numba import NumbaPerformanceWarning, cuda  # type: ignore


@cuda.jit
def _batch_crop_2d(new_array, array, corners):
    x, y, z = cuda.grid(3)
    if (x < new_array.shape[0]) & (y < new_array.shape[1]) & (z < new_array.shape[2]):
        new_array[x, y, z] = array[x, corners[x, 0] + y, corners[x, 1] + z]


def batch_crop_2d(
    array: cp.ndarray, corners: cp.ndarray, new_shape: tuple[int, int]
) -> cp.ndarray:
    threads_per_block = (1, 32, 32)

    blocks_per_grid_x = int(np.ceil(corners.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(new_shape[0] / threads_per_block[1]))
    blocks_per_grid_z = int(np.ceil(new_shape[1] / threads_per_block[2]))

    blockspergrid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    result = cp.zeros((len(array),) + new_shape, dtype=array.dtype)

    _batch_crop_2d[blockspergrid, threads_per_block](result, array, corners)
    return result


@cuda.jit()
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

    threadsperblock = (1024,)
    blockspergrid = int(np.ceil(result.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        _sum_run_length_encoded[blockspergrid, threadsperblock](
            array, result, separators
        )


_interpolate_radial_raw = cp.RawKernel(r"""
extern "C" __global__ void interpolate_radial(
    float* __restrict__ array,
    const float* __restrict__ positions,   // (n_atoms, 3) row-major
    const int* __restrict__ disk_indices,  // (n_disk, 2)  row-major
    const float sampling_0,
    const float sampling_1,
    const float* __restrict__ radial_gpts, // (n_radial,)
    const float* __restrict__ radial_funcs,// (n_atoms, n_radial) row-major
    const float* __restrict__ radial_deriv,// (n_atoms, n_radial) row-major
    const float dt,
    const float r0,                        // radial_gpts[0]
    const int n_radial,
    const int n_disk,
    const int n_atoms,
    const int rows,
    const int cols
) {
    // x-dim: disk index,  y-dim: atom index
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_atoms || j >= n_disk) return;

    float px = positions[i * 3];
    float py = positions[i * 3 + 1];

    int k = __float2int_rn(px / sampling_0) + disk_indices[j * 2];
    int m = __float2int_rn(py / sampling_1) + disk_indices[j * 2 + 1];

    if (k < 0 || k >= rows || m < 0 || m >= cols) return;

    float dx = k * sampling_0 - px;
    float dy = m * sampling_1 - py;
    float r = sqrtf(dx * dx + dy * dy);

    int idx = __float2int_rd(logf(r / r0 + 1e-12f) / dt);

    float val;
    int base = i * n_radial;
    if (idx < 0) {
        val = radial_funcs[base];
    } else if (idx < n_radial - 1) {
        float slope = radial_deriv[base + idx];
        val = radial_funcs[base + idx] + (r - radial_gpts[idx]) * slope;
    } else {
        return;
    }

    atomicAdd(&array[k * cols + m], val);
}
""", "interpolate_radial")


def interpolate_radial_functions(
    array,
    positions,
    disk_indices,
    sampling,
    radial_gpts,
    radial_functions,
    radial_derivative,
):
    if len(positions) == 0:
        return array

    n_disk = disk_indices.shape[0]
    n_atoms = positions.shape[0]
    n_radial = radial_gpts.shape[0]
    rows, cols = array.shape

    dt = (cp.log(radial_gpts[-1] / radial_gpts[0]) / (n_radial - 1)).item()
    r0 = float(radial_gpts[0].item())

    # x-axis carries disk_indices (can be hundreds of millions → needs
    # gridDim.x limit of 2^31-1); y-axis carries positions (atoms per slice,
    # typically O(100) → well within gridDim.y limit of 65535).
    block = (256, 1, 1)
    grid = (math.ceil(n_disk / 256), n_atoms, 1)

    # Ensure contiguous float32 / int32 arrays.
    positions_f = cp.ascontiguousarray(positions[:, :2].astype(cp.float32))
    # Pack into (n_atoms, 3) with a dummy column so stride matches kernel.
    pos3 = cp.zeros((n_atoms, 3), dtype=cp.float32)
    pos3[:, :2] = positions_f
    disk_i32 = cp.ascontiguousarray(disk_indices.astype(cp.int32))
    rg = cp.ascontiguousarray(radial_gpts.astype(cp.float32))
    rf = cp.ascontiguousarray(radial_functions.astype(cp.float32))
    rd = cp.ascontiguousarray(radial_derivative.astype(cp.float32))

    _interpolate_radial_raw(
        grid, block,
        (array, pos3, disk_i32,
         cp.float32(sampling[0]), cp.float32(sampling[1]),
         rg, rf, rd,
         cp.float32(dt), cp.float32(r0),
         np.int32(n_radial), np.int32(n_disk), np.int32(n_atoms),
         np.int32(rows), np.int32(cols)),
    )


def interpolate_bilinear(x, v, u, vw, uw):
    B, H, W = x.shape
    out_H, out_W = v.shape
    y = cp.empty((B, out_H, out_W), dtype=x.dtype)

    cp.ElementwiseKernel(
        "raw T x, S v, S u, T vw, T uw, S H, S W, S outsize",
        "T y",
        """
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
        """,
        "resize_images_interpolate_bilinear",
    )(x, v, u, vw, uw, H, W, out_H * out_W, y)

    return y
