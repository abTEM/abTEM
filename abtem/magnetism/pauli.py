from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numba  # type: ignore
import numpy as np
from numba import cuda, njit, prange

from abtem.core.backend import get_array_module

# 1D first-derivative centered stencils
first_derivative_fd_coefficients = {
    2: [-0.5, 0.0, 0.5],
    4: [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12],
    6: [-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60],
    8: [
        1 / 280,
        -4 / 105,
        1 / 5,
        -4 / 5,
        0.0,
        4 / 5,
        -1 / 5,
        4 / 105,
        -1 / 280,
    ],
}


def _launch_3d(kernel, shape, *args):
    """Launch a CUDA kernel over a (batch, x, y) grid of `shape`."""
    threads = (4, 8, 8)
    blocks = tuple(math.ceil(size / t) for size, t in zip(shape, threads))
    kernel[blocks, threads](*args)


@lru_cache(maxsize=None)
def _a_dot_gradient_stencil(
    accuracy: int,
    sampling: tuple[float, float],
    dtype=np.complex64,
    device: str = "cpu",
) -> Callable:
    """
    Build (and cache) a fused stencil function computing
    A_x * d/dx + A_y * d/dy of a wave-function array with periodic (wrap)
    boundaries.

    The returned function takes (waves, A_x, A_y, A_x_uniform, A_y_uniform)
    where waves has shape (..., nx, ny) -- axis -2 is x and axis -1 is y,
    following the abTEM array convention -- and A_x, A_y have shape (nx, ny).
    Neighbours across the boundary are found by wrapping the index, so
    neither the waves nor the fields are padded. Its `series_term` attribute
    computes one Taylor-series term of exp(scale * A.grad), adds it to an
    accumulator and sums its absolute values in the same pass.

    Cached on (accuracy, sampling, dtype, device): a fresh `@njit` closure
    forces numba to recompile from scratch even when functionally identical
    to one already compiled, and `ADotGradientOperator` is instantiated
    anew inside every `pauli_multislice_and_detect` call -- once per lazy
    dask chunk on the scanned path -- so without this cache, every chunk
    would pay a full JIT compile (order of a second) independently.
    """
    if accuracy not in first_derivative_fd_coefficients:
        raise ValueError(
            f"derivative_accuracy must be one of "
            f"{sorted(first_derivative_fd_coefficients)}, got {accuracy}"
        )

    c = np.array(first_derivative_fd_coefficients[accuracy])
    cx = (c / sampling[0]).astype(dtype)
    cy = (c / sampling[1]).astype(dtype)
    cx = np.roll(cx, -(len(cx) // 2))
    cy = np.roll(cy, -(len(cy) // 2))
    n = len(c) // 2

    # `Ax_u` / `Ay_u` carry a per-ensemble-member *uniform* contribution to
    # A_x / A_y, used by the probe-centred gauge (see `pauli_multislice`'s
    # `gauge_origin`). For an axial B_avg the uniform vector potential
    # A_np = 1/2 B_avg x (r - r0) is separable: A_x depends only on y (and on
    # the member's own gauge origin) and A_y only on x, so a per-member
    # (B, W) / (B, H) pair replaces what would otherwise be a full (B, H, W)
    # field per member -- hundreds of MB at production grid sizes. The
    # leading axis is indexed modulo its length, so passing length-1 zeros
    # disables the term with no branching in the inner loop.

    @njit(inline="always", fastmath=True)
    def _point_cpu(a, m, i, j, ax, ay):
        _, H, W = a.shape
        cumul = dtype(0.0)
        for k in range(-n, n + 1):
            ik = i + k
            if ik < 0:
                ik += H
            elif ik >= H:
                ik -= H
            jk = j + k
            if jk < 0:
                jk += W
            elif jk >= W:
                jk -= W
            cumul += ax * cx[k] * a[m, ik, j]
            cumul += ay * cy[k] * a[m, i, jk]
        return cumul

    # One pass over the rows of every member: rows rather than members are
    # the parallel unit, so a single probe position still uses every core.
    @njit(parallel=True, fastmath=True)
    def _stencil_cpu(a, Ax, Ay, Ax_u, Ay_u, out):
        M, H, W = a.shape
        Bu = Ax_u.shape[0]
        for r in prange(M * H):
            m = r // H
            i = r % H
            bu = m % Bu
            for j in range(W):
                ax = Ax[i, j] + Ax_u[bu, j]
                ay = Ay[i, j] + Ay_u[bu, i]
                out[m, i, j] = _point_cpu(a, m, i, j, ax, ay)

    @njit(parallel=True, fastmath=True)
    def _series_term_cpu(
        a, Ax, Ay, Ax_u, Ay_u, scale, out, accumulator, accumulate, row_amplitude
    ):
        M, H, W = a.shape
        Bu = Ax_u.shape[0]
        for r in prange(M * H):
            m = r // H
            i = r % H
            bu = m % Bu
            amplitude = 0.0
            for j in range(W):
                ax = Ax[i, j] + Ax_u[bu, j]
                ay = Ay[i, j] + Ay_u[bu, i]
                value = scale * _point_cpu(a, m, i, j, ax, ay)
                out[m, i, j] = value
                if accumulate:
                    accumulator[m, i, j] += value
                amplitude += abs(value)
            row_amplitude[r] = amplitude

    @cuda.jit(device=True)
    def _point_gpu(a, m, i, j, ax, ay):
        _, H, W = a.shape
        cumul = dtype(0.0)
        for k in range(-n, n + 1):
            ik = i + k
            if ik < 0:
                ik += H
            elif ik >= H:
                ik -= H
            jk = j + k
            if jk < 0:
                jk += W
            elif jk >= W:
                jk -= W
            cumul += ax * cx[k] * a[m, ik, j]
            cumul += ay * cy[k] * a[m, i, jk]
        return cumul

    @cuda.jit
    def _stencil_gpu(a, Ax, Ay, Ax_u, Ay_u, out):
        m, i, j = cuda.grid(3)
        M, H, W = a.shape
        if m < M and i < H and j < W:
            bu = m % Ax_u.shape[0]
            ax = Ax[i, j] + Ax_u[bu, j]
            ay = Ay[i, j] + Ay_u[bu, i]
            out[m, i, j] = _point_gpu(a, m, i, j, ax, ay)

    @cuda.jit
    def _series_term_gpu(a, Ax, Ay, Ax_u, Ay_u, scale, out, accumulator, accumulate):
        m, i, j = cuda.grid(3)
        M, H, W = a.shape
        if m < M and i < H and j < W:
            bu = m % Ax_u.shape[0]
            ax = Ax[i, j] + Ax_u[bu, j]
            ay = Ay[i, j] + Ay_u[bu, i]
            value = scale[0] * _point_gpu(a, m, i, j, ax, ay)
            out[m, i, j] = value
            if accumulate:
                accumulator[m, i, j] += value

    def zero_uniform(xp, n_points):
        """A disabled uniform term: one row of zeros, broadcast by the
        modulo indexing in the kernels."""
        return xp.zeros((1, n_points), dtype=dtype)

    def _prepare(a, Ax_uniform, Ay_uniform):
        if a.shape[-2] <= n or a.shape[-1] <= n:
            raise ValueError(
                f"the grid {a.shape[-2:]} is too small for a derivative stencil "
                f"reaching {n} points"
            )
        xp = get_array_module(a)
        a = xp.ascontiguousarray(a.reshape(-1, *a.shape[-2:]))
        if Ax_uniform is None:
            Ax_uniform = zero_uniform(xp, a.shape[2])
        if Ay_uniform is None:
            Ay_uniform = zero_uniform(xp, a.shape[1])
        return xp, a, Ax_uniform, Ay_uniform

    def _stencil(a, Ax, Ay, Ax_uniform=None, Ay_uniform=None):
        original_shape = a.shape
        xp, a, Ax_uniform, Ay_uniform = _prepare(a, Ax_uniform, Ay_uniform)
        out = xp.empty_like(a)

        if device == "cpu":
            _stencil_cpu(a, Ax, Ay, Ax_uniform, Ay_uniform, out)
        elif device == "gpu":
            _launch_3d(_stencil_gpu, a.shape, a, Ax, Ay, Ax_uniform, Ay_uniform, out)
        else:
            raise ValueError(f"Unsupported device: {device}")

        return out.reshape(original_shape)

    def series_term(
        a, Ax, Ay, Ax_uniform, Ay_uniform, scale, accumulator=None, out=None
    ):
        """
        One Taylor-series term, `scale * (A_x d/dx + A_y d/dy) a`, and the sum
        of its absolute values. When `accumulator` is given (a C-contiguous
        array of the shape of `a`, not `a` itself), the term is also added to
        it in the same pass. `out` may be passed to reuse a buffer of the
        shape of `a`.
        """
        original_shape = a.shape
        xp, a, Ax_uniform, Ay_uniform = _prepare(a, Ax_uniform, Ay_uniform)
        if out is None:
            out = xp.empty_like(a)
        out = out.reshape(a.shape)

        accumulate = accumulator is not None
        if accumulate:
            if not accumulator.flags.c_contiguous:
                raise ValueError("the accumulator must be C-contiguous")
            accumulator = accumulator.reshape(a.shape)
        else:
            accumulator = out

        scale = dtype(scale)
        if device == "cpu":
            row_amplitude = np.empty(a.shape[0] * a.shape[1], dtype=a.real.dtype)
            _series_term_cpu(
                a, Ax, Ay, Ax_uniform, Ay_uniform, scale, out, accumulator,
                accumulate, row_amplitude,
            )
            amplitude = row_amplitude.sum()
        elif device == "gpu":
            _launch_3d(
                _series_term_gpu, a.shape, a, Ax, Ay, Ax_uniform, Ay_uniform,
                xp.asarray([scale], dtype=dtype), out, accumulator, accumulate,
            )
            amplitude = xp.abs(out).sum()
        else:
            raise ValueError(f"Unsupported device: {device}")

        return out.reshape(original_shape), float(amplitude)

    _stencil.series_term = series_term
    _stencil.zero_uniform = zero_uniform
    # sum |c_k| per direction: the L1 norm of A.grad is at most
    # max|A_x| * sum|c_x| + max|A_y| * sum|c_y| (each value of the waves
    # enters 2n + 1 outputs, weighted by |c_k|).
    _stencil.coefficient_sums = (float(np.abs(cx).sum()), float(np.abs(cy).sum()))

    return _stencil


@njit(parallel=True)
def _spin_rotation_cpu(a, t, u, u_up_down, u_down_up, row_amplitude):
    P, _, Q, H, W = a.shape
    for r in prange(P * Q * H):
        p = r // (Q * H)
        q = (r // H) % Q
        i = r % H
        amplitude = 0.0
        for j in range(W):
            up = a[p, 0, q, i, j]
            down = a[p, 1, q, i, j]
            new_up = (u[i, j] * up + u_up_down[i, j] * down) * t[i, j]
            new_down = (u[i, j].conjugate() * down + u_down_up[i, j] * up) * t[i, j]
            a[p, 0, q, i, j] = new_up
            a[p, 1, q, i, j] = new_down
            amplitude += abs(new_up) + abs(new_down)
        row_amplitude[r] = amplitude


@cuda.jit
def _spin_rotation_gpu(a, t, u, u_up_down, u_down_up):
    pq, i, j = cuda.grid(3)
    P, _, Q, H, W = a.shape
    if pq < P * Q and i < H and j < W:
        p = pq // Q
        q = pq % Q
        up = a[p, 0, q, i, j]
        down = a[p, 1, q, i, j]
        a[p, 0, q, i, j] = (u[i, j] * up + u_up_down[i, j] * down) * t[i, j]
        a[p, 1, q, i, j] = (
            u[i, j].conjugate() * down + u_down_up[i, j] * up
        ) * t[i, j]


def spin_rotation(array, spin_axis, t, u, u_up_down, u_down_up):
    """
    Apply, in place and in one pass, the per-pixel spin rotation
    [[u, u_up_down], [u_down_up, conj(u)]] followed by the scalar
    transmission `t` to spinor waves whose spin axis is `spin_axis`, and
    return the sum of the absolute values of the result. The per-pixel
    factors have the shape of the last two axes; `array` must be
    C-contiguous.
    """
    if not array.flags.c_contiguous:
        raise ValueError("the spinor array must be C-contiguous")
    shape = array.shape
    P = math.prod(shape[:spin_axis])
    Q = math.prod(shape[spin_axis + 1 : -2])
    view = array.reshape((P, 2, Q) + shape[-2:])
    xp = get_array_module(array)
    if xp is np:
        row_amplitude = np.empty(P * Q * shape[-2], dtype=array.real.dtype)
        _spin_rotation_cpu(view, t, u, u_up_down, u_down_up, row_amplitude)
        return float(row_amplitude.sum())
    _launch_3d(
        _spin_rotation_gpu, (P * Q,) + shape[-2:], view, t, u, u_up_down,
        u_down_up,
    )
    return float(xp.abs(array).sum())


@njit(parallel=True)
def _add_inplace_cpu(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    for i in prange(a.shape[0]):
        a[i] += b[i]


def add_inplace(a, b):
    """`a += b` for C-contiguous arrays of the same shape, threaded on CPU."""
    if get_array_module(a) is np:
        _add_inplace_cpu(a, b)
    else:
        a += b
    return a


class ADotGradientOperator:
    """
    Cached fused operator computing A_x * d/dx(psi) + A_y * d/dy(psi) with a
    centered finite-difference stencil and periodic boundaries, mirroring
    `abtem.finite_difference.LaplaceOperator`.

    Parameters
    ----------
    accuracy : int
        Centered finite-difference stencil accuracy (one of 2, 4, 6, 8).
    """

    def __init__(self, accuracy: int):
        self._accuracy = accuracy
        self._key = None
        self._stencil = None

    def get_stencil(self, waves, device: str = "cpu") -> Callable:
        key = (waves.sampling, waves.array.dtype, device)

        if key == self._key:
            return self._stencil

        self._stencil = _a_dot_gradient_stencil(
            self._accuracy,
            sampling=waves.sampling,
            dtype=waves.array.dtype.type,
            device=device,
        )
        self._key = key
        return self._stencil


@numba.jit(nopython=True, parallel=True)
def central_difference_gradient_pbc(X, dx=1.0, dy=1.0):
    """
    Compute the gradient of a 2D array using central differences with periodic boundary
    conditions and Numba.

    Parameters
    ----------
    X : numpy.ndarray
        Array of shape (..., N, M) where the last two dimensions are the 2D grid.
    dx : float
        Spacing between points in the x direction.
    dy : float
        Spacing between points in the y direction.

    Returns
    -------
    grad_x : numpy.ndarray
        Gradient of X with respect to x.
    grad_y : numpy.ndarray
        Gradient of X with respect to y.
    """
    original_shape = X.shape
    N, M = original_shape[-2], original_shape[-1]

    X = X.reshape(-1, N, M)
    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(X)

    for idx in prange(X.shape[0]):
        for i in range(N):
            for j in range(M):
                grad_x[idx, i, j] = (
                    X[idx, i, (j + 1) % M] - X[idx, i, (j - 1) % M]
                ) / (2 * dx)
                grad_y[idx, i, j] = (
                    X[idx, (i + 1) % N, j] - X[idx, (i - 1) % N, j]
                ) / (2 * dy)

    grad_x = grad_x.reshape(original_shape)
    grad_y = grad_y.reshape(original_shape)
    return grad_x, grad_y


@numba.jit(nopython=True, parallel=True)
def central_difference_gradient_cbc(X, dx=1.0, dy=1.0):
    """
    Compute the gradient of a 2D array using central differences with constant boundary
    conditions and Numba.

    Parameters
    ----------
    X : numpy.ndarray
        Array of shape (..., N, M) where the last two dimensions are the 2D grid.
    dx : float
        Spacing between points in the x direction.
    dy : float
        Spacing between points in the y direction.

    Returns
    -------
    grad_x : numpy.ndarray
        Gradient of X with respect to x.
    grad_y : numpy.ndarray
        Gradient of X with respect to y.
    """
    original_shape = X.shape
    N, M = original_shape[-2], original_shape[-1]

    X = X.reshape(-1, N, M)
    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(X)

    for idx in prange(X.shape[0]):
        for i in range(N):
            for j in range(M):
                if j == 0:
                    grad_x[idx, i, j] = (X[idx, i, j + 1] - X[idx, i, j]) / dx
                elif j == M - 1:
                    grad_x[idx, i, j] = (X[idx, i, j] - X[idx, i, j - 1]) / dx
                else:
                    grad_x[idx, i, j] = (X[idx, i, j + 1] - X[idx, i, j - 1]) / (2 * dx)

                if i == 0:
                    grad_y[idx, i, j] = (X[idx, i + 1, j] - X[idx, i, j]) / dy
                elif i == N - 1:
                    grad_y[idx, i, j] = (X[idx, i, j] - X[idx, i - 1, j]) / dy
                else:
                    grad_y[idx, i, j] = (X[idx, i + 1, j] - X[idx, i - 1, j]) / (2 * dy)

    grad_x = grad_x.reshape(original_shape)
    grad_y = grad_y.reshape(original_shape)

    return grad_x, grad_y


def apply_A_xy_dot_nabla_xy(A, wave_functions, sampling):
    r"""
    Compute the action of the operator $A_{xy} \cdot \nabla_{xy}$ on the wave functions
    using the central difference gradient.

    Parameters
    ----------
    A : numpy.ndarray
        Vector field of shape (2, N, M) representing the operator A.
    wave_functions : numpy.ndarray
        Array of shape (..., N, M) representing the wave functions.
    sampling : two floats
        Spacing between points in the x and y directions.

    Returns
    -------
    result : numpy.ndarray
        Result of the operation $A \cdot \nabla_{xy} \psi$.
    """
    grad_x, grad_y = central_difference_gradient_pbc(
        wave_functions, dx=sampling[0], dy=sampling[1]
    )

    A = np.broadcast_to(A, (2,) + wave_functions.shape[-2:])

    result = A[0] * grad_x + A[1] * grad_y

    return result
