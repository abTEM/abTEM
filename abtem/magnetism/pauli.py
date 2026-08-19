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

    The returned function takes (waves, A_x_padded, A_y_padded) where waves
    has shape (..., nx, ny) — axis -2 is x and axis -1 is y, following the
    abTEM array convention — and A_x_padded, A_y_padded are A_x, A_y
    pre-padded with the same wrap boundary via the returned function's
    `.pad_field` attribute.

    A_x and A_y are constant across all Taylor-series terms of a slice's
    exponential (only the wave function changes between terms), so the
    caller pads them once per slice with `.pad_field` and reuses the result,
    rather than re-padding on every term.

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
    padding = n + 1

    # `Ax_u` / `Ay_u` carry a per-ensemble-member *uniform* contribution to
    # A_x / A_y, used by the probe-centred gauge (see `pauli_multislice`'s
    # `gauge_origin`). For an axial B_avg the uniform vector potential
    # A_np = 1/2 B_avg x (r - r0) is separable: A_x depends only on y (and on
    # the member's own gauge origin) and A_y only on x, so a per-member
    # (B, W) / (B, H) pair replaces what would otherwise be a full (B, H, W)
    # field per member -- hundreds of MB at production grid sizes. The
    # leading axis is indexed modulo its length, so passing length-1 zeros
    # disables the term with no branching in the inner loop.

    @njit(parallel=True, fastmath=True)
    def _stencil_cpu_batch(a, Ax, Ay, Ax_u, Ay_u):
        M, H, W = a.shape
        Bu = Ax_u.shape[0]
        # a.copy() + slice-fill rather than np.zeros_like: some numba/numpy
        # pairings (observed: numba 0.64.0 + numpy 2.4.3) fail to type numba's
        # internal np.zeros_like -> np.empty_like lowering inside @njit, while
        # ndarray.copy() (used identically by finite_difference.py's Laplace
        # stencil) is unaffected.
        out = a.copy()
        out[:] = 0
        for m in prange(M):
            bu = m % Bu
            for i in range(n, H - n):
                for j in range(n, W - n):
                    ax = Ax[i, j] + Ax_u[bu, j]
                    ay = Ay[i, j] + Ay_u[bu, i]
                    cumul = dtype(0.0)
                    for k in range(-n, n + 1):
                        cumul += ax * cx[k] * a[m, i + k, j]
                        cumul += ay * cy[k] * a[m, i, j + k]
                    out[m, i, j] = cumul
        return out

    @cuda.jit
    def _stencil_func_gpu_batch(a, Ax, Ay, Ax_u, Ay_u, out):
        m, i, j = cuda.grid(3)
        M, H, W = a.shape
        Bu = Ax_u.shape[0]
        if m < M and n <= i < H - n and n <= j < W - n:
            bu = m % Bu
            ax = Ax[i, j] + Ax_u[bu, j]
            ay = Ay[i, j] + Ay_u[bu, i]
            cumul = dtype(0.0)
            for k in range(-n, n + 1):
                cumul += ax * cx[k] * a[m, i + k, j]
                cumul += ay * cy[k] * a[m, i, j + k]
            out[m, i, j] = cumul

    def _stencil_gpu(a, Ax, Ay, Ax_u, Ay_u):
        xp = get_array_module(a)
        out = xp.zeros_like(a)

        threads_x = 8
        threads_y = 8

        target_threads = 256
        threads_m = max(1, target_threads // (threads_x * threads_y))

        threadsperblock = (threads_m, threads_x, threads_y)

        blockspergrid = (
            math.ceil(a.shape[0] / threadsperblock[0]),
            math.ceil(a.shape[1] / threadsperblock[1]),
            math.ceil(a.shape[2] / threadsperblock[2]),
        )
        _stencil_func_gpu_batch[blockspergrid, threadsperblock](
            a, Ax, Ay, Ax_u, Ay_u, out
        )

        return out

    def pad_field(A):
        """Pad a static A_x/A_y field with the stencil's wrap boundary.
        Call once per slice and reuse across all Taylor-series terms."""
        xp = get_array_module(A)
        return xp.pad(xp.asarray(A), padding, mode="wrap")

    def zero_uniform(xp, n_padded):
        """A disabled uniform term: one row of zeros, broadcast by the
        modulo indexing in the kernels."""
        return xp.zeros((1, n_padded), dtype=dtype)

    def _stencil(a, Ax_padded, Ay_padded, Ax_uniform=None, Ay_uniform=None):
        xp = get_array_module(a)

        original_shape = a.shape
        a = a.reshape(-1, *a.shape[-2:])

        pad_width = [(0, 0), (padding,) * 2, (padding,) * 2]
        a = xp.pad(a, pad_width=pad_width, mode="wrap")

        if Ax_uniform is None:
            Ax_uniform = zero_uniform(xp, a.shape[2])
        if Ay_uniform is None:
            Ay_uniform = zero_uniform(xp, a.shape[1])

        if device == "cpu":
            result = _stencil_cpu_batch(a, Ax_padded, Ay_padded,
                                        Ax_uniform, Ay_uniform)
        elif device == "gpu":
            result = _stencil_gpu(a, Ax_padded, Ay_padded,
                                  Ax_uniform, Ay_uniform)
        else:
            raise ValueError(f"Unsupported device: {device}")

        result = result[:, padding:-padding, padding:-padding]
        return result.reshape(original_shape)

    _stencil.pad_field = pad_field
    _stencil.zero_uniform = zero_uniform
    _stencil.padding = padding

    return _stencil


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
    X : np.ndarray
        Array of shape (..., N, M) where the last two dimensions are the 2D grid.
    dx : float
        Spacing between points in the x direction.
    dy : float
        Spacing between points in the y direction.

    Returns
    -------
    grad_x : np.ndarray
        Gradient of X with respect to x.
    grad_y : np.ndarray
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
    X : np.ndarray
        Array of shape (..., N, M) where the last two dimensions are the 2D grid.
    dx : float
        Spacing between points in the x direction.
    dy : float
        Spacing between points in the y direction.

    Returns
    -------
    grad_x : np.ndarray
        Gradient of X with respect to x.
    grad_y : np.ndarray
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
    A : np.ndarray
        Vector field of shape (2, N, M) representing the operator A.
    wave_functions : np.ndarray
        Array of shape (..., N, M) representing the wave functions.
    sampling : two floats
        Spacing between points in the x and y directions.

    Returns
    -------
    result : np.ndarray
        Result of the operation $A \cdot \nabla_{xy} \psi$.
    """
    grad_x, grad_y = central_difference_gradient_pbc(
        wave_functions, dx=sampling[0], dy=sampling[1]
    )

    A = np.broadcast_to(A, (2,) + wave_functions.shape[-2:])

    result = A[0] * grad_x + A[1] * grad_y

    return result
