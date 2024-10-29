from __future__ import annotations

import numba  # type: ignore
import numpy as np
from numba import prange


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
