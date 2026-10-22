"""Functions of a Hermitian operator applied to a wave, by a Lanczos iteration.

The exact one-way root of the multislice operator, ``F = sqrt(k^2 + laplacian +
2 k sigma V) - k``, is a function of the Hermitian operator ``O = laplacian / (2 k) +
sigma V``. Its action on a wave is evaluated here in the Krylov space of the wave and
the operator, one application of the operator per iteration, rather than as a
truncated Taylor series, which diverges where ``2 O / k`` reaches 1.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from abtem.core.backend import asnumpy


def inner(a, b, xp):
    """Per-member inner product over the last two axes, shape (B,)."""
    return xp.sum(xp.conj(a) * b, axis=(-2, -1))


def _check_propagating(argument: np.ndarray) -> None:
    if argument.min() <= 0.0:
        raise NotImplementedError(
            "The operator has eigenvalues at or below the edge of the propagating "
            "band (an exceptionally strong potential); lower propagating_cutoff or "
            "use a finer slicing."
        )


def exponential_of_root(k: float, thickness: float) -> Callable:
    """``t -> exp(i dz k (sqrt(1 + 2 t / k) - 1))``, the step of one slice."""

    def function(values):
        argument = 1.0 + 2.0 * values / k
        _check_propagating(argument)
        return np.exp(1j * thickness * k * (np.sqrt(argument) - 1.0))

    return function


def wave_number(k: float) -> Callable:
    """``t -> k sqrt(1 + 2 t / k)``, the local wave number operator."""

    def function(values):
        argument = 1.0 + 2.0 * values / k
        _check_propagating(argument)
        return (k * np.sqrt(argument)).astype(complex)

    return function


def inverse_wave_number(k: float) -> Callable:
    """``t -> 1 / (k sqrt(1 + 2 t / k))``."""

    def function(values):
        argument = 1.0 + 2.0 * values / k
        _check_propagating(argument)
        return (1.0 / (k * np.sqrt(argument))).astype(complex)

    return function


def lanczos_function(
    a0, apply: Callable, function: Callable, tolerance: float, max_iterations: int, xp
):
    """
    ``function(O) a0`` for every member of the batch ``a0``, of shape (B, nx, ny).

    One application of the Hermitian operator ``apply`` per iteration, with full
    reorthogonalisation (the number of iterations is small). The small tridiagonal
    matrices are diagonalised on the host, in double precision, and ``function`` maps
    their eigenvalues (B, m) to complex values of the same shape.

    The iteration stops when the coefficients on the Krylov basis change by less than
    ``tolerance``.
    """
    from abtem.finite_difference import NotConvergedError

    dtype = a0.dtype
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    norm0 = xp.sqrt(inner(a0, a0, xp).real)
    norm0_host = asnumpy(norm0).astype(np.float64)
    if (norm0_host == 0).all():
        return xp.zeros_like(a0)

    safe = xp.asarray(np.where(norm0_host > 0, norm0_host, 1.0).astype(real_dtype))
    basis = [a0 / safe[:, None, None]]
    alphas, betas = [], []
    coefficients_previous = None

    for _ in range(max_iterations):
        w = apply(basis[-1])
        alpha = asnumpy(inner(basis[-1], w, xp).real).astype(np.float64)
        w = w - xp.asarray(alpha.astype(real_dtype))[:, None, None] * basis[-1]
        if betas:
            w = w - xp.asarray(betas[-1].astype(real_dtype))[:, None, None] * basis[-2]
        for vector in basis:
            w = w - inner(vector, w, xp)[:, None, None] * vector
        alphas.append(alpha)

        size = len(alphas)
        tridiagonal = np.zeros((a0.shape[0], size, size))
        index = np.arange(size)
        tridiagonal[:, index, index] = np.stack(alphas, axis=1)
        if betas:
            off = np.stack(betas, axis=1)
            tridiagonal[:, index[:-1], index[1:]] = off
            tridiagonal[:, index[1:], index[:-1]] = off
        values, vectors = np.linalg.eigh(tridiagonal)

        coefficients = np.einsum(
            "bij,bj,bj->bi", vectors, function(values), vectors[:, 0, :].conj()
        )

        if coefficients_previous is not None:
            padded = np.pad(coefficients_previous, ((0, 0), (0, 1)))
            if np.linalg.norm(coefficients - padded, axis=1).max() < tolerance:
                break

        beta = asnumpy(xp.sqrt(inner(w, w, xp).real)).astype(np.float64)
        if (beta < 1e-14).all():
            break
        betas.append(beta)
        basis.append(
            w
            / xp.asarray(np.where(beta > 1e-14, beta, 1.0).astype(real_dtype))[
                :, None, None
            ]
        )
        coefficients_previous = coefficients
    else:
        raise NotConvergedError(
            f"the Lanczos iteration did not converge to {tolerance} in "
            f"{max_iterations} iterations"
        )

    result = xp.zeros_like(a0)
    for j, vector in enumerate(basis[: coefficients.shape[1]]):
        weight = xp.asarray((coefficients[:, j] * norm0_host).astype(dtype))
        result = result + weight[:, None, None] * vector
    return result
