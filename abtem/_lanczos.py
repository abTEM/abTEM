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

from abtem.core.backend import asnumpy, cp


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


def exponential_of_root(
    k: float, thickness: float, reference: float | None = None
) -> Callable:
    """``t -> exp(i dz (k sqrt(1 + 2 t / k) - reference))``, the step of one slice;
    the reference wave number is ``k`` unless a tilted wave, stored without the carrier
    of its direction, takes out its own."""
    reference = k if reference is None else reference

    def function(values):
        argument = 1.0 + 2.0 * values / k
        _check_propagating(argument)
        return np.exp(1j * thickness * (k * np.sqrt(argument) - reference))

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

    Two implementations of the same iteration: on the host a loop over the basis (BLAS
    threads on products this small cost more than they gain), and on a device the basis
    in one array, with batched products, a normalisation that stays on the device and
    one transfer to the host per iteration, since there the cost is the number of
    kernels and of transfers, not the arithmetic.
    """
    if xp is np:
        return _lanczos_host(a0, apply, function, tolerance, max_iterations, xp)
    return _lanczos_device(a0, apply, function, tolerance, max_iterations, xp)


def _lanczos_host(
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


def _lanczos_device(
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

    On a GPU the cost of this routine is the number of kernels and of transfers, not
    arithmetic: the basis is one array, the orthogonalisation takes four kernels
    whatever its size, everything that the next iteration needs stays on the device,
    and one transfer (the diagonal element and the norm of the residual, together)
    goes to the host per iteration.
    """
    from abtem.finite_difference import NotConvergedError

    dtype = a0.dtype
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    members, shape = a0.shape[0], a0.shape[1:]
    norm0 = xp.sqrt(inner(a0, a0, xp).real)
    # the Krylov basis of every member, (members, iterations + 1, nx * ny); a member
    # with nothing in it stays zero
    size_xy = shape[0] * shape[1]
    basis = xp.zeros((members, max_iterations + 1, size_xy), dtype)
    basis[:, 0] = (
        a0 / xp.where(norm0 > 0, norm0, 1.0).astype(real_dtype)[:, None, None]
    ).reshape(members, size_xy)
    alphas, betas = [], []
    coefficients_previous = None

    if cp is not None and xp is cp:

        def project_out(vectors, w):
            """w without its components along the vectors (members, m, nx * ny), and
            the overlaps <vectors, w> (members, m): two batched matrix products."""
            overlaps = xp.conj(xp.matmul(vectors, xp.conj(w)[:, :, None]))[:, :, 0]
            return w - xp.matmul(overlaps[:, None, :], vectors)[:, 0], overlaps

    else:

        def project_out(vectors, w):
            """The same with a product and a sum each, whatever m (a batched matrix
            product is wrong for complex numbers on Metal)."""
            overlaps = xp.sum(xp.conj(vectors) * w[:, None, :], axis=-1)
            return w - xp.sum(overlaps[:, :, None] * vectors, axis=1), overlaps

    for step in range(max_iterations):
        vectors = basis[:, : step + 1]
        w = apply(basis[:, step].reshape((members,) + shape)).reshape(members, size_xy)
        # Gram-Schmidt against the whole basis, twice (the three-term recurrence is
        # contained in it): alpha is the overlap with the current vector
        w, overlaps = project_out(vectors, w)
        w, _ = project_out(vectors, w)
        beta_device = xp.sqrt(xp.sum(xp.abs(w) ** 2, axis=-1))
        host = asnumpy(xp.stack([overlaps[:, step].real, beta_device])).astype(
            np.float64
        )
        alphas.append(host[0])
        beta = host[1]

        size = len(alphas)
        tridiagonal = np.zeros((members, size, size))
        index = np.arange(size)
        tridiagonal[:, index, index] = np.stack(alphas, axis=1)
        if betas:
            off = np.stack(betas, axis=1)
            tridiagonal[:, index[:-1], index[1:]] = off
            tridiagonal[:, index[1:], index[:-1]] = off
        values, vectors_host = np.linalg.eigh(tridiagonal)

        coefficients = np.einsum(
            "bij,bj,bj->bi",
            vectors_host,
            function(values),
            vectors_host[:, 0, :].conj(),
        )

        if coefficients_previous is not None:
            difference = coefficients.copy()
            difference[:, :-1] -= coefficients_previous
            if np.linalg.norm(difference, axis=1).max() < tolerance:
                break

        if (beta < 1e-14).all():
            break
        betas.append(beta)
        basis[:, step + 1] = (
            w / xp.where(beta_device > 1e-14, beta_device, 1.0)[:, None]
        )
        coefficients_previous = coefficients
    else:
        raise NotConvergedError(
            f"the Lanczos iteration did not converge to {tolerance} in "
            f"{max_iterations} iterations"
        )

    # the result, in one transfer to the device and one contraction
    weights = xp.asarray(coefficients.astype(dtype)) * norm0.astype(dtype)[:, None]
    result = xp.sum(
        weights[:, :, None] * basis[:, : coefficients.shape[1]], axis=1
    )
    return result.reshape((members,) + shape)
