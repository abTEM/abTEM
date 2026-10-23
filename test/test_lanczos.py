"""The Lanczos iteration of abtem._lanczos: a function of a Hermitian operator applied
to a wave."""

import functools

import numpy as np
import pytest

from abtem._lanczos import _lanczos_device, _lanczos_host, lanczos_function

SHAPE = (6, 5)


def hermitian_matrix(seed):
    rng = np.random.default_rng(seed)
    n = SHAPE[0] * SHAPE[1]
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (matrix + matrix.conj().T) / (2 * np.sqrt(n))


def apply_matrix(matrix):
    return lambda a: (a.reshape(len(a), -1) @ matrix.T).reshape(a.shape)


def function(values):
    return np.exp(1j * 0.7 * np.sqrt(4.0 + 2.0 * values))


def dense_reference(matrix, a0):
    """function(matrix) applied to the waves a0 by diagonalising the matrix."""
    values, vectors = np.linalg.eigh(matrix)
    flat = a0.reshape(len(a0), -1)
    # for the column vector x, V f(L) V^H x; the waves are the rows
    return ((flat @ vectors.conj()) * function(values)) @ vectors.T


# the host implementation, the device one with the tridiagonal matrices diagonalised on
# the host, and with them diagonalised on the device (as with CuPy; here the module is
# NumPy, whose batched eigh has the same interface)
IMPLEMENTATIONS = [
    pytest.param(_lanczos_host, id="host"),
    pytest.param(
        functools.partial(_lanczos_device, diagonalise_on_device=False),
        id="device-host-eigh",
    ),
    pytest.param(
        functools.partial(_lanczos_device, diagonalise_on_device=True),
        id="device-device-eigh",
    ),
]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_function_of_a_hermitian_operator_equals_its_dense_evaluation(implementation):
    # Oracle: diagonalise the operator (30 x 30), apply the function to its
    # eigenvalues; two waves of the batch, each with its own Krylov space.
    matrix = hermitian_matrix(0)
    rng = np.random.default_rng(1)
    a0 = (rng.normal(size=(2,) + SHAPE) + 1j * rng.normal(size=(2,) + SHAPE)).astype(
        np.complex128
    )
    result = implementation(a0, apply_matrix(matrix), function, 1e-10, 60, np)
    expected = dense_reference(matrix, a0).reshape(a0.shape)
    np.testing.assert_allclose(result, expected, atol=1e-7)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_wave_that_is_an_eigenvector_closes_the_krylov_space_at_once(implementation):
    # Oracle: f(O) v = f(lambda) v for an eigenvector v; the Krylov space has one
    # vector, and the iteration must stop (or carry zero vectors) without a failure.
    matrix = hermitian_matrix(4)
    values, vectors = np.linalg.eigh(matrix)
    a0 = vectors[:, 3].reshape((1,) + SHAPE) * np.array([[[2.0]]])
    result = implementation(a0, apply_matrix(matrix), function, 1e-10, 60, np)
    np.testing.assert_allclose(result, function(values[3]) * a0, atol=1e-9)


def test_the_implementations_for_the_host_and_for_a_device_agree():
    # Oracle: each other, in single precision, for a batch with a wave that is zero
    # (it must stay zero and not disturb the others).
    matrix = hermitian_matrix(2).astype(np.complex64)
    rng = np.random.default_rng(3)
    a0 = (rng.normal(size=(3,) + SHAPE) + 1j * rng.normal(size=(3,) + SHAPE)).astype(
        np.complex64
    )
    a0[1] = 0.0
    apply = apply_matrix(matrix)
    host = _lanczos_host(a0, apply, function, 1e-6, 60, np)
    for on_device in (False, True):
        device = _lanczos_device(
            a0, apply, function, 1e-6, 60, np, diagonalise_on_device=on_device
        )
        np.testing.assert_allclose(device, host, atol=1e-4)
        assert np.abs(device[1]).max() == 0.0


def test_a_zero_batch_gives_zero():
    a0 = np.zeros((2,) + SHAPE, np.complex64)
    result = lanczos_function(a0, lambda a: a, function, 1e-8, 10, np)
    assert np.abs(result).max() == 0.0
