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


def test_the_basis_grows_beyond_its_first_allocation():
    # Oracle: the dense evaluation of an oscillatory function of an operator with 30
    # well separated eigenvalues, which takes more than the 16 vectors that the basis
    # starts with (counted by the applications of the operator); the three-term
    # recurrence, with no reorthogonalisation, gives it to 1e-8.
    n = SHAPE[0] * SHAPE[1]
    rng = np.random.default_rng(5)
    unitary, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    spectrum = np.linspace(-1.0, 1.0, n) ** 3 * 1.9
    matrix = (unitary * spectrum) @ unitary.conj().T
    matrix = (matrix + matrix.conj().T) / 2
    a0 = (rng.normal(size=(1,) + SHAPE) + 1j * rng.normal(size=(1,) + SHAPE)).astype(
        np.complex128
    )

    def oscillatory(values):
        return np.exp(25j * values)

    applications = []
    apply = apply_matrix(matrix)

    def counted(a):
        applications.append(1)
        return apply(a)

    result = _lanczos_device(
        a0,
        counted,
        oscillatory,
        1e-10,
        60,
        np,
        diagonalise_on_device=True,
    )
    values, vectors = np.linalg.eigh(matrix)
    expected = ((a0.reshape(1, -1) @ vectors.conj()) * oscillatory(values)) @ vectors.T
    assert len(applications) > 16
    np.testing.assert_allclose(result.reshape(1, -1), expected, atol=1e-8)


def other_function(values):
    return np.cos(0.5 * values) / (3.0 + values**2)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_several_functions_come_from_one_krylov_space(implementation):
    # Oracle: each function on its own, with the dense evaluation of the first as a
    # check of the whole; the operator is applied once per iteration for all of them.
    matrix = hermitian_matrix(6)
    rng = np.random.default_rng(7)
    a0 = (rng.normal(size=(2,) + SHAPE) + 1j * rng.normal(size=(2,) + SHAPE)).astype(
        np.complex128
    )
    applications = []

    def apply(a):
        applications.append(1)
        return apply_matrix(matrix)(a)

    together = implementation(a0, apply, [function, other_function], 1e-10, 60, np)
    count_together, applications = len(applications), []
    alone = [
        implementation(a0, apply, f, 1e-10, 60, np) for f in (function, other_function)
    ]
    count_alone = len(applications)

    assert len(together) == 2
    for a, b in zip(together, alone):
        np.testing.assert_allclose(a, b, atol=1e-8)
    np.testing.assert_allclose(
        together[0], dense_reference(matrix, a0).reshape(a0.shape), atol=1e-7
    )
    assert count_together < count_alone


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_function_of_the_stepped_wave_is_the_product_of_the_functions(implementation):
    # Oracle: g(O) f(O) a0 = (g f)(O) a0, which is what lets the wave number of a slice
    # act on the stepped wave from the Krylov space of the step. The left side is two
    # iterations, the right side one.
    matrix = hermitian_matrix(8)
    rng = np.random.default_rng(9)
    a0 = (rng.normal(size=(2,) + SHAPE) + 1j * rng.normal(size=(2,) + SHAPE)).astype(
        np.complex128
    )
    apply = apply_matrix(matrix)
    stepped = implementation(a0, apply, function, 1e-10, 60, np)
    twice = implementation(stepped, apply, other_function, 1e-10, 60, np)
    product = implementation(
        a0, apply, [function, lambda v: other_function(v) * function(v)], 1e-10, 60, np
    )[1]
    np.testing.assert_allclose(product, twice, atol=1e-7)


def test_each_function_is_converged_to_its_own_tolerance():
    # Oracle: a loose tolerance for the second function does not stop the iteration of
    # the first early: the first is as accurate as when it runs alone.
    matrix = hermitian_matrix(10)
    rng = np.random.default_rng(11)
    a0 = (rng.normal(size=(1,) + SHAPE) + 1j * rng.normal(size=(1,) + SHAPE)).astype(
        np.complex128
    )
    apply = apply_matrix(matrix)
    tight, loose = lanczos_function(
        a0, apply, [function, other_function], [1e-10, 1e-2], 60, np
    )
    np.testing.assert_allclose(
        tight, dense_reference(matrix, a0).reshape(a0.shape), atol=1e-7
    )
    with pytest.raises(ValueError, match="one tolerance for each function"):
        lanczos_function(a0, apply, [function, other_function], [1e-6], 60, np)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_callable_with_outputs_returns_all_functions_stacked(implementation):
    # Oracle: the same functions given one by one.
    matrix = hermitian_matrix(12)
    rng = np.random.default_rng(13)
    a0 = (rng.normal(size=(2,) + SHAPE) + 1j * rng.normal(size=(2,) + SHAPE)).astype(
        np.complex128
    )

    def both(values):
        return np.stack([function(values), other_function(values)])

    both.outputs = 2
    apply = apply_matrix(matrix)
    stacked = implementation(a0, apply, both, [1e-10, 1e-6], 60, np)
    separate = implementation(
        a0, apply, [function, other_function], [1e-10, 1e-6], 60, np
    )
    assert len(stacked) == 2
    for a, b in zip(stacked, separate):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_the_step_and_the_wave_number_of_a_slice_are_the_functions_stacked():
    # Oracle: exponential_of_root and excess_wave_number of the same slice.
    from abtem._lanczos import (
        excess_wave_number,
        exponential_of_root,
        step_and_current_wave_number,
    )

    k, thickness, reference = 16.0, 0.05, 15.5
    values = np.random.default_rng(14).uniform(-3.0, 3.0, (2, 7))
    both = step_and_current_wave_number(k, thickness, reference)
    step = exponential_of_root(k, thickness, reference)(values)
    excess = excess_wave_number(k)(values)
    assert both.outputs == 2
    np.testing.assert_allclose(both(values)[0], step, atol=1e-14)
    np.testing.assert_allclose(both(values)[1], excess * step, atol=1e-13)
