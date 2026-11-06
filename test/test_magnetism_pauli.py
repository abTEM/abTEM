import numpy as np

from abtem.magnetism.pauli import apply_A_xy_dot_nabla_xy


def test_apply_A_xy_dot_nabla_xy_is_the_periodic_central_difference():
    """The operator is checked against ``np.roll`` central differences, on a
    stack of non-square arrays so that a transposed axis changes the result."""
    rng = np.random.default_rng(0)
    wave = rng.standard_normal((4, 12, 9)) + 1j * rng.standard_normal((4, 12, 9))
    A = rng.standard_normal((2, 12, 9))
    dx, dy = 0.1, 0.25

    grad_x = (np.roll(wave, -1, axis=-1) - np.roll(wave, 1, axis=-1)) / (2 * dx)
    grad_y = (np.roll(wave, -1, axis=-2) - np.roll(wave, 1, axis=-2)) / (2 * dy)

    result = apply_A_xy_dot_nabla_xy(A, wave, (dx, dy))

    np.testing.assert_allclose(result, A[0] * grad_x + A[1] * grad_y, atol=1e-12)
