"""Module to efficiently handle numerical integrals with singularities by a double exponential Tanh–Sinh quadrature."""
import numpy as np
import scipy.special
from typing import Callable, Sequence


def _error_estimate(eps: float,
                    value_estimates: Sequence[float],
                    left_summands: Sequence[float],
                    right_summands: Sequence[float]):
    # TODO: improve error estimate using the potential derivatives
    """
    Internal function to estimate the error made by the quadrature.

    Parameters
    ----------
    eps: float
        The error tolerance.
    value_estimates: list of float
        Sequence of estimated integral values
    left_summands: list of float
    right_summands: list of float

    Returns
    -------
    float
        The estimated error of the integration value (upper bound).
    """

    if len(value_estimates) < 3:
        error_estimate = 1

    elif value_estimates[0] == value_estimates[-1]:
        error_estimate = 0

    else:
        # e1 = abs(value_estimates[-1] - value_estimates[-2])
        # e2 = abs(value_estimates[-1] - value_estimates[-3])
        # e3 = eps * max(max(abs(left_summands)), max(abs(right_summands)))
        # e4 = max(abs(left_summands[-1]), abs(right_summands[-1]))
        # n = np.float((np.log(e1 + np.finfo(e1).eps) / (np.log(e2 + np.finfo(e2).eps))))

        e1 = abs(value_estimates[-1] - value_estimates[-2])
        e2 = abs(value_estimates[-1] - value_estimates[-3])
        e3 = eps * max(max(abs(left_summands)), max(abs(right_summands)))
        e4 = max(abs(left_summands[-1]), abs(right_summands[-1]))

        with np.errstate(over='ignore'):
            error_estimate = max(e1 ** (np.log(e1) / np.log(e2)), e1 ** 2, e3, e4)

        #
        #    error_estimate = max(e1 ** n, e1 ** 2, e3, e4)
        # error_estimate = max(e1, e4)

    return error_estimate


def _solve_expx_x_logx(tau, tol, max_steps=10):
    """Internal function to numerically calculate an auxiliary function."""
    x = np.log(2 / np.pi * np.log(np.pi / tau))

    def f0(x):
        return np.pi / 2 * np.exp(x) - x - np.log(x * np.pi / tau)

    def f1(x):
        return np.pi / 2 * np.exp(x) - 1 - 1 / x

    f0x = f0(x)
    success = False

    for _ in range(max_steps):
        x -= f0x / f1(x)
        f0x = f0(x)
        if abs(f0x) < tol:
            success = True
            break

    assert success
    return x


def integrate(f: Callable, a: float, b: float, eps: float, max_steps: int = 20, max_order=2000):
    """
    Integrate a function using the Tanh-Sinh quadrature method.

    Parameters
    ----------
    f: callable
        Function to integrate.
    a: float
        Lower integration limit.
    b: float
        Upper integration limit.
    eps: float
        Integration error tolerance.
    max_steps: int
        Maximum number of refinement steps allowed before an error is raised.

    Returns
    -------
    integral_value: float
        The estimated integration value.
    error_estimate: float
        The estimated error of the integration value (upper bound).
    step_size: float
        The quadrature step size used to obtain the final integral value.
    order: int
        The quadrature order used to obtain the final integral value.
    """

    def f_left(s):
        return f(a + s)

    def f_right(s):
        return f(b - s)

    interval = b - a

    def lambertw(x, k):
        out = scipy.special.lambertw(x, k)
        assert abs(out.imag) < 1.0e-15
        return scipy.special.lambertw(x, k).real

    h = _solve_expx_x_logx(eps ** 2, tol=1e-7)

    order = 0
    success = False
    for level in range(max_steps + 1):
        assert eps ** 2 * np.exp(np.pi / 2) < np.pi * h
        j = int(np.log(-2 / np.pi * lambertw(-(eps ** 2) / h / 2, -1)) / h)

        if level == 0:
            t = [0]
        else:
            t = h * np.arange(1, j + 1, 2)

        sinh_t = np.pi / 2 * np.sinh(t)
        cosh_t = np.pi / 2 * np.cosh(t)
        cosh_sinh_t = np.cosh(sinh_t)
        exp_sinh_t = np.exp(sinh_t)

        y0 = interval / 2 / exp_sinh_t / cosh_sinh_t

        weights = h * interval / 2 * cosh_t / cosh_sinh_t ** 2
        left_summands = f_left(y0) * weights
        right_summands = f_right(y0) * weights

        if level == 0:
            value_estimates = list(left_summands)
        else:
            value_estimates.append(value_estimates[-1] / 2 + np.sum(left_summands) + np.sum(right_summands))

        error_estimate = _error_estimate(eps, value_estimates, left_summands, right_summands)

        if order + j + 1 > max_order:
            success = True
            break

        if abs(error_estimate) < eps:
            success = True
            break

        order += j + 1
        h /= 2

    assert success
    return value_estimates[-1], error_estimate, h, order - 1


def tanh_sinh_nodes_and_weights(step_size: float, order: int):
    """
    Calculate the nodes and weights for the tanh-sinh quadrature with a given step size and order.

    Parameters
    ----------
    step_size: float
        The quadrature step size.
    order: int
        The quadrature order.

    Returns
    -------
    array, array
        Nodes and weights
    """

    xk = np.zeros(2 * order + 1, dtype=np.float)
    wk = np.zeros(2 * order + 1, dtype=np.float)
    for i, k in enumerate(range(-order, order + 1)):
        xk[i] = np.tanh(np.pi / 2 * np.sinh(k * step_size))
        numerator = step_size / 2 * np.pi * np.cosh(k * step_size)
        denominator = np.cosh(np.pi / 2 * np.sinh(k * step_size)) ** 2
        wk[i] = numerator / denominator
    return xk, wk
