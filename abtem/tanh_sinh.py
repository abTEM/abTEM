import numpy as np
import scipy.special

def _error_estimate(eps, value_estimates, left_summands, right_summands):
    if len(value_estimates) < 3:
        error_estimate = 1

    elif value_estimates[0] == value_estimates[-1]:
        error_estimate = 0

    else:
        e1 = abs(value_estimates[-1] - value_estimates[-2])
        #e2 = abs(value_estimates[-1] - value_estimates[-3])
        #e3 = eps * max(max(abs(left_summands)), max(abs(right_summands)))
        e4 = max(abs(left_summands[-1]), abs(right_summands[-1]))
        #n = np.float((np.log(e1 + np.finfo(e1).eps) / (np.log(e2 + np.finfo(e2).eps))))

        #with np.errstate(over='ignore'):
        #    error_estimate = max(e1 ** n, e1 ** 2, e3, e4)
        #print(e1, e2, e3, e4, error_estimate)

        error_estimate = min(e1, e4)

    return error_estimate


def _solve_expx_x_logx(tau, tol, max_steps=10):
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


def integrate(f, a, b, eps, max_steps=20):
    f_left = lambda s: f(a + s)
    f_right = lambda s: f(b - s)
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
        if abs(error_estimate) < eps:
            success = True
            break
        order += j + 1
        h /= 2

    assert success
    return value_estimates[-1], error_estimate, h, order - 1


def tanh_sinh_nodes_and_weights(step_size, order):
    xk = np.zeros(2 * order + 1, dtype=np.float)
    wk = np.zeros(2 * order + 1, dtype=np.float)
    for i, k in enumerate(range(-order, order + 1)):
        xk[i] = np.tanh(np.pi / 2 * np.sinh(k * step_size))
        numerator = step_size / 2 * np.pi * np.cosh(k * step_size)
        denominator = np.cosh(np.pi / 2 * np.sinh(k * step_size)) ** 2
        wk[i] = numerator / denominator
    return xk, wk