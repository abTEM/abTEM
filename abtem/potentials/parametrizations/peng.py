import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    return (p[0, 0] * np.exp(-p[1, 0] * k ** 2.) +
            p[0, 1] * np.exp(-p[1, 1] * k ** 2.) +
            p[0, 2] * np.exp(-p[1, 2] * k ** 2.) +
            p[0, 3] * np.exp(-p[1, 3] * k ** 2.) +
            p[0, 4] * np.exp(-p[1, 4] * k ** 2.))


def scattering_factor_ionic(k, p, charge):
    return scattering_factor(k, p) + 0.023934 * charge / k ** 2


def potential(r, p):
    return scattering_factor(r, p)


def potential_ionic(r, p, charge):
    return potential(r, p) + 0.023934 * charge / r * 2 * np.pi ** 2

