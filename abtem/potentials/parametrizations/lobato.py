import numpy as np
from ase import units
from numba import jit
from scipy.special import kn


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    k2 = k ** 2
    return ((p[0, 0] * (2. + p[1, 0] * k2) / (1. + p[1, 0] * k2) ** 2) +
            (p[0, 1] * (2. + p[1, 1] * k2) / (1. + p[1, 1] * k2) ** 2) +
            (p[0, 2] * (2. + p[1, 2] * k2) / (1. + p[1, 2] * k2) ** 2) +
            (p[0, 3] * (2. + p[1, 3] * k2) / (1. + p[1, 3] * k2) ** 2) +
            (p[0, 4] * (2. + p[1, 4] * k2) / (1. + p[1, 4] * k2) ** 2))


@jit(nopython=True, nogil=True)
def potential(r, p):
    return (p[0, 0] * (2. / (p[1, 0] * r) + 1.) * np.exp(-p[1, 0] * r) +
            p[0, 1] * (2. / (p[1, 1] * r) + 1.) * np.exp(-p[1, 1] * r) +
            p[0, 2] * (2. / (p[1, 2] * r) + 1.) * np.exp(-p[1, 2] * r) +
            p[0, 3] * (2. / (p[1, 3] * r) + 1.) * np.exp(-p[1, 3] * r) +
            p[0, 4] * (2. / (p[1, 4] * r) + 1.) * np.exp(-p[1, 4] * r))


@jit(nopython=True, nogil=True)
def potential_derivative(r, p):
    dvdr = - (p[0, 0] * (2. / (p[1, 0] * r ** 2) + 2. / r + p[1, 0]) * np.exp(-p[1, 0] * r) +
              p[0, 1] * (2. / (p[1, 1] * r ** 2) + 2. / r + p[1, 1]) * np.exp(-p[1, 1] * r) +
              p[0, 2] * (2. / (p[1, 2] * r ** 2) + 2. / r + p[1, 2]) * np.exp(-p[1, 2] * r) +
              p[0, 3] * (2. / (p[1, 3] * r ** 2) + 2. / r + p[1, 3]) * np.exp(-p[1, 3] * r) +
              p[0, 4] * (2. / (p[1, 4] * r ** 2) + 2. / r + p[1, 4]) * np.exp(-p[1, 4] * r))
    return dvdr


@jit(nopython=True, nogil=True)
def charge(r, p):
    n = (2 * np.pi ** 4 * units.Bohr * p[0, 0] / p[1, 0] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 0])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 1] / p[1, 1] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 1])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 2] / p[1, 2] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 2])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 3] / p[1, 3] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 3])) +
         2 * np.pi ** 4 * units.Bohr * p[0, 4] / p[1, 4] ** (5 / 2) * np.exp(-2 * np.pi * r / np.sqrt(p[1, 4])))
    return n


def projected_potential(r, p):
    return 2 * (2 * p[0][:, None] / p[1][:, None] * kn(0, r[None] * p[1][:, None]) +
                p[0][:, None] * r[None] * kn(1, r[None] * p[1][:, None])).sum(0)


def projected_scattering_factor(k, p):
    f = 8 * np.pi * (p[0][:, None] / p[1][:, None] * 1 / (4 * np.pi ** 2 * k[None] ** 2 + p[1][:, None] ** 2) +
                     p[0][:, None] * p[1][:, None] / (4 * np.pi ** 2 * k[None] ** 2 + p[1][:, None] ** 2) ** 2).sum(0)
    return f
