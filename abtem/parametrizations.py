"""Module to describe independent atom model parametrizations of the scattering potential."""
import csv
import os
from scipy.special import kn

import numpy as np
from numba import jit

from abtem.utils import _set_path


def load_parameters(filename):
    """Function to load parameters from a CSV file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    parameters = {}
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        keys = next(reader)
        for _, row in enumerate(reader):
            values = list(map(float, row))
            parameters[int(row[0])] = dict(zip(keys, values))
    return parameters


def load_lobato_parameters(convert=True):
    """Function to load the default Lobato parameters (doi:10.1107/S205327331401643X)."""
    parameters = {}

    for key, value in load_parameters(_set_path('lobato.txt')).items():
        a = np.array([value[key] for key in ('a1', 'a2', 'a3', 'a4', 'a5')])
        b = np.array([value[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5')])

        if convert:
            a = np.pi ** 2 * a / b ** (3 / 2.)
            b = 2 * np.pi / np.sqrt(b)

        parameters[key] = np.vstack((a, b))

    return parameters


@jit(nopython=True, nogil=True)
def lobato_scattering(k2, p):
    return ((p[0, 0] * (2. + p[1, 0] * k2) / (1. + p[1, 0] * k2) ** 2) +
            (p[0, 1] * (2. + p[1, 1] * k2) / (1. + p[1, 1] * k2) ** 2) +
            (p[0, 2] * (2. + p[1, 2] * k2) / (1. + p[1, 2] * k2) ** 2) +
            (p[0, 3] * (2. + p[1, 3] * k2) / (1. + p[1, 3] * k2) ** 2) +
            (p[0, 4] * (2. + p[1, 4] * k2) / (1. + p[1, 4] * k2) ** 2))


@jit(nopython=True, nogil=True)
def lobato(r, p):
    return (p[0, 0] * (2. / (p[1, 0] * r) + 1.) * np.exp(-p[1, 0] * r) +
            p[0, 1] * (2. / (p[1, 1] * r) + 1.) * np.exp(-p[1, 1] * r) +
            p[0, 2] * (2. / (p[1, 2] * r) + 1.) * np.exp(-p[1, 2] * r) +
            p[0, 3] * (2. / (p[1, 3] * r) + 1.) * np.exp(-p[1, 3] * r) +
            p[0, 4] * (2. / (p[1, 4] * r) + 1.) * np.exp(-p[1, 4] * r))


@jit(nopython=True, nogil=True)
def dvdr_lobato(r, p):
    dvdr = - (p[0, 0] * (2. / (p[1, 0] * r ** 2) + 2. / r + p[1, 0]) * np.exp(-p[1, 0] * r) +
              p[0, 1] * (2. / (p[1, 1] * r ** 2) + 2. / r + p[1, 1]) * np.exp(-p[1, 1] * r) +
              p[0, 2] * (2. / (p[1, 2] * r ** 2) + 2. / r + p[1, 2]) * np.exp(-p[1, 2] * r) +
              p[0, 3] * (2. / (p[1, 3] * r ** 2) + 2. / r + p[1, 3]) * np.exp(-p[1, 3] * r) +
              p[0, 4] * (2. / (p[1, 4] * r ** 2) + 2. / r + p[1, 4]) * np.exp(-p[1, 4] * r))

    return dvdr


@jit(nopython=True, nogil=True)
def d2vdr2_lobato(r, p):
    d2vdr2 = (p[0, 0] * (2 * (p[1, 0] * r + 2) / (p[1, 0] * r ** 3) +
                         2 * (p[1, 0] * r + 1) / r ** 2 + p[1, 0] ** 2) * np.exp(-p[1, 0] * r) +
              p[0, 1] * (2 * (p[1, 1] * r + 2) / (p[1, 1] * r ** 3) +
                         2 * (p[1, 1] * r + 1) / r ** 2 + p[1, 1] ** 2) * np.exp(-p[1, 1] * r) +
              p[0, 2] * (2 * (p[1, 2] * r + 2) / (p[1, 2] * r ** 3) +
                         2 * (p[1, 2] * r + 1) / r ** 2 + p[1, 2] ** 2) * np.exp(-p[1, 2] * r) +
              p[0, 3] * (2 * (p[1, 3] * r + 2) / (p[1, 3] * r ** 3) +
                         2 * (p[1, 3] * r + 1) / r ** 2 + p[1, 3] ** 2) * np.exp(-p[1, 3] * r) +
              p[0, 4] * (2 * (p[1, 4] * r + 2) / (p[1, 4] * r ** 3) +
                         2 * (p[1, 4] * r + 1) / r ** 2 + p[1, 4] ** 2) * np.exp(-p[1, 4] * r))

    return d2vdr2


def load_kirkland_parameters(convert=True):
    """Function to load the Kirkland parameters (doi:10.1007/978-1-4419-6533-2)."""
    parameters = {}

    for key, value in load_parameters(_set_path('kirkland.txt')).items():
        a = np.array([value[key] for key in ('a1', 'a2', 'a3')])
        b = np.array([value[key] for key in ('b1', 'b2', 'b3')])
        c = np.array([value[key] for key in ('c1', 'c2', 'c3')])
        d = np.array([value[key] for key in ('d1', 'd2', 'd3')])
        if convert:
            a = np.pi * a
            b = 2. * np.pi * np.sqrt(b)
            c = np.pi ** (3. / 2.) * c / d ** (3. / 2.)
            d = np.pi ** 2 / d

        parameters[key] = np.vstack((a, b, c, d))

    return parameters


@jit(nopython=True, nogil=True)
def kirkland(r, p):
    return (p[0, 0] * np.exp(-p[1, 0] * r) / r + p[2, 0] * np.exp(-p[3, 0] * r ** 2.) +
            p[0, 1] * np.exp(-p[1, 1] * r) / r + p[2, 1] * np.exp(-p[3, 1] * r ** 2.) +
            p[0, 2] * np.exp(-p[1, 2] * r) / r + p[2, 2] * np.exp(-p[3, 2] * r ** 2.))


@jit(nopython=True, nogil=True)
def kirkland_scattering(k, p):
    return (p[0, 0] / (p[1, 0] + k ** 2) + p[2, 0] * np.exp(-p[3, 0] * k ** 2) +
            p[0, 1] / (p[1, 1] + k ** 2) + p[2, 1] * np.exp(-p[3, 1] * k ** 2) +
            p[0, 2] / (p[1, 2] + k ** 2) + p[2, 2] * np.exp(-p[3, 2] * k ** 2))


@jit(nopython=True, nogil=True)
def dvdr_kirkland(r, p):
    dvdr = (- p[0, 0] * (1 / r + p[1, 0]) * np.exp(-p[1, 0] * r) / r -
            2 * p[2, 0] * p[3, 0] * r * np.exp(-p[3, 0] * r ** 2)
            - p[0, 1] * (1 / r + p[1, 1]) * np.exp(-p[1, 1] * r) / r -
            2 * p[2, 1] * p[3, 1] * r * np.exp(-p[3, 1] * r ** 2)
            - p[0, 2] * (1 / r + p[1, 2]) * np.exp(-p[1, 2] * r) / r -
            2 * p[2, 2] * p[3, 2] * r * np.exp(-p[3, 2] * r ** 2))
    return dvdr


def kirkland_projected(r, p):
    v = (2 * p[0, 0] * kn(0, p[1, 0] * r) + np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.exp(-p[3, 0] * r ** 2.) +
         2 * p[0, 1] * kn(0, p[1, 1] * r) + np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.exp(-p[3, 1] * r ** 2.) +
         2 * p[0, 2] * kn(0, p[1, 2] * r) + np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.exp(-p[3, 2] * r ** 2.))
    return v


def kirkland_projected_fourier(k, p):
    f = (4 * np.pi * p[0, 0] / (4 * np.pi ** 2 * k ** 2 + p[1, 0] ** 2) +
         np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.pi / p[3, 0] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 0]) +
         4 * np.pi * p[0, 1] / (4 * np.pi ** 2 * k ** 2 + p[1, 1] ** 2) +
         np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.pi / p[3, 1] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 1]) +
         4 * np.pi * p[0, 2] / (4 * np.pi ** 2 * k ** 2 + p[1, 2] ** 2) +
         np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.pi / p[3, 2] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 2]))
    return f
