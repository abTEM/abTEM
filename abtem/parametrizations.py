import csv
import os

import numpy as np
from numba import jit, prange


def load_parameters(filename):
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


def convert_lobato(parameters):
    a = np.array([parameters[key] for key in ('a1', 'a2', 'a3', 'a4', 'a5')]).astype(np.float32)
    b = np.array([parameters[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5')]).astype(np.float32)
    a = np.float32(np.pi) ** 2 * a / b ** (3 / 2.)
    b = 2 * np.float32(np.pi) / np.sqrt(b)
    return (a, b)


@jit(nopython=True, nogil=True)
def lobato_scattering(k2, a, b):
    return ((a[0] * (2. + b[0] * k2) / (1. + b[0] * k2) ** 2) +
            (a[1] * (2. + b[1] * k2) / (1. + b[1] * k2) ** 2) +
            (a[2] * (2. + b[2] * k2) / (1. + b[2] * k2) ** 2) +
            (a[3] * (2. + b[3] * k2) / (1. + b[3] * k2) ** 2) +
            (a[4] * (2. + b[4] * k2) / (1. + b[4] * k2) ** 2))


@jit(nopython=True, nogil=True)
def lobato(r, a, b):
    return (a[0] * (2. / (b[0] * r) + 1.) * np.exp(-b[0] * r) +
            a[1] * (2. / (b[1] * r) + 1.) * np.exp(-b[1] * r) +
            a[2] * (2. / (b[2] * r) + 1.) * np.exp(-b[2] * r) +
            a[3] * (2. / (b[3] * r) + 1.) * np.exp(-b[3] * r) +
            a[4] * (2. / (b[4] * r) + 1.) * np.exp(-b[4] * r))


@jit(nopython=True, nogil=True)
def dvdr_lobato(r, a, b):
    dvdr = - (a[0] * (2. / (b[0] * r ** 2) + 2. / r + b[0]) * np.exp(-b[0] * r) +
              a[1] * (2. / (b[1] * r ** 2) + 2. / r + b[1]) * np.exp(-b[1] * r) +
              a[2] * (2. / (b[2] * r ** 2) + 2. / r + b[2]) * np.exp(-b[2] * r) +
              a[3] * (2. / (b[3] * r ** 2) + 2. / r + b[3]) * np.exp(-b[3] * r) +
              a[4] * (2. / (b[4] * r ** 2) + 2. / r + b[4]) * np.exp(-b[4] * r))

    return dvdr


@jit(nopython=True, nogil=True, fastmath=True)
def lobato_soft(r, r_cut, v_cut, dvdr_cut, a, b):
    v = (a[0] * (2. / (b[0] * r) + 1.) * np.exp(-b[0] * r) +
         a[1] * (2. / (b[1] * r) + 1.) * np.exp(-b[1] * r) +
         a[2] * (2. / (b[2] * r) + 1.) * np.exp(-b[2] * r) +
         a[3] * (2. / (b[3] * r) + 1.) * np.exp(-b[3] * r) +
         a[4] * (2. / (b[4] * r) + 1.) * np.exp(-b[4] * r))

    return v - v_cut - (r - r_cut) * dvdr_cut


@jit(nopython=True, nogil=True, parallel=True)
def lobato_projected_finite(r, r_cut, v_cut, dvdr_cut, z0, z1, num_samples, a, b):
    projected = np.zeros((len(z0), len(r)), dtype=np.float32)

    for i in prange(z0.shape[0]):
        xk = np.linspace(z0[i], z1[i], num_samples)
        wk = (z1[i] - z0[i]) / np.float32(num_samples)

        for j in range(len(r)):
            rxy = np.sqrt(r[j] ** np.float32(2.) + xk ** np.float32(2.))
            rxy[rxy > r_cut] = r_cut
            projected[i, j] = np.sum(lobato_soft(rxy, r_cut, v_cut, dvdr_cut, a, b), axis=0) * wk

    return projected


def convert_kirkland(parameters):
    a = np.array([parameters[key] for key in ('a1', 'a2', 'a3')])
    b = np.array([parameters[key] for key in ('b1', 'b2', 'b3')])
    c = np.array([parameters[key] for key in ('c1', 'c2', 'c3')])
    d = np.array([parameters[key] for key in ('d1', 'd2', 'd3')])
    a = np.pi * a
    b = 2. * np.pi * np.sqrt(b)
    c = np.pi ** (3. / 2.) * c / d ** (3. / 2.)
    d = np.pi ** 2 / d
    return a, b, c, d


@jit(nopython=True, nogil=True)
def kirkland(r, a, b, c, d):
    return (a[0] * np.exp(-b[0] * r) / r + c[0] * np.exp(-d[0] * r ** 2.) +
            a[1] * np.exp(-b[1] * r) / r + c[1] * np.exp(-d[1] * r ** 2.) +
            a[2] * np.exp(-b[2] * r) / r + c[2] * np.exp(-d[2] * r ** 2.))


@jit(nopython=True, nogil=True)
def dvdr_kirkland(r, a, b, c, d):
    dvdr = (- a[0] * (1 / r + b[0]) * np.exp(-b[0] * r) / r -
            2 * c[0] * d[0] * r * np.exp(-d[0] * r ** 2)
            - a[1] * (1 / r + b[1]) * np.exp(-b[1] * r) / r -
            2 * c[1] * d[1] * r * np.exp(-d[1] * r ** 2)
            - a[2] * (1 / r + b[2]) * np.exp(-b[2] * r) / r -
            2 * c[2] * d[2] * r * np.exp(-d[2] * r ** 2))
    return dvdr


@jit(nopython=True, nogil=True, fastmath=True)
def kirkland_soft(r, r_cut, v_cut, dvdr_cut, a, b, c, d):
    v = (a[0] * np.exp(-b[0] * r) / r + c[0] * np.exp(-d[0] * r ** 2.) +
         a[1] * np.exp(-b[1] * r) / r + c[1] * np.exp(-d[1] * r ** 2.) +
         a[2] * np.exp(-b[2] * r) / r + c[2] * np.exp(-d[2] * r ** 2.))

    return v - v_cut - (r - r_cut) * dvdr_cut


@jit(nopython=True, nogil=True, parallel=True)
def kirkland_projected_finite(r, r_cut, v_cut, dvdr_cut, z0, z1, num_samples, a, b, c, d):
    projected = np.zeros((len(z0), len(r)))

    for i in prange(z0.shape[0]):
        xk = np.linspace(z0[i], z1[i], num_samples)
        wk = (z1[i] - z0[i]) / num_samples

        for j in range(len(r)):
            rxy = np.sqrt(r[j] ** 2 + xk ** 2)
            rxy[rxy > r_cut] = r_cut
            projected[i, j] = np.sum(kirkland_soft(rxy, r_cut, v_cut, dvdr_cut, a, b, c, d), axis=0) * wk

    return projected


