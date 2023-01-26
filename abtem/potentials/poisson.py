from numbers import Number

import numpy as np
from ase import units
from abtem.potentials.gpaw import unpack2
from scipy.special import sph_harm


def spherical_coordinates(box, gpts, origin):
    xyz = tuple(
        np.linspace(-o, L - o, num=n, endpoint=False)
        for L, n, o in zip(box, gpts, origin)
    )
    x, y, z = np.meshgrid(*xyz, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.zeros_like(r)
    theta[r != 0.0] = np.arccos(z[r != 0.0] / r[r != 0.0])
    phi = np.arctan2(y, x) + np.pi
    return r, theta, phi


def spline_orders(splines):
    new_splines = {}
    index = 0
    for i, spline in enumerate(splines):
        for k in range(-spline.l, spline.l + 1):
            new_splines[index] = spline, spline.l, k
            index += 1
    return new_splines


# def Y(order, degree, theta, phi):
#     if order > 0:
#         return (1 / np.sqrt(2) * (
#                     sph_harm(order, degree, theta, phi) + (-1) ** order * sph_harm(-order, degree, theta, phi))).real
#     elif order == 0:
#         return sph_harm(order, degree, theta, phi).real
#     else:
#         return (1 / (np.sqrt(2) * 1.j) * (
#                     sph_harm(-order, degree, theta, phi) - (-1) ** order * sph_harm(order, degree, theta, phi))).real


def real_sph_harm(order, degree, theta, phi):
    if order < 0:
        return (
            np.sqrt(2)
            * (-1) ** order
            * sph_harm(np.abs(order), degree, theta, phi).imag
        )
    elif order == 0:
        return sph_harm(order, degree, theta, phi).real
    else:
        return np.sqrt(2) * (-1) ** order * sph_harm(order, degree, theta, phi).imag


def unpack_density_matrix(packed_density_matrix):
    return unpack2(packed_density_matrix)


def sum_spherical_basis_functions(splines, density_matrix, r, theta, phi):
    density = np.zeros_like(r)

    r = r / units.Bohr
    splines = spline_orders(splines)

    for (i, j) in np.ndindex(density_matrix.shape):
        spline_i, degree_i, order_i = splines[i]
        spline_j, degree_j, order_j = splines[j]
        density_element = density_matrix[i, j]

        if np.abs(density_element) < 1e-8:
            continue

        density += (
            density_element
            * spline_i.map(r)
            * spline_j.map(r)
            * r ** (degree_i + degree_j)
            * real_sph_harm(order_i, degree_i, phi, theta)
            * real_sph_harm(order_j, degree_j, phi, theta)
        )
    return density


def sum_radial_basis_functions(splines, scales, r):
    density = np.zeros_like(r)

    r = r / units.Bohr

    if isinstance(scales, Number):
        scales = (scales,) * len(splines)

    for spline, scale in zip(splines, scales):
        density += scale * spline.map(r) * real_sph_harm(0, 0, 0.0, 0.0)

    return density
