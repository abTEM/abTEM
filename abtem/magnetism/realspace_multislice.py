import numpy as np
from numba import stencil, njit
import scipy.ndimage

from abtem.core.energy import energy2sigma, energy2wavelength


@njit(parallel=True, fastmath=True)
def anisotropic_5_point_stencil(array_in, array_out=None):
    @stencil
    def _stencil_func(a):
        c3 = np.complex64(4.0)
        return -c3 * a[0, 0] + a[0, 1] + a[-1, 0] + a[1, 0] + a[0, -1]

    if array_out is not None:
        return _stencil_func(array_in, out=array_out)
    else:
        return _stencil_func(array_in)


@njit(parallel=True, fastmath=True)
def isotropic_9_point_stencil(array_in, c=1., array_out=None):
    @stencil
    def _stencil_func(a, c):
        c1 = np.complex128(1.0 / 6.0 * c)
        c2 = np.complex128(2.0 / 3.0 * c)
        c3 = np.complex128(-10.0 / 3.0 * c)
        return (
            c1 * (a[1, 1] + a[1, -1] + a[-1, -1] + a[-1, 1])
            + c2 * (a[1, 0] + a[0, -1] + a[-1, 0] + a[0, 1])
            + c3 * a[0, 0]
        )

    if array_out is not None:
        return _stencil_func(array_in, c, out=array_out)
    else:
        return _stencil_func(array_in, c)


def make_3x3_stencil(c1, c2, c3):
    return np.array([[c1, c2, c1], [c2, c3, c2], [c1, c2, c1]])


def make_5x5_stencil(c1, c2, c3, c4, c5, c6):
    return np.array(
        [
            [c1, c2, c3, c2, c1],
            [c2, c4, c5, c4, c2],
            [c3, c5, c6, c5, c3],
            [c2, c4, c5, c4, c2],
            [c1, c2, c3, c2, c1],
        ]
    )


def make_stencil(template):
    if template == "ani5":
        args = [0, 1, -4]
        size = 3
    elif template == "iso9":
        args = [1 / 6, 2 / 3, -10 / 3]
        size = 3
    elif template == "ani9":
        args = [0, 0, -1 / 12, 0, 4 / 3, -5]
        size = 5
    elif template == "iso17":
        args = [-1 / 120, 0, -1 / 15, 2 / 15, 16 / 15, -9 / 2]
        size = 5
    elif template == "iso21":
        args = [0, -1 / 30, -1 / 60, 4 / 15, 13 / 15, -21 / 5]
        size = 5
    else:
        raise ValueError()

    if size == 3:
        return make_3x3_stencil(*args)
    elif size == 5:
        return make_5x5_stencil(*args)
    else:
        raise RuntimeError


def laplace(array, c):
    stencil = make_stencil("iso21") * c
    return scipy.ndimage.convolve(array, stencil, mode="wrap")



def step(array, potential_slice, sampling, dz, energy, m):
    wavelength = energy2wavelength(energy)
    c = 1.0j * wavelength * dz / (4 * np.pi) / np.prod(sampling)

    transmission_function = potential_slice * 1.0j * energy2sigma(200e3)

    temp = laplace(array, c) + transmission_function * array
    array += temp
    for i in range(2, m + 1):
        temp = (laplace(temp, c) + temp * transmission_function) / i
        array += temp

    return array

# def step(array, potential_slice, sampling, dz, wavelength, m):
#     potential_slice = potential_slice * 1.j * energy2sigma(200e3)

#     temp = potential_slice
#     array += temp
#     for i in range(2, m + 1):
#         temp = temp * potential_slice / i
#         print(i)
#         array += temp

#     return array