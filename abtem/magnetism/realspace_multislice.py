from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
from numba import stencil, njit

from abtem.core.energy import energy2sigma
import math

if TYPE_CHECKING:
    from abtem.waves import Waves
    from abtem.potentials.iam import PotentialArray


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
def isotropic_9_point_stencil(
    array_in, prefactor: float = 1.0, array_out: np.ndarray = None
):
    @stencil
    def _stencil_func(a, c):
        c1 = np.complex128(1.0 / 6.0 * prefactor)
        c2 = np.complex128(2.0 / 3.0 * prefactor)
        c3 = np.complex128(-10.0 / 3.0 * prefactor)
        return (
            c1 * (a[1, 1] + a[1, -1] + a[-1, -1] + a[-1, 1])
            + c2 * (a[1, 0] + a[0, -1] + a[-1, 0] + a[0, 1])
            + c3 * a[0, 0]
        )

    if array_out is not None:
        return _stencil_func(array_in, prefactor, out=array_out)
    else:
        return _stencil_func(array_in, prefactor)


def _get_central_offsets(derivative, accuracy):
    assert accuracy % 2 == 0
    num_central = 2 * math.floor((derivative + 1) / 2) - 1 + accuracy
    num_side = num_central // 2
    offsets = list(range(-num_side, num_side + 1))
    return offsets


def _build_matrix(offsets):
    A = [([1 for _ in offsets])]
    for i in range(1, len(offsets)):
        A.append([j ** i for j in offsets])

    return np.array(A, dtype=float)


def _build_rhs(offsets, derivative):
    b = [0 for _ in offsets]
    b[derivative] = math.factorial(derivative)
    return np.array(b, dtype=float)


def _finite_difference_coefficients(derivative, accuracy):
    offsets = _get_central_offsets(derivative, accuracy)
    A = _build_matrix(offsets)
    b = _build_rhs(offsets, derivative)
    coefs = np.linalg.solve(A, b)
    return coefs


def _laplace_stencil_array(accuracy):
    coefficients = _finite_difference_coefficients(2, accuracy)
    stencil = np.zeros((len(coefficients),) * 2)

    stencil[len(coefficients)//2, :] = coefficients
    stencil[:, len(coefficients) // 2] += coefficients
    return stencil


def _laplace_operator_func(accuracy, prefactor, dtype=np.complex128):
    c = _finite_difference_coefficients(2, accuracy)
    c = c * prefactor
    c = c.astype(dtype)
    c = np.roll(c, -(len(c) // 2))
    n = len(c) // 2

    @stencil(neighborhood=((-n, n + 1), (-n, n + 1)))
    def stencil_func(a):
        cumul = dtype(0.)
        for i in range(-n, n + 1):
            cumul += c[i] * a[i, 0] + c[i] * a[0, i]
        return cumul

    @njit(parallel=True, fastmath=True)
    def _laplace_stencil(a):
        return stencil_func(a)

    return _laplace_stencil


def _laplace_operator_func_slow(accuracy, prefactor):
    stencil = _laplace_stencil_array(accuracy) * prefactor
    def func(array):
        return scipy.ndimage.convolve(array, stencil, mode="wrap")
    return func
    #stencil = make_stencil_array(stencil) * prefactor
    #return scipy.ndimage.convolve(array, stencil, mode="wrap")


def _multislice_exponential_series(
    waves: np.ndarray,
    transmission_function: np.ndarray,
    num_terms: int,
    laplace: callable,
):
    temp = laplace(waves) + waves * transmission_function
    waves += temp
    for i in range(2, num_terms + 1):
        temp = (laplace(temp) + temp * transmission_function) / i
        waves += temp
    return waves


def multislice_step(
    waves: Waves,
    potential_slice: PotentialArray,
    laplace: callable,
    num_terms: int = 1,
):

    if num_terms < 1:
        raise ValueError()

    prefactor = (
        1.0j
        * waves.wavelength
        * potential_slice.thickness
        / (4 * np.pi)
        / np.prod(waves.sampling)
    )


    transmission_function = 1.0j * potential_slice.array[0] * energy2sigma(waves.energy)

    waves._array = _multislice_exponential_series(
        waves.array, transmission_function, num_terms, laplace
    )
    return waves
