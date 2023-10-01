from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
from numba import stencil, njit

from abtem.core.energy import energy2sigma

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


def make_3x3_stencil_array(c1, c2, c3):
    return np.array([[c1, c2, c1], [c2, c3, c2], [c1, c2, c1]])


def make_5x5_stencil_array(c1, c2, c3, c4, c5, c6):
    return np.array(
        [
            [c1, c2, c3, c2, c1],
            [c2, c4, c5, c4, c2],
            [c3, c5, c6, c5, c3],
            [c2, c4, c5, c4, c2],
            [c1, c2, c3, c2, c1],
        ]
    )


def make_7x7_stencil_array(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    return np.array(
        [
            [c1, c2, c3, c4, c3, c2, c1],
            [c2, c5, c6, c7, c6, c5, c2],
            [c3, c6, c8, c9, c8, c6, c3],
            [c4, c7, c9, c10, c9, c7, c4],
            [c3, c6, c8, c9, c8, c6, c3],
            [c2, c5, c6, c7, c6, c5, c2],
            [c1, c2, c3, c4, c3, c2, c1],
        ]
    )

def make_9x9_stencil_array(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    return np.array(
        [
            [c1, c2, c3, c4, c5, c4, c3, c2, c1],
            [c2, c6, c7, c8, c9, c8, c7, c6, c2],
            [c3, c7, c9, c9, c8, c6, c3, c7, c3],
            [c4, c8, c10, c10, c9, c7, c4, c8, c4],
            [c5, c9, c9, c10, c9, c7, c4, c9, c6],
            [c4, c8, c10, c10, c9, c7, c4, c8, c4],
            [c3, c7, c8, c9, c8, c6, c3, c7, c3],
            [c2, c6, c7, c8, c9, c5, c2, c6, c2],
            [c1, c2, c3, c4, c5, c4, c3, c2, c1],
        ]
    )



def make_stencil_array(template):
    if template == "anisotropic_5":
        coefficients = [0, 1, -4]
        size = 3
    elif template == "isotropic_9":
        coefficients = [1 / 6, 2 / 3, -10 / 3]
        size = 3
    elif template == "anisotropic_9":
        coefficients = [0, 0, -1 / 12, 0, 4 / 3, -5]
        size = 5
    elif template == "isotropic_17":
        coefficients = [-1 / 120, 0, -1 / 15, 2 / 15, 16 / 15, -9 / 2]
        size = 5
    elif template == "isotropic_21":
        coefficients = [0, -1 / 30, -1 / 60, 4 / 15, 13 / 15, -21 / 5]
        size = 5
    elif template == "anisotropic_13":
        coefficients = [0, 0, 0, 1 / 90, 0, 0, -3 / 20, 0, 3 / 2, -49 / 18]
        size = 7
    elif template == "anisotropic_13":
        coefficients = [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72]
    else:
        raise ValueError()

    if size == 3:
        return make_3x3_stencil_array(*coefficients)
    elif size == 5:
        return make_5x5_stencil_array(*coefficients)
    elif size == 7:
        return make_7x7_stencil_array(*coefficients)
    else:
        raise RuntimeError


def laplace(array, prefactor, stencil):
    stencil = make_stencil_array(stencil) * prefactor
    return scipy.ndimage.convolve(array, stencil, mode="wrap")


def _multislice_step(
    waves: np.ndarray,
    prefactor: complex,
    transmission_function: np.ndarray,
    num_terms: int,
    stencil: str,
):
    temp = laplace(waves, prefactor, stencil) + transmission_function * waves
    waves += temp
    for i in range(2, num_terms + 1):
        temp = (laplace(temp, prefactor, stencil) + temp * transmission_function) / i
        waves += temp
    return waves


def multislice_step(
    waves: Waves,
    potential_slice: PotentialArray,
    num_terms: int = 1,
    stencil: str = "isotropic_9",
):
    if num_terms < 1:
        raise ValueError()

    wavelength = waves.wavelength

    prefactor = (
        1.0j
        * wavelength
        * potential_slice.thickness
        / (4 * np.pi)
        / np.prod(waves.sampling)
    )

    transmission_function = 1.0j * potential_slice.array[0] * energy2sigma(waves.energy)

    waves._array = _multislice_step(
        waves.array, prefactor, transmission_function, num_terms, stencil
    )
    return waves
