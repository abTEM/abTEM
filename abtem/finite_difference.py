from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.ndimage  # type: ignore
from numba import njit, stencil, cuda  # type: ignore

from abtem.core.backend import get_array_module, get_scipy_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.antialias import antialias_aperture, AntialiasAperture
from abtem.core.fft import fft2_convolve

if TYPE_CHECKING:
    from abtem.potentials.iam import PotentialArray
    from abtem.waves import Waves


fd_coefficients = {
    2: [1.0, -2.0, 1.0],
    4: [
        -0.08333333333333333,
        1.3333333333333333,
        -2.5,
        1.3333333333333333,
        -0.08333333333333333,
    ],
    6: [
        0.011111111111111112,
        -0.15,
        1.5,
        -2.7222222222222223,
        1.5,
        -0.15,
        0.011111111111111112,
    ],
    8: [
        -0.0017857142857142857,
        0.025396825396825397,
        -0.2,
        1.6,
        -2.8472222222222223,
        1.6,
        -0.2,
        0.025396825396825397,
        -0.0017857142857142857,
    ],
    10: [
        0.00031746031746031746,
        -0.00496031746031746,
        0.03968253968253968,
        -0.23809523809523808,
        1.6666666666666667,
        -2.9272222222222224,
        1.6666666666666667,
        -0.23809523809523808,
        0.03968253968253968,
        -0.00496031746031746,
        0.00031746031746031746,
    ],
    12: [
        -6.012506012506013e-05,
        0.001038961038961039,
        -0.008928571428571428,
        0.05291005291005291,
        -0.26785714285714285,
        1.7142857142857142,
        -2.9827777777777778,
        1.7142857142857142,
        -0.26785714285714285,
        0.05291005291005291,
        -0.008928571428571428,
        0.001038961038961039,
        -6.012506012506013e-05,
    ],
    14: [
        1.1892869035726179e-05,
        -0.00022662522662522663,
        0.0021212121212121214,
        -0.013257575757575758,
        0.06481481481481481,
        -0.2916666666666667,
        1.75,
        -3.02359410430839,
        1.75,
        -0.2916666666666667,
        0.06481481481481481,
        -0.013257575757575758,
        0.0021212121212121214,
        -0.00022662522662522663,
        1.1892869035726179e-05,
    ],
    16: [
        -2.428127428127428e-06,
        5.074290788576503e-05,
        -0.000518000518000518,
        0.003480963480963481,
        -0.017676767676767676,
        0.07542087542087542,
        -0.3111111111111111,
        1.7777777777777777,
        -3.05484410430839,
        1.7777777777777777,
        -0.3111111111111111,
        0.07542087542087542,
        -0.017676767676767676,
        0.003480963480963481,
        -0.000518000518000518,
        5.074290788576503e-05,
        -2.428127428127428e-06,
    ],
    18: [
        5.078436450985471e-07,
        -1.1569313039901276e-05,
        0.00012844298558584272,
        -0.0009324009324009324,
        0.005034965034965035,
        -0.022027972027972027,
        0.08484848484848485,
        -0.32727272727272727,
        1.8,
        -3.0795354623330815,
        1.8,
        -0.32727272727272727,
        0.08484848484848485,
        -0.022027972027972027,
        0.005034965034965035,
        -0.0009324009324009324,
        0.00012844298558584272,
        -1.1569313039901276e-05,
        5.078436450985471e-07,
    ],
}


def _build_matrix(offsets: list[int]):
    import sympy  # type: ignore

    """Constructs the equation system matrix for the finite difference coefficients"""
    A = [([1 for _ in offsets])]
    for i in range(1, len(offsets)):
        A.append([j**i for j in offsets])
    return sympy.Matrix(A)


def _build_rhs(offsets: list[int], deriv: int):
    import sympy  # type: ignore

    """The right hand side of the equation system matrix"""
    b = [0 for _ in offsets]
    b[deriv] = math.factorial(deriv)
    return sympy.Matrix(b)


def _calculate_finite_difference_coefficient(derivative: int, accuracy: int = 2):
    import sympy  # type: ignore

    num_central = 2 * math.floor((derivative + 1) / 2) - 1 + accuracy
    num_side = num_central // 2
    offsets = list(range(-num_side, num_side + 1))

    matrix = _build_matrix(offsets)
    rhs = _build_rhs(offsets, derivative)
    coefs = sympy.linsolve((matrix, rhs))
    coefs = np.array([float(coef) for coef in tuple(coefs)[0]])
    return coefs


def finite_difference_coefficients(derivative: int, accuracy: int = 2):
    if accuracy % 2 == 1 or accuracy <= 0:
        raise ValueError("accuracy order must be a positive even integer")

    if derivative < 0:
        raise ValueError("derivative degree must be a positive integer")

    if accuracy <= 18:
        return np.array(fd_coefficients[accuracy])

    return _calculate_finite_difference_coefficient(derivative, accuracy)


def _laplace_stencil_array(accuracy):
    coefficients = finite_difference_coefficients(2, accuracy)
    stencil = np.zeros((len(coefficients),) * 2)

    stencil[len(coefficients) // 2, :] = coefficients
    stencil[:, len(coefficients) // 2] += coefficients
    return stencil


def _laplace_operator_stencil(
    accuracy, prefactor, mode: str = "wrap", dtype=np.complex64, device: str = "cpu"
):
    c = finite_difference_coefficients(2, accuracy)
    c = c * prefactor
    c = c.astype(dtype)
    c = np.roll(c, -(len(c) // 2))
    n = len(c) // 2
    padding = n + 1

    @stencil(neighborhood=((-n, n + 1), (-n, n + 1)))
    def stencil_func(a):
        cumul = dtype(0.0)
        for i in range(-n, n + 1):
            cumul += c[i] * a[i, 0] + c[i] * a[0, i]
        return cumul
    
    @cuda.jit
    def stencil_func_gpu(a, out):
        i, j = cuda.grid(2)
        H, W = a.shape
        if 1 <= i < H - n and 1 <= j < W - n:
            cumul = dtype(0.0)
            for k in range(-n, n + 1):
                cumul += c[k] * a[i + k, j] + c[k] * a[i, j + k]
            out[i, j] = cumul

    @njit(parallel=True, fastmath=True)
    def _laplace_stencil_cpu(a):
        return stencil_func(a)

    def _laplace_stencil_gpu(a):
        xp = get_array_module(a)
        out = xp.zeros_like(a)
        threadsperblock = (16, 16) # ToDo: threadsperblock hardcoded
        blockspergrid_x = math.ceil(a.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(a.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        stencil_func_gpu[blockspergrid, threadsperblock](a, out)
        return out

    def _laplace_stencil(a):
        if device == "cpu":
            return _laplace_stencil_cpu(a)
        elif device == 'gpu':
            return _laplace_stencil_gpu(a)
        else:
            raise ValueError()

    def _apply_boundary(mode, padding):
        pad_width = [(padding,) * 2, (padding,) * 2]

        def stencil_with_boundary(func):
            def func_wrapper(a):
                xp = get_array_module(a)
                a = xp.pad(a, pad_width=pad_width, mode=mode)
                res = func(a)
                return res[padding:-padding, padding:-padding]

            return func_wrapper

        return stencil_with_boundary

    if mode != "none":
        return _apply_boundary(mode="wrap", padding=padding)(_laplace_stencil)
    else:
        return _laplace_stencil

def _laplace_operator_func_slow(accuracy, prefactor):
    stencil = _laplace_stencil_array(accuracy) * prefactor

    def func(array):
        return scipy.ndimage.convolve(array, stencil, mode="wrap")

    return func


class LaplaceOperator:
    def __init__(self, accuracy):
        self._accuracy = accuracy
        self._key = None
        self._stencil = None

    def _get_new_stencil(self, key, device: str = "cpu"):
        wavelength, sampling, thickness = key
        prefactor = 1j * float(wavelength) * float(thickness) / (4 * np.pi) / np.prod(np.array(sampling, dtype=float))
        return _laplace_operator_stencil(self._accuracy, prefactor, mode="wrap", device=device) # currently gpu hardcoded but should be dynamic

    def get_stencil(self, waves: Waves, thickness: float, device: str = "cpu") -> Callable:
        key = (
            waves.wavelength,
            waves.sampling,
            thickness,
        )

        if key == self._key:
            return self._stencil

        self._stencil = self._get_new_stencil(key, device=device)
        self._key = key
        return self._stencil

    def apply(self, waves, thickness):
        laplace_stencil = self.get_stencil(waves, thickness, device=waves.device)
        waves._array = laplace_stencil(waves._array)
        return waves
    



class DivergedError(Exception):
    def __init__(self, message="the multislice exponential series diverged"):
        super().__init__(message)


class NotConvergedError(Exception):
    def __init__(self, message="the series did not converge"):
        super().__init__(message)


def _multislice_exponential_series(
    waves: np.ndarray,
    transmission_function: np.ndarray,
    laplace: Callable,
    wavelength: float,
    thickness: float,
    sampling: tuple[float, float],
    tolerance: float = 1e-16,
    max_terms: int = 300,
    correction: None | str = None,
    order: int = 4,

) -> np.ndarray:
    xp = get_array_module(waves)
    initial_amplitude = xp.abs(waves).sum()

    # kernel = antialias_aperture(
    #     waves.shape, sampling, xp)
    # return aperture.bandlimit(waves)

    if correction == "propagator":
        temp = propagator_corrected_taylor_series(
                waves, order=order, laplace=laplace, wavelength=wavelength, thickness=thickness, sampling=sampling
            ) + waves * transmission_function
    else:
        temp = laplace(waves) + waves * transmission_function

    waves += temp
    # waves = fft2_convolve(waves, kernel)

    for i in range(2, max_terms + 1):
        if correction == "propagator":
            temp = (propagator_corrected_taylor_series(
                temp, order=order, laplace=laplace, wavelength=wavelength, thickness=thickness, sampling=sampling
            ) + temp * transmission_function) / i
        else:
            temp = (laplace(temp) + temp * transmission_function) / i
        waves += temp
        # waves = fft2_convolve(waves, kernel)
        temp_amplitude = xp.abs(temp).sum()
        # print(f"Term {i}, temp amplitude: {temp_amplitude}, ratio: {temp_amplitude / initial_amplitude}, tolerance: {tolerance}")
        if temp_amplitude / initial_amplitude <= tolerance:
            break

        if temp_amplitude > initial_amplitude:
            raise DivergedError()
    else:
        raise NotConvergedError(
            f"series did not converge to a tolerance of {tolerance} in {max_terms}"
            "terms"
        )
    return waves

def propagator_corrected_taylor_series(
        waves: np.ndarray,
        order: int,
        laplace: Callable,
        wavelength: float,
        thickness: float,
        sampling: tuple[float, float]
        ) -> np.ndarray:
    xp = get_array_module(waves)
    if order < 1:
        raise ValueError("order must be a positive integer and at least 2")
    laplace_waves = laplace(waves)
    series = laplace_waves.copy()
    temp = laplace_waves.copy()
    alpha = np.prod(sampling) / (1.0j * thickness)
    for i in range(2, order + 1):
        temp = laplace(temp) * alpha
        series += temp * wavelength ** (i-1) / ((-1) ** (i+1) * 2.0 ** i * np.pi ** (i-1))

    # series = series / 2.0 + laplace_waves / 2.0

    # laplace_waves = laplace(waves)
    # series = laplace_waves.copy()
    # temp = laplace_waves.copy()
    # alpha = np.prod(sampling) * wavelength / (-2.0 * np.pi * 1.0j * thickness)
    # for i in range(2, order + 1):
    #     temp = laplace(temp) * alpha
    #     series += temp

    # series = series / 2.0 + laplace_waves / 2.0
    return series


# def _multislice_exponential_series(
#     waves,
#     num_terms: int,
#     laplace: callable,
# ):
#     temp = laplace(waves)
#
#     waves += temp
#     for i in range(2, num_terms + 1):
#         temp = (laplace(temp)) / i
#         waves += temp
#     return waves


def multislice_step(
    waves: Waves,
    potential_slice: PotentialArray,
    laplace: LaplaceOperator,
    tolerance: float = 1e-16,
    max_terms: int = 300,
    correction: None | str = None,
    order: int = 4,
) -> Waves:
    if max_terms < 1:
        raise ValueError()

    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    # if isinstance(potential_slice, TransmissionFunction):
    #     transmission_function = potential_slice

    # else:
    #     transmission_function = potential_slice.transmission_function(
    #         energy=waves._valid_energy
    #     )

    thickness = potential_slice.thickness
    transmission_function_array = (
        1.0j * potential_slice.array[0] * energy2sigma(waves._valid_energy) 
    )

    # waves = transmission_function.transmit(waves)
    laplace_stencil = laplace.get_stencil(waves, thickness, device=waves.device)

    wavelength = energy2wavelength(waves._valid_energy)

    waves._array = _multislice_exponential_series(
        waves._eager_array,
        transmission_function_array,
        laplace_stencil,
        wavelength,
        thickness,
        waves.sampling,
        tolerance,
        max_terms,
        correction,
        order,
    )
    aperture = AntialiasAperture()
    return aperture.bandlimit(waves)
    # return waves

def printtest():
    print("test")

# def multislice_step(
#     waves,
#     laplace,
#     potential_slice,
#     tolerance: float = 1e-16,
#     # accuracy: int = 4,
#     max_terms: int = 180,
# ) -> np.ndarray:
#
#     if waves.device != potential_slice.device:
#         potential_slice = potential_slice.copy_to_device(device=waves.device)
#
#     if isinstance(potential_slice, TransmissionFunction):
#         transmission_function = potential_slice
#
#     else:
#         transmission_function = potential_slice.transmission_function(
#             energy=waves.energy
#         )
#
#     thickness = transmission_function.slice_thickness[0]
#
#     array = waves.array.copy()
#     transmission_function = transmission_function.array
#
#     array = array * transmission_function[0]
#
#     laplace = laplace.get_stencil(waves, thickness)
#     #waves = transmission_function.transmit(waves)
#
#     temp = laplace(array)
#
#     array += temp
#
#     for i in range(2, max_terms + 1):
#         temp = laplace(temp) / i
#         array += temp
#
#
#     waves._array = array
#     return waves
#
#     # # initial_amplitude = np.abs(wave).sum()
#     # temp = laplace.apply(waves, thickness)
#     #
#     # # if t is not None:
#     # #    temp += t * wave
#     #
#     # waves += temp
#     #
#     # for i in range(2, max_terms + 1):
#     #     temp = laplace.apply(temp, thickness) / i
#     #
#     #     # if t is not None:
#     #     #    temp += t * temp / i
#     #
#     #     waves += temp
#     #
#     #     # temp_amplitude = np.abs(temp).sum()
#     #     # if temp_amplitude / initial_amplitude <= tolerance:
#     #     #     break
#     #     #
#     #     # if temp_amplitude > initial_amplitude:
#     #     #     raise DivergedError()
#     # # else:
#     # #     raise NotConvergedError(
#     # #         f"series did not converge to a tolerance of {tolerance} in {max_terms}
# terms"
#     # #     )
#     #
#     # return waves
