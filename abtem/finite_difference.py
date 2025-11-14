from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.ndimage  # type: ignore
from numba import cuda, njit, stencil  # type: ignore

from abtem.antialias import AntialiasAperture
from abtem.core.backend import get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength

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
    def stencil_func_2d(a):
        cumul = dtype(0.0)
        for i in range(-n, n + 1):
            cumul += c[i] * a[i, 0] + c[i] * a[0, i]
        return cumul

    @njit(parallel=True, fastmath=True)
    def _laplace_stencil_cpu_batch(a):
        out = np.zeros_like(a)
        for m in range(a.shape[0]):
            out[m] = stencil_func_2d(a[m])
        return out

    @cuda.jit
    def stencil_func_gpu_batch(a, out):
        m, i, j = cuda.grid(3)
        M, H, W = a.shape
        if m < M and n <= i < H - n and n <= j < W - n:
            cumul = dtype(0.0)
            for k in range(-n, n + 1):
                cumul += c[k] * a[m, i + k, j] + c[k] * a[m, i, j + k]
            out[m, i, j] = cumul

    def _laplace_stencil_gpu(a):
        xp = get_array_module(a)
        out = xp.zeros_like(a)

        M, H, W = a.shape

        threads_x = 32
        threads_y = 32
        max_threads = 1024
        threads_m = max_threads // (threads_x * threads_y)
        threadsperblock = (threads_m, threads_x, threads_y)

        blockspergrid_m = math.ceil(a.shape[0] / threadsperblock[0])
        blockspergrid_x = math.ceil(a.shape[1] / threadsperblock[1])
        blockspergrid_y = math.ceil(a.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_m, blockspergrid_x, blockspergrid_y)
        stencil_func_gpu_batch[blockspergrid, threadsperblock](a, out)

        return out

    def _laplace_stencil(a):
        # Store original shape and reshape to 3D
        original_shape = a.shape
        if a.ndim == 2:
            a = a.reshape(1, *a.shape)
        elif a.ndim > 3:
            a = a.reshape(-1, *a.shape[-2:])
        elif a.ndim != 3:
            raise ValueError(f"Array must have at least 2 dimensions, got {a.ndim}")

        # Apply stencil
        if device == "cpu":
            result = _laplace_stencil_cpu_batch(a)
        elif device == "gpu":
            result = _laplace_stencil_gpu(a)
        else:
            raise ValueError(f"Unsupported device: {device}")

        # Reshape back to original shape
        return result.reshape(original_shape)

    def _apply_boundary(mode, padding):
        def stencil_with_boundary(func):
            def func_wrapper(a):
                xp = get_array_module(a)

                # Build pad_width for arbitrary dimensions
                # Only pad the last two spatial dimensions
                pad_width = [(0, 0)] * (a.ndim - 2) + [(padding,) * 2, (padding,) * 2]

                # Build slicing to remove padding from last two dimensions
                slicing = tuple([slice(None)] * (a.ndim - 2) +
                               [slice(padding, -padding), slice(padding, -padding)])

                a = xp.pad(a, pad_width=pad_width, mode=mode)
                res = func(a)
                return res[slicing]

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
        prefactor = (
            1j
            * float(wavelength)
            * float(thickness)
            / (4 * np.pi)
            / np.prod(np.array(sampling, dtype=float))
        )
        return _laplace_operator_stencil(
            self._accuracy, prefactor, mode="wrap", device=device
        )

    def get_stencil(
        self, waves: Waves, thickness: float, device: str = "cpu"
    ) -> Callable:
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
    order: int = 1,
):
    xp = get_array_module(waves)
    initial_amplitude = xp.abs(waves).sum()

    temp = (
        propagator_taylor_series(
            waves,
            order=order,
            laplace=laplace,
            wavelength=wavelength,
            thickness=thickness,
            sampling=sampling,
        )
        + waves * transmission_function
    )

    waves += temp

    for i in range(2, max_terms + 1):
        temp = (
            propagator_taylor_series(
                temp,
                order=order,
                laplace=laplace,
                wavelength=wavelength,
                thickness=thickness,
                sampling=sampling,
            )
            + temp * transmission_function
        ) / i

        waves += temp
        temp_amplitude = xp.abs(temp).sum()
        if temp_amplitude / initial_amplitude <= tolerance:
            break

        if temp_amplitude > initial_amplitude:
            raise DivergedError()
    else:
        raise NotConvergedError(
            f"series did not converge to a tolerance of {tolerance} in {max_terms}terms"
        )
    return waves


def propagator_taylor_series(
    waves: np.ndarray,
    order: int,
    laplace: Callable,
    wavelength: float,
    thickness: float,
    sampling: tuple[float, float],
):
    if order < 1:
        raise ValueError("order must be a positive integer and at least 1")

    laplace_waves = laplace(waves)
    if order == 1:
        return laplace_waves

    series = laplace_waves.copy()
    temp = laplace_waves.copy()

    alpha = np.prod(sampling) / (1.0j * thickness)  # removes laplace prefactor
    for i in range(2, order + 1):
        prefactor = (wavelength / (-2.0 * np.pi)) ** (i - 1) * 0.5
        temp = laplace(temp) * alpha
        series += temp * prefactor

    return series


def multislice_step(
    waves: Waves,
    potential_slice: PotentialArray,
    laplace: LaplaceOperator,
    tolerance: float = 1e-16,
    max_terms: int = 300,
    order: int = 1,
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
        order,
    )

    aperture = AntialiasAperture()  # bandlimit to compare with Fourier
    return aperture.bandlimit(waves)
