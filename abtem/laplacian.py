from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
from numba import stencil, njit

from abtem.potentials.iam import TransmissionFunction
from abtem.core.energy import energy2sigma

if TYPE_CHECKING:
    from abtem.waves import Waves
    from abtem.potentials.iam import PotentialArray


def _get_central_offsets(derivative, accuracy):
    assert accuracy % 2 == 0
    num_central = 2 * math.floor((derivative + 1) / 2) - 1 + accuracy
    num_side = num_central // 2
    offsets = list(range(-num_side, num_side + 1))
    return offsets


def _build_matrix(offsets):
    A = [([1 for _ in offsets])]
    for i in range(1, len(offsets)):
        A.append([j**i for j in offsets])

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

    stencil[len(coefficients) // 2, :] = coefficients
    stencil[:, len(coefficients) // 2] += coefficients
    return stencil


def _laplace_operator_stencil(
    accuracy, prefactor, mode: str = "wrap", dtype=np.complex64
):
    c = _finite_difference_coefficients(2, accuracy)
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

    @njit(parallel=True, fastmath=True)
    def _laplace_stencil(a):
        return stencil_func(a)

    def _apply_boundary(mode, padding):
        pad_width = [(padding,) * 2, (padding,) * 2]

        def stencil_with_boundary(func):
            def func_wrapper(a: np.ndarray):
                a = np.pad(a, pad_width=pad_width, mode=mode)
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

    def _get_new_stencil(self, key):
        wavelength, sampling, thickness = key
        prefactor = 1.0j * wavelength * thickness / (4 * np.pi) / np.prod(sampling)
        return _laplace_operator_stencil(self._accuracy, prefactor, mode="wrap")

    def get_stencil(self, waves: Waves, thickness: float) -> callable:
        key = (
            waves.wavelength,
            waves.sampling,
            thickness,
        )

        # tilt_axes = _get_tilt_axes(waves)
        # tilt_axes_metadata = [waves.ensemble_axes_metadata[i] for i in tilt_axes]
        # if tilt_axes:
        #    key += (copy.deepcopy(tilt_axes_metadata),)

        if key == self._key:
            return self._stencil

        self._stencil = self._get_new_stencil(key)
        self._key = key
        return self._stencil

    def apply(self, waves, thickness):
        laplace_stencil = self.get_stencil(waves, thickness)
        waves._array = laplace_stencil(waves._array)
        return waves


def _multislice_exponential_series(
    waves,
    transmission_function,
    num_terms: int,
    laplace: callable,
):
    temp = laplace(waves) + waves * transmission_function

    waves += temp
    for i in range(2, num_terms + 1):
        temp = (laplace(temp) + temp * transmission_function) / i
        waves += temp
    return waves


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
    laplace: callable,
    max_terms: int = 1,
):
    # if num_terms < 1:
    #     raise ValueError()

    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    if isinstance(potential_slice, TransmissionFunction):
        transmission_function = potential_slice

    else:
        transmission_function = potential_slice.transmission_function(
            energy=waves.energy
        )


    thickness = potential_slice.thickness
    #transmission_function = 1.0j * potential_slice.array[0] * energy2sigma(waves.energy)

    #waves = transmission_function.transmit(waves)

    laplace = laplace.get_stencil(waves, thickness)

    waves._array = _multislice_exponential_series(
        waves._array, max_terms, laplace
    )
    return waves


class DivergedError(Exception):
    def __init__(self, message="the series diverged"):
        super().__init__(message)


class NotConvergedError(Exception):
    def __init__(self, message="the series did not converge"):
        super().__init__(message)


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
#     # #         f"series did not converge to a tolerance of {tolerance} in {max_terms} terms"
#     # #     )
#     #
#     # return waves
