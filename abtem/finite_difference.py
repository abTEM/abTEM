from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import scipy.ndimage  # type: ignore
from numba import njit, prange  # type: ignore

from abtem._lanczos import (
    exponential_of_root,
    inverse_wave_number,
    lanczos_function,
    wave_number,
)
from abtem.antialias import AntialiasAperture
from abtem.core import config
from abtem.core.backend import check_cupy_is_installed, cp, get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.fft import fft2, ifft2
from abtem.core.utils import get_dtype

if TYPE_CHECKING:
    from abtem.potentials.iam import PotentialArray
    from abtem.waves import Waves

import dask.array as da

# 1D second-derivative centered stencils
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
    A = [[1 for _ in offsets]]
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


_LAPLACE_STENCIL_KERNEL = r"""
#include <cupy/complex.cuh>

// Batched 2-D Laplacian finite-difference stencil: for each batch element m,
// out[m, i, j] = sum_k c[k + n] * (a[m, i + k, j] + a[m, i, j + k]) over
// k in [-n, n], evaluated on the interior i, j in [n, H - n) x [n, W - n).
// The boundary is left zero; the caller handles it via wrap padding.
//
// Grid: x covers W (fastest axis, coalesced), y covers H, z strides over the
// batch so any batch size is supported.
//
// Template parameter T is the complex floating-point type.
template<typename T>
__device__ __forceinline__ void laplace_stencil_impl(
    T* __restrict__ out,
    const T* __restrict__ a,
    const T* __restrict__ c,
    const int n,
    const int M, const int H, const int W
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n || i >= H - n || j < n || j >= W - n) return;

    for (int m = blockIdx.z; m < M; m += gridDim.z) {
        const T* am = a + (long long)m * H * W;
        T cumul = T(0);
        for (int k = -n; k <= n; k++) {
            cumul += c[k + n] * (am[(long long)(i + k) * W + j]
                                 + am[(long long)i * W + (j + k)]);
        }
        out[(long long)m * H * W + (long long)i * W + j] = cumul;
    }
}

extern "C" __global__ void laplace_stencil_c64(
    complex<float>* out, const complex<float>* a, const complex<float>* c,
    int n, int M, int H, int W
) {
    laplace_stencil_impl<complex<float> >(out, a, c, n, M, H, W);
}

extern "C" __global__ void laplace_stencil_c128(
    complex<double>* out, const complex<double>* a, const complex<double>* c,
    int n, int M, int H, int W
) {
    laplace_stencil_impl<complex<double> >(out, a, c, n, M, H, W);
}
"""

_laplace_stencil_c64 = None
_laplace_stencil_c128 = None


def _init_laplace_stencil_kernels():
    global _laplace_stencil_c64, _laplace_stencil_c128
    if _laplace_stencil_c64 is None:
        # no --std option: the dialect requirement comes from CuPy's bundled
        # headers, and each backend's default tracks them (CuPy itself pins no
        # dialect); pinning one broke NVRTC when CuPy 14's complex.cuh started
        # requiring C++17 via CCCL
        mod = cp.RawModule(code=_LAPLACE_STENCIL_KERNEL)
        # _laplace_stencil_c64 doubles as the initialized-guard, so it is
        # assigned last: a concurrent thread that observes it non-None is then
        # guaranteed to also observe _laplace_stencil_c128 (redundant module
        # compilation from two racing threads is harmless — CuPy caches it)
        _laplace_stencil_c128 = mod.get_function("laplace_stencil_c128")
        _laplace_stencil_c64 = mod.get_function("laplace_stencil_c64")


@njit(parallel=True, fastmath=True)
def _laplace_stencil_cpu_kernel(a, c, n):
    """The 5-point-per-axis Laplacian stencil of `_laplace_operator_stencil` on
    the CPU, for a stack of 2D arrays. `c` holds the coefficients rolled so that
    c[k] is that of offset k, and only the interior (n from each edge) is written.

    Module level, with the coefficients as arguments, so that Numba compiles it
    once per array type: as a closure over them it was compiled anew -- ~1 s
    with parallel=True -- every time an operator was made, once per dask task
    of a lazy real-space multislice.
    """
    M, H, W = a.shape
    out = a.copy()
    out[:] = 0
    for m in prange(M):
        for i in range(n, H - n):
            for j in range(n, W - n):
                cumul = c[0] * 0
                for k in range(-n, n + 1):
                    cumul += c[k] * a[m, i + k, j] + c[k] * a[m, i, j + k]
                out[m, i, j] = cumul
    return out


def _laplace_operator_stencil(
    accuracy, prefactor, mode: str = "wrap", dtype=None, device: str = "cpu"
):
    if dtype is None:
        dtype = get_dtype(complex=True)
    c = finite_difference_coefficients(2, accuracy)
    c = c * prefactor
    c = c.astype(dtype)
    c = np.roll(c, -(len(c) // 2))
    n = len(c) // 2
    padding = n + 1

    def _laplace_stencil_cpu_batch(a):
        return _laplace_stencil_cpu_kernel(a, c, n)

    if device == "gpu":
        check_cupy_is_installed()
        # the kernel indexes the coefficients in natural order (index k + n for
        # offset k), so undo the roll applied for the CPU stencil above
        c_gpu = cp.asarray(np.roll(c, n))

    def _laplace_stencil_gpu(a):
        xp = get_array_module(a)
        a = xp.ascontiguousarray(a)
        out = xp.zeros_like(a)

        M, H, W = a.shape

        _init_laplace_stencil_kernels()
        if a.dtype == xp.complex128:
            kernel = _laplace_stencil_c128
        elif a.dtype == xp.complex64:
            kernel = _laplace_stencil_c64
        else:
            raise TypeError(
                "the GPU Laplacian stencil requires a complex64 or complex128 "
                f"array, got {a.dtype}"
            )

        threads_x = 16
        threads_y = 16
        block = (threads_x, threads_y, 1)
        grid = (
            math.ceil(W / threads_x),
            math.ceil(H / threads_y),
            min(M, 65535),
        )
        kernel(
            grid,
            block,
            (
                out,
                a,
                c_gpu.astype(a.dtype, copy=False),
                np.int32(n),
                np.int32(M),
                np.int32(H),
                np.int32(W),
            ),
        )

        return out

    def _laplace_stencil_mps(a):
        """The same stencil on Metal, as shifted slices rather than a kernel.

        The sum is separable -- each output is the coefficients applied along
        the row and along the column -- so the whole of it is 2 * (2n + 1)
        strided adds over the interior, which Metal runs as well as a bespoke
        kernel would. That avoids writing and compiling Metal Shading Language
        for an operation torch already expresses.

        Only the interior is written, exactly as the CPU and CUDA stencils do:
        the caller pads by n + 1 beforehand and crops afterwards, so the band
        left at zero here never survives into the result.
        """
        xp = get_array_module(a)

        if a.dtype not in (np.complex64, np.complex128):
            raise TypeError(
                "the GPU Laplacian stencil requires a complex64 or complex128 "
                f"array, got {a.dtype}"
            )

        _, height, width = a.shape
        out = xp.zeros_like(a)

        if height <= 2 * n or width <= 2 * n:
            return out

        rows = slice(n, height - n)
        columns = slice(n, width - n)

        total = out[:, rows, columns]
        for k in range(-n, n + 1):
            # c is rolled for the CPU stencil's negative indexing; keep it
            coefficient = complex(c[k])
            total = total + coefficient * a[:, n + k : height - n + k, columns]
            total = total + coefficient * a[:, rows, n + k : width - n + k]

        out[:, rows, columns] = total
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
        elif device == "mps":
            result = _laplace_stencil_mps(a)
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
                slicing = tuple(
                    [slice(None)] * (a.ndim - 2)
                    + [slice(padding, -padding), slice(padding, -padding)]
                )

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
        """
        Centered finite-difference laplacian operator.

        Operates on complex wave arrays: the stencil coefficients are cast to
        the configured complex dtype, so the stencils are specialized for
        complex64/complex128 input even though the Laplacian itself is a real
        operator.

        Parameters
        ----------
        accuracy: int
            centered finite-difference stencil accuracy
        """
        self._accuracy = accuracy
        self._key = None
        self._stencil = None

    def _get_new_stencil(self, key, device: str = "cpu"):
        wavelength, sampling = key
        prefactor = 1 / np.prod(np.array(sampling, dtype=float))
        return _laplace_operator_stencil(
            self._accuracy,
            prefactor,
            mode="wrap",
            dtype=get_dtype(complex=True),
            device=device,
        )

    def get_stencil(self, waves: Waves, device: str = "cpu") -> Callable:
        """
        Cached method to return finite-difference stencil using specified
        waves parameters, namely wavelength and sampling for the prefactor.

        Parameters
        ----------
        waves: Waves
            Waves object stencil will ultimately be applied to
        device: str, optional
            Device to evaluate stencil on, "cpu" or "gpu"

        Returns
        ----------
        stencil: Callable
            Cached stencil function with waves prefactor, on specified device
        """
        key = (
            energy2wavelength(waves._valid_energy),
            waves.sampling,
        )

        if key == self._key:
            return self._stencil

        self._stencil = self._get_new_stencil(key, device=device)
        self._key = key
        return self._stencil

    def apply(self, waves):
        laplace_stencil = self.get_stencil(waves, device=waves.device)
        waves._array = laplace_stencil(waves._array)
        return waves


class DivergedError(Exception):
    def __init__(self, message="the multislice exponential series diverged"):
        super().__init__(message)


class NotConvergedError(Exception):
    def __init__(self, message="the series did not converge"):
        super().__init__(message)


def _multislice_exponential_series(
    waves: np.ndarray | da.core.Array,
    transmission_function: np.ndarray,
    laplace: Callable,
    wavelength: float,
    thickness: float,
    tolerance: float = 1e-16,
    max_terms: int = 300,
    order: int = 1,
    fully_corrected: bool = False,
):
    xp = get_array_module(waves)
    initial_amplitude = xp.abs(waves).sum()

    if initial_amplitude == 0:
        # Every term of the series is linear in the wave, so a zero wave is
        # returned unchanged; the convergence ratio below would be 0 / 0.
        return waves

    if fully_corrected:
        temp = full_series(
            waves, laplace, transmission_function, order, wavelength, thickness
        )
    else:
        temp = propagator_taylor_series(
            waves,
            order=order,
            laplace=laplace,
            transmission_function=transmission_function,
            wavelength=wavelength,
            thickness=thickness,
        )

    waves += temp

    for i in range(2, max_terms + 1):
        if fully_corrected:
            temp = (
                full_series(
                    temp, laplace, transmission_function, order, wavelength, thickness
                )
                / i
            )
        else:
            temp = (
                propagator_taylor_series(
                    temp,
                    order=order,
                    laplace=laplace,
                    transmission_function=transmission_function,
                    wavelength=wavelength,
                    thickness=thickness,
                )
            ) / i

        waves += temp
        temp_amplitude = xp.abs(temp).sum()
        if temp_amplitude / initial_amplitude <= tolerance:
            break

        if not xp.all(xp.isfinite(temp)) or temp_amplitude > initial_amplitude:
            raise DivergedError()
    else:
        raise NotConvergedError(
            f"series did not converge to a tolerance of {tolerance} in {max_terms}terms"
        )
    return waves


def conventional_operator(
    waves: np.ndarray | da.core.Array,
    laplace: Callable,
    transmission_function: np.ndarray,
    wavelength: float,
):
    """
    Split-step real-space multislice operator used in all higher-order expansions.

    Parameters
    ----------
    waves: Waves
        Waves object to apply multislice operator on
    laplace: Callable
        Fast laplace operator stencil function
    transmission_function: numpy.ndarray,
        Scaled potential slice to multiply incoming waves with
    wavelength: float
        Waves wavelength
    """
    K0 = 1 / wavelength
    return laplace(waves) / (4 * np.pi * K0) + transmission_function * waves


def propagator_taylor_series(
    waves: np.ndarray | da.core.Array,
    order: int,
    laplace: Callable,
    transmission_function: np.ndarray,
    wavelength: float,
    thickness: float,
):
    """
    Taylor series expansion of the propagator term in the MS equation.
    Eq.(8) in Ultramicroscopy 134 (2013) 135-143.
    """
    if order < 1:
        raise ValueError("order must be a positive integer and at least 1")

    if order == 1:
        return (
            conventional_operator(waves, laplace, transmission_function, wavelength)
            * 1.0j
            * thickness
        )

    K0 = 1 / wavelength
    laplace_waves = laplace(waves) / (4 * np.pi * K0)
    series = laplace_waves.copy()
    temp = laplace_waves.copy()

    for i in range(2, order + 1):
        prefactor = (wavelength / (-2.0 * np.pi)) ** (i - 1) * 0.5
        temp = laplace(temp) / (4 * np.pi * K0)
        series += temp * prefactor

    return (series + waves * transmission_function) * 1.0j * thickness


def full_series(
    waves: np.ndarray | da.core.Array,
    laplace: Callable,
    transmission_function: np.ndarray,
    order: int,
    wavelength: float,
    thickness: float,
    override_prefactor: list[float] = [],
):
    """
    Full Taylor series expansion of the MS Eq.(14) in Ultramicrscopy 134 (2013) 135-143.
    override_prefactor used in backscatter call, Eq. (13) in Micron 190 (2025) 103778.
    """
    series = conventional_operator(waves, laplace, transmission_function, wavelength)
    temp = series.copy()
    for i in range(2, order + 1):
        if override_prefactor:
            prefactor = override_prefactor[
                i - 1
            ]  # Note that the first prefactor always gets skipped and is always 1
        else:
            prefactor = (wavelength / (-2.0 * np.pi)) ** (i - 1) * 0.5
        temp = conventional_operator(temp, laplace, transmission_function, wavelength)
        series += temp * prefactor
    return series * 1.0j * thickness


def _member_tilts(waves: Waves) -> np.ndarray:
    """The total tilt [mrad] of every member of the ensemble of ``waves``: the base tilt
    plus the value of each tilt axis, in the order of the axes. Shape (..., 2)."""
    axes = waves.ensemble_axes_metadata
    shape = waves.shape[:-2]
    tilts = np.broadcast_to(
        np.asarray(waves.base_tilt, dtype=float), shape + (2,)
    ).copy()
    for i, axis in enumerate(axes):
        if hasattr(axis, "tilt"):
            values = np.asarray(axis.tilt, dtype=float).reshape(-1, 2)
            extent = [1] * len(axes) + [2]
            extent[i] = len(values)
            tilts += values.reshape(extent)
    return tilts


def _propagating_band(waves: Waves, propagating_cutoff: float, shift, xp):
    """Mask of the Fourier components that the exact root keeps: the propagating band
    ``|q + k_t| < propagating_cutoff / wavelength`` around the in-plane wave vector of
    a tilted wave (``shift`` = k_t), within the antialiasing aperture ``|q|``."""
    nx, ny = waves._valid_gpts
    dx, dy = waves._valid_sampling
    qx = xp.asarray(np.fft.fftfreq(nx, dx))[:, None]
    qy = xp.asarray(np.fft.fftfreq(ny, dy))[None, :]
    antialias = config.get("antialias.cutoff") / (2 * max(dx, dy))
    band = propagating_cutoff / energy2wavelength(waves._valid_energy)
    return ((qx + shift[0]) ** 2 + (qy + shift[1]) ** 2 < band**2) & (
        qx**2 + qy**2 < antialias**2
    )


def _gradient(laplace: LaplaceOperator, sampling, xp):
    """The centred finite-difference gradient (d/dx, d/dy) of the accuracy of
    ``laplace``, for arrays (..., nx, ny) with periodic boundaries."""
    coefficients = _calculate_finite_difference_coefficient(1, laplace._accuracy)
    half = len(coefficients) // 2

    def derivative(array, axis, step):
        return (
            sum(
                float(coefficients[k + half]) * xp.roll(array, -k, axis=axis)
                for k in range(-half, half + 1)
                if coefficients[k + half] != 0.0
            )
            / step
        )

    return (
        lambda a: derivative(a, -2, sampling[0]),
        lambda a: derivative(a, -1, sampling[1]),
    )


def _exact_root_step(
    waves: Waves,
    potential_slice: PotentialArray,
    next_slice: PotentialArray | None,
    laplace: LaplaceOperator,
    propagating_cutoff: float,
    tolerance: float,
    max_iterations: int,
) -> Sequence[Waves]:
    """A multislice step with the exact one-way root of the full operator, and the
    backscattered wave of the next slice, by Lanczos iterations (no Taylor series).

    A tilted wave travels along (tan x, tan y, 1) and, as in multislice, is stored
    without its carrier exp(2 pi i (k_t . r + k_zt z)): with K_t = 2 pi k_t the
    operator acting on the stored wave is (laplacian + 2 i K_t . gradient - |K_t|^2) /
    (2 k) + sigma V, on the band around -k_t, and the step removes the phase of k_zt."""
    xp = get_array_module(waves.array)

    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)
        if next_slice is not None:
            next_slice = next_slice.copy_to_device(device=waves.device)

    thickness = potential_slice.thickness
    sigma = energy2sigma(waves._valid_energy)
    wavelength = energy2wavelength(waves._valid_energy)
    k = 2 * np.pi / wavelength
    laplace_stencil = laplace.get_stencil(waves, device=waves.device)
    gradient_x, gradient_y = _gradient(laplace, waves._valid_sampling, xp)
    potentials = [potential_slice.array[0] * sigma / thickness]
    if next_slice is not None:
        potentials.append(next_slice.array[0] * sigma / thickness)

    def advance(batch, tilt):
        """The forward and the backscattered wave of a batch of one tilt."""
        tangent = np.tan(np.asarray(tilt, dtype=float) / 1e3)
        norm = np.sqrt(1.0 + np.sum(tangent**2))
        shift = tangent / (wavelength * norm)  # k_t [1/Å]
        reference = k / norm  # k_zt
        momentum = 2 * np.pi * shift  # K_t
        tilted = bool(np.any(shift))
        band = _propagating_band(waves, propagating_cutoff, shift, xp)

        def project(array):
            return ifft2(fft2(array) * band)

        def operator(potential, stacked=False):
            """The operator of a potential, or of a stack of two (2, 1, nx, ny), for a
            batch made of two halves, one for each potential."""

            def apply(a):
                kinetic = laplace_stencil(a)
                if tilted:
                    kinetic = (
                        kinetic
                        + 2j
                        * (momentum[0] * gradient_x(a) + momentum[1] * gradient_y(a))
                        - float(np.sum(momentum**2)) * a
                    )
                if stacked:
                    local = (a.reshape((2, -1) + a.shape[1:]) * potential).reshape(
                        a.shape
                    )
                else:
                    local = potential * a
                return project(kinetic / (2 * k) + local)

            return apply

        # the backscattered component is a correction of a few percent: its functions
        # are converged to at most 1e-5, which changes the intensity of a beam by
        # less than 1e-4 of itself and saves about 40 % of the iterations
        correction_tolerance = max(tolerance, 1e-5)
        current = operator(potentials[0])
        batch = lanczos_function(
            project(batch),
            current,
            exponential_of_root(k, thickness, reference),
            tolerance,
            max_iterations,
            xp,
        )

        backscatter = xp.zeros_like(batch)
        if next_slice is not None:
            following = operator(potentials[1])
            # B = (k_next - k_current) / (2 k_next), of the paper: Eq. 7 in Micron 190
            # (2025) 103778 with the operators of both slices
            # (the two wave numbers act on the same wave: one batch of two halves, one
            # iteration, instead of two, which is what a GPU, not the arithmetic, pays)
            members = batch.shape[0]
            both = lanczos_function(
                xp.concatenate([batch, batch]),
                operator(xp.stack([potentials[1], potentials[0]])[:, None], True),
                wave_number(k),
                correction_tolerance,
                max_iterations,
                xp,
            )
            difference = both[:members] - both[members:]
            backscatter = 0.5 * lanczos_function(
                difference,
                following,
                inverse_wave_number(k),
                correction_tolerance,
                max_iterations,
                xp,
            )
            batch = batch - backscatter
        return batch, backscatter

    shape = waves._array.shape
    members = waves._array.reshape((-1,) + shape[-2:])
    # the members of an ensemble of tilts have a band and an operator each
    unique, inverse = np.unique(
        _member_tilts(waves).reshape(-1, 2), axis=0, return_inverse=True
    )
    if len(unique) == 1:
        forward, backscatter = advance(members, tuple(unique[0]))
    else:
        forward, backscatter = xp.empty_like(members), xp.empty_like(members)
        for number, tilt in enumerate(unique):
            group = xp.asarray(np.flatnonzero(inverse.ravel() == number))
            forward[group], backscatter[group] = advance(members[group], tuple(tilt))

    waves._array = forward.reshape(shape)
    kwargs = waves._copy_kwargs(exclude=("array",))
    return waves, waves.__class__(backscatter.reshape(shape), **kwargs)


def multislice_step(
    waves: Waves,
    potential_slice: PotentialArray,
    next_slice: PotentialArray | None,
    laplace: LaplaceOperator,
    tolerance: float = 1e-16,
    max_terms: int = 300,
    order: int | str = 1,
    fully_corrected: bool = False,
    propagating_cutoff: float = 0.95,
    lanczos_tolerance: float = 1e-8,
    max_iterations: int = 60,
) -> Waves | Sequence[Waves]:
    """
    Performs a single multislice step.

    Parameters
    ----------
    waves: Waves
        Waves object to apply multislice operator on
    potential_slice: PotentialArray
        PotentialArray of slice to be used in transmission operator
    next_slice: PotentialArray | None
        PotentialArray of next slice (used to 'look ahead' for backscattered waves)
    laplace: Callable
        Fast laplace operator stencil function
    tolerance: float
        Convergence tolerance for exponent Taylor series amplitude
    max_terms: int
        Maximum terms in exponent Taylor series
    order: int
        Multislice operator expansion order
    fully_corrected: bool
        If True, transmission and propagator operators are expanded to specified order.
        If next_slice is not None, the backscattered wave component is also computed.
    propagating_cutoff, lanczos_tolerance, max_iterations
        For ``order="exact"``, which evaluates the root of the full operator by a
        Lanczos iteration instead of a series (it requires ``fully_corrected``): the
        propagating band, the convergence of the iteration and its maximum length.

    """
    if order == "exact":
        if not fully_corrected:
            raise ValueError('order="exact" requires the full expansion scope')
        return _exact_root_step(
            waves,
            potential_slice,
            next_slice,
            laplace,
            propagating_cutoff,
            lanczos_tolerance,
            max_iterations,
        )

    xp = get_array_module(waves.array)
    if max_terms < 1:
        raise ValueError()

    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)
        if next_slice is not None:
            next_slice = next_slice.copy_to_device(device=waves.device)

    thickness = potential_slice.thickness
    transmission_function_array = (
        potential_slice.array[0] * energy2sigma(waves._valid_energy) / thickness
    )
    if fully_corrected and next_slice is not None:
        transmission_function_array_next_slice = (
            next_slice.array[0] * energy2sigma(waves._valid_energy) / thickness
        )

    laplace_stencil = laplace.get_stencil(waves, device=waves.device)

    wavelength = energy2wavelength(waves._valid_energy)

    # Forward scattering term
    waves._array = _multislice_exponential_series(
        waves._array,
        transmission_function_array,
        laplace_stencil,
        wavelength,
        thickness,
        tolerance,
        max_terms,
        order,
        fully_corrected,
    )

    # Bandlimit before the correction term to suppress high-k content that
    # would be amplified by repeated conventional_operator applications at
    # low energies (large 1/K0) or after many slices (channeling build-up).
    aperture = AntialiasAperture()
    waves = aperture.bandlimit(waves)

    # Correction-term contributions
    backscatter = xp.zeros_like(waves._array)

    if fully_corrected and next_slice is not None:
        # constants and prefactors
        K0 = 1 / wavelength

        # Eq. 7 in Micron 190 (2025) 103778.
        backscatter = (
            1
            / (2 * np.pi * 1.0j * thickness)
            * (
                full_series(
                    waves._array,
                    laplace_stencil,
                    transmission_function_array_next_slice,
                    order,
                    wavelength,
                    thickness,
                )
                - full_series(
                    waves._array,
                    laplace_stencil,
                    transmission_function_array,
                    order,
                    wavelength,
                    thickness,
                )
            )
        )

        # 1/k series with custom prefactors
        prefactors = [1]
        for i in range(1, order + 1):
            prefactors.append(prefactors[-1] * (1 - 2 * i) / (2 * i))
        for i in range(len(prefactors)):
            prefactors[i] = prefactors[i] / (1.0j * thickness) / (np.pi * K0) ** i

        backscatter *= (
            1
            / (2 * K0)
            * (
                1
                + full_series(
                    waves._array,
                    laplace_stencil,
                    transmission_function_array_next_slice,
                    order,
                    wavelength,
                    thickness,
                    override_prefactor=prefactors,
                )
            )
        )

        if not xp.all(xp.isfinite(backscatter)):
            raise DivergedError(
                "overflow in correction-term computation; try using a smaller "
                "slice_thickness or a lower expansion order"
            )

        # Eq.10 in Micron 190 (2025) 103778.
        waves._array = waves._array - backscatter

    if fully_corrected:
        kwargs = waves._copy_kwargs(exclude=("array",))
        backscatter_waves = waves.__class__(backscatter, **kwargs)
        return waves, aperture.bandlimit(backscatter_waves)

    return waves
