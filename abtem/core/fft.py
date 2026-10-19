"""Module for handling Fourier transforms and convolution in *ab*TEM."""

import warnings
from typing import Tuple, TypeVar, overload

import dask.array as da
import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore

from abtem.core import config
from abtem.core.backend import check_cupy_is_installed, get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.grid import spatial_frequencies
from abtem.core.utils import get_dtype

try:
    import pyfftw  # type: ignore
except (ModuleNotFoundError, ImportError):
    pyfftw = None

try:
    import mkl_fft  # type: ignore
except ModuleNotFoundError:
    mkl_fft = None

try:
    import cupy as cp  # type: ignore
except ModuleNotFoundError:
    cp = None
except ImportError:
    if config.get("device") == "gpu":
        warnings.warn(
            "The CuPy library could not be imported. Please check your installation, or"
            "change your configuration to use CPU."
        )
    cp = None


def _raise_fft_lib_not_present(lib_name: str):
    raise RuntimeError(
        f"FFT library {lib_name} not present. Install this package or change the FFT"
        "library in your configuration."
    )


def _fft_name_to_fftw_direction(name: str) -> str:
    if name[0] == "i":
        direction = "FFTW_BACKWARD"
    else:
        direction = "FFTW_FORWARD"
    return direction


def _fftw_aligned_copy(array: np.ndarray) -> np.ndarray:
    """A C-contiguous copy of ``array`` aligned for FFTW's SIMD codelets."""
    copy = pyfftw.empty_aligned(array.shape, dtype=array.dtype)
    copy[...] = array
    return copy


def _fftw_aligned(array: np.ndarray) -> np.ndarray:
    """
    ``array`` in the one layout the FFTW plans in this module are made for.

    pyfftw quietly adds ``FFTW_UNALIGNED`` to a plan whose buffer is not
    ``pyfftw.simd_alignment``-aligned, and an unaligned plan selects different
    codelets that round differently (~1e-7 relative, the order of float32
    epsilon). That alignment is 32 bytes on x86 while malloc only guarantees 16,
    so planning for whatever buffer arrives would make the result depend on
    where malloc happened to place it. Every buffer is instead brought to an
    aligned, C-contiguous layout -- by copying when it is not in it already,
    which changes no arithmetic -- so that it always gets the same plan.

    A copy is a new buffer, so a caller passing ``overwrite_x=True`` must use the
    returned result rather than expect its own array to hold it.
    """
    if array.flags.c_contiguous and pyfftw.is_byte_aligned(array):
        return array
    return _fftw_aligned_copy(array)


def _new_fftw_object(
    array: np.ndarray,
    name: str,
    flags: tuple[str, ...] = (),
    axes: tuple[int, ...] = (-2, -1),
):
    # Planning overwrites the buffer, hence the scratch array. It must be laid out
    # like `array` and as aligned (see `_fftw_aligned`), both for
    # `update_arrays` to accept `array` and for the wisdom it leaves behind to be
    # the wisdom for `array`'s plan.
    dummy = pyfftw.zeros_aligned(array.shape, dtype=array.dtype)

    direction = _fft_name_to_fftw_direction(name)

    fftw_object = pyfftw.FFTW(
        dummy,
        dummy,
        axes=axes,
        direction=direction,
        threads=config.get("fftw.threads"),
        flags=(config.get("fftw.planning_effort"),) + flags,
        planning_timelimit=config.get("fftw.planning_timelimit"),
    )

    fftw_object.update_arrays(array, array)

    return fftw_object


class CachedFFTWConvolution:
    def __init__(self):
        self._fftw_objects = None
        self._shape = None

    def __call__(
        self, array: np.ndarray, kernel: np.ndarray, overwrite_x: bool
    ) -> np.ndarray:
        array = _fftw_aligned(array) if overwrite_x else _fftw_aligned_copy(array)

        if array.shape != self._shape:
            self._fftw_objects = None

        if self._fftw_objects is None:
            fftw_objects = {
                name: _new_fftw_object(array, name=name) for name in ("ifft2", "fft2")
            }
            self._fftw_objects = fftw_objects

        for fftw_object in self._fftw_objects.values():
            fftw_object.update_arrays(array, array)

        array = self._fftw_objects["fft2"]()
        array *= kernel
        array = self._fftw_objects["ifft2"]()
        return array


def get_fftw_object(
    array: np.ndarray,
    name: str,
    allow_new_wisdom: bool = True,
    overwrite_x: bool = False,
    axes: tuple[int, ...] = (-2, -1),
):
    """
    Get an in-place pyfftw object for a given array and a given FFT function.
    The plan is made from FFTW wisdom, which is created and kept the first time a
    layout is seen.

    The object transforms ``array`` itself if it is aligned and C-contiguous, and
    otherwise an aligned copy of it (see ``_fftw_aligned``), so the result must be
    taken from calling the object, not read back from ``array``.

    Parameters
    ----------
    array : np.ndarray
        Array to create the FFT object for.
    name : str
        Name of the FFT function.
    allow_new_wisdom : bool, optional
        Allow new wisdom to be created.
    overwrite_x : bool, optional
        Allow the input array to be overwritten.
    axes : tuple of int, optional
        Axes over which to compute the FFT.

    Returns
    -------
    pyfftw.FFTW
        FFTW object.
    """
    array = _fftw_aligned(array)

    direction = _fft_name_to_fftw_direction(name)

    flags: tuple[str, ...] = (config.get("fftw.planning_effort"),)
    if overwrite_x:
        flags += ("FFTW_DESTROY_INPUT",)

    try:
        fftw = pyfftw.FFTW(
            array,
            array,
            axes=axes,
            direction=direction,
            threads=config.get("fftw.threads"),
            flags=flags + ("FFTW_WISDOM_ONLY",),
        )
    except RuntimeError as e:
        if str(e) != "No FFTW wisdom is known for this plan.":
            raise

        if not allow_new_wisdom and config.get("fftw.allow_fallback"):
            return getattr(pyfftw.builders, name)(array, axes=axes)

        elif not allow_new_wisdom:
            raise

        _new_fftw_object(array, name, flags=flags, axes=axes)

        return get_fftw_object(
            array, name, allow_new_wisdom=False, overwrite_x=overwrite_x, axes=axes
        )

    return fftw


def _mkl_fft_dispatch(
    x: np.ndarray, func_name: str, overwrite_x: bool, **kwargs
) -> np.ndarray:
    if mkl_fft is None:
        _raise_fft_lib_not_present("mkl_fft")

    with threadpool_limits(limits=config.get("mkl.threads")):
        return getattr(mkl_fft, func_name)(x, overwrite_x=overwrite_x, **kwargs)


def _fftw_dispatch(
    x: np.ndarray, func_name: str, overwrite_x: bool, **kwargs
) -> np.ndarray:
    if pyfftw is None:
        _raise_fft_lib_not_present("pyfftw")

    if not overwrite_x:
        # the copy is made aligned, so `get_fftw_object` need not copy again
        x = _fftw_aligned_copy(x)

    return get_fftw_object(x, func_name, overwrite_x=overwrite_x, **kwargs)()


U = TypeVar("U", np.ndarray, da.core.Array)


def _fft_dispatch(
    x: U,
    func_name: str,
    overwrite_x: bool = False,
    **kwargs: dict,
) -> U:
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        if config.get("fft") == "mkl":
            return _mkl_fft_dispatch(x, func_name, overwrite_x, **kwargs)
        elif config.get("fft") == "fftw":
            return _fftw_dispatch(x, func_name, overwrite_x, **kwargs)
        elif config.get("fft") == "numpy":
            return getattr(np.fft, func_name)(x, **kwargs)
        else:
            raise RuntimeError()

    if isinstance(x, da.core.Array):
        return da.map_blocks(
            _fft_dispatch,
            x,
            func_name=func_name,
            overwrite_x=overwrite_x,
            **kwargs,
            meta=xp.array((), dtype=get_dtype(complex=True)),
        )

    check_cupy_is_installed()  # type: ignore

    if isinstance(x, cp.ndarray):
        return getattr(cp.fft, func_name)(x, **kwargs)


@overload
def fft2(x: np.ndarray, overwrite_x: bool = False, **kwargs) -> np.ndarray:
    ...


@overload
def fft2(x: da.core.Array, overwrite_x: bool = False, **kwargs) -> da.core.Array:
    ...


def fft2(x: U, overwrite_x: bool = False, **kwargs) -> U:
    """
    Compute the 2-dimensional discrete Fourier Transform.
    Using the FFT library specified in the configuration.
    """
    return _fft_dispatch(x, func_name="fft2", overwrite_x=overwrite_x, **kwargs)


@overload
def ifft2(x: np.ndarray, overwrite_x: bool = False, **kwargs) -> np.ndarray:
    ...


@overload
def ifft2(x: da.core.Array, overwrite_x: bool = False, **kwargs) -> da.core.Array:
    ...


def ifft2(x: U, overwrite_x: bool = False, **kwargs) -> U:
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.
    Using the FFT library specified in the configuration.
    """
    return _fft_dispatch(x, func_name="ifft2", overwrite_x=overwrite_x, **kwargs)


@overload
def fftn(x: np.ndarray, overwrite_x: bool = False, **kwargs) -> np.ndarray:
    ...


@overload
def fftn(x: da.core.Array, overwrite_x: bool = False, **kwargs) -> da.core.Array:
    ...


def fftn(x: U, overwrite_x: bool = False, **kwargs) -> U:
    """Compute the n-dimensional discrete Fourier Transform. Using the FFT library
    specified in the configuration."""
    return _fft_dispatch(x, func_name="fftn", overwrite_x=overwrite_x, **kwargs)


@overload
def ifftn(x: np.ndarray, overwrite_x: bool = False, **kwargs) -> np.ndarray:
    ...


@overload
def ifftn(x: da.core.Array, overwrite_x: bool = False, **kwargs) -> da.core.Array:
    ...


def ifftn(x: U, overwrite_x: bool = False, **kwargs) -> U:
    """
    Compute the n-dimensional inverse discrete Fourier Transform.
    Using the FFT library specified in the configuration.
    """
    return _fft_dispatch(x, func_name="ifftn", overwrite_x=overwrite_x, **kwargs)


def _fft2_convolve(x: U, kernel: U, overwrite_x: bool = False) -> U:
    x = fft2(x, overwrite_x=overwrite_x)
    try:
        x *= kernel
    except ValueError:
        x = x * kernel
    return ifft2(x, overwrite_x=overwrite_x)


@overload
def fft2_convolve(
    x: np.ndarray, kernel: np.ndarray, overwrite_x: bool = False
) -> np.ndarray:
    ...


@overload
def fft2_convolve(
    x: da.core.Array, kernel: np.ndarray, overwrite_x: bool = False
) -> da.core.Array:
    ...


def fft2_convolve(x: U, kernel: np.ndarray, overwrite_x: bool = False) -> U:
    """
    Compute the 2-dimensional convolution of an array with a kernel.

    Parameters
    ----------
    x : np.ndarray or da.core.Array
        Array to convolve.
    kernel : np.ndarray
        Convolution kernel.
    overwrite_x : bool, optional
        Overwrite the input array.

    Returns
    -------
    np.ndarray or da.core.Array
        Convolved array.
    """
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)

    if isinstance(x, da.core.Array):
        return da.map_blocks(
            _fft2_convolve,
            x,
            kernel=kernel,
            overwrite_x=overwrite_x,
            meta=xp.array((), dtype=get_dtype(complex=True)),
        )

    check_cupy_is_installed()  # type: ignore

    if isinstance(x, cp.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)


def fft_shift_kernel(positions: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Create an array representing one or more phase ramp(s) for shifting another array.

    Parameters
    ----------
    positions : np.ndarray
        Array of positions to shift the array to. The last dimension should be the
        number of dimensions to shift.
    shape : tuple
        Shape of the array to shift.

    Returns
    -------
    np.ndarray
        Array representing the phase ramp(s).
    """
    xp = get_array_module(positions)

    if cp is None or not isinstance(positions, cp.ndarray):
        positions = np.array(positions)

    assert positions.shape[-1] == len(shape)
    dims = positions.shape[-1]
    n = len(positions.shape) - 1
    k = list(spatial_frequencies(shape, (1.0,) * dims, xp=xp))

    for i in range(dims):
        d = list(range(0, n)) + list(range(n, n + dims))
        del d[i + n]
        expanded_positions = np.expand_dims(
            positions[..., i], tuple(range(n, n + dims))
        )

        k[i] = complex_exponential(
            -2 * np.pi * np.expand_dims(k[i], tuple(d)) * expanded_positions
        )

    array = k[0]
    for i in range(1, dims):
        array = array * k[i]

    return array


def fft_shift(array: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Shift an array in real space using Fourier space interpolation.

    Parameters
    ----------
    array : np.ndarray
        Array to shift.
    positions : np.ndarray
        Array of positions to shift the array to. The last dimension should be
        the number of dimensions to shift.

    Returns
    -------
    np.ndarray
        Shifted array
    """
    return ifft2(fft2(array) * fft_shift_kernel(positions, array.shape[-2:]))


def _fft_interpolation_masks_1d(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    mask1 = np.zeros(n1, dtype=bool)
    mask2 = np.zeros(n2, dtype=bool)

    if n2 > n1:
        mask1[:] = True

        if n1 == 1:
            mask2[0] = True
        elif n1 % 2 == 0:
            mask2[: n1 // 2] = True
            mask2[-n1 // 2 :] = True
        else:
            mask2[: n1 // 2 + 1] = True
            mask2[-n1 // 2 + 1 :] = True
    else:
        if n2 == 1:
            mask1[0] = True
        elif n2 % 2 == 0:
            mask1[: n2 // 2] = True
            mask1[-n2 // 2 :] = True
        else:
            mask1[: n2 // 2 + 1] = True
            mask1[-n2 // 2 + 1 :] = True

        mask2[:] = True

    return mask1, mask2


def fft_interpolation_masks(
    shape_in: tuple[int, ...], shape_out: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create boolean masks for interpolating between two arrays using Fourier space
    interpolation.

    Parameters
    ----------
    shape_in : tuple of int
        Shape of the input array to interpolate from.
    shape_out : tuple of int
        Shape of the output array to interpolate to.

    Returns
    -------
    tuple of np.ndarray
        Masks for the input and output arrays.
    """
    mask1_1d = []
    mask2_1d = []

    for i, (n1, n2) in enumerate(zip(shape_in, shape_out)):
        m1, m2 = _fft_interpolation_masks_1d(n1, n2)

        s = [slice(None) if j == i else np.newaxis for j in range(len(shape_in))]

        mask1_1d += [m1[tuple(s)]]
        mask2_1d += [m2[tuple(s)]]

    mask1 = mask1_1d[0]
    for m in mask1_1d[1:]:
        mask1 = mask1 * m

    mask2 = mask2_1d[0]
    for m in mask2_1d[1:]:
        mask2 = mask2 * m

    return mask1, mask2


def fft_crop(array: np.ndarray, new_shape: tuple[int, ...], normalize: bool = False):
    """
    Crop an array. It is assumed that the array is centered in Fourier space, this is
    used for real-space interpolation.

    Parameters
    ----------
    array : np.ndarray
        Array to crop.
    new_shape : tuple of int
        New shape of the array. If the new shape is smaller than the input array,
        each preceding dimension is treated as a batch dimension.
    normalize : bool, optional
        If True, renormalize the array to conserve the total amplitude.

    Returns
    -------
    np.ndarray
        Cropped array.
    """
    xp = get_array_module(array)

    if len(new_shape) < len(array.shape):
        # Not `array.shape[: -len(new_shape)]`: -0 is 0, not "the end", so
        # for `new_shape == ()` that slice was `[:0]`, always empty, rather
        # than `[:len(array.shape)]` -- every dimension is a batch
        # dimension when none are being resized.
        n_batch_dims = len(array.shape) - len(new_shape)
        new_shape = array.shape[:n_batch_dims] + new_shape

    mask_in, mask_out = fft_interpolation_masks(array.shape, new_shape)

    new_array = xp.zeros(new_shape, dtype=array.dtype)

    new_array[mask_out] = array[mask_in]

    if normalize:
        new_array = new_array * np.prod(new_array.shape) / np.prod(array.shape)

    return new_array


def fft_interpolate(
    array: np.ndarray,
    new_shape: Tuple[int, ...],
    normalization: str = "values",
    overwrite_x: bool = False,
):
    """
    Interpolate an array using Fourier space interpolation.

    Parameters
    ----------
    array : np.ndarray
        Array to interpolate.
    new_shape : tuple of int
        New shape of the array.
    normalization : str, optional
        Normalization to apply to the array. Can be 'values' or 'amplitude'.
    overwrite_x : bool, optional
        Overwrite the input array.

    Returns
    -------
    np.ndarray
        Interpolated array.
    """
    # Not `array.shape[-len(new_shape):]`: the same -0 problem, mirrored --
    # for `new_shape == ()` that slice was `[0:]`, everything, rather than
    # `[len(array.shape):]`, nothing (no axes are being resized).
    n_batch_dims = len(array.shape) - len(new_shape)
    old_size = np.prod(array.shape[n_batch_dims:])

    is_complex = np.iscomplexobj(array)

    array = array.astype(get_dtype(complex=True))

    if len(new_shape) == 2:
        array = fft2(array, overwrite_x=overwrite_x)
        array = fft_crop(array, new_shape)
        array = ifft2(array, overwrite_x=overwrite_x)
    else:
        if len(new_shape) != len(array.shape):
            axes = tuple(range(len(array.shape) - len(new_shape), len(array.shape)))
        else:
            axes = tuple(range(len(array.shape)))

        array = fftn(array, overwrite_x=overwrite_x, axes=axes)
        array = fft_crop(array, new_shape)
        array = ifftn(array, overwrite_x=overwrite_x, axes=axes)

    if not is_complex:
        array = array.real

    if normalization == "values":
        # See the comment above old_size: same -0 fix, same reasoning.
        n_batch_dims = len(array.shape) - len(new_shape)
        array *= np.prod(array.shape[n_batch_dims:]) / old_size
    elif normalization in ("amplitude", "intensity"):
        pass
    else:
        raise ValueError(f"Normalization '{normalization}' not recognized.")

    return array
