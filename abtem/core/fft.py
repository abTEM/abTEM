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


def _new_fftw_object(array: np.ndarray, name: str, flags: tuple[str, ...] = ()):
    dummy = np.zeros_like(array)

    direction = _fft_name_to_fftw_direction(name)

    fftw_object = pyfftw.FFTW(
        dummy,
        dummy,
        axes=(-2, -1),
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
        if array.shape != self._shape:
            self._fftw_objects = None

        if self._fftw_objects is None:
            fftw_objects = {
                name: _new_fftw_object(array, name=name) for name in ("ifft2", "fft2")
            }
            self._fftw_objects = fftw_objects

        if not overwrite_x:
            array = array.copy()
            self._fftw_objects["fft2"].update_arrays(array, array)
            self._fftw_objects["ifft2"].update_arrays(array, array)

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
    Get a pyfftw object for a given array and a given FFT function.
    The object is cached and reused if the array shape is the same.

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

    Returns
    -------
    pyfftw.FFTW
        FFTW object.
    """

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
            return getattr(pyfftw.builders, name)(array)

        elif not allow_new_wisdom:
            raise

        _new_fftw_object(array, name, flags=flags)

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
        x = x.copy()

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
        new_shape = array.shape[: -len(new_shape)] + new_shape

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
    old_size = np.prod(array.shape[-len(new_shape) :])

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
        array *= np.prod(array.shape[-len(new_shape) :]) / old_size

    elif normalization == "amplitude":
        pass
    else:
        pass
        # raise ValueError(f"Normalization [{normalization}] not recognized.")

    # elif normalization != "intensity":
    #    raise ValueError()

    # else:
    #    raise ValueError(f"Normalization {normalization} not recognized.")

    return array
