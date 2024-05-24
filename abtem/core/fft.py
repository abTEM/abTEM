from typing import Tuple

import dask.array as da
import numpy as np

from abtem.core import config
from abtem.core.backend import get_array_module, check_cupy_is_installed
from abtem.core.complex import complex_exponential
from abtem.core.grid import spatial_frequencies
from threadpoolctl import threadpool_limits
import warnings

from abtem.core.utils import get_dtype

try:
    import pyfftw
except (ModuleNotFoundError, ImportError):
    pyfftw = None

try:
    import mkl_fft
except ModuleNotFoundError:
    mkl_fft = None

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
except ImportError:
    if config.get("device") == "gpu":
        warnings.warn(
            "The CuPy library could not be imported. Please check your installation, or change your configuration to "
            "use CPU."
        )
    cp = None


def raise_fft_lib_not_present(lib_name):
    raise RuntimeError(
        f"FFT library {lib_name} not present. Install this package or change the FFT library in your "
        f"configuration."
    )


def _fft_name_to_fftw_direction(name: str):
    if name[0] == "i":
        direction = "FFTW_BACKWARD"
    else:
        direction = "FFTW_FORWARD"
    return direction


def _new_fftw_object(array: np.ndarray, name: str, flags=()):
    dummy = np.zeros_like(array)

    direction = _fft_name_to_fftw_direction(name)

    fftw_object = pyfftw.FFTW(
        dummy,
        dummy,
        axes=(-2, -1),
        direction=direction,
        threads=config.get("fftw.threads"),  # noqa
        flags=(config.get("fftw.planning_effort"),) + flags,  # noqa
        planning_timelimit=config.get("fftw.planning_timelimit"),  # noqa
    )

    fftw_object.update_arrays(array, array)

    return fftw_object


class CachedFFTWConvolution:
    def __init__(self):
        self._fftw_objects = None
        self._shape = None

    def __call__(self, array, kernel, overwrite_x):
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
    direction = _fft_name_to_fftw_direction(name)
    
    flags = (config.get("fftw.planning_effort"),)
    if overwrite_x:
        flags += ("FFTW_DESTROY_INPUT",)

    try:
        fftw = pyfftw.FFTW(
            array,
            array,
            axes=axes,
            direction=direction,
            threads=config.get("fftw.threads"),  # noqa
            flags=flags + ("FFTW_WISDOM_ONLY",),  # noqa
        )
        
    except RuntimeError as e:
        if not str(e) == "No FFTW wisdom is known for this plan.":
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


def _mkl_fft_dispatch(x: np.ndarray, func_name: str, overwrite_x: bool, **kwargs):
    if mkl_fft is None:
        raise_fft_lib_not_present("mkl_fft")

    with threadpool_limits(limits=config.get("mkl.threads")):
        return getattr(mkl_fft, func_name)(x, overwrite_x=overwrite_x, **kwargs)


def _fftw_dispatch(x: np.ndarray, func_name: str, overwrite_x: bool, **kwargs):
    if pyfftw is None:
        raise_fft_lib_not_present("pyfftw")

    if not overwrite_x:
        x = x.copy()

    return get_fftw_object(x, func_name, overwrite_x=overwrite_x, **kwargs)()


def _fft_dispatch(x, func_name, overwrite_x: bool = False, **kwargs):
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
        return x.map_blocks(
            _fft_dispatch,
            func_name=func_name,
            overwrite_x=overwrite_x,
            **kwargs,
            meta=xp.array((), dtype=np.complex64),
        )

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return getattr(cp.fft, func_name)(x, **kwargs)


def fft2(x: np.ndarray, overwrite_x: bool = False, **kwargs) -> np.ndarray:
    return _fft_dispatch(x, func_name="fft2", overwrite_x=overwrite_x, **kwargs)


def ifft2(x: np.ndarray, overwrite_x: bool = False, **kwargs) -> np.ndarray:
    return _fft_dispatch(x, func_name="ifft2", overwrite_x=overwrite_x, **kwargs)


def fftn(x, overwrite_x: bool = False, **kwargs):
    return _fft_dispatch(x, func_name="fftn", overwrite_x=overwrite_x, **kwargs)


def ifftn(x, overwrite_x=False, **kwargs):
    return _fft_dispatch(x, func_name="ifftn", overwrite_x=overwrite_x, **kwargs)


def _fft2_convolve(x, kernel, overwrite_x: bool = False):
    x = fft2(x, overwrite_x=overwrite_x)
    try:
        x *= kernel
    except ValueError:
        x = x * kernel
    return ifft2(x, overwrite_x=overwrite_x)


def fft2_convolve(x, kernel, overwrite_x: bool = False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(
            _fft2_convolve,
            kernel=kernel,
            overwrite_x=overwrite_x,
            meta=xp.array((), dtype=np.complex64),
        )

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)


def fft_shift_kernel(positions: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Create an array representing one or more phase ramp(s) for shifting another array.

    Parameters
    ----------
    positions : array of xy-positions
    shape : two int

    Returns
    -------

    """

    xp = get_array_module(positions)

    assert positions.shape[-1] == len(shape)
    dims = positions.shape[-1]
    n = len(positions.shape) - 1
    k = list(spatial_frequencies(shape, (1.0,) * dims, xp=xp))

    positions = [
        np.expand_dims(positions[..., i], tuple(range(n, n + dims)))
        for i in range(dims)
    ]

    for i in range(dims):
        d = list(range(0, n)) + list(range(n, n + dims))
        del d[i + n]
        k[i] = complex_exponential(
            -2 * np.pi * np.expand_dims(k[i], tuple(d)) * positions[i]
        )

    array = k[0]
    for i in range(1, dims):
        array = array * k[i]

    return array


def fft_shift(array, positions):
    xp = get_array_module(array)
    return xp.fft.ifft2(
        xp.fft.fft2(array) * fft_shift_kernel(positions, array.shape[-2:])
    )


def _fft_interpolation_masks_1d(n1, n2):
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


def fft_interpolation_masks(shape1, shape2):
    mask1_1d = []
    mask2_1d = []

    for i, (n1, n2) in enumerate(zip(shape1, shape2)):
        m1, m2 = _fft_interpolation_masks_1d(n1, n2)

        s = [np.newaxis] * len(shape1)
        s[i] = slice(None)

        mask1_1d += [m1[tuple(s)]]
        mask2_1d += [m2[tuple(s)]]

    mask1 = mask1_1d[0]
    for m in mask1_1d[1:]:
        mask1 = mask1 * m

    mask2 = mask2_1d[0]
    for m in mask2_1d[1:]:
        mask2 = mask2 * m

    return mask1, mask2


def fft_crop(array, new_shape, normalize: bool = False):
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
    xp = get_array_module(array)
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

    elif normalization != "intensity":
        raise ValueError()

    return array
