import functools
from typing import Tuple

try:
    import pyfftw
except ModuleNotFoundError:
    pyfftw = None

import dask.array as da
import numpy as np

from abtem.core import config
from abtem.core.backend import get_array_module, check_cupy_is_installed
from abtem.core.complex import complex_exponential
from abtem.core.grid import spatial_frequencies

try:
    import mkl_fft
except ModuleNotFoundError:
    mkl_fft = None

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def raise_fft_lib_not_present(lib_name):
    raise RuntimeError(f'FFT library {lib_name} not present. Install this package or change the FFT library in your '
                       f'configuration')


def _fft_dispatch(x, func_name, overwrite_x: bool = False, **kwargs):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        if config.get('fft') == 'mkl':
            if mkl_fft is None:
                raise_fft_lib_not_present('mkl_fft')

            return getattr(mkl_fft, func_name)(x, overwrite_x=overwrite_x, **kwargs)
        elif config.get('fft') == 'fftw':
            if pyfftw is None:
                raise_fft_lib_not_present('pyfftw')

            fftw_obj = getattr(pyfftw.builders, func_name)(x,
                                                           overwrite_input=overwrite_x,
                                                           planner_effort=config.get('fftw.planning_effort'),
                                                           threads=config.get('fftw.threads'),
                                                           avoid_copy=False,
                                                           **kwargs
                                                           )

            return fftw_obj()
        else:
            raise RuntimeError()

    if isinstance(x, da.core.Array):
        return x.map_blocks(_fft_dispatch, func_name=func_name, overwrite_x=overwrite_x, **kwargs,
                            meta=xp.array((), dtype=np.complex64))

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return getattr(cp.fft, func_name)(x, **kwargs)


def fft2(x, overwrite_x=False, **kwargs):
    return _fft_dispatch(x, func_name='fft2', overwrite_x=overwrite_x, **kwargs)


def ifft2(x, overwrite_x=False, **kwargs):
    return _fft_dispatch(x, func_name='ifft2', overwrite_x=overwrite_x, **kwargs)


def fftn(x, overwrite_x=False, **kwargs):
    return _fft_dispatch(x, func_name='fftn', overwrite_x=overwrite_x, **kwargs)


def ifftn(x, overwrite_x=False, **kwargs):
    return _fft_dispatch(x, func_name='ifftn', overwrite_x=overwrite_x, **kwargs)


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
        return x.map_blocks(_fft2_convolve, kernel=kernel, overwrite_x=overwrite_x,
                            meta=xp.array((), dtype=np.complex64))

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
    k = list(spatial_frequencies(shape, (1.,) * dims, xp=xp))

    positions = [np.expand_dims(positions[..., i], tuple(range(n, n + dims))) for i in range(dims)]

    for i in range(dims):
        d = list(range(0, n)) + list(range(n, n + dims))
        del d[i + n]
        k[i] = complex_exponential(- 2 * np.pi * np.expand_dims(k[i], tuple(d)) * positions[i])

    array = k[0]
    for i in range(1, dims):
        array = array * k[i]

    return array


def fft_shift(array, positions):
    xp = get_array_module(array)
    return xp.fft.ifft2(xp.fft.fft2(array) * fft_shift_kernel(positions, array.shape[-2:]))


def _fft_interpolation_masks_1d(n1, n2):
    mask1 = np.zeros(n1, dtype=bool)
    mask2 = np.zeros(n2, dtype=bool)

    if n2 > n1:
        mask1[:] = True

        if n1 == 1:
            mask2[0] = True
        elif n1 % 2 == 0:
            mask2[:n1 // 2] = True
            mask2[-n1 // 2:] = True
        else:
            mask2[:n1 // 2 + 1] = True
            mask2[-n1 // 2 + 1:] = True
    else:
        if n2 == 1:
            mask1[0] = True
        elif n2 % 2 == 0:
            mask1[:n2 // 2] = True
            mask1[-n2 // 2:] = True
        else:
            mask1[:n2 // 2 + 1] = True
            mask1[-n2 // 2 + 1:] = True

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
        new_shape = array.shape[:-len(new_shape)] + new_shape

    mask_in, mask_out = fft_interpolation_masks(array.shape, new_shape)

    new_array = xp.zeros(new_shape, dtype=array.dtype)

    new_array[mask_out] = array[mask_in]

    if normalize:
        new_array = new_array * np.prod(new_array.shape) / np.prod(array.shape)

    return new_array


def fft_interpolate(array: np.ndarray,
                    new_shape: Tuple[int, ...],
                    normalization: str = 'values',
                    overwrite_x: bool = False):
    xp = get_array_module(array)
    old_size = np.prod(array.shape[-len(new_shape):])

    is_complex = np.iscomplexobj(array)
    array = array.astype(xp.complex64)

    if len(new_shape) == 2:
        array = ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x)
    else:
        if len(new_shape) != len(array.shape):
            axes = tuple(range(len(array.shape) - len(new_shape), len(array.shape)))
        else:
            axes = None

        array = ifftn(fft_crop(fftn(array, axes=axes), new_shape), overwrite_x=overwrite_x, axes=axes)

    if not is_complex:
        array = array.real

    if normalization == 'values':
        array *= np.prod(array.shape[-len(new_shape):]) / old_size

    elif normalization != 'intensity':
        raise ValueError()

    return array
