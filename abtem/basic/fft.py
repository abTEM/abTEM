import cupy as cp
import dask.array as da
import mkl_fft
import numpy as np

from abtem.basic.backend import get_array_module
from abtem.basic.complex import complex_exponential
from abtem.basic.grid import spatial_frequencies


def fft2(x, overwrite_x=False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return mkl_fft.fft2(x, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(mkl_fft.fft2, meta=xp.array((), dtype=np.complex64))

    if isinstance(x, cp.ndarray):
        return cp.fft.fft2(x)


def ifft2(x, overwrite_x=False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return mkl_fft.ifft2(x, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(ifft2, meta=xp.array((), dtype=np.complex64))

    if isinstance(x, cp.ndarray):
        return cp.fft.ifft2(x)


def _fft2_convolve(x, kernel, overwrite_x=True):
    x = mkl_fft.fft2(x, overwrite_x=overwrite_x)
    x *= kernel
    return mkl_fft.ifft2(x, overwrite_x=overwrite_x)


def fft2_convolve(x, kernel, overwrite_x=True, ):
    if isinstance(x, np.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(_fft2_convolve, kernel=kernel, overwrite_x=overwrite_x, dtype=np.complex64)


def fft2_shift_kernel(positions: np.ndarray, shape: tuple) -> np.ndarray:
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

    kx, ky = spatial_frequencies(shape, (1.,) * 2, delayed=False, xp=xp)
    kx = kx.reshape((1,) * (len(positions.shape) - 1) + (-1, 1)).astype(np.float32)
    ky = ky.reshape((1,) * (len(positions.shape) - 1) + (1, -1)).astype(np.float32)
    x = positions[..., 0][..., None, None]
    y = positions[..., 1][..., None, None]

    twopi = np.float32(2. * np.pi)
    result = complex_exponential(-twopi * kx * x) * complex_exponential(-twopi * ky * y)
    return result


def fft2_interpolation_masks(shape1, shape2, xp=np, epsilon=1e-7):
    kx1 = xp.fft.fftfreq(shape1[-2], 1 / shape1[-2])
    ky1 = xp.fft.fftfreq(shape1[-1], 1 / shape1[-1])

    kx2 = xp.fft.fftfreq(shape2[-2], 1 / shape2[-2])
    ky2 = xp.fft.fftfreq(shape2[-1], 1 / shape2[-1])

    kx_min = max(xp.min(kx1), xp.min(kx2)) - epsilon
    kx_max = min(xp.max(kx1), xp.max(kx2)) + epsilon
    ky_min = max(xp.min(ky1), xp.min(ky2)) - epsilon
    ky_max = min(xp.max(ky1), xp.max(ky2)) + epsilon

    kx1, ky1 = xp.meshgrid(kx1, ky1, indexing='ij')
    kx2, ky2 = xp.meshgrid(kx2, ky2, indexing='ij')

    mask1 = (kx1 <= kx_max) & (kx1 >= kx_min) & (ky1 <= ky_max) & (ky1 >= ky_min)
    mask2 = (kx2 <= kx_max) & (kx2 >= kx_min) & (ky2 <= ky_max) & (ky2 >= ky_min)
    return mask1, mask2


def fft2_crop(array, new_shape):
    xp = get_array_module(array)
    mask_in, mask_out = fft2_interpolation_masks(array.shape, new_shape, xp=xp)

    if len(new_shape) < len(array.shape):
        new_shape = array.shape[:-2] + new_shape

    new_array = xp.zeros(new_shape, dtype=array.dtype)

    out_indices = xp.where(mask_out)
    in_indices = xp.where(mask_in)

    new_array[..., out_indices[0], out_indices[1]] = array[..., in_indices[0], in_indices[1]]
    return new_array


def fft2_interpolate(array, new_shape, normalization='values', overwrite_x=False):
    xp = get_array_module(array)

    old_size = array.shape[-2] * array.shape[-1]

    if np.iscomplexobj(array):
        cropped = fft2_crop(fft2(array), new_shape)
        array = ifft2(cropped, overwrite_x=overwrite_x)
    else:
        array = xp.complex64(array)
        array = ifft2(fft2_crop(fft2(array), new_shape), overwrite_x=overwrite_x).real

    if normalization == 'values':
        array *= array.shape[-1] * array.shape[-2] / old_size
    elif normalization == 'norm':
        array *= array.shape[-1] * array.shape[-2] / old_size
    elif (normalization != False) and (normalization != None):
        raise RuntimeError()

    return array
