import dask.array as da
import mkl_fft
import numpy as np

from abtem.basic.backend import get_array_module, check_cupy_is_installed
from abtem.basic.complex import complex_exponential
from abtem.basic.grid import spatial_frequencies

try:
    import cupy as cp
except:
    cp = None


def fft2(x, overwrite_x=False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return mkl_fft.fft2(x, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(fft2, meta=xp.array((), dtype=np.complex64))

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.fft.fft2(x)


def ifft2(x, overwrite_x=False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return mkl_fft.ifft2(x, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(ifft2, meta=xp.array((), dtype=np.complex64))

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.fft.ifft2(x)


def _fft2_convolve(x, kernel, overwrite_x=True):
    x = fft2(x, overwrite_x=overwrite_x)
    x *= kernel
    return ifft2(x, overwrite_x=overwrite_x)


def fft2_convolve(x, kernel, overwrite_x=True, ):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(_fft2_convolve, kernel=kernel, overwrite_x=overwrite_x,
                            meta=xp.array((), dtype=np.complex64))

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return _fft2_convolve(x, kernel, overwrite_x)


# def fft2_shift_kernel(positions: np.ndarray, shape: tuple) -> np.ndarray:
#     """
#     Create an array representing one or more phase ramp(s) for shifting another array.
#
#     Parameters
#     ----------
#     positions : array of xy-positions
#     shape : two int
#
#     Returns
#     -------
#
#     """
#     xp = get_array_module(positions)
#
#     kx, ky = spatial_frequencies(shape, (1.,) * 2, delayed=False, xp=xp)
#     kx = kx.reshape((1,) * (len(positions.shape) - 1) + (-1, 1)).astype(np.float32)
#     ky = ky.reshape((1,) * (len(positions.shape) - 1) + (1, -1)).astype(np.float32)
#     x = positions[..., 0][..., None, None]
#     y = positions[..., 1][..., None, None]
#
#     twopi = np.float32(2. * np.pi)
#     result = complex_exponential(-twopi * kx * x) * complex_exponential(-twopi * ky * y)
#     return result


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
    k = list(spatial_frequencies(shape, (1.,) * dims, delayed=False, xp=xp))

    positions = [np.expand_dims(positions[..., i], list(range(n, n + dims))) for i in range(dims)]

    for i in range(dims):
        d = list(range(0, n)) + list(range(n, n + dims))
        del d[i + n]
        k[i] = complex_exponential(- 2 * np.pi * np.expand_dims(k[i], d) * positions[i])

    array = k[0]
    for i in range(1, dims):
        array = array * k[i]

    return array


def _fft_interpolation_masks_1d(n1, n2):
    mask1 = np.zeros(n1, dtype=bool)
    mask2 = np.zeros(n2, dtype=bool)

    if n2 > n1:
        mask1[:] = True
    else:
        if n2 == 1:
            mask1[0] = True
        elif n2 % 2 == 0:
            mask1[:n2 // 2] = True
            mask1[-n2 // 2:] = True
        else:
            mask1[:n2 // 2 + 1] = True
            mask1[-n2 // 2 + 1:] = True

    if n1 > n2:
        mask2[:] = True
    else:
        if n1 == 1:
            mask2[0] = True
        elif n1 % 2 == 0:
            mask2[:n1 // 2] = True
            mask2[-n1 // 2:] = True
        else:
            mask2[:n1 // 2 + 1] = True
            mask2[-n1 // 2 + 1:] = True

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


def fft_crop(array, new_shape):
    xp = get_array_module(array)

    if len(new_shape) < len(array.shape):
        new_shape = array.shape[:-len(new_shape)] + new_shape

    mask_in, mask_out = fft_interpolation_masks(array.shape, new_shape)

    new_array = xp.zeros(new_shape, dtype=array.dtype)

    # out_indices = xp.where(mask_out)
    # in_indices = xp.where(mask_in)

    new_array[mask_out] = array[mask_in]

    return new_array


def fft2_interpolate(array, new_shape, normalization='values', overwrite_x=False):
    xp = get_array_module(array)

    old_size = array.shape[-2] * array.shape[-1]

    def _fft2_interpolate(array, new_shape):
        return ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x)

    is_complex = np.iscomplexobj(array)

    array = xp.complex64(array)

    array = ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x)

    if not is_complex:
        array = array.real

    # if np.iscomplexobj(array):
    #     # array = array.map_blocks(_fft2_interpolate, new_shape=new_shape[-2:],
    #     #                          chunks=array.chunks[:-2] + ((new_shape[-2],), (new_shape[-1],)),
    #     #                          meta=xp.array((), dtype=xp.complex64))
    #
    #     array = xp.complex64(array)
    #     array = ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x).real
    #
    # else:

    if normalization == 'values':
        array *= array.shape[-1] * array.shape[-2] / old_size
    elif normalization == 'norm':
        array *= array.shape[-1] * array.shape[-2] / old_size
    elif (normalization != False) and (normalization != None):
        raise RuntimeError()

    return array
