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
    import pyfftw
except ModuleNotFoundError:
    pyfftw = None

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def build_fftw_obj(x, allow_new_plan=False, overwrite_x=True, backward=False):
    # if backward:
    #     obj = pyfftw.builders.ifft2(x,
    #                                  overwrite_input=overwrite_x,
    #                                  planner_effort=config.get('fftw.planning_effort'),
    #                                  threads=config.get('fftw.threads'))
    # else:
    #     obj = pyfftw.builders.fft2(x,
    #                                 overwrite_input=overwrite_x,
    #                                 planner_effort=config.get('fftw.planning_effort'),
    #                                 threads=config.get('fftw.threads'))
    # return obj()

    # flags = (,)
    # if overwrite_x:
    #    flags += ('FFTW_DESTROY_INPUT',)
    # if not allow_new_plan:
    #    flags +=

    try:
        out = pyfftw.byte_align(np.zeros_like(x))
        fftw_obj = pyfftw.FFTW(x, out, axes=(-1, -2),
                               direction='FFTW_BACKWARD' if backward else 'FFTW_FORWARD',
                               threads=config.get('fftw.threads'),
                               flags=(config.get('fftw.planning_effort'), 'FFTW_WISDOM_ONLY',))
        fftw_obj()
        return out

    except RuntimeError:
        out = pyfftw.byte_align(np.zeros_like(x))
        fftw_obj = pyfftw.FFTW(x.copy(), out, axes=(-1, -2),
                               direction='FFTW_BACKWARD' if backward else 'FFTW_FORWARD',
                               threads=config.get('fftw.threads'),
                               flags=(config.get('fftw.planning_effort'),))

        return build_fftw_obj(x, allow_new_plan=True, overwrite_x=overwrite_x, backward=backward)

        # if not allow_new_plan:
        #     if backward:
        #         return pyfftw.builders.ifft2(x,
        #                                      overwrite_input=overwrite_x,
        #                                      planner_effort=config.get('fftw.planning_effort'),
        #                                      threads=config.get('fftw.threads'))
        #     else:
        #         return pyfftw.builders.fft2(x,
        #                                     overwrite_input=overwrite_x,
        #                                     planner_effort=config.get('fftw.planning_effort'),
        #                                     threads=config.get('fftw.threads'))

        # dummy = pyfftw.byte_align(np.zeros_like(x))
        #
        # pyfftw.FFTW(dummy, dummy,
        #             axes=(-1, -2),
        #             direction='FFTW_BACKWARD' if backward else 'FFTW_FORWARD',
        #             threads=config.get('fftw.threads'),
        #             flags=flags,
        #             planning_timelimit=120)
        #
        # return build_fftw_obj(x, allow_new_plan=False, overwrite_x=overwrite_x)


def fft2(x, overwrite_x=False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        if config.get('fft') == 'mkl':
            return mkl_fft.fft2(x, overwrite_x=overwrite_x)
        elif config.get('fft') == 'fftw':
            # if not overwrite_x:
            #    x = x.copy()

            x = build_fftw_obj(x, overwrite_x=overwrite_x, backward=False)
            return x
        else:
            raise RuntimeError()

    if isinstance(x, da.core.Array):
        return x.map_blocks(fft2, meta=xp.array((), dtype=np.complex64))

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.fft.fft2(x)


def ifft2(x, overwrite_x=False):
    xp = get_array_module(x)

    if isinstance(x, np.ndarray):
        if config.get('fft') == 'mkl':
            return mkl_fft.ifft2(x, overwrite_x=overwrite_x)
        elif config.get('fft') == 'fftw':

            x = build_fftw_obj(x, overwrite_x=overwrite_x, backward=True)
            return x
        else:
            raise RuntimeError()

    if isinstance(x, da.core.Array):
        return x.map_blocks(ifft2, meta=xp.array((), dtype=np.complex64))

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp.fft.ifft2(x)


def _fft2_convolve(x, kernel, overwrite_x=False):
    x = fft2(x, overwrite_x=overwrite_x)
    x *= kernel
    return ifft2(x, overwrite_x=overwrite_x)


def fft2_convolve(x, kernel, overwrite_x=False, ):
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

    # def _fft2_interpolate(array, new_shape):
    #    return ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x)

    is_complex = np.iscomplexobj(array)
    array = array.astype(xp.complex64)

    array = ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x)

    if not is_complex:
        array = array.real

    if normalization == 'values':
        #array *= old_size / np.prod(array.shape[-2:])
        #array *= array.shape[-1] * array.shape[-2] / old_size
        array *= np.prod(array.shape[-2:]) / old_size

    elif normalization == 'intensity':
        #array *= np.sqrt(np.prod(array.shape[-2:]) / old_size)
        array *= np.sqrt(old_size / np.prod(array.shape[-2:]))
    elif normalization != 'amplitude':
        raise RuntimeError()

    return array
