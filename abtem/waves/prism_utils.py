from typing import Tuple

import numpy as np

from abtem.core.backend import get_array_module, cp
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength
from abtem.core.utils import expand_dims_to_match


def batch_crop_2d(array: np.ndarray, corners: np.ndarray, new_shape: Tuple[int, int]):
    xp = get_array_module(array)

    if len(array.shape) > 3:
        old_shape = array.shape

        batch_shape = array.shape[:-len(corners.shape) - 1]
        array = array.reshape((-1,) + array.shape[-2:])
        corners = corners.reshape((-1, 2))

        if batch_shape:
            assert array.shape[0] == corners.shape[0] * np.prod(batch_shape)
            corners = np.tile(corners, (np.prod(batch_shape), 1))
    else:
        old_shape = None

    if xp is cp:
        i = xp.arange(array.shape[0])[:, None, None]
        ix = cp.arange(new_shape[0]) + cp.asarray(corners[:, 0, None])
        iy = cp.arange(new_shape[1]) + cp.asarray(corners[:, 1, None])
        ix = ix[:, :, None]
        iy = iy[:, None]
        array = array[i, ix, iy]
    else:
        array = np.lib.stride_tricks.sliding_window_view(array, (1,) + new_shape)
        array = array[xp.arange(array.shape[0]), corners[:, 0], corners[:, 1], 0]

    if old_shape is not None:
        array = array.reshape(old_shape[:-2] + array.shape[-2:])

    return array


def minimum_crop(positions: np.ndarray, shape):
    xp = get_array_module(positions)

    offset = (shape[0] // 2, shape[1] // 2)
    corners = xp.rint(positions - xp.asarray(offset)).astype(int)
    upper_corners = corners + xp.asarray(shape)

    crop_corner = (xp.min(corners[..., 0]).item(), xp.min(corners[..., 1]).item())

    size = (xp.max(upper_corners[..., 0]).item() - crop_corner[0],
            xp.max(upper_corners[..., 1]).item() - crop_corner[1])

    corners -= xp.asarray(crop_corner)
    return crop_corner, size, corners


def wrapped_slices(start: int, stop: int, n: int) -> Tuple[slice, slice]:
    if start < 0:
        if stop > n:
            raise RuntimeError(f'start = {start} stop = {stop}, n = {n}')

        a = slice(start % n, None)
        b = slice(0, stop)

    elif stop > n:
        if start < 0:
            raise RuntimeError(f'start = {start} stop = {stop}, n = {n}')

        a = slice(start, None)
        b = slice(0, stop - n)

    else:
        a = slice(start, stop)
        b = slice(0, 0)
    return a, b


def wrapped_crop_2d(array: np.ndarray, corner: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    upper_corner = (corner[0] + size[0], corner[1] + size[1])

    xp = get_array_module(array)

    try:
        a, c = wrapped_slices(corner[0], upper_corner[0], array.shape[-2])
        b, d = wrapped_slices(corner[1], upper_corner[1], array.shape[-1])
    except RuntimeError:
        padding = tuple((abs(min(c, 0)), max(c + k - l, 0)) for c, l, k in zip(corner, array.shape[-2:], size))
        slices = tuple(slice(c + p[0], c + p[0] + l) for c, l, p in zip(corner, size, padding))
        padding = ((0, 0),) * (len(array.shape) - 2) + padding
        slices = (slice(None),) * (len(array.shape) - 2) + slices
        array = xp.pad(array, padding, mode='wrap')[slices]
        return array

    A = array[..., a, b]
    B = array[..., c, b]
    D = array[..., c, d]
    C = array[..., a, d]

    if A.size == 0:
        AB = B
    elif B.size == 0:
        AB = A
    else:
        AB = xp.concatenate([A, B], axis=-2)

    if C.size == 0:
        CD = D
    elif D.size == 0:
        CD = C
    else:
        CD = xp.concatenate([C, D], axis=-2)

    if CD.size == 0:
        return AB

    if AB.size == 0:
        return CD

    return xp.concatenate([AB, CD], axis=-1)


def prism_wave_vectors(cutoff: float, extent: Tuple[float, float], energy: float,
                       interpolation: Tuple[int, int], xp=np) -> np.ndarray:
    wavelength = energy2wavelength(energy)

    n_max = int(np.ceil(cutoff / 1.e3 / (wavelength / extent[0] * interpolation[0])))
    m_max = int(np.ceil(cutoff / 1.e3 / (wavelength / extent[1] * interpolation[1])))

    n = np.arange(-n_max, n_max + 1, dtype=np.float32)
    w = np.asarray(extent[0], dtype=np.float32)
    m = np.arange(-m_max, m_max + 1, dtype=np.float32)
    h = np.asarray(extent[1], dtype=np.float32)

    kx = n / w * np.float32(interpolation[0])
    ky = m / h * np.float32(interpolation[1])

    mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (cutoff / 1.e3 / wavelength) ** 2

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kx = kx[mask]
    ky = ky[mask]
    return xp.asarray([kx, ky]).T


def plane_waves(wave_vectors: np.ndarray,
                extent: Tuple[float, float],
                gpts: Tuple[int, int],
                reverse: bool = False) -> np.ndarray:
    xp = get_array_module(wave_vectors)
    x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
    y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)

    sign = -1. if reverse else 1.

    array = (complex_exponential(sign * 2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]) *
             complex_exponential(sign * 2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]))

    return array


def prism_coefficients(positions, ctf, wave_vectors, xp):
    wave_vectors = xp.asarray(wave_vectors)

    alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * ctf.wavelength
    phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])
    basis = ctf.evaluate_with_alpha_and_phi(alpha, phi)

    coefficients = complex_exponential(-2. * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None])
    coefficients *= complex_exponential(-2. * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None])

    basis, coefficients = expand_dims_to_match(
        basis, coefficients, match_dims=[(-1,), (-1,)]
    )
    coefficients = coefficients * basis

    return coefficients
