from typing import Tuple

import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength, to_grid_angle
from abtem.core.utils import expand_dims_to_broadcast, get_dtype


def batch_crop_2d(array: np.ndarray, corners: np.ndarray, new_shape: Tuple[int, int]):
    """Crop ``new_shape`` windows from the last two axes of ``array`` at ``corners``.
    A window as large as the array can only sit at the origin, so ``array`` itself
    is returned, not a copy."""

    if tuple(new_shape) == tuple(array.shape[-2:]):
        return array

    xp = get_array_module(array)

    if len(array.shape) > 3:
        old_shape = array.shape

        batch_shape = array.shape[: -len(corners.shape) - 1]
        array = array.reshape((-1,) + array.shape[-2:])
        corners = corners.reshape((-1, 2))

        if batch_shape:
            assert array.shape[0] == corners.shape[0] * np.prod(batch_shape)
            corners = np.tile(corners, (np.prod(batch_shape), 1))
    else:
        old_shape = None

    # if xp is cp:
    i = xp.arange(array.shape[0])[:, None, None]
    ix = xp.arange(new_shape[0]) + xp.asarray(corners[:, 0, None])
    iy = xp.arange(new_shape[1]) + xp.asarray(corners[:, 1, None])
    ix = ix[:, :, None]
    iy = iy[:, None]
    array = array[i, ix, iy]
    # else:
    #     array = np.lib.stride_tricks.sliding_window_view(array, (1,) + new_shape)
    #     array = array[xp.arange(array.shape[0]), corners[:, 0], corners[:, 1], 0]

    if old_shape is not None:
        array = array.reshape(old_shape[:-2] + array.shape[-2:])

    return array


def minimum_crop(positions: np.ndarray, shape):
    xp = get_array_module(positions)

    offset = (shape[0] // 2, shape[1] // 2)
    corners = xp.rint(positions - xp.asarray(offset)).astype(int)
    upper_corners = corners + xp.asarray(shape)

    crop_corner = (xp.min(corners[..., 0]).item(), xp.min(corners[..., 1]).item())

    size = (
        xp.max(upper_corners[..., 0]).item() - crop_corner[0],
        xp.max(upper_corners[..., 1]).item() - crop_corner[1],
    )

    corners -= xp.asarray(crop_corner)
    return crop_corner, size, corners


def window_crop(positions: np.ndarray, window_gpts, gpts):
    """The crop box of windows of ``window_gpts`` points centred on ``positions``
    (in pixels of a grid of ``gpts`` points), as ``minimum_crop`` returns it, except
    that a window as large as the grid is the whole grid for every position, with
    each window's corner at the origin (the cell frame)."""
    if tuple(window_gpts) == tuple(gpts):
        xp = get_array_module(positions)
        return (0, 0), tuple(gpts), xp.zeros(positions.shape, dtype=int)

    return minimum_crop(positions, window_gpts)


def _periodic_runs(start: int, size: int, n: int) -> list[tuple[slice, slice]]:
    """The window ``[start, start + size)`` of a periodic axis of length ``n``, as
    contiguous runs of the axis.

    Each run is a pair ``(source, target)``: the slice of the axis it reads and
    the slice of the window it fills. ``start`` may be any integer and ``size``
    any length, also wider than one period: the first run starts at
    ``start % n``, and every later one at 0.
    """
    runs = []
    source = start % n
    done = 0
    while done < size:
        length = min(n - source, size - done)
        runs.append((slice(source, source + length), slice(done, done + length)))
        done += length
        source = 0
    return runs


def wrapped_crop_2d(
    array: np.ndarray, corner: Tuple[int, int], size: Tuple[int, int]
) -> np.ndarray:
    """The ``size`` window of the last two axes of ``array`` starting at ``corner``,
    with both axes periodic.

    ``corner`` may be any integers, also several periods outside the array, and
    ``size`` any lengths, also wider than one period.

    A window that does not wrap is returned as a view of ``array``: no copy, and
    a window as large as the array at the origin costs no memory. A window that
    wraps is copied run by run into a new array: one pass over the window, in
    slice copies (four at most for a window no wider than one period) and with
    no index arrays, which on the GPU means a few copy kernels and no gather.
    Callers must not write into the result.
    """
    rows = _periodic_runs(corner[0], size[0], array.shape[-2])
    columns = _periodic_runs(corner[1], size[1], array.shape[-1])

    if len(rows) == 1 and len(columns) == 1:
        return array[..., rows[0][0], columns[0][0]]

    xp = get_array_module(array)
    cropped = xp.empty_like(array, shape=array.shape[:-2] + tuple(size))
    for source_rows, target_rows in rows:
        for source_columns, target_columns in columns:
            cropped[..., target_rows, target_columns] = array[
                ..., source_rows, source_columns
            ]
    return cropped


def prism_wave_vectors(
    cutoff: float,
    extent: Tuple[float, float],
    energy: float,
    interpolation: Tuple[int, int],
    xp=np,
) -> np.ndarray:
    wavelength = energy2wavelength(energy)
    cutoff = to_grid_angle(cutoff)

    n_max = int(np.ceil(cutoff / 1.0e3 / (wavelength / extent[0] * interpolation[0])))
    m_max = int(np.ceil(cutoff / 1.0e3 / (wavelength / extent[1] * interpolation[1])))

    dtype = get_dtype(complex=False)
    n = np.arange(-n_max, n_max + 1, dtype=dtype)
    w = np.asarray(extent[0], dtype=dtype)
    m = np.arange(-m_max, m_max + 1, dtype=dtype)
    h = np.asarray(extent[1], dtype=dtype)

    kx = n / w * dtype(interpolation[0])
    ky = m / h * dtype(interpolation[1])

    mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (cutoff / 1.0e3 / wavelength) ** 2

    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    kx = kx[mask]
    ky = ky[mask]
    return xp.asarray([kx, ky]).T


def plane_waves(
    wave_vectors: np.ndarray,
    extent: Tuple[float, float],
    gpts: Tuple[int, int],
    reverse: bool = False,
) -> np.ndarray:
    xp = get_array_module(wave_vectors)
    dtype = get_dtype(complex=False)
    x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=dtype)
    y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=dtype)

    sign = -1.0 if reverse else 1.0

    array = complex_exponential(
        sign * 2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]
    ) * complex_exponential(
        sign * 2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]
    )

    return array


def _planewave_shift_coefficients(positions, wave_vectors):
    xp = get_array_module(positions)
    # wave_vectors = xp.asarray(wave_vectors)

    coefficients = complex_exponential(
        -2.0 * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None]
    )
    # print(coefficients.shape, coefficients.dtype)
    coefficients *= complex_exponential(
        -2.0 * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None]
    )

    return coefficients


def prism_coefficients(positions, wave_vectors, xp, ctf=None):
    wave_vectors = xp.asarray(wave_vectors)

    coefficients = _planewave_shift_coefficients(positions, wave_vectors)

    if ctf is not None:
        alpha = (
            xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * ctf.wavelength
        )
        phi = xp.arctan2(wave_vectors[:, 1], wave_vectors[:, 0])

        basis = ctf._evaluate_from_angular_grid(alpha, phi)
        basis, coefficients = expand_dims_to_broadcast(
            basis, coefficients, match_dims=[(-1,), (-1,)]
        )
        coefficients = coefficients * basis

    return coefficients
