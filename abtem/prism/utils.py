from typing import Tuple

import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength
from abtem.core.grid import reciprocal_cell
from abtem.core.utils import expand_dims_to_broadcast


def batch_crop_2d(array: np.ndarray, corners: np.ndarray, new_shape: Tuple[int, int]):
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


def wrapped_slices(start: int, stop: int, n: int) -> Tuple[slice, slice]:
    if start < 0:
        if stop > n:
            raise RuntimeError(f"start = {start} stop = {stop}, n = {n}")

        a = slice(start % n, None)
        b = slice(0, stop)

    elif stop > n:
        if start < 0:
            raise RuntimeError(f"start = {start} stop = {stop}, n = {n}")

        a = slice(start, None)
        b = slice(0, stop - n)

    else:
        a = slice(start, stop)
        b = slice(0, 0)
    return a, b


def wrapped_crop_2d(
    array: np.ndarray, corner: Tuple[int, int], size: Tuple[int, int]
) -> np.ndarray:
    upper_corner = (corner[0] + size[0], corner[1] + size[1])

    xp = get_array_module(array)

    try:
        a, c = wrapped_slices(corner[0], upper_corner[0], array.shape[-2])
        b, d = wrapped_slices(corner[1], upper_corner[1], array.shape[-1])
    except RuntimeError:
        padding = tuple(
            (abs(min(c, 0)), max(c + k - l, 0))
            for c, l, k in zip(corner, array.shape[-2:], size)
        )
        slices = tuple(
            slice(c + p[0], c + p[0] + l) for c, l, p in zip(corner, size, padding)
        )
        padding = ((0, 0),) * (len(array.shape) - 2) + padding
        slices = (slice(None),) * (len(array.shape) - 2) + slices
        array = xp.pad(array, padding, mode="wrap")[slices]
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


def prism_wave_vectors(
    cutoff: float,
    extent: Tuple[float, float],
    energy: float,
    interpolation: Tuple[int, int],
    xp=np,
    cell: np.ndarray = None,
) -> np.ndarray:
    """Cartesian transverse wave vectors ``(kx, ky)`` [1/Å] of the PRISM plane-wave
    expansion. The retained vectors are reciprocal-lattice points of the (interpolated)
    cell that fall within the radial ``cutoff`` [mrad]. ``cell`` is an optional 2x2
    real-space cell (rows are the in-plane lattice vectors); when ``None`` the cell is
    assumed orthogonal and the result is bit-identical to the original implementation."""
    wavelength = energy2wavelength(energy)

    n_max = int(np.ceil(cutoff / 1.0e3 / (wavelength / extent[0] * interpolation[0])))
    m_max = int(np.ceil(cutoff / 1.0e3 / (wavelength / extent[1] * interpolation[1])))

    n = np.arange(-n_max, n_max + 1, dtype=np.float32)
    m = np.arange(-m_max, m_max + 1, dtype=np.float32)

    if cell is None:
        w = np.asarray(extent[0], dtype=np.float32)
        h = np.asarray(extent[1], dtype=np.float32)

        kx = n / w * np.float32(interpolation[0])
        ky = m / h * np.float32(interpolation[1])

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (cutoff / 1.0e3 / wavelength) ** 2

        kx, ky = np.meshgrid(kx, ky, indexing="ij")
        kx = kx[mask]
        ky = ky[mask]
        return xp.asarray([kx, ky]).T

    # non-orthogonal cell: enumerate reciprocal-lattice points g = n*ix*b1 + m*iy*b2
    # (b1, b2 are the reciprocal basis vectors) and keep those within the cutoff sphere.
    reciprocal = reciprocal_cell(cell)
    nx = (n[:, None] * np.float32(interpolation[0]))
    my = (m[None, :] * np.float32(interpolation[1]))
    kx = nx * reciprocal[0, 0] + my * reciprocal[1, 0]
    ky = nx * reciprocal[0, 1] + my * reciprocal[1, 1]

    mask = kx**2 + ky**2 < (cutoff / 1.0e3 / wavelength) ** 2
    kx = kx[mask].astype(np.float32)
    ky = ky[mask].astype(np.float32)
    return xp.asarray([kx, ky]).T


def plane_waves(
    wave_vectors: np.ndarray,
    extent: Tuple[float, float],
    gpts: Tuple[int, int],
    reverse: bool = False,
    cell: np.ndarray = None,
) -> np.ndarray:
    """Build the plane waves ``exp(2 pi i g . r)`` sampled on the grid for each Cartesian
    wave vector ``g``. For a non-orthogonal ``cell`` (2x2, rows are the in-plane lattice
    vectors) the grid points sit at ``r = i a1 + j a2`` with sampling vectors
    ``a1 = cell[0] / gpts[0]``, ``a2 = cell[1] / gpts[1]``, so the phase stays separable in
    the pixel indices via ``g . a1`` and ``g . a2``. When ``cell`` is ``None`` the grid is
    orthogonal and the result is bit-identical to the original implementation."""
    xp = get_array_module(wave_vectors)

    sign = -1.0 if reverse else 1.0

    if cell is None:
        x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
        y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)

        array = complex_exponential(
            sign * 2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]
        ) * complex_exponential(
            sign * 2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]
        )

        return array

    cell = np.asarray(cell, dtype=float)
    a1 = cell[0] / gpts[0]  # sampling vector along axis 0 (Cartesian x, y)
    a2 = cell[1] / gpts[1]  # sampling vector along axis 1

    # per-pixel phase increment along each grid axis for every wave vector
    phase_i = (wave_vectors[:, 0] * a1[0] + wave_vectors[:, 1] * a1[1]).astype(
        np.float32
    )
    phase_j = (wave_vectors[:, 0] * a2[0] + wave_vectors[:, 1] * a2[1]).astype(
        np.float32
    )

    i = xp.arange(gpts[0], dtype=np.float32)
    j = xp.arange(gpts[1], dtype=np.float32)

    array = complex_exponential(
        sign * 2 * np.pi * phase_i[:, None, None] * i[None, :, None]
    ) * complex_exponential(
        sign * 2 * np.pi * phase_j[:, None, None] * j[None, None, :]
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
