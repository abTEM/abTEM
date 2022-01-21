from typing import Tuple

import numpy as np
from ase.data import chemical_symbols

from abtem.core.backend import get_array_module
from abtem.core.fft import fft2, ifft2, fft2_convolve
from abtem.core.grid import spatial_frequencies, polar_spatial_frequencies
from abtem.potentials.parametrizations.base import Parametrization


def _sinc(gpts: Tuple[int, int], sampling: Tuple[float, float], xp):
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp)
    sinc = np.sinc(np.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2))
    return sinc * sampling[0] * sampling[1]


def calculate_scattering_factor(gpts: Tuple[int, int],
                                sampling: Tuple[float, float],
                                number: int,
                                parametrization: Parametrization,
                                xp: str = 'numpy',
                                ) -> np.ndarray:
    xp = get_array_module(xp)
    # parametrization = parametrization_names[parametrization]
    # parameters = parametrization.load_parameters()

    k, _ = polar_spatial_frequencies(gpts, sampling, xp=xp)
    scattering_factors = xp.zeros(gpts, dtype=np.float32)

    # for i, number in enumerate(atomic_numbers):
    f = parametrization.projected_scattering_factor(k, chemical_symbols[number])

    # scattering_factors[i] = f

    return f / _sinc(gpts, sampling, xp)


def superpose_deltas(positions: np.ndarray, array: np.ndarray, slice_index) -> np.ndarray:
    xp = get_array_module(positions)
    shape = array.shape

    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    x = positions[:, 0] - rows
    y = positions[:, 1] - cols
    xy = x * y

    if slice_index is None:
        i = xp.array([rows, (rows + 1) % shape[0]] * 2)
        j = xp.array([cols] * 2 + [(cols + 1) % shape[1]] * 2)
        v = xp.array([1 + xy - y - x, x - xy, y - xy, xy])
        array[i, j] += v
    else:
        raise NotImplementedError

    # array[slice_idx, rows, cols] += 1 + xy - y - x  # (1 - x) * (1 - y)
    # array[slice_idx, (rows + 1) % shape[1], cols] += x - xy
    # array[slice_idx, rows, (cols + 1) % shape[2]] += y - xy
    # array[slice_idx, (rows + 1) % shape[1], (cols + 1) % shape[2]] += xy

    return array


def infinite_potential_projections(atoms, shape, sampling, scattering_factors, slice_index=None):
    xp = get_array_module(list(scattering_factors.values())[0])

    if len(atoms) == 0:
        return xp.zeros(shape, dtype=xp.float32)

    if slice_index is None:
        shape = shape[1:]

    array = xp.zeros(shape, dtype=xp.complex64)
    positions = xp.asarray(atoms.positions[:, :2] / sampling, dtype=xp.float32)

    unique = np.unique(atoms.numbers)
    if len(unique) > 1:
        temp = xp.zeros_like(array, dtype=np.complex64)

        for i, number in enumerate(unique):
            mask = atoms.numbers == number

            if i > 0:
                temp[:] = 0.

            if not np.any(mask):
                continue

            if slice_index is not None:
                masked_slice_index = slice_index[mask]
            else:
                masked_slice_index = None

            temp = superpose_deltas(positions[mask], temp, masked_slice_index)
            array += fft2(temp, overwrite_x=True) * scattering_factors[number]

        array = ifft2(array, overwrite_x=True).real
    else:
        superpose_deltas(positions, array, slice_index=slice_index)
        array = fft2_convolve(array, scattering_factors[unique[0]], overwrite_x=True).real

    if len(array.shape) == 2:
        array = array[None]

    return array
