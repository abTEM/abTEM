from typing import Tuple, Sequence

import numpy as np
from ase.data import chemical_symbols

from abtem.core.backend import get_array_module
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import spatial_frequencies, polar_spatial_frequencies
from abtem.potentials.parametrizations import names as parametrization_names


def _sinc(gpts: Tuple[int, int], sampling: Tuple[float, float], xp):
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp, delayed=False)
    sinc = np.sinc(np.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2))
    return sinc * sampling[0] * sampling[1]


def calculate_scattering_factors(gpts: Tuple[int, int],
                                 sampling: Tuple[float, float],
                                 atomic_numbers: Sequence[int],
                                 xp: str = 'numpy',
                                 parametrization: str = 'kirkland') -> np.ndarray:
    xp = get_array_module(xp)
    parametrization = parametrization_names[parametrization]
    parameters = parametrization.load_parameters()

    k, _ = polar_spatial_frequencies(gpts, sampling, delayed=False, xp=xp)
    scattering_factors = xp.zeros((len(atomic_numbers),) + gpts, dtype=np.float32)

    for i, number in enumerate(atomic_numbers):
        f = parametrization.projected_scattering_factor(k, parameters[chemical_symbols[number]])
        scattering_factors[i] = f

    return scattering_factors / _sinc(gpts, sampling, xp)


def superpose_deltas(positions: np.ndarray, slice_idx: np.ndarray, array: np.ndarray) -> np.ndarray:
    xp = get_array_module(positions)
    shape = array.shape

    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[slice_idx, rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[slice_idx, (rows + 1) % shape[1], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[slice_idx, rows, (cols + 1) % shape[2]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[slice_idx, (rows + 1) % shape[1], (cols + 1) % shape[2]] += (rows - positions[:, 0]) * (
            cols - positions[:, 1])

    return array


def infinite_potential_projections(positions, numbers, slice_idx, shape, sampling, scattering_factors):
    xp = get_array_module(scattering_factors)

    if len(positions) == 0:
        return xp.zeros(shape, dtype=xp.float32)

    array = xp.zeros(shape, dtype=xp.complex64)
    positions = xp.asarray(positions / sampling)
    temp = xp.zeros_like(array, dtype=np.complex64)

    unique = np.unique(numbers)
    for i, number in enumerate(unique):
        if len(unique) > 1:
            if i > 0:
                temp[:] = 0.

            mask = numbers == number

            if not np.any(mask):
                continue

            temp = superpose_deltas(positions[mask], slice_idx[mask], temp)
        else:
            temp = superpose_deltas(positions, slice_idx, temp)

        array += fft2(temp, overwrite_x=False) * scattering_factors[i]

    array = ifft2(array, overwrite_x=False).real
    return array
