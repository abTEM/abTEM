from typing import Tuple, Sequence

import numpy as np
from ase.data import chemical_symbols

from abtem.core.backend import get_array_module
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import spatial_frequencies, polar_spatial_frequencies
from abtem.potentials.parametrizations import names as parametrization_names


def _sinc(gpts: Tuple[int, int], sampling: Tuple[float, float], xp):
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp)
    sinc = np.sinc(np.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2))
    return sinc * sampling[0] * sampling[1]


def calculate_scattering_factor(gpts: Tuple[int, int],
                                sampling: Tuple[float, float],
                                number: int,
                                xp: str = 'numpy',
                                parametrization: str = 'kirkland') -> np.ndarray:
    xp = get_array_module(xp)
    parametrization = parametrization_names[parametrization]
    parameters = parametrization.load_parameters()

    k, _ = polar_spatial_frequencies(gpts, sampling, xp=xp)
    #scattering_factors = xp.zeros(gpts, dtype=np.float32)

    #for i, number in enumerate(atomic_numbers):
    f = parametrization.projected_scattering_factor(k, parameters[chemical_symbols[number]])
    #scattering_factors[i] = f

    return f / _sinc(gpts, sampling, xp)


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


def infinite_potential_projections(atoms, shape, sampling, scattering_factors, slice_index=None):
    xp = get_array_module(list(scattering_factors.values())[0])

    if len(atoms) == 0:
        return xp.zeros(shape, dtype=xp.float32)

    if slice_index is None:
        slice_index = np.zeros(len(atoms), dtype=int)

    array = xp.zeros(shape, dtype=xp.complex64)
    positions = xp.asarray(atoms.positions[:, :2] / sampling)
    temp = xp.zeros_like(array, dtype=np.complex64)

    unique = np.unique(atoms.numbers)
    for i, number in enumerate(unique):
        if len(unique) > 1:
            if i > 0:
                temp[:] = 0.

            mask = atoms.numbers == number

            if not np.any(mask):
                continue

            temp = superpose_deltas(positions[mask], slice_index[mask], temp)
        else:
            temp = superpose_deltas(positions, slice_index, temp)

        array += fft2(temp, overwrite_x=False) * scattering_factors[number]

    array = ifft2(array, overwrite_x=False).real
    return array
