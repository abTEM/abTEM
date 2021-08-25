import numpy as np
from ase.data import chemical_symbols
from abtem.potentials.parametrizations import names as parametrization_names
from abtem.potentials.utils import kappa
from abtem.basic.backend import get_array_module
from abtem.basic.fft import fft2, ifft2
from abtem.basic.grid import spatial_frequencies, polar_spatial_frequencies


def _sinc(gpts, sampling, xp):
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp, delayed=False)
    sinc = np.sinc(np.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2))
    return sinc * sampling[0] * sampling[1] * kappa


def calculate_scattering_factors(gpts, sampling, atomic_numbers, xp='numpy', parametrization='kirkland'):
    xp = get_array_module(xp)
    parametrization = parametrization_names[parametrization]
    parameters = parametrization.load_parameters()

    k, _ = polar_spatial_frequencies(gpts, sampling, delayed=False, xp=xp)
    scattering_factors = xp.zeros((len(atomic_numbers),) + gpts, dtype=np.float32)

    for i, number in enumerate(atomic_numbers):
        f = parametrization.projected_scattering_factor(k, parameters[chemical_symbols[number]])
        scattering_factors[i] = f

    return scattering_factors / _sinc(gpts, sampling, xp)


def superpose_deltas(positions, slice_idx, array):
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


def infinite_potential_projections(positions, numbers, slice_idx, shape, sampling, scattering_factors, unique):
    xp = get_array_module(scattering_factors)

    if len(positions) == 0:
        return xp.zeros(shape, dtype=xp.float32)

    array = xp.zeros(shape, dtype=xp.complex64)

    positions = xp.asarray(positions / sampling)

    temp = xp.zeros_like(array, dtype=np.complex64)
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

    return ifft2(array, overwrite_x=False).real
