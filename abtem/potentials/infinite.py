import dask.array as da
import numpy as np

from abtem.potentials.parametrizations import get_parameterization
from abtem.potentials.utils import kappa
from abtem.utils.backend import get_array_module
from abtem.utils.coordinates import spatial_frequencies
from abtem.utils.fft import fft2_convolve


def scattering_factor(gpts, sampling, atomic_number, xp='numpy', parametrization='kirkland'):
    kx, ky, k = spatial_frequencies(gpts, sampling, return_grid=True, return_radial=True, xp=xp)
    sinc = da.sinc(da.sqrt((kx * sampling[0]) ** 2 + (ky * sampling[1]) ** 2))
    parameters, funcs = get_parameterization(parametrization)

    f = funcs['projected_fourier'](k, parameters[atomic_number])
    return f / (sinc * sampling[0] * sampling[1] * kappa)


def superpose_deltas(positions, slice_idx, shape):
    xp = get_array_module(positions)
    array = np.zeros(shape, dtype=np.float32)

    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[slice_idx, rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[slice_idx, (rows + 1) % shape[1], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[slice_idx, rows, (cols + 1) % shape[2]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[slice_idx, (rows + 1) % shape[1], (cols + 1) % shape[2]] += (rows - positions[:, 0]) * (
            cols - positions[:, 1])

    return array


def infinite_potential_projections(positions, numbers, slice_idx, shape, sampling, xp='numpy'):
    xp = get_array_module(xp)

    array = xp.zeros(shape, dtype=xp.float32)

    if len(positions) == 0:
        return array

    positions = xp.asarray(positions / sampling)
    unique = np.unique(numbers)

    for number in unique:
        if len(unique) > 1:
            temp = superpose_deltas(positions[numbers == number], slice_idx[numbers == number], shape)
        else:
            temp = superpose_deltas(positions, slice_idx, shape)

        f = scattering_factor(shape[1:], sampling, number, xp).compute()

        array += fft2_convolve(temp, f).real

    return array
