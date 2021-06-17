import dask.array as da
import numpy as np

from abtem.potentials.parametrizations import get_parameterization
from abtem.potentials.utils import kappa
from abtem.utils.backend import get_array_module
from abtem.utils.coordinates import spatial_frequencies
from abtem.utils.fft import fft2, ifft2


def scattering_factor(gpts, sampling, atomic_number, xp='numpy', parametrization='kirkland'):
    kx, ky, k = spatial_frequencies(gpts, sampling, return_grid=True, return_radial=True, xp=xp)
    sinc = da.sinc(da.sqrt((kx * sampling[0]) ** 2 + (ky * sampling[1]) ** 2))
    parameters, funcs = get_parameterization(parametrization)

    f = funcs['projected_fourier'](k, parameters[atomic_number])
    return f / (sinc * sampling[0] * sampling[1] * kappa)


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


def infinite_potential_projections(positions, numbers, slice_idx, shape, sampling, scattering_factors, xp='numpy'):
    xp = get_array_module(xp)

    if len(positions) == 0:
        return xp.zeros(shape, dtype=xp.float32)

    array = xp.zeros(shape, dtype=xp.complex64)

    positions = xp.asarray(positions / sampling)
    unique = np.unique(numbers)

    temp = np.zeros_like(array, dtype=np.complex64)
    for i, number in enumerate(unique):
        if len(unique) > 1:
            if i > 0:
                temp[:] = 0.

            temp = superpose_deltas(positions[numbers == number], slice_idx[numbers == number], temp)
        else:
            temp = superpose_deltas(positions, slice_idx, temp)

        array += fft2(temp, overwrite_x=False) * scattering_factors[number]

    return ifft2(array, overwrite_x=False).real
