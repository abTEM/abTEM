from typing import Tuple

import numba as nb
import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength
from abtem.core.fft import ifft2
from abtem.core.grid import spatial_frequencies
from abtem.waves.natural_neighbors import pairwise_weights
from abtem.waves.transfer import CTF

def beamlet_weights(parent_wave_vectors, wave_vectors, gpts, sampling):
    kx, ky = spatial_frequencies(gpts, sampling)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.asarray((kx.ravel(), ky.ravel())).T

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12,12))
    # plt.plot(*k.T,'r.')
    # plt.plot(*wave_vectors.T, 'b.')
    # plt.show()
    # sss

    indices = np.argmax(np.all(np.isclose(wave_vectors[:, None, :], k), axis=2), axis=1)
    weights = np.zeros((len(parent_wave_vectors),) + kx.shape)

    point_weights = pairwise_weights(parent_wave_vectors, wave_vectors)
    for i, j in enumerate(indices):
        k, m = np.unravel_index(j, kx.shape)
        weights[:, k, m] = point_weights[:, i]

    return weights


def beamlet_basis(ctf, parent_wave_vectors, wave_vectors, gpts, sampling):
    basis = ctf.evaluate_on_grid(gpts=gpts, sampling=sampling)
    basis = beamlet_weights(parent_wave_vectors, wave_vectors, gpts, sampling) * basis / np.sqrt(len(wave_vectors))
    basis = ifft2(basis)
    basis = np.fft.fftshift(basis, axes=(1, 2))
    return basis


def remove_tilt(array, planewave_cutoff, extent, gpts, energy, interpolation, partitions, accumulated_defocus,
                block_info=None):
    xp = get_array_module(array)

    if block_info is None:
        start, end = 0, array.shape[-3]
    else:
        start, end = block_info[0]['array-location'][-3]

    if partitions is None:
        wave_vectors = prism_wave_vectors(planewave_cutoff, extent, energy, interpolation)
    else:
        wave_vectors = partitioned_prism_wave_vectors(planewave_cutoff, extent, energy, num_rings=partitions)

    wave_vectors = wave_vectors[start:end]

    wavelength = energy2wavelength(energy)

    alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * wavelength
    phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

    ctf_coefficients = CTF(defocus=accumulated_defocus, energy=energy).evaluate(alpha, phi)
    ctf_coefficients = np.expand_dims(ctf_coefficients, tuple(range(len(array.shape) - 3)) + (-2, -1))

    array = array * plane_waves(wave_vectors, extent, gpts, reverse=True) * ctf_coefficients
    return array


def interpolate_full(array, parent_wave_vectors, wave_vectors, extent, gpts, energy, defocus=0.):
    interpolated_array = plane_waves(wave_vectors, extent, gpts)

    weights = pairwise_weights(parent_wave_vectors, wave_vectors)

    alpha = np.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * energy2wavelength(energy)
    phi = np.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

    interpolated_array *= CTF(defocus=-defocus, energy=energy).evaluate(alpha, phi)[:, None, None]

    for i, plane_wave in enumerate(interpolated_array):
        plane_wave *= (array * weights[:, i, None, None]).sum(0)

    return interpolated_array


@nb.jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def reduce_beamlets_nearest_no_interpolation(waves, basis, parent_s_matrix, shifts):
    assert waves.shape[0] == shifts.shape[0]
    assert len(shifts.shape) == 2
    # assert basis.shape == parent_s_matrix.shape
    # assert waves.shape[1:] == parent_s_matrix.shape[:-1]

    for i in nb.prange(waves.shape[0]):
        for j in range(waves.shape[1]):
            for k in range(waves.shape[2]):
                # waves[i, j, k] = np.dot(basis[j, k, :],
                #                         parent_s_matrix[
                #                         (j + shifts[i, 0]) % parent_s_matrix.shape[0],
                #                         (k + shifts[i, 1]) % parent_s_matrix.shape[1], :])
                waves[i, j, k] = np.dot(basis[:, j, k],
                                        parent_s_matrix[:,
                                        (j + shifts[i, 0]) % parent_s_matrix.shape[1],
                                        (k + shifts[i, 1]) % parent_s_matrix.shape[2]])
    return waves



def partitioned_prism_wave_vectors(cutoff, extent, energy, num_rings, num_points_per_ring=6, xp=np):
    wavelength = energy2wavelength(energy)

    rings = [xp.array((0., 0.))]
    if num_rings == 1:
        raise NotImplementedError()

    n = num_points_per_ring
    for r in np.linspace(cutoff / (num_rings - 1), cutoff, num_rings - 1):
        angles = xp.arange(n, dtype=np.int32) * 2 * np.pi / n + np.pi / 2
        kx = xp.round(r * xp.sin(angles) / 1000. / wavelength * extent[0]) / extent[0]
        ky = xp.round(r * xp.cos(-angles) / 1000. / wavelength * extent[1]) / extent[1]
        n += num_points_per_ring
        rings.append(xp.array([kx, ky]).T)

    return xp.vstack(rings).astype(xp.float32)


def array_row_intersection(a, b):
    tmp = np.all(np.isclose(np.swapaxes(a[:, :, None], 1, 2), b), axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)
