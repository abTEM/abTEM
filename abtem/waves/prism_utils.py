from typing import Tuple

import dask.array
import numpy as np
from scipy.spatial import Delaunay
import dask
import dask.array as da

from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import energy2wavelength
from abtem.core.grid import spatial_frequencies
from abtem.waves.natural_neighbors import find_natural_neighbors, natural_neighbor_weights
from abtem.waves.transfer import CTF
from abtem.core.utils import generate_chunks


def prism_wave_vectors(cutoff: float, extent: Tuple[float, float],
                       energy: float, interpolation: Tuple[int, int]) -> np.ndarray:
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
    return np.asarray((kx, ky)).T


def partitioned_prism_wave_vectors(cutoff, extent, energy, num_rings, num_points_per_ring=6):
    wavelength = energy2wavelength(energy)

    rings = [np.array((0., 0.))]

    if num_rings == 1:
        raise NotImplementedError()

    n = num_points_per_ring
    for r in np.linspace(cutoff / (num_rings - 1), cutoff, num_rings - 1):
        angles = np.arange(n, dtype=np.int32) * 2 * np.pi / n + np.pi / 2
        kx = np.round(r * np.sin(angles) / 1000. / wavelength * extent[0]) / extent[0]
        ky = np.round(r * np.cos(-angles) / 1000. / wavelength * extent[1]) / extent[1]
        n += num_points_per_ring
        rings.append(np.array([kx, ky]).T)

    return np.vstack(rings)


def plane_waves(wave_vectors: np.ndarray, extent: Tuple[float, float], gpts: Tuple[int, int],
                normalize: bool = False, reverse: bool = False) -> np.ndarray:
    xp = get_array_module(wave_vectors)
    x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
    y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)

    sign = -1. if reverse else 1.

    array = (complex_exponential(sign * 2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]) *
             complex_exponential(sign * 2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]))

    if normalize == 'amplitudes':
        array = array / xp.sqrt((xp.abs(array[0]) ** 2).sum())
    elif normalize:
        raise RuntimeError()

    return array


def array_row_intersection(a, b):
    tmp = np.prod(np.isclose(np.swapaxes(a[:, :, None], 1, 2), b), axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)


def beamlet_weights(parent_wave_vectors, wave_vectors, gpts, sampling):
    n = len(parent_wave_vectors)
    tri = Delaunay(parent_wave_vectors)

    kx, ky = spatial_frequencies(gpts, sampling)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')

    k = np.asarray((kx.ravel(), ky.ravel())).T

    weights = np.zeros((n,) + kx.shape)

    intersection = np.where(array_row_intersection(k, wave_vectors))[0]
    members, circumcenters = find_natural_neighbors(tri, k)
    for i in intersection:
        j, l = np.unravel_index(i, kx.shape)
        weights[:, j, l] = natural_neighbor_weights(parent_wave_vectors, k[i], tri, members[i], circumcenters)

    return weights


def beamlet_basis(ctf, parent_wave_vectors, wave_vectors, gpts, sampling):
    basis = ctf.evaluate_on_grid(gpts=gpts, sampling=sampling)
    basis = beamlet_weights(parent_wave_vectors, wave_vectors, gpts, sampling) * basis
    return np.fft.ifft2(basis, axes=(1, 2))


def remove_s_matrix_beam_tilt(s_matrix_array):
    xp = get_array_module(s_matrix_array.array)

    def remove_tilt(array, cutoff, extent, gpts, energy, interpolation, partitions, accumulated_defocus,
                    block_info=None):
        start, end = block_info[0]['array-location'][-3]
        if partitions is None:
            wave_vectors = prism_wave_vectors(cutoff, extent, energy, interpolation)
        else:
            wave_vectors = partitioned_prism_wave_vectors(cutoff, extent, energy, num_rings=partitions)

        wave_vectors = wave_vectors[start:end]

        wavelength = energy2wavelength(energy)

        alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * wavelength
        phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

        ctf_coefficients = CTF(defocus=accumulated_defocus, energy=energy).evaluate(alpha, phi)

        ctf_coefficients = np.expand_dims(ctf_coefficients, tuple(range(len(array.shape) - 3)) + (-2, -1))

        array = array * plane_waves(wave_vectors, extent, gpts, reverse=True) * ctf_coefficients
        return array

    if s_matrix_array.is_lazy:
        array = s_matrix_array.array.map_blocks(remove_tilt,
                                                cutoff=s_matrix_array.planewave_cutoff,
                                                extent=s_matrix_array.extent,
                                                gpts=s_matrix_array.gpts,
                                                energy=s_matrix_array.energy,
                                                interpolation=s_matrix_array.interpolation,
                                                partitions=s_matrix_array.partitions,
                                                accumulated_defocus=s_matrix_array.accumulated_defocus,
                                                meta=xp.array((), dtype=xp.complex64))
    else:
        array = remove_tilt(s_matrix_array.array,
                            cutoff=s_matrix_array.planewave_cutoff,
                            extent=s_matrix_array.extent,
                            gpts=s_matrix_array.gpts,
                            energy=s_matrix_array.energy,
                            interpolation=s_matrix_array.interpolation,
                            partitions=s_matrix_array.partitions,
                            accumulated_defocus=s_matrix_array.accumulated_defocus)

    new_copy = s_matrix_array.copy(copy_array=False)
    new_copy._array = array
    return new_copy


def interpolate_full(s_matrix_array, chunks):
    s_matrix_array = remove_s_matrix_beam_tilt(s_matrix_array)

    wave_vectors = prism_wave_vectors(s_matrix_array.planewave_cutoff,
                                      s_matrix_array.extent,
                                      s_matrix_array.energy,
                                      s_matrix_array.interpolation)

    def calculate_weights(parent_wave_vectors, wave_vectors):
        tri = Delaunay(parent_wave_vectors)
        members, circumcenters = find_natural_neighbors(tri, wave_vectors)
        weights = np.zeros((len(parent_wave_vectors), len(wave_vectors)))

        for i, p in enumerate(wave_vectors):
            weights[:, i] = natural_neighbor_weights(s_matrix_array.wave_vectors, p, tri, members[i], circumcenters)

        return weights

    def interpolate_chunk(array, parent_wave_vectors, planewave_cutoff, extent, gpts, energy, interpolation,
                          accumulated_defocus, start, end):
        wave_vectors = prism_wave_vectors(planewave_cutoff, extent, energy, interpolation)

        wave_vectors = wave_vectors[start:end]
        interpolated_array = plane_waves(wave_vectors, extent, gpts)

        weights = calculate_weights(parent_wave_vectors, wave_vectors)

        alpha = np.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * s_matrix_array.wavelength
        phi = np.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

        interpolated_array *= CTF(defocus=-accumulated_defocus, energy=energy).evaluate(alpha, phi)[:, None, None]

        for i, plane_wave in enumerate(interpolated_array):
            plane_wave *= (array * weights[:, i, None, None]).sum(0)

        return interpolated_array

    arrays = []
    for start, end in generate_chunks(len(wave_vectors), chunks=chunks):
        array = dask.delayed(interpolate_chunk)(s_matrix_array.array,
                                                s_matrix_array.wave_vectors,
                                                s_matrix_array.planewave_cutoff,
                                                s_matrix_array.extent,
                                                s_matrix_array.gpts,
                                                s_matrix_array.energy,
                                                s_matrix_array.interpolation,
                                                s_matrix_array.accumulated_defocus,
                                                start,
                                                end)

        array = da.from_delayed(array, shape=(end - start,) + s_matrix_array.gpts, dtype=np.complex64)
        arrays.append(array)

    array = da.concatenate(arrays)

    return s_matrix_array.__class__(array,
                                    energy=s_matrix_array.energy,
                                    wave_vectors=wave_vectors,
                                    interpolation=s_matrix_array.interpolation,
                                    planewave_cutoff=s_matrix_array.planewave_cutoff,
                                    extent=s_matrix_array.extent,
                                    sampling=s_matrix_array.sampling,
                                    tilt=s_matrix_array.tilt,
                                    antialias_aperture=s_matrix_array.antialias_aperture,
                                    device=s_matrix_array._device,
                                    extra_axes_metadata=s_matrix_array._extra_axes_metadata,
                                    metadata=s_matrix_array.metadata)
