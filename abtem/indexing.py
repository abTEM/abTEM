from typing import Union, TYPE_CHECKING, List, Tuple

import numpy as np
from ase.cell import Cell
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from abtem.core.axes import AxisMetadata
from abtem.core.energy import energy2wavelength
from abtem.core.utils import label_to_index, CopyMixin
from abtem.visualize import (
    _show_indexed_diffraction_pattern,
    remove_annotations,
    add_miller_index_annotations,
)
import ipywidgets as widgets

if TYPE_CHECKING:
    from abtem.measurements import DiffractionPatterns

#
# def _frequency_bin_indices(shape):
#     x = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1 / shape[0])).astype(int)
#     y = np.fft.fftshift(np.fft.fftfreq(shape[1], d=1 / shape[1])).astype(int)
#     x, y = np.meshgrid(x, y, indexing="ij")
#     return np.array([x.ravel(), y.ravel()]).T
#
#
# def _find_linearly_independent_row(array, row, tol: float = 1e-6):
#     for other_row in array:
#         A = np.row_stack([row, other_row])
#         U, s, V = np.linalg.svd(A)
#         if np.all(np.abs(s) > tol):
#             break
#     else:
#         raise RuntimeError()
#
#     return other_row
#
#
# def _find_independent_spots(array):
#     spots = array > array.max() * 1e-2
#     half = array.shape[0] // 2, array.shape[1] // 2
#     spots = spots[half[0]:, half[1]:]
#
#     spots = np.array(np.where(spots)).T
#     intensities = array[half[0] + spots[:,0], half[1] + spots[:,1]]
#
#     spots = spots[np.argsort(-intensities)]
#     spot_0 = spots[0]
#     spot_1 = _find_linearly_independent_row(spots[1:], spot_0)
#     return spot_0, spot_1
#
#
# def _planar_spacing_from_bin_index(index, sampling):
#     d = 1 / np.sqrt((index[0] * sampling[0]) ** 2 + (index[1] * sampling[1]) ** 2)
#     return d
#
#
# def _planar_angle_from_bin_indices(index1, index2, sampling):
#     v1 = index1[0] * sampling[0], index1[1] * sampling[1]
#     v2 = index2[0] * sampling[0], index2[1] * sampling[1]
#     return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#
#
# def _orthorhombic_spacings(indices, d):
#     g = (
#             indices[:, None, None] ** 2 / d[0] ** 2
#             + indices[None, :, None] ** 2 / d[1] ** 2
#             + indices[None, None] ** 2 / d[2] ** 2
#     )
#
#     planes = np.zeros_like(g)
#     planes[g > 0.0] = 1 / np.sqrt(g[g > 0.0])
#     return planes
#
#
# def _closest_indices(array, value):
#     difference = np.abs(array - value)
#     return np.array(np.where(difference == np.min(difference))).T
#
#
# def _spacing_consistent_miller_indices(spacing, cell_edges):
#     max_index = int(np.ceil(max(cell_edges) / spacing)) + 1
#
#     indices = np.arange(-max_index, max_index + 1, 1)
#     d = _orthorhombic_spacings(indices, cell_edges)
#     return np.array([indices[i] for i in _closest_indices(d, spacing)])
#
#
# def _planar_angle(hkl1, hkl2, cell_edges):
#     h1, k1, l1 = hkl1
#     h2, k2, l2 = hkl2
#     a, b, c = cell_edges
#     d1 = 1 / a ** 2 * h1 ** 2 + 1 / b ** 2 * k1 ** 2 + 1 / c ** 2 * l1 ** 2
#     d2 = 1 / a ** 2 * h2 ** 2 + 1 / b ** 2 * k2 ** 2 + 1 / c ** 2 * l2 ** 2
#     d3 = 1 / a ** 2 * h1 * h2 + 1 / b ** 2 * k1 * k2 + 1 / c ** 2 * l1 * l2
#     return np.arccos(d3 / np.sqrt(d1 * d2))
#
#
# def _find_consistent_miller_index_pair(spacing_1, spacing_2, angle, cell_edges):
#     hkl1 = _spacing_consistent_miller_indices(spacing_1, cell_edges)[0]
#     hkl2 = _spacing_consistent_miller_indices(spacing_2, cell_edges)
#
#     angles = np.array([_planar_angle(hkl1, x, cell_edges) for x in hkl2])
#     hkl2 = hkl2[np.argmin(np.abs(angles - angle))]
#     return hkl1, hkl2
#
#
# def _bin_index_to_orthorhombic_miller(array, sampling, cell_edges):
#
#     bin1, bin2 = _find_independent_spots(array)
#
#     spacing1 = _planar_spacing_from_bin_index(bin1, sampling)
#     spacing2 = _planar_spacing_from_bin_index(bin2, sampling)
#     angle = _planar_angle_from_bin_indices(bin1, bin2, sampling)
#
#     hkl1, hkl2 = _find_consistent_miller_index_pair(
#         spacing1, spacing2, angle, cell_edges
#     )
#
#     return (bin1, bin2), (hkl1, hkl2)
#
#
# def _validate_cell_edges(cell):
#     if isinstance(cell, Number):
#         cell = [cell]
#
#     if not isinstance(cell, Cell):
#         if len(cell) == 1:
#             cell_edges = cell * 3
#         elif len(cell) == 2:
#             cell_edges = [cell[0]] * 2 + [cell[1]]
#         elif len(cell) != 3:
#             raise RuntimeError()
#         else:
#             raise RuntimeError()
#
#         hexagonal = False
#
#     elif is_cell_hexagonal(cell):
#         lengths = cell.lengths()
#         cell_edges = [lengths[0], np.sqrt(3) * lengths[1], lengths[2]]
#         hexagonal = True
#     elif is_cell_orthogonal(cell):
#         cell_edges = list(np.diag(cell))
#         hexagonal = False
#     else:
#         raise RuntimeError()
#
#     return cell_edges, hexagonal
#
#
#
# def _remap_vector_space(vector_space1, vector_space2, vectors):
#     A = np.linalg.inv(np.array(vector_space1))
#     bins_v = np.dot(vectors, A)
#     return np.dot(bins_v, vector_space2)
#
#
# def _map_all_bin_indices_to_miller_indices(array, sampling, cell):
#     cell_edges, hexagonal = _validate_cell_edges(cell)
#
#     (v1, v2), (u1, u2) = _bin_index_to_orthorhombic_miller(array, sampling, cell_edges)
#
#     bins = _frequency_bin_indices(array.shape)
#     hkl = _remap_vector_space((v1, v2), (u1, u2), bins)
#
#     # im = diffraction_patterns[-1].block_direct().to_cpu().array
#     #
#     # s = im > im.max() * .1
#     #
#     # s = np.array(np.where(s)).T - np.array(im.shape) // 2
#     #
#     # ss = s * diffraction_patterns.sampling
#     #
#     # d = np.linalg.norm(ss, axis=1)
#
#     # k = np.arange(15) * atoms.cell.reciprocal()[0, 0]
#     #
#     # D = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
#     #
#     # diff = np.abs(D - d[7])
#     # np.where(diff < diff.min() * 1.000000001)
#
#     # remove fractional planes
#     hkl = np.round(hkl).astype(int)
#     #mask = np.all(np.abs(hkl - np.round(hkl)) < 1e-3, axis=1)
#     #print((hkl == 4).sum())
#     #hkl = hkl[mask].astype(int)
#     #bins = bins[mask]
#
#     if hexagonal:
#         hkl[:, 1] = hkl[:, :-1].sum(axis=1) / 2
#         hkl = _miller_to_miller_bravais(hkl)
#
#     return bins, hkl


def get_frequency_bin_edges(diffraction_patterns):
    bin_edge_x = np.fft.fftshift(
        np.fft.fftfreq(
            diffraction_patterns.shape[-2], d=1 / diffraction_patterns.shape[-2]
        )
    ).astype(int)
    bin_edge_y = np.fft.fftshift(
        np.fft.fftfreq(
            diffraction_patterns.shape[-1], d=1 / diffraction_patterns.shape[-1]
        )
    ).astype(int)
    bin_edge_x = (
        bin_edge_x * diffraction_patterns.sampling[0]
        - diffraction_patterns.sampling[0] / 2
    )
    bin_edge_y = (
        bin_edge_y * diffraction_patterns.sampling[1]
        - diffraction_patterns.sampling[1] / 2
    )
    return bin_edge_x, bin_edge_y


def sphere_of_miller_index_grid_points(diffraction_patterns):
    max_index = min(diffraction_patterns.shape[-2:]) // 2

    hkl = np.meshgrid(*(np.arange(-max_index, max_index + 1),) * 3, indexing="ij")
    hkl = np.stack((hkl[0], hkl[1], hkl[2]), -1).reshape((-1, 3))

    return hkl[np.linalg.norm(hkl, axis=-1) < max_index]


def k_space_grid_points(hkl, cell):
    return (hkl[:, None] * cell.reciprocal()[None]).sum(-1)


def sagita(radius, chord):
    return radius - np.sqrt(radius**2 - (chord / 2) ** 2)


def digitize_k_space_grid(k_grid, diffraction_patterns):
    bin_edge_x, bin_edge_y = get_frequency_bin_edges(diffraction_patterns)

    n = np.digitize(k_grid[:, 0], bin_edge_x) - 1
    m = np.digitize(k_grid[:, 1], bin_edge_y) - 1

    nm = np.concatenate((n[:, None], m[:, None]), axis=1)

    return nm


def k_space_distances_to_ewald_sphere(k_grid, wavelength):
    k_norm = np.linalg.norm(k_grid[:, :2], axis=1)
    ewald_z = sagita(1 / wavelength, k_norm * 2)
    return ewald_z - k_grid[:, 2]


def _validate_cell(cell):
    if isinstance(cell, float):
        return Cell(np.diag([cell] * 3))
    else:
        return cell


def _index_diffraction_patterns(
    diffraction_patterns, cell, threshold, distance_threshold
):

    cell = _validate_cell(cell)

    shape = diffraction_patterns.shape[-2:]

    hkl = sphere_of_miller_index_grid_points(
        diffraction_patterns,
    )

    k = k_space_grid_points(hkl, cell)

    nm = digitize_k_space_grid(k, diffraction_patterns)

    mask = (
        np.all((nm > 0), axis=1)
        * (nm[:, 0] < diffraction_patterns.shape[-2])
        * (nm[:, 1] < diffraction_patterns.shape[-1])
    )

    k = k[mask]
    nm = nm[mask]
    hkl = hkl[mask]

    labels = np.ravel_multi_index(nm.T, shape)

    d_ewald = k_space_distances_to_ewald_sphere(k, diffraction_patterns.wavelength)

    ensemble_indices = tuple(range(len(diffraction_patterns.ensemble_shape)))
    max_intensities = diffraction_patterns.array.max(axis=ensemble_indices)

    selected_hkl = []
    intensities = []
    positions = []
    for label, indices in enumerate(label_to_index(labels)):

        if len(indices) == 0:
            continue

        n, m = np.unravel_index(label, shape)

        max_intensity = max_intensities[n, m]

        if max_intensity < threshold:
            continue

        if np.min(np.abs(d_ewald[indices])) > distance_threshold:
            continue

        min_index = np.argmin(np.abs(d_ewald[indices]))

        selected_hkl.append(hkl[indices][min_index])
        intensities.append(diffraction_patterns.array[..., n, m])
        positions.append(k[indices][min_index])

    return np.array(selected_hkl), np.array(intensities).T, np.array(positions)


def format_miller_indices(hkl):
    return "{} {} {}".format(*hkl)


def _miller_to_miller_bravais(hkl):
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    HKIL = np.zeros((len(hkl), 4), dtype=int)
    HKIL[:, 0] = 2 * h - k
    HKIL[:, 1] = 2 * k - h
    HKIL[:, 2] = -HKIL[:, 0] - HKIL[:, 1]
    HKIL[:, 3] = l

    # hkl[:, 1] = hkl[:, :-1].sum(axis=1) / 2
    # hkl = _miller_to_miller_bravais(hkl)

    return HKIL


def _equivalent_miller_indices(hkl):
    is_negation = np.zeros((len(hkl), len(hkl)), dtype=bool)

    for i in range(hkl.shape[1]):
        negated = hkl.copy()
        negated[:, i] = -negated[:, i]
        is_negation += np.all(hkl[:, None] == negated[None], axis=2)

    is_negation += np.all(hkl[:, None] == -hkl[None], axis=2)

    sorted = np.sort(hkl, axis=1)
    is_permutation = np.all(sorted[:, None] == sorted[None], axis=-1)

    is_connected = is_negation + is_permutation
    n, labels = connected_components(csr_matrix(is_connected))
    return labels


def _split_at_threshold(values, threshold):
    order = np.argsort(values)
    max_value = values.max()

    split = (np.diff(values[order]) > (max_value * threshold)) * (
        np.diff(values[order]) > 1e-6
    )

    split = np.insert(split, 0, False)
    return np.cumsum(split)[np.argsort(order)]


def _find_equivalent_spots(hkl, intensities, intensity_split: float = 1.0):
    labels = _equivalent_miller_indices(hkl)

    spots = np.zeros(len(hkl), dtype=bool)
    for indices in label_to_index(labels):
        sub_labels = _split_at_threshold(intensities[indices], intensity_split)
        for sub_indices in label_to_index(sub_labels):
            order = np.lexsort(np.rot90(hkl[indices][sub_indices]))
            spots[indices[sub_indices[order][-1]]] = True

    return spots


