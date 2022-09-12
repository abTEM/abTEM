from numbers import Number

import numpy as np
from ase.cell import Cell
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from abtem.core.utils import label_to_index
from abtem.structures.transform import (
    is_cell_hexagonal,
    is_cell_orthogonal,
)


def central_bin_index(array):
    return array.shape[0] // 2, array.shape[1] // 2


def frequency_bin_indices(shape):
    x = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1 / shape[0])).astype(int)
    y = np.fft.fftshift(np.fft.fftfreq(shape[1], d=1 / shape[1])).astype(int)
    x, y = np.meshgrid(x, y, indexing="ij")
    return np.array([x.ravel(), y.ravel()]).T


def find_linearly_independent_row(array, row, tol: float = 1e-6):

    for other_row in array:
        A = np.row_stack([row, other_row])
        U, s, V = np.linalg.svd(A)
        if np.all(np.abs(s) > tol):
            break
    else:
        raise RuntimeError()

    return other_row


def find_independent_spots(array):
    spots = array > array.max() * 1e-2
    half = central_bin_index(array)
    spots = spots[half[0] :, half[1] :]

    spots = np.array(np.where(spots)).T
    # print(spots)

    # plt.imshow(spots)
    # plt.show()

    # print(array )

    # sss
    # ind = np.array(np.unravel_index(np.argsort(-array, axis=None), array.shape)).T
    # ind = ind - central_bin_index(array)
    spot_0 = spots[0]
    spot_1 = find_linearly_independent_row(spots[1:], spot_0)
    return spot_0, spot_1


def planar_spacing_from_bin_index(index, sampling):
    d = 1 / np.sqrt((index[0] * sampling[0]) ** 2 + (index[1] * sampling[1]) ** 2)
    return d


def planar_angle_from_bin_indices(index1, index2, sampling):
    v1 = index1[0] * sampling[0], index1[1] * sampling[1]
    v2 = index2[0] * sampling[0], index2[1] * sampling[1]
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def orthorhombic_spacings(indices, d):
    g = (
        indices[:, None, None] ** 2 / d[0] ** 2
        + indices[None, :, None] ** 2 / d[1] ** 2
        + indices[None, None] ** 2 / d[2] ** 2
    )

    planes = np.zeros_like(g)
    planes[g > 0.0] = 1 / np.sqrt(g[g > 0.0])
    return planes


def closest_indices(array, value):
    difference = np.abs(array - value)
    return np.array(np.where(difference == np.min(difference))).T


def spacing_consistent_miller_indices(spacing, cell_edges):
    max_index = int(np.ceil(max(cell_edges) / spacing)) + 1

    indices = np.arange(-max_index, max_index + 1, 1)
    d = orthorhombic_spacings(indices, cell_edges)
    return np.array([indices[i] for i in closest_indices(d, spacing)])


def planar_angle(hkl1, hkl2, cell_edges):
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2
    a, b, c = cell_edges
    d1 = 1 / a ** 2 * h1 ** 2 + 1 / b ** 2 * k1 ** 2 + 1 / c ** 2 * l1 ** 2
    d2 = 1 / a ** 2 * h2 ** 2 + 1 / b ** 2 * k2 ** 2 + 1 / c ** 2 * l2 ** 2
    d3 = 1 / a ** 2 * h1 * h2 + 1 / b ** 2 * k1 * k2 + 1 / c ** 2 * l1 * l2
    return np.arccos(d3 / np.sqrt(d1 * d2))


def find_consistent_miller_index_pair(spacing_1, spacing_2, angle, cell_edges):
    hkl1 = spacing_consistent_miller_indices(spacing_1, cell_edges)[0]
    hkl2 = spacing_consistent_miller_indices(spacing_2, cell_edges)
    angles = np.array([planar_angle(hkl1, x, cell_edges) for x in hkl2])
    return hkl1, hkl2[np.argmin(np.abs(angles - angle))]


def bin_index_to_orthorhombic_miller(array, sampling, cell_edges):
    bin1, bin2 = find_independent_spots(array)
    spacing1 = planar_spacing_from_bin_index(bin1, sampling)
    spacing2 = planar_spacing_from_bin_index(bin2, sampling)
    angle = planar_angle_from_bin_indices(bin1, bin2, sampling)

    hkl1, hkl2 = find_consistent_miller_index_pair(
        spacing1, spacing2, angle, cell_edges
    )
    return (bin1, bin2), (hkl1, hkl2)


def validate_cell_edges(cell):

    if isinstance(cell, Number):
        cell = [cell]

    if not isinstance(cell, Cell):
        if len(cell) == 1:
            cell_edges = cell * 3
        elif len(cell) == 2:
            cell_edges = [cell[0]] * 2 + [cell[1]]
        elif len(cell) != 3:
            raise RuntimeError()
        else:
            raise RuntimeError()

        hexagonal = False

    elif is_cell_hexagonal(cell):
        lengths = cell.lengths()
        cell_edges = [lengths[0], np.sqrt(3) * lengths[1], lengths[2]]
        hexagonal = True
    elif is_cell_orthogonal(cell):
        cell_edges = list(np.diag(cell))
        hexagonal = False
    else:
        raise RuntimeError()

    return cell_edges, hexagonal


def miller_to_miller_bravais(hkl):
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    HKIL = np.zeros((len(hkl), 4), dtype=int)

    HKIL[:, 0] = 2 * h - k
    HKIL[:, 1] = 2 * k - h
    HKIL[:, 2] = -HKIL[:, 0] - HKIL[:, 1]
    HKIL[:, 3] = l
    return HKIL


def map_all_bin_indices_to_miller_indices(array, sampling, cell, tolerance=1e-6):
    cell_edges, hexagonal = validate_cell_edges(cell)
    (v1, v2), (u1, u2) = bin_index_to_orthorhombic_miller(array, sampling, cell_edges)

    bins = frequency_bin_indices(array.shape)
    A = np.linalg.inv(np.array([v1, v2]))
    bins_v = np.dot(bins, A)
    hkl = np.dot(bins_v, [u1, u2])
    mask = np.all(np.abs(hkl - np.round(hkl)) < tolerance, axis=1)
    hkl = hkl[mask].astype(int)
    bins = bins[mask]

    if hexagonal:
        hkl[:, 1] = hkl[:, :-1].sum(axis=1) / 2

    return bins, hkl


def equivalent_miller_indices(hkl, hexagonal=True):
    if hexagonal:
        hkl = miller_to_miller_bravais(hkl)

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


def split_at_threshold(values, threshold):
    order = np.argsort(values)
    max_value = values.max()

    split = (np.diff(values[order]) > (max_value * threshold)) * (
        np.diff(values[order]) > 1e-6
    )

    split = np.insert(split, 0, False)
    return np.cumsum(split)[np.argsort(order)]


def find_equivalent_spots(hkl, intensities, intensity_split=1.0, hexagonal: bool = True):

    labels = equivalent_miller_indices(hkl, hexagonal)

    spots = np.zeros(len(hkl), dtype=bool)
    for indices in label_to_index(labels):
        sub_labels = split_at_threshold(intensities[indices], intensity_split)
        for sub_indices in label_to_index(sub_labels):
            order = np.lexsort(np.rot90(hkl[indices][sub_indices]))
            spots[indices[sub_indices[order][-1]]] = True

    return spots


def tabulate_diffraction_pattern(
    diffraction_pattern,
    cell,
    return_data_frame: bool = False,
    normalize: bool = True,
    spot_threshold: float = 0.01,
    intensity_split=1.0,
):
    # if len(diffraction_pattern.ensemble_shape) > 0:
    # raise NotImplementedError("tabulating not implemented for ensembles, select a single pattern by indexing")

    bins, hkl = map_all_bin_indices_to_miller_indices(
        diffraction_pattern.array, diffraction_pattern.sampling, cell
    )

    intensities = diffraction_pattern.select_frequency_bin(bins)

    _, hexagonal = validate_cell_edges(cell)
    include = find_equivalent_spots(hkl, intensities=intensities, hexagonal=hexagonal, intensity_split=intensity_split)
    hkl, bins, intensities = hkl[include], bins[include], intensities[include]

    if hexagonal:
        hkl = miller_to_miller_bravais(hkl)

    order = np.lexsort(np.rot90(hkl))
    hkl, bins, intensities = hkl[order], bins[order], intensities[order]

    table = {
        "".join(map(str, list(hkli))): intensity
        for intensity, hkli in zip(intensities, hkl)
    }

    max_intensity = max(table.values())

    if normalize is True:
        normalization = max_intensity
    elif isinstance(normalize, str):
        normalization = table[normalize]
    else:
        normalization = 1.0

    table = {
        key: [value / normalization]
        for key, value in table.items()
        if value / max_intensity > spot_threshold
    }

    if return_data_frame:
        import pandas as pd

        return pd.DataFrame.from_dict(table)

    return table
