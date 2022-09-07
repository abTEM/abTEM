from numbers import Number

import numpy as np

from abtem.core.utils import label_to_index


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
    ind = np.array(np.unravel_index(np.argsort(-array, axis=None), array.shape)).T
    ind = ind - central_bin_index(array)
    spot_0 = ind[0]
    spot_1 = find_linearly_independent_row(ind[1:], spot_0)
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


def validate_cell_edges(cell_edges):

    if isinstance(cell_edges, Number):
        cell_edges = [cell_edges]

    if len(cell_edges) == 1:
        cell_edges = cell_edges * 3
    elif len(cell_edges) == 2:
        cell_edges = [cell_edges[0]] * 2 + [cell_edges[1]]
    elif len(cell_edges) != 3:
        raise RuntimeError()

    return cell_edges


def map_all_bin_indices_to_miller_indices(array, sampling, cell_edges, tolerance=1e-6):

    cell_edges = validate_cell_edges(cell_edges)

    (v1, v2), (u1, u2) = bin_index_to_orthorhombic_miller(array, sampling, cell_edges)

    bins = frequency_bin_indices(array.shape)
    A = np.linalg.inv(np.array([v1, v2]))
    bins_v = np.dot(bins, A)
    hkl = np.dot(bins_v, [u1, u2])
    mask = np.all(np.abs(hkl - np.round(hkl)) < tolerance, axis=1)
    hkl = hkl[mask].astype(int)
    bins = bins[mask]
    return bins, hkl


def tabulate_diffraction_pattern(
    diffraction_pattern,
    cell_edges,
    return_data_frame: bool = False,
    normalize: bool = True,
    threshold: float = 0.01,
):
    # if len(diffraction_pattern.ensemble_shape) > 0:
    # raise NotImplementedError("tabulating not implemented for ensembles, select a single pattern by indexing")

    cell_edges = validate_cell_edges(cell_edges)

    bins, hkl = map_all_bin_indices_to_miller_indices(
        diffraction_pattern.array, diffraction_pattern.sampling, cell_edges
    )
    intensities = diffraction_pattern.select_frequency_bin(bins)

    lengths = np.sum(hkl ** 2, axis=1)
    _, labels = np.unique(lengths, return_inverse=True)

    table = {}
    for indices in label_to_index(labels):
        i = np.lexsort(np.rot90(np.cumsum(hkl[indices], axis=1)))[-1]
        key = "".join(map(str, list(hkl[indices][i])))
        intensity = intensities[indices].mean()
        table[key] = intensity

    max_intensity = max(table.values())
    normalization = max_intensity if normalize else 1.

    table = {
        key: [value / normalization]
        for key, value in table.items()
        if value / max_intensity > threshold
    }

    if return_data_frame:
        import pandas as pd
        return pd.DataFrame.from_dict(table)

    return table