from typing import TYPE_CHECKING

import numpy as np
from ase.cell import Cell
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from abtem.core.utils import label_to_index

if TYPE_CHECKING:
    pass


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


