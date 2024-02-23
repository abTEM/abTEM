from __future__ import annotations

from numbers import Number

import numpy as np
from ase import Atoms
from ase.cell import Cell
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from abtem.atoms import euler_to_rotation, is_cell_orthogonal
from abtem.core.energy import energy2wavelength
from abtem.core.utils import label_to_index


def reciprocal_cell(cell):
    return np.linalg.pinv(cell).transpose()


def reciprocal_space_gpts(
    cell: np.ndarray,
    k_max: float | tuple[float, float, float],
) -> tuple[int, int, int]:
    if isinstance(k_max, Number):
        k_max = (k_max,) * 3

    assert len(k_max) == 3

    dk = np.linalg.norm(reciprocal_cell(cell), axis=1)

    gpts = (
        int(np.ceil(k_max[0] / dk[0])) * 2 + 1,
        int(np.ceil(k_max[1] / dk[1])) * 2 + 1,
        int(np.ceil(k_max[2] / dk[2])) * 2 + 1,
    )
    return gpts


def make_hkl_grid(
    cell: np.ndarray,
    k_max: float | tuple[float, float, float],
    axes=(0, 1, 2),
) -> np.ndarray:
    gpts = reciprocal_space_gpts(cell, k_max)

    freqs = tuple(np.fft.fftfreq(n, d=1 / n).astype(int) for n in gpts)

    freqs = tuple(freqs[axis] for axis in axes)

    hkl = np.meshgrid(*freqs, indexing="ij")
    hkl = np.stack(hkl, axis=-1)

    hkl = hkl.reshape((-1, len(axes)))
    return hkl


def _pixel_edges(shape, sampling):
    x = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1 / shape[0]))
    y = np.fft.fftshift(np.fft.fftfreq(shape[1], d=1 / shape[1]))
    x = (x - 0.5) * sampling[0]
    y = (y - 0.5) * sampling[1]
    return x, y


def _find_projected_pixel_index(g, shape, sampling):
    x, y = _pixel_edges(shape, sampling)

    n = np.digitize(g[:, 0], x) - 1
    m = np.digitize(g[:, 1], y) - 1

    nm = np.concatenate((n[:, None], m[:, None]), axis=1)
    return nm


def excitation_errors(g, energy):
    wavelength = energy2wavelength(energy)
    return g[..., 2] - 0.5 * wavelength * (g**2).sum(axis=-1)


def estimate_necessary_excitation_error(energy, k_max):
    hkl_corner = np.array([[np.sqrt(k_max), np.sqrt(k_max), 0]])
    sg = excitation_errors(hkl_corner, energy)
    return sg


def match_hkl_to_pixel(hkl, g_vec, shape, sampling, sg=None):
    nm = _find_projected_pixel_index(g_vec, shape, sampling)

    if sg is None:
        return nm

    unique, indices, inverse = np.unique(
        nm, return_index=True, return_inverse=True, axis=0
    )

    best_hkl = np.zeros((len(unique), 3), dtype=int)
    best_g_vec = np.zeros((len(unique), 3), dtype=float)
    best_sg = np.zeros((len(unique),), dtype=float)
    best_nm = np.zeros((len(unique), 2), dtype=int)
    for i, idx in enumerate(label_to_index(inverse)):
        closest = np.argmin(np.abs(sg[idx]))
        best_hkl[i] = hkl[idx][closest]
        best_sg[i] = sg[idx][closest]
        best_g_vec[i] = g_vec[idx][closest]
        best_nm[i] = nm[indices[i]]

    return best_hkl, best_g_vec, best_sg, best_nm


def filter_by_threshold(arrays, values, threshold) -> tuple[np.ndarray, ...]:
    mask = values < threshold
    shape = None
    out = ()
    for array in arrays:
        if shape and shape[0] != array.shape[0]:
            raise ValueError()

        shape = array.shape
        out += (array[mask],)

    return out


def validate_cell(
    cell: Atoms | Cell | float | tuple[float, float, float]
) -> np.ndarray:
    if isinstance(cell, Atoms):
        cell = cell.cell

    if np.isscalar(cell):
        cell = np.diag([cell] * 3)

    cell = np.array(cell)

    if isinstance(cell, np.ndarray) and cell.shape != (3, 3):
        cell = np.diag(cell)

    if not is_cell_orthogonal(Cell(cell)):
        raise NotImplementedError

    return cell


def _F_reflection_conditions(hkl):
    all_even = (hkl % 2 == 0).all(axis=1)
    all_odd = (hkl % 2 == 1).all(axis=1)
    return all_even + all_odd


def _I_reflection_conditions(hkl):
    return (hkl.sum(axis=1) % 2 == 0).all(axis=1)


def _A_reflection_conditions(hkl):
    return (hkl[1:].sum(axis=1) % 2 == 0).all(axis=1)


def _B_reflection_conditions(hkl):
    return (hkl[:, [0, 1]].sum(axis=1) % 2 == 0).all(axis=1)


def _C_reflection_conditions(hkl):
    return (hkl[:-1].sum(axis=1) % 2 == 0).all(axis=1)


def _reflection_condition_mask(hkl: np.ndarray, centering: str):
    if centering == "P":
        return np.ones(len(hkl), dtype=bool)
    elif centering == "F":
        return _F_reflection_conditions(hkl)
    elif centering == "I":
        return _I_reflection_conditions(hkl)
    elif centering == "A":
        return _A_reflection_conditions(hkl)
    elif centering == "B":
        return _B_reflection_conditions(hkl)
    elif centering == "C":
        return _C_reflection_conditions(hkl)
    else:
        raise ValueError("lattice centering must be one of P, I, F, A, B or C")


def index_diffraction_spots(
    array,
    sampling,
    cell,
    energy,
    k_max,
    sg_max,
    rotation=(0.0, 0.0, 0.0),
    rotation_axes="zxz",
    intensity_min=1e-12,
    centering: str = "P",
):
    R = euler_to_rotation(*rotation, axes=rotation_axes)

    cell = validate_cell(cell)

    cell = R @ cell

    hkl = make_hkl_grid(cell, k_max)

    mask = _reflection_condition_mask(hkl, centering=centering)
    hkl = hkl[mask]

    g_vec = hkl @ reciprocal_cell(cell)

    sg = excitation_errors(g_vec, energy)
    hkl, g_vec, sg = filter_by_threshold(
        arrays=(hkl, g_vec, sg), values=sg, threshold=sg_max
    )

    hkl, g_vec, sg, nm = match_hkl_to_pixel(hkl, g_vec, array.shape[-2:], sampling, sg)

    intensity = array[..., nm[:, 0], nm[:, 1]]

    if len(array.shape) == 2:
        max_intensity = intensity
    else:
        max_intensity = intensity.max(axis=tuple(range(0, len(intensity.shape) - 1)))

    hkl, g_vec, sg, nm = filter_by_threshold(
        arrays=(hkl, g_vec, sg, nm),
        values=-max_intensity,
        threshold=-intensity_min,
    )

    intensity = intensity[..., max_intensity > intensity_min]

    if len(intensity.shape) > 1:
        reps = intensity.shape[:-1] + (1, 1)
        g_vec = np.tile(g_vec[(None,) * (len(reps) - 2)], reps=reps)

    return hkl, g_vec, nm, intensity


def _format_miller_indices(hkl):
    return "{} {} {}".format(*hkl)


def _miller_to_miller_bravais(hkl):
    h, k, l = hkl

    H = 2 * h - k
    K = 2 * k - h
    I = -H - K
    L = l

    return H, K, I, L


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
