from __future__ import annotations


import numpy as np
from ase import Atoms
from ase.cell import Cell

from abtem.core.utils import is_broadcastable, label_to_index
from abtem.bloch.utils import excitation_errors


def _pixel_edges(
    shape: tuple[int, int], sampling: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1 / shape[0]))
    y = np.fft.fftshift(np.fft.fftfreq(shape[1], d=1 / shape[1]))
    x = (x - 0.5) * sampling[0]
    y = (y - 0.5) * sampling[1]
    return x, y


def _find_projected_pixel_index(
    g: tuple[np.ndarray, np.ndarray],
    shape: tuple[int, int],
    sampling: tuple[float, float],
) -> np.ndarray:
    x, y = _pixel_edges(shape, sampling)

    n = np.digitize(g[..., 0], x) - 1
    m = np.digitize(g[..., 1], y) - 1

    nm = np.concatenate((n[..., None], m[..., None]), axis=-1)
    return nm


def estimate_necessary_excitation_error(energy, k_max):
    hkl_corner = np.array([[np.sqrt(k_max), np.sqrt(k_max), 0]])
    sg = excitation_errors(hkl_corner, energy)
    return sg


def match_hkl_to_pixel(hkl, g_vec, shape, sampling, sg=None):
    nm = _find_projected_pixel_index(g_vec, shape, sampling)

    if sg is None:
        return nm

    best_hkl = np.zeros(g_vec.shape[:-2] + (len(unique), 3), dtype=int)
    # best_g_vec = np.zeros((len(unique), 3), dtype=float)
    best_sg = np.zeros(g_vec.shape[:-2] + (len(unique),), dtype=float)
    best_nm = np.zeros(g_vec.shape[:-2] + (len(unique), 2), dtype=int)

    for i in np.ndindex(nm.shape[:-2]):

        unique, indices, inverse = np.unique(
            nm[i], return_index=True, return_inverse=True, axis=-2
        )

        best_hkl = np.zeros((len(unique), 3), dtype=int)
        # best_g_vec = np.zeros((len(unique), 3), dtype=float)
        best_sg = np.zeros((len(unique),), dtype=float)
        best_nm = np.zeros((len(unique), 2), dtype=int)
        for j, idx in enumerate(label_to_index(inverse)):
            closest = np.argmin(np.abs(sg[idx]))
            best_hkl[j + (i,)] = hkl[idx][closest]
            best_sg[j + (i,)] = sg[idx][closest]
            # best_g_vec[i] = g_vec[idx][closest]
            best_nm[j + (i,)] = nm[indices[i]]

    return best_hkl, best_sg, best_nm


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

    return Cell(cell)


def prefix_indices(shape):
    return tuple(
        np.arange(n)[(slice(None),) + (None,) * (len(shape) - i)]
        for i, n in enumerate(shape)
    )


def overlapping_spots_mask(nm, sg):
    mask = np.zeros(nm.shape[:-1], dtype=bool)
    order = np.argsort(np.abs(sg), axis=-1)
    order_reverse = np.argsort(order, axis=-1)

    for i in np.ndindex(nm.shape[:-2]):
        _, indices = np.unique(nm[i][order[i]], return_index=True, axis=-2)
        mask[(i,) + (indices,)] = True

    mask = mask[prefix_indices(mask.shape[:-1]) + (order_reverse,)]
    return mask


def index_diffraction_spots(
    array: np.ndarray,
    hkl,
    sampling: tuple[float, float],
    cell: Atoms | Cell | float | tuple[float, float, float],
    energy: float,
    orientation_matrices: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Indexes diffraction spots in an array.

    Parameters:
        array : np.ndarray
            The input array containing diffraction spot intensities.
        sampling : tuple[float, float]
            The sampling rate of the array in the x and y directions [Å].
        cell : Atoms | Cell | float | tuple[float, float, float]
            The unit cell of the crystal structure.
        energy : float
            The energy of the incident electrons [eV].
        k_max : float
            The maximum value of the wavevector transfer [1/Å].
        sg_max : float
            The maximum value of the excitation error [1/Å].
        rotation : tuple[float, float, float], optional
            The Euler rotation angles of the crystal structure [rad.]. Defaults to (0.0, 0.0, 0.0).
        rotation_axes : str, optional
            The intrinsic Euler rotation axes convention. Defaults to "zxz".
        intensity_min : float, optional
            The minimum intensity threshold. Defaults to 1e-12.
        centering : str, optional
            The centering of the crystal structure. Defaults to "P".

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the indexed hkl values, wavevector transfer values, pixel coordinates, and intensities.
    """

    assert len(hkl.shape) == 2
    assert hkl.shape[1] == 3

    if orientation_matrices is None:
        orientation_matrices = np.eye(3)[(None,) * len(array.shape[:-2])]

    assert is_broadcastable(array.shape[:-1], orientation_matrices.shape[:-2])

    reciprocal_lattice_vectors = np.matmul(cell.reciprocal(), np.swapaxes(orientation_matrices, -2, -1))
    g_vec = hkl @ reciprocal_lattice_vectors

    nm = _find_projected_pixel_index(g_vec, array.shape[-2:], sampling)
    intensities = array[prefix_indices(array.shape[:-2]) + (nm[..., 0], nm[..., 1])]

    sg = excitation_errors(g_vec, energy)
    intensities = intensities * overlapping_spots_mask(nm, sg)

    return intensities


def miller_to_miller_bravais(hkl: tuple[int, int, int]):
    """
    Convert Miller indices to Miller-Bravais indices.

    Parameters
    ----------
    hkl : tuple
        The Miller indices (h, k, l).

    Returns
    -------
    tuple
        The Miller-Bravais indices (H, K, I, L).
    """
    h, k, l = hkl

    H = 2 * h - k
    K = 2 * k - h
    I = -H - K
    L = l

    return H, K, I, L


def check_translation_symmetry(atoms, translation, tol=1e-12):
    positions = atoms.get_scaled_positions()
    shifted_positions = positions + translation

    differences = shifted_positions[None] - positions[:, None]
    differences[differences > 0.5] = 1.0 - differences[differences > 0.5]
    distances = np.linalg.norm(differences, axis=-1)

    matching_index = np.argmin(distances, axis=1)

    min_distances = distances[matching_index, range(len(distances))]

    has_symmetry = np.all(
        (min_distances < tol) * (atoms.numbers == atoms.numbers[matching_index])
    )
    return has_symmetry
