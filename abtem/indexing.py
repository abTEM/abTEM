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
from typing import Union, Tuple


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

    n = np.digitize(g[:, 0], x) - 1
    m = np.digitize(g[:, 1], y) - 1

    nm = np.concatenate((n[:, None], m[:, None]), axis=1)
    return nm


def excitation_errors(g: np.ndarray, energy: float) -> np.ndarray:
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


def reflection_condition_mask(hkl: np.ndarray, centering: str) -> np.ndarray:
    """
    Returns a boolean mask indicating which reflections satisfy the reflection condition
    based on the given lattice centering.

    Parameters
    ----------
    hkl : np.ndarray
        Array of shape (N, 3) representing the Miller indices of reflections.
    centering : str
        The lattice centering type. Must be one of "P", "I", "F", "A", "B", or "C".

    Returns
    -------
    np.ndarray
        Boolean mask indicating which reflections satisfy the reflection condition.
    """

    if centering == "P":
        return np.ones(len(hkl), dtype=bool)
    elif centering == "F":
        all_even = (hkl % 2 == 0).all(axis=1)
        all_odd = (hkl % 2 == 1).all(axis=1)
        return all_even + all_odd
    elif centering == "I":
        return (hkl.sum(axis=1) % 2 == 0).all(axis=1)
    elif centering == "A":
        return (hkl[:, [1, 2]].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering == "B":
        return (hkl[:, [0, 2]].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering == "C":
        return (hkl[:, [0, 1]].sum(axis=1) % 2 == 0).all(axis=1)
    else:
        raise ValueError("lattice centering must be one of P, I, F, A, B or C")


def index_diffraction_spots(
    array: np.ndarray,
    sampling: tuple[float, float],
    cell: Atoms | Cell | float | tuple[float, float, float],
    energy: float,
    k_max: float,
    sg_max: float,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_axes: str = "zxz",
    intensity_min: float = 1e-12,
    centering: str = "P",
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
    R = euler_to_rotation(*rotation, axes=rotation_axes)

    cell = validate_cell(cell)

    cell = R @ cell

    hkl = make_hkl_grid(cell, k_max)

    mask = reflection_condition_mask(hkl, centering=centering)
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