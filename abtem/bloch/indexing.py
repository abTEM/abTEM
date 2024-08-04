from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.cell import Cell

from abtem.bloch.utils import excitation_errors
from abtem.core.utils import is_broadcastable


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
    sg = np.abs(excitation_errors(hkl_corner, energy).item())
    return sg


def validate_cell(cell: Atoms | Cell | float | tuple[float, float, float]) -> Cell:
    """
    Validate the cell input.

    Parameters:
    ----------
    cell : Atoms | Cell | float | tuple[float, float, float]
        The unit cell of the crystal structure.

    Returns:
    --------
    Cell
        The validated cell.
    """
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


def overlapping_spots_mask(nm: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """
    Create a mask for overlapping diffraction spots. Spots with the same h and k indices are considered overlapping.
    """
    mask = np.zeros(nm.shape[:-1], dtype=bool)
    order = np.argsort(np.abs(sg), axis=-1)
    order_reverse = np.argsort(order, axis=-1)

    for i in np.ndindex(nm.shape[:-2]):
        _, indices = np.unique(nm[i][order[i]], return_index=True, axis=-2)
        if len(i):
            indices = i + (indices,)
        mask[indices] = True

    mask = mask[prefix_indices(mask.shape[:-1]) + (order_reverse,)]
    return mask


def create_ellipse(a: int, b: int) -> np.ndarray:
    """
    Create an ellipse with semi-major and semi-minor axes.

    Parameters:
    ----------
    a : int
        The semi-major axis of the ellipse.
    b : int
        The semi-minor axis of the ellipse.

    Returns:
    --------
    np.ndarray
        The ellipse.
    """
    y, x = np.ogrid[-a : a + 1, -b : b + 1]
    a, b = max(a, 1), max(b, 1)
    return x**2 / b**2 + y**2 / a**2 <= 1


def integrate_ellipse_around_pixels(
    array: np.ndarray, nm: np.ndarray, a: int, b: int
) -> np.ndarray:
    """
    Integrate an ellipse around pixels in an array.

    Parameters:
    ----------
    array : np.ndarray
        The input array containing diffraction spot intensities.
    nm : np.ndarray
        The pixel coordinates of the diffraction spots.
    a : int
        The semi-major axis of the ellipse.
    b : int
        The semi-minor axis of the ellipse.

    Returns:
    --------
    np.ndarray
        The integrated intensities around the pixels.
    """
    ellipse = create_ellipse(a, b)
    structure = np.array(tuple(i - n for i, n in zip(np.where(ellipse), (a, b)))).T

    intensities = np.zeros_like(array, shape=array.shape[:-2] + (nm.shape[-2],))

    for i in range(nm.shape[-2]):
        nms = nm[..., i, :] - structure[(None,) * len(nm.shape[:-2])]
        nms = nms[
            (nms >= 0).all(-1)
            * (nms[..., 0] < array.shape[-2])
            * (nms[..., 1] < array.shape[-1])
        ]

        intensities[..., i] = array[
            prefix_indices(array.shape[:-2]) + (nms[:, 0], nms[:, 1])
        ].sum((-1,))

    return intensities


def index_diffraction_spots(
    array: np.ndarray,
    hkl,
    sampling: tuple[float, float],
    cell: Atoms | Cell | float | tuple[float, float, float],
    energy: float,
    orientation_matrices: np.ndarray = None,
    radius: float = None,
) -> np.ndarray:
    """
    Indexes diffraction spots in an array.

    Parameters
    ----------
    array : np.ndarray
        The input array containing diffraction spot intensities.
    hkl : np.ndarray
        The Miller indices to index.
    sampling : tuple[float, float]
        The sampling rate of the array in the x and y directions [Å].
    cell : Atoms | Cell | float | tuple[float, float, float]
        The unit cell of the crystal structure.
    energy : float
        The energy of the incident electrons [eV].
    orientation_matrices : np.ndarray, optional
        The orientation matrices of the crystal structure. Defaults to None.
    radius : float, optional
        The radius of the diffraction spots to integrate. Defaults to None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the indexed hkl values, wavevector transfer values, pixel coordinates, and intensities.
    """

    assert len(hkl.shape) == 2
    assert hkl.shape[1] == 3

    if orientation_matrices is None:
        orientation_matrices = np.eye(3)[(None,) * len(array.shape[:-2])]

    assert is_broadcastable(array.shape[:-2], orientation_matrices.shape[:-2])

    reciprocal_lattice_vectors = np.matmul(
        cell.reciprocal(), np.swapaxes(orientation_matrices, -2, -1)
    )
    g_vec = hkl @ reciprocal_lattice_vectors

    shape = (array.shape[-2], array.shape[-1])
    nm = _find_projected_pixel_index(g_vec, shape, sampling)
    intensities = array[prefix_indices(array.shape[:-2]) + (nm[..., 0], nm[..., 1])]

    if radius is not None:
        a, b = tuple(int(np.round(radius / d)) for d in sampling)
        intensities = integrate_ellipse_around_pixels(array, nm, a, b)

    sg = excitation_errors(g_vec, energy)

    mask = overlapping_spots_mask(nm, sg)

    intensities = intensities * mask

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
