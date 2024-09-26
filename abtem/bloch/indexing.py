from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms
from ase.cell import Cell

from abtem.bloch.utils import excitation_errors, reciprocal_cell
from abtem.core.grid import polar_spatial_frequencies


def _pixel_edges(
    shape: tuple[int, int], sampling: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the pixel edges of an array.

    Parameters:
    -----------
    shape : tuple[int, int]
        The shape of the array.
    sampling : tuple[float, float]
        The sampling rate of the array in the x and y directions [Å].

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        The pixel edges in reciprocal
    """
    x = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1 / shape[0]))
    y = np.fft.fftshift(np.fft.fftfreq(shape[1], d=1 / shape[1]))
    x = (x - 0.5) * sampling[0]
    y = (y - 0.5) * sampling[1]
    return x, y


def _find_projected_pixel_index(
    g: np.ndarray,
    shape: tuple[int, int],
    sampling: tuple[float, float],
) -> np.ndarray:
    x, y = _pixel_edges(shape, sampling)

    n = np.digitize(g[..., 0], x) - 1
    m = np.digitize(g[..., 1], y) - 1

    nm = np.concatenate((n[..., None], m[..., None]), axis=-1)
    return nm


def estimate_necessary_excitation_error(energy: float, k_max: float) -> float:
    hkl_corner = np.array([[np.sqrt(k_max), np.sqrt(k_max), 0]])
    sg = np.abs(excitation_errors(hkl_corner, energy).item())
    return sg


def validate_cell(
    cell: Atoms | Cell | np.ndarray | float | tuple[float, float, float],
) -> Cell:
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
        validated_cell = cell.cell

    elif isinstance(cell, float):
        validated_cell = np.diag([cell] * 3)

    elif isinstance(cell, tuple):
        validated_cell = np.array(cell)

    elif isinstance(cell, np.ndarray) and cell.shape != (3, 3):
        validated_cell = np.diag(cell)

    elif isinstance(cell, (np.ndarray, Cell)):
        validated_cell = cell

    else:
        raise ValueError(f"Invalid cell input, got {cell}")

    return Cell(validated_cell)


# def prefix_indices(shape):
#     return tuple(
#         np.arange(n)[(slice(None),) + (None,) * (len(shape) - i)]
#         for i, n in enumerate(shape)
#     )


def overlapping_spots_mask(nm: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """
    Create a mask for overlapping diffraction spots. Spots with the same h and k indices
    are considered overlapping.
    """
    mask = np.zeros(nm.shape[:-1], dtype=bool)
    order = np.argsort(np.abs(sg), axis=-1)
    order_reverse = np.argsort(order, axis=-1)

    for i in np.ndindex(nm.shape[:-2]):
        _, indices = np.unique(nm[i][order[i]], return_index=True, axis=-2)
        if len(i):
            indices = i + (indices,)
        mask[indices] = True

    mask = mask[..., order_reverse]
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


def antialiased_disk(r: float, sampling: tuple[float, float]) -> np.ndarray:
    """
    Create an array representing disk with antialiased edges.

    Parameters:
    ----------
    r : float
        The radius of the disk.
    sampling : two float
        The sampling rate of the array in the x and y directions. Units are arbitrary.

    Returns:
    --------
    np.ndarray
        A 2D array representing the disk.
    """
    gpts = 2 * int(np.ceil(r / sampling[0])) + 1, 2 * int(np.ceil(r / sampling[1])) + 1
    alpha, phi = polar_spatial_frequencies(
        gpts, (1 / (sampling[0] * gpts[0]), 1 / (sampling[1] * gpts[1]))
    )
    denominator = np.sqrt(
        (np.cos(phi) * sampling[0]) ** 2 + (np.sin(phi) * sampling[1]) ** 2
    )
    denominator[0, 0] = 1.0
    array = np.clip((r - alpha) / denominator + 0.5, a_min=0.0, a_max=1.0)
    array[0, 0] = 1.0
    array = np.fft.fftshift(array)
    return array


def integrate_ellipse_around_pixels(
    array: np.ndarray,
    nm: np.ndarray,
    r: float,
    sampling: tuple[float, float],
    priority: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Integrate an ellipse around pixels in an array.

    Parameters:
    ----------
    array : np.ndarray
        The input array containing diffraction spot intensities.
    nm : np.ndarray
        The pixel coordinates of the diffraction spots.

    Returns:
    --------
    np.ndarray
        The integrated intensities around the pixels.
    """
    weights = antialiased_disk(r, sampling)
    a, b = weights.shape[0] // 2, weights.shape[1] // 2
    intensities = np.zeros_like(array, shape=array.shape[:-2] + (nm.shape[-2],))

    masked_array = array.copy()

    assert len(nm.shape) == 2 and nm.shape[1] == 2

    if priority is None:
        order = np.arange(nm.shape[-2])
    else:
        order = np.argsort(priority, axis=-1)
    
    for i, (nmx, nmy) in enumerate(nm[order]):
        x_slice = slice(max(0, nmx - a), min(array.shape[-2], nmx + a + 1))
        y_slice = slice(max(0, nmy - b), min(array.shape[-1], nmy + b + 1))

        weights_slice_x = slice(a - (nmx - x_slice.start), a + (x_slice.stop - nmx))
        weights_slice_y = slice(b - (nmy - y_slice.start), b + (y_slice.stop - nmy))
        cropped_weigths = weights[weights_slice_x, weights_slice_y]

        integrated_intensity = (
            masked_array[..., x_slice, y_slice] * cropped_weigths
        ).sum((-2, -1))

        masked_array[..., x_slice, y_slice] *= 1 - cropped_weigths

        intensities[..., order[i]] = integrated_intensity

    return intensities


def index_diffraction_spots(
    array: np.ndarray,
    hkl: np.ndarray,
    sampling: tuple[float, float],
    cell: Cell | np.ndarray,
    energy: float,
    orientation_matrices: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
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
    cell : Cell | np.ndarray
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
        A tuple containing the indexed hkl values, wavevector transfer values, pixel
        coordinates, and intensities.
    """

    assert len(hkl.shape) == 2
    assert hkl.shape[1] == 3

    if orientation_matrices is None:
        orientation_matrices = np.eye(3)[(None,) * len(array.shape[:-2])]

    # assert is_broadcastable(array.shape[:-2], orientation_matrices.shape[:-2])

    orientation_matrices = np.squeeze(orientation_matrices)

    assert orientation_matrices.shape == (3, 3)

    reciprocal_lattice_vectors = np.matmul(
        reciprocal_cell(cell), orientation_matrices.T
    )

    g_vec = hkl @ reciprocal_lattice_vectors

    shape = (array.shape[-2], array.shape[-1])

    nm = _find_projected_pixel_index(g_vec, shape, sampling)

    sg = np.abs(excitation_errors(g_vec, energy))

    if radius is not None:
        # a, b = tuple(int(np.round(radius / d)) for d in sampling)
        intensities = integrate_ellipse_around_pixels(array, nm, radius, sampling, sg)
    else:
        intensities = array[..., nm[..., 0], nm[..., 1]]

    sg = excitation_errors(g_vec, energy)

    mask = overlapping_spots_mask(nm, sg)

    intensities = intensities * mask

    return intensities


def miller_to_miller_bravais(hkl: tuple[int, int, int]) -> tuple[int, int, int, int]:
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
    h, k, l = hkl  #  noqa: E741

    H = 2 * h - k
    K = 2 * k - h
    I = -H - K  # noqa: E741
    L = l

    return H, K, I, L
