from __future__ import annotations


import numpy as np
from numba import njit, prange

from abtem.core.energy import energy2wavelength
from ase.cell import Cell


def reciprocal_cell(cell):
    """
    Calculate the reciprocal cell of a unit cell.

    Parameters
    ----------
    cell : 3x3 np.ndarray
        The unit cell.

    Returns
    -------
    3x3 np.ndarray
        The reciprocal cell.
    """
    return np.linalg.pinv(cell).transpose()


def calculate_g_vec(hkl: np.ndarray, cell: Cell):
    """
    
    """
    return hkl @ cell.reciprocal()


def calculate_g_vec_length(hkl: np.ndarray, cell):
    return np.linalg.norm(calculate_g_vec(hkl, cell), axis=-1)


def hkl_strings_to_array(hkl):
    return np.array([tuple(map(int, hkli.split(" "))) for hkli in hkl])


def reciprocal_space_gpts(
    cell: np.ndarray,
    k_max: float,
) -> tuple[int, int, int]:
    # if isinstance(k_max, Number):
    #    k_max = (k_max,) * 3

    # assert len(k_max) == 3

    dk = np.linalg.norm(reciprocal_cell(cell), axis=1)

    gpts = (
        int(np.ceil(k_max / dk[0])) * 2 + 1,
        int(np.ceil(k_max / dk[1])) * 2 + 1,
        int(np.ceil(k_max / dk[2])) * 2 + 1,
    )
    return gpts


def make_hkl_grid(
    cell: np.ndarray,
    k_max: float,
    axes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    gpts = reciprocal_space_gpts(cell, k_max)

    freqs = tuple(np.fft.fftfreq(n, d=1 / n).astype(int) for n in gpts)

    freqs = tuple(freqs[axis] for axis in axes)

    hkl = np.meshgrid(*freqs, indexing="ij")
    hkl = np.stack(hkl, axis=-1)

    hkl = hkl.reshape((-1, len(axes)))
    g_vec = calculate_g_vec(hkl, cell)
    hkl = hkl[(g_vec**2).sum(-1) <= k_max**2]
    return hkl


def excitation_errors(g, energy, paraxial=False):
    assert g.shape[-1] == 3
    wavelength = energy2wavelength(energy)
    if paraxial:
        sg = (-2 * g[..., 2] - wavelength * (g[..., 0] ** 2 + g[..., 1] ** 2)) / 2.0
    else:
        sg = (-2 * g[..., 2] - wavelength * np.sum(g * g, axis=-1)) / 2.0
    return sg


def get_reflection_condition(hkl: np.ndarray, centering: str):
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
    if centering.lower() == "f":
        all_even = (hkl % 2 == 0).all(axis=1)
        all_odd = (hkl % 2 == 1).all(axis=1)
        return all_even + all_odd
    elif centering.lower() == "i":
        return hkl.sum(axis=1) % 2 == 0
    elif centering.lower() == "a":
        return (hkl[1:].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "b":
        return (hkl[:, [0, 1]].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "c":
        return (hkl[:-1].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "p":
        return np.ones(len(hkl), dtype=bool)
    else:
        raise ValueError()


@njit(parallel=True, fastmath=True, nogil=True, error_model="numpy")
def fast_filter_excitation_errors(mask, g, orientation_matrices, wavelength, sg_max):
    g_length_2 = (g**2).sum(axis=-1)

    b = 0.5 * wavelength * g_length_2
    for i in prange(len(orientation_matrices)):
        R = orientation_matrices[i]

        sg = -g[:, 0] * R[2, 0] - g[:, 1] * R[2, 1] - g[:, 2] * R[2, 2] - b

        mask += np.abs(sg) < sg_max


def filter_reciprocal_space_vectors(
    hkl: np.ndarray,
    cell: Cell,
    energy: float,
    sg_max: float,
    k_max: float,
    centering: str = "P",
    orientation_matrices: np.ndarray = None,
) -> np.ndarray:
    """
    Filter reciprocal space vectors based on excitation errors and reflection conditions.

    Parameters
    ----------
    hkl : np.ndarray
        Reciprocal space vectors.
    cell : Cell
        Unit cell.
    energy : float
        Electron energy [eV].
    sg_max : float
        Maximum excitation error [1/Å].
    k_max : float
        Maximum scattering vector length [1/Å].
    centering : str, optional
        Crystal centering must be one of 'P', 'I', 'A', 'B', 'C' or 'F'. Default is 'P'.
    orientation_matrices : np.ndarray, optional
        Orientation matrices for each crystallographic direction.

    Returns
    -------
    np.ndarray
        Mask for the reciprocal space vectors.
    """
    g = hkl @ cell.reciprocal()
    g_length = np.linalg.norm(g, axis=-1)

    if orientation_matrices is None:
        mask = np.abs(excitation_errors(g, energy, paraxial=False)) < sg_max

    else:
        if len(orientation_matrices.shape) == 2:
            orientation_matrices = orientation_matrices[None]

        orientation_matrices = orientation_matrices.reshape(-1, 3, 3)

        # if not len(orientation_matrices.shape) == 3:
        #     raise ValueError(
        #         "'orientation_matrices' must have shape (3, 3) or (n, 3, 3)"
        #     )

        mask = np.zeros(len(hkl), dtype=bool)

        fast_filter_excitation_errors(
            mask, g, orientation_matrices, energy2wavelength(energy), sg_max
        )

    mask *= get_reflection_condition(hkl, centering)
    mask *= g_length < k_max

    return mask
