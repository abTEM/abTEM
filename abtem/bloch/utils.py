from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from ase.cell import Cell
from numba import njit  # type: ignore

from abtem.core.backend import cp
from abtem.core.energy import energy2wavelength


def reciprocal_cell(cell: np.ndarray | Cell) -> np.ndarray:
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


def calculate_g_vec(hkl: np.ndarray, cell: np.ndarray | Cell) -> np.ndarray:
    return hkl @ reciprocal_cell(cell)


def calculate_g_vec_length(hkl: np.ndarray, cell: np.ndarray | Cell) -> np.ndarray:
    return np.linalg.norm(calculate_g_vec(hkl, cell), axis=-1)


def hkl_strings_to_array(hkl: list[str]) -> np.ndarray:
    return np.array([tuple(map(int, hkli.split(" "))) for hkli in hkl])


def generate_linear_combinations(
    vectors: np.ndarray, coefficients: Sequence[int], exclude_zero: bool = False
) -> np.ndarray:
    """
    Generate all possible linear combinations of the given vectors with the given
    coefficients.

    Parameters
    ----------
    vectors : np.array
        Array of vectors.
    coefficients : sequence of int
        Coefficients to use in the linear combinations.
    exclude_zero : bool, optional
        Whether to exclude the zero vector from the output.

    Returns
    -------
    np.array
        Array of linear combinations.
    """
    combinations = np.array(
        [
            sum(c * v for c, v in zip(coef_comb, vectors))
            for coef_comb in itertools.product(coefficients, repeat=len(vectors))
        ]
    )
    if exclude_zero:
        combinations = combinations[(combinations == 0).all(axis=1) == 0]
    return combinations


def get_shortest_g_vec_length(cell: Cell) -> float:
    """
    Get the length of the shortest reciprocal space vector in the given unit cell.

    Parameters
    ----------
    cell : Cell
        Unit cell.

    Returns
    -------
    float
        Length of the shortest reciprocal space vector [1/Å].
    """
    coefficients = [-1, 0, 1]
    reciprocal_cell = np.array(cell.reciprocal())
    combinations = generate_linear_combinations(
        reciprocal_cell, coefficients, exclude_zero=True
    )
    return np.min(np.linalg.norm(combinations, axis=1))


def reciprocal_space_gpts(
    cell: np.ndarray | Cell,
    g_max: float,
) -> tuple[int, int, int]:
    # if isinstance(g_max, Number):
    #    g_max = (g_max,) * 3

    # assert len(g_max) == 3

    dk = np.linalg.norm(reciprocal_cell(cell), axis=1)

    gpts = (
        int(np.ceil(g_max / dk[0])) * 2 + 1,
        int(np.ceil(g_max / dk[1])) * 2 + 1,
        int(np.ceil(g_max / dk[2])) * 2 + 1,
    )
    return gpts


def make_hkl_grid(
    cell: np.ndarray | Cell,
    g_max: float,
    axes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    gpts = reciprocal_space_gpts(cell, g_max)

    freqs = tuple(np.fft.fftfreq(n, d=1 / n).astype(int) for n in gpts)

    freqs = tuple(freqs[axis] for axis in axes)

    hkl_grids = np.meshgrid(*freqs, indexing="ij")
    hkl = np.stack(hkl_grids, axis=-1)

    hkl = hkl.reshape((-1, len(axes)))
    g_vec = calculate_g_vec(hkl, cell)
    hkl = hkl[(g_vec**2).sum(-1) <= g_max**2]
    return hkl


def excitation_errors(
    g: np.ndarray, energy: float, use_wave_eq: bool = False
) -> np.ndarray:
    """
    Calculate excitation errors for a set of reciprocal space vectors.

    Parameters
    ----------
    g : np.ndarray
        Reciprocal space vectors [1/Å], as an array of shape (N, 3).
    energy : float
        Electron energy [eV].
    use_wave_eq : bool, optional
        Whether to use the excitation errors derived from the wave equation.
        Default is False.

    Returns
    -------
    np.ndarray
        Excitation errors [1/Å].
    """
    assert g.shape[-1] == 3
    wavelength = energy2wavelength(energy)
    if use_wave_eq:
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


@njit(nogil=True, error_model="numpy")
def fast_filter_excitation_errors(mask, g, orientation_matrices, wavelength, sg_max):
    g_length_2 = (g**2).sum(axis=-1)

    b = 0.5 * wavelength * g_length_2
    for i in range(len(orientation_matrices)):
        R = orientation_matrices[i]

        sg = -g[:, 0] * R[2, 0] - g[:, 1] * R[2, 1] - g[:, 2] * R[2, 2] - b

        mask += np.abs(sg) < sg_max


def filter_reciprocal_space_vectors(
    hkl: np.ndarray,
    cell: Cell,
    energy: float,
    sg_max: float,
    g_max: float,
    centering: str = "P",
    orientation_matrices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Filter reciprocal space vectors based on excitation errors and reflection
    conditions.

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
    g_max : float
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
        mask = np.abs(excitation_errors(g, energy, use_wave_eq=False)) <= sg_max

    else:
        if len(orientation_matrices.shape) == 2:
            orientation_matrices = orientation_matrices[None]

        orientation_matrices = orientation_matrices.reshape((-1, 3, 3))

        # if not len(orientation_matrices.shape) == 3:
        #    raise ValueError(
        #        "'orientation_matrices' must have shape (3, 3) or (n, 3, 3)"
        #    )

        mask = np.zeros(len(g), dtype=bool)

        fast_filter_excitation_errors(
            mask, g, orientation_matrices, energy2wavelength(energy), sg_max
        )

    mask *= get_reflection_condition(hkl, centering)

    mask *= g_length <= g_max

    return mask


def ravel_hkl(hkl: np.ndarray, gpts: tuple[int, int, int]) -> np.ndarray:
    hkl = np.asarray(hkl)
    shift = np.array((gpts[0] // 2, gpts[1] // 2, gpts[2] // 2))
    hkl = hkl + shift
    multi_index = (hkl[..., 0], hkl[..., 1], hkl[..., 2])
    return np.ravel_multi_index(multi_index, gpts)


def raveled_hkl_to_hkl(
    array: np.ndarray,
    hkl_source: np.ndarray,
    hkl_destination: np.ndarray,
    gpts: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert a raveled array to a 3D array with the shape of the structure factor.

    Parameters
    ----------
    array : np.ndarray
        The raveled array.
    hkl_source : np.ndarray
        The reciprocal space vectors as Miller indices for the source array.
    hkl_destination : np.ndarray
        The reciprocal space vectors as Miller indices for the destination array.
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.

    Returns
    -------
    np.ndarray
        The 3D array.
    """

    hkl_source = ravel_hkl(hkl_source, gpts)
    hkl_destination = ravel_hkl(hkl_destination, gpts)

    if cp is not None and isinstance(array, cp.ndarray):
        convert_to_numpy = True
    else:
        convert_to_numpy = False

    if convert_to_numpy:
        array = cp.asnumpy(array)

    df = pd.Series(array, index=hkl_source)
    array = df.get(hkl_destination, default=0.0).to_numpy()  # type: ignore

    if convert_to_numpy:
        array = cp.asarray(array)

    return array
