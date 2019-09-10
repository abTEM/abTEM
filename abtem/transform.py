import numpy as np
from ase import Atoms
from scipy.interpolate import RegularGridInterpolator


def make_orthogonal_atoms(atoms, origin, extent, return_equivalent=False):
    origin = np.array(origin)
    extent = np.array(extent)

    P = np.array(atoms.cell[:2, :2])
    P_inv = np.linalg.inv(P)

    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower_corner = np.dot(origin_t, P)
    upper_corner = lower_corner + extent

    n, m = np.ceil(np.dot(upper_corner, P_inv)).astype(np.int)

    repeated_atoms = atoms.copy()
    displacement = atoms.cell[:2, :2].copy()

    repeated_atoms *= (n + 2, m + 2, 1)

    positions = repeated_atoms.get_positions()
    positions[:, :2] -= displacement.sum(axis=0)

    eps = 1e-16
    inside = ((positions[:, 0] > lower_corner[0] - eps) & (positions[:, 1] > lower_corner[1] - eps) &
              (positions[:, 0] < upper_corner[0]) & (positions[:, 1] < upper_corner[1]))

    atomic_numbers = repeated_atoms.get_atomic_numbers()[inside]
    positions = positions[inside]

    positions[:, :2] -= lower_corner

    new_atoms = Atoms(atomic_numbers, positions=positions, cell=[extent[0], extent[1], atoms.cell[2, 2]])

    if return_equivalent:
        equivalent = np.arange(0, len(atoms))
        equivalent = np.tile(equivalent, (n + 2) * (m + 2))
        equivalent = equivalent[inside]
        return new_atoms, equivalent
    else:
        return new_atoms


def make_orthogonal_array(array, original_cell, origin, extent, new_gpts=None):
    if new_gpts is None:
        new_gpts = array.shape

    origin = np.array(origin)
    extent = np.array(extent)

    P = np.array(original_cell)
    P_inv = np.linalg.inv(P)
    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower = np.dot(origin_t, P)
    upper = lower + extent

    n, m = np.ceil(np.dot(upper, P_inv)).astype(np.int)

    tiled = np.tile(array, (n + 2, m + 2))
    x = np.linspace(-1, n + 1, tiled.shape[0], endpoint=False)
    y = np.linspace(-1, m + 1, tiled.shape[1], endpoint=False)

    x_ = np.linspace(lower[0], upper[0], new_gpts[0], endpoint=False)
    y_ = np.linspace(lower[1], upper[1], new_gpts[1], endpoint=False)
    x_, y_ = np.meshgrid(x_, y_, indexing='ij')

    p = np.array([x_.ravel(), y_.ravel()]).T
    p = np.dot(p, P_inv)

    interpolated = RegularGridInterpolator((x, y), tiled)(p)

    return interpolated.reshape((new_gpts[0], new_gpts[1]))
