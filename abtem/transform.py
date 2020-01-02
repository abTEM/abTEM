import numpy as np
from ase import Atoms
from ase.build import niggli_reduce
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import RegularGridInterpolator

from abtem.points import LabelledPoints, fill_rectangle


def is_hexagonal(atoms):
    cell = atoms.get_cell()
    a = cell[0]
    b = cell[1]
    c = cell[2]

    A = np.linalg.norm(a, axis=0)
    B = np.linalg.norm(b, axis=0)
    C = np.linalg.norm(c, axis=0)
    angle = np.arccos(np.dot(a, b) / (A * B))

    return (np.isclose(A, B) & (np.isclose(angle, np.pi / 3) | np.isclose(angle, 2 * np.pi / 3)) & (C == cell[2, 2]))


def is_orthogonal(atoms, tol=1e-12):
    return not np.any(np.abs(atoms.cell[~np.eye(3, dtype=bool)]) > tol)


def merge_close(atoms, tol=1e-12):
    labels = fcluster(linkage(atoms.positions), tol, criterion='distance')

    _, idx = np.unique(labels, return_index=True)

    return atoms[idx]


def fill_rectangle_with_atoms(atoms, origin, extent, margin=0., return_atom_labels=False):
    non_zero = np.abs(atoms.cell) > 1e-12

    # if not (np.all(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.bool) == non_zero) |
    #         np.all(np.identity(3) == non_zero)):
    #     atoms.cell.pbc[:] = True
    #     niggli_reduce(atoms)

    if not np.isclose(atoms.cell[2].sum(), atoms.cell[2, 2]):
        raise RuntimeError()

    positions = atoms.positions[:, :2]
    cell = atoms.cell[:2, :2]

    points = LabelledPoints(positions, cell=cell, labels=np.arange(len(positions)))

    points = fill_rectangle(points, extent=extent, origin=origin, margin=margin)

    positions = np.hstack((points.positions, atoms.positions[points.labels, 2][:, None]))
    cell = [points.cell[0, 0], points.cell[1, 1], atoms.cell[2, 2]]

    new_atoms = Atoms(atoms.numbers[points.labels], positions=positions, cell=cell)

    if return_atom_labels:
        return new_atoms, points.labels
    else:
        return new_atoms


def orthogonalize_atoms(atoms, n=1, m=None, tol=1e-12):
    cell = np.abs(atoms.cell)
    cell = cell[np.argmax(cell, 1)]
    non_zero = np.abs(cell) > tol

    if not (np.all(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.bool) == non_zero) | np.all(
            np.identity(3) == non_zero)):
        raise RuntimeError()

    atoms.set_cell(cell)

    if m is None:
        m = np.abs(np.round(atoms.cell[0, 0] / atoms.cell[1, 0]))

    origin = [0, 0]
    extent = [atoms.cell[0, 0] * n, atoms.cell[1, 1] * m]

    atoms = fill_rectangle_with_atoms(atoms, origin=origin, extent=extent, margin=0., return_atom_labels=False)

    positions = atoms.positions
    cell = np.diag(atoms.cell)

    for i in range(3):
        close = (cell[i] - positions[:, i]) < tol
        positions[close, i] = positions[close, i] - cell[i]

    new_atoms = Atoms(atoms.numbers, positions=positions, cell=cell)

    return merge_close(new_atoms)


def orthogonalize_array(array, original_cell, origin, extent, new_gpts=None):
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
