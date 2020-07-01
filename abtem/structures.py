import numpy as np
from ase import Atoms


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


def is_orthogonal(atoms, tol=1e-11):
    return not np.any(np.abs(atoms.cell[~np.eye(3, dtype=bool)]) > tol)


def fill_rectangle(atoms, extent, origin=None, margin=0., eps=1e-12):
    # non_zero = np.abs(atoms.cell) > 1e-12

    if origin is None:
        origin = np.zeros(2)
    else:
        origin = np.array(origin)

    # if not (np.all(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.bool) == non_zero) |
    #         np.all(np.identity(3) == non_zero)):
    #     atoms.cell.pbc[:] = True
    #     niggli_reduce(atoms)

    if not np.isclose(atoms.cell[2].sum(), atoms.cell[2, 2]):
        raise RuntimeError()

    cell = atoms.cell.copy()

    extent = np.array(extent)

    P_inv = np.linalg.inv(cell[:2, :2])
    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0
    lower_corner = np.dot(origin_t, cell[:2, :2])
    upper_corner = lower_corner + extent

    corners = np.array([[-margin - eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, upper_corner[1].item() + margin + eps],
                        [-margin - eps, upper_corner[1].item() + margin + eps]])

    n0, m0 = 0, 0
    n1, m1 = 0, 0
    for corner in corners:
        new_n, new_m = np.ceil(np.dot(corner, P_inv)).astype(np.int)
        n0 = max(n0, new_n)
        m0 = max(m0, new_m)
        new_n, new_m = np.floor(np.dot(corner, P_inv)).astype(np.int)
        n1 = min(n1, new_n)
        m1 = min(m1, new_m)

    # TODO : this number of repetitions is very wasteful
    atoms *= ((1 + n0 - n1).item(), (1 + m0 - m1).item(), 1)
    new_positions = atoms.positions.copy()

    new_positions += cell[0] * n1 + cell[1] * m1 - np.concatenate((lower_corner, [0.]))[None]

    inside = ((new_positions[:, 0] >= - margin - eps) &
              (new_positions[:, 1] >= - margin - eps) &
              (new_positions[:, 0] < extent[0] + margin) &
              (new_positions[:, 1] < extent[1] + margin))

    new_positions = new_positions[inside]
    new_atomic_numbers = atoms.numbers[inside]
    new_cell = [extent[0], extent[1], cell[2, 2]]

    return Atoms(new_atomic_numbers, positions=new_positions, cell=new_cell)


def standardize_cell(atoms, tol=1e-12):
    """
    Permute the lattice vectors of an Atoms object without changing the positions. The lattice vectors will be sorted
    such that the first component is maximized.  This means that an orthorhombic cell will have a diagonal
    representation. Additionaly, negative lattice vectors are reversed (and atomic positions shifted to compensate).

    Parameters
    ----------
    atoms : Atoms object
        Atoms object to modify cell
    tol : float
        Tolerance for ensuring correct treatment of zero length lattice vectors.
    """
    old_cell = atoms.cell.copy()
    cell = np.abs(old_cell) + np.identity(3) * tol
    cell = cell[np.argmax(cell, 0)]
    atoms.set_cell(cell)
    atoms.positions += np.sum(cell - old_cell, axis=0) / 2
    return atoms


def orthogonalize(atoms, n=1, m=None, tol=1e-12):
    cell = np.abs(atoms.cell) + np.identity(3) * 2 * tol
    cell = cell[np.argmax(cell, 0)]
    non_zero = np.abs(cell) > tol

    if not (np.all(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.bool) == non_zero) | np.all(
            np.identity(3) == non_zero)):
        raise RuntimeError()

    atoms.set_cell(cell)

    if m is None:
        m = np.abs(np.round(atoms.cell[0, 0] / atoms.cell[1, 0]))

    origin = [0, 0]
    extent = [atoms.cell[0, 0] * n, atoms.cell[1, 1] * m]

    atoms = fill_rectangle(atoms, origin=origin, extent=extent, margin=0.)

    positions = atoms.positions
    cell = np.diag(atoms.cell)

    for i in range(3):
        close = (cell[i] - positions[:, i]) < tol
        positions[close, i] = positions[close, i] - cell[i]

    new_atoms = Atoms(atoms.numbers, positions=positions, cell=cell, pbc=True)
    new_atoms.wrap(eps=tol)

    #labels = fcluster(linkage(new_atoms.positions), tol, criterion='distance')
    #_, idx = np.unique(labels, return_index=True)
    return new_atoms#[idx]
