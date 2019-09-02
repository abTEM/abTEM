import numpy as np
from ase import Atoms


def create_orthogonal_view(origin, extent, atoms):
    origin = np.array(origin)
    extent = np.array(extent)

    P = np.array(atoms.cell[:2, :2])
    P_inv = np.linalg.inv(P)
    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower = np.dot(origin_t, P)
    upper = lower + extent

    n, m = np.ceil(np.dot(upper, P_inv)).astype(np.int)

    atoms = atoms.copy()

    mapping = np.arange(0, len(atoms))
    mapping = np.tile(mapping, (n + 2) * (m + 2))

    displacement = atoms.cell[:2, :2]

    atoms *= (n + 2, m + 2, 1)

    positions = atoms.get_positions()
    positions[:, :2] -= displacement.sum(axis=0)

    inside = ((positions[:, 0] > lower[0]) & (positions[:, 1] > lower[1]) &
              (positions[:, 0] < upper[0]) & (positions[:, 1] < upper[1]))

    atomic_numbers = atoms.get_atomic_numbers()[inside]
    positions = positions[inside]
    mapping = mapping[inside]

    positions[:, :2] -= lower

    new_atoms = Atoms(atomic_numbers, positions=positions, cell=[extent[0], extent[1], atoms.cell[2, 2]])
    return new_atoms, mapping
