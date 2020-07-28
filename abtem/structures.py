import numpy as np
from ase import Atoms
from fractions import Fraction


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


def standardize_cell(atoms, tol=1e-12):
    cell = np.array(atoms.cell)

    vertical_vector = np.where(np.all(np.abs(cell[:, :2]) < tol, axis=1))[0]
    if len(vertical_vector) != 1:
        raise RuntimeError('invalid cell, no vertical lattice vector')

    cell[[vertical_vector[0], 2]] = cell[[2, vertical_vector[0]]]

    cell[:2] = cell[:2][np.argsort(np.linalg.norm(cell[:2], axis=1))]

    atoms.set_cell(cell)

    r = np.arctan2(atoms.cell[0, 1], atoms.cell[0, 0]) / np.pi * 180

    atoms.rotate(-r, 'z', rotate_cell=True)

    if np.abs(atoms.cell[0, 0] - np.linalg.norm(atoms.cell[0])) > tol:
        raise RuntimeError()

    atoms.set_cell(np.abs(atoms.get_cell()))

    atoms.wrap()
    return atoms


def orthogonalize_cell(atoms, limit_denominator=10, tol=1e-12):
    atoms = standardize_cell(atoms, tol)

    fraction = atoms.cell[0, 0] / atoms.cell[1, 0]

    fraction = Fraction(fraction).limit_denominator(limit_denominator)

    atoms *= (fraction.denominator, fraction.numerator, 1)

    new_cell = atoms.cell.copy()
    new_cell[1, 0] = new_cell[0, 0]
    atoms.set_cell(new_cell, scale_atoms=True)

    atoms.set_cell(np.diag(atoms.cell))
    atoms.wrap()

    return atoms
