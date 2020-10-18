"""Module for modifying ASE atoms objects for use in abTEM."""
from fractions import Fraction

import numpy as np
from ase import Atoms
from scipy.linalg import polar


def is_cell_hexagonal(atoms: Atoms):
    """
    Function to check whether the cell of an ASE atoms object is hexagonal.
    """
    cell = atoms.get_cell()

    a = np.linalg.norm(cell[0], axis=0)
    b = np.linalg.norm(cell[1], axis=0)
    c = np.linalg.norm(cell[2], axis=0)
    angle = np.arccos(np.dot(cell[0], cell[1]) / (a * b))

    return np.isclose(a, b) & (np.isclose(angle, np.pi / 3) | np.isclose(angle, 2 * np.pi / 3)) & (c == cell[2, 2])


def is_cell_orthogonal(atoms: Atoms, tol: float = 1e-12):
    """
    Check whether an Atoms object has an orthogonal cell.
    """
    return not np.any(np.abs(atoms.cell[~np.eye(3, dtype=bool)]) > tol)


def is_cell_valid(atoms: Atoms, tol: float = 1e-12) -> bool:
    """
    Check whether the cell of an ASE atoms object can be converted to a structure that is usable by abTEM.

    Parameters
    ----------
    atoms: ASE atoms object
        The atoms that should be checked.
    tol: float
        Components of the lattice vectors below this value are considered to be zero.

    Returns
    -------
    bool
        If true, the atomic structure is usable by abTEM.
    """
    if np.abs(atoms.cell[0, 0] - np.linalg.norm(atoms.cell[0])) > tol:
        return False

    if np.abs(atoms.cell[1, 2]) > tol:
        return False

    if np.abs(atoms.cell[2, 2] - np.linalg.norm(atoms.cell[2])) > tol:
        return False

    return True


def standardize_cell(atoms: Atoms, tol: float = 1e-12):
    """
    Standardize the cell of an ASE atoms object. The atoms are rotated so one of the lattice vectors in the xy-plane
    aligns with the x-axis, then all of the lattice vectors are made positive.

    Parameters
    ----------
    atoms: ASE atoms object
        The atoms that should be standardized
    tol: float
        Components of the lattice vectors below this value are considered to be zero.

    Returns
    -------
    atoms: ASE atoms object
        The standardized atoms.
    """
    cell = np.array(atoms.cell)

    vertical_vector = np.where(np.all(np.abs(cell[:, :2]) < tol, axis=1))[0]

    if len(vertical_vector) != 1:
        raise RuntimeError('Invalid cell: no vertical lattice vector')

    cell[[vertical_vector[0], 2]] = cell[[2, vertical_vector[0]]]
    r = np.arctan2(atoms.cell[0, 1], atoms.cell[0, 0]) / np.pi * 180

    atoms.set_cell(cell)

    if r != 0.:
        atoms.rotate(-r, 'z', rotate_cell=True)

    if not is_cell_valid(atoms, tol):
        raise RuntimeError('This cell cannot be made orthogonal using currently implemented methods.')

    atoms.set_cell(np.abs(atoms.get_cell()))

    atoms.wrap()
    return atoms


def orthogonalize_cell(atoms: Atoms, limit_denominator: int = 10, return_strain=False):
    """
    Make the cell of an ASE atoms object orthogonal. This is accomplished by repeating the cell until the x-component
    of the lattice vectors in the xy-plane closely matches. If the ratio between the x-components is irrational this
    may not be possible without introducing some strain. However, the amount of strain can be made arbitrarily small
    by using many repetitions.

    Parameters
    ----------
    atoms: ASE atoms object
        The non-orthogonal atoms object.
    limit_denominator: int
        The maximum denominator in the rational approximation. Increase this to allow more repetitions and hence less
        strain.
    return_strain: bool
        If true, return the strain tensor that were applied to make the atoms orthogonal.

    Returns
    -------
    atoms: ASE atoms object
        The orthogonal atoms.
    strain_tensor: 2x2 array
        The applied strain tensor. Only provided if return_strain is true.
    """
    if is_cell_orthogonal(atoms):
        return atoms

    atoms = atoms.copy()
    atoms = standardize_cell(atoms)

    fraction = atoms.cell[0, 0] / atoms.cell[1, 0]
    fraction = Fraction(fraction).limit_denominator(limit_denominator)

    atoms *= (fraction.denominator, fraction.numerator, 1)

    new_cell = atoms.cell.copy()
    new_cell[1, 0] = new_cell[0, 0]

    a = np.linalg.solve(atoms.cell[:2, :2], new_cell[:2, :2])
    _, strain_tensor = polar(a, side='left')
    strain_tensor[0, 0] -= 1
    strain_tensor[1, 1] -= 1

    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.set_cell(np.diag(atoms.cell))
    atoms.wrap()

    if return_strain:
        return atoms, strain_tensor
    else:
        return atoms
