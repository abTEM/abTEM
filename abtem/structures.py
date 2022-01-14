"""Module for modifying ASE atoms objects for use in abTEM."""
from numbers import Number
from typing import Union, Sequence

import numpy as np
from ase import Atoms
from ase.build import cut
from ase.cell import Cell


def is_cell_hexagonal(atoms: Atoms):
    """
    Function to check whether the cell of an ASE atoms object is hexagonal.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms that should be checked.
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

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms that should be checked.
    tol : float
        Components of the lattice vectors below this value are considered to be zero.
    """
    return not np.any(np.abs(atoms.cell[~np.eye(3, dtype=bool)]) > tol)


def is_cell_valid(atoms: Atoms, tol: float = 1e-12) -> bool:
    """
    Check whether the cell of an ASE atoms object can be converted to a structure that is usable by abTEM.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms that should be checked.
    tol : float
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


_axes2tuple = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}


def rotation_matrix_to_euler(R, axes='sxyz', eps=1e-6):
    firstaxis, parity, repetition, frame = _axes2tuple[axes.lower()]

    i = firstaxis
    j = [1, 2, 0, 1][i + parity]
    k = [1, 2, 0, 1][i - parity + 1]

    R = np.array(R, dtype=float)
    if repetition:
        sy = np.sqrt(R[i, j] * R[i, j] + R[i, k] * R[i, k])
        if sy > eps:
            ax = np.arctan2(R[i, j], R[i, k])
            ay = np.arctan2(sy, R[i, i])
            az = np.arctan2(R[j, i], -R[k, i])
        else:
            ax = np.arctan2(-R[j, k], R[j, j])
            ay = np.arctan2(sy, R[i, i])
            az = 0.0
    else:
        cy = np.sqrt(R[i, i] * R[i, i] + R[j, i] * R[j, i])
        if cy > eps:
            ax = np.arctan2(R[k, j], R[k, k])
            ay = np.arctan2(-R[k, i], cy)
            az = np.arctan2(R[j, i], R[i, i])
        else:
            ax = np.arctan2(-R[j, k], R[j, j])
            ay = np.arctan2(-R[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def decompose_affine_transform(A):
    ZS = np.linalg.cholesky(np.dot(A.T, A)).T

    zoom = np.diag(ZS)

    shear = ZS / zoom[:, None]
    shear = shear[np.triu_indices(3, 1)]

    rotation = np.dot(A, np.linalg.inv(ZS))

    if np.linalg.det(rotation) < 0:
        zoom[0] *= -1
        ZS[0] *= -1
        rotation = np.dot(A, np.linalg.inv(ZS))

    return rotation, zoom, shear


def orthogonalize_cell(atoms: Atoms,
                       max_repetitions: int = 5,
                       return_transform: bool = False,
                       transform: Union[bool, str] = True):
    """
    Make the cell of an ASE atoms object orthogonal. This is accomplished by repeating the cell until lattice vectors
    are close to the three principal Cartesian directions. If the structure is not exactly orthogonal after the
    structure is repeated by a given maximum the remaining difference will be made up by applying strain.

    Parameters
    ----------
    atoms : ASE atoms object
        The non-orthogonal atoms object.
    max_repetitions : int
        The maximum number of repetions allowed. Increase this to allow more repetitions and hence less strain.
    return_transform : bool
        If true, return the transformations that were applied to make the atoms orthogonal.
    transform : bool
        If false no transformation is applied to make the cell orthogonal, hence a non-orthogonal cell may be returned.


    Returns
    -------
    atoms : ASE atoms object
        The orthogonal atoms.
    transform : tuple of arrays
        The applied transform in the form the euler angles
    """
    eps = 1e-8

    k = np.arange(-max_repetitions, max_repetitions + 1)
    l = np.arange(-max_repetitions, max_repetitions + 1)
    m = np.arange(-max_repetitions, max_repetitions + 1)

    a, b, c = atoms.cell

    vectors = np.abs(((k[:, None] * a[None])[:, None, None] +
                      (l[:, None] * b[None])[None, :, None] +
                      (m[:, None] * c[None])[None, None, :]))

    norm = np.linalg.norm(vectors, axis=-1)
    nonzero = norm > eps
    norm[nonzero == 0] = eps

    new_vectors = []
    for i in range(3):
        angles = vectors[..., i] / norm

        optimal = np.abs(angles.max() - angles < eps)

        optimal = np.where(optimal * nonzero)
        n = np.linalg.norm(vectors[optimal], axis=1)

        j = np.argmin(n)
        new_vector = np.array([k[optimal[0][j]], l[optimal[1][j]], m[optimal[2][j]]])

        new_vector = np.sign(np.dot(new_vector, atoms.cell)[i]) * new_vector
        new_vectors.append(new_vector)

    atoms = cut(atoms, *new_vectors)

    cell = Cell.new(np.linalg.norm(atoms.cell, axis=0))
    A = np.linalg.solve(atoms.cell.complete(), cell.complete())

    if transform is True:
        atoms.positions[:] = np.dot(atoms.positions, A)
        atoms.cell[:] = cell

    elif transform == 'raise':
        if not is_cell_orthogonal(atoms):
            raise RuntimeError()

    if return_transform and transform:
        rotation, zoom, shear = decompose_affine_transform(A)
        return atoms, (np.array(rotation_matrix_to_euler(rotation)), zoom, shear)
    else:
        return atoms


def standardize_cell(atoms: Atoms, tol: float = 1e-12):
    """
    Standardize the cell of an ASE atoms object. The atoms are rotated so one of the lattice vectors in the xy-plane
    aligns with the x-axis, then all of the lattice vectors are made positive.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms that should be standardized
    tol : float
        Components of the lattice vectors below this value are considered to be zero.

    Returns
    -------
    atoms : ASE atoms object
        The standardized atoms.
    """
    cell = np.array(atoms.cell)

    vertical_vector = np.where(np.all(np.abs(cell[:, :2]) < tol, axis=1))[0]

    # if len(vertical_vector) != 1:
    #    raise RuntimeError('Invalid cell: no vertical lattice vector')

    cell[[vertical_vector[0], 2]] = cell[[2, vertical_vector[0]]]
    r = np.arctan2(cell[0, 1], cell[0, 0]) / np.pi * 180

    atoms.set_cell(cell)

    if r != 0.:
        atoms.rotate(-r, 'z', rotate_cell=True)

    # if not is_cell_valid(atoms, tol):
    #    raise RuntimeError('This cell cannot be made orthogonal using currently implemented methods.')

    atoms.set_cell(np.abs(atoms.get_cell()))

    atoms.wrap()
    return atoms


# def orthogonalize_cell(atoms: Atoms,
#                        limit_denominator: int = 10,
#                        preserve_periodicity: bool = True,
#                        return_strain: bool = False):
#     """
#     Make the cell of an ASE atoms object orthogonal. This is accomplished by repeating the cell until lattice vectors
#     are close to the three principal Cartesian directions. If the structure is not exactly orthogonal after the
#     structure is repeated by a given maximum the remaining difference will be made up by applying strain.
#
#     Parameters
#     ----------
#     atoms : ASE atoms object
#         The non-orthogonal atoms object.
#     limit_denominator : int
#         The maximum denominator in the rational approximation. Increase this to allow more repetitions and hence less
#         strain.
#     preserve_periodicity : bool, optional
#         This function will make a structure periodic while preserving periodicity exactly, this will generally result in
#         repeating the structure. If preserving periodicity is not desired, this may be set to False. Default is True.
#     return_strain : bool
#         If true, return the strain tensor that were applied to make the atoms orthogonal.
#
#     Returns
#     -------
#     atoms : ASE atoms object
#         The orthogonal atoms.
#     strain_tensor : 2x2 array
#         The applied strain tensor. Only provided if return_strain is true.
#     """
#     if is_cell_orthogonal(atoms):
#         return atoms
#
#     atoms = atoms.copy()
#     atoms = standardize_cell(atoms)
#
#     if not preserve_periodicity:
#         return cut_rectangle(atoms, origin=(0, 0), extent=np.diag(atoms.cell)[:2])
#
#     fraction = Fraction(atoms.cell[0, 0] / atoms.cell[1, 0]).limit_denominator(limit_denominator)
#
#     new_cell = atoms.cell.copy()
#     new_cell[1, 0] = atoms.cell[0, 0] / fraction
#
#     a = np.linalg.solve(atoms.cell[:2, :2], new_cell[:2, :2])
#     _, strain_tensor = polar(a, side='left')
#     strain_tensor[0, 0] -= 1
#     strain_tensor[1, 1] -= 1
#
#     atoms.set_cell(new_cell, scale_atoms=True)
#
#     atoms *= (abs(fraction.denominator), abs(fraction.numerator), 1)
#
#     atoms.set_cell(np.diag(atoms.cell))
#     atoms.wrap()
#
#     if return_strain:
#         return atoms, strain_tensor
#     else:
#         return atoms


def cut_rectangle(atoms: Atoms, origin: Sequence[float], extent: Sequence[float], margin: float = 0.):
    """
    Cuts out a cell starting at the origin to a given extent from a sufficiently repeated copy of atoms.

    Parameters
    ----------
    atoms : ASE atoms object
        This should correspond to a repeatable unit cell.
    origin : two float
        Origin of the new cell. Units of Angstrom.
    extent : two float
        xy-extent of the new cell. Units of Angstrom.
    margin : float
        Atoms within margin from the border of the new cell will be included. Units of Angstrom. Default is 0.

    Returns
    -------
    ASE atoms object
    """

    # TODO : check that this works in edge cases

    atoms = atoms.copy()
    cell = atoms.cell.copy()

    extent = (extent[0], extent[1], atoms.cell[2, 2],)
    atoms.positions[:, :2] -= np.array(origin)

    a = atoms.cell.scaled_positions(np.array((extent[0] + 2 * margin, 0, 0)))
    b = atoms.cell.scaled_positions(np.array((0, extent[1] + 2 * margin, 0)))

    repetitions = (int(np.ceil(abs(a[0])) + np.ceil(abs(b[0]))),
                   int(np.ceil(abs(a[1])) + np.ceil(abs(b[1]))), 1)

    shift = (-np.floor(min(a[0], 0)) - np.floor(min(b[0], 0)),
             -np.floor(min(a[1], 0)) - np.floor(min(b[1], 0)), 0)
    atoms.set_scaled_positions(atoms.get_scaled_positions() - shift)

    atoms *= repetitions

    atoms.positions[:, :2] -= margin

    atoms.set_cell([extent[0], extent[1], cell[2, 2]])

    atoms = atoms[((atoms.positions[:, 0] >= -margin) &
                   (atoms.positions[:, 1] >= -margin) &
                   (atoms.positions[:, 0] < extent[0] + margin) &
                   (atoms.positions[:, 1] < extent[1] + margin))
    ]
    return atoms


def pad_atoms(atoms: Atoms, margin: float, directions='xy', in_place=False):
    """
    Repeat the atoms in x and y, retaining only the repeated atoms within the margin distance from the cell boundary.

    Parameters
    ----------
    atoms: ASE Atoms object
        The atoms that should be padded.
    margin: float
        The padding margin.

    Returns
    -------
    ASE Atoms object
        Padded atoms.
    """

    if not is_cell_orthogonal(atoms):
        raise RuntimeError('The cell of the atoms must be orthogonal.')

    if not in_place:
        atoms = atoms.copy()

    old_cell = atoms.cell.copy()

    axes = [{'x': 0, 'y': 1, 'z': 2}[direction] for direction in directions]

    reps = [1, 1, 1]
    for axis in axes:
        reps[axis] = int(1 + 2 * np.ceil(margin / atoms.cell[axis, axis]))

    if any([rep > 1 for rep in reps]):
        atoms *= reps
        atoms.positions[:] -= np.diag(old_cell) * [rep // 2 for rep in reps]
        atoms.cell = old_cell

    # import matplotlib.pyplot as plt
    # from abtem import show_atoms
    # show_atoms(atoms, plane='xz')
    # plt.show()

    to_keep = np.ones(len(atoms), dtype=bool)
    for axis in axes:
        to_keep *= (atoms.positions[:, axis] > -margin) * (atoms.positions[:, axis] < atoms.cell[axis, axis] + margin)

    atoms = atoms[to_keep]

    # for axis in axes:
    #     left = atoms[atoms.positions[:, axis] < margin]
    #     left.positions[:, axis] += atoms.cell[axis, axis]
    #     right = atoms[(atoms.positions[:, axis] > atoms.cell[axis, axis] - margin) &
    #                   (atoms.positions[:, axis] < atoms.cell[axis, axis])]
    #     right.positions[:, axis] -= atoms.cell[axis, axis]
    #     atoms += left + right
    return atoms


def plane_to_axes(plane):
    """Internal function for extracting axes from a plane."""
    axes = ()
    last_axis = [0, 1, 2]
    for axis in list(plane):
        if axis == 'x':
            axes += (0,)
            last_axis.remove(0)
        if axis == 'y':
            axes += (1,)
            last_axis.remove(1)
        if axis == 'z':
            axes += (2,)
            last_axis.remove(2)
    return axes + (last_axis[0],)


def rotate_atoms_to_plane(atoms, plane='xy'):
    if plane == 'xy':
        return atoms

    axes = plane_to_axes(plane)

    positions = atoms.positions[:, axes]
    cell = atoms.cell[:, axes]
    cell = cell[list(axes)]

    new_atoms = atoms.copy()
    new_atoms.positions[:] = positions
    new_atoms.cell[:] = cell
    new_atoms = standardize_cell(new_atoms)
    return new_atoms  # standardize_cell(atoms)


def flip_atoms(atoms):
    atoms = atoms.copy()
    atoms.positions[:] = atoms.cell[2, 2] - atoms.positions[:]
    return atoms


class SlicedAtoms:

    def __init__(self, atoms, slice_thicknesses):
        self._atoms = atoms
        self.slice_thicknesses = slice_thicknesses

    @property
    def atoms(self):
        return self._atoms

    @property
    def positions(self):
        return self.atoms.positions

    @property
    def numbers(self):
        return self.atoms.numbers

    def __len__(self):
        return len(self.atoms)

    @property
    def slice_thicknesses(self):
        return self._slice_thicknesses

    @slice_thicknesses.setter
    def slice_thicknesses(self, slice_thicknesses):
        if isinstance(slice_thicknesses, Number):
            num_slices = int(np.ceil(self._atoms.cell[2, 2] / slice_thicknesses))
            slice_thicknesses = np.full(num_slices, float(slice_thicknesses))
        self._slice_thicknesses = slice_thicknesses

    def flip(self):
        self._atoms.positions[:] = self._atoms.cell[2, 2] - self._atoms.positions[:]
        self._slice_thicknesses[:] = self._slice_thicknesses[::-1]

    def get_slice_entrance(self, i):
        return max(np.sum(self.slice_thicknesses[:i]), 0)

    def get_slice_exit(self, i):
        return min(self.get_slice_entrance(i) + self.slice_thicknesses[i], self.atoms.cell[2, 2])

    def get_subsliced_atoms(self,
                            start,
                            end=None,
                            atomic_number=None,
                            padding: Union[bool, float] = False,
                            z_margin=0.):

        if end is None:
            end = start + 1

        a = self.get_slice_entrance(start) - z_margin
        b = self.get_slice_entrance(end) + z_margin

        in_slice = (self.atoms.positions[:, 2] >= a) * (self.atoms.positions[:, 2] < b)

        if atomic_number is not None:
            in_slice = (self.atoms.numbers == atomic_number) * in_slice

        atoms = self.atoms[in_slice]

        if padding:
            atoms = pad_atoms(atoms, padding)

        return self.__class__(atoms, self.slice_thicknesses)

    @property
    def num_slices(self):
        """The number of projected potential slices."""
        return len(self._slice_thicknesses)

    def get_slice_thickness(self, i):
        """The thickness of the projected potential slices."""
        return self._slice_thicknesses[i]
