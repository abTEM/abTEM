"""Module for modifying ASE atoms objects for use in abTEM."""
from numbers import Number
from typing import Union, Tuple, Sequence

import numpy as np
from ase import Atoms
from ase.build.tools import rotation_matrix, cut
from ase.cell import Cell
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

_axes2tuple = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}


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

    return atoms


def rotation_matrix_to_euler(R: np.ndarray, axes: str = 'sxyz', eps: float = 1e-6):
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


def decompose_affine_transform(affined_transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ZS = np.linalg.cholesky(np.dot(affined_transform.T, affined_transform)).T

    zoom = np.diag(ZS)

    shear = ZS / zoom[:, None]
    shear = shear[np.triu_indices(3, 1)]

    rotation = np.dot(affined_transform, np.linalg.inv(ZS))

    if np.linalg.det(rotation) < 0:
        zoom[0] *= -1
        ZS[0] *= -1
        rotation = np.dot(affined_transform, np.linalg.inv(ZS))

    return rotation, zoom, shear


def _label_to_index_generator(labels: np.ndarray, first_label: int = 0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


def merge_close_atoms(atoms: Atoms, tol: float = 1e-7) -> Atoms:
    """
    Merge atoms within closer in distance than a given tolerance.

    Parameters
    ----------
    atoms : Atoms
        Atoms to merge.
    tol : float
        Atoms closer than this value are merged assuming they have identical atomic numbers.

    Returns
    -------
    merged_atoms : Atoms
    """
    if len(atoms) < 2:
        return atoms

    atoms = wrap_with_tolerance(atoms)

    points = atoms.positions
    numbers = atoms.numbers

    clusters = fcluster(linkage(pdist(points), method='complete'), tol, criterion='distance')
    new_points = np.zeros_like(points)
    new_numbers = np.zeros_like(numbers)

    k = 0
    for i, cluster in enumerate(_label_to_index_generator(clusters, 1)):
        new_points[i] = np.mean(points[cluster], axis=0)
        assert np.all(numbers[cluster] == numbers[0])
        new_numbers[i] = numbers[0]
        k += 1

    new_atoms = Atoms(positions=new_points[:k], numbers=new_numbers[:k], cell=atoms.cell)
    return new_atoms


def wrap_with_tolerance(atoms, tol: float = 1e-6):
    atoms = atoms.copy()

    atoms.wrap()
    d = np.linalg.norm(np.array(atoms.cell), axis=0)
    tol = tol / d
    scaled_positions = atoms.get_scaled_positions()
    scaled_positions = ((tol + scaled_positions) % 1) - tol

    atoms.positions[:] = atoms.cell.cartesian_positions(scaled_positions)
    return atoms


def shrink_cell(atoms, repetitions=(2, 3), tol=1e-6):
    atoms = wrap_with_tolerance(atoms, tol=tol)

    for repetition in repetitions:
        for i in range(3):
            while True:
                try:
                    atoms_copy = atoms.copy()
                    atoms_copy.cell[i] = atoms_copy.cell[i] / repetition

                    atoms_copy = wrap_with_tolerance(atoms_copy)

                    old_len = len(atoms_copy)
                    atoms_copy = merge_close_atoms(atoms_copy, tol=1e-5)

                    assert len(atoms_copy) == old_len // repetition

                    atoms = atoms_copy
                except AssertionError:
                    break

    return atoms


axis_mapping = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}


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


def rotation_matrix_from_plane(plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = 'xy'):
    x_vector, y_vector = plane

    if isinstance(x_vector, str):
        x_vector = np.array(axis_mapping[x_vector])

    if isinstance(y_vector, str):
        y_vector = np.array(axis_mapping[y_vector])

    old_x_vector = np.array([1., 0., 0.])
    old_y_vector = np.array([0., 1., 0.])

    if np.any(x_vector != old_x_vector) or np.any(y_vector != old_y_vector):
        return rotation_matrix(old_x_vector, x_vector, old_y_vector, y_vector)
    else:
        return np.eye(3)


def rotate_atoms_to_plane(atoms: Atoms,
                          plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = 'xy',
                          ) -> Atoms:
    if plane == 'xy':
        return atoms

    atoms = atoms.copy()
    R = rotation_matrix_from_plane(plane)

    atoms.positions[:] = np.dot(atoms.positions[:], R.T)
    atoms.cell[:] = np.dot(atoms.cell[:], R.T)
    return atoms


def flip_atoms(atoms: Atoms, axis: int = 2) -> Atoms:
    atoms = atoms.copy()
    atoms.positions[:, axis] = atoms.cell[axis, axis] - atoms.positions[:, axis]
    return atoms


def best_orthogonal_box(cell: np.ndarray, max_repetitions: int = 5, eps: float = 1e-12) -> np.ndarray:
    zero_vectors = np.linalg.norm(cell, axis=0) < eps

    if zero_vectors.sum() > 1:
        raise RuntimeError("two or more lattice vectors of the provided Atoms has no length")

    k = np.arange(-max_repetitions, max_repetitions + 1)
    l = np.arange(-max_repetitions, max_repetitions + 1)
    m = np.arange(-max_repetitions, max_repetitions + 1)

    a, b, c = cell
    vectors = np.abs(((k[:, None] * a[None])[:, None, None] +
                      (l[:, None] * b[None])[None, :, None] +
                      (m[:, None] * c[None])[None, None, :]))

    norm = np.linalg.norm(vectors, axis=-1)
    nonzero = norm > eps
    norm[nonzero == 0] = eps

    new_vectors = []
    for i in range(3):
        angles = vectors[..., i] / norm

        small_angles = np.abs(angles.max() - angles < eps)

        small_angles = np.where(small_angles * nonzero)

        shortest_small_angles = np.argmin(np.linalg.norm(vectors[small_angles], axis=1))

        new_vector = np.array([k[small_angles[0][shortest_small_angles]],
                               l[small_angles[1][shortest_small_angles]],
                               m[small_angles[2][shortest_small_angles]]])

        new_vector = np.sign(np.dot(new_vector, cell)[i]) * new_vector
        new_vectors.append(new_vector)

    cell = np.dot(new_vectors, np.array(cell))
    return np.linalg.norm(cell, axis=0)


def orthogonalize_cell(atoms: Atoms,
                       box: Tuple[float, float, float] = None,
                       max_repetitions: int = 5,
                       return_transform: bool = False,
                       allow_transform: bool = True,
                       origin: Tuple[float, float, float] = (0., 0., 0.),
                       plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = 'xy',
                       tolerance: float = 0.01):
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
    allow_transform : bool
        If false no transformation is applied to make the cell orthogonal, hence a non-orthogonal cell may be returned.


    Returns
    -------
    atoms : ASE atoms object
        The orthogonal atoms.
    transform : tuple of arrays
        The applied transform in the form the euler angles
    """

    if origin != (0., 0., 0.):
        atoms.translate(-np.array(origin))
        atoms.wrap()

    if plane != 'xy':
        atoms = rotate_atoms_to_plane(atoms, plane)

    if box is None:
        box = best_orthogonal_box(atoms.cell, max_repetitions=max_repetitions)

    inv = np.linalg.inv(atoms.cell)
    vectors = np.dot(np.diag(box), inv)
    vectors = np.round(vectors)

    atoms = cut(atoms, *vectors, tolerance=tolerance)

    A = np.linalg.solve(atoms.cell.complete(), np.diag(box))

    if allow_transform:
        atoms.positions[:] = np.dot(atoms.positions, A)
        atoms.cell[:] = np.diag(box)

    elif not np.allclose(A, np.eye(3)):
        raise RuntimeError()

    if return_transform:
        rotation, zoom, shear = decompose_affine_transform(A)
        return atoms, (np.array(rotation_matrix_to_euler(rotation)), zoom, shear)
    else:
        return atoms


# def orthogonalize_cell(atoms: Atoms,
#                        target_box: Tuple[float, float, float] = None,
#                        max_repetitions: int = None,
#                        return_transform: bool = False,
#                        allow_transform: bool = True,
#                        tolerance: float = 0.01):
#
#     eps = 1e-12
#
#     zero_vectors = np.linalg.norm(atoms.cell, axis=0) < eps
#
#     if zero_vectors.sum() > 1:
#         raise RuntimeError("two or more lattice vectors of the provided Atoms has no length")
#     elif zero_vectors.sum() == 1:
#         atoms.center(axis=np.where(zero_vectors)[0], vacuum=tolerance + eps)
#
#     a, b, c = atoms.cell
#
#     if target_box is not None and max_repetitions is None:
#         max_repetitions = int(np.ceil(np.sum(target_box) / np.min(np.linalg.norm(atoms.cell, axis=0))))
#     elif max_repetitions is None:
#         max_repetitions = 5
#
#     k = np.arange(-max_repetitions, max_repetitions + 1)
#     l = np.arange(-max_repetitions, max_repetitions + 1)
#     m = np.arange(-max_repetitions, max_repetitions + 1)
#
#     vectors = np.abs(((k[:, None] * a[None])[:, None, None] +
#                       (l[:, None] * b[None])[None, :, None] +
#                       (m[:, None] * c[None])[None, None, :]))
#
#     norm = np.linalg.norm(vectors, axis=-1)
#     nonzero = norm > eps
#     norm[nonzero == 0] = eps
#
#     new_vectors = []
#     for i in range(3):
#
#         if target_box:
#             target_vector = np.zeros(3)
#             target_vector[i] = target_box[i]
#             closest_vector = np.linalg.norm(vectors - target_vector[None, None, None], axis=-1)
#             closest_vector = np.where((closest_vector == np.min(closest_vector)) * nonzero)
#             new_vector = np.array([k[closest_vector[0][0]],
#                                    l[closest_vector[1][0]],
#                                    m[closest_vector[2][0]]])
#
#         else:
#             angles = vectors[..., i] / norm
#             small_angles = np.abs(angles.max() - angles < eps)
#             small_angles = np.where(small_angles * nonzero)
#             shortest_small_angles = np.argmin(np.linalg.norm(vectors[small_angles], axis=1))
#
#             new_vector = np.array([k[small_angles[0][shortest_small_angles]],
#                                    l[small_angles[1][shortest_small_angles]],
#                                    m[small_angles[2][shortest_small_angles]]])
#
#         new_vector = np.sign(np.dot(new_vector, atoms.cell)[i]) * new_vector
#         new_vectors.append(new_vector)
#
#     atoms = cut(atoms, *new_vectors, tolerance=tolerance)
#
#     if target_box:
#         cell = np.diag(target_box)
#     else:
#         cell = Cell.new(np.linalg.norm(atoms.cell, axis=0)).complete()
#
#     A = np.linalg.solve(atoms.cell.complete(), cell)
#
#     if allow_transform:
#         atoms.positions[:] = np.dot(atoms.positions, A)
#         atoms.cell[:] = cell
#
#     else:
#         if not is_cell_orthogonal(atoms):
#             raise RuntimeError()
#
#     atoms = shrink_cell(atoms, (2, 3))
#
#     if return_transform:
#         rotation, zoom, shear = decompose_affine_transform(A)
#         return atoms, (np.array(rotation_matrix_to_euler(rotation)), zoom, shear)
#     else:
#         return atoms


def atoms_in_box(atoms: Atoms,
                 box: Tuple[float, float, float],
                 margin: Tuple[float, float, float] = (0., 0., 0.),
                 origin: Tuple[float, float, float] = (0., 0., 0.)) -> Atoms:
    mask = np.all(atoms.positions >= (np.array(origin) - margin - 1e-12)[None], axis=1) * \
           np.all(atoms.positions < (np.array(origin) + box + margin)[None], axis=1)
    atoms = atoms[mask]
    return atoms


def cut_box(atoms: Atoms,
            box: Tuple[float, float, float] = None,
            plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = 'xy',
            origin: Tuple[float, float, float] = (0., 0., 0.),
            margin: Union[float, Tuple[float, float, float]] = 0.) -> Atoms:
    if box is None:
        box = best_orthogonal_box(atoms.cell)

    if isinstance(margin, Number):
        margin = (margin, margin, margin)

    atoms = atoms.copy()
    if not np.all(np.isclose(origin, (0., 0., 0.))):
        atoms.positions[:] = atoms.positions - origin
        atoms.wrap()

    atoms = rotate_atoms_to_plane(atoms, plane)

    new_cell = np.diag(np.array(box) + 2 * np.array(margin))
    new_cell = np.dot(atoms.cell.scaled_positions(new_cell), atoms.cell)

    scaled_margin = atoms.cell.scaled_positions(np.diag(margin))
    scaled_margin = np.sign(scaled_margin) * (np.ceil(np.abs(scaled_margin)))

    scaled_corners_new_cell = np.array([[0., 0., 0.], [0., 0., 1.],
                                        [0., 1., 0.], [0., 1., 1.],
                                        [1., 0., 0.], [1., 0., 1.],
                                        [1., 1., 0.], [1., 1., 1.]])

    corners = np.dot(scaled_corners_new_cell, new_cell)
    scaled_corners = np.linalg.solve(atoms.cell.T, corners.T).T
    repetitions = np.ceil(scaled_corners.ptp(axis=0)).astype('int') + 1
    new_atoms = atoms * repetitions

    center_translate = np.dot(np.floor(scaled_corners.min(axis=0)), atoms.cell)
    margin_translate = atoms.cell.cartesian_positions(scaled_margin).sum(0)

    new_atoms.positions[:] += center_translate - margin_translate

    new_atoms = atoms_in_box(new_atoms, box, margin=margin)
    new_atoms.cell = box

    # new_atoms = wrap_with_tolerance(new_atoms)
    return new_atoms


def pad_atoms(atoms: Atoms, margins: Union[float, Tuple[float, float, float]], directions: str = 'xyz') -> Atoms:
    """
    Repeat the atoms in x and y, retaining only the repeated atoms within the margin distance from the cell boundary.

    Parameters
    ----------
    atoms: ASE Atoms object
        The atoms that should be padded.
    margin: two float
        The padding margin.

    Returns
    -------
    ASE Atoms object
        Padded atoms.
    """

    if not is_cell_orthogonal(atoms):
        raise RuntimeError('The cell of the atoms must be orthogonal.')

    if isinstance(margins, Number):
        margins = (margins,) * 3

    atoms = atoms.copy()
    old_cell = atoms.cell.copy()

    axes = [{'x': 0, 'y': 1, 'z': 2}[direction] for direction in directions]

    reps = [1, 1, 1]
    for axis, margin in zip(axes, margins):
        reps[axis] = int(1 + 2 * np.ceil(margin / atoms.cell[axis, axis]))

    if any([rep > 1 for rep in reps]):
        atoms *= reps
        atoms.positions[:] -= np.diag(old_cell) * [rep // 2 for rep in reps]
        atoms.cell = old_cell

    atoms = atoms_in_box(atoms, np.diag(atoms.cell), margins)

    return atoms