from __future__ import annotations

import itertools
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.cell import Cell
from numba import njit  # type: ignore

from abtem.atoms import AtomProperties, validate_per_atom_property, validate_sigmas
from abtem.core.backend import get_array_module
from abtem.core.energy import energy2wavelength


def reciprocal_cell(cell: np.ndarray | Cell) -> np.ndarray:
    """
    Calculate the reciprocal cell of a unit cell.

    Parameters
    ----------
    cell : 3x3 numpy.ndarray
        The unit cell.

    Returns
    -------
    3x3 numpy.ndarray
        The reciprocal cell.
    """
    return np.linalg.pinv(cell).transpose()


def calculate_g_vec(hkl: np.ndarray, cell: np.ndarray | Cell) -> np.ndarray:
    g_vec: np.ndarray = hkl @ reciprocal_cell(cell)
    return g_vec


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
    vectors : numpy.array
        Array of vectors.
    coefficients : sequence of int
        Coefficients to use in the linear combinations.
    exclude_zero : bool, optional
        Whether to exclude the zero vector from the output.

    Returns
    -------
    numpy.array
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
    combinations = generate_linear_combinations(
        reciprocal_cell(cell), coefficients, exclude_zero=True
    )
    return np.min(np.linalg.norm(combinations, axis=1))


def cell_bounds(cell: np.ndarray | Cell) -> np.ndarray:
    cell = np.array(cell)
    origin = np.zeros(3)
    vertices = np.array(
        [
            origin,
            cell[0],
            cell[1],
            cell[2],
            cell[0] + cell[1],
            cell[0] + cell[2],
            cell[1] + cell[2],
            cell[0] + cell[1] + cell[2],
        ]
    )

    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)

    return max_bounds - min_bounds


def reciprocal_space_gpts(
    cell: np.ndarray | Cell,
    g_max: float,
) -> tuple[int, int, int]:
    # The largest Miller index along axis i over the ellipsoid {hkl : |hkl @ B| <=
    # g_max} is g_max * sqrt((B B^T)^-1)_ii = g_max * |a_i| (the lattice-vector
    # length). Using the Cartesian bounding box instead under-sizes the grid for a
    # non-orthogonal cell (the box can be smaller than |a_i| along some axis),
    # dropping reflection differences from the dynamical matrix. For an orthogonal
    # cell |a_i| equals the bounding box.
    lengths = np.linalg.norm(np.array(cell, dtype=float), axis=1)

    gpts = (
        int(np.ceil(g_max * lengths[0])) * 2 + 1,
        int(np.ceil(g_max * lengths[1])) * 2 + 1,
        int(np.ceil(g_max * lengths[2])) * 2 + 1,
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


def validate_use_wave_eq(use_wave_eq: bool | str) -> bool | Literal["exact"]:
    """Check that use_wave_eq is True, False or 'exact', and return it."""
    if isinstance(use_wave_eq, (bool, np.bool_)):
        return bool(use_wave_eq)
    if isinstance(use_wave_eq, str) and use_wave_eq == "exact":
        return "exact"
    raise ValueError(f"use_wave_eq must be True, False or 'exact', not {use_wave_eq!r}")


def excitation_errors(
    g: np.ndarray, energy: float, use_wave_eq: bool | Literal["exact"] = "exact"
) -> np.ndarray:
    """
    Calculate excitation errors for a set of reciprocal space vectors.

    Parameters
    ----------
    g : numpy.ndarray
        Reciprocal space vectors [1/Å], as an array of shape (N, 3).
    energy : float
        Electron energy [eV].
    use_wave_eq : bool or 'exact', optional
        If 'exact' (default), the non-paraxial form,
        ``-g_z + k (sqrt(1 - λ² g_⊥²) - 1)`` with ``k = 1 / λ``, the counterpart of
        ``FourierMultislice(order="exact")``; to first order in ``λ² g_⊥²`` it
        equals the paraxial form. Requires ``λ g_⊥ < 1`` (no evanescent beams).
        If True, the excitation errors derived from the paraxial wave equation,
        ``-g_z - λ g_⊥² / 2``; Bloch waves then solve the same equation as
        multislice with the first-order (Fresnel) propagator,
        ``FourierMultislice(order=1)``. If False, the standard excitation errors,
        ``-g_z - λ |g|² / 2``, from the Ewald sphere.

    Returns
    -------
    numpy.ndarray
        Excitation errors [1/Å].
    """
    assert g.shape[-1] == 3
    wavelength = energy2wavelength(energy)
    use_wave_eq = validate_use_wave_eq(use_wave_eq)
    if use_wave_eq == "exact":
        xp = get_array_module(g)
        x = wavelength**2 * (g[..., 0] ** 2 + g[..., 1] ** 2)
        if x.size and float(x.max()) >= 1.0:
            raise ValueError(
                "use_wave_eq='exact' requires wavelength * |g_perp| < 1 for every "
                "beam; reduce g_max or sg_max"
            )
        # k (sqrt(1 - x) - 1), evaluated stably as -k x / (sqrt(1 - x) + 1), the
        # same form as the exact multislice propagator
        sg = -g[..., 2] - x / (wavelength * (xp.sqrt(1.0 - x) + 1.0))
    elif use_wave_eq:
        sg = (-2 * g[..., 2] - wavelength * (g[..., 0] ** 2 + g[..., 1] ** 2)) / 2.0
    else:
        sg = (-2 * g[..., 2] - wavelength * np.sum(g * g, axis=-1)) / 2.0
    return sg


def get_reflection_condition(hkl: np.ndarray, centering: str) -> np.ndarray:
    """
    Returns a boolean mask indicating which reflections satisfy the reflection condition
    based on the given lattice centering.

    Parameters
    ----------
    hkl : numpy.ndarray
        Array of shape (N, 3) representing the Miller indices of reflections.
    centering : str
        The lattice centering type: one of "P", "I", "F", "A", "B" or "C", or several
        of them combined, such as "FI", for a cell with the lattice translations of
        each. A reflection must then satisfy all of their conditions.

    Returns
    -------
    numpy.ndarray
        Boolean mask indicating which reflections satisfy the reflection condition.
    """
    centering = centering.lower()
    if not centering or set(centering) - set("pifabc"):
        raise ValueError("Invalid crystal centering type.")

    mask = np.ones(len(hkl), dtype=bool)
    for letter in centering:
        if letter == "f":
            all_even = (hkl % 2 == 0).all(axis=1)
            all_odd = (hkl % 2 == 1).all(axis=1)
            mask &= all_even | all_odd
        elif letter == "i":
            mask &= hkl.sum(axis=1) % 2 == 0
        elif letter == "a":
            mask &= hkl[:, [1, 2]].sum(axis=1) % 2 == 0
        elif letter == "b":
            mask &= hkl[:, [0, 2]].sum(axis=1) % 2 == 0
        elif letter == "c":
            mask &= hkl[:, [0, 1]].sum(axis=1) % 2 == 0
    return mask


@njit(nogil=True)
def fast_filter_excitation_errors(
    mask: np.ndarray,
    g: np.ndarray,
    orientation_matrices: np.ndarray,
    wavelength: float,
    sg_max: float,
) -> None:
    g_length_2 = g[:, 0] ** 2 + g[:, 1] ** 2 + g[:, 2] ** 2

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
    hkl : numpy.ndarray
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
    orientation_matrices : numpy.ndarray, optional
        Orientation matrices for each crystallographic direction.

    Returns
    -------
    numpy.ndarray
        Mask for the reciprocal space vectors.
    """
    g = hkl @ reciprocal_cell(cell)
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


def _dense_structure_factor(
    array: np.ndarray, hkl_source: np.ndarray, gpts: tuple[int, int, int]
) -> np.ndarray:
    """The structure factors on the raveled `gpts` grid, zero away from `hkl_source`,
    on the device of `array`."""
    xp = get_array_module(array)
    dense = xp.zeros(int(np.prod(gpts)), dtype=array.dtype)
    dense[xp.asarray(ravel_hkl(hkl_source, gpts))] = array
    return dense


def retrieve_structure_factor_values(
    array: np.ndarray,
    hkl_source: np.ndarray,
    hkl_destination: np.ndarray,
    gpts: tuple[int, int, int],
) -> np.ndarray:
    """
    Look up the structure factors at a set of reciprocal space vectors.

    Parameters
    ----------
    array : numpy.ndarray
        The structure factors as a 1D array.
    hkl_source : numpy.ndarray
        The reciprocal space vectors as Miller indices for the source array.
    hkl_destination : numpy.ndarray
        The reciprocal space vectors as Miller indices to look up. Those on the grid
        but not in `hkl_source` (beyond its g_max) give zero.
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.

    Returns
    -------
    numpy.ndarray
        The structure factors at `hkl_destination`, on the device of `array`.
    """
    xp = get_array_module(array)
    dense = _dense_structure_factor(array, hkl_source, gpts)
    return dense[xp.asarray(ravel_hkl(hkl_destination, gpts))]


def structure_factor_coupling_matrix(
    array: np.ndarray,
    hkl_source: np.ndarray,
    hkl: np.ndarray,
    gpts: tuple[int, int, int],
    inversion_center: Optional[np.ndarray] = None,
    max_block_size: int = 2**26,
) -> np.ndarray:
    """The matrix of structure factors coupling a set of beams, A[i, j] = U(g_j - g_i).

    The matrix is gathered on the device of `array` from the structure factors on a
    dense grid, with the raveled index of g_j - g_i computed from those of g_i and
    g_j: no array of the beams x beams Miller index differences is made. The grid is
    enlarged as needed to hold every difference; differences beyond the g_max of
    `hkl_source` couple through zero.

    Parameters
    ----------
    array : numpy.ndarray
        The structure factors as a 1D array.
    hkl_source : numpy.ndarray
        The reciprocal space vectors of `array` as Miller indices, (M, 3).
    hkl : numpy.ndarray
        The Miller indices of the beams, (N, 3).
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.
    inversion_center : numpy.ndarray, optional
        A center of inversion symmetry t of the crystal in scaled coordinates (see
        `find_inversion_center`). If given, the matrix is that of the crystal moved
        by -t: real, U(g) exp(-2 pi i g.t), and of the real dtype. It is D A D^H with
        D = diag(exp(2 pi i g_i.t)).
    max_block_size : int
        The largest number of matrix elements gathered at once, bounding the memory
        of the temporary index array.

    Returns
    -------
    numpy.ndarray
        The (N, N) matrix, Fortran-ordered (the layout LAPACK and cuSOLVER use).
    """
    xp = get_array_module(array)
    hkl = np.asarray(hkl)
    hkl_source = np.asarray(hkl_source)

    if inversion_center is not None:
        phase = np.exp(-2j * np.pi * (hkl_source @ np.asarray(inversion_center)))
        array = (array * xp.asarray(phase, dtype=array.dtype)).real

    n = len(hkl)
    if n == 0:
        return xp.zeros((0, 0), dtype=array.dtype)

    # A grid holding both the structure factors and every difference g_j - g_i,
    # whose components span [-span, span]: the `gpts` grid can be too small when
    # the beams' g_max exceeds half the structure factor's.
    span = hkl.max(axis=0) - hkl.min(axis=0)
    nx, ny, nz = (max(int(m), 2 * int(s) + 1) for m, s in zip(gpts, span))
    shape = (nx, ny, nz)
    dense = _dense_structure_factor(array, hkl_source, shape)

    # The raveled index is linear in the Miller indices, so the index of g_i - g_j
    # is that of g_i less that of g_j (as the components are on the grid).
    strides = np.array((shape[1] * shape[2], shape[2], 1))
    offset = int(ravel_hkl(np.zeros(3, dtype=int), shape))
    index_dtype = np.int32 if dense.size < 2**31 else np.int64
    rows = xp.asarray(hkl @ strides + offset, dtype=index_dtype)
    columns = xp.asarray(hkl @ strides, dtype=index_dtype)

    # transposed[i, j] = U(g_i - g_j) = A[j, i]: gathering the transpose in C order
    # gives A in Fortran order
    transposed = xp.empty((n, n), dtype=array.dtype)
    block = max(1, max_block_size // n)
    for start in range(0, n, block):
        stop = min(start + block, n)
        transposed[start:stop] = dense[rows[start:stop, None] - columns[None, :]]
    return transposed.T


def relative_positions_for_centering() -> dict[str, np.ndarray]:
    """
    Returns the relative positions for each lattice centering type.

    Returns
    --------
    dict
        A dictionary where the keys are the centering types and the values are the
        relative positions.
    """
    return {
        "F": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ),
        "I": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]
        ),
        "A": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
            ]
        ),
        "B": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
            ]
        ),
        "C": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ),
        "P": np.array([[0.0, 0.0, 0.0]]),
    }


def wrapped_is_close(a, b):
    differences = (a[None] - b[:, None]) % 1.0
    is_close_to_zero = np.isclose(differences, 0.0)
    is_close_to_one = np.isclose(differences, 1.0)
    position_is_close = is_close_to_zero | is_close_to_one
    return position_is_close


def all_positions_have_relative_periodic_pair(
    positions: np.ndarray, relative_positions: np.ndarray
) -> bool:
    """
    Check if all positions have a relative periodic pair.

    Parameters
    ----------
    positions : numpy.ndarray
        The positions to check.
    relative_positions : numpy.ndarray
        The possible relative shifts of each position.

    Returns
    --------
    bool
        True if all positions have a relative periodic pair, False otherwise.
    """
    basis_size = len(positions) / (len(relative_positions))

    if not np.isclose(basis_size, np.round(basis_size), atol=1e-6):
        return False

    # Every translate of every position must be one of the positions. Counting
    # matches instead lets a position that matches several others (coincident
    # atoms) stand in for a translate that is missing.
    for position in positions:
        shifted_position = (position + relative_positions) % 1.0
        position_is_close = wrapped_is_close(shifted_position, positions)
        if not position_is_close.all(axis=2).any(axis=0).all():
            return False

    return True


def _sites_with_multiplicity(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident scaled positions (periodically) into sites, returning the
    distinct sites and the number of positions on each."""
    is_close = wrapped_is_close(positions, positions).all(axis=2)
    assigned = np.zeros(len(positions), dtype=bool)
    sites, multiplicities = [], []
    for i in range(len(positions)):
        if assigned[i]:
            continue
        on_site = is_close[i] & ~assigned
        assigned |= on_site
        sites.append(positions[i])
        multiplicities.append(int(on_site.sum()))
    return np.array(sites).reshape((-1, 3)), np.array(multiplicities, dtype=int)


def auto_detect_centering(
    atoms: Atoms, centerings_to_check: Optional[Iterable[str]] = None
) -> str:
    """
    Automatically detect the lattice centering of a crystal structure.

    A centering is detected when every atom has a periodic image of the same species
    at each of its lattice translations (see `relative_positions_for_centering`). The
    reflection conditions these translations imply depend only on the fractional
    translations, not on the cell metric, so any cell can be centered. Coincident
    atoms of the same species count as one site with a multiple occupancy, which the
    translations must preserve.

    A cell can have the translations of several centerings: a 2x2x2 supercell has
    those of both F and I, for example. All of them are then returned together, here
    "FI", and `get_reflection_condition` applies each of their conditions.

    Parameters
    ----------
    atoms : Atoms
        The crystal structure.
    centerings_to_check : iterable of str, optional
        The centering types to check. If None, all centering types are checked.

    Returns
    --------
    str
        The detected lattice centering: "P" if none is found, otherwise those found
        among "F", "I", "A", "B" and "C", in that order.
    """
    relative_positions = relative_positions_for_centering()

    if centerings_to_check is None:
        centerings_to_check = relative_positions.keys()
    else:
        centerings_to_check = {centering.upper() for centering in centerings_to_check}
        unknown = centerings_to_check - relative_positions.keys()
        if unknown:
            raise ValueError(f"Invalid crystal centering types: {sorted(unknown)}")

    positions = atoms.get_scaled_positions()

    # Coincident atoms of one species are one site with a multiple occupancy. A
    # centering translation must map each site onto a site of the same species
    # and the same occupancy, so the sites are grouped by both.
    groups = []
    for number in np.unique(atoms.numbers):
        sites, multiplicities = _sites_with_multiplicity(
            positions[atoms.numbers == number]
        )
        groups.extend(sites[multiplicities == m] for m in np.unique(multiplicities))

    detected = [
        centering
        for centering in "FIABC"
        if centering in centerings_to_check
        and all(
            all_positions_have_relative_periodic_pair(
                group, relative_positions[centering]
            )
            for group in groups
        )
    ]

    # The F-centering translations include those of A-, B- and C-centering.
    if "F" in detected:
        detected = [centering for centering in detected if centering not in "ABC"]

    return "".join(detected) or "P"


def _wrap_scaled(positions: np.ndarray) -> np.ndarray:
    """Scaled positions in [0, 1), as a periodic cKDTree requires (the remainder of a
    small negative number rounds to 1.0)."""
    positions = positions % 1.0
    positions[positions >= 1.0] = 0.0
    return positions


def find_inversion_center(
    atoms: Atoms,
    thermal_sigma: AtomProperties = 0.0,
    occupancy: AtomProperties = 1.0,
    tol: float = 1e-6,
) -> Optional[np.ndarray]:
    """
    Find a center of inversion symmetry of a crystal.

    The crystal is centrosymmetric about t when inverting it through t, r -> 2t - r,
    maps every atom onto a periodic image of an atom of the same species, thermal
    displacement and occupancy, with coincident atoms counted. Its structure factors
    are then U(g) = exp(2 pi i g.t) times a real number, which makes the Bloch-wave
    eigenproblem real symmetric up to a diagonal phase. If t is a center, so is
    t + n / 2 for any lattice vector n; one of them is returned.

    Parameters
    ----------
    atoms : Atoms
        The crystal structure.
    thermal_sigma : float, dict or sequence
        The thermal displacements of the atoms, as for the structure factors.
    occupancy : float, dict or sequence
        The occupancies of the atoms, as for the structure factors.
    tol : float
        The largest distance in scaled coordinates at which an inverted atom matches
        an atom. Small enough that the asymmetry it could hide is below the rounding
        of single precision.

    Returns
    -------
    numpy.ndarray or None
        The center in scaled coordinates, or None if the crystal has none.
    """
    from scipy.spatial import cKDTree  # type: ignore

    if len(atoms) == 0:
        return None

    sigmas = validate_sigmas(atoms, thermal_sigma, return_array=True)[0]
    occupancies = validate_per_atom_property(atoms, occupancy, return_array=True)
    assert isinstance(sigmas, np.ndarray)  # Type narrowing for mypy
    assert isinstance(occupancies, np.ndarray)  # Type narrowing for mypy

    # atoms that may map onto each other: same species, displacement and occupancy
    properties = np.column_stack(
        [atoms.numbers, np.reshape(sigmas, (len(atoms), -1)), occupancies]
    )
    kinds = np.unique(properties, axis=0, return_inverse=True)[1].ravel()

    positions = _wrap_scaled(atoms.get_scaled_positions(wrap=False))
    groups = [positions[kinds == kind] for kind in range(kinds.max() + 1)]
    trees = [cKDTree(group, boxsize=1.0) for group in groups]
    # the number of atoms on each site, which the inversion must preserve
    multiplicities = [
        tree.query_ball_point(group, tol, return_length=True)
        for tree, group in zip(trees, groups)
    ]

    # a center lies halfway between an atom of the rarest kind and its inverse
    rarest = groups[np.argmin([len(group) for group in groups])]
    for partner in rarest:
        center = (rarest[0] + partner) / 2
        if all(
            np.array_equal(
                tree.query_ball_point(
                    _wrap_scaled(2 * center - group), tol, return_length=True
                ),
                multiplicity,
            )
            for tree, group, multiplicity in zip(trees, groups, multiplicities)
        ):
            return center
    return None
