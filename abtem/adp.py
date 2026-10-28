"""Anisotropic displacement parameters (ADPs).

The displacement of an atom from its site, from thermal motion and static disorder,
is described by its covariance, the ADP tensor. Its components along the crystal
axes are the crystallographic ``U^ij`` [Å²] of the IUCr and CIF conventions, for
which the Debye-Waller factor of a reflection hkl is

``T(h) = exp(-2 pi^2 sum_ij h_i h_j a*_i a*_j U^ij)``.

In Cartesian coordinates the same tensor is ``U_cart = A N U^ij N A^T`` (see
`cartesian_adp`), and ``T = exp(-2 pi^2 g^T U_cart g)`` with ``g`` the Cartesian
reciprocal vector (``|g| = 1/d``). It enters every simulation mode in the same way:
the averaged potential of an atom is its static potential convolved with a Gaussian
of covariance ``U_cart``, so its scattering factor is damped by ``T``; frozen
phonons draw the displacements from the same Gaussian. An isotropic standard
deviation sigma [Å] along each direction is ``U_cart = sigma^2 I``, and
``U_eq = tr(U_cart) / 3``.

`set_adp` and `get_adp` speak ``U^ij`` relative to the cell of the atoms by default,
so that values from a CIF file or a paper mean what they say; ``cartesian=True``
gives and returns ``U_cart``. `read_cif_adp` reads them from a CIF file.

Inside, the atoms carry ``U_cart`` as the per-atom array named by `ADP_ARRAY`, with
shape ``(N, 3, 3)``, which is unchanged when abTEM repeats, cuts or pads the cell
and which every transformation abTEM applies to the positions also applies to the
tensors: rotating to a plane, orthogonalizing or standardizing the cell,
`abtem.atoms.rotate_atoms` and `abtem.atoms.flip_atoms`. A linear map ``F`` acting
on row vectors, ``r -> r @ F``, takes a tensor to ``F^T U_cart F``. A tensor
proportional to the identity is left as it is, as frozen phonons leave an isotropic
displacement: an isotropic Gaussian is the same along any rotated axes, and the
small strain an orthogonalization may need is not applied to it.

The atoms also record the cell their tensors refer to. Atoms rotated by other means
together with their cell, such as ``ase.Atoms.rotate(..., rotate_cell=True)``, have
their tensors rotated with them when abTEM uses them. Atoms whose positions are
rotated without their cell cannot be told from atoms that were not rotated; rotate
those with `abtem.atoms.rotate_atoms`, or transform the tensors with
`transform_adp`.
"""

from __future__ import annotations

import itertools
import warnings
from numbers import Number
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import atomic_numbers, chemical_symbols

#: The name of the per-atom array holding the Cartesian ADP tensors [Å²].
ADP_ARRAY = "adp"

#: The key of ``atoms.info`` holding the cell the tensors refer to.
ADP_CELL_INFO = "adp_cell"

_8PI2 = 8 * np.pi**2

# Relative tolerance for treating a tensor as isotropic.
_ISOTROPIC_RTOL = 1e-10

# Tensors that agree to this many decimals [Å²] are grouped as one kind of atom
# when the scattering factors are calculated. Typical tensors are 1e-3 to 1e-1 Å²,
# and rotating them leaves noise of order 1e-17.
ADP_DECIMALS = 12

# Tolerance of the frame check, on unit lattice directions and on integer matrices.
_FRAME_TOLERANCE = 1e-6


def is_isotropic(adp: np.ndarray) -> np.ndarray:
    """Whether each tensor is proportional to the identity.

    Parameters
    ----------
    adp : np.ndarray
        Tensors with shape (..., 3, 3).

    Returns
    -------
    isotropic : np.ndarray
        Boolean array with shape (...).
    """
    adp = np.asarray(adp, dtype=float)
    u_iso = np.trace(adp, axis1=-2, axis2=-1) / 3
    deviation = adp - u_iso[..., None, None] * np.eye(3)
    scale = np.max(np.abs(adp), axis=(-2, -1))
    return np.all(
        np.abs(deviation) <= _ISOTROPIC_RTOL * scale[..., None, None], axis=(-2, -1)
    )


def transform_adp(adp: np.ndarray, linear_map: np.ndarray) -> np.ndarray:
    """Transform Cartesian ADP tensors with the linear map applied to the positions.

    Parameters
    ----------
    adp : np.ndarray
        Cartesian tensors with shape (N, 3, 3) [Å²].
    linear_map : np.ndarray
        A 3x3 matrix acting on row vectors, ``r -> r @ linear_map``, such as the
        rotation ``R.T`` of positions transformed as ``R @ r``.

    Returns
    -------
    transformed : np.ndarray
        The tensors ``F^T U F``; tensors proportional to the identity are returned
        unchanged.
    """
    adp = np.asarray(adp, dtype=float)
    linear_map = np.asarray(linear_map, dtype=float)
    transformed = np.einsum("ai,nab,bj->nij", linear_map, adp, linear_map)
    isotropic = is_isotropic(adp)
    transformed[isotropic] = adp[isotropic]
    return transformed


def _record_adp_frame(atoms: Atoms) -> None:
    """Record the current cell of the atoms as the one their tensors refer to."""
    if atoms.has(ADP_ARRAY):
        atoms.info[ADP_CELL_INFO] = np.array(atoms.cell, dtype=float)
    else:
        atoms.info.pop(ADP_CELL_INFO, None)


def _transform_adp_array(atoms: Atoms, linear_map: np.ndarray) -> None:
    """Transform the ADP tensors of `atoms` in place with the linear map applied to
    their positions (a 3x3 matrix acting on row vectors), and record the cell they
    now refer to."""
    if not atoms.has(ADP_ARRAY):
        return
    atoms.set_array(ADP_ARRAY, transform_adp(atoms.get_array(ADP_ARRAY), linear_map))
    _record_adp_frame(atoms)


def _frame_rotation(reference: np.ndarray, cell: np.ndarray) -> Optional[np.ndarray]:
    """The rotation of the frame from a reference cell to a cell, if the cell is the
    reference cell rigidly rotated, with each lattice vector possibly scaled or
    reversed; otherwise None.

    A cell spanning the same lattice (a supercell, or reordered or reversed
    lattice vectors) or with lattice vectors along the same directions (vacuum
    added) has the same frame. A rotation is a proper orthogonal matrix acting on
    row vectors.
    """
    reference = np.asarray(reference, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if (
        reference.shape != (3, 3)
        or np.linalg.matrix_rank(reference) < 3
        or np.linalg.matrix_rank(cell) < 3
    ):
        return None

    lattice = cell @ np.linalg.inv(reference)
    if np.allclose(lattice, np.round(lattice), atol=_FRAME_TOLERANCE):
        return None

    old = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    new = cell / np.linalg.norm(cell, axis=1, keepdims=True)

    best, best_residual = None, np.inf
    for signs in itertools.product((1.0, -1.0), repeat=3):
        target = new * np.array(signs)[:, None]
        # orthogonal Procrustes: the rotation R minimizing |old @ R - target|
        u, _, vt = np.linalg.svd(old.T @ target)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            continue
        residual = np.abs(old @ rotation - target).max()
        if residual < best_residual:
            best, best_residual = rotation, residual

    if best is None or best_residual > _FRAME_TOLERANCE:
        return None
    if np.allclose(best, np.eye(3), atol=_FRAME_TOLERANCE):
        return None
    return best


def _resolve_adp_frame(atoms: Atoms) -> Atoms:
    """The atoms with their tensors in the frame of their current cell.

    If the cell has been rotated since the tensors were set (by ASE, for example),
    a copy with the tensors rotated with it is returned; otherwise the atoms."""
    if not atoms.has(ADP_ARRAY) or ADP_CELL_INFO not in atoms.info:
        return atoms

    rotation = _frame_rotation(atoms.info[ADP_CELL_INFO], atoms.cell)
    if rotation is None:
        return atoms

    atoms = atoms.copy()
    _transform_adp_array(atoms, rotation)
    return atoms


def _symbol_of(key: Any) -> str:
    if isinstance(key, (int, np.integer)):
        return chemical_symbols[key]
    if isinstance(key, str) and key in atomic_numbers:
        return key
    raise ValueError(f"Not a chemical symbol or atomic number: {key!r}")


def _as_tensor(value: Any) -> tuple[np.ndarray, bool]:
    """A U_iso, three diagonal elements, six components in the CIF order (11, 22,
    33, 12, 13, 23) or a 3x3 tensor [Å²], as a 3x3 tensor, and whether it was given
    as a U_iso."""
    value = np.asarray(value, dtype=float)
    if value.shape == ():
        return value * np.eye(3), True
    if value.shape == (3,):
        return np.diag(value), False
    if value.shape == (6,):
        return _tensor_from_cif_order(value), False
    if value.shape == (3, 3):
        return value, False
    raise ValueError(
        "An ADP must be a U_iso, three diagonal elements, six components "
        f"(11, 22, 33, 12, 13, 23) or a 3x3 tensor, got shape {value.shape}."
    )


def _tensor_from_cif_order(components: np.ndarray) -> np.ndarray:
    """Tensors from components in the CIF order (11, 22, 33, 12, 13, 23), shape
    (..., 6) to (..., 3, 3). The cross terms are the matrix elements, not doubled."""
    u11, u22, u33, u12, u13, u23 = np.moveaxis(np.asarray(components, float), -1, 0)
    return np.stack(
        [
            np.stack([u11, u12, u13], axis=-1),
            np.stack([u12, u22, u23], axis=-1),
            np.stack([u13, u23, u33], axis=-1),
        ],
        axis=-2,
    )


def _check_tensors(adp: np.ndarray, labels: Optional[list] = None) -> np.ndarray:
    """Symmetric and positive semi-definite tensors, or an error naming the atoms
    or sites that are not."""
    transposed = np.swapaxes(adp, -1, -2)
    if not np.allclose(adp, transposed, rtol=1e-8, atol=1e-14):
        raise ValueError("ADP tensors must be symmetric.")
    adp = (adp + transposed) / 2

    if len(adp):
        scale = np.max(np.abs(adp), axis=(-2, -1))
        npd = np.any(np.linalg.eigvalsh(adp) < -1e-8 * scale[:, None], axis=1)
        if np.any(npd):
            which = np.flatnonzero(npd)
            names = [labels[i] for i in which] if labels else list(which)
            raise ValueError(
                "ADP tensors must be positive semi-definite (an NPD tensor makes the "
                "Debye-Waller factor grow with the scattering vector); not so for "
                f"{'sites' if labels else 'atoms'} {names[:10]}"
                f"{' and more' if len(names) > 10 else ''}."
            )
    return adp


def validate_adp(atoms: Atoms, adp: Any, cartesian: bool = False) -> np.ndarray:
    """Validate ADPs and return one Cartesian tensor per atom.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms the ADPs belong to.
    adp : float, np.ndarray or dict
        ADPs [Å²], as one of

        - a float, ``U_iso`` for every atom;
        - an (N,) array, ``U_iso`` for each atom;
        - a (3, 3) tensor, or six components in the CIF order
          (11, 22, 33, 12, 13, 23), for every atom;
        - an (N, 3, 3) or (N, 6) array, one tensor per atom;
        - a dict keyed by chemical symbol or atomic number, of a ``U_iso``, three
          diagonal elements, six components or a (3, 3) tensor for each element.

        An isotropic ``U_iso`` is ``U_iso I`` in Cartesian coordinates, whatever the
        cell. Tensors are crystallographic ``U^ij`` relative to the cell of `atoms`
        (IUCr/CIF convention), unless `cartesian` is True.
    cartesian : bool
        If True, tensors are Cartesian ``U_cart`` in the frame of `atoms`.

    Returns
    -------
    adp : np.ndarray
        Cartesian tensors with shape (N, 3, 3) [Å²].
    """
    n = len(atoms)

    def to_cartesian(tensors: np.ndarray, isotropic: np.ndarray) -> np.ndarray:
        tensors = np.array(tensors, dtype=float)
        _check_symmetric(tensors)
        if cartesian or np.all(isotropic):
            return tensors
        if atoms.cell.rank < 3:
            raise ValueError(
                "Crystallographic U^ij need a cell with three lattice vectors; give "
                "Cartesian tensors with cartesian=True."
            )
        tensors[~isotropic] = cartesian_adp(atoms.cell, tensors[~isotropic])
        return tensors

    if isinstance(adp, dict):
        given = {_symbol_of(key): _as_tensor(value) for key, value in adp.items()}
        symbols = atoms.get_chemical_symbols()
        missing = set(symbols) - set(given)
        if missing:
            raise ValueError(f"ADPs are missing for the elements {sorted(missing)}.")
        tensors = np.array([given[symbol][0] for symbol in symbols]).reshape((n, 3, 3))
        isotropic = np.array([given[symbol][1] for symbol in symbols], dtype=bool)
        validated = to_cartesian(tensors, isotropic)
    elif isinstance(adp, Number) or np.shape(adp) in ((), (6,), (3, 3)):
        tensor, isotropic = _as_tensor(adp)
        validated = to_cartesian(
            np.tile(tensor, (n, 1, 1)), np.full(n, isotropic, dtype=bool)
        )
    elif np.shape(adp) == (n,):
        validated = np.asarray(adp, dtype=float)[:, None, None] * np.eye(3)
    elif np.shape(adp) == (n, 6):
        validated = to_cartesian(_tensor_from_cif_order(adp), np.zeros(n, dtype=bool))
    elif np.shape(adp) == (n, 3, 3):
        validated = to_cartesian(adp, np.zeros(n, dtype=bool))
    else:
        raise ValueError(
            f"ADPs must be given as a float, an ({n},) array of U_iso, a (3, 3) "
            f"tensor, six components, or an ({n}, 3, 3) or ({n}, 6) array for {n} "
            f"atoms, got shape {np.shape(adp)}; give three diagonal elements per "
            "element as a dict."
        )

    return _check_tensors(validated)


def _check_symmetric(tensors: np.ndarray) -> None:
    if not np.allclose(tensors, np.swapaxes(tensors, -1, -2), rtol=1e-8, atol=1e-14):
        raise ValueError("ADP tensors must be symmetric.")


def set_adp(atoms: Atoms, adp: Any, cartesian: bool = False) -> Atoms:
    """A copy of the atoms carrying the given ADPs.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms.
    adp : float, np.ndarray or dict
        ADPs [Å²], as for :func:`validate_adp`: tensors are crystallographic
        ``U^ij`` relative to the cell of `atoms` (IUCr/CIF convention, as in a CIF
        file or a paper), unless `cartesian` is True. None removes them.
    cartesian : bool
        If True, tensors are Cartesian ``U_cart`` in the frame of `atoms`, such as
        those of a phonon calculation.

    Returns
    -------
    atoms : ase.Atoms
        A copy of `atoms` carrying the Cartesian tensors in the array named by
        `ADP_ARRAY`, and their cell in ``atoms.info`` under `ADP_CELL_INFO`.
    """
    atoms = atoms.copy()
    if adp is None:
        if atoms.has(ADP_ARRAY):
            del atoms.arrays[ADP_ARRAY]
    else:
        atoms.set_array(ADP_ARRAY, validate_adp(atoms, adp, cartesian=cartesian))
    _record_adp_frame(atoms)
    return atoms


def _cartesian_adp(atoms: Atoms) -> Optional[np.ndarray]:
    """The Cartesian tensors the atoms carry, as stored, or None."""
    if not atoms.has(ADP_ARRAY):
        return None
    return np.asarray(atoms.get_array(ADP_ARRAY), dtype=float).reshape(
        (len(atoms), 3, 3)
    )


def get_adp(atoms: Atoms, cartesian: bool = False) -> Optional[np.ndarray]:
    """The ADP tensors the atoms carry [Å²], or None.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms.
    cartesian : bool
        If False (default), the crystallographic ``U^ij`` relative to the cell of
        `atoms` (IUCr/CIF convention); if True, the Cartesian ``U_cart``.

    Returns
    -------
    adp : np.ndarray or None
        Tensors with shape (N, 3, 3).
    """
    adp = _cartesian_adp(_resolve_adp_frame(atoms))
    if adp is None or cartesian:
        return adp
    if atoms.cell.rank < 3:
        raise ValueError(
            "Crystallographic U^ij need a cell with three lattice vectors; use "
            "cartesian=True."
        )
    return crystallographic_adp(atoms.cell, adp)


def sigmas_to_adp(sigmas: np.ndarray) -> np.ndarray:
    """Displacement standard deviations as Cartesian ADP tensors.

    A standard deviation sigma [Å] is the root-mean-square displacement along one
    direction, so ``U = sigma^2``.

    Parameters
    ----------
    sigmas : np.ndarray
        Standard deviations [Å], one per atom with shape (N,), or one per Cartesian
        direction for each atom with shape (N, 3).

    Returns
    -------
    adp : np.ndarray
        Tensors ``sigma^2 I`` or ``diag(sigma^2)`` with shape (N, 3, 3) [Å²].
    """
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.ndim == 1:
        return sigmas[:, None, None] ** 2 * np.eye(3)
    if sigmas.ndim == 2 and sigmas.shape[1] == 3:
        return sigmas[:, :, None] ** 2 * np.eye(3)
    raise ValueError(f"sigmas must have shape (N,) or (N, 3), got {sigmas.shape}.")


def resolve_adp(
    atoms: Atoms, sigmas: Optional[np.ndarray], name: str = "sigmas"
) -> Optional[np.ndarray]:
    """The Cartesian ADP tensors of the atoms, from either standard deviations or
    the tensors they carry, but not both.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms.
    sigmas : np.ndarray or None
        Validated standard deviations [Å] with shape (N,) or (N, 3), or None.
    name : str
        The name of the argument the standard deviations were given as, for the
        error message.

    Returns
    -------
    adp : np.ndarray or None
        Tensors with shape (N, 3, 3) [Å²], or None if neither is given.
    """
    adp = _cartesian_adp(_resolve_adp_frame(atoms))
    if sigmas is None:
        return adp
    if adp is not None:
        raise ValueError(
            f"The atoms carry ADP tensors (the '{ADP_ARRAY}' array) and `{name}` "
            "is also given, which would smear them twice (an explicit zero "
            "included). Give one of them: pass None, or remove the tensors with "
            "abtem.adp.set_adp(atoms, None)."
        )
    return sigmas_to_adp(sigmas)


def adp_kinds(
    numbers: np.ndarray,
    adp: Optional[np.ndarray],
    *properties: np.ndarray,
    in_plane: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Group atoms by atomic number, ADP tensor and other per-atom properties.

    Parameters
    ----------
    numbers : np.ndarray
        Atomic numbers with shape (N,).
    adp : np.ndarray or None
        ADP tensors with shape (N, 3, 3) [Å²], rounded to `ADP_DECIMALS` for the
        grouping.
    properties : np.ndarray
        Further per-atom properties with shape (N,) that the groups must share.
    in_plane : bool
        If True, only the upper left 2x2 block of the tensors is compared, as
        ``(U_xx, U_xy, U_yy)``.

    Returns
    -------
    kinds : np.ndarray
        One row per group: the atomic number, then the (rounded) tensor
        components, then the properties.
    inverse : np.ndarray
        The group of each atom, with shape (N,).
    """
    n = len(numbers)
    columns = [np.asarray(numbers, dtype=float).reshape((n, 1))]

    if adp is not None:
        adp = np.asarray(adp, dtype=float)
        if in_plane:
            components = np.stack([adp[:, 0, 0], adp[:, 0, 1], adp[:, 1, 1]], axis=1)
        else:
            components = adp[:, np.triu_indices(3)[0], np.triu_indices(3)[1]]
        # + 0.0 turns the negative zeros of rounding into zeros.
        columns.append(np.round(components, ADP_DECIMALS) + 0.0)

    for prop in properties:
        columns.append(np.asarray(prop, dtype=float).reshape((n, 1)))

    table = np.concatenate(columns, axis=1)
    return unique_rows(table)


def unique_rows(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The unique rows of a 2D array, and the index of each row among them.

    Rows are compared by their bytes, which is several times faster than
    ``np.unique(..., axis=0)``; a negative zero differs from a zero.

    Returns
    -------
    unique : np.ndarray
        The unique rows.
    inverse : np.ndarray
        The index of each row of `table` in `unique`, with shape (len(table),).
    """
    table = np.ascontiguousarray(table)
    if len(table) == 0:
        return table, np.zeros(0, dtype=int)
    rows = table.view(np.dtype((np.void, table.dtype.itemsize * table.shape[1])))
    _, index, inverse = np.unique(rows.ravel(), return_index=True, return_inverse=True)
    return table[index], inverse.reshape(-1)


def adp_square_roots(adp: np.ndarray) -> np.ndarray:
    """The symmetric square root S of each tensor, with ``S S = U``.

    A diagonal tensor ``diag(sigma**2)`` has the root ``diag(sigma)``. Identical
    tensors are decomposed once.

    Parameters
    ----------
    adp : np.ndarray
        Positive semi-definite tensors with shape (N, 3, 3).

    Returns
    -------
    roots : np.ndarray
        Tensors with shape (N, 3, 3).
    """
    adp = np.asarray(adp, dtype=float)
    kinds, inverse = unique_rows(adp.reshape((len(adp), 9)))
    eigenvalues, eigenvectors = np.linalg.eigh(kinds.reshape((-1, 3, 3)))
    roots = np.sqrt(np.clip(eigenvalues, 0.0, None))
    square_roots = np.einsum("kil,kl,kjl->kij", eigenvectors, roots, eigenvectors)
    diagonal = np.all(
        kinds.reshape((-1, 3, 3)) == np.eye(3) * kinds[:, [0, 4, 8], None], axis=(1, 2)
    )
    # exactly diag(sigma) for a diagonal tensor, without the rounding of eigh
    square_roots[diagonal] = np.sqrt(np.clip(kinds[diagonal][:, [0, 4, 8]], 0.0, None))[
        :, :, None
    ] * np.eye(3)
    return square_roots[inverse]


def kind_tensor(kind_components: np.ndarray) -> np.ndarray:
    """The 3x3 tensor of a row of upper-triangle components from `adp_kinds`."""
    tensor = np.zeros((3, 3))
    tensor[np.triu_indices(3)] = kind_components
    return tensor + np.triu(tensor, 1).T


def _convention_factor(cell: np.ndarray, convention: str) -> np.ndarray:
    """The factor taking ``U^ij`` to the given convention: 1 for ``'U'``, ``8 pi^2``
    for ``'B'`` and ``2 pi^2 a*_i a*_j`` for ``'beta'``."""
    if convention == "U":
        return np.ones((3, 3))
    if convention == "B":
        return np.full((3, 3), _8PI2)
    if convention == "beta":
        lengths = _reciprocal_lengths(cell)
        return 2 * np.pi**2 * np.outer(lengths, lengths)
    raise ValueError(f"convention must be 'U', 'B' or 'beta', not {convention!r}")


def _reciprocal_lengths(cell: np.ndarray) -> np.ndarray:
    """``|a*|, |b*|, |c*|``, without a factor of 2 pi [1/Å]."""
    return np.linalg.norm(np.linalg.inv(cell).T, axis=1)


def _as_matrices(adp: np.ndarray) -> np.ndarray:
    adp = np.asarray(adp, dtype=float)
    if adp.shape[-1] == 6:
        return _tensor_from_cif_order(adp)
    if adp.shape[-2:] != (3, 3):
        raise ValueError(
            f"ADPs must have shape (..., 6) or (..., 3, 3), got {adp.shape}."
        )
    return adp


def cartesian_adp(
    cell: Cell | np.ndarray, adp: np.ndarray, convention: str = "U"
) -> np.ndarray:
    """Convert crystallographic ADPs to Cartesian tensors.

    Parameters
    ----------
    cell : ase.cell.Cell or np.ndarray
        The unit cell, with the lattice vectors as rows [Å].
    adp : np.ndarray
        Crystallographic ADPs with shape (..., 6), in the CIF order
        (11, 22, 33, 12, 13, 23), or (..., 3, 3). The cross terms are the matrix
        elements, not doubled.
    convention : {'U', 'B', 'beta'}
        ``'U'`` for ``U^ij`` [Å²] (CIF ``_atom_site_aniso_U_ij``), ``'B'`` for
        ``B^ij = 8 pi^2 U^ij`` [Å²] (``_atom_site_aniso_B_ij``) and ``'beta'`` for
        the dimensionless ``beta^ij = 2 pi^2 a*_i a*_j U^ij``
        (``_atom_site_aniso_beta_ij``).

    Returns
    -------
    adp : np.ndarray
        Cartesian tensors ``U_cart = A N U^ij N A^T`` with shape (..., 3, 3) [Å²],
        with A the lattice vectors as columns and N = diag(|a*|, |b*|, |c*|).
    """
    cell = np.asarray(cell, dtype=float)
    u_ij = _as_matrices(adp) / _convention_factor(cell, convention)
    AN = cell.T * _reciprocal_lengths(cell)[None]
    return np.einsum("ia,...ab,jb->...ij", AN, u_ij, AN)


def crystallographic_adp(
    cell: Cell | np.ndarray, adp: np.ndarray, convention: str = "U"
) -> np.ndarray:
    """Convert Cartesian tensors to crystallographic ADPs, the inverse of
    `cartesian_adp`.

    Parameters
    ----------
    cell : ase.cell.Cell or np.ndarray
        The unit cell, with the lattice vectors as rows [Å].
    adp : np.ndarray
        Cartesian tensors ``U_cart`` with shape (..., 3, 3) [Å²].
    convention : {'U', 'B', 'beta'}
        Return ``U^ij`` [Å²], ``B^ij = 8 pi^2 U^ij`` [Å²] or
        ``beta^ij = 2 pi^2 a*_i a*_j U^ij``.

    Returns
    -------
    adp : np.ndarray
        ``U^ij = (A N)^-1 U_cart (A N)^-T``, or its B or beta, with shape
        (..., 3, 3).
    """
    cell = np.asarray(cell, dtype=float)
    inverse = np.linalg.inv(cell.T * _reciprocal_lengths(cell)[None])
    u_ij = np.einsum("ia,...ab,jb->...ij", inverse, _as_matrices(adp), inverse)
    return u_ij * _convention_factor(cell, convention)


# The CIF tags of anisotropic ADPs, by convention, as ASE stores them (lower case).
_CIF_ANISO_TAGS = {
    "U": "_atom_site_aniso_u_{}",
    "B": "_atom_site_aniso_b_{}",
    "beta": "_atom_site_aniso_beta_{}",
}
# The CIF tags of isotropic ADPs, and their factor to U_iso.
_CIF_ISO_TAGS = {
    "_atom_site_u_iso_or_equiv": 1.0,
    "_atom_site_b_iso_or_equiv": 1.0 / _8PI2,
}


def _cif_list(tags: dict, key: str) -> Optional[list]:
    values = tags.get(key)
    if values is None:
        return None
    return list(values) if isinstance(values, (list, tuple, np.ndarray)) else [values]


def _cif_floats(tags: dict, key: str) -> Optional[np.ndarray]:
    """The values of a CIF loop as floats, with NaN for '?', '.' and other text.
    ASE has already dropped the standard uncertainties, as in 0.0123(4)."""
    values = _cif_list(tags, key)
    if values is None:
        return None
    return np.array(
        [np.nan if isinstance(value, str) else value for value in values], dtype=float
    )


# The ADP types of _atom_site_adp_type with anisotropic values in the aniso loop.
_CIF_ANISO_TYPES = ("uani", "bani")

# A site whose ADPs deviate from its site symmetry by more than this [Å²] is
# reported; CIF values carry uncertainties of order 1e-4 Å² or less.
_SITE_SYMMETRY_TOLERANCE = 1e-4


def _cif_site_tensors(tags: dict, cell: np.ndarray) -> dict[str, np.ndarray]:
    """The Cartesian ADP tensor of each site of the asymmetric unit, keyed by its
    label.

    A site with a complete row in the aniso loop, matched by label, gets its
    anisotropic tensor; any other site its isotropic value, if any. A site
    declared anisotropic (``_atom_site_adp_type`` Uani or Bani) that falls back,
    and a site whose aniso row has unknown components, are reported.
    """
    labels = _cif_list(tags, "_atom_site_label")
    if labels is None:
        raise ValueError("The CIF has no _atom_site_label loop.")

    adp_types = _cif_list(tags, "_atom_site_adp_type") or _cif_list(
        tags, "_atom_site_thermal_displace_type"
    )
    adp_types = (
        {label: str(t).lower() for label, t in zip(labels, adp_types)}
        if adp_types
        else {}
    )

    tensors: dict[str, np.ndarray] = {}
    incomplete: list[str] = []

    aniso_labels = _cif_list(tags, "_atom_site_aniso_label")
    for convention, tag in _CIF_ANISO_TAGS.items():
        columns = [_cif_floats(tags, tag.format(ij)) for ij in (11, 22, 33, 12, 13, 23)]
        if aniso_labels is None or all(column is None for column in columns):
            continue
        # an omitted off-diagonal column is zero
        columns = [
            np.zeros(len(aniso_labels)) if column is None else column
            for column in columns
        ]
        cartesian = cartesian_adp(cell, np.stack(columns, axis=-1), convention)
        for label, tensor in zip(aniso_labels, cartesian):
            if label in tensors:
                continue
            if np.all(np.isfinite(tensor)):
                tensors[label] = tensor
            else:
                incomplete.append(label)

    declared_without_row = []
    for tag, factor in _CIF_ISO_TAGS.items():
        values = _cif_floats(tags, tag)
        if values is None:
            continue
        for label, value in zip(labels, values):
            if label in tensors or not np.isfinite(value):
                continue
            # an isotropic tensor, U_iso I in Cartesian coordinates
            tensors[label] = value * factor * np.eye(3)
            if adp_types.get(label) in _CIF_ANISO_TYPES and label not in incomplete:
                declared_without_row.append(label)

    if incomplete:
        warnings.warn(
            f"The anisotropic ADPs of the sites {sorted(set(incomplete))} have "
            "unknown components; their isotropic values are used, where given.",
            stacklevel=3,
        )
    if declared_without_row:
        warnings.warn(
            f"The sites {sorted(declared_without_row)} are anisotropic "
            "(_atom_site_adp_type), but have no row in the aniso loop; their "
            "equivalent isotropic values are used.",
            stacklevel=3,
        )
    multipole = [label for label, t in adp_types.items() if t == "umpe"]
    if multipole:
        warnings.warn(
            f"The multipole-expansion ADPs (Umpe) of the sites {sorted(multipole)} "
            "are not supported; their equivalent isotropic values are used.",
            stacklevel=3,
        )

    return tensors


def _site_symmetrized(
    tensor: np.ndarray, operations: np.ndarray
) -> tuple[np.ndarray, float]:
    """The tensor averaged over the Cartesian rotations of the site-symmetry group,
    and the largest change [Å²]."""
    symmetrized = np.mean(
        np.einsum("mia,ab,mjb->mij", operations, tensor, operations), 0
    )
    return symmetrized, float(np.abs(symmetrized - tensor).max())


def _cif_site_positions(tags: dict, cell: np.ndarray) -> np.ndarray:
    """The scaled positions of the sites of the asymmetric unit."""
    fractional = [_cif_floats(tags, f"_atom_site_fract_{axis}") for axis in "xyz"]
    if all(column is not None for column in fractional):
        return np.stack(fractional, axis=-1)

    cartesian = [_cif_floats(tags, f"_atom_site_cartn_{axis}") for axis in "xyz"]
    if all(column is not None for column in cartesian):
        return np.linalg.solve(cell.T, np.stack(cartesian, axis=0)).T

    raise ValueError("The CIF gives no positions of the sites.")


def adp_from_cif_tags(atoms: Atoms, symprec: float = 1e-4) -> Atoms:
    """The ADPs of atoms read from a CIF file by ASE, as Cartesian tensors.

    ``ase.io.read(filename, store_tags=True)`` keeps the tags of a CIF file in
    ``atoms.info``, the site of the asymmetric unit each atom was generated from
    in ``atoms.arrays["spacegroup_kinds"]`` and the space group in
    ``atoms.info["spacegroup"]``, but does not attach the ADPs to the atoms. This
    does: the ADPs of each site, anisotropic (``_atom_site_aniso_U_ij``,
    ``_atom_site_aniso_B_ij`` or ``_atom_site_aniso_beta_ij``) where given, and
    otherwise isotropic (``_atom_site_U_iso_or_equiv`` or
    ``_atom_site_B_iso_or_equiv``), are converted to Cartesian tensors (see
    `cartesian_adp`) and expanded with the space group: an atom generated by an
    operation with rotation R gets ``R_c U R_c^T``, where ``R_c = A R A^-1`` is the
    rotation in Cartesian coordinates and A has the lattice vectors as columns.

    A site falls back to its isotropic value only if it has no complete row in the
    aniso loop; a site declared anisotropic by ``_atom_site_adp_type`` that falls
    back is reported. Each site's tensor must be positive semi-definite, and is
    averaged over the operations that map the site onto itself, so that it has the
    symmetry of its site (a deviation beyond the uncertainty of CIF values is
    reported).

    The ADPs of a CIF are those of the temperature it was measured at
    (``_diffrn_ambient_temperature``), and a refinement includes static disorder
    and model errors in them.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms read by ``ase.io.read(filename, store_tags=True)`` from a CIF file.
    symprec : float
        The tolerance [scaled coordinates] for matching an atom to the symmetry
        operation that generated it.

    Returns
    -------
    atoms : ase.Atoms
        A copy of the atoms carrying their Cartesian ADP tensors [Å²] in the array
        named by `ADP_ARRAY`. Sites without ADPs get a zero tensor, with a
        warning; if the CIF gives no ADPs at all, the copy carries no tensors.
    """
    tags = atoms.info
    if "_atom_site_label" not in tags:
        raise ValueError(
            "The atoms carry no CIF tags; read them with "
            "ase.io.read(filename, store_tags=True), or use read_cif_adp(filename)."
        )

    atoms = atoms.copy()
    cell = np.array(atoms.cell)

    site_tensors = _cif_site_tensors(tags, cell)
    if not site_tensors:
        warnings.warn("The CIF gives no ADPs; the atoms are not smeared.", stacklevel=2)
        return atoms

    labels = _cif_list(tags, "_atom_site_label")
    assert labels is not None
    site_positions = _cif_site_positions(tags, cell)

    kinds = atoms.arrays.get("spacegroup_kinds")
    if kinds is None:
        if len(atoms) != len(labels):
            raise ValueError(
                "The atoms do not record the site of the CIF each was generated "
                "from (the 'spacegroup_kinds' array ASE sets when it reads a CIF)."
            )
        kinds = np.arange(len(atoms))

    missing = sorted(
        {labels[kind] for kind in kinds if labels[kind] not in site_tensors}
    )
    if missing:
        warnings.warn(
            f"The CIF gives no ADPs for the sites {missing}; they are not smeared.",
            stacklevel=2,
        )

    spacegroup = atoms.info.get("spacegroup")
    if spacegroup is not None:
        rotations, translations = zip(*spacegroup.get_symop())
        rotations = np.array(rotations, dtype=float)
        translations = np.array(translations, dtype=float)
    else:
        rotations = np.eye(3)[None]
        translations = np.zeros((1, 3))

    A = cell.T
    cartesian_rotations = A @ rotations @ np.linalg.inv(A)

    def maps(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Which operations map each position onto its target, (N, M)."""
        images = np.einsum("mij,nj->nmi", rotations, positions) + translations[None]
        difference = images - targets[:, None]
        difference -= np.round(difference)
        return np.all(np.abs(difference) < symprec, axis=-1)

    # Each site's tensor is checked, and averaged over the operations that map
    # the site onto itself, so that it has the symmetry of its site.
    site_labels = [label for label in labels if label in site_tensors]
    _check_tensors(
        np.array([site_tensors[label] for label in site_labels]), site_labels
    )
    stabilizers = maps(site_positions, site_positions)
    asymmetric = []
    for index, label in enumerate(labels):
        if label not in site_tensors:
            continue
        symmetrized, change = _site_symmetrized(
            site_tensors[label], cartesian_rotations[stabilizers[index]]
        )
        if change > _SITE_SYMMETRY_TOLERANCE:
            asymmetric.append(label)
        site_tensors[label] = symmetrized
    if asymmetric:
        warnings.warn(
            f"The ADPs of the sites {asymmetric} do not have the symmetry of their "
            f"sites (by more than {_SITE_SYMMETRY_TOLERANCE} Å²); they are "
            "averaged over the site-symmetry operations.",
            stacklevel=2,
        )

    # The operation that generated each atom: R x_site + t = x_atom (mod 1).
    matches = maps(site_positions[kinds], atoms.get_scaled_positions(wrap=False))
    if not np.all(np.any(matches, axis=1)):
        raise RuntimeError(
            "Could not find the symmetry operation that generated every atom."
        )
    operation = np.argmax(matches, axis=1)

    adp = np.zeros((len(atoms), 3, 3))
    for i, (kind, op) in enumerate(zip(kinds, operation)):
        tensor = site_tensors.get(labels[kind])
        if tensor is not None:
            R = cartesian_rotations[op]
            adp[i] = R @ tensor @ R.T

    atoms.set_array(ADP_ARRAY, adp)
    _record_adp_frame(atoms)
    return atoms


def read_cif_adp(filename: str, index: int = -1, symprec: float = 1e-4) -> Atoms:
    """Read a crystal and its ADPs from a CIF file.

    The file is read by ``ase.io.read(filename, index=index, store_tags=True)`` and
    the ADPs it gives are attached to the atoms by `adp_from_cif_tags`.

    Parameters
    ----------
    filename : str
        The CIF file.
    index : int
        The data block to read (default the last, as ``ase.io.read``).
    symprec : float
        The tolerance [scaled coordinates] for matching an atom to the symmetry
        operation that generated it.

    Returns
    -------
    atoms : ase.Atoms
        The atoms, carrying their Cartesian ADP tensors [Å²] in the array named by
        `ADP_ARRAY` if the file gives any.
    """
    import ase.io  # noqa: PLC0415

    atoms = ase.io.read(filename, index=index, format="cif", store_tags=True)
    return adp_from_cif_tags(atoms, symprec=symprec)


def _debye_integral(x: float) -> float:
    """``(1 / x^2) * integral_0^x t / (e^t - 1) dt``, which is ``phi(x) / x`` in the
    notation of Willis & Pryor; zero at x = inf."""
    from scipy import integrate  # noqa: PLC0415

    if np.isinf(x):
        return 0.0
    integral = integrate.quad(lambda t: t / np.expm1(t) if t > 0 else 1.0, 0.0, x)[0]
    return integral / x**2


def debye_adp(
    atoms: Atoms,
    temperature: float,
    debye_temperature: float | dict[str, float],
) -> np.ndarray:
    """Isotropic ADPs of the Debye model.

    The mean-square displacement along any direction of an atom of mass m in a
    Debye solid of Debye temperature Theta at temperature T is

    ``U_iso = 3 hbar^2 / (m k_B Theta) * (phi(x) / x + 1 / 4)``, with x = Theta / T

    and ``phi(x) = (1 / x) integral_0^x t / (e^t - 1) dt`` (Willis & Pryor,
    Thermal Vibrations in Crystallography, 1975). It is ``3 hbar^2 / (4 m k_B
    Theta)`` at T = 0, from the zero-point motion, and ``3 hbar^2 T / (m k_B
    Theta^2)`` at high T.

    The model is an estimate of thermal displacements only; refined ADPs, as in a
    CIF file, also contain static disorder and model errors. It is closest for
    elemental crystals with a Debye temperature fitted to their Debye-Waller
    factors, which can differ from the calorimetric one; in a compound, the Debye
    temperature of an element is not that of the element's own crystal. For the
    full tensors of a given crystal, see `adp_from_phonopy`.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms. Their masses are used.
    temperature : float
        The temperature [K].
    debye_temperature : float or dict
        The Debye temperature [K], for all atoms or keyed by chemical symbol.

    Returns
    -------
    u_iso : np.ndarray
        ``U_iso`` of each atom with shape (N,) [Å²]; attach them with
        ``set_adp(atoms, u_iso)``, which takes them as isotropic in any cell.
    """
    from scipy.constants import atomic_mass, hbar, k  # noqa: PLC0415

    if temperature < 0:
        raise ValueError("The temperature must not be negative.")

    symbols = atoms.get_chemical_symbols()
    if isinstance(debye_temperature, dict):
        thetas = {_symbol_of(key): value for key, value in debye_temperature.items()}
        missing = set(symbols) - set(thetas)
        if missing:
            raise ValueError(
                f"Debye temperatures are missing for the elements {sorted(missing)}."
            )
        theta = np.array([thetas[symbol] for symbol in symbols], dtype=float)
    else:
        theta = np.full(len(atoms), float(debye_temperature))

    if np.any(theta <= 0):
        raise ValueError("Debye temperatures must be positive.")

    masses = atoms.get_masses() * atomic_mass

    u_iso = np.zeros(len(atoms))
    for value in np.unique(theta):
        x = value / temperature if temperature > 0 else np.inf
        bracket = _debye_integral(x) + 0.25
        selected = theta == value
        u_iso[selected] = 3 * hbar**2 / (masses[selected] * k * value) * bracket

    return u_iso * 1e20


def adp_from_phonopy(
    phonon: Any,
    temperature: float,
    mesh: float | tuple[int, int, int] = 100.0,
    freq_min: float = 0.01,
    cell: str = "unitcell",
) -> Atoms:
    """Anisotropic ADPs from a harmonic phonon calculation with phonopy.

    The tensors are phonopy's thermal displacement matrices: for atom j,

    ``U_j = hbar / (2 m_j N_q) sum_{q, nu} e_j e_j^dagger coth(hbar omega /
    (2 k_B T)) / omega``

    summed over the phonon modes nu of the q-point mesh, with their eigenvectors e
    and frequencies omega. These are thermal displacements only; refined ADPs, as
    in a CIF file, also contain static disorder and model errors. The force
    constants may come from any calculator, such
    as DFT or a machine-learned potential, or from a database: phonondb
    (``phonopy_params.yaml.xz`` for each of ~10,000 Materials Project entries, on
    NIMS MDR) or the force constants of the Materials Project.

    Parameters
    ----------
    phonon : phonopy.Phonopy or str
        A phonopy object with force constants, or a file for ``phonopy.load``, such
        as ``phonopy_params.yaml.xz``.
    temperature : float
        The temperature [K].
    mesh : float or three int
        The q-point mesh, as for ``Phonopy.run_mesh``: a length that sets its
        density, or the numbers of points. The mesh is centred on Gamma, which keeps
        the symmetry of the crystal (a shifted even mesh does not: 12^3 gives fcc Cu
        a spurious U_xy of 2 % of U_xx). The low-frequency acoustic modes weigh most
        in U, so check that it is converged: for fcc Cu at 300 K, U_xx is 0.00608,
        0.00633 and 0.00645 Å² on meshes of 12^3, 24^3 and 48^3 (which the default
        gives for Cu).
    freq_min : float
        Modes below this frequency [THz] are left out, as for
        ``Phonopy.run_thermal_displacement_matrices``. It must exclude the acoustic
        modes at Gamma, whose frequencies are zero up to rounding (phonopy's own
        default of 0 includes them when the rounding makes them positive, and
        their 1 / omega dominates U), but no mode of another q-point: the lowest
        of those is ~0.2 THz for fcc Cu on a 48^3 mesh.
    cell : {'unitcell', 'primitive'}
        Return the unit cell of the phonon calculation, or its primitive cell.

    Returns
    -------
    atoms : ase.Atoms
        The cell, carrying the ADP tensors [Å²] of the phonon calculation (see
        `get_adp` for ``U^ij`` or ``U_cart``).
    """
    if isinstance(phonon, str):
        import phonopy  # noqa: PLC0415

        phonon = phonopy.load(phonon)

    if cell not in ("unitcell", "primitive"):
        raise ValueError(f"cell must be 'unitcell' or 'primitive', not {cell!r}")

    phonon.run_mesh(
        mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    frequencies = np.asarray(phonon.mesh.frequencies)
    if np.any(frequencies < -freq_min):
        warnings.warn(
            f"The phonons have imaginary modes, down to {frequencies.min():.3g} THz; "
            "they are left out of the ADPs, which are then not those of a stable "
            "crystal.",
            stacklevel=2,
        )
    phonon.run_thermal_displacement_matrices(
        temperatures=[temperature], freq_min=freq_min
    )
    matrices = phonon.thermal_displacement_matrices.thermal_displacement_matrices
    primitive_adp = np.asarray(matrices[0], dtype=float)

    def to_atoms(phonopy_atoms):
        return Atoms(
            symbols=list(phonopy_atoms.symbols),
            cell=np.array(phonopy_atoms.cell),
            scaled_positions=np.array(phonopy_atoms.scaled_positions),
            masses=np.array(phonopy_atoms.masses),
            pbc=True,
        )

    primitive = to_atoms(phonon.primitive)
    if cell == "primitive":
        return set_adp(primitive, primitive_adp, cartesian=True)

    # Each atom of the unit cell is a lattice translate of an atom of the
    # primitive cell, whose tensor it has.
    unitcell = to_atoms(phonon.unitcell)
    inverse_cell = np.linalg.inv(np.array(primitive.cell))
    difference = (
        unitcell.positions[:, None] - primitive.positions[None]
    ) @ inverse_cell
    distance = np.abs(difference - np.round(difference)).max(axis=-1)
    match = distance.argmin(axis=1)
    if np.any(distance[np.arange(len(unitcell)), match] > 1e-5):
        raise RuntimeError("Could not map the unit cell onto the primitive cell.")

    return set_adp(unitcell, primitive_adp[match], cartesian=True)


def debye_waller_factors(
    adp: np.ndarray, g_vec: np.ndarray, isotropic: Optional[np.ndarray] = None
) -> np.ndarray:
    """The Debye-Waller factors ``exp(-2 pi^2 g^T U g)``.

    Parameters
    ----------
    adp : np.ndarray
        Tensors with shape (K, D, D) [Å²].
    g_vec : np.ndarray
        Scattering vectors with shape (N_g, D) [1/Å].
    isotropic : np.ndarray, optional
        Whether each tensor is proportional to the identity, in which case the
        factor is evaluated from ``|g|^2``.

    Returns
    -------
    dwf : np.ndarray
        Factors with shape (K, N_g).
    """
    adp = np.asarray(adp, dtype=float)
    g_vec = np.asarray(g_vec, dtype=float)
    if isotropic is None:
        isotropic = (
            is_isotropic(adp) if adp.shape[-1] == 3 else np.zeros(len(adp), bool)
        )

    exponent = np.empty((len(adp), len(g_vec)))
    g2 = None
    for k, (tensor, iso) in enumerate(zip(adp, isotropic)):
        if iso:
            if g2 is None:
                g2 = np.sum(g_vec**2, axis=1)
            exponent[k] = np.trace(tensor) / len(tensor) * g2
        else:
            exponent[k] = np.sum((g_vec @ tensor) * g_vec, axis=1)

    return np.exp(-2 * np.pi**2 * exponent)
