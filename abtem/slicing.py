"""Module for slicing atoms for the multislice algorithm."""

from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import Any, Iterable, Optional, Sequence, TypeGuard, cast

import numpy as np
from ase import Atoms

from abtem.atoms import is_cell_orthogonal
from abtem.core.utils import EqualityMixin, label_to_index


def crystal_slice_thicknesses(atoms: Atoms, tolerance: float = 0.2) -> np.ndarray:
    """
    Calculates slice thicknesses to match the spacing between the crystal planes of a
    given collection of atoms.

    Parameters
    ----------
    atoms: Atoms
        The atoms to be sliced. Must have an orthogonal cell.
    tolerance : float, optional
        The tolerance between atomic positions in the z-direction

    Returns
    -------

    """
    z = atoms.positions[:, 2]
    z = np.insert(z, [0, -1], [0.0, atoms.cell[2, 2]])

    unique = (np.unique(np.floor(z / tolerance).astype(int)) + 0.5) * tolerance
    slice_positions = np.sort(unique)
    slice_thickness = np.diff(slice_positions)

    assert np.isclose(np.sum(slice_thickness), atoms.cell[2, 2])
    return slice_thickness


def commensurate_slice_thickness(
    atoms: Atoms,
    target_thickness: float = 1.0,
    tolerance: float = 0.2,
) -> tuple[float, ...]:
    """
    Find slice thicknesses commensurate with the crystal planes, closest to a target
    thickness.

    Unique z-positions (within `tolerance`) define candidate slice boundaries between 0
    and the cell height.  Adjacent plane-level slices are merged so that each resulting
    slice thickness is as close as possible to `target_thickness` while keeping
    boundaries aligned with atomic planes.

    Parameters
    ----------
    atoms : Atoms
        The atoms to be sliced. Must have an orthogonal cell.
    target_thickness : float
        Target slice thickness [Å].
    tolerance : float
        Tolerance for identifying distinct crystal planes [Å].

    Returns
    -------
    tuple of float
        Commensurate slice thicknesses.
    """
    cell_z = float(atoms.cell[2, 2])
    z = atoms.positions[:, 2] % cell_z

    unique_z = np.unique(np.round(z / tolerance).astype(int)) * tolerance
    unique_z = unique_z[(unique_z > tolerance / 2) & (unique_z < cell_z - tolerance / 2)]

    boundaries = np.sort(np.concatenate(([0.0], unique_z, [cell_z])))
    plane_thicknesses = np.diff(boundaries)

    merged = []
    acc = 0.0
    for t in plane_thicknesses:
        if acc > 0 and acc + t > target_thickness and acc >= target_thickness * 0.5:
            merged.append(acc)
            acc = t
        else:
            acc += t
    if acc > 0:
        merged.append(acc)

    result = tuple(float(t) for t in merged)
    assert np.isclose(sum(result), cell_z)
    return result


def commensurate_gpts(
    extent: tuple[float, float],
    positions: np.ndarray,
    target_sampling: float = 0.05,
    tolerance: float = 1e-3,
) -> tuple[int, int]:
    """
    Find grid points such that the sampling grid is commensurate with the atom
    positions in x and y, closest to a target sampling.

    For each axis the function identifies the unique atom planes and computes the
    GCD of the spacings between them.  This gives the primitive lattice spacing
    and therefore the required grid period p (= number of grid points per unit
    cell of the primitive lattice).  The result is invariant under rigid
    translations of the structure: a crystal that has been centered or otherwise
    shifted within the cell always yields the same p as the unshifted version.

    Parameters
    ----------
    extent : tuple of float
        Grid extent in x and y [Å].
    positions : np.ndarray
        Atom positions with shape (N, 3) or (N, 2).
    target_sampling : float
        Target grid sampling [Å].
    tolerance : float
        Tolerance for identifying distinct atom planes [Å].

    Returns
    -------
    tuple of int
        Number of grid points in x and y.
    """
    gpts = []
    for i in range(2):
        L = extent[i]
        n_target = int(np.ceil(L / target_sampling))

        x = positions[:, i] % L
        # Snap values at x ≈ L back to 0: floating-point modulo can leave atoms
        # from cell-transform at L−ε rather than 0, creating a spurious near-zero
        # spacing that breaks the GCD calculation.
        x[x > L - tolerance] = 0.0
        x = np.sort(x)

        # De-duplicate: keep one representative per distinct atom plane
        unique_mask = np.concatenate([[True], np.diff(x) > tolerance])
        unique_x = x[unique_mask]

        if len(unique_x) <= 1:
            gpts.append(n_target)
            continue

        # Spacings between consecutive unique planes, including the wrap-around gap
        spacings = np.concatenate([np.diff(unique_x), [L + unique_x[0] - unique_x[-1]]])

        min_spacing = float(np.min(spacings))
        if min_spacing < tolerance:
            gpts.append(n_target)
            continue

        # Check that every spacing is approximately an integer multiple of the
        # minimum spacing (i.e. the minimum is the primitive spacing).
        ratios = spacings / min_spacing
        k = np.round(ratios).astype(int)
        if np.any(np.abs(ratios - k) > 0.05) or np.any(k < 1):
            # No clear commensurability — fall back to the raw target
            gpts.append(n_target)
            continue

        # Primitive spacing refined as L / (total number of primitive periods).
        # Using L / sum(k) is more accurate than min_spacing alone because it
        # averages out any floating-point scatter in the individual spacings.
        n_periods = int(np.sum(k))
        n_multiple = max(round(n_target / n_periods), 1) * n_periods
        gpts.append(n_multiple)

    return tuple(gpts)


def is_number(value: Any) -> TypeGuard[int | float | np.ndarray]:
    """
    Check if the value is a number, including a NumPy array with a single element,
    an integer, or a float.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if the value is a number, False otherwise.
    """
    if isinstance(value, (int, float)):
        return True
    elif isinstance(value, np.ndarray) and value.size == 1:
        return isinstance(value.item(), (int, float))
    else:
        return False


def _validate_slice_thickness(
    slice_thickness: float | np.ndarray | Sequence[float],
    thickness: Optional[float] = None,
    num_slices: Optional[int] = None,
) -> tuple[float, ...]:
    if is_number(slice_thickness):
        st_value = slice_thickness.item() if isinstance(slice_thickness, np.ndarray) else float(slice_thickness)
        if st_value <= 0.0:
            raise ValueError(
                f"slice_thickness must be positive, got {slice_thickness}"
            )
        if thickness is not None:
            thickness = float(thickness)
            n = float(np.ceil(thickness / slice_thickness))
            validated_slice_thickness = (thickness / n,) * int(n)
        elif num_slices is not None:
            if isinstance(slice_thickness, np.ndarray):
                slice_thickness = cast(float, slice_thickness.item())
            validated_slice_thickness = (float(slice_thickness),) * num_slices
        else:
            raise RuntimeError("Either thickness or num_slices must be given.")
    elif isinstance(slice_thickness, Iterable):
        validated_slice_thickness = tuple(float(d) for d in slice_thickness)

    if thickness is not None:
        if not np.isclose(np.sum(validated_slice_thickness), thickness):
            raise RuntimeError(
                f"Sum of slice thicknesses must be equal to the depth of the cell. "
                f"Slice thicknesses: {np.sum(slice_thickness)}, thickness: {thickness}"
            )

    if num_slices is not None:
        if len(validated_slice_thickness) != num_slices:
            raise RuntimeError(
                "Number of slice thicknesses must match the number of slices."
            )

    return validated_slice_thickness


def slice_limits(slice_thickness) -> list[tuple[float, float]]:
    """The entrance and exit thicknesses of each slice [Å]."""

    cum_thickness = list(itertools.accumulate((0,) + slice_thickness))
    limits = [
        (cum_thickness[i], cum_thickness[i + 1]) for i in range(len(cum_thickness) - 1)
    ]
    return limits


def _unpack_item(item: int | slice, num_items: int) -> tuple[int, int]:
    """
    Unpacks an item to a first and last index.

    Parameters
    ----------
    item : int or slice
        The item to unpack.
    num_items : int
        The number of items.

    Returns
    -------
    first_index : int
        The first index.
    last_index : int
        The last index.
    """

    if isinstance(item, int):
        first_index = item
        last_index = first_index + 1
    elif isinstance(item, slice):
        first_index = 0 if item.start is None else item.start
        last_index = num_items if item.stop is None else item.stop
    else:
        raise RuntimeError("item must be an int or a slice")

    last_index = min(last_index, num_items)

    if first_index >= last_index:
        raise IndexError

    return first_index, last_index


class BaseSlicedAtoms(EqualityMixin):
    """
    Base class for sliced atoms used for grouping each atom in an ASE atoms object into
    a collection of slices along the z-direction.

    Parameters
    ----------
    atoms: Atoms
        The atoms to be sliced. Must have an orthgonal cell.
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å]
        (default is 0.5 Å).
        If given as a float, the number of slices is calculated by dividing the slice
        thickness into the `z`-height of
        supercell. The slice thickness may be given as a sequence of values for each
        slice, in which case an error will be thrown if the sum of slice thicknesses
        is not equal to the height of the atoms.
    """

    def __init__(self, atoms: Atoms, slice_thickness: float | Sequence[float] | str):
        if not is_cell_orthogonal(atoms):
            raise RuntimeError("atoms must have an orthogonal cell")

        self._atoms = atoms

        if isinstance(slice_thickness, str):
            raise NotImplementedError

        self._slice_thickness = _validate_slice_thickness(
            slice_thickness, thickness=atoms.cell[2, 2]
        )

    def __len__(self) -> int:
        return self.num_slices

    @property
    def atoms(self) -> Atoms:
        """The Atoms before slicing."""
        return self._atoms

    @property
    def box(self) -> tuple[float, float, float]:
        """The simulation box [Å]."""
        diag = np.diag(self._atoms.cell)
        return float(diag[0]), float(diag[1]), float(diag[2])

    @property
    def num_slices(self) -> int:
        """Number of projected potential slices."""
        return len(self._slice_thickness)

    @property
    def slice_thickness(self) -> tuple[float, ...]:
        """Slice thicknesses for each slice."""
        return self._slice_thickness

    @property
    def slice_limits(self) -> list[tuple[float, float]]:
        """The entrance and exit thicknesses of each slice [Å]."""
        return slice_limits(self.slice_thickness)

    def check_slice_idx(self, index: int):
        """Raises an error if index is greater than the number of slices."""
        if index >= self.num_slices:
            raise RuntimeError(
                f"Slice index {index} too large for sliced atoms with {self.num_slices}"
                f"slices"
            )

    @abstractmethod
    def get_atoms_in_slices(
        self,
        first_slice: int,
        last_slice: Optional[int] = None,
        atomic_number: Optional[int] = None,
    ) -> Atoms:
        """
        Get the atoms between two slice indices.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the atoms to return.
        last_slice : int, optional
            Index of the last slice of the atoms to return.
        atomic_number : int, optional
            If given, only atoms with the given atomic number is returned.

        Returns
        -------
        atoms : Atoms
        """

    def generate_atoms_in_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        atomic_number: Optional[int] = None,
    ):
        """
        Generate atoms in slices.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the atoms to return.
        last_slice : int, optional
            Index of the last slice of the atoms to return.
        atomic_number : int, optional
            If given, only atoms with the given atomic number is returned.

        Yields
        ------
        atoms : Atoms
            The atoms in each slice.
        """
        if last_slice is None:
            last_slice = len(self)

        for i in range(first_slice, last_slice):
            yield self.get_atoms_in_slices(i, atomic_number=atomic_number)

    def __getitem__(self, item: int | slice) -> Atoms:
        return self.get_atoms_in_slices(*_unpack_item(item, len(self)))


# def find_closest_indices(list1, list2):
#     # Convert lists to NumPy arrays
#     arr1 = np.array(list1)[:, np.newaxis]  # Convert to column vector
#     arr2 = np.array(list2)

#     # Calculate the absolute differences using broadcasting
#     differences = np.abs(arr1 - arr2)

#     # Find the indices of the minimum differences
#     closest_indices = np.argmin(differences, axis=1)

#     return closest_indices


class SliceIndexedAtoms(BaseSlicedAtoms):
    """
    Sliced atoms assigning each atom to a specific slice index.

    Parameters
    ----------
    atoms: Atoms
        The atoms to be sliced. Must have an orthgonal cell.
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å]
        (default is 0.5 Å).
        If given as a float, the number of slices is calculated by dividing the slice
        thickness into the `z`-height of
        supercell. The slice thickness may be given as a sequence of values for each
        slice, in which case an error will be thrown if the sum of slice thicknesses is
        not equal to the height of the atoms.
    """

    def __init__(
        self,
        atoms: Atoms,
        slice_thickness: float | Sequence[float],
    ):
        super().__init__(atoms, slice_thickness)

        bin_edges = np.array(self.slice_thickness).cumsum()

        # Guard against floating-point accumulation error in cumsum:
        # atoms exactly on a slice boundary can be assigned to the wrong
        # slice when the cumulative sum drifts by ~1e-14.  Nudge each
        # edge down by a small tolerance so that an atom sitting
        # precisely at z = sum(thicknesses[:k]) falls into slice k
        # (the next slice), not slice k-1.
        bin_edges -= 1e-12

        labels = np.digitize(self.atoms.positions[:, 2], bin_edges)

        self._slice_index = [
            indices for indices in label_to_index(labels, max_label=len(self) - 1)
        ]

    def get_atoms_in_slices(
        self,
        first_slice: int,
        last_slice: Optional[int] = None,
        atomic_number: Optional[int] = None,
    ) -> Atoms:
        if last_slice is None:
            last_slice = first_slice

        if last_slice - first_slice < 2:
            in_slice = self._slice_index[first_slice]
        else:
            in_slice = np.concatenate(self._slice_index[first_slice:last_slice])

        atoms = self.atoms[in_slice]

        if atomic_number is not None:
            atoms = atoms[(atoms.numbers == atomic_number)]

        slice_thickness = self.slice_thickness[first_slice:last_slice]
        atoms.cell[2, 2] = np.sum(slice_thickness)
        atoms.positions[:, 2] -= np.sum(self.slice_thickness[:first_slice])
        return atoms


class SlicedAtoms(BaseSlicedAtoms):
    """
    Sliced atoms assigning each atom to multiple slices.

    Parameters
    ----------
    atoms: Atoms
        The atoms to be sliced. Must have an orthgonal cell.
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å]
        (default is 0.5 Å).
        If given as a float, the number of slices is calculated by dividing the slice
        thickness into the `z`-height of
        supercell. The slice thickness may be given as a sequence of values for each
        slice, in which case an error will be thrown if the sum of slice thicknesses is
        not equal to the height of the atoms.
    xy_padding : float, optional
        Padding of the atoms in x and y included in each of the slices [Å].
    z_padding : float, optional
        Padding of the atoms along z in each slice included in the slices [Å].
    """

    def __init__(
        self,
        atoms: Atoms,
        slice_thickness: float | Sequence[float],
        xy_padding: float = 0.0,
        z_padding: float = 0.0,
    ):
        super().__init__(atoms, slice_thickness)
        self._xy_padding = xy_padding
        self._z_padding = z_padding

    def get_atoms_in_slices(
        self,
        first_slice: int,
        last_slice: Optional[int] = None,
        atomic_number: Optional[int] = None,
    ) -> Atoms:
        if last_slice is None:
            last_slice = first_slice

        a = self.slice_limits[first_slice][0]
        b = self.slice_limits[last_slice][1]

        in_slice = (self.atoms.positions[:, 2] >= (a - self._z_padding)) * (
            self.atoms.positions[:, 2] < (b + self._z_padding)
        )

        if atomic_number is not None:
            in_slice = (self.atoms.numbers == atomic_number) * in_slice

        atoms = self.atoms[in_slice]
        atoms.cell = tuple(np.diag(atoms.cell)[:2]) + (b - a,)
        return atoms
