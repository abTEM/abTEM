"""Module for slicing atoms for the multislice algorithm."""

from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import Any, Iterable, Optional, Sequence, TypeGuard, cast

import numpy as np
from ase import Atoms

from abtem.atoms import is_cell_orthogonal, _wrap_and_snap_atoms
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
    wrap : bool, optional
        If True (default), atoms outside the cell are wrapped into it before they
        are assigned to slices. The given atoms are not modified.
    """

    def __init__(
        self,
        atoms: Atoms,
        slice_thickness: float | Sequence[float],
        wrap: bool = True,
    ):
        # Atoms outside the cell must be wrapped before they are binned, and
        # not every caller has done so: Potential._prepare_atoms wraps, but
        # explicit core-loss ``sites`` and CrystalPotential's tiled atoms are
        # handed here raw. np.digitize returns 0 for any z below the first bin
        # edge -- including arbitrarily negative z -- so an unwrapped atom was
        # assigned to slice 0 whatever its true wrapped depth, and one above
        # the last edge was discarded by label_to_index. Both are silent, and
        # the mis-sliced one is the damaging case: it ionises at the wrong
        # depth, which is exactly what a depth-resolved measurement is trying
        # to resolve.
        #
        # ``wrap`` exists because that is only right for a periodic potential.
        # Potential(periodic=False) deliberately does not wrap -- its atoms are
        # cut from a larger repeated potential and randomised *after* padding,
        # so an edge atom displaced just outside the cell belongs at the face
        # it left, not at the opposite one. Callers that know the potential's
        # convention pass it through; the default wraps, since that is right
        # for everything else that reaches here.
        #
        # Validate first: wrapping reads np.diag(atoms.cell) and would
        # silently ignore the shear terms of a non-orthogonal cell before
        # BaseSlicedAtoms raises for it.
        super().__init__(atoms, slice_thickness)

        if wrap:
            self._atoms = _wrap_and_snap_atoms(self.atoms)

        bin_edges = np.array(self.slice_thickness).cumsum()

        # Guard against floating-point accumulation error in cumsum:
        # atoms exactly on a slice boundary can be assigned to the wrong
        # slice when the cumulative sum drifts by ~1e-14.  Nudge each
        # edge down by a small tolerance so that an atom sitting
        # precisely at z = sum(thicknesses[:k]) falls into slice k
        # (the next slice), not slice k-1.
        bin_edges -= 1e-12

        labels = np.digitize(self.atoms.positions[:, 2], bin_edges)

        # label_to_index silently discards labels outside [0, num_slices - 1].
        # np.digitize against increasing bins returns [0, len(bins)], never
        # negative, so only the upper end can escape: an atom below the entrance
        # face already lands in slice 0.
        #
        # With wrap=False (a non-periodic potential) an atom displaced just
        # outside the cell is expected, e.g. by frozen phonons randomised after
        # padding. It belongs to the face it left, so clamp it into the last
        # slice, mirroring what digitize already does at the entrance face. Its
        # true position is kept: the infinite projection does not depend on z
        # within a slice, so the projected potential is conserved exactly, where
        # dropping the atom lost its whole contribution.
        if not wrap:
            labels = np.minimum(labels, len(self) - 1)

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


def _sum_planes(array, start: int, stop: int):
    """
    The sum of the planes `start` up to, not including, `stop` of the grid that
    repeats `array` along its last axis, taken from the one period.
    """
    # The planes start:stop of the repeated grid are the end of one period, whole
    # periods and the start of one period. The partial periods are summed as views
    # and the whole periods as `periods` times the sum of the grid, so no plane is
    # copied.
    n = array.shape[-1]
    head_end = min(stop, (start // n + 1) * n)
    summed = np.sum(array[..., start % n : start % n + head_end - start], axis=-1)
    periods, tail = divmod(stop - head_end, n)
    if periods:
        summed = summed + periods * np.sum(array, axis=-1)
    if tail:
        summed = summed + np.sum(array[..., :tail], axis=-1)
    return summed


def _integrate_planes(array, start: float, stop: float):
    """
    The integral of the planes of `array`, repeated periodically along its last
    axis, from `start` to `stop` (`start` < `stop`), in units of the plane spacing:
    multiply by the spacing for an integral over heights.

    Plane k is the sample at the height k and stands for the heights from k - 1/2
    to k + 1/2, so each plane is weighted by the length of its heights that lie
    between the limits. Consecutive slices share their limit, so the weights of
    every plane add up to one. A limit on a plane splits it between the slices on
    either side (the trapezoidal rule), so the integral of a slice is centred on
    the slice; summing the planes from the lower limit up to the upper one whole (a
    left Riemann sum) would put it half a plane below the slice.
    """
    n = array.shape[-1]
    # The limits half a plane up, so that plane k stands for the interval from k
    # to k + 1. Python floats, so that the weights keep the dtype of `array`: a
    # NumPy float64 scalar would promote float32 planes to float64.
    u, v = float(start) + 0.5, float(stop) + 0.5
    first, last = int(np.floor(u)), int(np.floor(v))
    if first == last:
        return (v - u) * array[..., first % n]

    summed = (first + 1 - u) * array[..., first % n] + (v - last) * array[..., last % n]
    if last > first + 1:
        summed = summed + _sum_planes(array, first + 1, last)
    return summed
