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

    # Snap values at z ≈ cell_z back to 0 before deduplicating, so planes near
    # the periodic boundary merge with planes near 0 instead of being treated
    # as distinct. Identify unique planes by sorted-neighbor distance (as in
    # commensurate_gpts) rather than rounding to a fixed tolerance-wide grid
    # anchored at 0: fixed-grid rounding can split two atoms much closer
    # together than `tolerance` into different bins when they straddle a bin
    # edge, or merge genuinely distinct planes that land in the same bin.
    z[z > cell_z - tolerance] = 0.0
    z = np.sort(z)
    unique_mask = np.concatenate([[True], np.diff(z) > tolerance])
    unique_z = z[unique_mask]
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
        # z is periodic, so the last slice and the first slice are also
        # adjacent through the wrap at cell_z ≡ 0. Without this, a trailing
        # remainder that is real (spacing ≥ tolerance) but well below half
        # the target thickness would end up as its own near-degenerate final
        # slice, since nothing follows it to trigger the merge guard above.
        if merged and acc < target_thickness * 0.5:
            merged[-1] += acc
        else:
            merged.append(acc)

    result = tuple(float(t) for t in merged)
    assert np.isclose(sum(result), cell_z)
    return result


def commensurate_gpts(
    extent: tuple[float, float],
    positions: np.ndarray,
    target_sampling: float = 0.05,
    tolerance: float = 1e-3,
    round_to_fast_fft: bool = True,
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

    When `round_to_fast_fft` is enabled the number of grid points additionally
    factorizes completely into the primes 2, 3, 5 and 7 whenever that is
    compatible with commensurability, so FFTs run on fast radix kernels instead
    of the slow, memory-hungry Bluestein fallback.  The smallest such grid not
    coarser than the target sampling is chosen; if the grid period p itself
    contains a prime factor larger than 7 no multiple of it can be a fast FFT
    length, and commensurability takes precedence.

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
    round_to_fast_fft : bool
        If True (default), prefer grids that are also fast FFT sizes (all prime
        factors in {2, 3, 5, 7}), at the cost of a sampling slightly finer than
        commensurability alone would give.

    Returns
    -------
    tuple of int
        Number of grid points in x and y.
    """
    from abtem.core.fft import is_fast_fft_size, next_fast_fft_size

    def translational_period_count(unique_x: np.ndarray, L: float) -> int:
        # Largest m such that the set of atom planes is invariant under a shift
        # of L / m. Even when the intra-cell plane positions are irrational
        # (no commensurate grid exists), a repeated unit cell still imposes
        # this translational period, and symmetry-equivalent atoms only see
        # identical discretised potentials if gpts is a multiple of m.
        for m in range(len(unique_x), 1, -1):
            shifted = np.sort((unique_x + L / m) % L)
            idx = np.searchsorted(unique_x, shifted)
            neighbors = np.stack(
                [
                    unique_x[np.clip(idx - 1, 0, len(unique_x) - 1)],
                    unique_x[np.clip(idx, 0, len(unique_x) - 1)],
                ]
            )
            distances = np.abs(neighbors - shifted)
            distances = np.minimum(distances, L - distances)
            if np.all(distances.min(axis=0) < tolerance):
                return m
        return 1

    def fallback_gpts(n_target: int, unique_x=None, L=None) -> int:
        # The GCD search found no commensurate period. When fast-FFT rounding
        # is enabled, pick the fast size that (a) keeps gpts a multiple of the
        # translational period count m of the plane set and (b) best aligns
        # the planes with grid points: structures with an irrational internal
        # parameter (e.g. rutile u = 0.306, brookite) have no exactly
        # commensurate grid, but some fast sizes come much closer than others.
        if not round_to_fast_fft:
            return n_target
        if unique_x is None or L is None or len(unique_x) <= 1:
            return next_fast_fft_size(n_target)

        m = translational_period_count(unique_x, L)
        if not is_fast_fft_size(m):
            # No multiple of m can be a fast FFT size; translational
            # commensurability takes precedence, as in the main path.
            return max(round(n_target / m), 1) * m

        best = None
        multiplier = -(-n_target // m)
        while True:
            multiplier = next_fast_fft_size(multiplier)
            n = multiplier * m
            # Score by ALIGNABILITY, not realized alignment: the smallest
            # achievable max plane-to-gridpoint distance over all grid-origin
            # phases. The plane residues x_i*n/L mod 1 must cluster near a
            # common phase for the planes to sit near grid points; the tightest
            # arc containing all residues is 1 - (largest circular gap), and
            # centering the phase in that arc gives max distance (1 - gap)/2.
            # Unlike distance-to-nearest-gridpoint at a fixed origin, this
            # depends only on RELATIVE positions, so a rigidly translated
            # structure gets the same grid (a tested invariant of the
            # commensurate search that the fallback must preserve).
            residues = np.sort(np.mod(unique_x * (n / L), 1.0))
            gaps = np.diff(np.concatenate([residues, [residues[0] + 1.0]]))
            alignability = (1.0 - float(np.max(gaps))) / 2.0
            # Round the score so nearly-equal alignments prefer the smaller grid.
            score = (round(alignability, 2), n)
            if best is None or score < best[0]:
                best = (score, n)
            if n > n_target * 1.12:
                return best[1]
            multiplier += 1

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
            gpts.append(fallback_gpts(n_target, unique_x, L))
            continue

        # Spacings between consecutive unique planes, including the wrap-around gap
        spacings = np.concatenate([np.diff(unique_x), [L + unique_x[0] - unique_x[-1]]])

        min_spacing = float(np.min(spacings))
        if min_spacing < tolerance:
            gpts.append(fallback_gpts(n_target, unique_x, L))
            continue

        # Check that every spacing is approximately an integer multiple of the
        # minimum spacing (i.e. the minimum is the primitive spacing).
        ratios = spacings / min_spacing
        k = np.round(ratios).astype(int)
        if np.any(np.abs(ratios - k) > 0.05) or np.any(k < 1):
            # No clear commensurability — fall back to the raw target
            gpts.append(fallback_gpts(n_target, unique_x, L))
            continue

        # Primitive spacing refined as L / (total number of primitive periods).
        # Using L / sum(k) is more accurate than min_spacing alone because it
        # averages out any floating-point scatter in the individual spacings.
        n_periods = int(np.sum(k))
        if round_to_fast_fft and is_fast_fft_size(n_periods):
            # A multiple m * n_periods is a fast FFT size exactly when both
            # factors are, so the smallest fast commensurate grid not coarser
            # than the target uses the next fast multiplier. When n_periods
            # itself has a prime factor larger than 7, no multiple can be fast
            # and commensurability takes precedence (handled below).
            n_multiple = next_fast_fft_size(-(-n_target // n_periods)) * n_periods
        else:
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
