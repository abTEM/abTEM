from numbers import Number
from typing import List, Tuple

import dask.array as da
import dask.delayed
import numpy as np
from ase.data import atomic_numbers

from abtem.structures.structures import cut_box


class SliceIndexedAtoms:

    def __init__(self, atoms, num_slices):
        order = np.argsort(atoms.positions[:, 2])
        self._num_slices = num_slices

        self._positions = atoms.positions[order][:, :2]
        self._numbers = atoms.numbers[order]
        self._slice_idx = np.floor(atoms.positions[order][:, 2] / atoms.cell[2, 2] * num_slices).astype(np.int)

    def __len__(self):
        return self._num_slices

    def get_atoms_in_slices(self, first_slice, last_slice):
        def _get_atoms_in_slice(slice_idx, first_slice, last_slice, positions, numbers):
            start_idx = np.searchsorted(slice_idx, first_slice)
            end_idx = np.searchsorted(slice_idx, last_slice)

            if start_idx == end_idx:
                return np.zeros((0, 2), dtype=np.float32), np.zeros((0,)), np.zeros((0,))

            chunk_positions = positions[start_idx:end_idx]
            chunk_numbers = numbers[start_idx:end_idx]
            chunk_slice_idx = slice_idx[start_idx:end_idx] - first_slice
            return chunk_positions, chunk_numbers, chunk_slice_idx

        return _get_atoms_in_slice(self._slice_idx, first_slice, last_slice, self._positions, self._numbers)

        # positions, numbers, slice_idx = dask.delayed(_get_atoms_in_slice, nout=3)(self._slice_idx,
        #                                                                           first_slice,
        #                                                                           last_slice,
        #                                                                           self._atoms.positions,
        #                                                                           self._atoms.numbers)
        #
        # return positions, numbers, slice_idx


def _validate_slice_thickness(slice_thickness, thickness=None, num_slices=None):
    if isinstance(slice_thickness, Number):
        if thickness is not None:
            n = np.ceil(thickness / slice_thickness)
            slice_thickness = np.full(int(n), thickness / n)
        elif num_slices is not None:
            slice_thickness = [slice_thickness] * num_slices
        else:
            raise RuntimeError()

    slice_thickness = np.array(slice_thickness)

    if thickness is not None:
        if not np.isclose(np.sum(slice_thickness), thickness):
            raise RuntimeError()

    if num_slices is not None:
        if len(slice_thickness) != num_slices:
            raise RuntimeError()

    return slice_thickness


def _slice_limits(slice_thickness):
    cumulative_thickness = np.cumsum(np.concatenate(((0,), slice_thickness)))
    return [(cumulative_thickness[i], cumulative_thickness[i + 1]) for i in range(len(cumulative_thickness) - 1)]


class SlicedAtoms:

    def __init__(self, atoms, slice_thickness, plane='xy', box=None, origin=(0., 0., 0.), padding=0.):

        if box is None:
            box = np.diag(atoms.cell)

        if isinstance(padding, dict):
            new_padding = {}
            for key, value in padding.items():
                if isinstance(key, str):
                    key = atomic_numbers[key]
                new_padding[key] = value
            padding = new_padding
            max_padding = max(padding.values())
        elif isinstance(padding, Number):
            max_padding = padding
        else:
            raise ValueError()

        atoms = cut_box(atoms, box, plane, origin=origin, margin=max_padding)

        self._padding = padding
        self._atoms = atoms
        self._slice_thickness = _validate_slice_thickness(slice_thickness, box[2])
        self._slice_limits = _slice_limits(self._slice_thickness)
        self._padding = padding

    @property
    def atoms(self):
        return self._atoms

    @property
    def num_slices(self):
        """The number of projected potential slices."""
        return len(self._slice_thickness)

    def __len__(self):
        return self.num_slices

    @property
    def slice_thickness(self):
        return self._slice_thickness

    @property
    def slice_limits(self) -> List[Tuple[float, float]]:
        return self._slice_limits

    # def flip(self):
    #     self._atoms.positions[:] = self._atoms.cell[2, 2] - self._atoms.positions[:]
    #     self._slice_thicknesses[:] = self._slice_thicknesses[::-1]

    def get_atoms_in_slices(self, start, end=None, atomic_number=None):

        if isinstance(atomic_number, str):
            atomic_number = atomic_numbers[atomic_number]

        if end is None:
            end = start

        if isinstance(self._padding, dict):
            if atomic_number is None:
                padding = max(self._padding.values())
            else:
                padding = self._padding[atomic_number]
        elif isinstance(self._padding, Number):
            padding = self._padding
        else:
            raise RuntimeError()

        a = self.slice_limits[start][0]
        b = self.slice_limits[end][1]

        in_slice = (self.atoms.positions[:, 2] >= (a - padding)) * (self.atoms.positions[:, 2] < (b + padding))

        if atomic_number is not None:
            in_slice = (self.atoms.numbers == atomic_number) * in_slice

        atoms = self.atoms[in_slice]
        atoms.cell = tuple(np.diag(atoms.cell)[:2]) + (b - a,)
        return atoms
