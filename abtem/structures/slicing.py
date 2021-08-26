from numbers import Number
from typing import Union

import numpy as np

from abtem.structures.structures import pad_atoms


class SliceIndexedAtoms:

    def __init__(self, atoms, num_slices):
        self._atoms = atoms.copy()
        self._num_slices = num_slices
        self._unique_numbers = np.unique(self.atoms.numbers)
        self.atoms.wrap(pbc=True)

        positions = self.atoms.positions.astype(np.float32)
        order = np.argsort(positions[:, 2])
        positions = positions[order]

        self._positions = positions[:, :2]
        self._numbers = self.atoms.numbers[order]
        self._slice_idx = np.floor(positions[:, 2] / self.atoms.cell[2, 2] * self._num_slices).astype(np.int)

    @property
    def atoms(self):
        return self._atoms

    def get_atoms_in_slices(self, first_slice, last_slice):
        start_idx = np.searchsorted(self._slice_idx, first_slice)
        end_idx = np.searchsorted(self._slice_idx, last_slice)

        if start_idx == end_idx:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,)), np.zeros((0,))

        chunk_positions = self._positions[start_idx:end_idx]
        chunk_numbers = self._numbers[start_idx:end_idx]
        chunk_slice_idx = self._slice_idx[start_idx:end_idx] - first_slice
        return chunk_positions, chunk_numbers, chunk_slice_idx


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

    def get_atoms_in_slices(self,
                            start,
                            end=None,
                            atomic_number=None,
                            padding: Union[bool, float] = False,
                            z_margin=0.):

        if end is None:
            end = start + 1

        a = self.get_slice_entrance(start) - z_margin
        b = self.get_slice_entrance(end) + z_margin

        in_slice = (self.atoms.positions[:, 2] > a) * (self.atoms.positions[:, 2] < b)

        if atomic_number is not None:
            in_slice = (self.atoms.numbers == atomic_number) * in_slice

        atoms = self.atoms[in_slice]

        if padding:
            atoms = pad_atoms(atoms, padding)

        return atoms

    @property
    def num_slices(self):
        """The number of projected potential slices."""
        return len(self._slice_thicknesses)

    def get_slice_thickness(self, i):
        """The thickness of the projected potential slices."""
        return self._slice_thicknesses[i]
