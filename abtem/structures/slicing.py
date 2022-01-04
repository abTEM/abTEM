from numbers import Number
from typing import List, Tuple, Union, Sequence

import numpy as np
from ase.data import atomic_numbers

from abtem.structures.structures import cut_box
from ase import Atoms
from abc import abstractmethod


def _validate_slice_thickness(slice_thickness: Union[float, np.ndarray, Sequence[float]],
                              thickness: float = None,
                              num_slices: int = None) -> np.ndarray:
    if np.isscalar(slice_thickness):
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


def unpack_item(item, num_items):
    if isinstance(item, int):
        first_index = item
        last_index = first_index + 1
    elif isinstance(item, slice):
        first_index = item.start
        last_index = item.stop
    else:
        raise RuntimeError()

    if last_index is None:
        last_index = num_items
    else:
        last_index = min(last_index, num_items)

    if first_index >= last_index:
        raise IndexError

    return first_index, last_index


class AbstractSlicedAtoms:

    def __init__(self, atoms: Atoms, slice_thickness: Union[float, np.ndarray, str], reverse: bool = False):
        self._atoms = atoms

        if isinstance(slice_thickness, str):
            raise NotImplementedError

        self._slice_thickness = _validate_slice_thickness(slice_thickness, thickness=atoms.cell[2, 2])
        self._reverse = reverse

    def __len__(self):
        return self.num_slices

    @property
    def num_slices(self):
        return len(self._slice_thickness)

    @property
    def reverse(self):
        return self._reverse

    @property
    def slice_thickness(self):
        return self._slice_thickness

    @property
    def slice_limits(self):
        return _slice_limits(self.slice_thickness)

    def check_slice_idx(self, i):
        """Raises an error if i is greater than the number of slices."""
        if i >= self.num_slices:
            raise RuntimeError('Slice index {} too large for sliced atoms with {} slices'.format(i, self.num_slices))

    @abstractmethod
    def _get_atoms_in_slices(self, first_slice, last_slice):
        pass

    def __getitem__(self, item):
        return self._get_atoms_in_slices(*unpack_item(item, len(self)))


class SliceIndexedAtoms(AbstractSlicedAtoms):

    def __init__(self,
                 atoms: Atoms,
                 slice_thickness: Union[float, np.ndarray, str],
                 slice_index=None,
                 reverse: bool = False):


        super().__init__(atoms, slice_thickness, reverse)

        self._slice_index = np.digitize(self.atoms.positions[:, 2], np.cumsum(self.slice_thickness))

    @property
    def atoms(self):
        return self._atoms

    @property
    def slice_index(self):
        return self._slice_index

    def _get_atoms_in_slices(self, first_slice: int, last_slice: int):
        if last_slice - first_slice == 1:
            is_in_slices = self.slice_index == first_slice
        else:
            is_in_slices = (self.slice_index >= first_slice) * (self.slice_index < last_slice)

        atoms = self.atoms[is_in_slices]
        slice_thickness = self.slice_thickness[first_slice:last_slice]
        atoms.cell[2, 2] = np.sum(slice_thickness)

        atoms.positions[:, 2] -= np.sum(self.slice_thickness[:first_slice])

        if last_slice - first_slice > 1:
            slice_index = self.slice_index[is_in_slices]
            raise NotImplementedError
            return self.__class__
        else:
            return atoms


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

    @slice_thickness.setter
    def slice_thickness(self, value):
        self._slice_thickness = value
        self._slice_limits = _slice_limits(self._slice_thickness)

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
