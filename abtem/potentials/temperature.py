"""Module to describe the effect of temperature on the atomic positions."""
import inspect
import itertools
from abc import abstractmethod, ABCMeta
from copy import copy
from functools import partial
from numbers import Number
from typing import Mapping, Union, Sequence, List, Iterable, Tuple

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.core.axes import FrozenPhononsAxis, AxisMetadata
from abtem.core.ensemble import Ensemble
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.utils import CopyMixin, EqualityMixin
from dask.delayed import Delayed
from ase import data


class AbstractFrozenPhonons(Ensemble, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """Abstract base class for frozen phonons objects."""

    def __init__(self,
                 atomic_numbers,
                 cell,
                 ensemble_mean: bool = True):
        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._ensemble_mean = ensemble_mean

    @property
    def is_lazy(self):
        return isinstance(self.atoms, Delayed)

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers

    @property
    def numbers(self) -> np.ndarray:
        return self._atomic_numbers

    @property
    def cell(self) -> Cell:
        return self._cell

    def _validate_atomic_numbers_and_cell(self, atoms, atomic_numbers, cell):
        if isinstance(atoms, Delayed) and (atomic_numbers is None or cell is None):
            raise ValueError()

        if cell is None:
            cell = atoms.cell.copy()

        if atomic_numbers is None:
            atomic_numbers = np.unique(atoms.numbers)
        else:
            atomic_numbers = np.array(atomic_numbers, dtype=int)

        return atomic_numbers, cell

    @property
    @abstractmethod
    def atoms(self) -> Union[Atoms, Delayed]:
        pass

    @abstractmethod
    def randomize(self, atoms: Atoms) -> Atoms:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class DummyFrozenPhonons(AbstractFrozenPhonons):

    def __init__(self,
                 atoms: Union[Atoms, Delayed],
                 num_configs: int = None,
                 atomic_numbers: Union[np.ndarray, Sequence[int]] = None,
                 cell: Union[Cell, np.ndarray] = None):

        self._atoms = atoms
        self._num_configs = num_configs
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, atomic_numbers, cell)
        super().__init__(atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=True)

    @property
    def num_configs(self):
        return self._num_configs

    @property
    def ensemble_shape(self):
        if self._num_configs is None:
            return ()
        else:
            return self._num_configs,

    @property
    def default_ensemble_chunks(self):
        if self._num_configs is None:
            return ()
        else:
            return 1,

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if self._num_configs is None:
            return []
        else:
            return [FrozenPhononsAxis(values=tuple(range(len(self))), _ensemble_mean=self.ensemble_mean)]

    def randomize(self, atoms):
        return atoms

    @property
    def atoms(self):
        return self._atoms

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        return cls(args[0])

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('atoms',))
        return partial(self._from_partitioned_args_func, **kwargs)

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        def _dummy_frozen_phonons(atoms):
            arr = np.zeros((1,), dtype=object)
            arr.itemset(0, atoms)
            return arr

        if not lazy and self.is_lazy:
            atoms = self.atoms.compute()
        else:
            atoms = self.atoms

        if lazy:
            array = da.from_delayed(dask.delayed(_dummy_frozen_phonons)(atoms), shape=(1,), dtype=object)
        else:
            array = _dummy_frozen_phonons(atoms)

        return array,

    def __len__(self):
        if self._num_configs is None:
            return 1
        else:
            return self._num_configs


def validate_seeds(seeds: Union[int, Tuple[int, ...]], num_seeds: int = None) -> Tuple[int, ...]:
    if seeds is None or np.isscalar(seeds):
        if num_seeds is None:
            raise ValueError('provide `num_configs` or a seed for each configuration')

        rng = np.random.default_rng(seed=seeds)
        seeds = ()
        while len(seeds) < num_seeds:
            seed = rng.integers(np.iinfo(np.int32).max)
            if seed not in seeds:
                seeds += (seed,)
    else:
        if not hasattr(seeds, '__len__'):
            raise ValueError

        if num_seeds is not None:
            assert num_seeds == len(seeds)

    return seeds


class FrozenPhonons(AbstractFrozenPhonons):
    """
    The frozen phonons randomly displaces the atomic positions of an ASE Atoms object to emulate thermal vibrations.

    Parameters
    ----------
    atoms: ASE Atoms object
        Atoms with the average atomic configuration.
    num_configs: int
        Number of frozen phonon configurations.
    sigmas: float or dict or list
        If float, the standard deviation of the displacements is assumed to be identical for all atoms.
        If dict, a displacement standard deviation should be provided for each species. The atomic species can be
        specified as atomic number or a symbol, using the ASE standard.
        If list or array, a displacement standard deviation should be provided for each atom.
    directions: str, optional
        The displacement directions of the atoms as a string; for example 'xy' for displacement in the x- and
        y-direction. Default is 'xy'.
    ensemble_mean : True, optional
        If True, the mean of the ensemble of results from a multislice simulation is calculated, otherwise, the result
        of every frozen phonon is returned.
    seeds: int or sequence of int
        Seed for the random number generator(rng), or one seed for each rng in the frozen phonon ensemble.
    """

    def __init__(self,
                 atoms: Atoms,
                 num_configs: int,
                 sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
                 directions: str = 'xyz',
                 ensemble_mean: bool = True,
                 seeds: Union[int, Tuple[int, ...]] = None):

        if isinstance(sigmas, dict):
            atomic_numbers = [data.atomic_numbers[symbol] for symbol in sigmas.keys()]
        else:
            atomic_numbers = None

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, atomic_numbers, cell=None)

        unique_symbols = [chemical_symbols[number] for number in atomic_numbers]

        self._atoms = atoms

        if isinstance(sigmas, Number):
            new_sigmas = {}
            for symbol in unique_symbols:
                new_sigmas[symbol] = sigmas

            sigmas = new_sigmas

        elif isinstance(sigmas, dict):
            if not all([symbol in unique_symbols for symbol in sigmas.keys()]):
                raise RuntimeError('displacement standard deviation must be provided for all atomic species')

        elif isinstance(sigmas, Iterable):
            sigmas = np.array(sigmas, dtype=np.float32)
            if len(sigmas) != len(atoms):
                raise RuntimeError('displacement standard deviation must be provided for all atoms')
        else:
            raise ValueError()

        self._sigmas = sigmas
        self._directions = directions

        self._seeds = validate_seeds(seeds, num_seeds=num_configs)

        super().__init__(atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean)

    @property
    def ensemble_shape(self):
        return len(self),

    @property
    def default_ensemble_chunks(self):
        return 1,

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return [FrozenPhononsAxis(values=tuple(range(len(self))), _ensemble_mean=self.ensemble_mean)]

    @property
    def num_configs(self) -> int:
        return len(self._seeds)

    @property
    def seeds(self) -> Tuple[int, ...]:
        return self._seeds

    @property
    def sigmas(self) -> Union[Mapping[Union[str, int], float], np.ndarray]:
        return self._sigmas

    @property
    def atoms(self) -> Union[Atoms, Delayed]:
        return self._atoms

    @property
    def directions(self) -> str:
        return self._directions

    def __len__(self) -> int:
        return self.num_configs

    @property
    def axes(self) -> List[int]:
        axes = []
        for direction in list(set(self._directions.lower())):
            if direction == 'x':
                axes += [0]
            elif direction == 'y':
                axes += [1]
            elif direction == 'z':
                axes += [2]
            else:
                raise RuntimeError('Directions must be "x", "y" or "z" not {}.')
        return axes

    def randomize(self, atoms: Atoms) -> Atoms:
        sigmas = self.sigmas

        if isinstance(sigmas, Mapping):
            temp = np.zeros(len(atoms.numbers), dtype=np.float32)
            for unique in np.unique(atoms.numbers):
                temp[atoms.numbers == unique] = np.float32(sigmas[chemical_symbols[unique]])
            sigmas = temp

        elif not isinstance(sigmas, np.ndarray):
            raise RuntimeError()

        atoms = atoms.copy()

        rng = np.random.default_rng(self.seeds[0])
        r = rng.normal(size=(len(atoms), 3))

        for axis in self.axes:
            atoms.positions[:, axis] += sigmas * r[:, axis]

        return atoms

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = args[0]
        if hasattr(args, 'item'):
            args = args.item()

        kwargs['atoms'] = args['atoms']
        kwargs['seeds'] = args['seeds']
        kwargs['num_configs'] = args['num_configs']
        return cls(**kwargs)

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('atoms', 'seeds', 'num_configs'))
        return partial(self._from_partitioned_args_func, **kwargs)

    @staticmethod
    def _frozen_phonons_args(atoms, seeds):
        arr = np.zeros((1,), dtype=object)
        arr.itemset(0, {'atoms': atoms, 'seeds': seeds, 'num_configs': len(seeds)})
        return arr

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        if not lazy and self.is_lazy:
            atoms = self.atoms.compute()
        else:
            atoms = self.atoms

        array = np.zeros((len(chunks[0]),), dtype=object)
        for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
            seeds = self.seeds[start:stop]

            if lazy:
                lazy_atoms = dask.delayed(atoms)
                lazy_frozen_phonon = dask.delayed(self._frozen_phonons_args)(atoms=lazy_atoms, seeds=seeds)
                array.itemset(i, da.from_delayed(lazy_frozen_phonon, shape=(1,), dtype=object))
            else:
                array.itemset(i, self._frozen_phonons_args(atoms=atoms, seeds=seeds))

        if lazy:
            array = da.concatenate(list(array))

        return array,

    def to_md_frozen_phonons(self):
        trajectory = []
        for b in self.generate_blocks(1):
            trajectory.append(b[-1].randomize(b[-1].atoms))
        return MDFrozenPhonons(trajectory)


class MDFrozenPhonons(AbstractFrozenPhonons):
    """
    Molecular dynamics frozen phonons.

    Parameters
    ----------
    trajectory: List of ASE Atoms
        Sequence of Atoms objects representing a thermal distribution of atomic configurations.
    ensemble_mean : True, optional
        If True, the mean of the ensemble of results from a multislice simulation is calculated, otherwise, the result
        of every frozen phonon is returned.
    """

    def __init__(self,
                 trajectory: Sequence[Atoms],
                 atomic_numbers=None,
                 cell=None,
                 ensemble_mean: bool = True):

        if isinstance(trajectory, Atoms):
            trajectory = [trajectory]

        self._trajectory = trajectory

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(trajectory[0], atomic_numbers, cell)

        super().__init__(atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean)

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return [FrozenPhononsAxis(values=tuple(range(len(self))), _ensemble_mean=self.ensemble_mean)]

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def atoms(self) -> Atoms:
        return self._trajectory[0]

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return len(self),

    @property
    def default_ensemble_chunks(self) -> Tuple[int, ...]:
        return 1,

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        def md_frozen_phonons(atoms):
            arr = np.zeros((1,), dtype=object)
            arr.itemset(0, atoms)
            return arr

        array = np.zeros((len(chunks[0]),), dtype=object)
        start = 0
        for i, chunk in enumerate(chunks[0]):
            stop = start + chunk
            trajectory = self._trajectory[start:stop]

            if lazy:
                atoms = dask.delayed(lambda *args: list(args))(*trajectory)

                delayed_frozen_phonon = dask.delayed(md_frozen_phonons)(atoms=atoms)

                array.itemset(i, da.from_delayed(delayed_frozen_phonon, shape=(1,), dtype=object))
            else:
                trajectory = [atoms.compute() if hasattr(atoms, 'compute') else atoms for atoms in trajectory]
                array.itemset(i, trajectory)

            start = stop

        if lazy:
            array = da.concatenate(list(array))

        return array,

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('trajectory',))
        return partial(MDFrozenPhonons, **kwargs)

    def randomize(self, atoms):
        return atoms

    # def __getitem__(self, item) -> Atoms:
    #    atoms = self._trajectory[item]
    #    return atoms

    def standard_deviations(self) -> np.ndarray:
        mean_positions = np.mean([atoms.positions for atoms in self], axis=0)
        squared_deviations = [(atoms.positions - mean_positions) ** 2 for atoms in self]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))

    def __copy__(self):
        return self.__class__(trajectory=[atoms.copy() for atoms in self._trajectory])
