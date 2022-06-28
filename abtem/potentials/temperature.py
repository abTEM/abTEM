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


class AbstractFrozenPhonons(Ensemble, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """Abstract base class for frozen phonons objects."""

    def __init__(self, ensemble_mean: bool = True):
        self._ensemble_mean = ensemble_mean

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    @abstractmethod
    def atoms(self) -> Atoms:
        pass

    @property
    def numbers(self) -> np.ndarray:
        return self.atoms.numbers

    @abstractmethod
    def randomize(self, atoms: Atoms) -> Atoms:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def cell(self) -> Cell:
        pass


class DummyFrozenPhonons(AbstractFrozenPhonons):

    def __init__(self, atoms):
        self._atoms = atoms
        super().__init__(ensemble_mean=True)

    @property
    def ensemble_shape(self):
        return ()

    @property
    def default_ensemble_chunks(self):
        return ()

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return []

    def randomize(self, atoms):
        return atoms

    @property
    def atoms(self):
        return self._atoms

    @property
    def cell(self):
        return self._atoms.cell

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        return cls(args[0])

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('atoms',))
        return partial(self._from_partitioned_args_func, **kwargs)

    def __iter__(self):
        return self.generate_blocks()

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        # chunks = validate_chunks(self.ensemble_shape, chunks)
        atoms = self.atoms

        array = np.zeros((1,), dtype=object)
        array.itemset(0, atoms)

        # array = np.zeros(len(chunks[0]), dtype=object)
        # for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
        #     seeds = self.seeds[start:stop]
        #     block = {'atoms': atoms, 'seeds': seeds}
        #     array[i] = block

        if lazy:
            array = da.from_array(array, chunks=1)

        return array,

    def __len__(self):
        return 1


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
                 sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
                 num_configs: int = None,
                 directions: str = 'xyz',
                 ensemble_mean: bool = True,
                 seeds: Union[int, Sequence[int]] = None):

        self._unique_numbers = np.unique(atoms.numbers)
        unique_symbols = [chemical_symbols[number] for number in self._unique_numbers]
        self._atoms = atoms

        if isinstance(sigmas, Number):
            new_sigmas = {}
            for symbol in unique_symbols:
                new_sigmas[symbol] = sigmas

            sigmas = new_sigmas

        elif isinstance(sigmas, dict):
            if not all([symbol in unique_symbols for symbol in sigmas.keys()]):
                raise RuntimeError('Displacement standard deviation must be provided for all atomic species.')

        elif isinstance(sigmas, Iterable):
            sigmas = np.array(sigmas, dtype=np.float32)
            if len(sigmas) != len(atoms):
                raise RuntimeError('Displacement standard deviation must be provided for all atoms.')
        else:
            raise ValueError()

        self._sigmas = sigmas
        self._directions = directions

        # elif hasattr(seeds, '__len__') and num_configs is not None:
        #    if not len(seeds) == num_configs:
        #        raise RuntimeError()
        if seeds is None or np.isscalar(seeds):
            rng = np.random.default_rng(seed=seeds)
            seeds = ()
            while len(seeds) < num_configs:
                seed = rng.integers(np.iinfo(np.int32).max)
                if seed not in seeds:
                    seeds += (seed,)
        else:
            if not hasattr(seeds, '__len__'):
                raise ValueError

            if num_configs is not None:
                assert num_configs == len(seeds)

        self._seeds = seeds

        super().__init__(ensemble_mean)

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
    def seeds(self) -> Tuple[int]:
        return self._seeds

    @property
    def sigmas(self) -> Union[Mapping[Union[str, int], float], np.ndarray]:
        return self._sigmas

    @property
    def cell(self) -> np.ndarray:
        return self._atoms.cell

    @property
    def atoms(self) -> Atoms:
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
        kwargs['atoms'] = args[0]['atoms']
        kwargs['seeds'] = args[0]['seeds']
        return cls(**kwargs)

    @classmethod
    def merge_blocks(cls, blocks):
        kwargs = blocks[0].copy_kwargs(exclude=('seeds', 'num_configs'))
        kwargs['seeds'] = tuple(itertools.chain(*(block.seeds for block in blocks)))
        return cls(**kwargs)

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('atoms', 'seeds', 'num_configs'))
        return partial(self._from_partitioned_args_func, **kwargs)

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        atoms = self.atoms

        array = np.zeros(len(chunks[0]), dtype=object)
        for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
            seeds = self.seeds[start:stop]
            block = {'atoms': atoms, 'seeds': seeds}
            array[i] = block

        if lazy:
            array = da.from_array(array, chunks=1)

        return array,


class LazyAtoms:

    def __init__(self, atoms, numbers, cell):
        self._atoms = atoms
        self._numbers = numbers
        self._cell = cell

    @property
    def atoms(self):
        return self._atoms

    @property
    def cell(self):
        return self._cell

    @property
    def numbers(self):
        return self._numbers


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

    def __init__(self, trajectory: Sequence[Atoms], ensemble_mean: bool = True):

        if isinstance(trajectory, Atoms):
            trajectory = [trajectory]

        self._trajectory = trajectory

        super().__init__(ensemble_mean=ensemble_mean)

    def generate_configurations(self):
        for frozen_phonon in self:
            yield MDFrozenPhonons([frozen_phonon])

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return [FrozenPhononsAxis(values=tuple(range(len(self))), _ensemble_mean=self.ensemble_mean)]

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def atoms(self) -> Atoms:
        return self[0]

    @property
    def cell(self) -> np.ndarray:
        return self[0].cell

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return len(self),

    @property
    def default_ensemble_chunks(self) -> Tuple[int, ...]:
        return 1,

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        def md_frozen_phonons(atoms, **kwargs):
            arr = np.empty((1,), dtype=object)
            arr[0] = MDFrozenPhonons(atoms, **kwargs)
            return arr

        array = []
        start = 0
        for chunk in chunks[0]:
            stop = start + chunk

            atoms = self.atoms[start:stop]

            delayed_frozen_phonon = dask.delayed(md_frozen_phonons)(atoms=atoms,
                                                                    ensemble_mean=self.ensemble_mean,
                                                                    )

            array.append(da.from_delayed(delayed_frozen_phonon, shape=(1,), dtype=object))
            start = stop

        return da.concatenate(array),

    def from_partitioned_args(self):
        return lambda x: x

    def randomize(self, atoms):
        return atoms

    def __getitem__(self, item) -> Atoms:
        atoms = self._trajectory[item]
        return atoms

    def standard_deviations(self) -> np.ndarray:
        mean_positions = np.mean([atoms.positions for atoms in self], axis=0)
        squared_deviations = [(atoms.positions - mean_positions) ** 2 for atoms in self]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))

    def __copy__(self):
        return self.__class__(trajectory=[atoms.copy() for atoms in self._trajectory])
