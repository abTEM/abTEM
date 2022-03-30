"""Module to describe the effect of temperature on the atomic positions."""
from abc import abstractmethod, ABCMeta
from copy import copy
from numbers import Number
from typing import Mapping, Union, Sequence, List, Iterable

import dask
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
import dask.array as da
from dask.graph_manipulation import clone
from dask.delayed import Delayed
from abtem.core.axes import FrozenPhononsAxis
from abtem.measure.detect import stack_measurements


def stack_frozen_phonons(measurements, detectors):
    measurements = list(map(list, zip(*measurements)))
    stacked = []
    for measurement, detector in zip(measurements, detectors):
        new = stack_measurements(measurement, axes_metadata=FrozenPhononsAxis())
        stacked.append(new)
    return stacked


class LazyAtoms:

    def __init__(self, atoms: Union[Atoms, Delayed], cell: np.ndarray = None, numbers: np.ndarray = None):
        self._atoms = atoms

        if cell is None:
            if self.is_lazy:
                raise RuntimeError('cannot infer cell from lazy atoms')
            cell = atoms.cell

        if numbers is None:
            if self.is_lazy:
                raise RuntimeError('cannot atomic numbers cell from lazy atoms')
            numbers = np.unique(atoms.numbers)

        self._cell = cell
        self._numbers = numbers

    @property
    def is_lazy(self):
        if hasattr(self.atoms, 'compute'):
            return True

        return False

    @property
    def numbers(self):
        return self._numbers

    @property
    def cell(self):
        return self._cell

    @property
    def atoms(self):
        return self._atoms

    def compute(self, **kwargs):
        if self.is_lazy:
            self._atoms = self.atoms.compute(**kwargs)

        return self._atoms

    def delay_atoms(self):
        if self.is_lazy:
            return self

        def delay_atoms(atoms):
            return atoms

        atoms = dask.delayed(delay_atoms)(self.atoms)
        return self.__class__(atoms.copy(), cell=copy(self.cell), numbers=copy(self.numbers))

    def apply_transformation(self, func, new_cell=None, new_numbers=None, **kwargs):
        if new_cell is None:
            new_cell = self._cell

        if new_numbers is None:
            new_numbers = self._numbers

        if self.is_lazy:
            atoms = dask.delayed(func)(atoms=self.atoms, **kwargs)
        else:
            atoms = func(atoms=self.atoms, **kwargs)
        return self.__class__(atoms, cell=new_cell, numbers=new_numbers)

    def clone(self):
        if not self.is_lazy:
            return self

        atoms = clone(self.atoms.copy(), assume_layers=False)
        return self.__class__(atoms, cell=copy(self.cell), numbers=copy(self.numbers))


class AbstractFrozenPhonons(metaclass=ABCMeta):
    """Abstract base class for frozen phonons objects."""

    def __init__(self, ensemble_mean: bool = True):
        self._ensemble_mean = ensemble_mean

    @abstractmethod
    def apply_transformation(self, func=None, *args, **kwargs):
        pass

    @property
    def axes_metadata(self):
        return FrozenPhononsAxis(ensemble_mean=self.ensemble_mean)

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def cell(self):
        pass

    @abstractmethod
    def get_configurations(self):
        pass

    def __getitem__(self, item):
        configurations = self.get_configurations()
        return configurations[item]

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


class DummyFrozenPhonons(AbstractFrozenPhonons):

    def __init__(self):
        super().__init__(ensemble_mean=True)

    @property
    def cell(self):
        return None

    def get_configurations(self, lazy: bool = True):
        return [None]

    def __len__(self):
        return 1

    def __copy__(self):
        return self.__class__()

    def apply_transformation(self, func=None, *args, **kwargs):
        return self


class FrozenPhonons(AbstractFrozenPhonons):
    """
    Frozen phonons object.

    Generates atomic configurations for thermal diffuse scattering.
    Randomly displaces the atomic positions of an ASE Atoms object to emulate thermal vibrations.

    Parameters
    ----------
    atoms: ASE Atoms object
        Atoms with the average atomic configuration.
    num_configs: int
        Number of frozen phonon configurations.
    sigmas: float or dict or list
        If float, the standard deviation of the displacements is assumed to be identical for all atoms.
        If dict, a displacement standard deviation should be provided for each species. The atomic species can be
        specified as atomic number or symbol.
        If list or array, a displacement standard deviation should be provided for each atom.
    directions: str
        The displacement directions of the atoms as a string; for example 'xy' for displacement in the x- and
        y-direction.
    ensemble_mean : True
    seed: int
        Seed for random number generator.
    """

    def __init__(self,
                 atoms: Union[Atoms, LazyAtoms],
                 num_configs: int,
                 sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
                 directions: str = 'xyz',
                 ensemble_mean: bool = True,
                 seed: int = None):

        self._unique_numbers = np.unique(atoms.numbers)
        unique_symbols = [chemical_symbols[number] for number in self._unique_numbers]

        if not hasattr(atoms, 'compute'):
            atoms = LazyAtoms(atoms)

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
        self._num_configs = num_configs
        self._seed = seed

        super().__init__(ensemble_mean)

    def apply_transformation(self, func, *args, **kwargs):
        atoms = self._atoms.apply_transformation(func, *args, **kwargs)
        return self.__class__(atoms, num_configs=len(self), sigmas=copy(self.sigmas), seed=self.seed,
                              directions=self.directions)

    def delay_atoms(self):
        atoms = self._atoms.delay_atoms()
        return self.__class__(atoms, num_configs=len(self), sigmas=copy(self.sigmas),
                              seed=self.seed, directions=self.directions)

    @property
    def is_lazy(self):
        return self._atoms.is_lazy

    @property
    def num_configs(self) -> int:
        return self._num_configs

    @property
    def seed(self) -> int:
        return self._seed

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
        return self._num_configs

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

    def get_random_state(self):
        if self.seed is not False and self.seed is not None:
            if self.is_lazy:
                random_state = dask.delayed(np.random.RandomState)(seed=self.seed)
            else:
                random_state = np.random.RandomState(seed=self.seed)
        else:
            random_state = None

        return random_state

    @staticmethod
    def _jiggle_atoms(atoms, sigmas, axes, random_state):

        if isinstance(sigmas, Mapping):
            temp = np.zeros(len(atoms.numbers), dtype=np.float32)
            for unique in np.unique(atoms.numbers):
                temp[atoms.numbers == unique] = np.float32(sigmas[chemical_symbols[unique]])
            sigmas = temp

        elif not isinstance(sigmas, np.ndarray):
            raise RuntimeError()

        atoms = atoms.copy()

        if random_state:
            r = random_state.normal(size=(len(atoms), 3))
        else:
            r = np.random.normal(size=(len(atoms), 3))

        for axis in axes:
            atoms.positions[:, axis] += sigmas * r[:, axis]

        return atoms

    def get_configurations(self):
        random_state = self.get_random_state()

        configurations = []
        for i in range(self.num_configs):
            configuration = self._atoms.clone()
            configuration = configuration.apply_transformation(self._jiggle_atoms,
                                                               sigmas=self.sigmas,
                                                               axes=self.axes,
                                                               random_state=random_state)

            configurations.append(configuration)

        return configurations

    def __copy__(self) -> 'FrozenPhonons':
        return self.__class__(atoms=self.atoms.copy(), num_configs=len(self), sigmas=copy(self.sigmas),
                              seed=self.seed, directions=self.directions)


class MDFrozenPhonons(AbstractFrozenPhonons):
    """
    Molecular dynamics frozen phonons object.

    Parameters
    ----------
    trajectory: List of ASE Atoms objects
        Sequence of Atoms objects representing a thermal distribution of atomic configurations.
    """

    def __init__(self, trajectory: Sequence[Atoms], ensemble_mean: bool = True):
        self._trajectory = trajectory

        super().__init__(ensemble_mean=ensemble_mean)

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def atoms(self) -> Atoms:
        return self[0]

    @property
    def cell(self) -> np.ndarray:
        return self[0].cell

    def __getitem__(self, item) -> Atoms:
        return self._trajectory[0]

    def standard_deviations(self) -> np.ndarray:
        mean_positions = np.mean([atoms.positions for atoms in self], axis=0)
        squared_deviations = [(atoms.positions - mean_positions) ** 2 for atoms in self]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))

    def get_frozen_phonon_atoms(self, lazy: bool = True):
        return self._trajectory

    def __copy__(self):
        return self.__class__(trajectory=[atoms.copy() for atoms in self._trajectory])
