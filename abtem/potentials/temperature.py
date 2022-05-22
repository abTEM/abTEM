"""Module to describe the effect of temperature on the atomic positions."""
from abc import abstractmethod, ABCMeta
from copy import copy
from numbers import Number
from typing import Mapping, Union, Sequence, List, Iterable

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

from abtem.core.axes import FrozenPhononsAxis
from abtem.core.blockwise import Ensemble
from abtem.core.dask import validate_chunks


class AbstractFrozenPhonons(Ensemble, metaclass=ABCMeta):
    """Abstract base class for frozen phonons objects."""

    def __init__(self, ensemble_mean: bool = True):
        self._ensemble_mean = ensemble_mean

    @abstractmethod
    def generate_configurations(self):
        pass

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    @abstractmethod
    def atoms(self):
        pass

    @abstractmethod
    def randomize(self, atoms):
        pass

    @property
    def ensemble_shape(self):
        return len(self),

    @property
    def default_ensemble_chunks(self):
        return 1,

    @property
    def ensemble_axes_metadata(self):
        return [FrozenPhononsAxis(_ensemble_mean=self.ensemble_mean)]

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def cell(self):
        pass

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


class DummyFrozenPhonons(AbstractFrozenPhonons):

    def __init__(self):
        super().__init__(ensemble_mean=True)

    def ensemble_blocks(self, chunks):
        raise NotImplementedError

    def ensemble_partial(self):
        raise NotImplementedError

    def generate_configurations(self):
        raise NotImplementedError

    def randomize(self, atoms):
        raise NotImplementedError

    @property
    def atoms(self):
        return None

    @property
    def cell(self):
        return None

    def __len__(self):
        return 1

    def __copy__(self):
        return self.__class__()


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
                 atoms: Atoms,
                 num_configs: int,
                 sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
                 directions: str = 'xyz',
                 ensemble_mean: bool = True,
                 random_state=None):

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
        self._num_configs = num_configs
        self._random_state = random_state

        super().__init__(ensemble_mean)

    @property
    def num_configs(self) -> int:
        return self._num_configs

    @property
    def random_state(self):
        return self._random_state

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

        if self.random_state:
            r = self.random_state.normal(size=(len(atoms), 3))
        else:
            r = np.random.normal(size=(len(atoms), 3))

        for axis in self.axes:
            atoms.positions[:, axis] += sigmas * r[:, axis]

        return atoms

    def generate_configurations(self):
        kwargs = self._copy_as_dict()
        kwargs['num_configs'] = 1

        for i in range(self.num_configs):
            yield FrozenPhonons(**kwargs)

    def ensemble_partial(self):
        return lambda x: x

    def ensemble_blocks(self, chunks: int = 1):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        random_state = dask.delayed(self.random_state)
        atoms = dask.delayed(self.atoms)

        def frozen_phonons(**kwargs):
            arr = np.empty((1,), dtype=object)
            arr[0] = FrozenPhonons(**kwargs)
            return arr

        array = []
        for chunk in chunks[0]:
            delayed_frozen_phonon = dask.delayed(frozen_phonons)(atoms=atoms,
                                                                 sigmas=self.sigmas,
                                                                 num_configs=chunk,
                                                                 directions=self.directions,
                                                                 ensemble_mean=self.ensemble_mean,
                                                                 random_state=random_state)

            array.append(da.from_delayed(delayed_frozen_phonon, shape=(1,), dtype=object))

        return da.concatenate(array),

    def _copy_as_dict(self, copy_atoms: bool = True) -> dict:

        kwargs = {'num_configs': len(self),
                  'sigmas': copy(self.sigmas),
                  'random_state': copy(self.random_state),
                  'ensemble_mean': self.ensemble_mean,
                  'directions': self.directions}

        if copy_atoms:
            kwargs['atoms'] = self.atoms.copy()

        return kwargs


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
    Molecular dynamics frozen phonons object.

    Parameters
    ----------
    trajectory: List of ASE Atoms objects
        Sequence of Atoms objects representing a thermal distribution of atomic configurations.
    """

    def __init__(self, trajectory: Sequence[Atoms], ensemble_mean: bool = True):

        if isinstance(trajectory, Atoms):
            trajectory = [trajectory]

        self._trajectory = trajectory

        super().__init__(ensemble_mean=ensemble_mean)

    def generate_configurations(self):
        for frozen_phonon in self:
            yield MDFrozenPhonons([frozen_phonon])

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def atoms(self) -> Atoms:
        return self[0]

    @property
    def cell(self) -> np.ndarray:
        return self[0].cell

    def ensemble_blocks(self, chunks: int = 1):
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

    def ensemble_partial(self):
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
