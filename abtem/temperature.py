"""Module to describe the effect of temperature on the atomic positions."""
from abc import abstractmethod, ABCMeta
from typing import Mapping, Union, Sequence
from numbers import Number
from collections import Iterable

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from copy import copy


class AbstractFrozenPhonons(metaclass=ABCMeta):
    """Abstract base class for frozen phonons objects."""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def generate_atoms(self):
        """
        Generate frozen phonon configurations.
        """
        pass

    def __iter__(self):
        return self.generate_atoms()

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


class DummyFrozenPhonons(AbstractFrozenPhonons):
    """
    Dummy frozen phonons object.

    Generates the input Atoms object. Used as a stand-in for simulations without frozen phonons.

    Parameters
    ----------
    atoms: ASE Atoms object
        Generated Atoms object.
    """

    def __init__(self, atoms: Atoms):
        self._atoms = atoms.copy()

    def __len__(self):
        return 1

    def generate_atoms(self):
        yield self._atoms

    def __copy__(self):
        return self.__class__(self._atoms.copy())


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
    seed: int
        Seed for random number generator.
    """

    def __init__(self,
                 atoms: Atoms,
                 num_configs: int,
                 sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
                 directions: str = 'xyz',
                 seed=None):

        if isinstance(sigmas, Number):
            sigmas_array = np.array(sigmas, dtype=np.float)
            sigmas_array = np.tile(sigmas_array, len(atoms))

        elif isinstance(sigmas, dict):
            sigmas_array = np.zeros(len(atoms), dtype=np.float)
            new_sigmas = {}
            for key, sigma in sigmas.items():
                try:
                    new_sigmas[atomic_numbers[key]] = sigma
                except KeyError:
                    pass

            for unique in np.unique(atoms.numbers):
                try:
                    sigmas_array[atoms.numbers == unique] = new_sigmas[unique]
                except KeyError:
                    raise RuntimeError('Displacement standard deviation must be provided for all atomic species.')

        elif isinstance(sigmas, Iterable):
            sigmas_array = np.array(sigmas, dtype=np.float)

        else:
            raise ValueError()

        if len(sigmas_array) != len(atoms):
            raise RuntimeError('Displacement standard deviation must be provided for all atoms.')

        self._sigmas = sigmas_array

        new_directions = []
        for direction in list(set(directions.lower())):
            if direction == 'x':
                new_directions += [0]
            elif direction == 'y':
                new_directions += [1]
            elif direction == 'z':
                new_directions += [2]
            else:
                raise RuntimeError('Directions must be "x", "y" or "z" not {}.')

        self._directions = new_directions
        self._atoms = atoms
        self._num_configs = num_configs
        self._seed = seed

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    def __len__(self):
        return self._num_configs

    def generate_atoms(self):
        if self._seed:
            np.random.seed(self._seed)

        for i in range(len(self)):
            atoms = self._atoms.copy()
            positions = atoms.get_positions()

            for direction in self._directions:
                positions[:, direction] += self._sigmas * np.random.randn(len(positions))

            atoms.set_positions(positions)

            yield atoms

    def __copy__(self):
        return self.__class__(atoms=self.atoms.copy(), num_configs=len(self), sigmas=self._sigmas.copy(),
                              seed=self._seed)


class MDFrozenPhonons(AbstractFrozenPhonons):
    """
    Molecular dynamics frozen phonons object.

    Parameters
    ----------
    trajectory: List of ASE Atoms objects
        Sequence of Atoms objects representing a thermal distribution of atomic configurations.
    """

    def __init__(self, trajectory: Sequence[Atoms]):
        self._trajectory = trajectory

    def __len__(self):
        return len(self._trajectory)

    def standard_deviations(self):
        mean_positions = np.mean([atoms.positions for atoms in self], axis=0)
        squared_deviations = [(atoms.positions - mean_positions) ** 2 for atoms in self]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))

    def generate_atoms(self):
        for i in range(len(self)):
            yield self._trajectory[i]

    def __copy__(self):
        return self.__class__(trajectory=[atoms.copy() for atoms in self._trajectory])
