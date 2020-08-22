"""Module to describe the effect of temperature on the atomic positions."""
from abc import abstractmethod, ABCMeta
from typing import Mapping, Union, Sequence

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
    sigmas: dict
        Mapping from atomic species to the standard deviation of the displacements of that atomic species.
        The atomic species can be specified as atomic number or symbol.
    seed: int
        Seed for random number generator.
    """

    def __init__(self, atoms: Atoms, num_configs: int, sigmas: Mapping[Union[str, int], float], seed=None):

        new_sigmas = {}
        for key, sigma in sigmas.items():
            try:
                new_sigmas[atomic_numbers[key]] = sigma
            except KeyError:
                pass

        unique_atomic_numbers = np.unique(atoms.get_atomic_numbers())

        if len(set(unique_atomic_numbers).intersection(set(new_sigmas.keys()))) != len(unique_atomic_numbers):
            raise RuntimeError('Displacement standard deviation not provided for all atomic species.')

        self._sigmas = new_sigmas
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

            for number, sigma in self._sigmas.items():
                indices = np.where(atoms.get_atomic_numbers() == number)[0]
                positions[indices] += sigma * np.random.randn(len(indices), 3)

            atoms.set_positions(positions)
            atoms.wrap()

            yield atoms

    def __copy__(self):
        return self.__class__(atoms=self.atoms.copy(), num_configs=len(self), sigma=self._sigmas.copy(),
                              seed=self._seed)


class MDFrozenPhonons(AbstractFrozenPhonons):
    """
    Molecular dynamics frozen phonons.

    Parameters
    ----------
    trajectory: List of ASE Atoms objects
        Sequence of Atoms objects representing a thermal distribution of atomic configurations.
    """

    def __init__(self, trajectory: Sequence[Atoms]):
        self._trajectory = trajectory

    def __len__(self):
        return len(self._trajectory)

    def generate_atoms(self):
        for i in range(len(self)):
            yield self._trajectory[i]

    def __copy__(self):
        return self.__class__(trajectory=[atoms.copy() for atoms in self._trajectory])
