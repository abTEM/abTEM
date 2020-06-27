from abc import abstractmethod, ABCMeta
from typing import Mapping, Union

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


class AbstractTDS(metaclass=ABCMeta):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def generate_atoms(self):
        pass


class TDS(AbstractTDS):

    def __init__(self, atoms: Atoms, num_configs, sigmas: Mapping[Union[str, int], float], seed=None):
        """
        Generates atomic configurations for thermal diffuse scattering.

        Randomly displaces the atomic positions in of an ASE Atoms object to simulate thermal vibrations.

        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms object with the average atomic configuration.
        sigmas : Mapping[Union[str, int], float]
            Mapping from atomic species to the variance of the displacements of that atomic species. The atomic species
            can be specified as atomic number or symbol.
        """
        new_sigmas = {}
        for key, sigma in sigmas.items():
            try:
                new_sigmas[atomic_numbers[key]] = sigma
            except KeyError:
                pass

        unique_atomic_numbers = np.unique(atoms.get_atomic_numbers())

        if len(set(unique_atomic_numbers).intersection(set(new_sigmas.keys()))) != len(unique_atomic_numbers):
            raise RuntimeError('provide sigma for all atomic species')

        self._sigmas = new_sigmas
        self._atoms = atoms
        self._num_configs = num_configs
        self._seed = seed

    def __len__(self):
        return self._num_configs

    def generate_atoms(self):
        if self._seed:
            np.random.seed(self._seed)

        for i in range(self._num_configs):
            atoms = self._atoms.copy()
            positions = atoms.get_positions()

            for number, sigma in self._sigmas.items():
                indices = np.where(atoms.get_atomic_numbers() == number)[0]
                positions[indices] += sigma * np.random.randn(len(indices), 3)

            atoms.set_positions(positions)
            atoms.wrap()

            yield atoms
