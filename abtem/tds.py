import numpy as np
from ase.data import atomic_numbers


class TDS(object):

    def __init__(self, atoms, sigmas, n):
        self._n = n
        new_sigmas = {}
        for key, sigma in sigmas.items():
            try:
                new_sigmas[atomic_numbers[key]] = sigma
            except KeyError:
                pass

        self._sigmas = new_sigmas
        self._atoms = atoms

    def get_displaced_atoms(self):
        atoms = self._atoms.copy()

        atomic_numbers = np.unique(atoms.get_atomic_numbers())

        if len(set(atomic_numbers).intersection(set(self._sigmas.keys()))) != len(atomic_numbers):
            raise RuntimeError('provide sigma for all atomic species')

        positions = atoms.get_positions()

        for number in atomic_numbers:
            indices = np.where(atoms.get_atomic_numbers() == number)[0]
            positions[indices] += self._sigmas[number] * np.random.randn(len(indices), 3)

        atoms.set_positions(positions)
        return atoms

    def generate(self):
        for i in range(self._n):
            yield self.get_displaced_atoms()
