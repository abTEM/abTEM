"""Module to describe the effect of temperature on the atomic positions."""
from __future__ import annotations

from abc import abstractmethod, ABCMeta
from functools import partial
from numbers import Number
from typing import Sequence, Iterable

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase import data
from ase.cell import Cell
from ase.data import chemical_symbols
from ase.io import read
from ase.io.trajectory import read_atoms
from dask.delayed import Delayed

from abtem.core.axes import FrozenPhononsAxis, AxisMetadata, UnknownAxis
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.utils import CopyMixin, EqualityMixin

try:
    from gpaw.io import Reader  # noqa
except:
    Reader = None


def _safe_read_atoms(calculator, clean: bool = True):
    if isinstance(calculator, str):
        with Reader(calculator) as reader:
            atoms = read_atoms(reader.atoms)
    else:
        atoms = calculator.atoms

    if clean:
        atoms.constraints = None
        atoms.calc = True

    return atoms


class BaseFrozenPhonons(Ensemble, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """Base class for frozen phonons objects. Documented in the subclasses."""

    def __init__(self, atomic_numbers, cell, ensemble_mean: bool = True):
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

    @staticmethod
    def _validate_atomic_numbers_and_cell(atoms: Atoms | Delayed, atomic_numbers, cell):
        if isinstance(atoms, (Delayed, da.core.Array)) and (
            atomic_numbers is None or cell is None
        ):
            atoms = atoms.compute()

        if cell is None:
            cell = atoms.cell.copy()
        else:
            if not isinstance(cell, Cell):
                cell = Cell(cell)

            if not np.allclose(atoms.cell.array, cell.array):
                raise RuntimeError("cell of provided Atoms did not match provided cell")

        if atomic_numbers is None:
            atomic_numbers = np.unique(atoms.numbers)
        else:
            atomic_numbers = np.array(atomic_numbers, dtype=int)

        return atomic_numbers, cell

    @property
    @abstractmethod
    def atoms(self) -> Atoms | Delayed:
        pass

    @abstractmethod
    def randomize(self, atoms: Atoms) -> Atoms:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        for _, _, fp in self.generate_blocks(1):
            fp = fp.item()
            yield fp.randomize(fp.atoms)


class DummyFrozenPhonons(BaseFrozenPhonons):
    """Class to allow all potentials to be treated in the same way."""

    def __init__(
        self,
        atoms: Atoms | Delayed,
        num_configs: int = None,
    ):

        self._atoms = atoms
        self._num_configs = num_configs
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, None, None)
        super().__init__(atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=True)

    @property
    def num_configs(self):
        """Number of atomic configurations."""
        return self._num_configs

    @property
    def ensemble_shape(self):
        if self._num_configs is None:
            return ()
        else:
            return (self._num_configs,)

    @property
    def _default_ensemble_chunks(self):
        if self._num_configs is None:
            return ()
        else:
            return (1,)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        if self._num_configs is None:
            return []
        else:
            return [FrozenPhononsAxis(_ensemble_mean=self.ensemble_mean)]

    def randomize(self, atoms: Atoms) -> Atoms:
        return atoms

    @property
    def atoms(self):
        return self._atoms

    @classmethod
    def _from_partitioned_args_func(cls, args, **kwargs):

        if hasattr(args, "item"):
            args = args.item()

        atoms = args

        new_dummy_frozen_phonons = cls(atoms=atoms, **kwargs)

        return _wrap_with_array(new_dummy_frozen_phonons, 1)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("atoms",))
        return partial(self._from_partitioned_args_func, **kwargs)

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        if lazy:
            lazy_args = dask.delayed(_wrap_with_array)(self.atoms, ndims=1)

            array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
        else:
            array = _wrap_with_array(self.atoms, 1)
        return (array,)

    def __len__(self):
        if self._num_configs is None:
            return 1
        else:
            return self._num_configs


def _validate_seeds(
    seeds: int | tuple[int, ...] | None, num_seeds: int = None
) -> tuple[int, ...]:
    if seeds is None or np.isscalar(seeds):
        if num_seeds is None:
            raise ValueError("Provide `num_configs` or a seed for each configuration.")

        rng = np.random.default_rng(seed=seeds)
        seeds = ()
        while len(seeds) < num_seeds:
            seed = rng.integers(np.iinfo(np.int32).max)
            if seed not in seeds:
                seeds += (seed,)
    else:
        if not hasattr(seeds, "__len__"):
            raise ValueError

        if num_seeds is not None:
            assert num_seeds == len(seeds)

    return seeds


class FrozenPhonons(BaseFrozenPhonons):
    """
    The frozen phonons randomly displace the atomic positions to emulate thermal vibrations.

    Parameters
    ----------
    atoms : ASE.Atoms
        Atomic configuration used for displacements.
    num_configs : int
        Number of frozen phonon configurations.
    sigmas : float or dict or list
        If float, the standard deviation of the displacements is assumed to be identical for all atoms.
        If dict, a displacement standard deviation should be provided for each species. The atomic species can be
        specified as atomic number or a symbol, using the ASE standard.
        If list or array, a displacement standard deviation should be provided for each atom.
    directions : str, optional
        The displacement directions of the atoms as a string; for example 'xy' (default) for displacement in the `x`- and
        `y`-direction (ie. perpendicular to the propagation direction).
    ensemble_mean : bool, optional
        If True (default), the mean of the ensemble of results from a multislice simulation is calculated, otherwise,
        the result of every frozen phonon configuration is returned.
    seed : int or sequence of int
        Seed(s) for the random number generator used to generate the displacements, or one seed for each configuration in
         the frozen phonon ensemble.
    """

    def __init__(
        self,
        atoms: Atoms,
        num_configs: int,
        sigmas: float | dict[str | int, float] | Sequence[float],
        directions: str = "xyz",
        ensemble_mean: bool = True,
        seed: int | tuple[int, ...] = None,
    ):
        if isinstance(sigmas, dict):
            atomic_numbers = [data.atomic_numbers[symbol] for symbol in sigmas.keys()]
        else:
            atomic_numbers = None

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(
            atoms, atomic_numbers, cell=None
        )

        self._sigmas = sigmas
        self._directions = directions
        self._atoms = atoms
        self._seed = _validate_seeds(seed, num_seeds=num_configs)

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

    def _validate_sigmas(self, atoms):
        unique_symbols = [chemical_symbols[number] for number in self.atomic_numbers]
        sigmas = self._sigmas

        if isinstance(sigmas, Number):
            new_sigmas = {}
            for symbol in unique_symbols:
                new_sigmas[symbol] = sigmas

            sigmas = new_sigmas

        elif isinstance(sigmas, dict):
            if not all([symbol in unique_symbols for symbol in sigmas.keys()]):
                raise RuntimeError(
                    "Displacement standard deviation must be provided for all atomic species."
                )

        elif isinstance(sigmas, Iterable):
            sigmas = np.array(sigmas, dtype=np.float32)
            if len(sigmas) != len(atoms):
                raise RuntimeError(
                    "Displacement standard deviation must be provided for all atoms."
                )
        else:
            raise ValueError()

        return sigmas

    @property
    def ensemble_shape(self):
        return (len(self),)

    @property
    def _default_ensemble_chunks(self):
        return (1,)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return [FrozenPhononsAxis(_ensemble_mean=self.ensemble_mean)]

    @property
    def num_configs(self) -> int:
        return len(self._seed)

    @property
    def seed(self) -> tuple[int, ...]:
        return self._seed

    @property
    def sigmas(self) -> float | dict[str | int, float] | Sequence[float]:
        return self._sigmas

    @property
    def atoms(self) -> Atoms | Delayed:
        return self._atoms

    @property
    def directions(self) -> str:
        return self._directions

    def __len__(self) -> int:
        return self.num_configs

    @property
    def axes(self) -> list[int]:
        axes = []
        for direction in list(set(self._directions.lower())):
            if direction == "x":
                axes += [0]
            elif direction == "y":
                axes += [1]
            elif direction == "z":
                axes += [2]
            else:
                raise RuntimeError(f"Directions must be 'x', 'y' or 'z', not {axes}.")
        return axes

    def randomize(self, atoms: Atoms) -> Atoms:
        sigmas = self._validate_sigmas(atoms)

        if isinstance(sigmas, dict):
            temp = np.zeros(len(atoms.numbers), dtype=np.float32)
            for unique in np.unique(atoms.numbers):
                temp[atoms.numbers == unique] = np.float32(
                    sigmas[chemical_symbols[unique]]
                )
            sigmas = temp

        elif not isinstance(sigmas, np.ndarray):
            raise RuntimeError()

        atoms = atoms.copy()

        rng = np.random.default_rng(self.seed[0])
        r = rng.normal(size=(len(atoms), 3)) / np.sqrt(3)

        for axis in self.axes:
            atoms.positions[:, axis] += sigmas * r[:, axis]

        return atoms

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = unpack_blockwise_args(args)
        atoms, seed = args[0]

        new = cls(atoms=atoms, seed=seed, num_configs=len(seed), **kwargs)
        new = _wrap_with_array(new, len(new.ensemble_shape))
        return new

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("atoms", "seed", "num_configs"))
        output = partial(self._from_partitioned_args_func, **kwargs)
        return output

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        if lazy:
            arrays = []
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                seeds = self.seed[start:stop]
                lazy_atoms = dask.delayed(self.atoms)
                lazy_args = dask.delayed(_wrap_with_array)((lazy_atoms, seeds), ndims=1)
                lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
                arrays.append(lazy_array)

            array = da.concatenate(arrays)
        else:
            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                array.itemset(i, (self.atoms, self.seed[start:stop]))

        return (array,)

    def to_atoms_ensemble(self):
        """
        Convert the frozen phonons to an ensemble of atoms.

        Returns
        -------
        atoms_ensemble : AtomsEnsemble
        """
        trajectory = []
        for block in self.generate_blocks(1):
            trajectory.append(block[-1].randomize(block[-1].atoms))
        return AtomsEnsemble(trajectory)


class AtomsEnsemble(BaseFrozenPhonons):
    """
    Frozen phonons based on a molecular dynamics simulation.

    Parameters
    ----------
    trajectory : sequence of ASE.Atoms
        Sequence of atoms representing a thermal distribution of atomic configurations.
    ensemble_mean : True, optional
        If True, the mean of the ensemble of results from a multislice simulation is calculated, otherwise, the result
        of every frozen phonon is returned.i
    """

    def __init__(
        self,
        trajectory: Sequence[Atoms],
        ensemble_mean: bool = True,
        cell: Cell = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        ensemble_shape: tuple[int, ...] = None,
    ):

        if isinstance(trajectory, Atoms):
            trajectory = [trajectory]

        elif isinstance(trajectory, str):
            trajectory = read(trajectory, index=":")

        if isinstance(trajectory, (list, tuple)):
            if isinstance(trajectory[0], str):
                trajectory = [_safe_read_atoms(path) for path in trajectory]

            trajectory_list = trajectory

            trajectory = np.zeros(len(trajectory), dtype=object).reshape(ensemble_shape)
            for i, atoms in enumerate(trajectory_list):
                trajectory.itemset(i, atoms)

        if isinstance(trajectory, np.ndarray):
            trajectory = trajectory.reshape(ensemble_shape)

        elif not isinstance(trajectory, da.core.Array):
            raise ValueError()

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = [FrozenPhononsAxis(_ensemble_mean=ensemble_mean)]
        elif isinstance(ensemble_axes_metadata, AxisMetadata):
            ensemble_axes_metadata = [ensemble_axes_metadata]
        elif not isinstance(ensemble_axes_metadata, list):
            raise ValueError()

        assert len(ensemble_axes_metadata) == len(trajectory.shape)

        atoms = trajectory.ravel()[0]

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, None, cell)

        self._trajectory = trajectory

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

        self._ensemble_axes_metadata = ensemble_axes_metadata

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def atoms(self) -> Atoms:
        return self._trajectory.ravel()[0]

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        if isinstance(self._trajectory, (da.core.Array, np.ndarray)):
            return self._trajectory.shape
        return (len(self),)

    @property
    def _default_ensemble_chunks(self) -> tuple[int, ...]:
        if isinstance(self._trajectory, (da.core.Array, np.ndarray)):
            return (1,) * len(self.ensemble_shape)
        return (1,)

    def _partition_args(self, chunks: int = 1, lazy: bool = True):

        if lazy:
            if isinstance(self._trajectory, da.core.Array):
                return (self._trajectory,)
            else:
                return (da.from_array(self._trajectory, chunks=chunks),)
        else:
            if isinstance(self._trajectory, da.core.Array):
                return self._trajectory.compute()
            else:
                return self._trajectory

        # #if :
        # return self._trajectory,
        #
        # #print(self._trajectory)
        #
        # chunks = validate_chunks(self.ensemble_shape, chunks)
        #
        # def md_frozen_phonons(atoms):
        #     arr = np.zeros((1,), dtype=object)
        #     arr.itemset(0, atoms)
        #     return arr
        #
        # array = np.zeros((len(chunks[0]),), dtype=object)
        # start = 0
        # for i, chunk in enumerate(chunks[0]):
        #     stop = start + chunk
        #     trajectory = self._trajectory[start:stop]
        #
        #     if lazy:
        #         atoms = dask.delayed(lambda *args: list(args))(*trajectory)
        #
        #         delayed_frozen_phonon = dask.delayed(md_frozen_phonons)(atoms=atoms)
        #
        #         array.itemset(
        #             i,
        #             da.from_delayed(delayed_frozen_phonon, shape=(1,), dtype=object),
        #         )
        #     else:
        #         pass
        #
        #         # print("aaaaa", trajectory)
        #         # trajectory = [
        #         #     atoms.compute() if hasattr(atoms, "compute") else atoms
        #         #     for atoms in trajectory
        #         # ]
        #         # array.itemset(i, trajectory)
        #
        #     start = stop
        #
        # if lazy:
        #     array = da.concatenate(list(array))
        #
        # return (array,)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("trajectory", "ensemble_shape"))
        kwargs["cell"] = self.cell.array
        kwargs["ensemble_shape"] = (1,) * len(self.ensemble_shape)
        kwargs["ensemble_axes_metadata"] = [UnknownAxis()] * len(self.ensemble_shape)
        return partial(AtomsEnsemble, **kwargs)

    def randomize(self, atoms: Atoms) -> Atoms:
        return atoms

    def standard_deviations(self) -> np.ndarray:
        mean_positions = np.mean(
            [atoms.positions for atoms in self._trajectory], axis=0
        )
        squared_deviations = [
            (atoms.positions - mean_positions) ** 2 for atoms in self._trajectory
        ]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))
