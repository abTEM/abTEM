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
except ImportError:
    Reader = None


def _safe_read_atoms(calculator, clean: bool = True) -> Atoms:
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
    """Base class for frozen phonons."""

    def __init__(
        self, atomic_numbers: np.ndarray, cell: Cell, ensemble_mean: bool = True
    ):
        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._ensemble_mean = ensemble_mean

    @property
    def ensemble_mean(self):
        """The mean of the ensemble of results from a multislice simulation is calculated."""
        return self._ensemble_mean

    @property
    def atomic_numbers(self) -> np.ndarray:
        """The unique atomic number of the atoms."""
        return self._atomic_numbers

    @property
    def cell(self) -> Cell:
        """The cell of the atoms."""
        return self._cell

    @staticmethod
    def _validate_atomic_numbers_and_cell(atoms: Atoms, atomic_numbers, cell):
        if isinstance(atoms, da.core.Array) and (
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
    def atoms(self) -> Atoms:
        """Base atomic configuration used for displacements."""
        pass

    @abstractmethod
    def randomize(self, atoms: Atoms) -> Atoms:
        """
        Randomize the atoms.

        Parameters
        ----------
        atoms : Atoms
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def num_configs(self):
        """Number of atomic configurations."""
        pass

    def __iter__(self):
        for _, _, fp in self.generate_blocks(1):
            fp = fp.item()
            yield fp.randomize(fp.atoms)


class DummyFrozenPhonons(BaseFrozenPhonons):
    """Class to allow all potentials to be treated in the same way."""

    def __init__(
        self,
        atoms: Atoms,
        num_configs: int = None,
    ):
        self._atoms = atoms
        self._num_configs = num_configs
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, None, None)
        super().__init__(atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=True)

    @property
    def num_configs(self):
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
            atoms = self.atoms
            array = _wrap_with_array(atoms, 1)
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

        Anistropic displacements may be given by providing a standard deviation for each principal direction.
        This may be a tuple of three numbers for identical displacements for all atoms. A dict of tuples of three
        numbers to specify displacements for each species. A list or array with three numbers for each atom.

    directions : str, optional
        The displacement directions of the atoms as a string; for example 'xy' (default) for displacement in the `x`-
        and `y`-direction (i.e. perpendicular to the propagation direction).
    ensemble_mean : bool, optional
        If True (default), the mean of the ensemble of results from a multislice simulation is calculated, otherwise,
        the result of every frozen phonon configuration is returned.
    seed : int or sequence of int
        Seed(s) for the random number generator used to generate the displacements, or one seed for each configuration
        in the frozen phonon ensemble.
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

            anisotropic = False
            sigmas = new_sigmas

        elif isinstance(sigmas, dict):
            anisotropic = any(hasattr(value, "__len__") for value in sigmas.values())

            if anisotropic and not all(len(value) == 3 for value in sigmas.values()):
                raise RuntimeError("Three values for each element must be given for anisotropic displacements.")

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

            if len(sigmas.shape) == 2:
                if sigmas.shape[1] == 3:
                    anisotropic = True
                else:
                    raise RuntimeError(
                        "Three values for each atom must be given for anisotropic displacements."
                    )
            elif len(sigmas.shape) == 1:
                anisotropic = False
            else:
                raise RuntimeError()
        else:
            raise ValueError()

        return sigmas, anisotropic

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
        """Random seed for each displacement configuration."""
        return self._seed

    @property
    def sigmas(self) -> float | dict[str | int, float] | Sequence[float]:
        """Displacement standard deviation for each atom."""
        return self._sigmas

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def directions(self) -> str:
        """The directions of the random displacements."""
        return self._directions

    def __len__(self) -> int:
        return self.num_configs

    @property
    def _axes(self) -> list[int]:
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
        sigmas, anisotropic = self._validate_sigmas(atoms)

        if isinstance(sigmas, dict):
            if anisotropic:
                temp = np.zeros((len(atoms.numbers), 3), dtype=np.float32)
            else:
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

        if anisotropic:
            r = rng.normal(size=(len(atoms), 3))
            for axis in self._axes:
                atoms.positions[:, axis] += sigmas[:, axis] * r[:, axis]
        else:
            r = rng.normal(size=(len(atoms), 3)) / np.sqrt(3)

            for axis in self._axes:
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
            atoms = self.atoms
            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                array.itemset(i, (atoms, self.seed[start:stop]))

        return (array,)

    def to_atoms_ensemble(self):
        """
        Convert the frozen phonons to an ensemble of atoms.

        Returns
        -------
        atoms_ensemble : AtomsEnsemble
        """
        trajectory = []
        for _, _, block in self.generate_blocks(1):
            block = block.item()
            trajectory.append(block.randomize(block.atoms))
        return AtomsEnsemble(trajectory)


class AtomsEnsemble(BaseFrozenPhonons):
    """
    Frozen phonons based on a molecular dynamics simulation.

    Parameters
    ----------
    trajectory : list of ASE.Atoms, dask.core.Array, list of dask.Delayed
        Sequence of atoms representing a distribution of atomic configurations.
    ensemble_mean : True, optional
        If True, the mean of the ensemble of results from a multislice simulation is calculated, otherwise, the result
        of every frozen phonon is returned.
    ensemble_axes_metadata : list of AxesMetadata, optional
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    cell : Cell, optional
    """

    def __init__(
        self,
        trajectory: Sequence[Atoms],
        ensemble_mean: bool = True,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        cell: Cell = None,
    ):
        if isinstance(trajectory, str):
            trajectory = read(trajectory, index=":")

        elif isinstance(trajectory, Atoms):
            trajectory = [trajectory]

        if isinstance(trajectory, (list, tuple)):
            if isinstance(trajectory[0], str):
                trajectory = [_safe_read_atoms(path) for path in trajectory]

            if isinstance(trajectory[0], Delayed):
                stack = []
                for atoms in trajectory:
                    atoms = dask.delayed(_wrap_with_array)(atoms, 1)
                    atoms = da.from_delayed(atoms, shape=(1,), dtype=object)
                    stack.append(atoms)

                trajectory = da.concatenate(stack)

            else:
                stack = np.empty(len(trajectory), dtype=object)
                for i, atoms in enumerate(trajectory):
                    stack.itemset(i, atoms)

                trajectory = stack

        assert isinstance(trajectory, (np.ndarray, da.core.Array))

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
    def trajectory(self) -> np.ndarray | da.core.Array:
        """Array of atoms representing an ensemble of atomic configurations."""
        return self._trajectory

    def __getitem__(self, item):
        new_trajectory = self._trajectory[item]
        kwargs = self._copy_kwargs(exclude=("trajectory",))
        return AtomsEnsemble(new_trajectory, **kwargs)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def num_configs(self) -> int:
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
        chunks = validate_chunks(self.ensemble_shape, chunks)
        if lazy:
            arrays = []
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                trajectory = self.trajectory[start:stop]
                lazy_args = dask.delayed(_wrap_with_array)(trajectory, ndims=1)
                lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
                arrays.append(lazy_array)

            array = da.concatenate(arrays)
        else:
            trajectory = self.trajectory
            if isinstance(trajectory, da.core.Array):
                trajectory = trajectory.compute()

            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                array.itemset(i, _wrap_with_array(trajectory[start:stop], 1))

        return (array,)

    @staticmethod
    def _from_partition_args_func(*args, **kwargs):
        args = unpack_blockwise_args(args)
        trajectory = args[0]
        atoms_ensemble = AtomsEnsemble(trajectory, **kwargs)
        return _wrap_with_array(atoms_ensemble, 1)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("trajectory", "ensemble_shape"))
        kwargs["cell"] = self.cell.array

        kwargs["ensemble_axes_metadata"] = [UnknownAxis()] * len(self.ensemble_shape)
        return partial(self._from_partition_args_func, **kwargs)

    def randomize(self, atoms: Atoms) -> Atoms:
        return atoms

    def standard_deviations(self) -> np.ndarray:
        """
        Standard deviation of the positions of each atom in each direction.
        """
        ensemble_positions = [atoms.positions for atoms in self._trajectory]

        num_atoms = len(ensemble_positions[0])
        if not all(len(positions) == num_atoms for positions in ensemble_positions):
            raise RuntimeError()

        mean_positions = np.mean(ensemble_positions, axis=0)
        squared_deviations = [
            (atoms.positions - mean_positions) ** 2 for atoms in self._trajectory
        ]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))
