"""Module to describe the effect of temperature on the atomic positions."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Optional, Sequence, TypeGuard, Union

import dask
import dask.array as da
import numpy as np
from ase import Atoms, data
from ase.cell import Cell
from ase.data import chemical_symbols
from ase.io import read
from ase.io.trajectory import read_atoms
from dask.delayed import Delayed

from abtem.core.axes import AxisMetadata, FrozenPhononsAxis, EnergyAxis, UnknownAxis
from abtem.core.chunks import Chunks, chunk_ranges, validate_chunks, iterate_chunk_ranges
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype, itemset

Reader: Optional[Callable] = None
try:
    from gpaw.io import Reader  # noqa
except ImportError:
    Reader = None


def _safe_read_atoms(calculator, clean: bool = True) -> Atoms:
    if isinstance(calculator, str):
        assert Reader is not None
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
        """The mean of the ensemble of results from a multislice simulation is
        calculated."""
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
    def _validate_atomic_numbers_and_cell(
        atoms: Atoms | np.ndarray,
        atomic_numbers: Optional[np.ndarray] = None,
        cell: Optional[Cell] = None,
    ) -> tuple[np.ndarray, Cell]:
        if isinstance(atoms, da.core.Array) and (
            atomic_numbers is None or cell is None
        ):
            atoms = atoms.compute()

        if isinstance(atoms, np.ndarray):
            atoms = atoms.item()

        assert isinstance(atoms, Atoms)

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

    @abstractmethod
    def randomize(self, atoms: Atoms) -> Atoms:
        """
        Randomize the atoms.

        Parameters
        ----------
        atoms : Atoms
        """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def num_configs(self):
        """Number of atomic configurations."""

    def __iter__(self):
        for _, _, fp in self.generate_blocks(1):
            fp = fp.item()
            yield fp.randomize(fp.atoms)


class DummyFrozenPhonons(BaseFrozenPhonons):
    """Class to allow all potentials to be treated in the same way."""

    def __init__(
        self,
        atoms: Atoms,
        num_configs: Optional[int] = None,
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
    def numbers(self):
        """The atomic numbers of the atoms."""
        return self.atoms.numbers

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

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = 1

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
    seeds: int | tuple[int, ...] | None, num_seeds: Optional[int] = None
) -> tuple[int, ...]:
    if seeds is None or isinstance(seeds, int):
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


def ensure_all_values_are_tuples(
    props: dict[Any, Any],
) -> TypeGuard[Dict[str, tuple[float | int, ...]]]:
    return all(isinstance(value, tuple) for value in props.values())


def all_keys_are_ints(props: dict[Any, Any]) -> TypeGuard[dict[int, Any]]:
    return all(isinstance(key, int) for key in props.keys())


AtomProperties = Union[
    float,
    np.ndarray,
    dict[str, float],
    dict[str, tuple[float, ...]],
    dict[str, np.ndarray],
    Sequence[float],
]


def validate_per_atom_property(
    atoms: Atoms,
    props: AtomProperties,
    return_array: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    atomic_numbers = np.unique(atoms.numbers)
    unique_symbols = [chemical_symbols[number] for number in atomic_numbers]

    validated_props: np.ndarray | dict[str, np.ndarray]
    dtype = get_dtype(complex=False)

    if isinstance(props, Number):
        validated_props = {
            symbol: np.array(props, dtype=dtype) for symbol in unique_symbols
        }

    elif isinstance(props, dict):
        if all_keys_are_ints(props):
            validated_props = {
                chemical_symbols[key]: np.array(value) for key, value in props.items()
            }
        elif not all(isinstance(key, str) for key in props.keys()):
            raise RuntimeError(
                "Keys in the properties dictionary must be either all "
                "atomic numbers or all chemical symbols."
            )

        if not set(unique_symbols).issubset(set(props.keys())):
            raise RuntimeError(
                "Property must be provided for all atomic species."
                f" symbols: {unique_symbols}, keys: {props.keys()}"
            )

        if ensure_all_values_are_tuples(props):
            first_attr = next(iter(props.values()))

            if not all(len(attr) == len(first_attr) for attr in props.values()):
                raise RuntimeError("All values must have the same length.")

        validated_props = {
            symbol: np.array(value, dtype=dtype) for symbol, value in props.items()
        }

    elif isinstance(props, (list, tuple, np.ndarray)):
        validated_props = np.array(props, dtype=dtype)
        if len(props) != len(atoms):
            raise RuntimeError("Property must be provided for all atoms.")
    else:
        raise ValueError("Invalid type for `props`.")

    if return_array and isinstance(validated_props, dict):
        return atom_property_dict_to_atom_property_array(atoms, validated_props)

    return validated_props


def validate_sigmas(
    atoms: Atoms, sigmas: AtomProperties, return_array: bool = False
) -> tuple[np.ndarray | dict[str, np.ndarray], bool]:
    """
    Validate the standard deviations of displacement for atoms in an atomic structure.

    Parameters
    ----------
    atoms : Atoms
        The atomic structure which standard deviations of displacement are to be
        validated.
    sigmas : float, dict[str, float] or Sequence[float]
        It can be either:

        - a single float value specifying the standard deviation for all atoms,
        - a dictionary mapping each atom's symbol or atomic number to a corresponding
          standard deviation,
        - a sequence of float values providing the standard deviation for each atom
          individually.

        For anisotropic displacements, either three values for each atom or for each
        element must be provided.

    Returns
    -------
    sigmas : dict[str, float] or np.ndarray
        The validated standard deviations
    anisotropic : bool
        A boolean value indicating whether the displacements are anisotropic.

    Raises
    ------
    ValueError
        If the type of `sigmas` is not float, dict, or list.
    RuntimeError
        If the length of `sigmas` does not match the length of `atoms`, or
        three values for each atom or each element are not given for anisotropic
        displacements.
    """

    validated_sigmas = validate_per_atom_property(
        atoms, sigmas, return_array=return_array
    )

    if isinstance(validated_sigmas, dict):
        sigmas_array = next(iter(validated_sigmas.values()))
    else:
        sigmas_array = validated_sigmas

    if (
        sigmas_array.shape
        and len(sigmas_array.shape) == 2
        and sigmas_array.shape[-1] == (3,)
    ):
        anisotropic = True
    elif len(sigmas_array.shape) < 2:
        anisotropic = False
    else:
        raise RuntimeError("Anisotropic displacements must be given as three values.")

    return validated_sigmas, anisotropic


def atom_property_dict_to_atom_property_array(
    atoms: Atoms, props: dict[str, np.ndarray]
) -> np.ndarray:
    dtype = get_dtype(complex=False)

    n = next(iter(props.values())).shape
    array = np.zeros((len(atoms.numbers),) + n, dtype=dtype)

    for unique in np.unique(atoms.numbers):
        array[atoms.numbers == unique] = np.array(
            props[chemical_symbols[unique]], dtype=dtype
        )

    return array


class FrozenPhonons(BaseFrozenPhonons):
    """
    The frozen phonons randomly displace the atomic positions to emulate thermal
    vibrations.

    Parameters
    ----------
    atoms : ASE.Atoms
        Atomic configuration used for displacements.
    num_configs : int
        Number of frozen phonon configurations.
    sigmas : float or dict or list
        If float, the standard deviation of the displacements is assumed to be identical
        for all atoms. If dict, a displacement standard deviation should be provided for
        each species. The atomic species can be specified as atomic number or a symbol,
        using the ASE standard. If list or array, a displacement standard deviation
        should be provided for each atom.

        Anistropic displacements may be given by providing a standard deviation for each
        principal direction. This may be a tuple of three numbers for identical
        displacements for all atoms. A dict of tuples of three numbers to specify
        displacements for each species. A list or array with three numbers for each
        atom.

    directions : str, optional
        The displacement directions of the atoms as a string; for example 'xy' (default)
        for displacement in the `x`- and `y`-direction (i.e. perpendicular to the
        propagation direction).
    ensemble_mean : bool, optional
        If True (default), the mean of the ensemble of results from a multislice
        simulation is calculated, otherwise, the result of every frozen phonon
        configuration is returned.
    seed : int or sequence of int
        Seed(s) for the random number generator used to generate the displacements, or
        one seed for each configuration in the frozen phonon ensemble.
    """

    def __init__(
        self,
        atoms: Atoms,
        num_configs: int,
        sigmas: (
            float | dict[str, float] | dict[str, tuple[float, ...]] | Sequence[float]
        ),
        directions: str = "xyz",
        ensemble_mean: bool = True,
        seed: Optional[int | tuple[int, ...]] = None,
    ):
        if isinstance(sigmas, dict):
            atomic_numbers = np.array(
                [data.atomic_numbers[symbol] for symbol in sigmas.keys()]
            )
        else:
            atomic_numbers = None

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(
            atoms, atomic_numbers, cell=None
        )

        self._sigmas = validate_sigmas(atoms, sigmas)[0]
        self._directions = directions
        self._atoms = atoms
        self._seed = _validate_seeds(seed, num_seeds=num_configs)

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

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
    def sigmas(self) -> np.ndarray | dict[str, np.ndarray]:
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

    def _validate_sigmas(self, atoms: Atoms):
        return validate_sigmas(atoms, self._sigmas)

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
            sigmas = atom_property_dict_to_atom_property_array(atoms, sigmas)

        assert isinstance(sigmas, np.ndarray)

        atoms = atoms.copy()

        rng = np.random.default_rng(self.seed[0])

        if anisotropic:
            r = rng.normal(size=(len(atoms), 3))
            for axis in self._axes:
                atoms.positions[:, axis] += sigmas[:, axis] * r[:, axis]
        else:
            r = rng.normal(size=(len(atoms), 3))

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

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = 1
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
                itemset(array, i, (atoms, self.seed[start:stop]))

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
        If True, the mean of the ensemble of results from a multislice simulation is
        calculated, otherwise, the result of every frozen phonon is returned.
    ensemble_axes_metadata : list of AxesMetadata, optional
        Axis metadata for each ensemble axis. The axis metadata must be compatible with
        the shape of the array.
    cell : Cell, optional
    """

    def __init__(
        self,
        trajectory: Sequence[Atoms],
        ensemble_mean: bool = True,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        cell: Optional[Cell] = None,
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
                stack_array = np.empty(len(trajectory), dtype=object)
                for i, atoms in enumerate(trajectory):
                    itemset(stack_array, i, atoms)

                trajectory = stack_array

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

    @property
    def numbers(self):
        """The atomic numbers of the atoms."""
        return self.trajectory[0].numbers

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
        atoms = self._trajectory.ravel()[0]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.item()
        return atoms

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

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = 1
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
                itemset(array, i, _wrap_with_array(trajectory[start:stop], 1))

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

    def mean_squared_deviations(self) -> np.ndarray:
        """
        Squared deviation of the positions of each atom in each direction.
        """
        positions = np.stack([atoms.positions for atoms in self.trajectory])
        return ((positions - positions.mean(0)) ** 2).mean(0)

    def standard_deviations(self) -> np.ndarray:
        """
        Standard deviation of the positions of each atom in each direction.
        """
        positions = np.stack([atoms.positions for atoms in self.trajectory])
        return (positions - positions.mean(0)).std()



class EnergyResolvedAtomsEnsemble(BaseFrozenPhonons):
    """
    An energy-resolved ensemble of frozen-phonon ensembles (list of lists of Atoms).
    Describes a list of AtomsEnsemble objects with associated energies.

    Currently requires all frozen-phonon ensembles to be of the same size.

    Parameters
    ----------
    list of trajectories : list of lists of ASE.Atoms
        List of sequences of atoms representing a distribution of atomic configurations corresponding
        to phonon displacement at specific energies.
    energies : array
        Array of energies corresponding in order to the energies of the atom sequences.
    ensemble_mean : True, optional
        If True, the mean of the ensemble of results from a multislice simulation is
        calculated, otherwise, the result of every frozen phonon is returned.

    """
    def __init__(
        self,
        energy_resolved_snapshots: list[Sequence[Atoms]],
        energies: np.ndarray,
        ensemble_mean: bool = True,
        cell: Optional[Cell] = None,
    ):
        
        assert (len(energy_resolved_snapshots) == len(energies)), "Number of snaphots needs to match the number of energies"

        snapshots_stack_array = np.empty(len(energies), dtype=object)

        for ene, trajectory in enumerate(energy_resolved_snapshots):
            itemset(snapshots_stack_array, ene, AtomsEnsemble(trajectory))

        snapshots_stack = snapshots_stack_array

        assert isinstance(snapshots_stack, np.ndarray)

        ensemble_axes_metadata = [EnergyAxis(label=r"energy loss", values=energies, units="eV"),
                                  FrozenPhononsAxis(_ensemble_mean=ensemble_mean)]

        #assert len(ensemble_axes_metadata) == 2

        atoms = snapshots_stack[0][0].atoms

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, None, cell)

        self._snapshots_stack = snapshots_stack
        self._energies = energies

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

        self._ensemble_axes_metadata = ensemble_axes_metadata

    @property
    def snapshots_stack(self) -> np.ndarray:
        """Array of arrays of atoms representing an energy-resolved ensemble of atomic configurations."""
        return self._snapshots_stack

    @property
    def energies(self):
        """The energy bins for the energy-resolved snapshots."""
        return self._energies

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        new_snapshots_stack = self._snapshots_stack[item]
        new_snapshots_stack = [[trj for trj in snap_stack.trajectory] for snap_stack in new_snapshots_stack]
        new_energies = self._energies[item]
        kwargs = self._copy_kwargs(exclude=("energy_resolved_snapshots","energies",))
        return EnergyResolvedAtomsEnsemble(new_snapshots_stack, np.array(new_energies), **kwargs)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    def __len__(self) -> int:
        return len(self._snapshots_stack)

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        if isinstance(self._snapshots_stack, (da.core.Array, np.ndarray)):
            return (len(self._snapshots_stack), len(self._snapshots_stack[0]))
        return (len(self),)

    @property
    def num_configs(self) -> int:
        return len(self._snapshots_stack[0]) # Assumes all stacks have equal number of snapshots

    @property
    def atoms(self) -> Atoms:
        atoms = self._snapshots_stack[0][0]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.item()
        return atoms

    def randomize(self, atoms: Atoms) -> Atoms:
        return atoms

    @property
    def _default_ensemble_chunks(self) -> tuple[int, ...]:
        if isinstance(self._snapshots_stack, (da.core.Array, np.ndarray)):
            return (1,) * len(self.ensemble_shape)
        return (1,)

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = 1
        chunks = validate_chunks(self.ensemble_shape, chunks)
        if lazy:
            arrays = []
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                snapshots_stack = self._snapshots_stack[start:stop]
                for sstack in snapshots_stack:
                    arrays.append(sstack._partition_args(chunks[1:], lazy=lazy))

            array = da.concatenate(arrays)

            
            # arrays = []
            # for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
            #     snapshots_stack = self._snapshots_stack[start:stop]
            #     energies = self.energies[start:stop]
                
            #     tmp = []
                
            #     for energy, sstack in zip(energies, snapshots_stack):
                    
            #         tmpargs = sstack._partition_args(chunks[1:])
            #         tmp.append(tmpargs)
                
                
            #     print('tmp', tmp)
            #     arrays.append((da.concatenate(tmp), energies))
            
            # print('arrays', arrays)            
            # print('array', array)
        else:
            snapshots_stack = self._snapshots_stack
            if isinstance(snapshots_stack, da.core.Array):
                snapshots_stack = snapshots_stack.compute()

            array = np.empty(tuple(len(x) for x in chunks), dtype=object)
            
            for index, slic in iterate_chunk_ranges(chunks):
                tmp_snapshots_stack = snapshots_stack[slic]
                itemset(array,  index, _wrap_with_array(tmp_snapshots_stack, 1))
        
            # snapshots_stack = self._snapshots_stack
            # if isinstance(snapshots_stack, da.core.Array):
            #     snapshots_stack = snapshots_stack.compute()

            # array = np.zeros((len(chunks[0]),), dtype=object)
            # for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
            #     itemset(array, i, _wrap_with_array(snapshots_stack[start:stop], 1))

        return (array,)

    @staticmethod
    def _from_partition_args_func(*args, **kwargs):
        args = unpack_blockwise_args(args)
        energy_resolved_snapshots = args[0]
        energies = args[1]
        snapshots_stack = EnergyResolvedAtomsEnsemble(energy_resolved_snapshots, energies, **kwargs)
        return _wrap_with_array(snapshots_stack, 2) # Should this be 2?

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("energy_resolved_snapshots", "energies", "ensemble_shape"))
        kwargs["cell"] = self.cell.array
        kwargs["ensemble_axes_metadata"] = [UnknownAxis()] * len(self.ensemble_shape)
        return partial(self._from_partition_args_func, **kwargs)



def validate_energy_resolved_snaps(energies, energy_resolved_snapshots):
    
    assert (len(energy_resolved_snapshots) == len(energies)), "Number of snaphots needs to match the number of energies"
    
    nsnaps = len(energy_resolved_snapshots[0])
    for trajectory in energy_resolved_snapshots[1:]:
        assert len(trajectory) == nsnaps



class EnergyResolvedAtomsEnsemble2(BaseFrozenPhonons):
    """
    An energy-resolved ensemble of frozen-phonon ensembles (list of lists of Atoms).
    Describes a list of AtomsEnsemble objects with associated energies.

    Currently requires all frozen-phonon ensembles to be of the same size.

    Parameters
    ----------
    list of trajectories : list of lists of ASE.Atoms
        List of sequences of atoms representing a distribution of atomic configurations corresponding
        to phonon displacement at specific energies.
    energies : array
        Array of energies corresponding in order to the energies of the atom sequences.
    ensemble_mean : True, optional
        If True, the mean of the ensemble of results from a multislice simulation is
        calculated, otherwise, the result of every frozen phonon is returned.

    """
    def __init__(
        self,
        energy_resolved_snapshots: list[Sequence[Atoms]],
        energies: np.ndarray,
        ensemble_mean: bool = True,
        cell: Optional[Cell] = None,
    ):

        validate_energy_resolved_snaps(energies, energy_resolved_snapshots)

        snapshots_stack_array = np.empty((len(energies), len(energy_resolved_snapshots[0])), dtype=object)

        for ene, trajectory in enumerate(energy_resolved_snapshots):
            for i, atoms in enumerate(trajectory):
                itemset(snapshots_stack_array, (ene, i), atoms)
        
        snapshots_stack = snapshots_stack_array

        assert isinstance(snapshots_stack, np.ndarray)

        ensemble_axes_metadata = [EnergyAxis(label=r"energy loss", values=energies, units="eV"),
                                  FrozenPhononsAxis(_ensemble_mean=ensemble_mean)]
        
        #assert len(ensemble_axes_metadata) == 2

        atoms = snapshots_stack[0,0]

        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, None, None)
        
        self._snapshots_stack = snapshots_stack
        self._energies = energies

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

        self._ensemble_axes_metadata = ensemble_axes_metadata

    @property
    def snapshots_stack(self) -> np.ndarray:
        """Array of arrays of atoms representing an energy-resolved ensemble of atomic configurations."""
        return self._snapshots_stack

    @property
    def energies(self):
        """The energy bins for the energy-resolved snapshots."""
        return self._energies

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        new_snapshots_stack = self._snapshots_stack[item]
        new_snapshots_stack = [[trj for trj in snap_stack.trajectory] for snap_stack in new_snapshots_stack]
        new_energies = self._energies[item]
        kwargs = self._copy_kwargs(exclude=("energy_resolved_snapshots","energies",))
        return EnergyResolvedAtomsEnsemble(new_snapshots_stack, np.array(new_energies), **kwargs)
    
    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    def __len__(self) -> int:
        return len(self._snapshots_stack)

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        if isinstance(self._snapshots_stack, (da.core.Array, np.ndarray)):
            return self._snapshots_stack.shape
        return (len(self),)

    @property
    def num_configs(self) -> int:
        return len(self._snapshots_stack[0,0]) # Assumes all stacks have equal number of snapshots

    @property
    def atoms(self) -> Atoms:
        atoms = self._snapshots_stack[0,0]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.item()
        return atoms

    def randomize(self, atoms: Atoms) -> Atoms:
        return atoms

    @property
    def _default_ensemble_chunks(self) -> tuple[int, ...]:
        if isinstance(self._snapshots_stack, (da.core.Array, np.ndarray)):
            return (1,) * len(self.ensemble_shape)
        return (1,)

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = 1
        
        chunks = validate_chunks(self.ensemble_shape, chunks)
        if lazy:

            array = np.empty(tuple(len(x) for x in chunks), dtype=object)
            
            for index, slic in iterate_chunk_ranges(chunks):
                
                snapshots_stack = self._snapshots_stack[slic]
                energies = self.energies[slic[0]]
                
                lazy_args = dask.delayed(_wrap_with_array)((snapshots_stack, energies), ndims=1)
                lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
                
                itemset(array, index, lazy_array)
            
            shape = array.shape
            array = da.concatenate(array.flatten())
            array = array.reshape(shape)

        else:
            snapshots_stack = self._snapshots_stack
            if isinstance(snapshots_stack, da.core.Array):
                snapshots_stack = snapshots_stack.compute()

            array = np.empty(tuple(len(x) for x in chunks), dtype=object)
            
            for index, slic in iterate_chunk_ranges(chunks):
                tmp_snapshots_stack = snapshots_stack[slic]
                energies = self.energies[slic[0]]
                itemset(array,  index, _wrap_with_array((tmp_snapshots_stack, energies), 1))
        
        return (array,)
    
    @staticmethod
    def _from_partition_args_func(*args, **kwargs):
        args = unpack_blockwise_args(args)
        energy_resolved_snapshots, energies = args[0]
        snapshots_stack = EnergyResolvedAtomsEnsemble2(energy_resolved_snapshots, energies, **kwargs)
        return _wrap_with_array(snapshots_stack, len(snapshots_stack)) # Should this be 2?

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("energy_resolved_snapshots", "energies", "ensemble_shape"))
        kwargs["cell"] = self.cell.array
 #       kwargs["ensemble_axes_metadata"] = [UnknownAxis()] * len(self.ensemble_shape)
        return partial(self._from_partition_args_func, **kwargs)
    
    
    