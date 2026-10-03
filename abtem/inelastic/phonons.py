"""Module to describe the effect of temperature on the atomic positions."""

from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Union,
)

import dask
import dask.array as da
import numpy as np
from ase import Atoms, data
from ase.cell import Cell
from ase.io import read
from ase.io.trajectory import read_atoms
from dask.delayed import Delayed

from abtem.core.axes import (
    AxisMetadata,
    EnergyLossAxis,
    FrozenPhononsAxis,
    UnknownAxis,
)
from abtem.core.chunks import Chunks, chunk_ranges, iterate_chunk_ranges, validate_chunks
from abtem.core.ensemble import (
    Ensemble,
    _wrap_with_array,
    shared_constant_arg,
    unpack_blockwise_args,
)
from abtem.core.utils import CopyMixin, EqualityMixin, itemset

if TYPE_CHECKING:
    pass


Reader: Optional[Callable] = None
try:
    from gpaw.io import Reader  # noqa
except ImportError:
    Reader = None


# Per-atom array naming, for each atom of a transformed or repeated structure, the
# index of the atom of the original structure it is a copy of. A potential sets it
# before transforming the atoms, so that per-atom displacement standard deviations
# can follow their atoms into a structure with a different number of atoms.
SOURCE_INDEX = "abtem_source_index"


def _is_identity(frame: Optional[np.ndarray]) -> bool:
    """Whether a frame is the identity, to within rounding of a rotation or an
    orthogonalization; None is the identity."""
    if frame is None:
        return True
    return np.allclose(_as_frame(frame), np.eye(3), rtol=0.0, atol=1e-12)


def _as_frame(frame: Optional[np.ndarray]) -> np.ndarray:
    frame = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
    if frame.shape != (3, 3):
        raise ValueError(f"A frame must be a 3x3 matrix, not of shape {frame.shape}.")
    return frame


def _accepts_keyword(method: Callable, name: str) -> bool:
    """Whether a method accepts a keyword argument of the given name."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _safe_read_atoms(calculator, clean: bool = True) -> Atoms:
    if isinstance(calculator, str):
        assert Reader is not None
        with Reader(calculator) as reader:
            atoms = read_atoms(reader.atoms)
    else:
        atoms = calculator.atoms

    if clean:
        atoms.constraints = None
        atoms.calc = None

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
            atoms = atoms.compute(scheduler="single-threaded")

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

    def _randomize_transformed(self, atoms: Atoms, frame: np.ndarray) -> Atoms:
        """Randomize atoms that a potential transformed from :attr:`atoms`.

        `frame` is the linear map from the Cartesian axes of :attr:`atoms` to
        those of `atoms`, acting on row vectors. Subclasses whose displacements
        have no direction ignore it; this keeps subclasses that implement only
        `randomize(atoms)` working.
        """
        return self.randomize(atoms)

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
        return _wrap_with_array(new_dummy_frozen_phonons, 0)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("atoms",))
        return partial(self._from_partitioned_args_func, **kwargs)

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        # This ensemble has no chunking: a single constant travels as one
        # graph node. `chunks` is part of the Ensemble signature only.
        return (shared_constant_arg(self.atoms, lazy=lazy),)

    def __len__(self):
        if self._num_configs is None:
            return 1
        else:
            return self._num_configs


def validate_seeds(
    seeds: int | tuple[int, ...] | None,
    num_seeds: Optional[int] = None,
) -> tuple[int, ...]:
    if num_seeds is None and seeds is None:
        num_seeds = 1

    if isinstance(seeds, int) and num_seeds is None:
        seeds = (seeds,)

    elif seeds is None:
        if num_seeds is None:
            raise ValueError("Provide `num_seeds` or a seed for each configuration.")

        rng = np.random.default_rng(seed=seeds)
        seeds = ()
        while len(seeds) < num_seeds:
            seed = rng.integers(np.iinfo(np.int32).max)
            if seed not in seeds:
                seeds += (seed,)
    elif isinstance(seeds, int) and num_seeds is not None:
        seeds_to_make_new_seeds = seeds
        rng = np.random.default_rng(seed=seeds_to_make_new_seeds)
        seeds = ()
        while len(seeds) < num_seeds:
            seed = rng.integers(np.iinfo(np.int32).max)
            if seed not in seeds:
                seeds += (seed,)
    else:
        if not hasattr(seeds, "__len__"):
            raise ValueError("Invalid type for `seeds`.")

        seeds = tuple(seeds)

        if num_seeds is not None:
            assert num_seeds == len(seeds)

    return seeds


from abtem.atoms import (
    AtomProperties,
    B_to_sigma,
    sigma_to_B,
    validate_per_atom_property,
    validate_sigmas,
)


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

        The standard deviation applies to each displaced direction separately: every
        direction in `directions` receives an independent Gaussian displacement with
        standard deviation sigma, so sigma squared is the mean-square displacement
        along one direction (the isotropic displacement parameter U_iso), not the
        total mean-square displacement, which is 3 sigma squared for the default
        `directions`.

        Anisotropic displacements may be given by providing a standard deviation for
        each Cartesian direction (`x`, `y`, `z`). This may be a tuple of three numbers
        for identical displacements for all atoms. A dict of tuples of three numbers to
        specify displacements for each species. A list or array with three numbers for
        each atom.

        The directions of anisotropic standard deviations are the Cartesian axes of
        `atoms` as given. A potential that rotates the atoms to another `plane`, or
        orthogonalizes their cell, applies the same linear map to the displacements,
        including the small strain an orthogonalization may need. Isotropic standard
        deviations, equal in all three directions for every atom, are applied along
        the potential's axes without that map: an isotropic Gaussian is the same
        along any rotated axes, and the seeded displacements stay those of
        `sigmas * r`. Per-atom standard deviations follow their atoms when the
        potential repeats the cell.

    directions : str, optional
        The directions in which the atoms are displaced, as a string of one or more of
        'x', 'y' and 'z'. In a potential, these are the axes of the potential: 'x' and
        'y' perpendicular to the propagation direction and 'z' along it, whatever
        `plane` the atoms are rotated to. The default, 'xyz', displaces the atoms in all
        three directions; 'xy' keeps them in place along the propagation direction.
        Iterating the frozen phonons, or calling :meth:`to_atoms_ensemble`, displaces
        `atoms` without a potential, and the directions are then the axes of `atoms`.
        :meth:`~abtem.potentials.iam.Potential.to_atoms_ensemble` returns the
        configurations as a potential simulates them.
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
        self._axes  # raises on an invalid direction now rather than when displacing
        self._atoms = atoms
        self._seed = validate_seeds(seed, num_seeds=num_configs)

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
                raise RuntimeError(
                    f"Directions must be 'x', 'y' or 'z', not {direction!r}."
                )
        return axes

    def _sigmas_of(self, atoms: Atoms) -> np.ndarray:
        """The standard deviations of the given atoms, one row per atom.

        Per-species standard deviations apply to any structure. Per-atom ones
        belong to :attr:`atoms`; a structure with a different number of atoms
        needs the :data:`SOURCE_INDEX` array, which a potential sets before
        transforming the atoms.
        """
        if isinstance(self._sigmas, dict):
            sigmas = validate_per_atom_property(atoms, self._sigmas, return_array=True)
            assert isinstance(sigmas, np.ndarray)
            return sigmas

        sigmas = np.asarray(self._sigmas)
        if SOURCE_INDEX in atoms.arrays:
            return sigmas[atoms.arrays[SOURCE_INDEX]]
        if len(sigmas) != len(atoms):
            raise RuntimeError(
                f"Per-atom displacement standard deviations are given for "
                f"{len(sigmas)} atoms, but {len(atoms)} atoms are displaced; give "
                "them per species instead."
            )
        return sigmas

    def randomize(self, atoms: Atoms, frame: Optional[np.ndarray] = None) -> Atoms:
        """
        Randomly displace the atoms.

        Parameters
        ----------
        atoms : Atoms
            The atoms to displace: :attr:`atoms`, or a copy a potential has
            transformed (rotated, orthogonalized or repeated).
        frame : 3x3 array, optional
            The linear map from the Cartesian axes of :attr:`atoms` to those of
            `atoms`, acting on row vectors: a displacement `d` along the axes of
            :attr:`atoms` is `d @ frame` along those of `atoms`. Only anisotropic
            standard deviations depend on it. The default is the identity.

        Returns
        -------
        displaced : Atoms
        """
        return self._randomize(atoms, frame=frame)

    def _randomize_transformed(self, atoms: Atoms, frame: np.ndarray) -> Atoms:
        if type(self).randomize is FrozenPhonons.randomize:
            return self._randomize(atoms, frame=frame)
        # A subclass's own randomize, which may take the atoms only.
        if _accepts_keyword(self.randomize, "frame"):
            return self.randomize(atoms, frame=frame)
        return self.randomize(atoms)

    def _randomize(self, atoms: Atoms, frame: Optional[np.ndarray] = None) -> Atoms:
        sigmas = self._sigmas_of(atoms)

        atoms = atoms.copy()

        rng = np.random.default_rng(self.seed[0])
        r = rng.normal(size=(len(atoms), 3))

        if sigmas.ndim == 2:
            displacements = sigmas * r
            # Anisotropic standard deviations are given along the axes of
            # self.atoms; the displacements are drawn along those axes and
            # mapped to the axes of `atoms`. An isotropic Gaussian is the same
            # distribution along any rotated axes, so isotropic standard
            # deviations are not mapped, and their seeded displacements are
            # the same with any frame.
            isotropic = np.all(sigmas == sigmas[:, :1])
            if not isotropic and not _is_identity(frame):
                displacements = displacements @ _as_frame(frame)
        else:
            displacements = sigmas[:, None] * r

        for axis in self._axes:
            atoms.positions[:, axis] += displacements[:, axis]

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
            # One graph node shared by every chunk; created inside the loop,
            # each chunk would get its own copy under a fresh UUID key.
            lazy_atoms = dask.delayed(self.atoms)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                seeds = self.seed[start:stop]
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

        The configurations displace `atoms` as given. A potential that rotates,
        orthogonalizes or cuts the cell simulates configurations of the
        transformed atoms instead, which
        :meth:`~abtem.potentials.iam.Potential.to_atoms_ensemble` returns.

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

                trajectory_array = da.concatenate(stack)

            else:
                stack_array = np.empty(len(trajectory), dtype=object)
                for i, atoms in enumerate(trajectory):
                    itemset(stack_array, i, atoms)

                trajectory_array = stack_array
        elif isinstance(trajectory, (np.ndarray, da.core.Array)):
            trajectory_array = trajectory
        else:
            raise ValueError(f"Invalid type for `trajectory`, got {type(trajectory)}")
        # assert isinstance(trajectory, (np.ndarray, da.core.Array))

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = [FrozenPhononsAxis(_ensemble_mean=ensemble_mean)]
        elif isinstance(ensemble_axes_metadata, AxisMetadata):
            ensemble_axes_metadata = [ensemble_axes_metadata]
        elif not isinstance(ensemble_axes_metadata, list):
            raise ValueError()

        assert len(ensemble_axes_metadata) == len(trajectory_array.shape)

        atoms = trajectory_array.ravel()[0]
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(atoms, None, cell)

        self._trajectory = trajectory_array

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
    Energy-resolved ensemble of frozen-phonon configurations.

    Wraps a 2D array of Atoms objects ``(n_energies, n_configs)`` with an
    :class:`~abtem.core.axes.EnergyLossAxis` and
    :class:`~abtem.core.axes.FrozenPhononsAxis` as ensemble axes.  The energy
    values typically come from external phonon calculations.

    All inner configuration lists must have the same length.

    Notes
    -----
    When running a multi-slice :class:`~abtem.potentials.iam.Potential`
    (more than one slice) over this ensemble, prefer ``projection="finite"``
    over the default ``projection="infinite"``. The default assigns each
    atom to exactly one slice with a hard cutoff and no padding
    (:class:`~abtem.slicing.SliceIndexedAtoms`); if the configurations
    include out-of-plane (z) displacement, an atom sitting near a slice
    boundary can flip its entire potential contribution between slices
    across otherwise near-identical configurations, producing spurious
    discontinuities in the resulting spectra. ``projection="finite"`` uses
    padded slicing (:class:`~abtem.slicing.SlicedAtoms`) where such atoms
    blend gradually into the neighbouring slice instead. With a single
    slice there is no boundary to cross, so this does not arise regardless
    of projection method.

    Parameters
    ----------
    energy_resolved_snapshots : list of lists of ASE Atoms, or 2D numpy.ndarray
        Outer index is energy, inner index is configuration.
    energies : array-like
        Energy values [eV] corresponding to each outer entry.
    ensemble_mean : bool, optional
        If True (default), average over frozen-phonon configurations.
    cell : Cell, optional
    """

    def __init__(
        self,
        energy_resolved_snapshots: list[Sequence[Atoms]] | np.ndarray,
        energies: np.ndarray | Sequence[float],
        ensemble_mean: bool = True,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        cell: Optional[Cell] = None,
    ):
        energies = np.asarray(energies)

        if isinstance(energy_resolved_snapshots, np.ndarray):
            snapshots = energy_resolved_snapshots
        else:
            if len(energy_resolved_snapshots) != len(energies):
                raise ValueError(
                    "Number of snapshot groups must match the number of energies "
                    f"({len(energy_resolved_snapshots)} != {len(energies)})"
                )
            n_configs = len(energy_resolved_snapshots[0])
            for i, trajectory in enumerate(energy_resolved_snapshots[1:], 1):
                if len(trajectory) != n_configs:
                    raise ValueError(
                        f"All configuration lists must have the same length; "
                        f"entry 0 has {n_configs}, entry {i} has {len(trajectory)}"
                    )

            snapshots = np.empty((len(energies), n_configs), dtype=object)
            for i, trajectory in enumerate(energy_resolved_snapshots):
                for j, atoms in enumerate(trajectory):
                    itemset(snapshots, (i, j), atoms)

        atoms = snapshots.ravel()[0]
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(
            atoms, None, cell
        )

        self._snapshots = snapshots
        self._energies = energies

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

        if ensemble_axes_metadata is not None:
            self._ensemble_axes_metadata = ensemble_axes_metadata
        else:
            self._ensemble_axes_metadata = [
                EnergyLossAxis(
                    values=tuple(float(e) for e in energies),
                    units="eV",
                ),
                FrozenPhononsAxis(_ensemble_mean=ensemble_mean),
            ]

    @property
    def snapshots(self) -> np.ndarray:
        """2D object array of Atoms ``(n_energies, n_configs)``."""
        return self._snapshots

    @property
    def energies(self) -> np.ndarray:
        """Energy values [eV] for each snapshot group."""
        return self._energies

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self._snapshots.shape

    @property
    def num_configs(self) -> int:
        return self._snapshots.shape[1]

    @property
    def atoms(self) -> Atoms:
        atoms = self._snapshots.ravel()[0]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.item()
        return atoms

    def __len__(self) -> int:
        return len(self._energies)

    def __getitem__(self, item):
        new_snapshots = self._snapshots[item]
        new_energies = (
            self._energies[item]
            if not isinstance(item, tuple)
            else self._energies[item[0]]
        )
        if new_snapshots.ndim < 2:
            # `snapshots` is (n_energies, n_configs). A 1D result means one
            # of the two axes collapsed to a scalar index -- which one
            # determines whether the surviving values are energies (config
            # axis collapsed, item[1] is an int) or configs (energy axis
            # collapsed, the ordinary `ensemble[k]`/`ensemble[k, :]` case).
            energy_axis_collapsed = not (
                isinstance(item, tuple)
                and len(item) > 1
                and not isinstance(item[0], (int, np.integer))
            )
            if energy_axis_collapsed:
                new_snapshots = new_snapshots.reshape(1, -1)
            else:
                new_snapshots = new_snapshots.reshape(-1, 1)
        if np.ndim(new_energies) == 0:
            new_energies = np.atleast_1d(new_energies)
        kwargs = self._copy_kwargs(
            exclude=("energy_resolved_snapshots", "energies")
        )
        return EnergyResolvedAtomsEnsemble(
            new_snapshots, new_energies, **kwargs
        )

    def randomize(self, atoms: Atoms) -> Atoms:
        return atoms

    @property
    def _default_ensemble_chunks(self) -> tuple[int, ...]:
        return (1,) * len(self.ensemble_shape)

    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ):
        if chunks is None:
            chunks = 1
        chunks = validate_chunks(self.ensemble_shape, chunks)

        if lazy:
            array = np.empty(tuple(len(c) for c in chunks), dtype=object)
            for index, slic in iterate_chunk_ranges(chunks):
                snapshots_chunk = self._snapshots[slic]
                energies_chunk = self._energies[slic[0]]
                lazy_args = dask.delayed(_wrap_with_array)(
                    (snapshots_chunk, energies_chunk), ndims=1
                )
                lazy_array = da.from_delayed(
                    lazy_args, shape=(1,), dtype=object
                )
                itemset(array, index, lazy_array)

            shape = array.shape
            array = da.concatenate(array.flatten()).reshape(shape)
        else:
            snapshots = self._snapshots
            if isinstance(snapshots, da.core.Array):
                snapshots = snapshots.compute()

            array = np.empty(tuple(len(c) for c in chunks), dtype=object)
            for index, slic in iterate_chunk_ranges(chunks):
                snapshots_chunk = snapshots[slic]
                energies_chunk = self._energies[slic[0]]
                itemset(
                    array,
                    index,
                    _wrap_with_array((snapshots_chunk, energies_chunk), 1),
                )

        return (array,)

    @staticmethod
    def _from_partition_args_func(*args, **kwargs):
        args = unpack_blockwise_args(args)
        snapshots_chunk, energies_chunk = args[0]
        ensemble = EnergyResolvedAtomsEnsemble(
            snapshots_chunk, energies_chunk, **kwargs
        )
        return _wrap_with_array(ensemble)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("energy_resolved_snapshots", "energies")
        )
        kwargs["cell"] = self.cell.array
        kwargs["ensemble_axes_metadata"] = [UnknownAxis()] * len(
            self.ensemble_shape
        )
        return partial(self._from_partition_args_func, **kwargs)
