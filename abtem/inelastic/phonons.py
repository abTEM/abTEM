"""Module to describe the effect of temperature on the atomic positions."""

from __future__ import annotations

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
from ase.geometry import find_mic
from ase.io import read
from ase.io.trajectory import read_atoms
from dask.delayed import Delayed

from abtem.core.axes import (
    AxisMetadata,
    EnergyLossAxis,
    FrozenPhononsAxis,
    PhononParityAxis,
    PhononRestParityAxis,
    UnknownAxis,
)
from abtem.core.chunks import Chunks, chunk_ranges, iterate_chunk_ranges, validate_chunks
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.utils import CopyMixin, EqualityMixin, itemset

if TYPE_CHECKING:
    pass


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
        if chunks is None:
            chunks = 1

        if lazy:
            lazy_args = dask.delayed(_wrap_with_array)(self.atoms, ndims=0)
            array = da.from_delayed(lazy_args, shape=(), dtype=object)
        else:
            atoms = self.atoms
            array = _wrap_with_array(atoms, ndims=0)
        return (array,)

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
    atom_property_dict_to_atom_property_array,
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
                # If sigmas is 2D (anisotropic), extract the sigma for this axis
                sigma_axis = sigmas[:, axis] if sigmas.ndim == 2 else sigmas
                atoms.positions[:, axis] += sigma_axis * r[:, axis]
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


def _validate_parity_snapshot(
    atoms: Atoms,
    equilibrium_atoms: Atoms,
    max_displacement: Optional[float],
    index: tuple,
) -> None:
    """Check that ``atoms`` is a displacement of ``equilibrium_atoms`` with
    the same atoms in the same order, so that ``2 * R_eq - R`` is a
    meaningful twin. ``index`` is the (energy, config) position, for the
    error message only."""
    if len(atoms) != len(equilibrium_atoms):
        raise ValueError(
            f"snapshot {index} has {len(atoms)} atoms but equilibrium_atoms "
            f"has {len(equilibrium_atoms)}; every snapshot must be a "
            "displacement of the same structure."
        )
    if not np.array_equal(atoms.numbers, equilibrium_atoms.numbers):
        raise ValueError(
            f"snapshot {index} has a different species sequence than "
            "equilibrium_atoms; atoms must be in the same order in every "
            "snapshot and in equilibrium_atoms."
        )
    if max_displacement is None:
        return

    displacement = atoms.positions - equilibrium_atoms.positions
    # minimum image, so snapshots wrapped back into the cell (an atom near a
    # boundary displaced across it) are not flagged. find_mic (rather than a
    # hand-rolled fractional-coordinate wrap requiring an invertible 3x3
    # cell) also handles the common case of a 2D material with a degenerate
    # or undefined out-of-plane cell vector (e.g. ase.build.graphene()'s
    # default cell has rank 2, pbc=(True, True, False)): it wraps only the
    # periodic directions and leaves the rest untouched.
    # abTEM treats the potential as periodic along every non-degenerate cell
    # vector whatever `atoms.pbc` says (hand-built ASE Atoms default to
    # pbc=False), so wrap along those directions rather than only the
    # pbc-flagged ones; a zero cell vector (2D materials built by ASE) is
    # simply not wrapped.
    periodic = np.array([np.linalg.norm(vector) > 0 for vector in atoms.cell])
    if periodic.any():
        displacement, _ = find_mic(displacement, atoms.cell, pbc=periodic)
    largest = np.linalg.norm(displacement, axis=1)
    worst = int(np.argmax(largest))
    if largest[worst] > max_displacement:
        raise ValueError(
            f"atom {worst} of snapshot {index} is displaced by "
            f"{largest[worst]:.3f} Å from equilibrium_atoms, more than "
            f"max_displacement={max_displacement} Å. This usually means the "
            "snapshot and equilibrium_atoms are ordered differently (or "
            "equilibrium_atoms is not the structure the snapshots were "
            "displaced from). Pass max_displacement=None to disable the "
            "check if the displacement is genuine."
        )


def _validate_rest_snapshots(
    rest_snapshots,
    shape: tuple[int, int],
    equilibrium_atoms: Atoms,
    max_displacement: Optional[float],
) -> Optional[np.ndarray]:
    """Bring ``rest_snapshots`` into an ``(n_energies, n_configs)`` object
    array matching the bin snapshots (a flat list of ``n_configs`` fields is
    reused for every energy), and validate each against the equilibrium
    structure. Returns None if no rest snapshots were given."""
    if rest_snapshots is None:
        return None

    n_energies, n_configs = shape
    rest = np.empty(shape, dtype=object)
    if len(rest_snapshots) > 0 and isinstance(rest_snapshots[0], Atoms):
        if len(rest_snapshots) != n_configs:
            raise ValueError(
                "a flat list of rest_snapshots must have one entry per "
                f"configuration ({n_configs}), got {len(rest_snapshots)}."
            )
        for i in range(n_energies):
            for j, atoms in enumerate(rest_snapshots):
                itemset(rest, (i, j), atoms)
    else:
        if len(rest_snapshots) != n_energies or any(
            len(group) != n_configs for group in rest_snapshots
        ):
            raise ValueError(
                "rest_snapshots must have the same (energy, configuration) "
                f"layout as energy_resolved_snapshots, {shape}."
            )
        for i, group in enumerate(rest_snapshots):
            for j, atoms in enumerate(group):
                itemset(rest, (i, j), atoms)

    for index in np.ndindex(shape):
        _validate_parity_snapshot(
            rest[index], equilibrium_atoms, max_displacement, ("rest",) + index
        )
        if not np.allclose(rest[index].cell, equilibrium_atoms.cell, atol=1e-6):
            raise ValueError(
                f"rest snapshot {index} has a different cell than "
                "equilibrium_atoms; the rest displacement field must be a "
                "displacement of the same structure."
            )
    return rest


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
        Outer index is energy, inner index is configuration. If
        ``parity_projection`` is True, this must still be given in this
        ordinary (real-configuration-only) shape -- the displacement-reversed
        twin is built automatically.
    energies : array-like
        Energy values [eV] corresponding to each outer entry.
    equilibrium_atoms : ASE.Atoms, optional
        The shared undisplaced/equilibrium structure every snapshot in
        ``energy_resolved_snapshots`` is a displacement of. Required if
        ``parity_projection`` is True (used to build each snapshot's
        displacement-reversed twin, ``2 * equilibrium_atoms.positions -
        snapshot.positions``); unused otherwise. Must have the same number
        of atoms, in the same order and of the same species, as every
        snapshot -- a reordered or otherwise mismatched structure would
        silently produce meaningless twins, so this is checked.
    max_displacement : float, optional
        Sanity bound [Å] on the (minimum-image) displacement of every atom
        in every snapshot from ``equilibrium_atoms``, checked when
        ``parity_projection`` is True. Thermal and zero-point displacements
        are a few hundredths to a few tenths of an Å; a larger value almost
        always means the snapshot and equilibrium atoms are ordered
        differently. Default 1.0 Å; ``None`` disables the check.
    rest_snapshots : list of lists of ASE Atoms, or list of ASE Atoms, optional
        Displacement fields of the *rest* of the phonon spectrum, all modes
        outside the energy bin (or, simpler and adequate for narrow bins,
        drawn from the full thermal ensemble), each a displacement of
        ``equilibrium_atoms``. Given per snapshot in the same
        ``(energy, configuration)`` layout as ``energy_resolved_snapshots``,
        or as one list of ``n_configs`` fields reused for every energy bin.
        Requires ``parity_projection=True``. Every snapshot is then
        propagated at the four structures ``R_eq ± u_bin ± u_rest``, and a
        :class:`~abtem.core.axes.PhononRestParityAxis`
        (``values=("plus", "minus")``) follows the parity axis. Averaging
        the exit waves over the rest axis keeps the part even in the rest
        displacement, which carries the Debye-Waller damping of the bin's
        one-phonon amplitude by all other modes -- the factor a
        bin-restricted snapshot lacks -- while the part odd in the rest
        displacement (one bin phonon plus one rest phonon, mis-binned at
        this energy) cancels exactly. With rest fields,
        :func:`~abtem.measurements.phonon_loss_diffraction_patterns`
        returns the one-phonon channel only, unless
        ``rest_static_reference`` is set. Costs four multislice runs per
        snapshot instead of two. Drawing the rest field from the full
        thermal ensemble (bin modes included) double-counts the bin modes'
        own damping, an error of about ``2 M_bin`` in the one-phonon
        intensity, i.e. ``2 M / n_bins`` for bins of comparable weight --
        negligible for narrow bins, a few percent for a handful of bins.
    rest_static_reference : bool, optional
        If True (default False), also propagate, per rest realization, the
        rest-displaced structure without the bin displacement
        (``R_eq ± u_rest``) as a third parity member ``"static"``. It is
        the per-realization reference the multi-phonon channel subtracts;
        without it that channel would be dominated by two-rest-phonon
        fluctuations (order ``u_rest**4``), so it is only computed when
        this reference is present. Six multislice runs per snapshot, and
        :func:`~abtem.measurements.phonon_loss_diffraction_patterns` then
        returns all three channels as without rest fields. Requires
        ``rest_snapshots``.
    parity_projection : bool, optional
        If True (default False), separate one-phonon from multi-phonon
        scattering by also propagating, for every snapshot, its
        displacement-reversed twin -- see issue #373. This adds a leading
        :class:`~abtem.core.axes.PhononParityAxis` (``values=("real",
        "twin")``) to the ensemble. Requires ``equilibrium_atoms``. Since
        the whole point is to keep every individual configuration's exit
        wave (rather than only their mean), ``ensemble_mean`` is forced to
        False automatically when this is True -- averaging over frozen
        phonons before forming the parity combination would defeat it.
    ensemble_mean : bool, optional
        If True (default), average over frozen-phonon configurations.
        Ignored (forced to False) if ``parity_projection`` is True.
    cell : Cell, optional
    """

    def __init__(
        self,
        energy_resolved_snapshots: list[Sequence[Atoms]] | np.ndarray,
        energies: np.ndarray | Sequence[float],
        equilibrium_atoms: Optional[Atoms] = None,
        parity_projection: bool = False,
        max_displacement: Optional[float] = 1.0,
        rest_snapshots: Optional[list[Sequence[Atoms]] | Sequence[Atoms]] = None,
        rest_static_reference: bool = False,
        ensemble_mean: bool = True,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        cell: Optional[Cell] = None,
        _validated: bool = False,
    ):
        if rest_snapshots is not None and not parity_projection:
            raise ValueError("rest_snapshots requires parity_projection=True.")
        if (
            rest_snapshots is not None
            and isinstance(energy_resolved_snapshots, np.ndarray)
            and energy_resolved_snapshots.ndim != 2
        ):
            raise ValueError(
                "rest_snapshots can only be combined with real-configuration "
                "snapshots of shape (n_energies, n_configs); a pre-built "
                f"{energy_resolved_snapshots.ndim}D snapshot array already "
                "carries its parity (and rest) members."
            )

        if parity_projection and equilibrium_atoms is None:
            raise ValueError(
                "parity_projection=True requires equilibrium_atoms (the "
                "shared undisplaced structure every snapshot displaces "
                "from), used to build each snapshot's displacement-reversed "
                "twin."
            )

        if parity_projection:
            ensemble_mean = False

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

        if parity_projection and snapshots.ndim == 2 and rest_static_reference and rest_snapshots is None:
            # only meaningful for fresh input; a chunk reconstruction passes
            # rest_snapshots=None with the members already baked in
            raise ValueError("rest_static_reference requires rest_snapshots.")

        if parity_projection and snapshots.ndim == 2:
            # Fresh, real-configuration-only input (the public contract of
            # this constructor): build the displacement-reversed twin and
            # stack it as a new leading axis. A `snapshots.ndim >= 3` input
            # is either a dask/chunk reconstruction of an ensemble that
            # already has the parity axis (and, if used, the rest-parity
            # axis) baked in (see `_from_partition_args_func` below), or a
            # pre-built array passed directly -- never re-twinned, but the
            # latter is validated below.
            eq_positions = equilibrium_atoms.positions
            rest = _validate_rest_snapshots(
                rest_snapshots, snapshots.shape, equilibrium_atoms, max_displacement
            )
            twin = np.empty_like(snapshots)
            for index in np.ndindex(snapshots.shape):
                atoms = snapshots[index]
                _validate_parity_snapshot(
                    atoms, equilibrium_atoms, max_displacement, index
                )
                twin_atoms = atoms.copy()
                twin_atoms.positions = 2 * eq_positions - atoms.positions
                itemset(twin, index, twin_atoms)
            snapshots = np.stack([snapshots, twin], axis=0)

            if rest is not None:
                # (parity, rest sign, energy, config): R_eq + s u_bin + t u_rest
                # for the "real" (s = +1) and "twin" (s = -1) members and,
                # if rest_static_reference, a third "static" member (s = 0):
                # R_eq + t u_rest, the rest-displaced structure without the
                # bin displacement. That member is the per-realization
                # reference the multi-phonon channel subtracts, so that the
                # two-rest-phonon fluctuations (of order u_rest^4, typically
                # larger than the bin's own two-phonon signal) cancel per
                # realization instead of contaminating that channel; the
                # one-phonon channel does not need it.
                n_members = 3 if rest_static_reference else 2
                with_rest = np.empty((n_members, 2) + rest.shape, dtype=object)
                for index in np.ndindex(rest.shape):
                    u_rest = rest[index].positions - eq_positions
                    for sign_index, sign in enumerate((1.0, -1.0)):
                        for parity in range(2):
                            atoms = snapshots[(parity,) + index].copy()
                            atoms.positions = atoms.positions + sign * u_rest
                            itemset(with_rest, (parity, sign_index) + index, atoms)
                        if rest_static_reference:
                            static = snapshots[(0,) + index].copy()
                            static.positions = eq_positions + sign * u_rest
                            itemset(with_rest, (2, sign_index) + index, static)
                snapshots = with_rest
        elif parity_projection and snapshots.ndim >= 3 and not _validated:
            # A `snapshots.ndim >= 3` input with `_validated=True` is a
            # dask/chunk reconstruction of an ensemble that was already
            # validated once (see `_from_partitioned_args`/
            # `_from_partition_args_func` below, which forward `_validated`
            # via `_copy_kwargs`) -- re-checking every chunk would be
            # redundant. `_validated=False` (the default for anyone calling
            # this constructor directly, which is not the documented way to
            # build a parity_projection ensemble but is not prevented
            # either) means this 3D array has never been checked against
            # equilibrium_atoms, so validate both halves now.
            for index in np.ndindex(snapshots.shape):
                _validate_parity_snapshot(
                    snapshots[index], equilibrium_atoms, max_displacement, index
                )

        atoms = snapshots.ravel()[0]
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(
            atoms, None, cell
        )

        self._snapshots = snapshots
        self._energies = energies
        self._equilibrium_atoms = equilibrium_atoms
        self._parity_projection = parity_projection
        self._max_displacement = max_displacement
        self._rest_snapshots = rest_snapshots
        self._rest_static_reference = rest_static_reference
        # Every snapshot has now been checked (or checking was skipped
        # because a validated instance is being re-chunked) -- record this
        # so a downstream dask reconstruction of this instance (which
        # forwards constructor kwargs via _copy_kwargs) skips redundant
        # re-validation instead of inferring it from array rank alone.
        self._validated = True

        super().__init__(
            atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=ensemble_mean
        )

        if ensemble_axes_metadata is not None:
            self._ensemble_axes_metadata = ensemble_axes_metadata
        else:
            energy_and_config_axes = [
                EnergyLossAxis(
                    values=tuple(float(e) for e in energies),
                    units="eV",
                ),
                FrozenPhononsAxis(
                    _ensemble_mean=ensemble_mean,
                    _ensemble_mean_forced=parity_projection,
                ),
            ]
            if parity_projection:
                if snapshots.ndim == 4:
                    members = (
                        ("real", "twin", "static")
                        if snapshots.shape[0] == 3
                        else ("real", "twin")
                    )
                    leading = [
                        PhononParityAxis(values=members),
                        PhononRestParityAxis(values=("plus", "minus")),
                    ]
                else:
                    leading = [PhononParityAxis(values=("real", "twin"))]
                self._ensemble_axes_metadata = leading + energy_and_config_axes
            else:
                self._ensemble_axes_metadata = energy_and_config_axes

    @property
    def snapshots(self) -> np.ndarray:
        """Object array of Atoms, ``(n_energies, n_configs)`` normally,
        ``(2, n_energies, n_configs)`` if ``parity_projection`` is True, or
        ``(2 or 3, 2, n_energies, n_configs)`` if ``rest_snapshots`` were
        given as well (parity member real/twin[/static], rest sign, energy,
        configuration)."""
        return self._snapshots

    @property
    def energies(self) -> np.ndarray:
        """Energy values [eV] for each snapshot group."""
        return self._energies

    @property
    def equilibrium_atoms(self) -> Optional[Atoms]:
        """The shared undisplaced/equilibrium structure, if given."""
        return self._equilibrium_atoms

    @property
    def parity_projection(self) -> bool:
        """Whether this ensemble carries the displacement-reversed twin
        needed to separate one-phonon from multi-phonon scattering."""
        return self._parity_projection

    @property
    def max_displacement(self) -> Optional[float]:
        """Sanity bound [Å] on snapshot displacements from
        ``equilibrium_atoms`` (``None`` if disabled)."""
        return self._max_displacement

    @property
    def rest_snapshots(self):
        """The rest-displacement fields, if given (see the constructor)."""
        return self._rest_snapshots

    @property
    def rest_parity(self) -> bool:
        """Whether the ensemble carries the rest-parity axis (both signs of a
        rest displacement field on top of every bin snapshot)."""
        return self._snapshots.ndim == 4

    @property
    def rest_static_reference(self) -> bool:
        """Whether the rest-displaced static reference member is included
        (needed for the multi-phonon channel with rest fields)."""
        return self._rest_static_reference

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self._snapshots.shape

    @property
    def num_configs(self) -> int:
        return self._snapshots.shape[-1]

    @property
    def atoms(self) -> Atoms:
        atoms = self._snapshots.ravel()[0]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.item()
        return atoms

    def __len__(self) -> int:
        return len(self._energies)

    @staticmethod
    def _getitem_2d(snapshots_2d, energies, item):
        """Index a plain (n_energies, n_configs) snapshots array + energies
        array, collapsing whichever axis a scalar index hits the same way
        a non-parity ensemble's __getitem__ always has."""
        new_snapshots = snapshots_2d[item]
        new_energies = (
            energies[item] if not isinstance(item, tuple) else energies[item[0]]
        )
        if isinstance(new_snapshots, Atoms):
            # both axes collapsed (``ensemble[i, j]``): a single configuration
            single = np.empty((1, 1), dtype=object)
            itemset(single, (0, 0), new_snapshots)
            new_snapshots = single
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
        return new_snapshots, new_energies

    def __getitem__(self, item):
        # `ensemble_axes_metadata` is deliberately not copied: the parent's
        # EnergyLossAxis carries every energy, but the sliced ensemble has
        # fewer, so the constructor must rebuild the axes from the new
        # energies (a stale axis makes the next Potential/multislice raise
        # an ordinal-axis size mismatch).
        kwargs = self._copy_kwargs(
            exclude=("energy_resolved_snapshots", "energies", "ensemble_axes_metadata")
        )

        if self._parity_projection and self._snapshots.ndim == 4:
            raise NotImplementedError(
                "Indexing an EnergyResolvedAtomsEnsemble with rest_snapshots "
                "(parity, rest sign, energy, configuration) is not supported; "
                "slice the inputs before constructing it instead."
            )

        if self._parity_projection:
            # The (parity, energy, config) 3D layout makes the plain/2D-
            # style syntax below ambiguous (e.g. `ensemble[i]` would mean
            # "parity member i" here but "energy i" for a non-parity
            # ensemble), so only the unambiguous form -- an explicit
            # leading ':' keeping the parity axis whole -- is supported:
            # `ensemble[:, energy_slice]` or
            # `ensemble[:, energy_slice, config_slice]`.
            # isinstance guards first: comparing a numpy index array to a
            # slice would evaluate elementwise and raise on `not (...)`
            is_bare_full_slice = isinstance(item, slice) and item == slice(None)
            is_leading_full_slice = (
                isinstance(item, tuple)
                and len(item) >= 1
                and isinstance(item[0], slice)
                and item[0] == slice(None)
            )
            if not (is_bare_full_slice or is_leading_full_slice):
                raise NotImplementedError(
                    "Indexing a parity_projection=True "
                    "EnergyResolvedAtomsEnsemble requires an explicit "
                    "leading ':' to keep the parity axis whole, e.g. "
                    "ensemble[:, energy_slice] or "
                    "ensemble[:, energy_slice, config_slice]. Slice "
                    "energy_resolved_snapshots/energies before constructing "
                    "it instead if you need something else."
                )
            if is_bare_full_slice:
                sub_item = slice(None)
            else:
                sub_item = item[1:]
                if len(sub_item) == 0:
                    sub_item = slice(None)
                elif len(sub_item) == 1:
                    sub_item = sub_item[0]

            real_snapshots, new_energies = self._getitem_2d(
                self._snapshots[0], self._energies, sub_item
            )
            twin_snapshots, _ = self._getitem_2d(
                self._snapshots[1], self._energies, sub_item
            )
            new_snapshots = np.stack([real_snapshots, twin_snapshots], axis=0)
            # Both halves are still an untouched subset of a previously-
            # validated real/twin pairing -- selecting a subset cannot
            # introduce a mismatch, so re-validation would be redundant.
            kwargs["_validated"] = True
            return EnergyResolvedAtomsEnsemble(
                new_snapshots, new_energies, **kwargs
            )

        new_snapshots, new_energies = self._getitem_2d(
            self._snapshots, self._energies, item
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
                energies_chunk = self._energies[slic[-2]]
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
                energies_chunk = self._energies[slic[-2]]
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
            exclude=("energy_resolved_snapshots", "energies", "rest_snapshots")
        )
        # the chunk snapshots already contain the rest-displaced members;
        # only the fresh (2D) constructor path consumes rest_snapshots
        kwargs["rest_snapshots"] = None
        kwargs["cell"] = self.cell.array
        kwargs["ensemble_axes_metadata"] = [UnknownAxis()] * len(
            self.ensemble_shape
        )
        return partial(self._from_partition_args_func, **kwargs)
