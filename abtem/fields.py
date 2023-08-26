"""Module for describing electrostatic potentials using the independent atom model."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial, reduce
from numbers import Number
from operator import mul
from typing import Type

import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.array import _validate_lazy, ArrayObject
from abtem.atoms import (
    is_cell_orthogonal,
    orthogonalize_cell,
    best_orthogonal_cell,
    cut_cell,
    pad_atoms,
    plane_to_axes,
    rotate_atoms_to_plane,
)
from abtem.core.axes import (
    ThicknessAxis,
    FrozenPhononsAxis,
    AxisMetadata,
    _find_axes_type,
)
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import generate_chunks, Chunks
from abtem.core.chunks import validate_chunks
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args, Ensemble
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.utils import EqualityMixin, CopyMixin
from abtem.inelastic.phonons import (
    BaseFrozenPhonons,
    AtomsEnsemble,
    DummyFrozenPhonons,
)
from abtem.measurements import Images
from abtem.slicing import (
    _validate_slice_thickness,
    SliceIndexedAtoms,
    SlicedAtoms,
    BaseSlicedAtoms,
)


class BaseField(Ensemble, HasGridMixin, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    @property
    @abstractmethod
    def base_shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def num_configurations(self):
        """Number of frozen phonons in the ensemble of potentials."""
        pass

    @property
    @abstractmethod
    def base_axes_metadata(self):
        pass

    def _get_exit_planes_axes_metadata(self):
        return ThicknessAxis(label="z", values=tuple(self.exit_thicknesses))

    @property
    @abstractmethod
    def exit_planes(self) -> tuple[int]:
        """The "exit planes" of the potential. The indices of slices where a measurement is returned."""
        pass

    @property
    def _exit_plane_after(self):
        exit_plane_index = 0
        exit_planes = self.exit_planes

        if exit_planes[0] == -1:
            exit_plane_index += 1

        is_exit_plane = np.zeros(len(self), dtype=bool)
        for i in range(len(is_exit_plane)):
            if i == exit_planes[exit_plane_index]:
                is_exit_plane[i] = True
                exit_plane_index += 1

        return is_exit_plane

    @property
    def exit_thicknesses(self) -> tuple[float]:
        """The "exit thicknesses" of the potential. The thicknesses in the potential where a measurement is returned."""
        thicknesses = np.cumsum(self.slice_thickness)

        if self.exit_planes[0] == -1:

            return tuple(
                np.insert(
                    thicknesses[np.array(self.exit_planes[1:], dtype=int)], 0, 0.0
                )
            )
        else:
            return tuple(thicknesses[np.array(self.exit_planes, dtype=int)])

    @property
    def num_exit_planes(self) -> int:
        """Number of exit planes."""
        return len(self.exit_planes)

    @abstractmethod
    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        pass

    @abstractmethod
    def build(
        self,
        first_slice: int = 0,
        last_slice: int = None,
        chunks: int = 1,
        lazy: bool = None,
    ):
        pass

    def __len__(self) -> int:
        return self.num_slices

    @property
    def num_slices(self) -> int:
        """Number of projected potential slices."""
        return len(self.slice_thickness)

    @property
    @abstractmethod
    def slice_thickness(self) -> np.ndarray:
        """Slice thicknesses for each slice."""
        pass

    @property
    def slice_limits(self) -> list[tuple[float, float]]:
        """The entrance and exit thicknesses of each slice [Å]."""
        cumulative_thickness = np.cumsum(np.concatenate(((0,), self.slice_thickness)))
        return [
            (cumulative_thickness[i], cumulative_thickness[i + 1])
            for i in range(len(cumulative_thickness) - 1)
        ]

    @property
    def thickness(self) -> float:
        """Thickness of the potential [Å]."""
        return sum(self.slice_thickness)

    def __iter__(self):
        for slic in self.generate_slices():
            yield slic

    def project(self) -> Images:
        """
        Sum of the potential slices as an image.

        Returns
        -------
        projected : Images
            The projected potential.
        """
        return self.build().project()

    @property
    def _default_ensemble_chunks(self) -> tuple:
        return validate_chunks(self.ensemble_shape, (1,) * len(self.ensemble_shape))

    def to_images(self):
        """
        Converts the potential to an ensemble of images.

        Returns
        -------
        image_ensemble : Images
            The potential slices as images.
        """
        return self.build().complex_images()

    def show(self, project: bool = True, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        project : bool, optional
            Show the projected potential (True, default) or show all potential slices. It is recommended to index a
            subset of the potential slices when this keyword set to False.
        kwargs :
            Additional keyword arguments for the show method of :class:`.Images`.
        """
        if project:
            return self.project().show(**kwargs)
        else:
            if "explode" not in kwargs.keys():
                kwargs["explode"] = True

            return self.to_images().show(**kwargs)


def _require_cell_transform(cell, box, plane, origin):

    if box == tuple(np.diag(cell)):
        return False

    if not is_cell_orthogonal(cell):
        return True

    if box is not None:
        return True

    if plane != "xy":
        return True

    if origin != (0.0, 0.0, 0.0):
        return True

    return False


def _validate_exit_planes(exit_planes, num_slices):
    if isinstance(exit_planes, int):
        if exit_planes >= num_slices:
            return (num_slices - 1,)

        exit_planes = list(range(exit_planes - 1, num_slices, exit_planes))
        if exit_planes[-1] != (num_slices - 1):
            exit_planes.append(num_slices - 1)
        exit_planes = (-1,) + tuple(exit_planes)
    elif exit_planes is None:
        exit_planes = (num_slices - 1,)

    return exit_planes


def _validate_frozen_phonons(atoms):
    if isinstance(atoms, Atoms):
        atoms = atoms.copy()
        atoms.calc = None

    if not hasattr(atoms, "randomize"):
        if isinstance(atoms, (list, tuple)):
            frozen_phonons = AtomsEnsemble(atoms)
        elif isinstance(atoms, Atoms):
            frozen_phonons = DummyFrozenPhonons(atoms)
        else:
            raise ValueError()
    else:
        frozen_phonons = atoms

    return frozen_phonons


class _FieldBuilder(BaseField):
    def __init__(
        self,
        array_object: Type[FieldArray],
        slice_thickness: float | tuple[float, ...],
        exit_planes: int | tuple[int, ...],
        cell: np.ndarray | Cell,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        box: tuple[float, float, float] = None,
        plane: str
        | tuple[tuple[float, float, float], tuple[float, float, float]] = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        periodic: bool = True,
        device: str = None,
    ):
        self._array_object = array_object
        if _require_cell_transform(cell, box=box, plane=plane, origin=origin):
            axes = plane_to_axes(plane)
            cell = cell[:, list(axes)]
            box = tuple(best_orthogonal_cell(cell))

        elif box is None:
            box = tuple(np.diag(cell))

        self._grid = Grid(
            extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True
        )
        self._device = validate_device(device)

        self._box = box
        self._plane = plane
        self._origin = origin
        self._periodic = periodic

        self._slice_thickness = _validate_slice_thickness(
            slice_thickness, thickness=box[2]
        )
        self._exit_planes = _validate_exit_planes(
            exit_planes, len(self._slice_thickness)
        )

    @property
    def slice_thickness(self) -> tuple[float, ...]:
        return self._slice_thickness

    @property
    def exit_planes(self) -> tuple[int]:
        return self._exit_planes

    @property
    def device(self) -> str:
        """The device where the potential is created."""
        return self._device

    @property
    def periodic(self) -> bool:
        """Specifies whether the potential is periodic."""
        return self._periodic

    @property
    def plane(self) -> str:
        """The plane relative to the atoms mapped to `xy` plane of the potential, i.e. the plane is perpendicular to the
        propagation direction."""
        return self._plane

    @property
    def box(self) -> tuple[float, float, float]:
        """The extent of the potential in `x`, `y` and `z`."""
        return self._box

    @property
    def origin(self) -> tuple[float, float, float]:
        """The origin relative to the provided atoms mapped to the origin of the potential."""
        return self._origin

    def __getitem__(self, item) -> FieldArray:
        return self.build(lazy=False)[item]

    @staticmethod
    def _wrap_build_potential(potential, first_slice, last_slice):
        potential = potential.item()
        array = potential.build(first_slice, last_slice, lazy=False).array
        return array

    def build(
        self,
        first_slice: int = 0,
        last_slice: int = None,
        max_batch: int | str = 1,
        lazy: bool = None,
    ) -> FieldArray:
        """
        Build the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential
        max_batch : int or str, optional
            Maximum number of slices to calculate in task. Default is 1.
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If None, this defaults to the
            value set in the configuration file.

        Returns
        -------
        potential_array : PotentialArray
            The built potential as an array.
        """

        lazy = _validate_lazy(lazy)

        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if lazy:
            blocks = self.ensemble_blocks(self._default_ensemble_chunks)

            xp = get_array_module(self.device)
            chunks = validate_chunks(self.ensemble_shape, self._default_ensemble_chunks)
            chunks = chunks + self.base_shape

            if self.ensemble_shape:
                new_axis = tuple(
                    range(
                        len(self.ensemble_shape),
                        len(self.ensemble_shape) + len(self.base_shape),
                    )
                )
            else:
                new_axis = tuple(range(1, len(self.base_shape)))

            array = blocks.map_blocks(
                self._wrap_build_potential,
                new_axis=new_axis,
                first_slice=first_slice,
                last_slice=last_slice,
                chunks=chunks,
                meta=xp.array((), dtype=np.float32),
            )

        else:
            xp = get_array_module(self.device)

            array = xp.zeros(
                self.ensemble_shape + (last_slice - first_slice,) + self.base_shape[1:],
                dtype=xp.float32,
            )

            if self.ensemble_shape:
                for i, _, potential in self.generate_blocks(1):
                    potential = potential.item()
                    i = np.unravel_index((0,), self.ensemble_shape)

                    for j, slic in enumerate(
                        potential.generate_slices(first_slice, last_slice)
                    ):

                        array[i + (j,)] = slic.array[0]

            else:
                for j, slic in enumerate(self.generate_slices(first_slice, last_slice)):
                    array[j] = slic.array[0]

        potential = self._array_object(
            array,
            sampling=(self.sampling[0], self.sampling[1]),
            slice_thickness=self.slice_thickness[first_slice:last_slice],
            exit_planes=self.exit_planes,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
        )
        return potential


class _FieldBuilderFromAtoms(_FieldBuilder):
    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons,
        array_object: Type[FieldArray],
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        slice_thickness: float | tuple[float, ...] = 1,
        exit_planes: int | tuple[int, ...] = None,
        plane: str
        | tuple[tuple[float, float, float], tuple[float, float, float]] = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: tuple[float, float, float] = None,
        periodic: bool = True,
        integrator=None,
        device: str = None,
    ):

        self._frozen_phonons = _validate_frozen_phonons(atoms)
        self._integrator = integrator
        self._sliced_atoms = None
        self._array_object = array_object

        super().__init__(
            array_object=array_object,
            gpts=gpts,
            sampling=sampling,
            cell=self._frozen_phonons.cell,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
        )

    @property
    def frozen_phonons(self) -> BaseFrozenPhonons:
        """Ensemble of atomic configurations representing frozen phonons."""
        return self._frozen_phonons

    @property
    def num_configurations(self) -> int:
        """Size of the ensemble of atomic configurations representing frozen phonons."""
        return len(self.frozen_phonons)

    @property
    def integrator(self):
        """The integrator determining how the projection integrals for each slice is calculated."""
        return self._integrator

    def _cutoffs(self):
        atoms = self.frozen_phonons.atoms
        unique_numbers = np.unique(atoms.numbers)
        return tuple(
            self._integrator.cutoff(chemical_symbols[number])
            for number in unique_numbers
        )

    def get_transformed_atoms(self):
        """
        The atoms used in the multislice algorithm, transformed to the given plane, origin and box.

        Returns
        -------
        transformed_atoms : Atoms
            Transformed atoms.
        """
        atoms = self.frozen_phonons.atoms

        if is_cell_orthogonal(atoms.cell) and self.plane != "xy":
            atoms = rotate_atoms_to_plane(atoms, self.plane)

        elif tuple(np.diag(atoms.cell)) != self.box:
            if self.periodic:
                atoms = orthogonalize_cell(
                    atoms,
                    box=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    return_transform=False,
                    allow_transform=True,
                )
                return atoms
            else:
                cutoffs = self._cutoffs()
                atoms = cut_cell(
                    atoms,
                    cell=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    margin=max(cutoffs) if cutoffs else 0.0,
                )

        return atoms

    def _prepare_atoms(self):

        atoms = self.get_transformed_atoms()

        if self.integrator.finite:
            cutoffs = self._cutoffs()
            margins = max(cutoffs) if len(cutoffs) else 0.0
        else:
            margins = 0.0

        if self.periodic:
            atoms = self.frozen_phonons.randomize(atoms)
            atoms.wrap()

        if not self.integrator.periodic and self.integrator.finite:
            atoms = pad_atoms(atoms, margins=margins)
        elif self.integrator.periodic:
            atoms = pad_atoms(atoms, margins=margins, directions="z")

        if not self.periodic:
            atoms = self.frozen_phonons.randomize(atoms)

        if self.integrator.finite:
            sliced_atoms = SlicedAtoms(
                atoms=atoms, slice_thickness=self.slice_thickness, z_padding=margins
            )
        else:
            sliced_atoms = SliceIndexedAtoms(
                atoms=atoms, slice_thickness=self.slice_thickness
            )

        return sliced_atoms

    def get_sliced_atoms(self) -> BaseSlicedAtoms:
        """
        The atoms grouped into the slices given by the slice thicknesses.

        Returns
        -------
        sliced_atoms : BaseSlicedAtoms
        """
        if self._sliced_atoms is not None:
            return self._sliced_atoms

        self._sliced_atoms = self._prepare_atoms()

        return self._sliced_atoms

    def generate_slices(
        self, first_slice: int = 0, last_slice: int = None, return_depth: float = False
    ):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        return_depth : bool
            If True, return the depth of each generated slice.

        Yields
        ------
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        xp = get_array_module(self.device)

        sliced_atoms = self.get_sliced_atoms()

        numbers = np.unique(sliced_atoms.atoms.numbers)

        exit_plane_after = self._exit_plane_after

        cumulative_thickness = np.cumsum(self.slice_thickness)

        for start, stop in generate_chunks(
            last_slice - first_slice, chunks=1, start=first_slice
        ):

            if len(numbers) > 1 or stop - start > 1:
                array = xp.zeros(
                    (stop - start,) + self.base_shape[1:], dtype=np.float32
                )
            else:
                array = None

            for i, slice_idx in enumerate(range(start, stop)):

                # for Z, integrator in integrators.items():
                atoms = sliced_atoms.get_atoms_in_slices(slice_idx)

                new_array = self._integrator.integrate_on_grid(
                    atoms,
                    a=sliced_atoms.slice_limits[slice_idx][0],
                    b=sliced_atoms.slice_limits[slice_idx][1],
                    gpts=self.gpts,
                    sampling=self.sampling,
                    device=self.device,
                )

                if array is not None:
                    array[i] += new_array
                else:
                    array = new_array[None]

            if array is None:
                array = xp.zeros(
                    (stop - start,) + self.base_shape[1:], dtype=np.float32
                )

            # array -= array.min()

            exit_planes = tuple(np.where(exit_plane_after[start:stop])[0])

            potential_array = self._array_object(
                array,
                slice_thickness=self.slice_thickness[start:stop],
                exit_planes=exit_planes,
                extent=self.extent,
            )

            if return_depth:
                depth = cumulative_thickness[stop - 1]
                yield depth, potential_array
            else:
                yield potential_array

    @property
    def ensemble_axes_metadata(self):
        return self.frozen_phonons.ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self.frozen_phonons.ensemble_shape

    @classmethod
    def _from_partitioned_args_func(cls, *args, frozen_phonons_partial, **kwargs):
        args = unpack_blockwise_args(args)

        frozen_phonons = frozen_phonons_partial(*args)
        frozen_phonons = frozen_phonons.item()

        new_potential = cls(frozen_phonons, **kwargs)

        ndims = max(len(new_potential.ensemble_shape), 1)
        new_potential = _wrap_with_array(new_potential, ndims)
        return new_potential

    def _from_partitioned_args(self, *args, **kwargs):
        frozen_phonons_partial = self.frozen_phonons._from_partitioned_args()
        kwargs = self._copy_kwargs(exclude=("atoms", "sampling"))

        return partial(
            self._from_partitioned_args_func,
            frozen_phonons_partial=frozen_phonons_partial,
            **kwargs,
        )

    def _partition_args(self, chunks: Chunks = (1,), lazy: bool = True):
        return self.frozen_phonons._partition_args(chunks, lazy=lazy)


class FieldArray(BaseField, ArrayObject):
    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: float | tuple[float, ...] = None,
        extent: float | tuple[float, float] = None,
        sampling: float | tuple[float, float] = None,
        exit_planes: int | tuple[int, ...] = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        self._slice_thickness = _validate_slice_thickness(
            slice_thickness, num_slices=array.shape[-self._base_dims]
        )

        self._exit_planes = _validate_exit_planes(
            exit_planes, len(self._slice_thickness)
        )
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling)

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def num_configurations(self):
        indices = _find_axes_type(self, FrozenPhononsAxis)
        if indices:
            return reduce(mul, tuple(self.array.shape[i] for i in indices))
        else:
            return 1

    @property
    def slice_thickness(self) -> tuple[float, ...]:
        return self._slice_thickness

    @property
    def exit_planes(self) -> tuple[int, ...]:
        return self._exit_planes

    def build(
        self,
        first_slice: int = 0,
        last_slice: int = None,
        chunks: int = 1,
        lazy: bool = None,
    ):
        raise RuntimeError("potential is already built")

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.

        Yields
        ------
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        exit_plane_after = self._exit_plane_after
        cum_thickness = np.cumsum(self.slice_thickness)
        start = first_slice
        stop = first_slice + 1

        for i in range(first_slice, last_slice):
            s = (0,) * (len(self.array.shape) - 3) + (i,)
            array = self.array[s][None]

            slic = self.__class__(
                array, self.slice_thickness[i : i + 1], extent=self.extent
            )

            exit_planes = tuple(np.where(exit_plane_after[start:stop])[0])

            slic._exit_planes = exit_planes

            start += 1
            stop += 1

            yield slic

    def __getitem__(self, items):
        if isinstance(items, (Number, slice)):
            items = (items,)

        ensemble_items = items[: len(self.ensemble_shape)]
        slic_items = items[len(self.ensemble_shape) :]

        if len(ensemble_items):
            potential_array = super().__getitem__(ensemble_items)
        else:
            potential_array = self

        if len(slic_items) == 0:
            return potential_array

        padded_items = (slice(None),) * len(potential_array.ensemble_shape) + slic_items

        array = potential_array._array[padded_items]
        slice_thickness = np.array(potential_array.slice_thickness)[slic_items]

        if len(array.shape) < len(potential_array.shape):
            array = array[
                (slice(None),) * len(potential_array.ensemble_shape) + (None,)
            ]
            slice_thickness = slice_thickness[None]

        kwargs = potential_array._copy_kwargs(exclude=("array", "slice_thickness"))
        kwargs["array"] = array
        kwargs["slice_thickness"] = slice_thickness
        kwargs["sampling"] = None
        return potential_array.__class__(**kwargs)

    def tile(self, repetitions: tuple[int, int] | tuple[int, int, int]):
        """
        Tile the potential.

        Parameters
        ----------
        repetitions: two or three int
            The number of repetitions of the potential along each axis. NOTE: if three integers are given, the first
            represents the number of repetitions along the `z`-axis.

        Returns
        -------
        PotentialArray object
            The tiled potential.
        """
        if len(repetitions) == 2:
            repetitions = tuple(repetitions) + (1,)

        new_array = np.tile(
            self.array, (repetitions[2], repetitions[0], repetitions[1])
        )

        new_extent = (self.extent[0] * repetitions[0], self.extent[1] * repetitions[1])
        new_slice_thickness = tuple(np.tile(self.slice_thickness, repetitions[2]))

        return self.__class__(
            array=new_array,
            slice_thickness=new_slice_thickness,
            extent=new_extent,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
        )

    def to_hyperspy(self):
        return self.to_images().to_hyperspy()

    def to_images(self):
        """Convert slices of the potential to a stack of images."""
        metadata = {"label": "potential", "units": "eV / e"}

        return Images(
            array=self._array,
            sampling=(self.sampling[0], self.sampling[1]),
            metadata=metadata,
            ensemble_axes_metadata=self.axes_metadata[:-2],
        )

    def project(self) -> Images:
        """
        Create a 2D array representing a projected image of the potential(s).

        Returns
        -------
        images : Images
            One or more images of the projected potential(s).
        """
        metadata = {"label": "potential", "units": "eV / e"}
        array = self.array.sum(-self._base_dims)
        # array -= array.min((-2, -1), keepdims=True)

        ensemble_axes_metadata = (
            self.ensemble_axes_metadata + self.base_axes_metadata[1:-2]
        )

        return Images(
            array=array,
            sampling=self.sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )
