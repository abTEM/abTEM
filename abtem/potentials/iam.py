"""Module for describing electrostatic potentials using the independent atom model."""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from functools import partial, reduce
from numbers import Number
from operator import mul
from typing import TYPE_CHECKING, Optional, Sequence, Type

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.array import ArrayObject, _validate_lazy
from abtem.atoms import (
    best_orthogonal_cell,
    cut_cell,
    is_cell_orthogonal,
    orthogonalize_cell,
    pad_atoms,
    plane_to_axes,
    rotate_atoms_to_plane,
)
from abtem.core.axes import (
    AxisMetadata,
    FrozenPhononsAxis,
    RealSpaceAxis,
    ThicknessAxis,
    _find_axes_type,
)
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import Chunks, chunk_ranges, generate_chunks, validate_chunks
from abtem.core.complex import complex_exponential
from abtem.core.energy import Accelerator, HasAcceleratorMixin, energy2sigma
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.grid import Grid, HasGrid2DMixin
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype, itemset
from abtem.inelastic.phonons import (
    AtomsEnsemble,
    BaseFrozenPhonons,
    DummyFrozenPhonons,
    _validate_seeds,
)
from abtem.integrals import (
    QuadratureProjectionIntegrals,
    ScatteringFactorProjectionIntegrals,
)
from abtem.measurements import Images
from abtem.slicing import (
    BaseSlicedAtoms,
    SlicedAtoms,
    SliceIndexedAtoms,
    _validate_slice_thickness,
    slice_limits,
)

if TYPE_CHECKING:
    from abtem.integrals import FieldIntegrator
    from abtem.parametrizations import Parametrization
    from abtem.waves import BaseWaves, Waves


class BaseField(Ensemble, HasGrid2DMixin, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    # @property
    # @abstractmethod
    # def device(self) -> str:
    #     pass
    device: str

    @property
    def base_shape(self):
        """Shape of the base axes of the potential."""
        return (self.num_slices,) + self.gpts

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
    def exit_thicknesses(self) -> tuple[float, ...]:
        """The "exit thicknesses" of the potential. The thicknesses in the potential where a measurement is returned."""
        thicknesses = np.cumsum(self.slice_thickness)
        exit_indices = np.array(self.exit_planes, dtype=int)
        exit_thicknesses = tuple(thicknesses[i] for i in exit_indices)
        if self.exit_planes[0] == -1:
            return (0.0,) + exit_thicknesses[1:]
        else:
            return exit_thicknesses

    @property
    def num_exit_planes(self) -> int:
        """Number of exit planes."""
        return len(self.exit_planes)

    @abstractmethod
    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
        pass

    @abstractmethod
    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunks: int = 1,
        lazy: Optional[bool] = None,
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
    def slice_thickness(self) -> tuple[float, ...]:
        """Slice thicknesses for each slice."""
        pass

    @property
    def slice_limits(self) -> list[tuple[float, float]]:
        """The entrance and exit thicknesses of each slice [Å]."""
        return slice_limits(self.slice_thickness)

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
        return self.build().to_images()

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


class BasePotential(BaseField, metaclass=ABCMeta):
    """Base class of all potentials. Documented in the subclasses."""

    @property
    def base_axes_metadata(self):
        """List of AxisMetadata for the base axes."""
        return [
            ThicknessAxis(
                label="z", values=tuple(np.cumsum(self.slice_thickness)), units="Å"
            ),
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="Å", endpoint=False
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="Å", endpoint=False
            ),
        ]


def validate_potential(
    potential: Atoms | BasePotential, waves: Optional[BaseWaves] = None
) -> BasePotential:
    if isinstance(potential, (Atoms, BaseFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves.device

        potential = Potential(potential, device=device)
    # elif not isinstance(potential, BasePotential):
    #    raise ValueError()

    if waves is not None and potential is not None:
        potential.grid.match(waves)

    return potential


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


class _FieldBuilder(BaseField):
    def __init__(
        self,
        array_object: Type[FieldArray],
        slice_thickness: float | tuple[float, ...],
        cell: np.ndarray | Cell,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        box: Optional[tuple[float, float, float]] = None,
        plane: (
            str | tuple[tuple[float, float, float], tuple[float, float, float]]
        ) = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        periodic: bool = True,
        device: Optional[str] = None,
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
    def plane(
        self,
    ) -> str | tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The plane relative to the atoms mapped to `xy` plane of the potential,
        i.e. the plane is perpendicular to the propagation direction."""
        return self._plane

    @property
    def box(self) -> tuple[float, float, float]:
        """The extent of the potential in `x`, `y` and `z`."""
        return self._box

    @property
    def origin(self) -> tuple[float, float, float]:
        """The origin relative to the provided atoms mapped to the origin of the
        potential."""
        return self._origin

    def __getitem__(self, item) -> PotentialArray:
        return self.build(lazy=False)[item]

    @staticmethod
    def _wrap_build_potential(potential, first_slice, last_slice):
        potential = potential.item()
        array = potential.build(first_slice, last_slice, lazy=False).array
        return array

    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        max_batch: int | str = 1,
        lazy: Optional[bool] = None,
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
            If True, create the wave functions lazily, otherwise, calculate instantly.
            If None, this defaults to the value set in the configuration file.

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

            array = da.map_blocks(
                self._wrap_build_potential,
                blocks,
                new_axis=new_axis,
                first_slice=first_slice,
                last_slice=last_slice,
                chunks=chunks,
                meta=xp.array((), dtype=get_dtype(complex=False)),
            )

        else:
            xp = get_array_module(self.device)

            array = xp.zeros(
                self.ensemble_shape + (last_slice - first_slice,) + self.base_shape[1:],
                dtype=get_dtype(complex=False),
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
            sampling=self._valid_sampling,
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
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: (
            str | tuple[tuple[float, float, float], tuple[float, float, float]]
        ) = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        integrator=None,
        device: Optional[str] = None,
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
            atoms.wrap(eps=0.0)

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
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        return_depth: float = False,
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
                    (stop - start,) + self.base_shape[1:],
                    dtype=get_dtype(complex=False),
                )
            else:
                array = None

            for i, slice_idx in enumerate(range(start, stop)):
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
                    (stop - start,) + self.base_shape[1:],
                    dtype=get_dtype(complex=False),
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


class _PotentialBuilder(_FieldBuilder, BasePotential):
    pass


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
            raise ValueError(
                f"Frozen phonons should be of types `FrozenPhonons`, `Atoms` or `AtomsEnsemble`, "
                f"not {atoms}"
            )
    else:
        frozen_phonons = atoms

    return frozen_phonons


class Potential(_FieldBuilderFromAtoms, BasePotential):
    """
    Calculate the electrostatic potential of a set of atoms or frozen phonon configurations. The potential is calculated
    with the Independent Atom Model (IAM) using a user-defined parametrization of the atomic potentials.

    Parameters
    ----------
    atoms : ase.Atoms or abtem.FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the independent atom model for calculating
        the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential. Provide either "sampling" (spacing
        between consecutive grid points) or "gpts" (total number of grid points).
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [Å]. Provide either "sampling" or "gpts".
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å] (default is 0.5 Å).
        If given as a float, the number of slices is calculated by dividing the slice thickness into the `z`-height of
        supercell. The slice thickness may be given as a sequence of values for each slice, in which case an error will
        be thrown if the sum of slice thicknesses is not equal to the height of the atoms.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available (by Lobato et al. and Kirkland; default is 'lobato').
        See the citation guide for references.
    projection : 'finite' or 'infinite', optional
        If 'finite' the 3D potential is numerically integrated between the slice boundaries. If 'infinite' (default),
        the infinite potential projection of each atom will be assigned to a single slice.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If string, it must be a concatenation of two of 'x', 'y' and 'z';
        the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential. This is equivalent to
        translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined from the atoms' cell.
        If the box size does not match an integer number of the atoms' supercell, an affine transformation may be
        necessary to preserve periodicity, determined by the `periodic` keyword.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines how the atomic structure is
        transformed. If True, the periodicity of the Atoms is preserved, which may require applying a small affine
        transformation to the atoms. If False, the transformed potential is effectively cut out of a larger repeated
        potential, which may not preserve periodicity.
    integrator : ProjectionIntegrator, optional
        Provide a custom integrator for the projection integrals of the potential slicing.
    device : str, optional
        The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
        configuration file.
    """

    _exclude_from_copy = ("parametrization", "projection")

    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons | None = None,
        gpts: int | tuple[int, int] | None = None,
        sampling: float | tuple[float, float] | None = None,
        slice_thickness: float | tuple[float, ...] = 1,
        parametrization: str | Parametrization = "lobato",
        projection: str = "infinite",
        exit_planes: int | tuple[int, ...] | None = None,
        plane: (
            str | tuple[tuple[float, float, float], tuple[float, float, float]]
        ) = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: tuple[float, float, float] | None = None,
        periodic: bool = True,
        integrator: FieldIntegrator | None = None,
        device: str | None = None,
    ):
        if integrator is None:
            if projection == "finite":
                integrator = QuadratureProjectionIntegrals(
                    parametrization=parametrization
                )
            elif projection == "infinite":
                integrator = ScatteringFactorProjectionIntegrals(
                    parametrization=parametrization
                )
            else:
                raise NotImplementedError

        super().__init__(
            atoms=atoms,
            array_object=PotentialArray,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
            integrator=integrator,
        )


class FieldArray(BaseField, ArrayObject):
    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: Optional[float | tuple[float, ...]] = None,
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        # assert len(array.shape) == self._base_dims

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
        last_slice: Optional[int] = None,
        chunks: int = 1,
        lazy: Optional[bool] = None,
    ):
        raise RuntimeError("potential is already built")

    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
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
        # cum_thickness = np.cumsum(self.slice_thickness)
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

        if not len(items) <= len(self.ensemble_shape) + 1:
            raise IndexError(
                f"Too many indices for potential array with {len(self.ensemble_shape)}"
                "ensemble axes. Only slice indices and ensemble indices are allowed."
            )

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

    def to_hyperspy(self, transpose: bool = True):
        return self.to_images().to_hyperspy(transpose=transpose)

    def to_images(self):
        """Convert slices of the potential to a stack of images."""
        return Images(
            array=self._array,
            sampling=(self.sampling[0], self.sampling[1]),
            metadata=self.metadata,
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
        array = self.array.sum(-self._base_dims)
        # array -= array.min((-2, -1), keepdims=True)

        ensemble_axes_metadata = (
            self.ensemble_axes_metadata + self.base_axes_metadata[1:-2]
        )

        return Images(
            array=array,
            sampling=self._valid_sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=self.metadata,
        )


class PotentialArray(BasePotential, FieldArray):
    """
    The potential array represents slices of the electrostatic potential as an array. All other potentials build
    potential arrays.

    Parameters
    ----------
    array: 3D np.ndarray
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thickness: float
        The thicknesses of potential slices [Å]. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
    extent: one or two float, optional
        Lateral extent of the potential [Å].
    sampling: one or two float, optional
        Lateral sampling of the potential [1 / Å].
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
    """

    _base_dims = 3

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: Optional[float | tuple[float, ...]] = None,
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        if metadata is None:
            metadata = {}
        metadata = {"label": "potential", "units": "eV / e", **metadata}

        super().__init__(
            array=array,
            slice_thickness=slice_thickness,
            extent=extent,
            sampling=sampling,
            exit_planes=exit_planes,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @staticmethod
    def _transmission_function(array, energy):
        xp = get_array_module(array)
        sigma = xp.array(energy2sigma(energy), dtype=get_dtype())
        array = complex_exponential(sigma * array)
        return array

    @classmethod
    def from_array_and_metadata(
        cls: type[ArrayObject],
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ) -> ArrayObjectType:
        pass

    def transmission_function(self, energy: float) -> TransmissionFunction:
        """
        Calculate the transmission functions for each slice for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        transmissionfunction : TransmissionFunction
            Transmission functions for each slice.
        """
        xp = get_array_module(self.array)

        if self.is_lazy:
            array = da.map_blocks(
                self._transmission_function,
                self._array.map_blocks,
                energy=energy,
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )
        else:
            array = self._transmission_function(self._array, energy=energy)

        t = TransmissionFunction(
            array,
            slice_thickness=self.slice_thickness,
            extent=self.extent,
            energy=energy,
        )
        return t

    def transmit(self, waves: Waves, conjugate: bool = False) -> Waves:
        """
        Transmit a wave function through a potential slice.

        Parameters
        ----------
        waves: Waves
            Waves object to transmit.
        conjugate : bool, optional
            If True, use the conjugate of the transmission function. Default is False.

        Returns
        -------
        transmission_function : TransmissionFunction
            Transmission function for the wave function through the potential slice.
        """

        transmission_function = self.transmission_function(waves.energy)

        return transmission_function.transmit(waves, conjugate=conjugate)


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions.

    Parameters
    ----------
    array : 3D np.ndarray
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thickness : float
        The thicknesses of potential slices [Å]. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
    extent : one or two float, optional
        Lateral extent of the potential [Å].
    sampling : one or two float, optional
        Lateral sampling of the potential [1 / Å].
    energy : float
        Electron energy [eV].
    """

    def __init__(
        self,
        array: np.ndarray,
        slice_thickness: float | Sequence[float],
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
    ):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thickness, extent, sampling)

    def get_chunk(self, first_slice, last_slice) -> TransmissionFunction:
        array = self.array[first_slice:last_slice]
        if len(array.shape) == 2:
            array = array[None]
        return self.__class__(
            array,
            self.slice_thickness[first_slice:last_slice],
            extent=self.extent,
            energy=self.energy,
        )

    def transmission_function(self, energy) -> TransmissionFunction:
        """
        Calculate the transmission functions for each slice for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        transmissionfunction : TransmissionFunction
            Transmission functions for each slice.
        """
        if energy != self.energy:
            raise RuntimeError()
        return self

    def transmit(self, waves: Waves, conjugate: bool = False) -> Waves:
        """
        Transmit a wave function through a potential slice.

        Parameters
        ----------
        waves: Waves
            Waves object to transmit.
        conjugate : bool, optional
            If True, use the conjugate of the transmission function. Default is False.

        Returns
        -------
        transmission_function : Waves
            Transmission function for the wave function through the potential slice.
        """
        self.accelerator.check_match(waves)
        self.grid.check_match(waves)

        xp = get_array_module(self.array[0])

        if conjugate:
            waves._array *= xp.conjugate(self.array[0])
        else:
            waves._array *= self.array[0]

        return waves


class CrystalPotential(_PotentialBuilder):
    """
    The crystal potential may be used to represent a potential consisting of a repeating
    unit. This may allow calculations to be performed with lower computational cost by
    calculating the potential unit once and repeating it.

    If the repeating unit is a potential with frozen phonons it is treated as an
    ensemble from which each repeating unit along the `z`-direction is randomly drawn.
    If `num_frozen_phonons` an ensemble of crystal potentials are created each with a
    random seed for choosing potential units.

    Parameters
    ----------
    potential_unit : BasePotential
        The potential unit to assemble the crystal potential from.
    repetitions : three int
        The repetitions of the potential in `x`, `y` and `z`.
    num_frozen_phonons : int, optional
        Number of frozen phonon configurations assembled from the potential units.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        a measurement will be collected every `exit_planes` number of slices.
    seeds: int or sequence of int
        Seed for the random number generator (RNG), or one seed for each RNG in the
        frozen phonon ensemble.
    """

    def __init__(
        self,
        potential_unit: BasePotential,
        repetitions: tuple[int, int, int],
        num_frozen_phonons: int | None = None,
        exit_planes: int | None = None,
        seeds: int | tuple[int, ...] | None = None,
    ):
        if num_frozen_phonons is None and seeds is None:
            self._seeds = None
        else:
            if num_frozen_phonons is None and seeds:
                num_frozen_phonons = len(seeds)
            elif num_frozen_phonons is None and seeds is None:
                num_frozen_phonons = 1

            self._seeds = _validate_seeds(seeds, num_frozen_phonons)

        if (
            (potential_unit.num_configurations == 1)
            and (num_frozen_phonons is not None)
            and (num_frozen_phonons > 1)
        ):
            warnings.warn(
                "'num_frozen_phonons' is greater than one, but the potential unit does"
                "not have frozen phonons"
            )

        # if (potential_unit.num_frozen_phonons > 1) and (num_frozen_phonons is not None):
        #     warnings.warn(
        #         "the potential unit has frozen phonons, but 'num_frozen_phonons' is not set"
        #     )

        gpts = (
            potential_unit._valid_gpts[0] * repetitions[0],
            potential_unit._valid_gpts[1] * repetitions[1],
        )
        extent = (
            potential_unit._valid_extent[0] * repetitions[0],
            potential_unit._valid_extent[1] * repetitions[1],
        )

        box = extent + (potential_unit.thickness * repetitions[2],)
        slice_thickness = potential_unit.slice_thickness * repetitions[2]
        super().__init__(
            array_object=PotentialArray,
            gpts=gpts,
            cell=Cell(np.diag(box)),
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=potential_unit.device,
            plane="xy",
            origin=(0.0, 0.0, 0.0),
            box=box,
            periodic=True,
        )

        self._potential_unit = potential_unit
        self._repetitions = repetitions

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        if self._seeds is None:
            return ()
        else:
            return (self.num_configurations,)

    @property
    def num_configurations(self):
        if self._seeds is None:
            return 1
        else:
            return len(self._seeds)

    @property
    def seeds(self):
        return self._seeds

    @property
    def potential_unit(self) -> BasePotential:
        return self._potential_unit

    @HasGrid2DMixin.gpts.setter
    def gpts(self, gpts):
        if not (
            (gpts[0] % self.repetitions[0] == 0)
            and (gpts[1] % self.repetitions[0] == 0)
        ):
            raise ValueError(
                "Number of grid points must be divisible by the number of potential repetitions."
            )
        self.grid.gpts = gpts
        self._potential_unit.gpts = (
            gpts[0] // self._repetitions[0],
            gpts[1] // self._repetitions[1],
        )

    @HasGrid2DMixin.sampling.setter
    def sampling(self, sampling):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

    @property
    def repetitions(self) -> tuple[int, int, int]:
        return self._repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        if self.seeds is None:
            return []
        else:
            return [FrozenPhononsAxis(_ensemble_mean=True)]

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = unpack_blockwise_args(args)
        potential, seed = args[0]
        if hasattr(potential, "item"):
            potential = potential.item()

        if seed is not None:
            num_frozen_phonons = len(seed)
        else:
            num_frozen_phonons = None

        new = cls(
            potential_unit=potential,
            seeds=seed,
            num_frozen_phonons=num_frozen_phonons,
            **kwargs,
        )
        return _wrap_with_array(new)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("potential_unit", "seeds", "num_frozen_phonons")
        )
        output = partial(self._from_partitioned_args_func, **kwargs)
        return output

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        if chunks == ():
            chunks = ((1,),)

        if lazy:
            arrays = []

            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                lazy_atoms = dask.delayed(self.potential_unit)
                lazy_args = dask.delayed(_wrap_with_array)((lazy_atoms, seeds), ndims=1)
                lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
                arrays.append(lazy_array)

            array = da.concatenate(arrays)
        else:
            potential_unit = self.potential_unit

            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                itemset(array, i, (potential_unit, seeds))

        return (array,)

    def generate_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        return_depth: bool = False,
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
        if hasattr(self.potential_unit, "array"):
            potentials = self.potential_unit
        else:
            potentials = self.potential_unit.build(lazy=False)

        if len(potentials.shape) == 3:
            potentials = potentials.expand_dims(axis=0)

        if self.seeds is None:
            rng = np.random.default_rng(self.seeds)
        else:
            rng = np.random.default_rng(self.seeds[0])

        exit_plane_after = self._exit_plane_after
        cum_thickness = np.cumsum(self.slice_thickness)
        start = first_slice
        stop = first_slice + 1

        for i in range(self.repetitions[2]):
            generator = potentials[
                rng.integers(0, potentials.shape[0])
            ].generate_slices()

            for j in range(len(self.potential_unit)):
                slic = next(generator).tile(self.repetitions[:2])

                exit_planes = tuple(np.where(exit_plane_after[start:stop])[0])

                slic._exit_planes = exit_planes

                start += 1
                stop += 1

                if return_depth:
                    yield cum_thickness[stop - 1], slic
                else:
                    yield slic

                if j == last_slice:
                    break
