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

from abtem.potentials.base import (
    BasePotential,
    PotentialArray,
    FieldArray,
    FieldBuilder,
    _validate_frozen_phonons
)

if TYPE_CHECKING:
    from abtem.integrals import FieldIntegrator
    from abtem.parametrizations import Parametrization
    from abtem.waves import BaseWaves, Waves


class _FieldBuilderFromAtoms(FieldBuilder):
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
        self._sliced_atoms: Optional[BaseSlicedAtoms] = None
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
        """The integrator determining how the projection integrals for each slice is
        calculated."""
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
        The atoms used in the multislice algorithm, transformed to the given plane,
        origin and box.

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

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = (1,)

        return self.frozen_phonons._partition_args(chunks, lazy=lazy)


class Potential(_FieldBuilderFromAtoms, BasePotential):
    """
    Calculate the electrostatic potential of a set of atoms or frozen phonon
    configurations. The potential is calculated with the Independent Atom Model (IAM)
    using a user-defined parametrization of the atomic potentials.

    Parameters
    ----------
    atoms : ase.Atoms or abtem.FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the
        independent atom model for calculating the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential.
        Provide either "sampling" (spacing between consecutive grid points) or "gpts"
        (total number of grid points).
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [Å].
        Provide either "sampling" or "gpts".
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å]
        (default is 1 Å).
        If given as a float, the number of slices is calculated by dividing the slice
        thickness into the `z`-height of supercell. The slice thickness may be given as
        a sequence of values for each slice, in which case an error will be thrown if
        the sum of slice thicknesses is not equal to the height of the atoms.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential
        for each element. Two of the most accurate parametrizations are available
        (by Lobato et al. and Kirkland; default is 'lobato').
        See the citation guide for references.
    projection : 'finite' or 'infinite', optional
        If 'finite' the 3D potential is numerically integrated between the slice
        boundaries. If 'infinite' (default), the infinite potential projection of each
        atom will be assigned to a single slice.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential,
        i.e. provided plane is perpendicular to the propagation direction. If string,
        it must be a concatenation of two of 'x', 'y' and 'z'; the default value 'xy'
        indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped
        to the `x` and `y` directions of the potential, respectively. The length of the
        vectors has no influence. If the vectors are not perpendicular, the second
        vector is rotated in the plane to become perpendicular to the first.
        Providing a value of ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing
        'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential.
        This is equivalent to translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined
        from the atoms' cell. If the box size does not match an integer number of the
        atoms' supercell, an affine transformation may be necessary to preserve
        periodicity, determined by the `periodic` keyword.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines
        how the atomic structure is transformed. If True, the periodicity of the Atoms
        is preserved, which may require applying a small affine transformation to the
        atoms. If False, the transformed potential is effectively cut out of a larger
        repeated potential, which may not preserve periodicity.
    integrator : ProjectionIntegrator, optional
        Provide a custom integrator for the projection integrals of the potential
        slicing.
    device : str, optional
        The device used for calculating the potential, 'cpu' or 'gpu'. The default is
        determined by the user configuration file.
    """

    _exclude_from_copy = ("parametrization", "projection")

    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons,
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

        @property
        @abstractmethod
        def base_axes_metadata(self):
            pass


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