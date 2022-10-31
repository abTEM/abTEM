"""Module to calculate electrostatic potentials using the independent atom model."""
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from functools import reduce
from numbers import Number
from operator import mul
from typing import Sequence, TYPE_CHECKING
from typing import Tuple, List, Union

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.core.array import HasArray, validate_lazy
from abtem.core.axes import (
    HasAxes,
    RealSpaceAxis,
    _find_axes_type,
)
from abtem.core.axes import ThicknessAxis, FrozenPhononsAxis, AxisMetadata
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import chunk_shape, Chunks, generate_chunks
from abtem.core.chunks import validate_chunks, iterate_chunk_ranges
from abtem.core.complex import complex_exponential
from abtem.core.device import HasDeviceMixin
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.core.ensemble import Ensemble
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.integrals.base import ProjectionIntegrator
from abtem.core.integrals.gaussians import GaussianProjectionIntegrals
from abtem.core.integrals.infinite import InfinitePotentialProjections
from abtem.core.integrals.quadrature import ProjectionQuadratureRule
from abtem.core.utils import EqualityMixin, CopyMixin
from abtem.inelastic.phonons import (
    BaseFrozenPhonons,
    DummyFrozenPhonons,
    _validate_seeds,
)
from abtem.inelastic.phonons import FrozenPhonons
from abtem.measurements import Images
from abtem.slicing import (
    _validate_slice_thickness,
    SliceIndexedAtoms,
    SlicedAtoms,
    unpack_item,
)
from abtem.atoms import (
    is_cell_orthogonal,
    orthogonalize_cell,
    best_orthogonal_cell,
    cut_cell,
    rotation_matrix_from_plane,
    pad_atoms,
)

if TYPE_CHECKING:
    from abtem.waves import Waves
    from abtem.core.parametrizations.base import Parametrization


class BasePotential(
    Ensemble,
    HasAxes,
    HasGridMixin,
    EqualityMixin,
    CopyMixin,
    HasDeviceMixin,
    metaclass=ABCMeta,
):
    """Base class of all potentials. Documented in the subclasses."""

    @property
    @abstractmethod
    def ensemble_shape(self):
        pass

    @property
    @abstractmethod
    def ensemble_axes_metadata(self):
        pass

    @property
    def base_shape(self):
        return (self.num_slices,) + self.gpts

    @property
    @abstractmethod
    def num_frozen_phonons(self):
        pass

    @property
    def exit_planes_axes_metadata(self):
        return ThicknessAxis(label="z", values=tuple(self.exit_thicknesses))

    @property
    def base_axes_metadata(self):
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

    @property
    @abstractmethod
    def exit_planes(self) -> Tuple[int]:
        pass

    @property
    def exit_thicknesses(self) -> Tuple[float]:
        thicknesses = np.insert(np.cumsum(self.slice_thickness), 0, 0)
        return tuple(thicknesses[list(self.exit_planes)])

    @property
    def num_exit_planes(self) -> int:
        """
        Number of exit planes.
        """
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
        """
        Number of projected potential slices.
        """
        return len(self.slice_thickness)

    @property
    @abstractmethod
    def slice_thickness(self) -> np.ndarray:
        pass

    @property
    def slice_limits(self) -> List[Tuple[float, float]]:
        """
        The entrance and exit depths of each slice [Å].
        """
        cumulative_thickness = np.cumsum(np.concatenate(((0,), self.slice_thickness)))
        return [
            (cumulative_thickness[i], cumulative_thickness[i + 1])
            for i in range(len(cumulative_thickness) - 1)
        ]

    @property
    def thickness(self) -> float:
        """
        The thickness of the potential [Å].
        """
        return sum(self.slice_thickness)

    def __iter__(self):
        for slic in self.generate_slices():
            yield slic

    def project(self) -> "Images":
        return self.build().project()

    @property
    def _default_ensemble_chunks(self) -> Tuple:
        return validate_chunks(self.ensemble_shape, (1,))

    def images(self):
        return self.build().images()

    def show(self, project: bool = True, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        project : bool, optional
            Show the projected potential (True, default) or show all potential slices. It is recommended to index a subset
            of the potential slices when this keyword set to False.
        kwargs :
            Additional keyword arguments for the show method of :class:`.Images`.
        """
        if project:
            return self.project().show(**kwargs)
        else:
            if "explode" not in kwargs.keys():
                kwargs["explode"] = True

            return self.images().show(**kwargs)


def _validate_potential(
    potential: Union[Atoms, BasePotential], waves: "Waves" = None
) -> BasePotential:
    if isinstance(potential, (Atoms, BaseFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves.device

        potential = Potential(potential, device=device)

    if waves is not None and potential is not None:
        potential.grid.match(waves)

    return potential


def _validate_exit_planes(exit_planes, num_slices):
    if isinstance(exit_planes, int):
        exit_planes = list(range(0, num_slices, exit_planes))
        if exit_planes[-1] != num_slices:
            exit_planes.append(num_slices)
        exit_planes = tuple(exit_planes)
    elif exit_planes is None:
        exit_planes = (num_slices,)

    return exit_planes


class _PotentialBuilder(BasePotential):
    def __init__(
        self,
        slice_thickness: Union[float, Tuple[float, ...]],
        exit_planes: Union[int, Tuple[int, ...]],
        cell: Union[np.ndarray, Cell],
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        box: Tuple[float, float, float] = None,
        plane: Union[
            str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = "xy",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        periodic: bool = True,
        device: str = None,
    ):

        if self._require_cell_transform(cell, box=box, plane=plane, origin=origin):
            R = rotation_matrix_from_plane(plane)
            cell = np.dot(cell, R.T)
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
    def slice_thickness(self) -> Tuple[float, ...]:
        """
        Thickness of each slice [Å].
        """
        return self._slice_thickness

    @property
    def exit_planes(self) -> Tuple[int]:
        return self._exit_planes

    @property
    def device(self) -> str:
        return self._device

    @property
    def periodic(self) -> bool:
        """
        Specify whether the potential is periodic.
        """
        return self._periodic

    @property
    def plane(self) -> str:
        return self._plane

    @property
    def box(self) -> Tuple[float, float, float]:
        return self._box

    @property
    def origin(self) -> Tuple[float, float, float]:
        return self._origin

    def _require_cell_transform(self, cell, box, plane, origin):

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

    def __getitem__(self, item) -> "PotentialArray":
        return self.build(*unpack_item(item, len(self)), lazy=False)

    @staticmethod
    def _wrap_build_potential(potential, first_slice, last_slice):
        potential = potential.item()
        array = potential.build(first_slice, last_slice, lazy=False).array
        return array

    def build(
        self,
        first_slice: int = 0,
        last_slice: int = None,
        max_batch: Union[int, str] = 1,
        lazy: bool = None,
    ) -> "PotentialArray":
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
            Object representing the potential.
        """

        lazy = validate_lazy(lazy)

        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if lazy:
            blocks = self.ensemble_blocks(self._default_ensemble_chunks)

            xp = get_array_module(self.device)
            chunks = validate_chunks(self.ensemble_shape, (1,))
            chunks = chunks + ((len(self),), (self.gpts[0],), (self.gpts[1],))

            if self.ensemble_shape:
                new_axis = tuple(
                    range(len(self.ensemble_shape), len(self.ensemble_shape) + 3)
                )
            else:
                new_axis = tuple(range(1, 3))

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
                self.ensemble_shape + (last_slice - first_slice,) + self.gpts,
                dtype=xp.float32,
            )

            if self.ensemble_shape:

                for i, _, potential in self.generate_blocks():
                    for j, slic in enumerate(
                        potential.generate_slices(first_slice, last_slice)
                    ):
                        array[i, j] = slic.array[0]
            else:
                for j, slic in enumerate(self.generate_slices(first_slice, last_slice)):

                    array[j] = slic.array[0]

        potential = PotentialArray(
            array,
            sampling=self.sampling,
            slice_thickness=self.slice_thickness[first_slice:last_slice],
            exit_planes=self.exit_planes,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
        )

        # if not lazy:
        #    potential = potential.compute()

        return potential


class Potential(_PotentialBuilder):
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
        Sampling of the potential in `x` and `y` [1 / Å]. Provide either "sampling" or "gpts".
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å] (default is 0.5 Å).
        If given as a float, the number of slices is calculated by dividing the slice thickness into the `z`-height of supercell.
        The slice thickness may be given as a sequence of values for each slice, in which case an error will be thrown
        if the sum of slice thicknesses is not equal to the height of the atoms.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available (by Lobato et al. and Kirkland; default is 'lobato').
        See the citation guide for references.
    projection : 'finite' or 'infinite', optional
        If 'finite' the 3D potential is numerically integrated between the slice boundaries. If 'infinite' (default),
        the infinite potential projection of each atom will be assigned to a single slice.
    integral_method : {'quadrature', 'analytic'}, optional
        Specifies whether to perform projection integrals in real space or reciprocal space. By default, finite projection
        integrals are computed in real space and infinite projection integrals are performed in reciprocal space.
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
        The origin relative to the provided atoms mapped to the origin of the potential. This is equivalent to translating
        the atoms. The default is (0., 0., 0.).
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
        The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user configuration file.
    """

    _exclude_from_copy = ("parametrization", "projection", "integral_method")

    def __init__(
        self,
        atoms: Union[Atoms, BaseFrozenPhonons] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        slice_thickness: Union[float, Tuple[float, ...]] = 1,
        parametrization: Union[str, "Parametrization"] = "lobato",
        projection: str = "infinite",
        integral_method: str = None,
        exit_planes: Union[int, Tuple[int, ...]] = None,
        plane: Union[
            str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = "xy",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Tuple[float, float, float] = None,
        periodic: bool = True,
        integrator: ProjectionIntegrator = None,
        device: str = None,
    ):

        if not hasattr(atoms, "randomize"):
            self._frozen_phonons = DummyFrozenPhonons(atoms)
        else:
            self._frozen_phonons = atoms

        if projection == "infinite" and integral_method is None:
            integral_method = "analytic"

        elif projection == "finite" and integral_method is None:
            integral_method = "quadrature"

        if integrator is None:
            if projection == "finite" and integral_method == "quadrature":
                integrator = ProjectionQuadratureRule(parametrization=parametrization)
            elif projection == "finite" and integral_method == "analytic":
                integrator = GaussianProjectionIntegrals(
                    correction_parametrization=parametrization
                )
            elif projection == "infinite" and integral_method == "analytic":
                integrator = InfinitePotentialProjections(
                    parametrization=parametrization
                )
            else:
                raise NotImplementedError

        self._integrator = integrator
        self._sliced_atoms = None

        super().__init__(
            gpts=gpts,
            sampling=sampling,
            cell=self._frozen_phonons.atoms.cell,
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
        return self._frozen_phonons

    @property
    def num_frozen_phonons(self) -> int:
        return len(self.frozen_phonons)

    @property
    def integrator(self):
        return self._integrator

    def _cutoffs(self):
        atoms = self.frozen_phonons.atoms
        unique_numbers = np.unique(atoms.numbers)
        return tuple(self._integrator.cutoff(number) for number in unique_numbers)

    def _transformed_atoms(self):
        atoms = self.frozen_phonons.atoms

        cutoffs = self._cutoffs()

        if tuple(np.diag(atoms.cell)) != self.box:
            if self.periodic:
                atoms = orthogonalize_cell(
                    atoms,
                    box=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    return_transform=False,
                    allow_transform=True,
                )
            else:
                atoms = cut_cell(
                    atoms,
                    cell=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    margin=max(cutoffs) if cutoffs else 0.0,
                )

        return atoms

    def _prepare_atoms(self):

        atoms = self._transformed_atoms()
        cutoffs = self._cutoffs()

        if self.integrator.finite:
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

    @property
    def sliced_atoms(self):
        if self._sliced_atoms is not None:
            return self._sliced_atoms

        self._sliced_atoms = self._prepare_atoms()

        return self._sliced_atoms

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        Returns
        -------
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        xp = get_array_module(self.device)

        sliced_atoms = self.sliced_atoms

        numbers = np.unique(sliced_atoms.atoms.numbers)

        integrators = {
            number: self.integrator.build(
                chemical_symbols[number],
                gpts=self.gpts,
                sampling=self.sampling,
                device=self.device,
            )
            for number in numbers
        }

        for start, stop in generate_chunks(
            last_slice - first_slice, chunks=1, start=first_slice
        ):
            array = xp.zeros((stop - start,) + self.gpts, dtype=np.float32)
            for i, slice_idx in enumerate(range(start, stop)):

                for Z, integrator in integrators.items():
                    atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)

                    array[i] += integrator.integrate_on_grid(
                        positions=atoms.positions,
                        a=sliced_atoms.slice_limits[slice_idx][0],
                        b=sliced_atoms.slice_limits[slice_idx][1],
                        gpts=self.gpts,
                        sampling=self.sampling,
                        device=self.device,
                    )

            array -= array.min()

            yield PotentialArray(
                array,
                slice_thickness=self.slice_thickness[start:stop],
                extent=self.extent,
            )

    @property
    def ensemble_axes_metadata(self):
        return self.frozen_phonons.ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.frozen_phonons.ensemble_shape

    @staticmethod
    def potential(*args, frozen_phonons_partial, **kwargs):
        frozen_phonons = frozen_phonons_partial(*args)
        return Potential(frozen_phonons, **kwargs)

    def _from_partitioned_args(self, *args, **kwargs):
        frozen_phonons_partial = self.frozen_phonons._from_partitioned_args()
        kwargs = self._copy_kwargs(exclude=("atoms", "sampling"))
        return partial(
            self.potential, frozen_phonons_partial=frozen_phonons_partial, **kwargs
        )

    def _partition_args(self, chunks=(1,), lazy: bool = True):
        return self.frozen_phonons._partition_args(chunks, lazy=lazy)


class PotentialArray(BasePotential, HasArray):
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
        array: Union[np.ndarray, da.core.Array],
        slice_thickness: Union[float, Tuple[float, ...]] = None,
        extent: Union[float, Tuple[float, float]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        exit_planes: Union[int, Tuple[int, ...]] = None,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: dict = None,
    ):

        if len(array.shape) < 3:
            raise RuntimeError(f"PotentialArray must be 3D, not {len(array.shape)}D.")

        self._array = array
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling)
        self._slice_thickness = _validate_slice_thickness(
            slice_thickness, num_slices=array.shape[-3]
        )
        self._exit_planes = _validate_exit_planes(
            exit_planes, len(self._slice_thickness)
        )

        ensemble_axes_metadata = (
            [] if ensemble_axes_metadata is None else ensemble_axes_metadata
        )

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._check_axes_metadata()
        self._metadata = {} if metadata is None else metadata

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.array.shape[: -self._base_dims]

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def num_frozen_phonons(self):
        indices = _find_axes_type(self, FrozenPhononsAxis)
        if indices:
            return reduce(mul, tuple(self.array.shape[i] for i in indices))
        else:
            return 1

    @property
    def slice_thickness(self) -> Tuple[float, ...]:
        return self._slice_thickness

    @property
    def exit_planes(self) -> Tuple[int, ...]:
        return self._exit_planes

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

    def build(
        self,
        first_slice: int = 0,
        last_slice: int = None,
        chunks: int = 1,
        lazy: bool = None,
    ):
        raise RuntimeError("potential is already built")

    @staticmethod
    def _wrap_partition_args(*args):
        wrapped = np.zeros((1,), dtype=object)
        if len(args) > 1:
            wrapped.itemset(0, {"array": args[0], "ensemble_axes_metadata": args[1]})
        else:
            wrapped.itemset(0, {"array": args[0][0], "ensemble_axes_metadata": []})
        return wrapped

    def _partition_args(self, chunks: int = None, lazy: bool = True):

        if chunks is None and self.is_lazy:
            chunks = self.array.chunks[: -len(self.base_shape)]
        elif chunks is None:
            chunks = (1,) * len(self.ensemble_shape)

        chunks = self._validate_chunks(chunks)

        if lazy:
            array = self.ensure_lazy().array

            if chunks != array.chunks:
                array = array.rechunk(chunks + array.chunks[len(chunks) :])

            if len(self.ensemble_shape) == 0:
                array = array[None]
                blocks = da.blockwise(
                    self._wrap_partition_args,
                    (0,),
                    array,
                    tuple(range(len(array.shape))),
                    concatenate=True,
                    dtype=object,
                )
            else:
                ensemble_axes_metadata = self._partition_ensemble_axes_metadata(
                    chunks=chunks
                )

                blocks = da.blockwise(
                    self._wrap_partition_args,
                    tuple(range(max(len(self.ensemble_shape), 1))),
                    array,
                    tuple(range(len(array.shape))),
                    ensemble_axes_metadata,
                    tuple(range(max(len(self.ensemble_shape), 1))),
                    # chunks=chunks,
                    align_arrays=False,
                    concatenate=True,
                    dtype=object,
                )

        else:
            array = self.compute().array
            if len(self.ensemble_shape) == 0:
                blocks = np.zeros((1,), dtype=object)
            else:
                blocks = np.zeros(chunk_shape(chunks), dtype=object)
            ensemble_axes_metadata = self._partition_ensemble_axes_metadata(
                chunks, lazy=False
            )

            for block_indices, chunk_range in iterate_chunk_ranges(chunks):
                blocks[block_indices] = {
                    "array": array[chunk_range],
                    "ensemble_axes_metadata": ensemble_axes_metadata[block_indices],
                }

        return (blocks,)

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        kwargs["array"] = args[0]["array"]

        ensemble_axes_metadata = args[0]["ensemble_axes_metadata"]

        if hasattr(ensemble_axes_metadata, "item"):
            ensemble_axes_metadata = ensemble_axes_metadata.item()

        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return cls(**kwargs)

    def _from_partitioned_args(self):
        return partial(
            self._from_partitioned_args_func,
            **self._copy_kwargs(
                exclude=("array", "sampling", "ensemble_axes_metadata")
            ),
        )

    def generate_slices(self, first_slice=0, last_slice=None):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        Returns
        -------
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        for i in range(first_slice, last_slice):
            s = (0,) * (len(self.array.shape) - 3) + (i,)
            array = self.array[s][None]
            yield self.__class__(
                array, self.slice_thickness[i : i + 1], extent=self.extent
            )

    def transmission_function(self, energy: float) -> "TransmissionFunction":
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

        def _transmission_function(array, energy):
            array = complex_exponential(xp.float32(energy2sigma(energy)) * array)
            return array

        if self.is_lazy:
            array = self._array.map_blocks(
                _transmission_function,
                energy=energy,
                meta=xp.array((), dtype=xp.complex64),
            )
        else:
            array = _transmission_function(self._array, energy=energy)

        t = TransmissionFunction(
            array,
            slice_thickness=self.slice_thickness,
            extent=self.extent,
            energy=energy,
        )
        return t

    def tile(self, multiples: Union[Tuple[int, int], Tuple[int, int, int]]):
        """
        Tile the potential.

        Parameters
        ----------
        multiples: two or three int
            The number of repetitions of the potential along each axis. NOTE: if three integers are given, the first represents
            the number of repetitions along the `z`-axis.

        Returns
        -------
        PotentialArray object
            The tiled potential.
        """
        if len(multiples) == 2:
            multiples = tuple(multiples) + (1,)

        new_array = np.tile(self.array, (multiples[2], multiples[0], multiples[1]))

        new_extent = (self.extent[0] * multiples[0], self.extent[1] * multiples[1])
        new_slice_thickness = tuple(np.tile(self.slice_thickness, multiples[2]))

        return self.__class__(
            array=new_array,
            slice_thickness=new_slice_thickness,
            extent=new_extent,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
        )

    @property
    def metadata(self):
        return self._metadata

    def to_hyperspy(self):
        return self.images().to_hyperspy()

    def transmit(self, waves: "Waves", conjugate: bool = False) -> "Waves":
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
        transmissionfunction : TransmissionFunction
            Transmission function for the wave function through the potential slice.
        """

        transmission_function = self.transmission_function(waves.energy)

        return transmission_function.transmit(waves, conjugate=conjugate)

    def images(self):
        """Convert slices of the potential to a stack of images."""
        metadata = {"label": "potential", "units": "eV / e"}

        return Images(
            array=self._array,
            sampling=self.sampling,
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
        array = self.array.sum(-3)

        return Images(
            array=array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )


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
        slice_thickness: Union[float, Sequence[float]],
        extent: Union[float, Tuple[float, float]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        energy: float = None,
    ):

        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thickness, extent, sampling)

    def get_chunk(self, first_slice, last_slice) -> "TransmissionFunction":
        array = self.array[first_slice:last_slice]
        if len(array.shape) == 2:
            array = array[None]
        return self.__class__(
            array,
            self.slice_thickness[first_slice:last_slice],
            extent=self.extent,
            energy=self.energy,
        )

    def transmission_function(self, energy) -> "TransmissionFunction":
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

    def transmit(self, waves: "Waves", conjugate: bool = False) -> "Waves":
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
        transmissionfunction : TransmissionFunction
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
    The crystal potential may be used to represent a potential consisting of a repeating unit. This may allow
    calculations to be performed with lower computational cost by calculating the potential unit once and repeating it.

    If the repeating unit is a potential with frozen phonons it is treated as an ensemble from which each repeating
    unit along the `z`-direction is randomly drawn. If `num_frozen_phonons` an ensemble of crystal potentials are created
    each with a random seed for choosing potential units.

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
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    seeds: int or sequence of int
        Seed for the random number generator (RNG), or one seed for each RNG in the frozen phonon ensemble.
    """

    def __init__(
        self,
        potential_unit: BasePotential,
        repetitions: Tuple[int, int, int],
        num_frozen_phonons: int = None,
        exit_planes: int = None,
        seeds: Union[int, Tuple[int, ...]] = None,
    ):

        if num_frozen_phonons is None and seeds is None:
            self._seeds = None
        else:
            if num_frozen_phonons is None and seeds:
                num_frozen_phonons = len(seeds)
            elif num_frozen_phonons is None and seeds is None:
                num_frozen_phonons = 1

            self._seeds = _validate_seeds(seeds, num_frozen_phonons)

        if (potential_unit.num_frozen_phonons == 1) and (num_frozen_phonons is not None) and (num_frozen_phonons > 1):
            warnings.warn(
                "'num_frozen_phonons' is greater than one, but the potential unit does not have frozen phonons"
            )

        if (potential_unit.num_frozen_phonons > 1) and (num_frozen_phonons is not None):
            warnings.warn(
                "the potential unit has frozen phonons, but 'num_frozen_phonon' is not set"
            )

        gpts = (
            potential_unit.gpts[0] * repetitions[0],
            potential_unit.gpts[1] * repetitions[1],
        )
        extent = (
            potential_unit.extent[0] * repetitions[0],
            potential_unit.extent[1] * repetitions[1],
        )

        box = extent + (potential_unit.thickness * repetitions[2],)
        slice_thickness = potential_unit.slice_thickness * repetitions[2]
        super().__init__(
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
    def ensemble_shape(self) -> Tuple[int, ...]:
        if self._seeds is None:
            return ()
        else:
            return self.num_frozen_phonons,

    @property
    def num_frozen_phonons(self):
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

    @HasGridMixin.gpts.setter
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

    @HasGridMixin.sampling.setter
    def sampling(self, sampling):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

    @property
    def repetitions(self) -> Tuple[int, int, int]:
        return self._repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if self.seeds is None:
            return []
        else:
            return [FrozenPhononsAxis(_ensemble_mean=True)]

    @staticmethod
    def _wrap_partition_args(*args):
        arr = np.zeros((1,), dtype=object)
        arr.itemset(
            0,
            {
                "potential_args": args[0],
                "seeds": args[1],
                "num_frozen_phonons": args[2],
            },
        )
        return arr

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        if len(self.ensemble_shape) == 0:
            array = np.zeros((1,), dtype=object)
            chunks = ((1,),)
        else:
            array = np.zeros(len(chunks[0]), dtype=object)

        for block_indices, chunk_range in iterate_chunk_ranges(chunks):
            if self.seeds is None:
                seeds = None
                num_frozen_phonons = None
            else:
                seeds = self.seeds[chunk_range[0]]
                num_frozen_phonons = len(seeds)

            potential_unit = self.potential_unit._partition_args(-1, lazy=lazy)[0]

            if lazy:
                block = dask.delayed(self._wrap_partition_args)(
                    potential_unit, seeds, num_frozen_phonons
                )
                block = da.from_delayed(block, shape=(1,), dtype=object)
            else:
                block = self._wrap_partition_args(
                    potential_unit, seeds, num_frozen_phonons
                )

            array.itemset(block_indices[0], block)

        if lazy:
            array = da.concatenate(array)

        return (array,)

    @staticmethod
    def _crystal_potential(*args, potential_partial, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()

        potential_args = args["potential_args"]
        if hasattr(potential_args, "item"):
            potential_args = potential_args.item()

        potential_unit = potential_partial(potential_args)

        kwargs["seeds"] = args["seeds"]
        kwargs["num_frozen_phonons"] = args["num_frozen_phonons"]
        potential = CrystalPotential(potential_unit, **kwargs)

        return potential

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("potential_unit", "seeds", "num_frozen_phonons")
        )
        potential_partial = self.potential_unit._from_partitioned_args()
        return partial(
            self._crystal_potential, potential_partial=potential_partial, **kwargs
        )

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        Returns
        -------
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

        for i in range(self.repetitions[2]):
            generator = potentials[
                rng.integers(0, potentials.shape[0])
            ].generate_slices()
            for i in range(len(self.potential_unit)):
                slic = next(generator).tile(self.repetitions[:2])

                yield slic
