"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from functools import partial, reduce
from numbers import Number
from operator import mul
from typing import Union, Sequence, Tuple, List, TYPE_CHECKING

import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.core.array import HasArray, validate_lazy
from abtem.core.axes import ThicknessAxis, HasAxes, RealSpaceAxis, FrozenPhononsAxis, AxisMetadata
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import iterate_chunk_ranges, validate_chunks, chunk_shape, Chunks
from abtem.core.complex import complex_exponential
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.core.ensemble import Ensemble
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.utils import generate_chunks, EqualityMixin, CopyMixin
from abtem.measure.measure import Images
from abtem.potentials.integrals import ProjectionQuadratureRule, GaussianProjectionIntegrals, \
    InfinitePotentialProjections
from abtem.potentials.parametrizations.base import Parametrization
from abtem.potentials.temperature import AbstractFrozenPhonons, FrozenPhonons, DummyFrozenPhonons
from abtem.structures.slicing import validate_slice_thickness, SliceIndexedAtoms, SlicedAtoms, unpack_item
from abtem.structures.transform import is_cell_orthogonal, orthogonalize_cell, best_orthogonal_box, cut_box, \
    rotation_matrix_from_plane, pad_atoms

if TYPE_CHECKING:
    import Waves


class AbstractPotential(Ensemble, HasAxes, HasGridMixin, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """
    device: str

    @property
    @abstractmethod
    def num_frozen_phonons(self):
        pass

    @property
    def exit_planes_axes_metadata(self):
        return ThicknessAxis(label='z', values=tuple(self.exit_thicknesses))

    @property
    def base_axes_metadata(self):
        return [ThicknessAxis(label='z', values=tuple(np.cumsum(self.slice_thickness)), units='Å'),
                RealSpaceAxis(label='x', sampling=self.sampling[0], units='Å', endpoint=False),
                RealSpaceAxis(label='y', sampling=self.sampling[1], units='Å', endpoint=False)]

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
        return len(self.exit_planes)

    @abstractmethod
    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        pass

    @abstractmethod
    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              chunks: int = 1,
              lazy: bool = None):
        pass

    def __len__(self) -> int:
        return self.num_slices

    @property
    def num_slices(self) -> int:
        """The number of projected potential slices."""
        return len(self.slice_thickness)

    @property
    @abstractmethod
    def slice_thickness(self) -> np.ndarray:
        pass

    @property
    def slice_limits(self) -> List[Tuple[float, float]]:
        cumulative_thickness = np.cumsum(np.concatenate(((0,), self.slice_thickness)))
        return [(cumulative_thickness[i], cumulative_thickness[i + 1]) for i in range(len(cumulative_thickness) - 1)]

    @property
    def thickness(self) -> float:
        return sum(self.slice_thickness)

    def __iter__(self):
        for slic in self.generate_slices():
            yield slic

    def project(self) -> 'Images':
        return self.build().project()

    @property
    def default_ensemble_chunks(self) -> Tuple:
        return validate_chunks(self.ensemble_shape, (1,))

    def images(self):
        return self.build().images()

    def show(self, explode: bool = False, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for abtem.plot.show_image.
        """
        if explode:
            return self.images().show(explode=True, **kwargs)
        else:
            return self.project().show(**kwargs)


def validate_potential(potential: Union[Atoms, AbstractPotential], waves: 'Waves' = None) -> AbstractPotential:
    if isinstance(potential, (Atoms, AbstractFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves.device

        potential = Potential(potential, device=device)

    if waves is not None and potential is not None:
        potential.grid.match(waves)

    return potential


def validate_exit_planes(exit_planes, num_slices):
    if isinstance(exit_planes, int):
        exit_planes = list(range(0, num_slices, exit_planes))
        if exit_planes[-1] != num_slices:
            exit_planes.append(num_slices)
        exit_planes = tuple(exit_planes)
    elif exit_planes is None:
        exit_planes = (num_slices,)

    return exit_planes


class PotentialBuilder(AbstractPotential):

    def __init__(self,
                 gpts: Union[int, Tuple[int, int]],
                 sampling: Union[float, Tuple[float, float]],
                 slice_thickness: Union[float, Tuple[float, ...]],
                 exit_planes: Union[int, Tuple[int, ...]],
                 cell: Union[np.ndarray, Cell],
                 box: Tuple[float, float, float],
                 plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
                 origin: Tuple[float, float, float],
                 periodic: bool,
                 device: str,
                 ):

        if self._require_cell_transform(cell,
                                        box=box,
                                        plane=plane,
                                        origin=origin):
            R = rotation_matrix_from_plane(plane)
            cell = np.dot(cell, R.T)
            box = tuple(best_orthogonal_box(cell))

        elif box is None:
            box = tuple(np.diag(cell))

        self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)
        self._device = validate_device(device)

        self._box = box
        self._plane = plane
        self._origin = origin
        self._periodic = periodic

        self._slice_thickness = validate_slice_thickness(slice_thickness, thickness=box[2])
        self._exit_planes = validate_exit_planes(exit_planes, len(self._slice_thickness))

    @property
    def slice_thickness(self) -> Tuple[float, ...]:
        return self._slice_thickness

    @property
    def exit_planes(self) -> Tuple[int]:
        return self._exit_planes

    @property
    def device(self) -> str:
        return self._device

    @property
    def periodic(self) -> bool:
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

        if plane != 'xy':
            return True

        if origin != (0., 0., 0.):
            return True

        return False

    def __getitem__(self, item) -> 'PotentialArray':
        return self.build(*unpack_item(item, len(self)), lazy=False)

    @staticmethod
    def _wrap_build_potential(potential, first_slice, last_slice):
        potential = potential.item()
        array = potential.build(first_slice, last_slice, lazy=False).array
        return array

    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              chunks: Chunks = 1,
              lazy: bool = None) -> 'PotentialArray':
        """
        Build the potential producing a PotentialArray.

        Parameters
        ----------
        first_slice : int, optional
        last_slice : int, optional
        chunks : int or tuple of int or tuple of tuple of int
        lazy : bool

        Returns
        -------
        potential_array : PotentialArray
        """

        lazy = validate_lazy(lazy)

        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if lazy:
            blocks = self.ensemble_blocks(self.default_ensemble_chunks)

            xp = get_array_module(self.device)
            chunks = validate_chunks(self.ensemble_shape, (1,))
            chunks = chunks + ((len(self),), (self.gpts[0],), (self.gpts[1],))

            if self.ensemble_shape:
                new_axis = tuple(range(len(self.ensemble_shape), len(self.ensemble_shape) + 3))
            else:
                new_axis = tuple(range(1, 3))

            array = blocks.map_blocks(self._wrap_build_potential,
                                      new_axis=new_axis,
                                      first_slice=first_slice,
                                      last_slice=last_slice,
                                      chunks=chunks,
                                      meta=xp.array((), dtype=np.float32))

        else:
            xp = get_array_module(self.device)

            array = xp.zeros(self.ensemble_shape + (last_slice - first_slice,) + self.gpts, dtype=xp.float32)

            if self.ensemble_shape:

                for i, _, potential in self.generate_blocks():
                    for j, slic in enumerate(potential.generate_slices(first_slice, last_slice)):
                        array[i, j] = slic.array
            else:
                for j, slic in enumerate(self.generate_slices(first_slice, last_slice)):
                    array[j] = slic.array

        potential = PotentialArray(array,
                                   sampling=self.sampling,
                                   slice_thickness=self.slice_thickness[first_slice:last_slice],
                                   exit_planes=self.exit_planes,
                                   ensemble_axes_metadata=self.ensemble_axes_metadata)

        # if not lazy:
        #    potential = potential.compute()

        return potential


class Potential(PotentialBuilder):
    """
    The Potential is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the independent atom model for calculating
        the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points in x and y describing each slice of the potential. Provide either "sampling" or "gpts".
    sampling : one or two float, optional
        Sampling of the potential in x and y [1 / Å]. Provide either "sampling" or "gpts".
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in Å. If given as a float the number of slices are calculated by dividing the
        slice thickness into the z-height of supercell.
        The slice thickness may be given as a sequence of values for each slice, in which case an error will be thrown
        if the sum of slice thicknesses is not equal to the height of the atoms.
        Default is 0.5 Å.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available by Lobato et. al. and Kirkland. The abTEM default is 'lobato'.
        See the citation guide for references.
    projection : 'finite' or 'infinite', optional
        If 'finite' the 3d potential is numerically integrated between the slice boundaries. If 'infinite' the infinite
        potential projection of each atom will be assigned to a single slice. Default is 'infinite'.
    integral_method : {'quadrature', 'analytic'}, optional
        Specifies whether to perform projection integrals in real space or Fourier space. By default finite projection
        integrals are computed in real space and infinite projection integrals are performed in Fourier space.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided Atoms mapped to xy plane of the Potential, i.e. provided plane is
        perpendicular to the propagation direction. If string, it must be a combination of two of 'x', 'y' and 'z',
        the default value 'xy' indicates that potential slices are cuts the 'xy'-plane of the Atoms.
        The plane may also be specified with two arbitrary 3d vectors, which are mapped to the x and y directions of
        the potential, respectively. The length of the vectors has influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided Atoms mapped to the origin of the Potential. This is equivalent to shifting
        the atoms
        The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in x, y and z. If not given this is determined from the Atoms. If the box size does
        not match an integer number of the atoms supercell, an affine transformation may be necessary to preserve
        periodicity, determined by the `periodic` keyword.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines how the atomic structure is
        transformed. If True, the periodicity of the Atoms is preserved, which may require applying a small affine
        transformation to the atoms. If False, the transformed potential is effectively cut out of a larger repeated
        potential, which may not preserve periodicity.
    device : str, optional
        The device used for calculating the potential. The default is determined by the user configuration file.

    """
    _exclude_from_copy = ('parametrization', 'projection', 'integral_method')

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: Union[float, Tuple[float, ...]] = 1,
                 parametrization: Union[str, Parametrization] = 'lobato',
                 projection: str = 'infinite',
                 integral_method: str = None,
                 exit_planes: Union[int, Tuple[int, ...]] = None,
                 device: str = None,
                 plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = 'xy',
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 box: Tuple[float, float, float] = None,
                 periodic: bool = True,
                 integrator=None):

        if not hasattr(atoms, 'randomize'):
            self._frozen_phonons = DummyFrozenPhonons(atoms)
        else:
            self._frozen_phonons = atoms

        if projection == 'infinite' and integral_method is None:
            integral_method = 'analytic'

        elif projection == 'finite' and integral_method is None:
            integral_method = 'quadrature'

        if integrator is None:
            if projection == 'finite' and integral_method == 'quadrature':
                integrator = ProjectionQuadratureRule(parametrization=parametrization)
            elif projection == 'finite' and integral_method == 'analytic':
                integrator = GaussianProjectionIntegrals(correction_parametrization=parametrization)
            elif projection == 'infinite' and integral_method == 'analytic':
                integrator = InfinitePotentialProjections(parametrization=parametrization)
            else:
                raise NotImplementedError

        self._integrator = integrator
        self._sliced_atoms = None

        super().__init__(gpts=gpts,
                         sampling=sampling,
                         cell=self._frozen_phonons.atoms.cell,
                         slice_thickness=slice_thickness,
                         exit_planes=exit_planes,
                         device=device,
                         plane=plane,
                         origin=origin,
                         box=box,
                         periodic=periodic)

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
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
                atoms = orthogonalize_cell(atoms,
                                           box=self.box,
                                           plane=self.plane,
                                           origin=self.origin,
                                           return_transform=False,
                                           allow_transform=True)
            else:
                atoms = cut_box(atoms,
                                box=self.box,
                                plane=self.plane,
                                origin=self.origin,
                                margin=max(cutoffs) if cutoffs else 0.)

        return atoms

    def _prepare_atoms(self):

        atoms = self._transformed_atoms()
        cutoffs = self._cutoffs()

        if self.integrator.finite:
            margins = max(cutoffs) if len(cutoffs) else 0.
        else:
            margins = 0.

        if self.periodic:
            atoms = self.frozen_phonons.randomize(atoms)
            atoms.wrap()

        if not self.integrator.periodic and self.integrator.finite:
            atoms = pad_atoms(atoms, margins=margins)
        elif self.integrator.periodic:
            atoms = pad_atoms(atoms, margins=margins, directions='z')

        if not self.periodic:
            atoms = self.frozen_phonons.randomize(atoms)

        if self.integrator.finite:
            sliced_atoms = SlicedAtoms(atoms=atoms,
                                       slice_thickness=self.slice_thickness,
                                       z_padding=margins)
        else:
            sliced_atoms = SliceIndexedAtoms(atoms=atoms, slice_thickness=self.slice_thickness)

        return sliced_atoms

    @property
    def sliced_atoms(self):
        if self._sliced_atoms is not None:
            return self._sliced_atoms

        self._sliced_atoms = self._prepare_atoms()

        return self._sliced_atoms

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):

        if last_slice is None:
            last_slice = len(self)

        xp = get_array_module(self.device)

        sliced_atoms = self.sliced_atoms

        numbers = np.unique(sliced_atoms.atoms.numbers)

        integrators = {number: self.integrator.build(
            chemical_symbols[number], gpts=self.gpts, sampling=self.sampling, device=self.device)
            for number in numbers}

        for start, stop in generate_chunks(last_slice - first_slice, chunks=1, start=first_slice):
            array = xp.zeros((stop - start,) + self.gpts, dtype=np.float32)
            for i, slice_idx in enumerate(range(start, stop)):

                for Z, integrator in integrators.items():
                    atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)

                    array[i] += integrator.integrate_on_grid(positions=atoms.positions,
                                                             a=sliced_atoms.slice_limits[slice_idx][0],
                                                             b=sliced_atoms.slice_limits[slice_idx][1],
                                                             gpts=self.gpts,
                                                             sampling=self.sampling,
                                                             device=self.device)

            array -= array.min()

            yield PotentialArray(array,
                                 slice_thickness=self.slice_thickness[start:stop],
                                 extent=self.extent)

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

    def from_partitioned_args(self, *args, **kwargs):
        frozen_phonons_partial = self.frozen_phonons.from_partitioned_args()
        kwargs = self.copy_kwargs(exclude=('atoms',))
        return partial(self.potential, frozen_phonons_partial=frozen_phonons_partial, **kwargs)

    def partition_args(self, chunks=(1,), lazy: bool = True):
        return self.frozen_phonons.partition_args(chunks, lazy=lazy)


class PotentialArray(AbstractPotential, HasArray):
    """
    The potential array represents slices of the electrostatic potential as an array. All other potentials builds
    potential arrays.

    Parameters
    ----------
    array: 3D array
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thickness: float
        The thicknesses of potential slices in Å. If a float, the thickness is the same for all slices.
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

    """

    _base_dims = 3

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 slice_thickness: Union[float, Tuple[float, ...]] = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 exit_planes: Union[int, Tuple[int, ...]] = None,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        if len(array.shape) < 3:
            raise RuntimeError(f'PotentialArray must be 3d, not {len(array.shape)}d')

        self._array = array
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling)
        self._slice_thickness = validate_slice_thickness(slice_thickness, num_slices=array.shape[-3])
        self._exit_planes = validate_exit_planes(exit_planes, len(self._slice_thickness))

        ensemble_axes_metadata = [] if ensemble_axes_metadata is None else ensemble_axes_metadata

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self.check_axes_metadata()
        self._metadata = {} if metadata is None else metadata

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def num_frozen_phonons(self):
        indices = self.find_axes_type(FrozenPhononsAxis)
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

        ensemble_items = items[:len(self.ensemble_shape)]
        slic_items = items[len(self.ensemble_shape):]

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
            array = array[(slice(None),) * len(potential_array.ensemble_shape) + (None,)]
            slice_thickness = slice_thickness[None]

        kwargs = potential_array.copy_kwargs(exclude=('array', 'slice_thickness'))
        kwargs['array'] = array
        kwargs['slice_thickness'] = slice_thickness
        return potential_array.__class__(**kwargs)

    def build(self, first_slice: int = 0, last_slice: int = None, chunks: int = 1, lazy: bool = None):
        raise RuntimeError('potential is already built')

    @staticmethod
    def _wrap_partition_args(*args):
        wrapped = np.zeros((1,), dtype=object)
        if len(args) > 1:
            wrapped.itemset(0, {'array': args[0], 'ensemble_axes_metadata': args[1]})
        else:
            wrapped.itemset(0, {'array': args[0][0], 'ensemble_axes_metadata': []})
        return wrapped

    def partition_args(self, chunks: int = None, lazy: bool = True):

        if chunks is None and self.is_lazy:
            chunks = self.array.chunks[:-len(self.base_shape)]
        elif chunks is None:
            chunks = (1,) * len(self.ensemble_shape)

        chunks = self.validate_chunks(chunks)

        if lazy:
            array = self.ensure_lazy().array

            if chunks != array.chunks:
                array = array.rechunk(chunks + array.chunks[len(chunks):])

            if len(self.ensemble_shape) == 0:
                array = array[None]
                blocks = da.blockwise(self._wrap_partition_args, (0,),
                                      array, tuple(range(len(array.shape))),
                                      concatenate=True, dtype=object)
            else:
                ensemble_axes_metadata = self.partition_ensemble_axes_metadata(chunks=chunks)

                blocks = da.blockwise(self._wrap_partition_args,
                                      tuple(range(max(len(self.ensemble_shape), 1))),
                                      array, tuple(range(len(array.shape))),
                                      ensemble_axes_metadata, tuple(range(max(len(self.ensemble_shape), 1))),
                                      # chunks=chunks,
                                      align_arrays=False,
                                      concatenate=True, dtype=object)

        else:

            array = self.compute().array
            if len(self.ensemble_shape) == 0:
                blocks = np.zeros((1,), dtype=object)
            else:
                blocks = np.zeros(chunk_shape(chunks), dtype=object)
            ensemble_axes_metadata = self.partition_ensemble_axes_metadata(chunks, lazy=False)

            for block_indices, chunk_range in iterate_chunk_ranges(chunks):
                blocks[block_indices] = {'array': array[chunk_range],
                                         'ensemble_axes_metadata': ensemble_axes_metadata[block_indices]}

        return blocks,

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        kwargs['array'] = args[0]['array']

        ensemble_axes_metadata = args[0]['ensemble_axes_metadata']

        if hasattr(ensemble_axes_metadata, 'item'):
            ensemble_axes_metadata = ensemble_axes_metadata.item()

        kwargs['ensemble_axes_metadata'] = ensemble_axes_metadata
        return cls(**kwargs)

    def from_partitioned_args(self):
        return partial(self._from_partitioned_args_func,
                       **self.copy_kwargs(exclude=('array', 'ensemble_axes_metadata')))

    def generate_slices(self, first_slice=0, last_slice=None):

        if last_slice is None:
            last_slice = len(self)

        for i in range(first_slice, last_slice):
            s = (0,) * (len(self.array.shape) - 3) + (i,)
            array = self.array[s][None]
            yield self.__class__(array, self.slice_thickness[i:i + 1], extent=self.extent)

    def get_chunk(self, first_slice, last_slice):

        array = self.array[first_slice:last_slice]

        if len(array.shape) == 2:
            array = array[None]

        return self.__class__(array, self.slice_thickness[first_slice:last_slice], extent=self.extent)

    def transmission_function(self, energy: float) -> 'TransmissionFunction':
        """
        Calculate the transmission functions for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        TransmissionFunction object
        """

        xp = get_array_module(self.array)

        def _transmission_function(array, energy):
            array = complex_exponential(xp.float32(energy2sigma(energy)) * array)
            return array

        if self.is_lazy:
            array = self._array.map_blocks(_transmission_function, energy=energy, meta=xp.array((), dtype=xp.complex64))
        else:
            array = _transmission_function(self._array, energy=energy)

        t = TransmissionFunction(array, slice_thickness=self.slice_thickness, extent=self.extent, energy=energy)
        return t

    def tile(self, multiples: Union[Tuple[int, int], Tuple[int, int, int]]):
        """
        Tile the potential.

        Parameters
        ----------
        multiples: two or three int
            The number of repetitions of the potential along each axis. If three integers are given the first represents
            the number of repetitions along the z-axis.

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

        return self.__class__(array=new_array, slice_thickness=new_slice_thickness, extent=new_extent,
                              ensemble_axes_metadata=self.ensemble_axes_metadata)

    @property
    def metadata(self):
        return self._metadata

    def to_hyperspy(self):
        return self.images().to_hyperspy()

    def transmit(self, waves: 'Waves', conjugate: bool = False) -> 'Waves':
        """
        Transmit a wavefunction.

        Parameters
        ----------
        waves: Waves object
            Waves object to transmit.

        Returns
        -------
        TransmissionFunction
        """
        return self.transmission_function(waves.energy).transmit(waves, conjugate=conjugate)

    def images(self):
        metadata = {'label': 'potential', 'units': 'eV / e'}

        return Images(array=self._array, sampling=self.sampling, metadata=metadata,
                      ensemble_axes_metadata=self.axes_metadata[:-2])

    def project(self, cumulative: bool = False) -> Images:
        """
        Create a 2d xarray representing a measurement of the projected potential.

        Returns
        -------
        Measurement
        """
        metadata = {'label': 'potential', 'units': 'eV / e'}

        return Images(array=self._array.sum(-3), sampling=self.sampling,
                      ensemble_axes_metadata=self.ensemble_axes_metadata, metadata=metadata)


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions."""

    def __init__(self,
                 array: np.ndarray,
                 slice_thickness: Union[float, Sequence[float]],
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None):

        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thickness, extent, sampling)

    def get_chunk(self, first_slice, last_slice) -> 'TransmissionFunction':
        array = self.array[first_slice:last_slice]
        if len(array.shape) == 2:
            array = array[None]
        return self.__class__(array, self.slice_thickness[first_slice:last_slice], extent=self.extent,
                              energy=self.energy)

    def transmission_function(self, energy) -> 'TransmissionFunction':
        if energy != self.energy:
            raise RuntimeError()
        return self

    def transmit(self, waves: 'Waves', conjugate: bool = False) -> 'Waves':
        self.accelerator.check_match(waves)
        self.grid.check_match(waves)

        xp = get_array_module(self.array[0])

        if conjugate:
            waves._array *= xp.conjugate(self.array[0])
        else:
            waves._array *= self.array[0]
        # else:
        #    waves *= self.array

        return waves
