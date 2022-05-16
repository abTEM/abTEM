"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Tuple, List, Dict, Generator, TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import zarr
from ase import Atoms
from ase.data import chemical_symbols

from abtem.core.backend import get_array_module, _validate_device, copy_to_device
from abtem.core.complex import complex_exponential
from abtem.core.dask import HasDaskArray, validate_lazy
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.utils import generate_chunks
from abtem.measure.measure import Images
from abtem.potentials.atom import AtomicPotential, cutoff_distance
from abtem.potentials.infinite import calculate_scattering_factor, infinite_potential_projections
from abtem.potentials.parametrizations import parametrizations, Parametrization
from abtem.potentials.temperature import AbstractFrozenPhonons, FrozenPhonons, DummyFrozenPhonons, LazyAtoms
from abtem.structures.slicing import _validate_slice_thickness, SliceIndexedAtoms, SlicedAtoms, AbstractSlicedAtoms, \
    unpack_item
from abtem.structures.structures import is_cell_orthogonal, orthogonalize_cell, rotate_atoms_to_plane, \
    best_orthogonal_box, cut_box, rotation_matrix_from_plane, pad_atoms

if TYPE_CHECKING:
    import Waves


class AbstractPotential(HasGridMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """

    def __init__(self, slice_thickness: Union[Sequence, np.ndarray]):
        self._slice_thickness = slice_thickness

    @abstractmethod
    def to_delayed(self):
        pass

    @abstractmethod
    def build(self, **kwargs) -> 'PotentialArray':
        pass

    def __len__(self) -> int:
        return self.num_slices

    @property
    def num_slices(self) -> int:
        """The number of projected potential slices."""
        return len(self._slice_thickness)

    @property
    def slice_thickness(self) -> np.ndarray:
        return self._slice_thickness

    @property
    def slice_limits(self) -> List[Tuple[float, float]]:
        cumulative_thickness = np.cumsum(np.concatenate(((0,), self.slice_thickness)))
        return [(cumulative_thickness[i], cumulative_thickness[i + 1]) for i in range(len(cumulative_thickness) - 1)]

    @property
    def thickness(self) -> float:
        return sum(self._slice_thickness)

    @abstractmethod
    def get_chunk(self, first_slice, last_slice):
        pass

    def generate_slices(self, start: int = 0, stop: int = None, chunks: int = 1) -> Generator:

        if stop is None:
            stop = len(self)

        if start > stop:
            reverse = True
            start, stop = stop, start
        else:
            reverse = False

        start_slices = np.arange(start, stop)
        end_slices = np.arange(start + 1, stop + 1)

        if reverse:
            start_slices = start_slices[::-1]
            end_slices = end_slices[::-1]

        for start, stop in zip(start_slices, end_slices):
            yield self.get_chunk(first_slice=start, last_slice=stop)

    def __getitem__(self, item) -> 'PotentialArray':
        return self.get_chunk(*unpack_item(item, len(self)))

    def project(self) -> 'Images':
        return self.build().project()

    @property
    @abstractmethod
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        pass

    @property
    def num_frozen_phonons(self) -> int:
        return len(self.frozen_phonons)

    @property
    def ensemble_mean(self) -> bool:
        return self.frozen_phonons.ensemble_mean

    @abstractmethod
    def get_configurations(self, lazy: bool = False) -> List['AbstractPotential']:
        pass

    def show(self, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for abtem.plot.show_image.
        """
        return self.project().show(**kwargs)

    # @abstractmethod
    # def __copy__(self):
    #     pass

    def copy(self):
        """Make a copy."""
        return copy(self)


# class AbstractPotentialFromAtoms(AbstractPotential):
#
#     def __init__(self,
#                  box: Tuple[float, float, float],
#                  gpts: Union[int, Tuple[int, int]],
#                  sampling: Union[float, Tuple[float, float]],
#                  slice_thickness: Union[float, Sequence[float]],
#                  chunks: int = 1,
#                  device: str = None):
#
#         # self._periodic = periodic
#         # self._box = box
#         # self._x_vector = x_vector
#         # self._y_vector = y_vector
#         # self._origin = origin
#
#         # atoms = self._prepare_atoms(atoms)
#
#         # if np.abs(atoms.cell[2, 2]) < 1e-12:
#         #    raise RuntimeError('cell has no thickness')
#
#         # if hasattr(atoms, 'get_frozen_phonon_atoms'):
#         # self._frozen_phonons = frozen_phonons
#
#         # box = np.diag(frozen_phonons.cell)
#         # else:
#         #    self._frozen_phonons = FrozenPhonons(atoms, sigmas=0., num_configs=1)
#
#         # box = np.diag(atoms.cell)
#
#         # if box is None:
#         #     if not is_cell_orthogonal(atoms):
#         #         try:
#         #             ortho_atoms = orthogonalize_cell(atoms)
#         #         except:
#         #             ortho_atoms = atoms
#         #
#         #         box = np.diag(ortho_atoms.cell)
#         #     else:
#         #         box = np.diag(atoms.cell)
#
#         self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)
#
#         slice_thickness = _validate_slice_thickness(slice_thickness, box[2])
#
#         self._device = _validate_device(device)
#         self._chunks = chunks
#
#         super().__init__(box=box, gpts=gpts, sampling=sampling, slice_thickness=slice_thickness, chunks=chunks,
#                          device=device)
#
#     # def _prepare_atoms(self, atoms):
#     #
#     #     if self._periodic:
#     #         return self._prepare_atoms_periodic(atoms)
#     #
#     #     else:
#     #         return self._prepare_atoms_periodic(atoms)
#     #
#     # def _prepare_atoms_periodic(self, atoms):
#     #
#     #     if not is_cell_orthogonal(atoms):
#     #         try:
#     #             atoms, transform = orthogonalize_cell(atoms, return_transform=True, allow_transform=False)
#     #         except RuntimeError:
#     #             raise RuntimeError('The unit cell of the atoms is not orthogonal and could not be made orthogonal '
#     #                                'periodic without a. See our tutorial on making orthogonal cells '
#     #                                'https://abtem.readthedocs.io/en/latest/tutorials/orthogonal_cells.html')
#     #
#     #         return atoms
#     #     else:
#     #         return atoms
#
#     def _prepare_atoms_nonperiodic(self, atoms):
#         pass
#
#     @AbstractPotential.slice_thickness.setter
#     def slice_thickness(self, value):
#         self._slice_thickness = _validate_slice_thickness(value, self.atoms.cell[2, 2])
#
#     @property
#     def atoms(self) -> Union[Atoms]:
#         """Atoms object defining the atomic configuration."""
#         return self._frozen_phonons.atoms
#
#     @property
#     def frozen_phonons(self) -> AbstractFrozenPhonons:
#         """FrozenPhonons object defining the atomic configuration(s)."""
#         return self._frozen_phonons
#
#     @property
#     def num_frozen_phonons(self) -> int:
#         return len(self._frozen_phonons)
#
#     def build(self, lazy: bool = True) -> 'PotentialArray':
#         self.grid.check_is_defined()
#
#         xp = get_array_module(self._device)
#
#         def get_chunk(potential, first_slice, last_slice):
#             return potential.get_chunk(first_slice, last_slice).array
#
#         array = []
#
#         if hasattr(self.atoms, 'atoms'):
#             potential = self.to_delayed()
#             lazy = True
#         else:
#             potential = self
#             lazy = False
#
#         for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
#             shape = (last_slice - first_slice,) + self.gpts
#
#             if lazy:
#                 new_chunk = dask.delayed(get_chunk)(potential, first_slice, last_slice)
#                 new_chunk = da.from_delayed(new_chunk, shape=shape, meta=xp.array((), dtype=np.float32))
#             else:
#                 new_chunk = get_chunk(potential, first_slice, last_slice)
#
#             array.append(new_chunk)
#
#         if lazy:
#             array = da.concatenate(array)
#         else:
#             array = np.concatenate(array)
#
#         return PotentialArray(array, self.slice_thickness, extent=self.extent)


def validate_potential(potential: Union[Atoms, AbstractPotential], waves: 'Waves' = None) -> AbstractPotential:
    if isinstance(potential, (Atoms, AbstractFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves._device

        potential = Potential(potential, device=device)

    if waves is not None and potential is not None:
        potential.grid.match(waves)

    return potential


class PotentialConfiguration(AbstractPotential):

    def __init__(self, atoms, slice_thickness, parametrization, gpts=None, sampling=None, chunks: int = 1,
                 device: str = None):
        box = np.diag(atoms.cell)

        self._atoms = atoms
        self._parametrization = parametrization
        self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)
        self._device = _validate_device(device)
        self._chunks = chunks

        slice_thickness = _validate_slice_thickness(slice_thickness, box[2])

        super().__init__(slice_thickness=slice_thickness)

    def to_delayed(self):
        def to_delayed(atoms, self):
            self._atoms = atoms
            return self

        return dask.delayed(to_delayed)(self.atoms.atoms, self)

    @property
    def chunks(self) -> int:
        """The projection method."""
        return self._chunks

    @property
    def device(self) -> str:
        return self._device

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        return DummyFrozenPhonons()

    def get_configurations(self, lazy: bool = False) -> List['AbstractPotential']:
        return [self]

    @property
    def atoms(self):
        return self._atoms

    @property
    def parametrization(self):
        return self._parametrization

    def build(self, lazy=None) -> 'PotentialArray':
        self.grid.check_is_defined()
        xp = get_array_module(self.device)

        lazy = validate_lazy(lazy)

        def get_chunk(potential, first_slice, last_slice):
            return potential.get_chunk(first_slice, last_slice).array

        array = []
        if hasattr(self.atoms, 'compute'):
            potential = self.to_delayed()
        else:
            potential = self

        for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
            shape = (last_slice - first_slice,) + self.gpts

            if hasattr(potential, 'compute'):
                new_chunk = dask.delayed(get_chunk)(potential, first_slice, last_slice)
                new_chunk = da.from_delayed(new_chunk, shape=shape, meta=xp.array((), dtype=np.float32))
            else:
                new_chunk = get_chunk(potential, first_slice, last_slice)

            array.append(new_chunk)

        if lazy:
            array = da.concatenate(array)
        else:
            array = np.concatenate(array)

        return PotentialArray(array, self.slice_thickness, extent=self.extent)


class InfinitePotentialConfiguration(PotentialConfiguration):

    def __init__(self,
                 atoms,
                 slice_thickness,
                 parametrization: Union['str', Parametrization] = 'lobato',
                 gpts: Tuple[int, int] = None,
                 sampling: Tuple[float, float] = None,
                 chunks: int = 1,
                 device: str = None):

        super().__init__(atoms=atoms,
                         slice_thickness=slice_thickness,
                         parametrization=parametrization,
                         gpts=gpts,
                         sampling=sampling,
                         chunks=chunks,
                         device=device)

        self._scattering_factors = None
        self._sliced_atoms = None

    def get_sliced_atoms(self) -> SliceIndexedAtoms:
        if self._sliced_atoms is None:
            self.atoms.wrap(pbc=True)
            self._sliced_atoms = SliceIndexedAtoms(atoms=self.atoms, slice_thickness=self.slice_thickness)

        return self._sliced_atoms

    def get_scattering_factors(self) -> np.ndarray:

        if self._scattering_factors is None:
            xp = get_array_module(self.device)
            scattering_factors = {}
            for number in np.unique(self.atoms.numbers):
                f = calculate_scattering_factor(self.gpts,
                                                self.sampling,
                                                number,
                                                parametrization=self.parametrization,
                                                xp=xp)
                scattering_factors[number] = f

            self._scattering_factors = scattering_factors

        return self._scattering_factors

    def get_chunk(self, first_slice: int, last_slice: int) -> Union['PotentialArray']:
        xp = get_array_module(self.device)
        scattering_factors = self.get_scattering_factors()
        sliced_atoms = self.get_sliced_atoms()

        atoms = sliced_atoms[first_slice: last_slice]
        shape = (last_slice - first_slice,) + self.gpts

        if len(atoms) == 0:
            array = xp.zeros(shape, dtype=xp.float32)
        else:
            array = infinite_potential_projections(atoms, shape, self.sampling, scattering_factors)

        potential = PotentialArray(array,
                                   slice_thickness=self.slice_thickness[first_slice:last_slice],
                                   extent=self.extent)
        return potential


class FinitePotentialConfiguration(PotentialConfiguration):

    def __init__(self,
                 atoms,
                 slice_thickness,
                 parametrization: Union['str', Parametrization] = 'lobato',
                 gpts: Tuple[int, int] = None,
                 sampling: Tuple[float, float] = None,
                 chunks: int = 1,
                 device: str = None,
                 cutoff_tolerance: float = 1e-3):

        self._cutoff_tolerance = cutoff_tolerance

        super().__init__(atoms=atoms,
                         slice_thickness=slice_thickness,
                         parametrization=parametrization,
                         gpts=gpts,
                         sampling=sampling,
                         chunks=chunks,
                         device=device)

        self._atomic_potentials = None
        self._sliced_atoms = None

    @property
    def cutoff_tolerance(self):
        return self._cutoff_tolerance

    def get_sliced_atoms(self) -> SlicedAtoms:
        if self._sliced_atoms is None:
            z_padding = max(self.get_finite_cutoffs().values()) if self.get_finite_cutoffs().values() else 0.
            self._sliced_atoms = SlicedAtoms(atoms=self.atoms, slice_thickness=self.slice_thickness,
                                             z_padding=z_padding)

        return self._sliced_atoms

    def get_finite_cutoffs(self):
        return {Z: atomic_potential.cutoff for Z, atomic_potential in self.get_atomic_potentials().items()}

    def get_atomic_potentials(self) -> Dict[int, AtomicPotential]:

        if self._atomic_potentials is None:
            atomic_potentials = {}
            for Z in np.unique(self.atoms.numbers):
                atomic_potentials[Z] = AtomicPotential(symbol=Z,
                                                       parametrization=self.parametrization,
                                                       inner_cutoff=min(self.sampling) / 2,
                                                       cutoff_tolerance=self.cutoff_tolerance)
                atomic_potentials[Z].build_integral_table()

            self._atomic_potentials = atomic_potentials

        return self._atomic_potentials

    def get_chunk(self, first_slice: int, last_slice: int) -> 'PotentialArray':
        xp = get_array_module(self._device)

        atomic_potentials = self.get_atomic_potentials()
        sliced_atoms = self.get_sliced_atoms()

        extent = np.diag(self.atoms.cell)[:2]

        sampling = (extent[0] / self.gpts[0], extent[1] / self.gpts[1])

        array = xp.zeros((last_slice - first_slice,) + self.gpts, dtype=np.float32)
        for i, slice_idx in enumerate(range(first_slice, last_slice)):
            for Z, atomic_potential in atomic_potentials.items():
                atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)

                a = sliced_atoms.slice_limits[slice_idx][0] - atoms.positions[:, 2]
                b = sliced_atoms.slice_limits[slice_idx][1] - atoms.positions[:, 2]

                atomic_potential.project_on_grid(array[i], sampling, atoms.positions, a, b)

        array -= array.min()
        return PotentialArray(array, slice_thickness=self.slice_thickness[first_slice:last_slice], extent=self.extent)


class Potential(AbstractPotential):
    """
    Potential object.

    The potential object is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the IAM of the electrostatic potential(s).
        The atoms are assumed to be periodic.
    gpts : one or two int, optional
        Number of grid points describing each slice of the potential.
    sampling : one or two float, optional
        Lateral sampling of the potential [1 / Å].
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in Å. If given as a float the number of slices are calculated by dividing the
        slice thickness into the z-height of supercell.
        The slice thickness may be as a sequence of values for each slice, in which case an error will be thrown is the
        sum of slice thicknesses is not equal to the z-height of the supercell.
        Default is 0.5 Å.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available by Lobato et. al. and Kirkland. The abTEM default is 'lobato'.
        See the citation guide for references.
    projection : 'finite' or 'infinite'
        If 'finite' the 3d potential is numerically integrated between the slice boundaries. If 'infinite' the infinite
        potential projection of each atom will be assigned to a single slice.
    x_vector : {'x', 'y', 'z'} or three float, optional
        Vector defining the direction of the x-axis of the plane defining the plane perpendicular to the propagation
        direction.
    y_vector : {'x', 'y', 'z'} or three float, optional
        Vector defining the direction of the x-axis of the plane defining the plane perpendicular to the propagation
        direction.
    origin : three float, optional
    box : three float, optional

    periodic : bool, optional
        If True (default),
    chunks : int, optional
        Number of potential slices in each chunk of a lazy calculation. Default is 1.
    device : str, optional
        The device used for calculating the potential. The default is determined by the user configuration file.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]. The cutoff is only
        relevant for potentials using the 'finite' projection scheme.
    """

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: Union[float, np.ndarray] = .5,
                 parametrization: Union[str, Parametrization] = 'lobato',
                 projection: str = 'infinite',
                 x_vector: Union[str, Tuple[float, float, float]] = 'x',
                 y_vector: Union[str, Tuple[float, float, float]] = 'y',
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 box: Tuple[float, float, float] = None,
                 periodic: bool = True,
                 chunks: int = 1,
                 device: str = None,
                 cutoff_tolerance: float = 1e-3):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization

        if isinstance(parametrization, str):
            try:
                parametrization = parametrizations[parametrization]()
            except KeyError:
                raise RuntimeError()

        self._parametrization = parametrization

        if projection not in ('finite', 'infinite'):
            raise RuntimeError('Projection must be "finite" or "infinite"')

        if not hasattr(atoms, 'get_configurations'):
            self._frozen_phonons = FrozenPhonons(atoms, sigmas=0., num_configs=1)
        else:
            self._frozen_phonons = atoms

        if box is None and self._atoms_require_preparation(atoms=atoms,
                                                           box=box,
                                                           x_vector=x_vector,
                                                           y_vector=y_vector,
                                                           origin=origin):

            R = rotation_matrix_from_plane(x_vector, y_vector)
            cell = self._frozen_phonons._atoms.cell
            cell = np.dot(cell, R.T)
            box = tuple(best_orthogonal_box(cell))

        elif box is None:
            box = tuple(np.diag(atoms.cell))

        self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)
        self._device = _validate_device(device)
        self._chunks = chunks

        self._box = box
        self._x_vector = x_vector
        self._y_vector = y_vector
        self._origin = origin
        self._periodic = periodic
        self._projection = projection

        slice_thickness = _validate_slice_thickness(slice_thickness, thickness=box[2])

        super().__init__(slice_thickness=slice_thickness)

    @property
    def chunks(self) -> int:
        """The projection method."""
        return self._chunks

    @property
    def device(self) -> str:
        return self._device

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        return self._frozen_phonons

    def _atoms_require_preparation(self, atoms, box, x_vector, y_vector, origin):
        if not is_cell_orthogonal(atoms):
            return True

        if box is not None:
            return True

        if x_vector != 'x':
            return True

        if y_vector != 'y':
            return True

        if origin != (0., 0., 0.):
            return True

        return False

    def _prepare_atoms(self, atoms):

        if not self._atoms_require_preparation(atoms, self.box, self.x_vector, self.y_vector, self.origin):
            return atoms

        if self.box is None:
            box = best_orthogonal_box(atoms.cell)
        else:
            box = self.box

        if self.periodic:
            atoms.translate(-np.array(self.origin))
            atoms.wrap()
            atoms = rotate_atoms_to_plane(atoms, self.x_vector, self.y_vector)
            atoms, transform = orthogonalize_cell(atoms, box=box, return_transform=True, allow_transform=True)
            if self.projection == 'finite':
                symbols = [chemical_symbols[number] for number in np.unique(atoms.numbers)]
                cutoffs = [cutoff_distance(symbol, self.parametrization, self.cutoff_tolerance) for symbol in symbols]
                atoms = pad_atoms(atoms, margins=max(cutoffs) if cutoffs else 0.)


        else:
            if self.projection == 'infinite':
                raise RuntimeError()

            symbols = [chemical_symbols[number] for number in np.unique(atoms.numbers)]
            cutoffs = [cutoff_distance(symbol, self.parametrization, self.cutoff_tolerance) for symbol in symbols]

            atoms = cut_box(atoms,
                            box=box,
                            x_vector=self.x_vector,
                            y_vector=self.y_vector,
                            origin=self.origin,
                            margin=max(cutoffs) if cutoffs else 0.)

        return atoms

    def to_delayed(self):
        if self.num_frozen_phonons > 1:
            raise RuntimeError()

        def delayed_potential(d, atoms):
            d['atoms'] = atoms
            return self.__class__(**d)

        d = self._copy_as_dict(copy_atoms=False)
        return dask.delayed(delayed_potential)(d, self.atoms.atoms)

    @property
    def periodic(self) -> bool:
        return self._periodic

    @property
    def x_vector(self) -> str:
        return self._x_vector

    @property
    def y_vector(self) -> str:
        return self._y_vector

    @property
    def box(self) -> Tuple[float, float, float]:
        return self._box

    @property
    def origin(self) -> Tuple[float, float, float]:
        return self._origin

    @property
    def parametrization(self) -> Parametrization:
        """The potential parametrization."""
        return self._parametrization

    @property
    def projection(self) -> str:
        """The projection method."""
        return self._projection

    @property
    def cutoff_tolerance(self) -> float:
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    def build(self, lazy=True) -> 'PotentialArray':
        lazy = validate_lazy(lazy)
        configuration = self.get_configurations(lazy=lazy)[0]
        return configuration.build(lazy=lazy)

    def get_chunk(self, first_slice: int, last_slice: int) -> 'PotentialArray':
        configuration = self.get_configurations(lazy=False)[0]
        return configuration.get_chunk(first_slice, last_slice)

    def get_configurations(self, lazy: bool = None):
        """
        Function to generate scattering potentials for a set of frozen phonon configurations.

        Returns
        -------
        generator
            Generator of potentials.
        """
        lazy = validate_lazy(lazy)
        frozen_phonons = self.frozen_phonons

        if lazy:
            frozen_phonons = frozen_phonons.delay_atoms()

        frozen_phonons = frozen_phonons.apply_transformation(self._prepare_atoms, new_cell=np.diag(self.box))

        potentials = []
        for atoms in frozen_phonons.get_configurations():

            if not lazy:
                atoms = atoms.atoms

            if self.projection == 'infinite':
                potentials.append(InfinitePotentialConfiguration(atoms,
                                                                 gpts=self.gpts,
                                                                 slice_thickness=self.slice_thickness,
                                                                 parametrization=self.parametrization))
            else:
                potentials.append(FinitePotentialConfiguration(atoms,
                                                               gpts=self.gpts,
                                                               slice_thickness=self.slice_thickness,
                                                               parametrization=self.parametrization,
                                                               cutoff_tolerance=self.cutoff_tolerance))

        return potentials

    def _copy_as_dict(self, copy_atoms: bool = True):
        d = {'gpts': self.gpts,
             'sampling': self.gpts,
             'slice_thickness': self.slice_thickness,
             'parametrization': self.parametrization,
             'projection': self.projection,
             'chunks': self.chunks,
             'device': self.device,
             'x_vector': self.x_vector,
             'y_vector': self.y_vector,
             'box': self.box,
             'origin': self.origin,
             'cutoff_tolerance': self.cutoff_tolerance}

        if copy_atoms:
            d['atoms'] = self.frozen_phonons.copy()

        return d

    def __copy__(self) -> 'Potential':
        return self.__class__(**self._copy_as_dict(copy_atoms=True))


class PotentialArray(AbstractPotential, HasGridMixin, HasDaskArray):
    """
    Potential array object

    The potential array represents slices of the electrostatic potential as an array.

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
    """

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 slice_thickness: Union[np.ndarray, float, Sequence[float]] = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None):

        if len(array.shape) != 3:
            raise RuntimeError(f'PotentialArray must be 2d or 3d, not {len(array.shape)}d')

        self._array = array
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling)
        slice_thickness = _validate_slice_thickness(slice_thickness, num_slices=len(array))
        super().__init__(slice_thickness=slice_thickness)

    def _copy_as_dict(self, copy_array: bool = True) -> dict:
        d = {'slice_thickness': self.slice_thickness.copy(),
             'extent': self.extent}
        if copy_array:
            d['array'] = self.array.copy()
        return d

    def to_delayed(self):
        def wrap(array, d):
            return PotentialArray(array, **d)

        d = self._copy_as_dict(copy_array=False)
        return dask.delayed(wrap)(self.array, d)

    @property
    def frozen_phonons(self):
        return DummyFrozenPhonons()

    def build(self):
        return self

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

        t = TransmissionFunction(array, slice_thickness=self.slice_thickness.copy(), extent=self.extent, energy=energy)
        return t

    @property
    def num_configurations(self) -> int:
        return 1

    def get_configurations(self, *args, **kwargs) -> List['PotentialArray']:
        return [self]

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
        new_slice_thickness = np.tile(self.slice_thickness, multiples[2])

        return self.__class__(array=new_array, slice_thickness=new_slice_thickness, extent=new_extent)

    def to_zarr(self, url: str, overwrite: bool = False):
        """
        Write potential to a zarr file.

        Parameters
        ----------
        url: str
            url to which the data is saved.
            See https://docs.dask.org/en/latest/generated/dask.array.to_zarr.html
        """

        self.array.to_zarr(url, component='array', overwrite=overwrite)

        with zarr.open(url, mode='a') as f:
            f.create_dataset('slice_thickness', data=self.slice_thickness, overwrite=overwrite)
            f.create_dataset('extent', data=self.extent, overwrite=overwrite)

    @classmethod
    def from_zarr(cls, url: str, chunks: bool = 1):
        """
        Read potential from zarr file.

        Parameters
        ----------
        url: str
            The file to read.

        Returns
        -------
        PotentialArray object
        """

        with zarr.open(url, mode='r') as f:
            slice_thickness = f['slice_thickness'][:]
            extent = f['extent'][:]

        array = da.from_zarr(url, component='array', chunks=(chunks, -1, -1))
        return cls(array=array, slice_thickness=slice_thickness, extent=extent)

    def to_hyperspy(self):
        from hyperspy._signals.signal2d import Signal2D

        axes = [
            {'scale': self.slice_thickness[0],
             'units': 'Å',
             'name': 'Depth',
             'size': self.shape[0],
             'offset': 0.,
             },
            {'scale': self.sampling[1],
             'units': 'Å',
             'name': 'y',
             'size': self.shape[2],
             'offset': 0.,
             },
            {'scale': self.sampling[0],
             'units': 'Å',
             'name': 'x',
             'size': self.shape[1],
             'offset': 0.,
             },
        ]
        s = Signal2D(np.transpose(self.array, (0, 2, 1)), axes=axes).squeeze()

        return s

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

    def project(self) -> Images:
        """
        Create a 2d xarray representing a measurement of the projected potential.

        Returns
        -------
        Measurement
        """
        return Images(array=self._array.sum(0), sampling=self.sampling)

    def __copy__(self, device: str = None):
        if device is not None:
            array = copy_to_device(self.array, device)
        else:
            array = self.array.copy()

        return self.__class__(array=array,
                              slice_thickness=self.slice_thickness.copy(),
                              extent=self.extent)


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

    def __copy__(self, device: str = None):
        if device is not None:
            array = copy_to_device(self.array, device)
        else:
            array = self.array.copy()

        return self.__class__(array=array,
                              slice_thickness=self.slice_thickness.copy(),
                              extent=self.extent,
                              energy=self.energy)
