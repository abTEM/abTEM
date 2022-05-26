"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from functools import partial
from typing import Union, Sequence, Tuple, List, TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import zarr
from ase import Atoms

from abtem.core.axes import ThicknessAxis, HasAxes, RealSpaceAxis
from abtem.core.backend import get_array_module, validate_device, copy_to_device, device_name_from_array_module
from abtem.core.blockwise import Ensemble
from abtem.core.complex import complex_exponential
from abtem.core.dask import HasDaskArray, validate_chunks
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.utils import generate_chunks
from abtem.measure.measure import Images
from abtem.potentials.atom import AtomicPotential
from abtem.potentials.infinite import calculate_scattering_factor, infinite_potential_projections
from abtem.potentials.parametrizations import parametrizations, Parametrization
from abtem.potentials.temperature import AbstractFrozenPhonons, FrozenPhonons, DummyFrozenPhonons
from abtem.structures.slicing import _validate_slice_thickness, SliceIndexedAtoms, SlicedAtoms, unpack_item
from abtem.structures.structures import is_cell_orthogonal, orthogonalize_cell, best_orthogonal_box, cut_box, \
    rotation_matrix_from_plane, pad_atoms

if TYPE_CHECKING:
    import Waves


class AbstractPotential(HasAxes, Ensemble, HasGridMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """

    @property
    @abstractmethod
    def device(self) -> str:
        pass

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

    def __getitem__(self, item) -> 'PotentialArray':
        return self.build(*unpack_item(item, len(self)))

    def project(self) -> 'Images':
        return self.build().project()

    @abstractmethod
    def generate_configurations(self):
        pass

    @property
    def default_ensemble_chunks(self) -> Tuple:
        return validate_chunks(self.ensemble_shape, (1, -1))

    def _exit_plane_blocks(self, chunks):
        if len(chunks[0]) > 1:
            raise NotImplementedError

        arr = np.empty((1,), dtype=object)
        arr.itemset(self.exit_planes)
        return da.from_array(arr, chunks=chunks[0])

    def show(self, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for abtem.plot.show_image.
        """
        return self.project().show(**kwargs)

    def copy(self):
        """Make a copy."""
        return copy(self)


def validate_potential(potential: Union[Atoms, AbstractPotential], waves: 'Waves' = None) -> AbstractPotential:
    if isinstance(potential, (Atoms, AbstractFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves._device

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

    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              chunks: int = 1,
              lazy: bool = None,
              keep_ensemble_dims: bool = False) -> 'PotentialArray':

        if last_slice is None:
            last_slice = len(self)

        def build(potential):
            potential = potential.item()
            return potential.build().array[None, None]

        if lazy:
            blocks = self._ensemble_blockwise(1)
            chunks = blocks.chunks + ((len(self),),) + ((self.gpts[0],), (self.gpts[1],))
            array = blocks.map_blocks(build,
                                      new_axis=(2, 3, 4),
                                      chunks=chunks,
                                      dtype=np.float32)

        else:
            xp = get_array_module(self.device)

            array = xp.zeros((len(self),) + self.gpts, dtype=xp.float32)

            for i, slic in enumerate(self.generate_slices(first_slice, last_slice)):
                array[i] = slic.array

        potential = PotentialArray(array,
                                   sampling=self.sampling,
                                   slice_thickness=self.slice_thickness,
                                   ensemble_axes_metadata=self.ensemble_axes_metadata)

        potential = potential.squeeze()

        if not lazy:
            potential = potential.compute()

        return potential


class Potential(PotentialBuilder):
    """
    Potential object.

    The potential object is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the IAM of the electrostatic potential(s).
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
                 exit_planes: int = None,
                 device: str = None,
                 plane: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = 'xy',
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 box: Tuple[float, float, float] = None,
                 periodic: bool = True,
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

        if not hasattr(atoms, 'randomize'):
            self._frozen_phonons = FrozenPhonons(atoms, sigmas=0., num_configs=1)
        else:
            self._frozen_phonons = atoms

        if self._require_cell_transform(atoms=atoms,
                                        box=box,
                                        plane=plane,
                                        origin=origin):
            R = rotation_matrix_from_plane(plane)
            cell = np.dot(self.frozen_phonons.atoms.cell, R.T)
            box = tuple(best_orthogonal_box(cell))

            if periodic and projection == 'infinite':
                raise RuntimeError()

        if box is None:
            box = tuple(np.diag(atoms.cell))

        self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)
        self._device = validate_device(device)

        self._box = box
        self._plane = plane
        self._origin = origin
        self._periodic = periodic
        self._projection = projection

        self._slice_thickness = _validate_slice_thickness(slice_thickness, thickness=box[2])
        self._exit_planes = validate_exit_planes(exit_planes, len(self._slice_thickness))

        self._sliced_atoms = None

        super().__init__()

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        return self._frozen_phonons

    @property
    def num_frozen_phonons(self) -> int:
        return len(self.frozen_phonons)

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        axes_metadata += self.frozen_phonons.ensemble_axes_metadata
        axes_metadata += [ThicknessAxis(values=self.exit_thicknesses)]
        return axes_metadata

    @property
    def slice_thickness(self) -> np.ndarray:
        return self._slice_thickness

    @property
    def exit_planes(self) -> Tuple[int]:
        return self._exit_planes

    @property
    def device(self) -> str:
        return self._device

    def _require_cell_transform(self, atoms, box, plane, origin):
        if box == tuple(np.diag(atoms.cell)):
            return False

        if not is_cell_orthogonal(atoms):
            return True

        if box is not None:
            return True

        if plane != 'xy':
            return True

        if origin != (0., 0., 0.):
            return True

        return False

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

    def _prepare_atoms_finite(self, cutoffs):
        atoms = self.frozen_phonons.atoms

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

        atoms = pad_atoms(atoms, margins=max(cutoffs) if cutoffs else 0.)
        atoms = self.frozen_phonons.randomize(atoms)

        z_padding = max(cutoffs.values()) if cutoffs.values() else 0.

        self._sliced_atoms = SlicedAtoms(atoms=atoms, slice_thickness=self.slice_thickness, z_padding=z_padding)
        return self._sliced_atoms

    def _generate_slices_finite(self, start: int, stop: int) -> 'PotentialArray':
        xp = get_array_module(self._device)

        atomic_potentials = {}
        for Z in np.unique(self.frozen_phonons.atoms.numbers):
            atomic_potentials[Z] = AtomicPotential(symbol=Z,
                                                   parametrization=self.parametrization,
                                                   inner_cutoff=min(self.sampling) / 2,
                                                   cutoff_tolerance=self.cutoff_tolerance)
            atomic_potentials[Z].build_integral_table()

        cutoffs = {Z: atomic_potential.cutoff for Z, atomic_potential in atomic_potentials.items()}

        if self._sliced_atoms is None:
            self._prepare_atoms_finite(cutoffs)

        sliced_atoms = self._sliced_atoms

        for start, stop in generate_chunks(stop - start, chunks=1, start=start):
            array = xp.zeros((stop - start,) + self.gpts, dtype=np.float32)

            for i, slice_idx in enumerate(range(start, stop)):
                for Z, atomic_potential in atomic_potentials.items():
                    atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)

                    a = sliced_atoms.slice_limits[slice_idx][0] - atoms.positions[:, 2]
                    b = sliced_atoms.slice_limits[slice_idx][1] - atoms.positions[:, 2]

                    atomic_potential.project_on_grid(array[i], self.sampling, atoms.positions, a, b)

            array -= array.min()
            yield PotentialArray(array,
                                 slice_thickness=self.slice_thickness[start:stop],
                                 extent=self.extent)

    def _prepare_atoms_infinite(self):
        atoms = self.frozen_phonons.atoms

        if tuple(np.diag(atoms.cell)) != self.box:
            atoms = orthogonalize_cell(atoms,
                                       box=self.box,
                                       plane=self.plane,
                                       origin=self.origin,
                                       return_transform=False,
                                       allow_transform=True)

        atoms = self.frozen_phonons.randomize(atoms)
        atoms.wrap()

        self._sliced_atoms = SliceIndexedAtoms(atoms=atoms, slice_thickness=self.slice_thickness)
        return self._sliced_atoms


    def _generate_slices_infinite(self, start: int, stop: int) -> 'PotentialArray':

        if self._sliced_atoms is None:
            self._prepare_atoms_infinite()

        sliced_atoms = self._sliced_atoms

        xp = get_array_module(self.device)
        scattering_factors = {}
        for number in np.unique(self.frozen_phonons.atoms.numbers):
            f = calculate_scattering_factor(self.gpts,
                                            self.sampling,
                                            number,
                                            parametrization=self.parametrization,
                                            xp=xp)
            scattering_factors[number] = f

        for start, stop in generate_chunks(stop - start, chunks=1, start=start):
            atoms = sliced_atoms[start: stop]
            shape = (stop - start,) + self.gpts

            if len(atoms) == 0:
                array = xp.zeros(shape, dtype=xp.float32)
            else:
                array = infinite_potential_projections(atoms, shape, self.sampling, scattering_factors)

            yield PotentialArray(array,
                                 slice_thickness=self.slice_thickness[start:stop],
                                 extent=self.extent)

    def generate_configurations(self):
        for frozen_phonon in self.frozen_phonons.generate_configurations():
            kwargs = self._copy_as_dict(copy_atoms=False)
            kwargs['atoms'] = frozen_phonon
            yield Potential(**kwargs)

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):

        if last_slice is None:
            last_slice = len(self)

        if self.projection == 'infinite':
            for slices in self._generate_slices_infinite(first_slice, last_slice):
                yield slices
        elif self.projection == 'finite':
            for slices in self._generate_slices_finite(first_slice, last_slice):
                yield slices
        else:
            raise RuntimeError()

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        shape = ()
        shape += self.frozen_phonons.ensemble_shape
        shape += (self.num_exit_planes,)
        return shape

    @property
    def base_shape(self):
        return self.gpts

    def ensemble_blocks(self, chunks=(1, -1)):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        frozen_phonons_blocks = self.frozen_phonons.ensemble_blocks(chunks[:1])
        return frozen_phonons_blocks[0], self._exit_plane_blocks(chunks[1:])

    def ensemble_partial(self):
        def potential(*args, potential_kwargs):
            frozen_phonons = args[0].item()
            exit_planes = args[1].item()
            arr = np.empty((1, 1), dtype=object)
            arr[0, 0] = Potential(frozen_phonons, exit_planes=exit_planes, **potential_kwargs)
            return arr

        potential_kwargs = self._copy_as_dict(copy_atoms=False)
        del potential_kwargs['exit_planes']

        return partial(potential, potential_kwargs=potential_kwargs)

    def _copy_as_dict(self, copy_atoms: bool = True):
        d = {'gpts': self.gpts,
             'sampling': self.gpts,
             'slice_thickness': self.slice_thickness,
             'parametrization': self.parametrization,
             'projection': self.projection,
             'device': self.device,
             'exit_planes': self.exit_planes,
             'plane': self.plane,
             'box': self.box,
             'origin': self.origin,
             'cutoff_tolerance': self.cutoff_tolerance}

        if copy_atoms:
            d['atoms'] = self.frozen_phonons.copy()

        return d

    def __copy__(self) -> 'Potential':
        return self.__class__(**self._copy_as_dict(copy_atoms=True))


class CrystalPotential(AbstractPotential):
    """
    Crystal potential object

    The crystal potential may be used to represent a potential consisting of a repeating unit. This may allow
    calculations to be performed with lower memory and computational cost.

    The crystal potential has an additional function in conjunction with frozen phonon calculations. The number of
    frozen phonon configurations are not given by the FrozenPhonon objects, rather the ensemble of frozen phonon
    potentials represented by a potential with frozen phonons represent a collection of units, which will be assembled
    randomly to represent a random potential. The number of frozen phonon configurations should be given explicitely.
    This may save computational cost since a smaller number of units can be combined to a larger frozen phonon ensemble.

    Parameters
    ----------
    potential_unit : AbstractPotential
        The potential unit that repeated will create the full potential.
    repetitions : three int
        The repetitions of the potential in x, y and z.
    num_frozen_phonon_configs : int
        Number of frozen phonon configurations.
    """

    def __init__(self,
                 potential_unit: AbstractPotential,
                 repetitions: Tuple[int, int, int],
                 num_frozen_phonon_configs: int = 1,
                 exit_planes=None,
                 random_state=None, ):

        self._potential_unit = potential_unit
        self._repetitions = repetitions
        self._num_frozen_phonon_configs = num_frozen_phonon_configs
        self._random_state = random_state

        # if (potential_unit.num_frozen_phonon_configs == 1) & (num_frozen_phonon_configs > 1):
        #     warnings.warn('"num_frozen_phonon_configs" is greater than one, but the potential unit does not have'
        #                   'frozen phonons')
        #
        # if (potential_unit.num_frozen_phonon_configs > 1) & (num_frozen_phonon_configs == 1):
        #     warnings.warn('the potential unit has frozen phonons, but "num_frozen_phonon_configs" is set to 1')

        gpts = (self._potential_unit.gpts[0] * self.repetitions[0],
                self._potential_unit.gpts[1] * self.repetitions[1])
        extent = (self._potential_unit.extent[0] * self.repetitions[0],
                  self._potential_unit.extent[1] * self.repetitions[1])

        self._grid = Grid(extent=extent, gpts=gpts, sampling=self._potential_unit.sampling, lock_extent=True)

        super().__init__()

        self._exit_planes = validate_exit_planes(exit_planes, len(self))

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        shape = ()
        shape += self.frozen_phonons.ensemble_shape
        shape += (self.num_exit_planes,)
        return shape

    @property
    def base_shape(self):
        return self.gpts

    @property
    def num_frozen_phonon_configs(self):
        return self._num_frozen_phonon_configs

    @property
    def random_state(self):
        return self._random_state

    @property
    def slice_thickness(self) -> np.ndarray:
        return np.tile(self._potential_unit.slice_thickness, self.repetitions[2])

    @property
    def exit_planes(self) -> Tuple[int]:
        return self._exit_planes

    @property
    def device(self):
        return self._potential_unit.device

    @property
    def potential_unit(self) -> AbstractPotential:
        return self._potential_unit

    @HasGridMixin.gpts.setter
    def gpts(self, gpts):
        if not ((gpts[0] % self.repetitions[0] == 0) and (gpts[1] % self.repetitions[0] == 0)):
            raise ValueError('gpts must be divisible by the number of potential repetitions')
        self.grid.gpts = gpts
        self._potential_unit.gpts = (gpts[0] // self._repetitions[0], gpts[1] // self._repetitions[1])

    @HasGridMixin.sampling.setter
    def sampling(self, sampling):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

    @property
    def repetitions(self) -> Tuple[int, int, int]:
        return self._repetitions

    @repetitions.setter
    def repetitions(self, repetitions: Tuple[int, int, int]):
        repetitions = tuple(repetitions)

        if len(repetitions) != 3:
            raise ValueError('repetitions must be sequence of length 3')

        self._repetitions = repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        return DummyFrozenPhonons()

    @property
    def num_frozen_phonons(self) -> int:
        return len(self.frozen_phonons)

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        axes_metadata += self.frozen_phonons.ensemble_axes_metadata
        axes_metadata += [ThicknessAxis(values=self.exit_thicknesses)]
        return axes_metadata

    def ensemble_blocks(self, chunks=(1, -1)):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        random_state = dask.delayed(self.random_state)

        potential_unit_blocks = self.potential_unit.ensemble_blocks((-1, -1))
        potential_partial = self.potential_unit.ensemble_partial()

        blocks = []
        for i in range(self.num_frozen_phonon_configs):
            p = dask.delayed(potential_partial)(*potential_unit_blocks)
            p = da.from_delayed(p, shape=(1,), dtype=object)
            blocks.append(p)

        blocks = da.concatenate(blocks)

        return blocks, self._exit_plane_blocks(chunks[1:])

    def ensemble_partial(self):

        def crystal_potential(*args, **kwargs):
            potential_unit = args[0].item()
            exit_planes = args[1].item()

            arr = np.zeros((1,) * len(args), dtype=object)
            arr.itemset(CrystalPotential(potential_unit,
                                         exit_planes=exit_planes,
                                         **kwargs))
            return arr

        kwargs = {'repetitions': self.repetitions}
        return partial(crystal_potential, **kwargs)

    def generate_configurations(self):
        for i in range(self.num_frozen_phonon_configs):
            kwargs = self._copy_as_dict()
            kwargs['num_frozen_phonon_configs'] = 1
            yield CrystalPotential(**kwargs)

    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              chunks: int = 1,
              lazy: bool = None) -> 'PotentialArray':

        if last_slice is None:
            last_slice = len(self)

        xp = get_array_module(self.device)

        array = xp.zeros((len(self),) + self.gpts, dtype=xp.float32)

        for i, slic in enumerate(self.generate_slices(first_slice, last_slice)):
            array[i] = slic.array

        return PotentialArray(array, sampling=self.sampling, slice_thickness=self.slice_thickness)

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        potentials = [potential.build() for potential in self.potential_unit.generate_configurations()]

        first_layer = first_slice // self._potential_unit.num_slices
        if last_slice is None:
            last_layer = self.repetitions[2]
        else:
            last_layer = last_slice // self._potential_unit.num_slices

        first_slice = first_slice % self._potential_unit.num_slices
        last_slice = None

        # configs = self._calculate_configs(energy, max_batch)

        # if len(configs) == 1:
        #    layers = configs * self.repetitions[2]
        # else:
        #    layers = [configs[np.random.randint(len(configs))] for _ in range(self.repetitions[2])]

        if self.random_state:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState()

        for i in range(self.repetitions[2]):

            j = random_state.randint(0, len(potentials))
            generator = potentials[j].generate_slices()

            for i in range(len(self.potential_unit)):
                yield next(generator).tile(self.repetitions[:2])

    def _copy_as_dict(self, copy_potential: bool = True):

        kwargs = {'repetitions': self.repetitions,
                  'num_frozen_phonon_configs': self.num_frozen_phonon_configs,
                  'exit_planes': self.exit_planes,
                  'random_state': self.random_state}

        if copy_potential:
            kwargs['potential_unit'] = self.potential_unit.copy()

        return kwargs

        # array = np.concatenate([potential_array.array for potential_array in potential_arrays], axis=1)

        # array = array.reshape((array.shape[0],) + self.gpts)

        # yield PotentialArray(array,
        #                      sampling=self.sampling,
        #                      slice_thickness=potential_arrays[0].slice_thickness,
        #                      ensemble_axes_metadata=potential_arrays[0].ensemble_axes_metadata)

        # for layer_num, layer in enumerate(layers[first_layer:last_layer]):
        #
        #     if layer_num == last_layer:
        #         last_slice = last_slice % self._potential_unit.num_slices
        #
        #     for start, end, potential_slice in layer.generate_slices(first_slice=first_slice,
        #                                                              last_slice=last_slice,
        #                                                              max_batch=max_batch):
        #         yield layer_num + start, layer_num + end, potential_slice
        #
        #         first_slice = 0

    # def _calculate_configs(self, energy, max_batch=1):
    #     potential_generators = self._potential_unit.generate_frozen_phonon_potentials(pbar=False)
    #
    #     potential_configs = []
    #     for potential in potential_generators:
    #
    #         if isinstance(potential, AbstractPotentialBuilder):
    #             potential = potential.build(max_batch=max_batch)
    #         elif not isinstance(potential, PotentialArray):
    #             raise RuntimeError()
    #
    #         if energy is not None:
    #             potential = potential.as_transmission_function(energy=energy, max_batch=max_batch, in_place=False)
    #
    #         potential = potential.tile(self.repetitions[:2])
    #         potential_configs.append(potential)
    #
    #     return potential_configs
    #
    # def _generate_slices_base(self, first_slice=0, last_slice=None, max_batch=1, energy=None):
    #
    #     first_layer = first_slice // self._potential_unit.num_slices
    #     if last_slice is None:
    #         last_layer = self.repetitions[2]
    #     else:
    #         last_layer = last_slice // self._potential_unit.num_slices
    #
    #     first_slice = first_slice % self._potential_unit.num_slices
    #     last_slice = None
    #
    #     configs = self._calculate_configs(energy, max_batch)
    #
    #     if len(configs) == 1:
    #         layers = configs * self.repetitions[2]
    #     else:
    #         layers = [configs[np.random.randint(len(configs))] for _ in range(self.repetitions[2])]
    #
    #     for layer_num, layer in enumerate(layers[first_layer:last_layer]):
    #
    #         if layer_num == last_layer:
    #             last_slice = last_slice % self._potential_unit.num_slices
    #
    #         for start, end, potential_slice in layer.generate_slices(first_slice=first_slice,
    #                                                                  last_slice=last_slice,
    #                                                                  max_batch=max_batch):
    #             yield layer_num + start, layer_num + end, potential_slice
    #
    #             first_slice = 0
    #
    # def generate_slices(self, first_slice=0, last_slice=None, max_batch=1):
    #     return self._generate_slices_base(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)
    #
    # def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
    #     return self._generate_slices_base(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch,
    #                                       energy=energy)


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
                 sampling: Union[float, Tuple[float, float]] = None,
                 exit_planes=None,
                 ensemble_axes_metadata=None):

        if len(array.shape) < 3:
            raise RuntimeError(f'PotentialArray must be 3d, not {len(array.shape)}d')

        self._array = array
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling)
        self._slice_thickness = _validate_slice_thickness(slice_thickness, num_slices=array.shape[-3])
        self._exit_planes = validate_exit_planes(exit_planes, len(self._slice_thickness))

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        super().__init__(array=array)

        self._ensemble_axes_metadata = ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.array.shape[:-3]

    @property
    def base_shape(self):
        return self.array.shape[-3:]

    def generate_configurations(self):
        raise NotImplementedError

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def base_axes_metadata(self):
        return [ThicknessAxis(label='z', values=tuple(self.exit_thicknesses), units='Å'),
                RealSpaceAxis(label='x', sampling=self.sampling[0], units='Å', endpoint=False),
                RealSpaceAxis(label='y', sampling=self.sampling[1], units='Å', endpoint=False)]

    def squeeze(self):
        d = self._copy_as_dict(copy_array=False)
        d['ensemble_axes_metadata'] = [m for s, m in zip(self.ensemble_shape, self.ensemble_axes_metadata) if
                                       s != 1]
        d['array'] = self.array[tuple(0 if s == 1 else slice(None) for s in self.ensemble_shape)]
        return self.__class__(**d)

    @property
    def slice_thickness(self) -> np.ndarray:
        return self._slice_thickness

    @property
    def exit_planes(self) -> Tuple[int]:
        return self._exit_planes

    @property
    def device(self):
        return device_name_from_array_module(get_array_module(self.array))

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

    def build(self, first_slice: int = 0, last_slice: int = None, chunks: int = 1, lazy: bool = None):
        return self

    def ensemble_blocks(self, chunks):
        raise NotImplementedError

    def ensemble_partial(self):
        raise NotImplementedError

    def generate_slices(self, first_slice=0, last_slice=None):

        if last_slice is None:
            last_slice = len(self)

        for i in range(first_slice, last_slice):
            array = self.array[i][None]
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

    def copy(self, device: str = None):
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

    def copy(self, device: str = None):
        if device is not None:
            array = copy_to_device(self.array, device)
        else:
            array = self.array.copy()

        return self.__class__(array=array,
                              slice_thickness=self.slice_thickness.copy(),
                              extent=self.extent,
                              energy=self.energy)
