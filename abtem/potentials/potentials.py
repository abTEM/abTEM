"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Tuple, List, Dict, Generator, TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import zarr
from ase import Atoms

from abtem.core.backend import get_array_module, _validate_device, copy_to_device
from abtem.core.complex import complex_exponential
from abtem.core.dask import HasDaskArray
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.utils import generate_chunks
from abtem.measure.measure import Images
from abtem.potentials.atom import AtomicPotential
from abtem.potentials.infinite import calculate_scattering_factor, infinite_potential_projections
from abtem.potentials.parametrizations import parametrizations, Parametrization
from abtem.potentials.temperature import AbstractFrozenPhonons, FrozenPhonons, DelayedAtoms, DummyFrozenPhonons
from abtem.structures.slicing import _validate_slice_thickness, SliceIndexedAtoms, SlicedAtoms, unpack_item
from abtem.structures.structures import is_cell_orthogonal, orthogonalize_cell

if TYPE_CHECKING:
    import Waves


class AbstractPotential(HasGridMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """

    def __init__(self, slice_thickness: Union[Sequence, np.ndarray]):
        self._slice_thickness = np.array(slice_thickness)

    @abstractmethod
    def to_delayed(self):
        pass

    @abstractmethod
    def build(self) -> 'PotentialArray':
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
    def get_frozen_phonon_potentials(self, lazy: bool = False) -> List['AbstractPotential']:
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

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """Make a copy."""
        return copy(self)


class AbstractPotentialFromAtoms(AbstractPotential):

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons],
                 gpts: Union[int, Tuple[int, int]],
                 sampling: Union[float, Tuple[float, float]],
                 slice_thickness: Union[float, Sequence[float]],
                 box: Tuple[float, float, float] = None,
                 plane: str = 'xy',
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 chunks: int = None,
                 device: str = None,
                 precalculate: bool = False):

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('cell has no thickness')

        if hasattr(atoms, 'get_frozen_phonon_atoms'):
            self._frozen_phonons = atoms
        else:
            self._frozen_phonons = FrozenPhonons(atoms, sigmas=0., num_configs=1)

        if box is None:
            if not is_cell_orthogonal(atoms):
                try:
                    ortho_atoms = orthogonalize_cell(atoms)
                except:
                    ortho_atoms = atoms

                box = np.diag(ortho_atoms.cell)
            else:
                box = np.diag(atoms.cell)

        self._box = box
        self._plane = plane
        self._origin = origin
        self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)
        slice_thickness = _validate_slice_thickness(slice_thickness, box[2])

        self._precalculate = precalculate

        self._device = _validate_device(device)

        if chunks == 'auto':
            if precalculate:
                chunks = int(np.floor(256 / (4 * np.prod(self.gpts) / 1e6)))
            else:
                chunks = 1

        self._chunks = chunks

        super().__init__(slice_thickness=slice_thickness)

    @property
    def chunks(self) -> int:
        """The projection method."""
        return self._chunks

    @property
    def device(self) -> str:
        return self._device

    @property
    def precalculate(self) -> bool:
        return self._precalculate

    @property
    def plane(self) -> str:
        return self._plane

    @property
    def box(self) -> Tuple[float, float, float]:
        return self._box

    @property
    def origin(self) -> Tuple[float, float, float]:
        return self._origin

    @AbstractPotential.slice_thickness.setter
    def slice_thickness(self, value):
        self._slice_thickness = _validate_slice_thickness(value, self.atoms.cell[2, 2])

    @property
    def atoms(self) -> Union[Atoms, DelayedAtoms]:
        """Atoms object defining the atomic configuration."""
        return self._frozen_phonons.atoms

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        """FrozenPhonons object defining the atomic configuration(s)."""
        return self._frozen_phonons

    @property
    def num_frozen_phonons(self) -> int:
        return len(self._frozen_phonons)

    def build(self, lazy: bool = True) -> 'PotentialArray':
        self.grid.check_is_defined()

        xp = get_array_module(self._device)

        def get_chunk(potential, first_slice, last_slice):
            return potential.get_chunk(first_slice, last_slice).array

        array = []

        if hasattr(self.atoms, 'atoms'):
            potential = self.to_delayed()
            lazy = True
        else:
            potential = self
            lazy = False

        for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
            shape = (last_slice - first_slice,) + self.gpts

            if lazy:
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


def validate_potential(potential: Union[Atoms, AbstractPotential], waves: 'Waves' = None) -> AbstractPotential:
    if isinstance(potential, (Atoms, AbstractFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves._device

        potential = Potential(potential, device=device)

    if waves is not None and potential is not None:
        potential.grid.match(waves)

    return potential


class Potential(AbstractPotentialFromAtoms):
    """
    Potential object.

    The potential object is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons object
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the IAM of the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points describing each slice of the potential.
    sampling : one or two float, optional
        Lateral sampling of the potential [1 / Å].
    slice_thickness : float, optional
        Thickness of the potential slices in Å for calculating the number of slices used by the multislice algorithm.
        Default is 0.5 Å.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available by Lobato et. al. and Kirkland. The abTEM default is 'lobato'.
        See the citation guide for references.
    projection : 'finite' or 'infinite'
        If 'finite' the 3d potential is numerically integrated between the slice boundaries. If 'infinite' the infinite
        potential projection of each atom will be assigned to a single slice.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]. The cutoff is only
        relevant for potentials using the 'finite' projection scheme.
    device : str, optional
        The device used for calculating the potential. The default is 'cpu'.
    """

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: Union[float, np.ndarray] = .5,
                 parametrization: Union[str, Parametrization] = 'lobato',
                 projection: str = 'infinite',
                 chunks: Union[int, str] = 'auto',
                 precalculate: bool = False,
                 device: str = None,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 cutoff_tolerance: float = 1e-3):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization

        if isinstance(parametrization, str):
            try:
                parametrization = parametrizations[parametrization]()
            except KeyError:
                raise RuntimeError()

        self._parametrization = parametrization

        # if parametrization not in parametrizations:
        #    raise RuntimeError('Parametrization must be "{}" or "{}"'.format(*parametrizations))

        if projection not in ('finite', 'infinite'):
            raise RuntimeError('Projection must be "finite" or "infinite"')

        self._projection = projection

        super().__init__(atoms=atoms,
                         gpts=gpts,
                         sampling=sampling,
                         slice_thickness=slice_thickness,
                         plane=plane,
                         box=box,
                         origin=origin,
                         device=device,
                         chunks=chunks,
                         precalculate=precalculate)

        self._sliced_atoms = None
        self._scattering_factors = None
        self._atomic_potentials = None

        def clear_data(*args):
            self._scattering_factors = None
            self._atomic_potentials = None

        self.grid.events.observe(clear_data, ('sampling', 'gpts', 'extent'))

    def is_lazy(self):
        return isinstance(self.atoms, DelayedAtoms)

    def to_delayed(self):
        if self.num_frozen_phonons > 1:
            raise RuntimeError()

        if isinstance(self.atoms, DelayedAtoms):
            atoms = self.atoms.atoms
        else:
            atoms = self.atoms

        def delayed_potential(atoms):
            d = self._copy_as_dict(copy_atoms=False)
            d['atoms'] = atoms
            return self.__class__(**d)

        return dask.delayed(delayed_potential)(atoms)

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

    def _calculate_atomic_potentials(self) -> Dict[int, AtomicPotential]:
        atomic_potentials = {}

        for Z in np.unique(self.atoms.numbers):
            atomic_potentials[Z] = AtomicPotential(symbol=Z,
                                                   parametrization=self.parametrization,
                                                   inner_cutoff=min(self.sampling) / 2,
                                                   cutoff_tolerance=self.cutoff_tolerance)
            atomic_potentials[Z].build_integral_table()

        return atomic_potentials

    def get_sliced_atoms(self) -> SliceIndexedAtoms:
        if self._sliced_atoms is None:
            atoms = self.atoms

            atoms.wrap(pbc=True)

            if self.projection == 'infinite':
                self._sliced_atoms = SliceIndexedAtoms(atoms=atoms, slice_thickness=self.slice_thickness)
            else:
                raise NotImplementedError

        return self._sliced_atoms

    def _get_scattering_factors(self) -> np.ndarray:

        if self._scattering_factors is None:
            xp = get_array_module(self._device)
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

    def _get_chunk_infinite(self, first_slice: int, last_slice: int) -> Union['PotentialArray']:
        xp = get_array_module(self._device)

        scattering_factors = self._get_scattering_factors()
        sliced_atoms = self.get_sliced_atoms()

        atoms = sliced_atoms[first_slice: last_slice]
        shape = (last_slice - first_slice,) + self.gpts

        if len(atoms) == 0:
            array = xp.zeros(shape, dtype=xp.float32)
        else:
            array = infinite_potential_projections(atoms, shape, self.sampling, scattering_factors)

        potential = PotentialArray(array, slice_thickness=self.slice_thickness[first_slice:last_slice],
                                   extent=self.extent)
        return potential

    def _get_chunk_finite(self, first_slice: int, last_slice: int, return_atoms: bool = False):
        xp = get_array_module(self._device)

        if self._atomic_potentials is None:
            self._atomic_potentials = self._calculate_atomic_potentials()

        extent = np.diag(self.atoms.cell)[:2]

        sampling = (extent[0] / self.gpts[0], extent[1] / self.gpts[1])

        array = xp.zeros((last_slice - first_slice,) + self.gpts, dtype=np.float32)

        cutoffs = {Z: atomic_potential.cutoff for Z, atomic_potential in self._atomic_potentials.items()}
        sliced_atoms = SlicedAtoms(self.atoms,
                                   self._slice_thickness,
                                   plane=self._plane,
                                   box=self._box,
                                   padding=cutoffs)

        for i, slice_idx in enumerate(range(first_slice, last_slice)):
            for Z, atomic_potential in self._atomic_potentials.items():
                atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)

                a = sliced_atoms.slice_limits[slice_idx][0] - atoms.positions[:, 2]
                b = sliced_atoms.slice_limits[slice_idx][1] - atoms.positions[:, 2]

                atomic_potential.project_on_grid(array[i], sampling, atoms.positions, a, b)

        array -= array.min()
        return PotentialArray(array, slice_thickness=self.slice_thickness[first_slice:last_slice], extent=self.extent)

    def get_chunk(self, first_slice: int, last_slice: int) -> 'PotentialArray':
        if self.projection == 'infinite':
            return self._get_chunk_infinite(first_slice, last_slice)
        else:
            return self._get_chunk_finite(first_slice, last_slice)

    def get_frozen_phonon_potentials(self, lazy: bool = True):
        """
        Function to generate scattering potentials for a set of frozen phonon configurations.

        Returns
        -------
        generator
            Generator of potentials.
        """

        def potential_configuration(atoms):
            return self.__class__(atoms=atoms,
                                  gpts=self.gpts,
                                  sampling=self.sampling,
                                  slice_thickness=self.slice_thickness,
                                  parametrization=self.parametrization,
                                  projection=self.projection,
                                  chunks=self.chunks,
                                  precalculate=self.precalculate,
                                  device=self._device,
                                  plane=self.plane,
                                  box=self.box,
                                  origin=self.origin,
                                  cutoff_tolerance=self.cutoff_tolerance)

        potentials = []
        for atoms in self.frozen_phonons.get_frozen_phonon_atoms(lazy=lazy):
            potentials.append(potential_configuration(atoms))

        return potentials

    def _copy_as_dict(self, copy_atoms: bool = True):
        d = {'gpts': self.gpts,
             'sampling': self.gpts,
             'slice_thickness': self.slice_thickness,
             'parametrization': self.parametrization,
             'projection': self.projection,
             'chunks': self.chunks,
             'precalculate': self.precalculate,
             'device': self.device,
             'plane': self.plane,
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
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling, lock_gpts=True)
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

    def get_frozen_phonon_potentials(self, *args, **kwargs) -> List['PotentialArray']:
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
