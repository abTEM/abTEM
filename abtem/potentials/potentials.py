"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Callable

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from ase import Atoms

from abtem.base_classes import Grid, HasGridMixin, HasAcceleratorMixin, Accelerator, watched_property, HasEventMixin
from abtem.device import HasDeviceMixin, get_available_memory
from abtem.measure.old_measure import Measurement
from abtem.potentials.infinite import infinite_potential_projections, scattering_factor
from abtem.structures.slicing import SliceIndexedAtoms
from abtem.structures.structures import is_cell_orthogonal
from abtem.temperature import AbstractFrozenPhonons, DummyFrozenPhonons
from abtem.utils import energy2sigma, ProgressBar, generate_batches
from abtem.utils.antialias import AntialiasFilter
from abtem.utils.complex import complex_exponential


class AbstractPotential(HasGridMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """

    def __len__(self):
        return self.num_slices

    @property
    @abstractmethod
    def num_slices(self):
        """The number of projected potential slices."""
        pass

    @property
    @abstractmethod
    def num_frozen_phonons(self):
        pass

    @property
    @abstractmethod
    def frozen_phonon_potentials(self):
        pass

    @property
    def thickness(self):
        return sum([self.get_slice_thickness(i) for i in range(len(self))])

    def check_slice_idx(self, i):
        """Raises an error if i is greater than the number of slices."""
        if i >= self.num_slices:
            raise RuntimeError('Slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    @abstractmethod
    def get_slice_thickness(self, i):
        """
        Get the slice thickness [Å].

        Parameters
        ----------
        i: int
            Slice index.
        """
        pass

    @abstractmethod
    def project(self):
        pass

    def plot(self, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for abtem.plot.show_image.
        """
        return self.project().abtem.plot(**kwargs)

    def copy(self):
        """Make a copy."""
        return copy(self)


class AbstractPotentialBuilder(AbstractPotential):
    """Potential builder abstract class."""

    def __init__(self, chunk_size: int = 1, device='cpu', storage='cpu'):
        self._chunk_size = chunk_size
        self._storage = storage
        self._device = device
        super().__init__()

    @property
    def storage(self):
        return self._storage

    @property
    def device(self):
        return self._device

    def _estimate_max_batch(self):
        memory_per_wave = 2 * 4 * self.gpts[0] * self.gpts[1]
        available_memory = .2 * get_available_memory(self._device)
        return min(int(available_memory / memory_per_wave), len(self))

    def __getitem__(self, items):
        return self.build()[items]

    @abstractmethod
    def build(self):
        pass

    def project(self):
        return self[:].compute().project()

    def show(self, **kwargs):
        return self.project().show(**kwargs)


class Potential(AbstractPotentialBuilder, HasDeviceMixin, HasEventMixin):
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
    storage : str, optional
        The device on which to store the created potential. The default is 'None', defaulting to the chosen device.
    """

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 slice_thickness: float = .5,
                 parametrization: str = 'lobato',
                 projection: str = 'finite',
                 device: str = 'cpu',
                 chunk_size: int = 1,
                 cutoff_tolerance: float = 1e-3,
                 z_periodic: bool = True,
                 storage: str = None):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness

        # if projection.lower() == 'finite':
        #     self._builder = InfinitePotentialProjections(atoms=atoms, gpts=gpts, sampling=sampling,
        #                                                  slice_thickness=slice_thickness)

        if projection not in ('finite', 'infinite'):
            raise RuntimeError('Projection must be "finite" or "infinite"')

        self._projection = projection

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('cell has no thickness')

        if not is_cell_orthogonal(atoms):
            raise RuntimeError('atoms are not orthogonal')

        if isinstance(atoms, AbstractFrozenPhonons):
            self._frozen_phonons = atoms
        else:
            self._frozen_phonons = DummyFrozenPhonons(atoms)

        self._grid = Grid(extent=np.diag(atoms.cell)[:2], gpts=gpts, sampling=sampling, lock_extent=True)

        super().__init__(chunk_size=chunk_size, device=device)

    @property
    def parametrization(self):
        """The potential parametrization."""
        return self._parameters

    @property
    def projection(self):
        """The projection method."""
        return self._projection

    @property
    def parameters(self):
        """The parameters of the potential parametrization."""
        return self._parameters

    @property
    def atoms(self):
        """Atoms object defining the atomic configuration."""
        return self._frozen_phonons.atoms

    @property
    def frozen_phonons(self):
        """FrozenPhonons object defining the atomic configuration(s)."""
        return self._frozen_phonons

    @property
    def num_frozen_phonons(self):
        return len(self.frozen_phonons)

    @property
    def cutoff_tolerance(self):
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def num_slices(self):
        """The number of projected potential slices."""
        return int(np.ceil(self.atoms.cell[2, 2] / self._slice_thickness))

    @property
    def slice_thickness(self):
        """The thickness of the projected potential slices."""
        return self._slice_thickness

    @slice_thickness.setter
    @watched_property('_event')
    def slice_thickness(self, value):
        self._slice_thickness = value

    def get_slice_thickness(self, i) -> float:
        return self.atoms.cell[2, 2] / self.num_slices

    def get_parameterized_function(self, number) -> Callable:
        return lambda r: self._function(r, self.parameters[number])

    def build(self):
        atoms = self.atoms

        slice_thicknesses = [self.get_slice_thickness(i) for i in range(len(self))]
        slice_index_atoms = SliceIndexedAtoms(atoms, self.num_slices)

        scattering_factors = {}
        for number in np.unique(self.atoms.numbers):
            scattering_factors[number] = scattering_factor(self.gpts, self.sampling, number, np)  # .compute()

        array = []
        for first_slice, last_slice in generate_batches(len(self), max_batch=self._chunk_size, start=0):
            positions, numbers, slice_idx = slice_index_atoms.get_atoms_in_slices(first_slice, last_slice)
            shape = (last_slice - first_slice,) + self.gpts

            chunk = dask.delayed(infinite_potential_projections)(positions, numbers, slice_idx, shape, self.sampling,
                                                                 scattering_factors)

            array.append(da.from_delayed(chunk, shape=shape, dtype=np.float32))

        return PotentialArray(da.concatenate(array), slice_thicknesses, extent=self.extent)

    def frozen_phonon_potentials(self):
        """
        Function to generate scattering potentials for a set of frozen phonon configurations.

        Returns
        -------
        generator
            Generator of potentials.
        """

        potentials = []


        for atoms in self.frozen_phonons:
            potential = Potential(atoms, slice_thickness=self.slice_thickness, gpts=self.gpts,
                                  chunk_size=self._chunk_size)
            potentials.append(potential)

        return potentials

    def __copy__(self):
        return self.__class__(atoms=self.frozen_phonons.copy(),
                              gpts=self.gpts,
                              slice_thickness=self.slice_thickness,
                              parametrization=self.parametrization,
                              cutoff_tolerance=self.cutoff_tolerance,
                              device=self.device,
                              storage=self._storage)


class PotentialArray(AbstractPotential, HasGridMixin):
    """
    Potential array object

    The potential array represents slices of the electrostatic potential as an array.

    Parameters
    ----------
    array: 3D array
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thicknesses: float
        The thicknesses of potential slices in Å. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
    extent: one or two float, optional
        Lateral extent of the potential [Å].
    sampling: one or two float, optional
        Lateral sampling of the potential [1 / Å].
    """

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 slice_thicknesses: Union[float, Sequence[float]] = None,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):

        if (len(array.shape) != 2) & (len(array.shape) != 3):
            raise RuntimeError(f'PotentialArray must be 2d or 3d, not {len(array.shape)}d')

        self._array = array

        slice_thicknesses = np.array(slice_thicknesses)

        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, array.shape[0])

        if (slice_thicknesses.shape != (array.shape[0],)) & (len(array.shape) == 3):
            raise ValueError()

        self._slice_thicknesses = slice_thicknesses
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling, lock_gpts=True)

        super().__init__()

    @property
    def array(self):
        """The potential array."""
        return self._array

    def compute(self):
        self._array = self.array.compute()
        return self

    def delayed(self, chunks=1):
        if isinstance(self.array, da.core.Array):
            return self

        array = da.from_array(self.array, name=str(id(self.array)), chunks=(chunks, -1, -1))
        return self.__class__(array=array, slice_thicknesses=self.slice_thicknesses.copy(), extent=self.extent)

    def split_chunks(self):
        return [self[a:a + b] for a, b in zip(np.cumsum((0,) + self.array.chunks[0]), self.array.chunks[0])]

    def __getitem__(self, items):
        if isinstance(items, (int, slice)):
            potential_array = self.array[items]
            if len(potential_array.shape) == 2:
                potential_array = potential_array[None]

            return self.__class__(potential_array, self._slice_thicknesses[items], extent=self.extent)
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def transmission_function(self, energy: float):
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

        array = complex_exponential(np.float32(energy2sigma(energy)) * self._array)

        array = AntialiasFilter()(array, sampling=self.sampling)


        t = TransmissionFunction(array, slice_thicknesses=self._slice_thicknesses.copy(), extent=self.extent,
                                 energy=energy)

        return t

    @property
    def num_frozen_phonons(self):
        return 1

    def frozen_phonon_potentials(self):
        return [self]

    @property
    def num_slices(self):
        return self._array.shape[0]

    def get_slice_thickness(self, i):
        return self._slice_thicknesses[i]

    @property
    def slice_thicknesses(self):
        return self._slice_thicknesses

    @property
    def thickness(self):
        return np.sum(self._slice_thicknesses)

    def tile(self, tile):
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

        if len(tile) == 2:
            tile = tuple(tile) + (1,)

        new_array = np.tile(self.array, (tile[2], tile[0], tile[1]))

        new_extent = (self.extent[0] * tile[0], self.extent[1] * tile[1])
        new_slice_thicknesses = np.tile(self._slice_thicknesses, tile[2])

        return self.__class__(array=new_array, slice_thicknesses=new_slice_thicknesses, extent=new_extent)

    def to_zarr(self, url, overwrite=False):
        """
        Write potential to a zarr file.

        Parameters
        ----------
        url: str
            url to which the data is saved.
        """

        self.array.to_zarr(url, component='array', overwrite=overwrite)

        with zarr.open('test.zarr', mode='a') as f:
            f.create_dataset('slice_thicknesses', data=self.slice_thicknesses, overwrite=overwrite)
            f.create_dataset('extent', data=self.extent, overwrite=overwrite)

    @classmethod
    def from_zarr(cls, url, chunks=1):
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
            slice_thicknesses = f['slice_thicknesses'][:]
            extent = f['extent'][:]

        array = da.from_zarr(url, component='array', chunks=(chunks, -1, -1))
        return cls(array=array, slice_thicknesses=slice_thicknesses, extent=extent)

    def transmit(self, waves):
        """
        Transmit a wavefunction.

        Parameters
        ----------
        waves: Waves object
            Wavefunction to transmit.

        Returns
        -------
        TransmissionFunction
        """
        return self.transmission_function(waves.energy).transmit(waves)

    def project(self):
        """
        Create a 2d xarray representing a measurement of the projected potential.

        Returns
        -------
        Measurement
        """
        coords = self.grid.coords()
        measurement = xr.DataArray(self.array.sum(0), coords=coords, dims=['x', 'y'], name='Projected potential',
                                   attrs={'units': 'eV Å / e'})

        return measurement

    def __copy___(self):
        return self.__class__(array=self.array.copy(),
                              slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent)


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions."""

    def __init__(self,
                 array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence[float]],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None):

        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thicknesses, extent, sampling)

    def __getitem__(self, items):
        if isinstance(items, (int, slice)):
            array = self.array[items]
            if len(array.shape) == 2:
                array = array[None]

            return self.__class__(array, self._slice_thicknesses[items], extent=self.extent, energy=self.energy)
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def delayed(self, chunks=1):
        new = super().delayed(chunks=chunks)
        new.energy = self.energy
        return new

    def transmission_function(self, energy):
        if energy != self.energy:
            raise RuntimeError()
        return self

    def transmit(self, waves):
        self.accelerator.check_match(waves)
        # try:
        # print(type(waves._array), type(self.array)
        waves._array *= self.array
        # except AttributeError:
        #    waves = self.transmit(waves.build())

        return waves
