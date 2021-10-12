"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Tuple

import dask
import dask.array as da
import numpy as np
import zarr
from ase import Atoms

from abtem.basic.antialias import antialias_kernel
from abtem.basic.backend import get_array_module, xp_to_str
from abtem.basic.complex import complex_exponential
from abtem.basic.dask import computable, HasDaskArray, requires_dask_array
from abtem.basic.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.basic.fft import fft2_convolve
from abtem.basic.grid import Grid, HasGridMixin
from abtem.basic.utils import generate_chunks
from abtem.measure.measure import Images
from abtem.potentials.atom import AtomicPotential
from abtem.potentials.infinite import infinite_potential_projections, calculate_scattering_factors
from abtem.potentials.temperature import AbstractFrozenPhonons, DummyFrozenPhonons
from abtem.structures.slicing import SliceIndexedAtoms, SlicedAtoms, _validate_slice_thickness
from abtem.structures.structures import is_cell_orthogonal, orthogonalize_cell


class AbstractPotential(HasGridMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """

    def __init__(self, slice_thickness):
        self._slice_thickness = slice_thickness

    # @property
    # @abstractmethod
    # def num_frozen_phonons(self) -> int:
    #     pass
    #
    # @property
    # @abstractmethod
    # def frozen_phonon_potentials(self):
    #     pass
    #
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
    def slice_limits(self):
        cumulative_thickness = np.cumsum(np.concatenate(((0,), self.slice_thickness)))
        return [(cumulative_thickness[i], cumulative_thickness[i + 1]) for i in range(len(cumulative_thickness) - 1)]

    @property
    def thickness(self) -> float:
        return sum(self._slice_thickness)

    def check_slice_idx(self, i):
        """Raises an error if i is greater than the number of slices."""
        if i >= self.num_slices:
            raise RuntimeError('Slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    def __getitem__(self, items):
        return self.build()[items]

    def project(self):
        return self.build().project()

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


class AbstractPotentialFromAtoms(AbstractPotential):

    def __init__(self,
                 atoms,
                 gpts,
                 sampling,
                 slice_thickness,
                 box=None,
                 plane='xy',
                 origin=(0., 0., 0.),
                 precalculate=False):

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('cell has no thickness')

        self._frozen_phonons = atoms

        # elif isinstance(atoms, AbstractFrozenPhonons):
        #     self._frozen_phonons = atoms.get_configurations()
        #
        # else:
        #     raise NotImplementedError()

        if isinstance(atoms, AbstractFrozenPhonons):
            self._frozen_phonons = atoms
        else:
            self._frozen_phonons = DummyFrozenPhonons(atoms)

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

        super().__init__(slice_thickness=slice_thickness)

    @property
    def precalculate(self):
        return self._precalculate

    @property
    def plane(self):
        return self._plane

    @property
    def box(self):
        return self._box

    @property
    def origin(self):
        return self._origin

    @AbstractPotential.slice_thickness.setter
    def slice_thickness(self, value):
        self._slice_thickness = _validate_slice_thickness(value, self._frozen_phonons[0].cell[2, 2])

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
        return len(self._frozen_phonons)


class ChunkIterator:

    def __init__(self, size, chunks=1):
        self._chunks = [x for x in generate_chunks(size, chunks=chunks)]
        self._i = 0

    def _get_chunk(self, first_slice, last_slice):
        pass

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i == len(self._chunks):
            raise StopIteration
        first_slice, last_slice = self._chunks[self._i]
        self._i += 1
        return self._get_chunk(first_slice, last_slice)


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
    storage : str, optional
        The device on which to store the created potential. The default is 'None', defaulting to the chosen device.
    """

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: Union[float, np.ndarray] = .5,
                 parametrization: str = 'lobato',
                 projection: str = 'infinite',
                 chunks: Union[int, str] = 'auto',
                 precalculate: bool = False,
                 device: str = 'cpu',
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 cutoff_tolerance: float = 1e-3):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization

        if projection not in ('finite', 'infinite'):
            raise RuntimeError('Projection must be "finite" or "infinite"')

        self._projection = projection
        self._atomic_potentials = None
        self._device = device

        super().__init__(atoms=atoms,
                         gpts=gpts,
                         sampling=sampling,
                         slice_thickness=slice_thickness,
                         plane=plane,
                         box=box,
                         origin=origin,
                         precalculate=precalculate)

        if chunks == 'auto':
            if precalculate:
                chunks = int(np.floor(256 / (4 * np.prod(self.gpts) / 1e6)))
            else:
                chunks = 1

        self._chunks = chunks

    @property
    def parametrization(self):
        """The potential parametrization."""
        return self._parametrization

    @property
    def projection(self):
        """The projection method."""
        return self._projection

    @property
    def cutoff_tolerance(self):
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    def _build_finite(self):

        def calculate_atomic_potentials():
            if self._atomic_potentials is None:
                core_size = min(self.sampling)
                self._atomic_potentials = {}
                for Z in np.unique(self.atoms.numbers):
                    self._atomic_potentials[Z] = AtomicPotential(Z, self.parametrization, core_size,
                                                                 cutoff_tolerance=self.cutoff_tolerance)
                    self._atomic_potentials[Z].build_integral_table()

            return self._atomic_potentials

        def calculate_potential_chunk(first_slice,
                                      last_slice,
                                      atoms,
                                      slice_thickness,
                                      plane,
                                      box,
                                      atomic_potentials):

            array = xp.zeros((last_slice - first_slice,) + self.gpts, dtype=np.float32)

            cutoffs = {Z: atomic_potential.cutoff for Z, atomic_potential in atomic_potentials.items()}
            sliced_atoms = SlicedAtoms(atoms, slice_thickness, plane=plane, box=box, padding=cutoffs)

            for i, slice_idx in enumerate(range(first_slice, last_slice)):
                for Z, atomic_potential in atomic_potentials.items():
                    atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)
                    a = sliced_atoms.slice_limits[slice_idx][0] - atoms.positions[:, 2]
                    b = sliced_atoms.slice_limits[slice_idx][1] - atoms.positions[:, 2]
                    atomic_potential.project_on_grid(array[i], self.sampling, atoms.positions, a, b)

            return array

        xp = get_array_module(self._device)
        atomic_potentials = dask.delayed(calculate_atomic_potentials, pure=True)()

        array = []
        for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
            shape = (last_slice - first_slice,) + self.gpts
            chunk = dask.delayed(calculate_potential_chunk)(first_slice,
                                                            last_slice,
                                                            self.atoms,
                                                            self.slice_thickness,
                                                            self.plane,
                                                            self.box,
                                                            atomic_potentials)

            array.append(da.from_delayed(chunk, shape=shape, meta=xp.array((), dtype=np.float32)))

        return PotentialArray(da.concatenate(array), self.slice_thickness, extent=self.extent)

    def _scattering_factors(self):
        xp = get_array_module(self._device)
        return calculate_scattering_factors(self.gpts, self.sampling, np.unique(self.atoms.numbers), xp_to_str(xp))

    @computable
    def build(self):
        self.grid.check_is_defined()
        # self.frozen_phonon_potentials()[0]

    def _frozen_phonon_potentials_infinite(self, precalculate):
        scattering_factors = dask.delayed(self._scattering_factors)()

        class InfinitePotentialIterator(ChunkIterator):

            def __init__(self, sliced_atoms, scattering_factors, gpts, sampling, chunks=1):
                self._sliced_atoms = sliced_atoms
                self._scattering_factors = scattering_factors
                self._gpts = gpts
                self._sampling = sampling
                super().__init__(size=len(sliced_atoms), chunks=chunks)

            def _get_chunk(self, first_slice, last_slice):
                positions, numbers, slice_idx = self._sliced_atoms.get_atoms_in_slices(first_slice, last_slice)
                shape = (last_slice - first_slice,) + self._gpts
                return infinite_potential_projections(positions, numbers, slice_idx, shape, self._sampling,
                                                      self._scattering_factors)

        def slice_atoms(configuration):
            return SliceIndexedAtoms(configuration, self.num_slices)

        def calculate_potential_chunk(first_slice, last_slice, slice_index_atoms, scattering_factors):
            positions, numbers, slice_idx = slice_index_atoms.get_atoms_in_slices(first_slice, last_slice)
            shape = (last_slice - first_slice,) + self.gpts
            return infinite_potential_projections(positions, numbers, slice_idx, shape, self.sampling,
                                                  scattering_factors)

        def potential_generator(sliced_atoms, scattering_factors):
            return InfinitePotentialIterator(sliced_atoms, scattering_factors, gpts=self.gpts, sampling=self.sampling,
                                             chunks=self._chunks)

        def potential_builder(slice_index_atoms, scattering_factors):
            xp = get_array_module(self._device)

            array = []
            for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
                chunk = dask.delayed(calculate_potential_chunk)(first_slice,
                                                                last_slice,
                                                                slice_index_atoms,
                                                                scattering_factors)

                shape = (last_slice - first_slice,) + self.gpts
                array.append(da.from_delayed(chunk, shape=shape, meta=xp.array((), dtype=np.float32)))

            return PotentialArray(da.concatenate(array), self.slice_thickness, extent=self.extent)

        potentials = []
        for configuration in self.frozen_phonons.get_configurations():
            sliced_atoms = dask.delayed(slice_atoms)(configuration)

            if precalculate:
                potentials.append(potential_builder(sliced_atoms, scattering_factors))
            else:
                gen = dask.delayed(potential_generator)(sliced_atoms, scattering_factors)
                potentials.append(PotentialGenerator(gen,
                                                     slice_thickness=self.slice_thickness,
                                                     extent=self.extent))

        return potentials

    def frozen_phonon_potentials(self):
        """
        Function to generate scattering potentials for a set of frozen phonon configurations.

        Returns
        -------
        generator
            Generator of potentials.
        """

        if self._projection == 'finite':
            raise NotImplementedError
        else:
            return self._frozen_phonon_potentials_infinite(self.precalculate)

    def __copy__(self):
        return self.__class__(atoms=self.frozen_phonons.copy(),
                              gpts=self.gpts,
                              slice_thickness=self.slice_thickness,
                              parametrization=self.parametrization,
                              cutoff_tolerance=self.cutoff_tolerance,
                              device=self.device)


class PotentialGenerator(AbstractPotential, HasGridMixin):

    def __init__(self, iterator, slice_thickness, gpts=None, sampling=None, extent=None):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._iterator = iterator
        super().__init__(slice_thickness=slice_thickness)

    @property
    def array(self):
        return self._iterator

    def build(self):
        pass


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
                 slice_thickness: Union[float, Sequence[float]] = None,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):

        if len(array.shape) != 3:
            raise RuntimeError(f'PotentialArray must be 3d, not {len(array.shape)}d')

        self._array = array
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling, lock_gpts=True)
        slice_thickness = _validate_slice_thickness(slice_thickness, num_slices=array.shape[0])

        super().__init__(slice_thickness=slice_thickness)

    def build(self):
        return self

    def split_chunks(self):
        return [self[a:a + b] for a, b in zip(np.cumsum((0,) + self.array.chunks[0]), self.array.chunks[0])]

    def __getitem__(self, items):
        if isinstance(items, (int, slice)):
            potential_array = self.array[items]
            if len(potential_array.shape) == 2:
                potential_array = potential_array[None]

            return self.__class__(potential_array, self.slice_thickness[items], extent=self.extent)
        else:
            raise TypeError('Potential must be indexed with integers or slices, not {}'.format(type(items)))

    def transmission_function(self, energy: float, antialias=True):
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

        if self.is_delayed:
            array = self._array.map_blocks(_transmission_function, energy=energy, meta=xp.array((), dtype=xp.complex64))

        else:
            array = _transmission_function(self._array, energy=energy)

        t = TransmissionFunction(array, slice_thickness=self.slice_thickness.copy(), extent=self.extent, energy=energy)
        return t

    @property
    def num_frozen_phonons(self):
        return 1

    def frozen_phonon_potentials(self):
        return [self]

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
        new_slice_thickness = np.tile(self.slice_thickness, tile[2])

        return self.__class__(array=new_array, slice_thickness=new_slice_thickness, extent=new_extent)

    def to_zarr(self, url, overwrite=False):
        """
        Write potential to a zarr file.

        Parameters
        ----------
        url: str
            url to which the data is saved.
        """

        self.array.to_zarr(url, component='array', overwrite=overwrite)

        with zarr.open(url, mode='a') as f:
            f.create_dataset('slice_thickness', data=self.slice_thickness, overwrite=overwrite)
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
            slice_thickness = f['slice_thickness'][:]
            extent = f['extent'][:]

        array = da.from_zarr(url, component='array', chunks=(chunks, -1, -1))
        return cls(array=array, slice_thickness=slice_thickness, extent=extent)

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
        return Images(array=self._array.sum(0), sampling=self.sampling)

    def __copy___(self):
        return self.__class__(array=self.array.copy(),
                              slice_thickness=self.slice_thickness.copy(),
                              extent=self.extent)


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions."""

    def __init__(self,
                 array: np.ndarray,
                 slice_thickness: Union[float, Sequence[float]],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None):

        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thickness, extent, sampling)

    def __getitem__(self, items):
        if isinstance(items, (int, slice)):
            array = self.array[items]
            if len(array.shape) == 2:
                array = array[None]

            return self.__class__(array, self.slice_thickness[items], extent=self.extent, energy=self.energy)
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
