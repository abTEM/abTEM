"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Callable, Generator

import h5py
import numpy as np
from ase import Atoms
from ase import units
from scipy.optimize import brentq

from abtem.base_classes import Grid, HasGridMixin, Cache, cached_method, HasAcceleratorMixin, Accelerator, Event, \
    watched_property, AntialiasFilter
from abtem.device import get_device_function, get_array_module, get_array_module_from_device, copy_to_device, \
    HasDeviceMixin, asnumpy, get_available_memory
from abtem.measure import calibrations_from_grid, Measurement
from abtem.parametrizations import kirkland, dvdr_kirkland, load_kirkland_parameters, kirkland_projected_fourier
from abtem.parametrizations import lobato, dvdr_lobato, load_lobato_parameters
from abtem.structures import is_cell_orthogonal
from abtem.tanh_sinh import integrate, tanh_sinh_nodes_and_weights
from abtem.temperature import AbstractFrozenPhonons, DummyFrozenPhonons
from abtem.utils import energy2sigma, ProgressBar, generate_batches, _disc_meshgrid
from abtem.structures import pad_atoms

# Vacuum permitivity in ASE units
eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

# Conversion factor from unitless potential parametrizations to ASE potential units
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


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

    def check_slice_idx(self, i):
        """Raises an error if i is greater than the number of slices."""
        if i >= self.num_slices:
            raise RuntimeError('Slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
        """
        Generate the transmission functions one slice at a time.

        Parameters
        ----------
        energy: float
            Electron energy [eV].
        first_slice: int
            First potential slice to generate.
        last_slice: int, optional
            Last potential slice generate.
        max_batch: int
            Maximum number of potential slices calculated in parallel.

        Returns
        -------
        generator of PotentialArray objects
        """
        antialias_filter = AntialiasFilter()

        for start, end, potential_slice in self.generate_slices(first_slice, last_slice, max_batch=max_batch):
            yield start, end, potential_slice.as_transmission_function(energy,
                                                                       in_place=True,
                                                                       max_batch=max_batch,
                                                                       antialias_filter=antialias_filter)

    @abstractmethod
    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1):
        """
        Generate the potential slices.

        Parameters
        ----------
        first_slice: int
            First potential slice to generate.
        last_slice: int, optional
            Last potential slice generate.
        max_batch: int
            Maximum number of potential slices calculated in parallel.

        Returns
        -------
        generator of PotentialArray objects
        """

        pass

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

    def __iter__(self):
        for _, __, p in self.generate_slices(max_batch=1):
            yield p

    # def show(self, **kwargs):
    #     """
    #     Show the potential projection. This requires building all potential slices.
    #
    #     Parameters
    #     ----------
    #     kwargs:
    #         Additional keyword arguments for abtem.plot.show_image.
    #     """
    #
    #     self[:].show(**kwargs)

    def copy(self):
        """Make a copy."""
        return copy(self)


class AbstractPotentialBuilder(AbstractPotential):
    """Potential builder abstract class."""

    def __init__(self, precalculate=True, device='cpu', storage='cpu'):
        self._precalculate = precalculate
        self._storage = storage
        self._device = device

    def _estimate_max_batch(self):
        memory_per_wave = 2 * 4 * self.gpts[0] * self.gpts[1]
        available_memory = .2 * get_available_memory(self._device)
        return min(int(available_memory / memory_per_wave), len(self))

    def __getitem__(self, items):
        if isinstance(items, int):
            if items >= self.num_slices:
                raise StopIteration
            return next(self.generate_slices(items, items + 1, max_batch=1))[2]

        elif isinstance(items, slice):
            if items.start is None:
                first_slice = 0
            else:
                first_slice = items.start

            if items.stop is None:
                last_slice = len(self)
            else:
                last_slice = items.stop

            if items.step is None:
                step = 1
            else:
                step = items.step

            return self.build(first_slice, last_slice, pbar=False)[::step]
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              energy: float = None,
              max_batch: int = None,
              pbar: Union[bool, ProgressBar] = False,
              ) -> 'PotentialArray':
        """
        Precalcaulate the potential as a potential array.

        Parameters
        ----------
        first_slice: int
            First potential slice to generate.
        last_slice: int, optional
            Last potential slice generate.
        energy: float
            Electron energy [eV]. If given, the transmission functions will be returned.
        max_batch: int
            Maximum number of potential slices calculated in parallel.
        pbar: bool
            If true, show progress bar.

        Returns
        -------
        PotentialArray object
        """

        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if max_batch is None:
            max_batch = self._estimate_max_batch()

        storage_xp = get_array_module_from_device(self._storage)

        if energy is None:
            array = storage_xp.zeros((last_slice - first_slice,) + (self.gpts[0], self.gpts[1]), dtype=np.float32)
            generator = self.generate_slices(max_batch=max_batch, first_slice=first_slice, last_slice=last_slice)
        else:
            array = storage_xp.zeros((last_slice - first_slice,) + (self.gpts[0], self.gpts[1]), dtype=np.complex64)
            generator = self.generate_transmission_functions(energy=energy,
                                                             max_batch=max_batch,
                                                             first_slice=first_slice,
                                                             last_slice=last_slice)

        slice_thicknesses = np.zeros(last_slice - first_slice)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)
            close_pbar = True
        else:
            close_pbar = False

        pbar.reset()
        for start, end, potential_slice in generator:
            array[start:end] = copy_to_device(potential_slice.array, self._storage)
            slice_thicknesses[start:end] = potential_slice.slice_thicknesses
            pbar.update(end - start)

        pbar.refresh()

        if close_pbar:
            pbar.close()

        if energy is None:
            return PotentialArray(array, slice_thicknesses=slice_thicknesses, extent=self.extent)
        else:
            return TransmissionFunction(array, slice_thicknesses=slice_thicknesses, extent=self.extent, energy=energy)

    def project(self):
        projected = self[0]
        max_batch = self._estimate_max_batch()
        for _, _, projected_chunk in self.generate_slices(max_batch=max_batch):
            projected._array += projected_chunk.array.sum(0)
        return projected.project()


class AbstractTDSPotentialBuilder(AbstractPotentialBuilder):
    """Thermal diffuse scattering builder abstract class."""

    def __init__(self, precalculate=True, device='cpu', storage='cpu'):
        super().__init__(precalculate=precalculate, device=device, storage=storage)

    @property
    @abstractmethod
    def frozen_phonons(self):
        pass

    @abstractmethod
    def generate_frozen_phonon_potentials(self, pbar):
        pass


class PotentialIntegrator:
    """
    Perform finite integrals of a radial function along a straight line.

    Parameters
    ----------
    function: callable
        Radial function to integrate.
    r: array of float
        The evaluation points of the integrals.
    cutoff: float, optional
        The radial function is assumed to be zero outside this threshold.
    cache_size: int, optional
        The maximum number of integrals that will be cached.
    cache_key_decimals: int, optional
        The number of decimals used in the cache keys.
    tolerance: float, optional
        The absolute error tolerance of the integrals.
    """

    def __init__(self,
                 function: Callable,
                 r: np.ndarray,
                 max_interval,
                 cutoff: float = None,
                 cache_size: int = 4096,
                 cache_key_decimals: int = 2,
                 tolerance: float = 1e-6):

        self._function = function
        self._r = r

        if cutoff is None:
            self._cutoff = r[-1]
        else:
            self._cutoff = cutoff

        self._cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals
        self._tolerance = tolerance

        def f(z):
            return self._function(np.sqrt(self.r[0] ** 2 + (z * max_interval / 2 + max_interval / 2) ** 2))

        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tolerance)

        self._xk, self._wk = tanh_sinh_nodes_and_weights(step_size, order)

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._cutoff

    def integrate(self, z, a: float, b: float, xp):
        """
        Evaulate the integrals of the radial function at the evaluation points.

        Parameters
        ----------
        a: float
            Lower limit of integrals.
        b: float
            Upper limit of integrals.

        Returns
        -------
        1d array
            The evaulated integrals.
        """

        vr = np.zeros((len(z), self.r.shape[0]), np.float32)
        dvdr = np.zeros((len(z), self.r.shape[0]), np.float32)
        a = np.round(a - z, self._cache_key_decimals)
        b = np.round(b - z, self._cache_key_decimals)

        split = a * b < 0

        a, b = np.abs(a), np.abs(b)
        a, b = np.minimum(a, b), np.minimum(np.maximum(a, b), self.cutoff)

        for i, (ai, bi) in enumerate(zip(a, b)):
            if split[i]:  # split the integral
                values1, derivatives1 = self._do_integrate(0, ai)
                values2, derivatives2 = self._do_integrate(0, bi)
                result = (values1 + values2, derivatives1 + derivatives2)
            else:
                result = self._do_integrate(ai, bi)

            vr[i] = result[0]
            dvdr[i, :-1] = result[1]

        return vr, dvdr

    @cached_method('_cache')
    def _do_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.

        def f(z):
            return self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))

        values = np.sum(f(self._xk[None]) * self._wk[None], axis=1) * zm
        derivatives = np.diff(values) / np.diff(self.r)

        return values, derivatives


def superpose_deltas(positions, z, array):
    shape = array.shape[-2:]
    xp = get_array_module(array)
    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[z, rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[z, (rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[z, rows, (cols + 1) % shape[1]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[z, (rows + 1) % shape[0], (cols + 1) % shape[1]] += (rows - positions[:, 0]) * (cols - positions[:, 1])


class Potential(AbstractTDSPotentialBuilder, HasDeviceMixin):
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
    precalculate : bool
        If True, precalculate the potential else the potential will be calculated on-the-fly and immediately discarded.
        Default is True.
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
                 cutoff_tolerance: float = 1e-3,
                 device='cpu',
                 precalculate: bool = True,
                 storage=None):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness

        self._storage = storage

        if parametrization.lower() == 'lobato':
            self._parameters = load_lobato_parameters()
            self._function = lobato
            self._derivative = dvdr_lobato

        elif parametrization.lower() == 'kirkland':
            self._parameters = load_kirkland_parameters()
            self._function = kirkland
            self._derivative = dvdr_kirkland
        else:
            raise RuntimeError('Parametrization {} not recognized'.format(parametrization))

        if projection == 'infinite':
            if parametrization.lower() != 'kirkland':
                raise RuntimeError('Infinite projections are only implemented for the Kirkland parametrization')
        elif (projection != 'finite'):
            raise RuntimeError('Projection must be "finite" or "infinite"')

        self._projection = projection

        if isinstance(atoms, AbstractFrozenPhonons):
            self._frozen_phonons = atoms
        else:
            self._frozen_phonons = DummyFrozenPhonons(atoms)

        atoms = next(iter(self._frozen_phonons))

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('Atoms cell has no thickness')

        if not is_cell_orthogonal(atoms):
            raise RuntimeError('Atoms are not orthogonal')

        self._atoms = atoms
        self._grid = Grid(extent=np.diag(atoms.cell)[:2], gpts=gpts, sampling=sampling, lock_extent=True)

        self._cutoffs = {}
        self._integrators = {}
        self._disc_indices = {}

        def grid_changed_callback(*args, **kwargs):
            self._integrators = {}
            self._disc_indices = {}

        self.grid.changed.register(grid_changed_callback)
        self.changed = Event()

        if storage is None:
            storage = device

        super().__init__(precalculate=precalculate, device=device, storage=storage)

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
    def function(self):
        """The potential function of the parametrization."""
        return self._function

    @property
    def atoms(self):
        """Atoms object defining the atomic configuration."""
        return self._atoms

    @property
    def frozen_phonons(self):
        """FrozenPhonons object defining the atomic configuration(s)."""
        return self._frozen_phonons

    @property
    def cutoff_tolerance(self):
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def num_slices(self):
        """The number of projected potential slices."""
        return int(np.ceil(self._atoms.cell[2, 2] / self._slice_thickness))

    @property
    def slice_thickness(self):
        """The thickness of the projected potential slices."""
        return self._slice_thickness

    @slice_thickness.setter
    @watched_property('changed')
    def slice_thickness(self, value):
        self._slice_thickness = value

    def get_slice_thickness(self, i):
        return self._atoms.cell[2, 2] / self.num_slices

    def get_cutoff(self, number: int) -> float:
        """
        Cutoff distance for atomic number given an error tolerance.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        cutoff: float
            The potential cutoff.
        """

        try:
            return self._cutoffs[number]
        except KeyError:
            def f(r):
                return self.function(r, self.parameters[number]) - self.cutoff_tolerance

            self._cutoffs[number] = brentq(f, 1e-7, 1000)
            return self._cutoffs[number]

    def get_tapered_function(self, number: int) -> Callable:
        """
        Tapered potential function for atomic number.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        callable
        """

        cutoff = self.get_cutoff(number)
        rolloff = .85 * cutoff

        def soft_function(r):
            result = np.zeros_like(r)
            valid = r < cutoff
            transition = valid * (r > rolloff)
            result[valid] = self._function(r[valid], self.parameters[number])
            result[transition] *= (np.cos(np.pi * (r[transition] - rolloff) / (cutoff - rolloff)) + 1.) / 2
            return result

        return soft_function

    def get_integrator(self, number: int) -> PotentialIntegrator:
        """
        Potential integrator for atomic number.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        PotentialIntegrator object
        """

        try:
            return self._integrators[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            soft_function = self.get_tapered_function(number)
            inner_cutoff = np.min(self.sampling) / 2.

            num_points = int(np.ceil(cutoff / np.min(self.sampling) * 10.))
            r = np.geomspace(inner_cutoff, cutoff, num_points)
            max_interval = self.slice_thickness
            self._integrators[number] = PotentialIntegrator(soft_function, r, max_interval, cutoff)
            return self._integrators[number]

    def _get_radial_interpolation_points(self, number):
        """Internal function for the indices of the radial interpolation points."""
        try:
            return self._disc_indices[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            margin = np.int(np.ceil(cutoff / np.min(self.sampling)))
            rows, cols = _disc_meshgrid(margin)
            self._disc_indices[number] = np.hstack((rows[:, None], cols[:, None]))
            return self._disc_indices[number]

    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if self.projection == 'finite':
            return self._generate_slices_finite(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)
        else:
            return self._generate_slices_infinite(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)

    def _generate_slices_infinite(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        xp = get_array_module_from_device(self._device)

        fft2_convolve = get_device_function(xp, 'fft2_convolve')

        atoms = self.atoms.copy()
        atoms.wrap()
        positions = atoms.get_positions().astype(np.float32)
        numbers = atoms.get_atomic_numbers()
        unique = np.unique(numbers)
        order = np.argsort(positions[:, 2])

        positions = positions[order]
        numbers = numbers[order]

        kx = xp.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = xp.fft.fftfreq(self.gpts[1], self.sampling[1])
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')
        k = xp.sqrt(kx ** 2 + ky ** 2)

        sinc = xp.sinc(xp.sqrt((kx * self.sampling[0]) ** 2 + (kx * self.sampling[1]) ** 2))

        scattering_factors = {}
        for atomic_number in unique:
            f = kirkland_projected_fourier(k, self.parameters[atomic_number])
            scattering_factors[atomic_number] = (f / (sinc * self.sampling[0] * self.sampling[1] * kappa)).astype(
                xp.complex64)

        slice_idx = np.floor(positions[:, 2] / atoms.cell[2, 2] * self.num_slices).astype(np.int)

        start, end = next(generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice))

        array = xp.zeros((end - start,) + self.gpts, dtype=xp.complex64)
        temp = xp.zeros((end - start,) + self.gpts, dtype=xp.complex64)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            array[:] = 0.
            start_idx = np.searchsorted(slice_idx, start)
            end_idx = np.searchsorted(slice_idx, end)

            if start_idx != end_idx:
                for j, number in enumerate(unique):
                    temp[:] = 0.
                    chunk_positions = positions[start_idx:end_idx]
                    chunk_slice_idx = slice_idx[start_idx:end_idx] - start

                    if len(unique) > 1:
                        chunk_positions = chunk_positions[numbers[start_idx:end_idx] == number]
                        chunk_slice_idx = chunk_slice_idx[numbers[start_idx:end_idx] == number]

                    chunk_positions = xp.asarray(chunk_positions[:, :2] / self.sampling)

                    superpose_deltas(chunk_positions, chunk_slice_idx, temp)
                    fft2_convolve(temp, scattering_factors[number])

                    array += temp

            slice_thicknesses = [self.get_slice_thickness(i) for i in range(start, end)]
            yield start, end, PotentialArray(array.real[:end - start], slice_thicknesses, extent=self.extent)

    def _generate_slices_finite(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        xp = get_array_module_from_device(self._device)

        interpolate_radial_functions = get_device_function(xp, 'interpolate_radial_functions')

        atoms = self.atoms.copy()
        atoms.wrap()
        indices_by_number = {number: np.where(atoms.numbers == number)[0] for number in np.unique(atoms.numbers)}

        start, end = next(generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice))
        array = xp.zeros((end - start,) + self.gpts, dtype=xp.float32)

        slice_edges = np.linspace(0, self.atoms.cell[2, 2], self.num_slices + 1)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            array[:] = 0.

            for number, indices in indices_by_number.items():
                species_atoms = atoms[indices]
                integrator = self.get_integrator(number)
                disc_indices = xp.asarray(self._get_radial_interpolation_points(number))

                a = slice_edges[start]
                b = slice_edges[end]
                chunk_atoms = species_atoms[(species_atoms.positions[:, 2] > a - integrator.cutoff) *
                                            (species_atoms.positions[:, 2] < b + integrator.cutoff)]
                chunk_atoms = pad_atoms(chunk_atoms, integrator.cutoff)
                chunk_positions = chunk_atoms.positions

                if len(chunk_atoms) == 0:
                    continue

                positions = np.zeros((0, 3), dtype=xp.float32)
                A = np.zeros((0,), dtype=xp.float32)
                B = np.zeros((0,), dtype=xp.float32)
                run_length_enconding = np.zeros((end - start + 1,), dtype=xp.int32)

                for i, j in enumerate(range(start, end)):
                    a = slice_edges[j]
                    b = slice_edges[j + 1]
                    slice_positions = chunk_positions[(chunk_positions[:, 2] > a - integrator.cutoff) *
                                                      (chunk_positions[:, 2] < b + integrator.cutoff)]

                    positions = np.vstack((positions, slice_positions))
                    A = np.concatenate((A, [a] * len(slice_positions)))
                    B = np.concatenate((B, [b] * len(slice_positions)))

                    run_length_enconding[i + 1] = run_length_enconding[i] + len(slice_positions)

                vr, dvdr = integrator.integrate(positions[:, 2], A, B, xp=xp)

                vr = xp.asarray(vr, dtype=xp.float32)
                dvdr = xp.asarray(dvdr, dtype=xp.float32)
                r = xp.asarray(integrator.r, dtype=xp.float32)
                sampling = xp.asarray(self.sampling, dtype=xp.float32)

                interpolate_radial_functions(array,
                                             run_length_enconding,
                                             disc_indices,
                                             positions,
                                             vr,
                                             r,
                                             dvdr,
                                             sampling)

            slice_thicknesses = [self.get_slice_thickness(i) for i in range(start, end)]

            yield start, end, PotentialArray(array[:end - start] / kappa,
                                             slice_thicknesses,
                                             extent=self.extent)

    def generate_frozen_phonon_potentials(self, pbar: Union[ProgressBar, bool] = True):
        """
        Function to generate scattering potentials for a set of frozen phonon configurations.

        Parameters
        ----------
        pbar: bool, optional
            Display a progress bar. Default is True.

        Returns
        -------
        generator
            Generator of potentials.
        """

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=(not pbar) or (not self._precalculate))

        for atoms in self.frozen_phonons:
            self.atoms.positions[:] = atoms.positions
            # self.atoms.wrap()
            pbar.reset()

            if self._precalculate:
                yield self.build(pbar=pbar)
            else:
                yield self

        pbar.refresh()
        pbar.close()

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
                 array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence[float]],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):

        if (len(array.shape) != 2) & (len(array.shape) != 3):
            raise RuntimeError()

        slice_thicknesses = np.array(slice_thicknesses)

        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, array.shape[0])
        elif (slice_thicknesses.shape != (array.shape[0],)) & (len(array.shape) == 3):
            raise ValueError()

        self._array = array
        self._slice_thicknesses = slice_thicknesses
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling, lock_gpts=True)

    def __getitem__(self, items):
        if isinstance(items, int):
            return PotentialArray(self.array[items][None], self._slice_thicknesses[items][None], extent=self.extent)

        elif isinstance(items, slice):
            return PotentialArray(self.array[items], self._slice_thicknesses[items], extent=self.extent)
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def as_transmission_function(self, energy: float, in_place: bool = True, max_batch: int = 1,
                                 antialias_filter: AntialiasFilter = None):
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
        complex_exponential = get_device_function(xp, 'complex_exponential')

        array = self._array
        if not in_place:
            array = array.copy()

        array = complex_exponential(energy2sigma(energy) * array)

        t = TransmissionFunction(array,
                                 slice_thicknesses=self._slice_thicknesses.copy(),
                                 extent=self.extent,
                                 energy=energy)

        if antialias_filter is None:
            antialias_filter = AntialiasFilter()

        for start, end, potential_slices in t.generate_slices(max_batch=max_batch):
            antialias_filter.bandlimit(potential_slices)

        return t

    @property
    def array(self):
        """The potential array."""
        return self._array

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

    def generate_slices(self, first_slice: int = 0, last_slice: int = None, max_batch: int = 1):
        if last_slice is None:
            last_slice = len(self)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            slice_thicknesses = np.array([self.get_slice_thickness(i) for i in range(start, end)])
            yield start, end, self.__class__(self.array[start:end],
                                             slice_thicknesses=slice_thicknesses,
                                             extent=self.extent)

    def tile(self, multiples):
        """
        Tile the potential.

        Parameters
        ----------
        multiples: two int
            The number of repetitions of the potential along each axis.

        Returns
        -------
        PotentialArray object
            The tiled potential.
        """

        assert len(multiples) == 2
        new_array = np.tile(self.array, (1,) + multiples)
        new_extent = (self.extent[0] * multiples[0], self.extent[1] * multiples[1])
        return self.__class__(array=new_array, slice_thicknesses=self._slice_thicknesses, extent=new_extent)

    def write(self, path):
        """
        Write potential to file.

        Parameters
        ----------
        path: str
            Path to which the data is saved.
        """
        with h5py.File(path, 'w') as f:
            f.create_dataset('array', data=self.array)
            f.create_dataset('slice_thicknesses', data=self._slice_thicknesses)
            f.create_dataset('extent', data=self.extent)

    @classmethod
    def read(cls, path):
        """
        Read potentia from hdf5 file.

        Parameters
        ----------
        path: str
            The file to read.

        Returns
        -------
        PotentialArray object
        """
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        return cls(array=datasets['array'], slice_thicknesses=datasets['slice_thicknesses'], extent=datasets['extent'])

    def transmit(self, waves):
        """
        Transmit a wave function.

        Parameters
        ----------
        waves: Waves object
            Wave function to transmit.

        Returns
        -------
        TransmissionFunction
        """
        return self.as_transmission_function(waves.energy).transmit(waves)

    def project(self):
        """
        Create a 2d measurement of the projected potential.

        Returns
        -------
        Measurement
        """
        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, names=['x', 'y'])
        array = asnumpy(self.array.sum(0))
        array -= array.min()
        return Measurement(array, calibrations)

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

    def as_transmission_function(self, energy, in_place=True, max_batch=1, antialias_filter=None):
        if energy != self.energy:
            raise RuntimeError()

        return self

    def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
        if energy != self.energy:
            raise RuntimeError()
        return self.generate_slices(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)

    def transmit(self, waves):
        self.accelerator.check_match(waves)
        xp = get_array_module(waves._array)
        waves._array *= copy_to_device(self.array, xp)
        return waves
