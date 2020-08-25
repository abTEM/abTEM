"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Callable, Generator

import h5py
import numpy as np
from copy import copy

from ase import Atoms
from ase import units

from scipy.optimize import brentq

from abtem.base_classes import Grid, HasGridMixin, Cache, cached_method, HasAcceleratorMixin, Accelerator, Event, \
    watched_property
from abtem.device import get_device_function, get_array_module, get_array_module_from_device, copy_to_device, \
    HasDeviceMixin
from abtem.measure import calibrations_from_grid
from abtem.parametrizations import kirkland, dvdr_kirkland, load_kirkland_parameters, kirkland_projected_fourier
from abtem.parametrizations import lobato, dvdr_lobato, load_lobato_parameters
from abtem.plot import show_image
from abtem.structures import is_cell_orthogonal
from abtem.tanh_sinh import integrate, tanh_sinh_nodes_and_weights
from abtem.temperature import AbstractFrozenPhonons, DummyFrozenPhonons
from abtem.utils import energy2sigma, ProgressBar

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

    @abstractmethod
    def generate_slices(self, start=0, end=None):
        """
        Generate the potential one slice at a time.

        Parameters
        ----------
        start: int
            First potential slice.
        end: int, optional
            Last potential slice.

        Returns
        -------
        generator of ProjectedPotential objects
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

    def get_projected_potential(self, items) -> 'ProjectedPotentialArray':
        """
        Get projected potential for the slice or slices.

        Parameters
        ----------
        items: int or slice
            The potential slice(s) (either a single index or a range, ie. a Python slice [start:end]) to include in the
            projection.

        Returns
        -------
        ProjectedPotential object
            The projected potential.
        """

        if isinstance(items, int):
            if items >= self.num_slices:
                raise StopIteration
            return next(self.generate_slices(items, items + 1))
        elif isinstance(items, slice):
            if items.start is None:
                start = 0
            else:
                start = items.start

            if items.stop is None:
                stop = len(self)
            else:
                stop = items.stop

            if items.step is None:
                step = 1
            else:
                step = items.step

            projected = self[start]
            for i in range(start + 1, stop, step):
                projected._array += self[i]._array
                projected._thickness += self.get_slice_thickness(i)
            return projected
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def __getitem__(self, items):
        return self.get_projected_potential(items)

    def show(self, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for abtem.plot.show_image.
        """

        self[:].show(**kwargs)

    def copy(self):
        """Make a copy."""
        return copy(self)


class AbstractPotentialBuilder(AbstractPotential):
    """Potential builder abstract class."""

    def __init__(self, storage='cpu'):
        self._storage = storage

    def build(self, pbar: Union[bool, ProgressBar] = False) -> 'PotentialArray':
        """
        Precalcaulate the potential as a potential array.

        Parameters
        ----------
        pbar: bool
            If true, show progress bar.

        Returns
        -------
        PotentialArray object
        """

        self.grid.check_is_defined()

        storage_xp = get_array_module_from_device(self._storage)

        array = storage_xp.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]), dtype=np.float32)
        slice_thicknesses = np.zeros(self.num_slices)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)

        pbar.reset()
        for i, potential_slice in enumerate(self.generate_slices()):
            array[i] = copy_to_device(potential_slice.array, self._storage)
            slice_thicknesses[i] = potential_slice.thickness
            pbar.update(1)

        pbar.refresh()

        return PotentialArray(array, slice_thicknesses, self.extent)


class AbstractTDSPotentialBuilder(AbstractPotentialBuilder):
    """Thermal diffuse scattering builder abstract class."""

    def __init__(self, storage='cpu'):
        super().__init__(storage)

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

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._cutoff

    def integrate(self, a: float, b: float):
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

        a = round(a, self._cache_key_decimals)
        b = round(b, self._cache_key_decimals)

        split = a * b < 0
        a = max(min(a, b), -self.cutoff)
        b = min(max(a, b), self.cutoff)
        if split:  # split the integral
            values1, derivatives1 = self._do_integrate(0, abs(a))
            values2, derivatives2 = self._do_integrate(0, abs(b))
            result = (values1 + values2, derivatives1 + derivatives2)
        else:
            result = self._do_integrate(a, b)
        return result

    @cached_method('_cache')
    def _do_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.

        def f(z):
            return self._function(np.sqrt(self.r[0] ** 2 + (z * zm + zp) ** 2))

        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tolerance)
        xk, wk = tanh_sinh_nodes_and_weights(step_size, order)

        def f(z):
            return self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))

        values = np.sum(f(xk[None]) * wk[None], axis=1) * zm
        derivatives = np.diff(values) / np.diff(self.r)
        return values, derivatives


class ProjectedPotentialArray(HasGridMixin):
    """
    Projected potential array object.

    Parameters
    ----------
    array: 2D array
        The array describing the potential array [eV / e * Å]
    thickness: float
        The thickness of the projected potential.
    extent: one or two float
        Lateral extent of the potential [Å].
    sampling: one or two float
        Lateral sampling of the potential [1 / Å].
    """

    def __init__(self, array: np.ndarray, thickness: float, extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):
        if len(array.shape) != 2:
            raise RuntimeError('The projected potential array must be 2d.')

        self._array = array
        self._thickness = thickness
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)

    @property
    def thickness(self) -> float:
        """The thickness of the projected potential [Å]."""
        return self._thickness

    @property
    def array(self):
        """The array describing the potential array [eV / e * Å]."""
        return self._array

    def show(self, **kwargs):
        """
        Show the projected potential.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for abtem.plot.show_image.
        """

        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, names=['x', 'y'])
        return show_image(self.array, calibrations, **kwargs)


def pad_atoms(atoms: Atoms, margin: float):
    """
    Repeat the atoms in x and y, retaining only the repeated atoms within the margin distance from the cell boundary.

    Parameters
    ----------
    atoms: ASE Atoms object
        The atoms that should be padded.
    margin: float
        The padding margin.

    Returns
    -------
    ASE Atoms object
        Padded atoms.
    """

    if not is_cell_orthogonal(atoms):
        raise RuntimeError('The cell of the atoms must be orthogonal.')

    left = atoms[atoms.positions[:, 0] < margin]
    left.positions[:, 0] += atoms.cell[0, 0]
    right = atoms[atoms.positions[:, 0] > atoms.cell[0, 0] - margin]
    right.positions[:, 0] -= atoms.cell[0, 0]

    atoms += left + right

    top = atoms[atoms.positions[:, 1] < margin]
    top.positions[:, 1] += atoms.cell[1, 1]
    bottom = atoms[atoms.positions[:, 1] > atoms.cell[1, 1] - margin]
    bottom.positions[:, 1] -= atoms.cell[1, 1]
    atoms += top + bottom
    return atoms


def _disc_meshgrid(r):
    """Internal function to return all indices inside a disk with a given radius."""
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= r ** 2
    return rows[inside], cols[inside]


def superpose_deltas(positions, array):
    shape = array.shape
    xp = get_array_module(array)
    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[(rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[rows, (cols + 1) % shape[1]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[(rows + 1) % shape[0], (cols + 1) % shape[1]] += (rows - positions[:, 0]) * (cols - positions[:, 1])


class Potential(AbstractTDSPotentialBuilder, HasDeviceMixin):
    """
    Potential object.

    The potential object is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms: Atoms or FrozenPhonons object
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the IAM of the electrostatic potential(s).
    gpts: one or two int, optional
        Number of grid points describing each slice of the potential.
    sampling: one or two float, optional
        Lateral sampling of the potential [1 / Å].
    slice_thickness: float, optional
        Thickness of the potential slices in Å for calculating the number of slices used by the multislice algorithm.
        Default is 0.5 Å.
    parametrization: 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available by Lobato et. al. and Kirkland. The abTEM default is 'lobato'.
        See the citation guide for references.
    projection: 'finite' or 'infinite'
        If 'finite' the 3d potential is numerically integrated between the slice boundaries. If 'infinite' the infinite
        potential projection of each atom will be assigned to a single slice.
    cutoff_tolerance: float, optional
        The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]. The cutoff is only
        relevant for potentials using the 'finite' projection scheme.
    device: str, optional
        The device used for calculating the potential. The default is 'cpu'.
    storage: str, optional
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

        self._device = device

        if storage is None:
            storage = device

        super().__init__(storage=storage)

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
            inner_cutoff = np.min(self.sampling) / 5.
            num_points = int(np.ceil(cutoff / np.min(self.sampling) * 5.))
            r = np.geomspace(inner_cutoff, cutoff, num_points)
            self._integrators[number] = PotentialIntegrator(soft_function, r, cutoff)
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

    def generate_slices(self, start=0, end=None) -> Generator:
        self.grid.check_is_defined()

        if end is None:
            end = len(self)

        if self.projection == 'finite':
            return self._generate_slices_finite(start=start, end=end)
        else:
            return self._generate_slices_infinite(start=start, end=end)

    def _generate_slices_infinite(self, start=0, end=None) -> Generator:
        xp = get_array_module_from_device(self._device)
        atoms = self.atoms.copy()
        atoms.wrap()
        indices_by_number = {number: np.where(atoms.numbers == number)[0] for number in np.unique(atoms.numbers)}

        kx = xp.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = xp.fft.fftfreq(self.gpts[1], self.sampling[1])
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')
        k = xp.sqrt(kx ** 2 + ky ** 2)

        sinc = xp.sinc(xp.sqrt((kx * self.sampling[0]) ** 2 + (kx * self.sampling[1]) ** 2))

        scattering_factors = {}
        for atomic_number in indices_by_number.keys():
            f = kirkland_projected_fourier(k, self.parameters[atomic_number])
            scattering_factors[atomic_number] = (f / (sinc * self.sampling[0] * self.sampling[1] * kappa)).astype(
                xp.complex64)

        array = xp.zeros((len(indices_by_number),) + self.gpts, dtype=xp.complex64)
        a = np.sum([self.get_slice_thickness(i) for i in range(0, start)])

        fft2_convolve = get_device_function(xp, 'fft2_convolve')

        for i in range(start, end):
            array[:] = 0.
            b = a + self.get_slice_thickness(i)

            for j, (number, indices) in indices_by_number.items():
                slice_atoms = atoms[indices]
                slice_atoms = slice_atoms[(slice_atoms.positions[:, 2] > a) *
                                          (slice_atoms.positions[:, 2] < b)]

                positions = xp.asarray(slice_atoms.positions[:, :2] / self.sampling)

                superpose_deltas(positions, array[j])
                fft2_convolve(array[j], scattering_factors[number])

            # array += cupyx.scipy.fft.ifft2(cupyx.scipy.fft.ifft2(new_array, overwrite_x=True) *
            #                               scattering_factors[number], overwrite_x=True).real

            a = b
            yield ProjectedPotentialArray(array.real.sum(0), self.get_slice_thickness(i), extent=self.extent)

    def _generate_slices_finite(self, start=0, end=None) -> Generator:
        xp = get_array_module_from_device(self._device)

        interpolate_radial_functions = get_device_function(xp, 'interpolate_radial_functions')

        atoms = self.atoms.copy()
        atoms.wrap()
        indices_by_number = {number: np.where(atoms.numbers == number)[0] for number in np.unique(atoms.numbers)}

        array = xp.zeros(self.gpts, dtype=xp.float32)
        a = np.sum([self.get_slice_thickness(i) for i in range(0, start)])
        for i in range(start, end):
            array[:] = 0.
            b = a + self.get_slice_thickness(i)

            for number, indices in indices_by_number.items():
                slice_atoms = atoms[indices]

                integrator = self.get_integrator(number)

                disc_indices = xp.asarray(self._get_radial_interpolation_points(number))

                slice_atoms = slice_atoms[(slice_atoms.positions[:, 2] > a - integrator.cutoff) *
                                          (slice_atoms.positions[:, 2] < b + integrator.cutoff)]

                slice_atoms = pad_atoms(slice_atoms, integrator.cutoff)

                if len(slice_atoms) == 0:
                    continue

                vr = np.zeros((len(slice_atoms), integrator.r.shape[0]), np.float32)
                dvdr = np.zeros((len(slice_atoms), integrator.r.shape[0]), np.float32)
                for j, atom in enumerate(slice_atoms):
                    am, bm = a - atom.z, b - atom.z
                    vr[j], dvdr[j, :-1] = integrator.integrate(am, bm)
                vr = xp.asarray(vr, dtype=xp.float32)
                dvdr = xp.asarray(dvdr, dtype=xp.float32)
                r = xp.asarray(integrator.r, dtype=xp.float32)

                slice_positions = xp.asarray(slice_atoms.positions[:, :2], dtype=xp.float32)
                sampling = xp.asarray(self.sampling, dtype=xp.float32)

                interpolate_radial_functions(array,
                                             disc_indices,
                                             slice_positions,
                                             vr,
                                             r,
                                             dvdr,
                                             sampling)
            a = b

            yield ProjectedPotentialArray(array / kappa, self.get_slice_thickness(i), extent=self.extent)

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
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)

        for atoms in self.frozen_phonons:
            self.atoms.positions[:] = atoms.positions
            self.atoms.wrap()
            pbar.reset()
            yield self.build(pbar=pbar)

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

    def __init__(self, array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence[float]],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):

        if len(array.shape) != 3:
            raise RuntimeError()

        slice_thicknesses = np.array(slice_thicknesses)

        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, array.shape[0])
        elif slice_thicknesses.shape != (array.shape[0],):
            raise ValueError()

        self._array = array
        self._slice_thicknesses = slice_thicknesses
        self._grid = Grid(extent=extent, gpts=self.array.shape[1:], sampling=sampling, lock_gpts=True)

    def as_transmission_functions(self, energy):
        """
        Calculate the transmission functions for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        TransmissionFunctions object
        """

        xp = get_array_module(self.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        array = complex_exponential(energy2sigma(energy) * self._array)
        return TransmissionFunctions(array, slice_thicknesses=self._slice_thicknesses, extent=self.extent,
                                     energy=energy)

    @property
    def array(self):
        """The potential array."""
        return self._array

    @property
    def num_slices(self):
        return self._array.shape[0]

    def get_slice_thickness(self, i):
        return self._slice_thicknesses[i]

    def generate_slices(self, start=0, end=None):
        if end is None:
            end = len(self)

        for i in range(start, end):
            yield ProjectedPotentialArray(self.array[i], thickness=self.get_slice_thickness(i), extent=self.extent)

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
        with h5py.File(path, 'w') as f:
            f.create_dataset('array', data=self.array)
            f.create_dataset('slice_thicknesses', data=self._slice_thicknesses)
            f.create_dataset('extent', data=self.extent)

    @classmethod
    def read(cls, path):
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        return cls(array=datasets['array'], slice_thicknesses=datasets['slice_thicknesses'], extent=datasets['extent'])

    def __copy___(self):
        return self.__class__(array=self.array.copy(),
                              slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent)


class TransmissionFunctions(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions."""

    def __init__(self,
                 array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence[float]],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thicknesses, extent, sampling)
