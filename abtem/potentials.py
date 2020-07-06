from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence

import cupy as cp
import h5py
import numpy as np
from ase import Atoms
from ase import units
from scipy.optimize import brentq

from abtem.structures import fill_rectangle, is_orthogonal
from abtem.bases import Grid, HasGridMixin, Cache, cached_method, HasAcceleratorMixin, Accelerator, watched_method, \
    Event
from abtem.cpu_kernels import complex_exponential
from abtem.measure import calibrations_from_grid
from abtem.parametrizations import kirkland, dvdr_kirkland, load_kirkland_parameters
from abtem.parametrizations import lobato, dvdr_lobato, load_lobato_parameters
from abtem.plot import show_image
from abtem.tanh_sinh import integrate, tanh_sinh_nodes_and_weights
from abtem.temperature import AbstractTDS
from abtem.utils import energy2sigma, ProgressBar
from abtem.device import get_device_function, get_array_module

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


class AbstractPotential(HasGridMixin, metaclass=ABCMeta):

    def __len__(self):
        return self.num_slices

    def get_slice_entrance(self, i):
        self.check_slice_idx(i)
        return float(sum([self.get_slice_thickness(j) for j in range(i)]))

    def get_slice_exit(self, i):
        self.check_slice_idx(i)
        return float(sum([self.get_slice_thickness(j) for j in range(i + 1)]))

    @property
    @abstractmethod
    def num_slices(self):
        pass

    def check_slice_idx(self, i):
        if i >= self.num_slices:
            raise RuntimeError('slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    @abstractmethod
    def get_slice_thickness(self, i):
        pass

    @property
    @abstractmethod
    def thickness(self):
        pass

    @property
    @abstractmethod
    def tds(self):
        pass

    @property
    def has_tds(self):
        return self.tds is not None

    @property
    def num_tds_configs(self):
        if self.tds is None:
            return 1
        else:
            return len(self.tds)

    @property
    @abstractmethod
    def generate_tds_potentials(self):
        pass

    def get_slice(self, i):
        return PotentialSlice(self, i)

    @abstractmethod
    def calculate_slice(self, i, xp=np):
        pass

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError('potential indices must be integers, not {}'.format(type(item)))
        if item >= self.num_slices:
            raise StopIteration
        return self.get_slice(item)

    def calculate(self, start=None, end=None, xp=np, pbar: Union[bool, ProgressBar] = False) -> 'ArrayPotential':
        self.grid.check_is_defined()

        if start is None:
            start = 0

        if end is None:
            end = self.num_slices

        array = xp.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]), dtype=self.get_slice(0).calculate(xp).dtype)
        slice_thicknesses = np.zeros(self.num_slices)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)

        pbar.reset()
        for i in range(start, end):
            potential_slice = self.get_slice(i)

            array[i] = potential_slice.calculate(xp)
            slice_thicknesses[i] = potential_slice.thickness
            pbar.update(1)

        pbar.refresh()
        return ArrayPotential(array, slice_thicknesses, self.extent)

    def show(self, start=None, end=None, **kwargs):
        if (start is None) & (end is None):
            start = 0
            end = len(self)
        if start is None:
            start = 0
        if end is None:
            end = start + 1

        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, names=['x', 'y'])
        return show_image(self.calculate(start, end).array.sum(0), calibrations, **kwargs)


class PotentialIntegrator:

    def __init__(self, function, r, cache_size=2048, cache_key_decimals=2, tolerance=1e-6):
        self._function = function
        self._r = r
        self.cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals
        self._tolerance = tolerance

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._r[-1]

    def integrate(self, a, b):
        a = round(a, self._cache_key_decimals)
        b = round(b, self._cache_key_decimals)

        split = np.sign(a) * np.sign(b)
        a = max(min(a, b), -self.cutoff)
        b = min(max(a, b), self.cutoff)
        if split < 0:  # split integral
            values1, derivatives1 = self.cached_integrate(0, abs(a))
            values2, derivatives2 = self.cached_integrate(0, abs(b))
            result = (values1 + values2, derivatives1 + derivatives2)
        else:

            result = self.cached_integrate(a, b)
        return result

    @cached_method('cache')
    def cached_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.
        f = lambda z: self._function(np.sqrt(self.r[0] ** 2 + (z * zm + zp) ** 2))
        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tolerance)
        xk, wk = tanh_sinh_nodes_and_weights(step_size, order)
        f = lambda z: self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))
        values = np.sum(f(xk[None]) * wk[None], axis=1) * zm
        derivatives = np.diff(values) / np.diff(self.r)
        return values, derivatives


class PotentialSlice(HasGridMixin):

    def __init__(self, potential: AbstractPotential, index: int):
        self._potential = potential
        self._index = index

    @property
    def _grid(self):
        return self._potential.grid

    @property
    def index(self) -> int:
        return self._index

    @property
    def thickness(self) -> float:
        return self._potential.get_slice_thickness(self.index)

    def calculate(self, xp=np):
        return self._potential.calculate_slice(self.index, xp)

    def show(self, **kwargs):
        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, names=['x', 'y'])
        return show_image(self.calculate(), calibrations, **kwargs)


class Potential(AbstractPotential):
    """
    Potential object

    The potential object is used to calculate electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated in the Independent Atom Model (IAM) with a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms : Atoms object
        Atoms object defining the atomic configuration used in the IAM of the electrostatic potential.
    origin : two floats, float, optional
        xy-origin of the electrostatic potential relative to the xy-origin of the Atoms object. Units of Angstrom.
    extent : two floats, float, optional
        Lateral extent of potential, if the unit cell of the atoms is too small it will be repeated. Units of Angstrom.
    gpts : two ints, int, optional
        Number of grid points describing each slice of the potential.
    sampling : two floats, float, optional
        Lateral sampling of the potential. Units of 1 / Angstrom.
    slice_thickness : float, optional
        Thickness of the potential slices in Angstrom for calculating the number of slices used by the multislice
        algorithm. Default is 0.5 Angstrom.
    num_slices : int, optional
        Number of slices used by the multislice algorithm. If `num_slices` is set, then `slice_thickness` is disabled.
    parametrization : 'lobato' or 'kirkland'
        The potential parametrization describes the radial dependence of the potential for each element. Two of the most
        accurate parametrizations are available by Lobato et. al. and EJ Kirkland.
        See the citation guide for references.
    cutoff_tolerance : float
        The error tolerance used for deciding the radial cutoff distance of the potential in units of eV / e.
        Default is 1e-3.
    """

    def __init__(self,
                 atoms: Atoms = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 slice_thickness: float = .5,
                 parametrization: str = 'lobato',
                 cutoff_tolerance: float = 1e-3,
                 storage='device'):

        if isinstance(atoms, AbstractTDS):
            self._tds = atoms
            self._atoms = next(atoms.generate_atoms())
        elif isinstance(atoms, Atoms):
            self._tds = None
            self._atoms = atoms.copy()
        else:
            raise RuntimeError()

        if np.abs(self._atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        if not is_orthogonal(self._atoms):
            raise RuntimeError('atoms are non-orthogonal')

        self._thickness = self._atoms.cell[2, 2]

        self._grid = Grid(gpts=gpts, sampling=sampling, extent=np.diag(self._atoms.cell)[:2])
        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness
        self._storage = storage

        self._atomic_numbers = np.unique(self._atoms.numbers)

        if parametrization == 'lobato':
            self._parameters = load_lobato_parameters()
            self._function = lobato  # lambda r: lobato(r, parameters[atomic_number])
            self._derivative = dvdr_lobato  # lambda r: dvdr_lobato(r, parameters[atomic_number])

        elif parametrization == 'kirkland':
            self._parameters = load_kirkland_parameters()
            self._function = kirkland  # lambda r: kirkland(r, parameters[atomic_number])
            self._derivative = dvdr_kirkland  # lambda r: dvdr_kirkland(r, parameters[atomic_number])
        else:
            raise RuntimeError('parametrization {} not recognized'.format(parametrization))

        self._cutoffs = {}
        for number in self._atomic_numbers:
            self._cutoffs[number] = brentq(lambda r: self.function(r, self.parameters[number])
                                                     - self.cutoff_tolerance, 1e-7, 1000)

        self._padded_positions = {}
        self._integrators = {}

        def positions_changed_callback(*args, **kwargs):
            self._padded_positions = {}

        self.positions_changed = Event()
        self.positions_changed.register(positions_changed_callback)

        def grid_changed_callback(*args, **kwargs):
            self._integrators = {}

        self.grid.changed.register(grid_changed_callback)

    @watched_method('positions_changed')
    def displace_atoms(self, new_positons):
        self._atoms.positions[:] = new_positons
        self._atoms.wrap()

    @property
    def parameters(self):
        return self._parameters

    @property
    def function(self):
        return self._function

    @property
    def atoms(self):
        return self._atoms

    @property
    def tds(self):
        return self._tds

    @property
    def precalculate(self):
        return self._precalculate

    @property
    def cutoff_tolerance(self):
        return self._cutoff_tolerance

    @property
    def thickness(self):
        return self._thickness

    @property
    def num_slices(self):
        return int(np.ceil(self.thickness / self._slice_thickness))

    def get_slice_thickness(self, i):
        return self.thickness / self.num_slices

    def _get_cutoff(self, number):
        return self._cutoffs[number]

    def _get_integrator(self, number):
        try:
            return self._integrators[number]
        except KeyError:
            cutoff = self._get_cutoff(number)
            cutoff_value = self.function(cutoff, self.parameters[number])
            cutoff_derivative = self._derivative(cutoff, self.parameters[number])

            def soft_potential(r):
                result = np.array(self._function(r, self.parameters[number]) - cutoff_value -
                                  (r - cutoff) * cutoff_derivative)
                result[r > cutoff] = 0
                return result

            r = np.geomspace(np.min(self.sampling), cutoff, int(np.ceil(cutoff / np.min(self.sampling) * 10)))

            margin = np.int(np.ceil(cutoff / np.min(self.sampling)))
            rows, cols = disc_meshgrid(margin)
            disc_indices = np.hstack((rows[:, None], cols[:, None]))
            self._integrators[number] = (PotentialIntegrator(soft_potential, r), disc_indices)
            return self._integrators[number]

    def _get_padded_positions(self, number):
        try:
            return self._padded_positions[number]
        except KeyError:
            cutoff = self._get_cutoff(number)
            self._padded_positions[number] = fill_rectangle(self.atoms[self.atoms.numbers == number],
                                                            self.extent,
                                                            [0., 0.],
                                                            margin=cutoff).positions
            return self._padded_positions[number]

    def calculate_slice(self, i, xp=np):
        self.check_slice_idx(i)
        self.grid.check_is_defined()

        interpolate_radial_functions = get_device_function(xp, 'interpolate_radial_functions')

        a = self.get_slice_entrance(i)
        b = self.get_slice_exit(i)

        array = xp.zeros(self.gpts, dtype=xp.float32)
        for number in self._atomic_numbers:
            integrator, disc_indices = self._get_integrator(number)
            disc_indices = xp.asarray(disc_indices)

            positions = self._get_padded_positions(number)

            positions = positions[(positions[:, 2] > a - integrator.cutoff) *
                                  (positions[:, 2] < b + integrator.cutoff)]

            if len(positions) == 0:
                continue

            vr = np.zeros((len(positions), len(integrator.r)), xp.float32)
            dvdr = np.zeros((len(positions), len(integrator.r)), xp.float32)
            for i, (am, bm) in enumerate(np.nditer((a - positions[:, 2], b - positions[:, 2]))):
                vr[i], dvdr[i, :-1] = integrator.integrate(am.item(), bm.item())
            vr = xp.asarray(vr, dtype=xp.float32)
            dvdr = xp.asarray(dvdr, dtype=xp.float32)
            r = xp.asarray(integrator.r, dtype=xp.float32)

            positions = xp.asarray(positions[:, :2], dtype=xp.float32)
            sampling = xp.asarray(self.sampling, dtype=xp.float32)

            interpolate_radial_functions(array,
                                         disc_indices,
                                         positions,
                                         vr,
                                         r,
                                         dvdr,
                                         sampling)

        return array / kappa

    def generate_tds_potentials(self, xp=np, pbar: Union[ProgressBar, bool] = True):
        if not self.has_tds:
            raise RuntimeError()

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)

        for atoms in self._tds.generate_atoms():
            self.displace_atoms(atoms.positions)
            pbar.reset()
            yield self.calculate(xp=xp, pbar=False)
            pbar.refresh()


def disc_meshgrid(r):
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= r ** 2
    return rows[inside], cols[inside]


class ArrayPotential(AbstractPotential, HasGridMixin):

    def __init__(self, array: np.ndarray, slice_thicknesses: Union[float, Sequence], extent: np.ndarray = None,
                 sampling: np.ndarray = None):

        """
        Precalculated potential object.

        Parameters
        ----------
        array : 3d ndarray
            The array representing the potential slices. The first dimension is the slice index and the last two are the
            spatial dimensions.
        slice_thicknesses : float, sequence of floats
            The thicknesses of the potential slices in Angstrom. If float, the thickness is the same for all slices.
            If sequence, the length must equal the length of potential array.
        extent : two floats, float, optional
            Lateral extent of potential, if the unit cell of the atoms is too small it will be repeated. Units of Angstrom.
        sampling : two floats, float, optional
            Lateral sampling of the potential. Units of 1 / Angstrom.
        """

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
        xp = get_array_module(self.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        array = complex_exponential(energy2sigma(energy) * self._array)
        t = TransmissionFunctions(array, slice_thicknesses=self._slice_thicknesses, energy=energy)
        t._grid = copy(self.grid)
        return t

    def generate_tds_potentials(self):
        yield self

    @property
    def tds(self):
        return None

    @property
    def array(self):
        return self._array

    @property
    def thickness(self):
        return self._slice_thicknesses.sum()

    @property
    def num_slices(self):
        return self._array.shape[0]

    def get_slice_thickness(self, i):
        return self._slice_thicknesses[i]

    def calculate_slice(self, i, xp=np):
        return xp.asarray(self._array[i])

    def repeat(self, multiples):
        assert len(multiples) == 2
        new_array = np.tile(self._array, (1,) + multiples)
        new_extent = multiples * self.extent
        return self.__class__(array=new_array, slice_thicknesses=self._slice_thicknesses, extent=new_extent)

    def write(self, path):
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('class', (1,), dtype='S100')
            dset[:] = np.string_('abtem.potentials.ArrayPotential')
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

    def copy(self):
        return self.__class__(array=cp.asarray(self.array), slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent.copy())


class TransmissionFunctions(ArrayPotential, HasAcceleratorMixin):

    def __init__(self, array: np.ndarray, slice_thicknesses: Union[float, Sequence], extent: np.ndarray = None,
                 sampling: np.ndarray = None, energy: float = None):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thicknesses, extent, sampling)
