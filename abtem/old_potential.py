from abc import ABCMeta
from contextlib import ExitStack
from copy import copy
from typing import Union, Sequence

import cupy as cp
import h5py
import numpy as np
from ase import Atoms
from ase import units
from scipy.optimize import brentq
from tqdm.auto import tqdm

from abtem.atoms import fill_rectangle, is_orthogonal
from abtem.bases import Grid, HasGridMixin, Event, Cache, cached_method, cache_clear_callback, \
    HasAcceleratorMixin, Accelerator
from abtem.cpu_kernels import interpolate_radial_functions, complex_exponential
from abtem.measure import calibrations_from_grid
from abtem.parametrizations import kirkland, dvdr_kirkland, load_kirkland_parameters
from abtem.parametrizations import lobato, dvdr_lobato, load_lobato_parameters
from abtem.plot import show_image
from abtem.tanh_sinh import integrate, tanh_sinh_nodes_and_weights
from abtem.utils import energy2sigma

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

DTYPE = np.float32


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
    def num_slices(self):
        raise NotImplementedError()

    def check_slice_idx(self, i):
        if i >= self.num_slices:
            raise RuntimeError('slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    def get_slice_thickness(self, i):
        raise NotImplementedError()

    @property
    def thickness(self):
        raise NotImplementedError()

    def get_slice(self, i):
        self.check_slice_idx(i)
        return PotentialSlice(self, i)

    def _calculate_slice(self, i):
        raise NotImplementedError()

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError('potential indices must be integers, not {}'.format(type(item)))
        if item >= self.num_slices:
            raise StopIteration
        return self.get_slice(item)

    def calculate(self, first_slice=None, last_slice=None, show_progress=False) -> 'ArrayPotential':
        self.grid.check_is_defined()

        if first_slice is None:
            first_slice = 0

        if last_slice is None:
            last_slice = self.num_slices

        array = np.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]), dtype=self[0].array.dtype)
        slice_thicknesses = np.zeros(self.num_slices)

        with tqdm(total=last_slice - first_slice) if show_progress else ExitStack() as pbar:
            for i in range(first_slice, last_slice):
                potential_slice = self.get_slice(i)
                array[i] = potential_slice.array
                slice_thicknesses[i] = potential_slice.thickness
                if show_progress:
                    pbar.update(1)

        return ArrayPotential(array, slice_thicknesses, self.extent)

    def show(self, **kwargs):
        return show_image(self.calculate().array.sum(0), calibrations_from_grid(self.grid, names=['x', 'y']), **kwargs)


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

    @property
    def array(self) -> np.ndarray:
        return self._potential._calculate_slice(self.index)

    def show(self, **kwargs):
        return show_image(self.array, calibrations_from_grid(self.grid, names=['x', 'y']), **kwargs)


class PotentialIntegrator:

    def __init__(self, function, r, tol, cutoff, cache_size=512, cache_key_decimals=2):
        self._function = function
        self._tol = tol
        self._r = r
        self._cutoff = cutoff
        self.cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals

    @property
    def r(self):
        return self._r

    def integrate(self, a, b):
        a = round(a, self._cache_key_decimals)
        b = round(b, self._cache_key_decimals)

        a = max(min(a, b), -self._cutoff)
        b = min(max(a, b), self._cutoff)
        if np.sign(a) * np.sign(b) < 0:  # split integral
            result = self.cached_integrate(0, abs(a))
            result = result + self.cached_integrate(0, abs(b))

        else:
            result = self.cached_integrate(a, b)
        return result

    @cached_method('cache')
    def cached_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.
        f = lambda z: self._function(np.sqrt(self.r[0] ** 2 + (z * zm + zp) ** 2))
        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tol)
        xk, wk = tanh_sinh_nodes_and_weights(step_size, order)
        f = lambda z: self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))
        return np.sum(f(xk[None]) * wk[None], axis=1) * zm


class PotentialInterpolator:

    def __init__(self, atomic_number, parametrization, tolerance, sampling):

        if parametrization == 'lobato':
            parameters = load_lobato_parameters()
            self._function = lambda r: lobato(r, parameters[atomic_number])
            self._derivative = lambda r: dvdr_lobato(r, parameters[atomic_number])

        elif parametrization == 'kirkland':
            self._function = lambda r: kirkland(r, parameters[atomic_number])
            self._derivative = lambda r: dvdr_kirkland(r, parameters[atomic_number])
            parameters = load_kirkland_parameters()

        else:
            raise RuntimeError('parametrization {} not recognized'.format(parametrization))

        cutoff = brentq(lambda r: self._function(r) - tolerance, 1e-7, 1000)

        cutoff_value = self._function(cutoff)
        cutoff_derivative = self._derivative(cutoff)

        def soft_potential(r):
            result = np.array(self._function(r) - cutoff_value - (r - cutoff) * cutoff_derivative)
            result[r > cutoff] = 0
            return result

        self._sampling = sampling

        r = np.linspace(np.min(sampling) / 2, cutoff, int(np.ceil(cutoff / np.min(sampling) * 4)))

        self._projector = PotentialIntegrator(soft_potential, r, 1e-9, cutoff)
        self._margin = np.int(np.ceil(r[-1] / np.min(sampling)))

        m = self._margin
        cols = np.zeros((2 * m + 1, 2 * m + 1), dtype=np.int32)
        cols[:] = np.linspace(0, 2 * m, 2 * m + 1) - m
        rows = cols.copy().T
        r2 = rows ** 2 + cols ** 2
        inside = r2 < m ** 2
        self._disc_indices = (rows[inside], cols[inside])

    @property
    def function(self):
        return self._function

    @property
    def derivative(self):
        return self._derivative

    @property
    def cutoff(self):
        return self.projector.r[-1]

    @property
    def sampling(self):
        return self._sampling

    @property
    def projector(self):
        return self._projector

    @property
    def margin(self):
        return self._margin

    def interpolate(self, shape, positions, limits):
        assert len(positions) == len(limits[0]) == len(limits[1])

        padded_shape = [num_elem + 2 * self.margin for num_elem in shape]

        array = np.zeros(padded_shape, dtype=np.float32)

        v = np.zeros((len(positions), len(self.projector.r)))
        for i, (a, b) in enumerate(np.nditer(limits)):
            v[i] = self.projector.integrate(a.item(), b.item())

        positions = positions / self._sampling + self.margin

        position_indices = np.ceil(positions).astype(np.int)[:, 0] * padded_shape[1] + \
                           np.ceil(positions).astype(np.int)[:, 1]

        disc_indices = self._disc_indices[0] * padded_shape[1] + self._disc_indices[1]

        rows, cols = np.indices(padded_shape)

        interpolate_radial_functions(array.ravel(),
                                     rows.ravel(),
                                     cols.ravel(),
                                     position_indices,
                                     disc_indices,
                                     positions,
                                     v,
                                     self.projector.r / np.min(self._sampling))
        array = array.reshape(padded_shape)[self.margin:-self.margin, self.margin:-self.margin]
        return array


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
                 cutoff_tolerance: float = 1e-2):

        self._grid = Grid(gpts=gpts, sampling=sampling)

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness
        self._interpolators = {}
        self._positions = {}

        if atoms is not None:
            self.set_atoms(atoms)

    def set_atoms(self, atoms):
        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        if not is_orthogonal(atoms):
            raise RuntimeError('atoms are non-orthogonal')

        atoms = atoms.copy()
        atoms.wrap()

        self._thickness = atoms.cell[2, 2]
        self.extent = np.diag(atoms.cell)[:2]
        self.grid.check_is_defined()

        for number in np.unique(atoms.numbers):
            if not number in self._interpolators:
                self._interpolators[number] = PotentialInterpolator(
                    number,
                    self._parametrization,
                    self.cutoff_tolerance,
                    self.sampling
                )

            self._positions[number] = fill_rectangle(atoms[atoms.numbers == number],
                                                     self.extent,
                                                     [0., 0.],
                                                     margin=self.interpolators[number].cutoff).positions

    @property
    def interpolators(self):
        return self._interpolators

    @property
    def positions(self):
        return self._positions

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

    def get_slice(self, i):
        self.check_slice_idx(i)
        return PotentialSlice(self, i)

    def _calculate_slice(self, i):
        self.grid.check_is_defined()

        v = np.zeros(self.gpts, dtype=DTYPE)
        for number, interpolator in self.interpolators.items():
            positions = self.positions[number]
            a = self.get_slice_entrance(i)
            b = self.get_slice_exit(i)

            positions = positions[(positions[:, 2] > a - interpolator.cutoff) *
                                  (positions[:, 2] < b + interpolator.cutoff)]

            if len(positions) > 0:
                v += interpolator.interpolate(v.shape,
                                              positions[:, :2],
                                              (a - positions[:, 2], b - positions[:, 2]))

        return v / kappa


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
        array = complex_exponential(energy2sigma(energy) * self._array)
        t = TransmissionFunctions(array, slice_thicknesses=self._slice_thicknesses, energy=energy)
        t._grid = copy(self.grid)
        return t

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

    def _calculate_slice(self, i):
        return self._array[i]

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

    def copy(self):
        return self.__class__(array=cp.asarray(self.array), slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent.copy())


class TransmissionFunctions(ArrayPotential, HasAcceleratorMixin):

    def __init__(self, array: np.ndarray, slice_thicknesses: Union[float, Sequence], extent: np.ndarray = None,
                 sampling: np.ndarray = None, energy: float = None):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thicknesses, extent, sampling)
