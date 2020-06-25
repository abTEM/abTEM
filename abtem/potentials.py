from abc import ABCMeta, abstractmethod
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
from abtem.bases import Grid, HasGridMixin, Cache, cached_method, HasAcceleratorMixin, Accelerator, watched_method, \
    Event, cache_clear_callback
from abtem.cpu_kernels import interpolate_radial_functions, complex_exponential
from abtem.device import get_array_module
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
        raise NotImplementedError()

    @abstractmethod
    def get_slice(self, i):
        pass

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError('potential indices must be integers, not {}'.format(type(item)))
        if item >= self.num_slices:
            raise StopIteration
        return self.get_slice(item)

    def calculate(self, start=None, end=None, xp=np, show_progress=False) -> 'ArrayPotential':
        self.grid.check_is_defined()

        if start is None:
            start = 0

        if end is None:
            end = self.num_slices

        array = xp.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]), dtype=self.get_slice(0)[0].dtype)
        slice_thicknesses = xp.zeros(self.num_slices)

        with tqdm(total=end - start) if show_progress else ExitStack() as pbar:
            for i in range(start, end):
                array[i], slice_thicknesses[i] = self.get_slice(i)

                if show_progress:
                    pbar.update(1)

        return ArrayPotential(array, slice_thicknesses, self.extent)

    def show(self, **kwargs):
        return show_image(self.calculate().array.sum(0), calibrations_from_grid(self.grid, names=['x', 'y']), **kwargs)


class PotentialIntegrator:

    def __init__(self, function, r, cache_size=512, cache_key_decimals=2):
        self._function = function
        self._r = r
        self.cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._r[-1]

    def integrate(self, a, b):
        a = round(a, self._cache_key_decimals)
        b = round(b, self._cache_key_decimals)

        a = max(min(a, b), -self.cutoff)
        b = min(max(a, b), self.cutoff)
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
        value, error_estimate, step_size, order = integrate(f, -1, 1, 1e-9)
        xk, wk = tanh_sinh_nodes_and_weights(step_size, order)
        f = lambda z: self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))
        return np.sum(f(xk[None]) * wk[None], axis=1) * zm


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
                 cutoff_tolerance: float = 1e-3):

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        if not is_orthogonal(atoms):
            raise RuntimeError('atoms are non-orthogonal')

        self._atoms = atoms.copy()
        self._atoms.wrap()
        self._thickness = atoms.cell[2, 2]

        self._grid = Grid(gpts=gpts, sampling=sampling, extent=np.diag(atoms.cell)[:2])
        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness
        self._atomic_numbers = np.unique(atoms.numbers)

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

        self.positions_cache = Cache(len(self._atomic_numbers))
        self.positions_changed_event = Event()
        self.positions_changed_event.register(cache_clear_callback(self.positions_cache))

        self.integrators_cache = Cache(len(self._atomic_numbers))

    @watched_method('positions_changed_event')
    def displace_atoms(self, new_positons):
        self._atoms.positions[:] = new_positons

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
        return brentq(lambda r: self.function(r, self.parameters[number]) - self.cutoff_tolerance, 1e-7, 1000)

    @cached_method('integrators_cache')
    def _get_integrator(self, number):
        cutoff = self._get_cutoff(number)
        cutoff_value = self.function(cutoff, self.parameters[number])
        cutoff_derivative = self._derivative(cutoff, self.parameters[number])

        def soft_potential(r):
            result = np.array(self._function(r, self.parameters[number]) - cutoff_value -
                              (r - cutoff) * cutoff_derivative)
            result[r > cutoff] = 0
            return result

        r = np.geomspace(np.min(self.sampling), cutoff, int(np.ceil(cutoff / np.min(self.sampling) * 4)))
        return PotentialIntegrator(soft_potential, r)

    @cached_method('positions_cache')
    def _get_padded_positions(self, number):
        cutoff = self._get_cutoff(number)
        return fill_rectangle(self.atoms[self.atoms.numbers == number],
                              self.extent,
                              [0., 0.],
                              margin=cutoff).positions

    def get_slice(self, i, xp=np):
        self.check_slice_idx(i)
        self.grid.check_is_defined()
        a = self.get_slice_entrance(i)
        b = self.get_slice_exit(i)

        v = xp.zeros(self.gpts, dtype=xp.float32)
        for number in self._atomic_numbers:
            integrator = self._get_integrator(number)
            positions = self._get_padded_positions(number)

            positions = positions[(positions[:, 2] > a - integrator.cutoff) *
                                  (positions[:, 2] < b + integrator.cutoff)]

            if len(positions) == 0:
                continue

            vr = np.zeros((len(positions), len(integrator.r)))
            for i, (a, b) in enumerate(np.nditer((a - positions[:, 2], b - positions[:, 2]))):
                vr[i] = integrator.integrate(a.item(), b.item())

            pixel_positions = positions[:, :2] / self.sampling

            position_indices = np.ceil(pixel_positions).astype(np.int)[:, 0] * v.shape[1] + \
                               np.ceil(pixel_positions).astype(np.int)[:, 1]

            disc_rows, disc_cols = disc_meshgrid(np.int(np.ceil(integrator.r[-1] / np.min(self.sampling))))
            rows, cols = np.indices(v.shape)

            interpolate_radial_functions(v.ravel(),
                                         rows.ravel(),
                                         cols.ravel(),
                                         position_indices,
                                         disc_rows * v.shape[1] + disc_cols,
                                         pixel_positions,
                                         vr,
                                         integrator.r / np.min(self.sampling))
            v = v.reshape(self.gpts)

        return v / kappa, b - a


def disc_meshgrid(r):
    cols = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.int32)
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

    def get_slice(self, i):
        return self._array[i], self.get_slice_thickness(i)

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
