from typing import Union, Sequence

import h5py
import numpy as np
from ase import Atoms
from ase import units
from numba import jit
from scipy.optimize import brentq
from tqdm.auto import tqdm

from abtem.bases import Grid, cached_method, ArrayWithGrid, ArrayWithGrid2D, Cache, cached_method_with_args
from abtem.interpolation import interpolation_kernel_parallel
from abtem.parametrizations import load_lobato_parameters, load_kirkland_parameters, kirkland, dvdr_kirkland, \
    project_tanh_sinh
from abtem.parametrizations import lobato, dvdr_lobato
from abtem.transform import fill_rectangle_with_atoms
import cupy as cp

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

QUADRATURE_PARAMETER_RATIO = 4
DTYPE = np.float32


class PotentialBase:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def num_slices(self):
        raise NotImplementedError()

    def check_slice_idx(self, i):
        if i >= self.num_slices:
            raise RuntimeError('slice index {} too large'.format(i))

    def slice_thickness(self, i):
        raise NotImplementedError()

    def _get_slice_array(self, i):
        raise NotImplementedError()

    def get_slice(self, i):
        raise NotImplementedError()

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise RuntimeError()

        if item >= self.num_slices:
            raise StopIteration
        return self.get_slice(item)


class CalculatedPotentialBase(PotentialBase, Grid):

    def __init__(self, atoms, origin=None, extent=None, gpts=None, sampling=None, num_slices=None, **kwargs):

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        self._atoms = atoms.copy()
        self._atoms.wrap()

        if origin is None:
            self._origin = np.array([0., 0.])
        else:
            self._origin = origin

        if extent is None:
            extent = np.diag(atoms.cell)[:2]

        self._num_slices = num_slices

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, **kwargs)

    @property
    def atoms(self):
        return self._atoms

    @property
    def origin(self):
        return self._origin

    @property
    def thickness(self):
        return self._atoms.cell[2, 2]

    @property
    def num_slices(self):
        return self._num_slices

    def get_slice(self, i):
        return PotentialSlice(self._get_slice_array(i)[None], thickness=self.slice_thickness(i), extent=self.extent)

    def __getitem__(self, i):
        if isinstance(i, int):
            if i >= self.num_slices:
                raise StopIteration
            return self.get_slice(i)
        elif isinstance(i, slice):
            return self.precalculate(show_progress=False, first_slice=slice.start, last_slice=slice.stop)

    def precalculate(self, show_progress=False, first_slice=None, last_slice=None):
        if first_slice is None:
            first_slice = 0

        if last_slice is None:
            last_slice = self.num_slices

        array = np.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]))
        slice_thicknesses = np.zeros(self.num_slices)

        for i in tqdm(range(first_slice, last_slice), disable=not show_progress):
            array[i] = self._get_slice_array(i)
            slice_thicknesses[i] = self.slice_thickness(i)

        return PrecalculatedPotential(array, slice_thicknesses, self.extent)


def tanh_sinh_quadrature(order, parameter):
    xk = np.zeros(2 * order, dtype=DTYPE)
    wk = np.zeros(2 * order, dtype=DTYPE)
    for i in range(0, 2 * order):
        k = i - order
        xk[i] = np.tanh(np.pi / 2 * np.sinh(k * parameter))
        numerator = parameter / 2 * np.pi * np.cosh(k * parameter)
        denominator = np.cosh(np.pi / 2 * np.sinh(k * parameter)) ** 2
        wk[i] = numerator / denominator
    return xk, wk


class PotentialSlice(ArrayWithGrid2D):

    def __init__(self, array, thickness, extent=None):
        self._thickness = thickness
        super().__init__(array=array, extent=extent)

    @property
    def thickness(self):
        return self._thickness


class Potential(CalculatedPotentialBase, Cache):
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
    method : 'finite'
        The method for calculating the potential, currently the only implemented method is 'interpolation'.
    cutoff_tolerance : float
        The error tolerance used for deciding the radial cutoff distance of the potential in units of eV / e.
        Default is 1e-3.
    quadrature_order : int, optional
        Order of the Tanh-Sinh quadrature for numerical integration the potential along the optical axis. Default is 40.
    interpolation_sampling : float, optional
        The average sampling used when calculating the radial dependence of the atomic potentials.
    """

    def __init__(self, atoms: Atoms,
                 origin: Union[float, Sequence[float]] = None,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 slice_thickness: float = .5,
                 num_slices: int = None,
                 parametrization: str = 'lobato',
                 method: str = 'finite',
                 cutoff_tolerance: float = 1e-2,
                 quadrature_order: int = 20,
                 interpolation_sampling: float = .01):

        # TODO : cell warning

        self._cutoff_tolerance = cutoff_tolerance
        self._method = method

        if isinstance(parametrization, str):
            if parametrization == 'lobato':
                self._potential_function = lobato
                self._potential_derivative = dvdr_lobato
                self._parameters = load_lobato_parameters()

            elif parametrization == 'kirkland':
                self._potential_function = kirkland
                self._potential_derivative = dvdr_kirkland
                self._parameters = load_kirkland_parameters()

            else:
                raise RuntimeError('parametrization {} not recognized'.format(parametrization))

        if num_slices is None:
            if slice_thickness is None:
                raise RuntimeError()

            num_slices = int(np.ceil(atoms.cell[2, 2] / slice_thickness))

        self._quadrature_order = quadrature_order
        self._interpolation_sampling = interpolation_sampling

        super().__init__(atoms=atoms.copy(), origin=origin, extent=extent, gpts=gpts, sampling=sampling,
                         num_slices=num_slices)

    def slice_thickness(self, i):
        return self.thickness / self.num_slices

    @property
    def cutoff_tolerance(self):
        return self._cutoff_tolerance

    @property
    def parameters(self):
        return self._parameters

    def slice_entrance(self, i):
        self.check_slice_idx(i)
        return np.sum([self.slice_thickness(j) for j in range(i)])

    def slice_exit(self, i):
        self.check_slice_idx(i)
        return np.sum([self.slice_thickness(j) for j in range(i + 1)])

    def evaluate_potential(self, r, atomic_number):
        return self._potential_function(r, self.parameters[atomic_number])

    def evaluate_potential_derivative(self, r, atomic_number):
        return self._potential_derivative(r, self.parameters[atomic_number])

    @cached_method_with_args(('tolerance', 'parametrization'))
    def get_cutoff(self, atomic_number):
        func = lambda r: self.evaluate_potential(r, atomic_number) - self.cutoff_tolerance
        return brentq(func, 1e-7, 1000)

    @cached_method(())
    def _get_unique_atomic_numbers(self):
        return np.unique(self._atoms.get_atomic_numbers())

    @cached_method(('tolerance',))
    def max_cutoff(self):
        return max([self.get_cutoff(number) for number in self._get_unique_atomic_numbers()])

    @cached_method(('tolerance', 'origin', 'extent'))
    def get_padded_atoms(self):
        cutoff = self.max_cutoff()
        return fill_rectangle_with_atoms(self.atoms, self._origin, self.extent, margin=cutoff, )

    @cached_method(('gpts',))
    def _allocate(self):
        self.check_is_grid_defined()
        return np.zeros(self.gpts, dtype=DTYPE)

    @cached_method_with_args(('tolerance',))
    def get_soft_potential_function(self, atomic_number):
        cutoff = self.get_cutoff(atomic_number)
        cutoff_value = self.evaluate_potential(cutoff, atomic_number)
        cutoff_derivative = self.evaluate_potential_derivative(cutoff, atomic_number)
        parameters = self.parameters[atomic_number]
        potential_function = self._potential_function

        @jit(nopython=True)
        def soft_potential(r):
            v = potential_function(r, parameters)
            return v - cutoff_value - (r - cutoff) * cutoff_derivative

        return soft_potential

    @cached_method_with_args(('tolerance', 'origin', 'extent'))
    def _get_atomic_positions(self, atomic_number):
        atoms = self.get_padded_atoms()
        return atoms.get_positions()[np.where(atoms.numbers == atomic_number)]

    @cached_method_with_args(('tolerance',))
    def _get_radial_coordinates(self, atomic_number):
        n = int(np.ceil(self.get_cutoff(atomic_number) / self._interpolation_sampling))
        return np.geomspace(min(self.sampling) / 2, self.get_cutoff(atomic_number), n)

    @cached_method(())
    def _get_quadrature(self):
        m = self._quadrature_order
        h = QUADRATURE_PARAMETER_RATIO / self._quadrature_order
        return tanh_sinh_quadrature(m, h)

    def _evaluate_interpolation(self, i):
        v = self._allocate()
        v[:, :] = 0.

        slice_thickness = self.slice_thickness(i)
        xk, wk = self._get_quadrature()

        for atomic_number in self._get_unique_atomic_numbers():
            positions = self._get_atomic_positions(atomic_number)
            cutoff = self.get_cutoff(atomic_number)
            r = self._get_radial_coordinates(atomic_number)

            positions = positions[np.abs((i + .5) * slice_thickness - positions[:, 2]) < (cutoff + slice_thickness / 2)]

            z0 = i * slice_thickness - positions[:, 2]
            z1 = (i + 1) * slice_thickness - positions[:, 2]

            vr = project_tanh_sinh(r, z0, z1, xk, wk, self.get_soft_potential_function(atomic_number))

            block_margin = int(cutoff / min(self.sampling))

            corner_positions = np.round(positions[:, :2] / self.sampling).astype(np.int) - block_margin
            block_positions = positions[:, :2] - self.sampling * corner_positions

            block_size = 2 * block_margin + 1

            x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size)
            y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size)

            interpolation_kernel_parallel(v, r, vr, corner_positions, block_positions, x, y)

        return v / kappa

    def get_projected(self):
        projected = self.get_slice(0)
        for i in range(1, self.num_slices):
            projected._array += self._get_slice_array(i)
        return projected

    def _get_slice_array(self, i):
        if self._method == 'finite':
            return self._evaluate_interpolation(i)
        elif self._method == 'fourier':
            raise NotImplementedError()
        else:
            raise NotImplementedError('method {} not implemented')

    def copy(self, copy_atoms=False):
        if copy_atoms:
            return self.__class__(self.atoms.copy())
        else:
            return self.__class__(self.atoms)


def import_potential(path):
    npzfile = np.load(path)
    return PrecalculatedPotential(npzfile['array'], npzfile['slice_thicknesses'], npzfile['extent'])


class PrecalculatedPotential(PotentialBase, ArrayWithGrid):

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

        self._slice_thicknesses = slice_thicknesses

        super().__init__(extent=extent, sampling=sampling, spatial_dimensions=2, array=array)

    @property
    def array(self):
        return self._array

    @property
    def thickness(self):
        return self._slice_thicknesses.sum()

    @property
    def num_slices(self):
        return self._array.shape[0]

    def slice_thickness(self, i):
        return self._slice_thicknesses[i]

    def repeat(self, multiples):
        assert len(multiples) == 2
        new_array = np.tile(self._array, (1,) + multiples)
        new_extent = multiples * self.extent
        return self.__class__(array=new_array, thickness=self.thickness, extent=new_extent)

    def get_slice(self, i):
        return PotentialSlice(self._get_slice_array(i)[None], thickness=self.slice_thickness(i), extent=self.extent)

    def _get_slice_array(self, i):
        return self._array[i]

    def write(self, path):
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('class', (1,), dtype='S100')
            dset[:] = np.string_('abtem.potentials.PrecalculatedPotential')
            f.create_dataset('array', data=self.array)
            f.create_dataset('slice_thicknesses', data=self._slice_thicknesses)
            f.create_dataset('extent', data=self.extent)

    def projection(self):
        array = np.zeros(self.gpts, dtype=DTYPE)
        for potential_slice in self:
            array += potential_slice.array[0] * potential_slice.thickness
        return ArrayWithGrid2D(array, extent=self.extent)

    def copy(self, to_gpu=True):
        return self.__class__(array=cp.asarray(self.array), slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent.copy())
