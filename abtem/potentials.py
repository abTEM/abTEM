import os
from typing import Union, Sequence

import numpy as np
from abtem.bases import Grid, cached_method, ArrayWithGrid, Cache, cached_method_with_args
from abtem.interpolation import interpolation_kernel_parallel
from abtem.parametrizations import load_lobato_parameters, load_kirkland_parameters, kirkland, dvdr_kirkland, \
    project_tanh_sinh
from abtem.parametrizations import lobato, dvdr_lobato
from abtem.transform import fill_rectangle_with_atoms
from ase import Atoms
from ase import units
from numba import jit
from scipy.optimize import brentq
from tqdm.auto import tqdm

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

QUADRATURE_PARAMETER_RATIO = 4
DTYPE = np.float32


class PotentialBase(Grid):

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
    def origin(self):
        return self._origin

    @property
    def num_slices(self):
        return self._num_slices

    @property
    def thickness(self):
        return self._atoms.cell[2, 2]

    def get_slice(self, i):
        raise NotImplementedError()

    def check_slice_idx(self, i):
        if i >= self.num_slices:
            raise RuntimeError('slice index {} too large'.format(i))

    def __getitem__(self, i):
        if i >= self.num_slices:
            raise StopIteration

        return self.get_slice(i)

    def slice_thickness(self, i):
        raise NotImplementedError()

    def precalculate(self, show_progress=False):
        array = np.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]))
        slice_thicknesses = np.zeros(self.num_slices)

        for i in tqdm(range(self.num_slices), disable=not show_progress):
            array[i] = self.get_slice(i)
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


class Potential(PotentialBase, Cache):
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
    method : 'interpolation'
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
                 method: str = 'interpolation',
                 cutoff_tolerance: float = 1e-2,
                 quadrature_order: int = 10,
                 interpolation_sampling: float = .01):

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
    def num_slices(self):
        return self._num_slices

    @property
    def atoms(self):
        return self._atoms

    @property
    def thickness(self):
        return self.atoms.cell[2, 2]

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

    def _get_slice_array(self, i):
        if self._method == 'interpolation':
            return self._evaluate_interpolation(i)
        elif self._method == 'fourier':
            raise NotImplementedError()
        else:
            raise NotImplementedError('method {} not implemented')

    def get_slice(self, i):
        
        return ArrayWithGrid(self._get_slice_array(i)[None], extent=self.extent, spatial_dimensions=2)

    def copy(self, copy_atoms=False):
        if copy_atoms:
            return self.__class__(self.atoms.copy())
        else:
            return self.__class__(self.atoms)


def import_potential(path):
    npzfile = np.load(path)
    return PrecalculatedPotential(npzfile['array'], npzfile['slice_thicknesses'], npzfile['extent'])


class PrecalculatedPotential(ArrayWithGrid):

    def __init__(self, array: np.ndarray, slice_thicknesses: Union[float, Sequence], extent: np.ndarray = None,
                 sampling: np.ndarray = None):

        """
        Precalculated potential object. This object


        Parameters
        ----------
        array :
        slice_thicknesses :
        extent :
        sampling :
        """

        slice_thicknesses = np.array(slice_thicknesses)

        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, array.shape[0])
        elif slice_thicknesses.shape != (array.shape[0],):
            raise ValueError()

        self._slice_thicknesses = slice_thicknesses

        super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling)

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

    def downsample(self):
        N, M = self.gpts

        X = np.fft.fft(self._array, axis=1)
        self._array = np.fft.ifft((X[:, :(N // 2), :] + X[:, -(N // 2):, :]) / 2., axis=1).real

        X = np.fft.fft(self._array, axis=2)
        self._array = np.fft.ifft((X[:, :, :(M // 2)] + X[:, :, -(M // 2):]) / 2., axis=2).real

        self.extent = self.extent

    def get_slice(self, i):
        return self._array[i]

    def extract(self, first, last):
        if last < 0:
            last = self.num_slices + last

        if first >= last:
            raise RuntimeError()

        slice_thicknesses = self._slice_thicknesses[first:last]
        return self.__class__(array=self.array[first:last], slice_thicknesses=slice_thicknesses, extent=self.extent)

    def export(self, path, overwrite=False):
        if (os.path.isfile(path) | os.path.isfile(path + '.npz')) & (not overwrite):
            raise RuntimeError('file {} already exists')

        np.savez(path, array=self.array, slice_thicknesses=self._slice_thicknesses, extent=self.extent)
