import os
import numbers
import numpy as np
from ase import units
from scipy.optimize import brentq
from tqdm.auto import tqdm

from abtem.bases import Grid, HasCache, cached_method, ArrayWithGrid
from abtem.interpolation import interpolation_kernel_parallel
from abtem.parametrizations import convert_kirkland, kirkland, kirkland_projected_finite, dvdr_kirkland, load_parameters
from abtem.parametrizations import convert_lobato, lobato, lobato_projected_finite, dvdr_lobato
from abtem.transform import make_orthogonal_atoms

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


class PotentialBase(Grid):

    def __init__(self, atoms, origin=None, extent=None, gpts=None, sampling=None, num_slices=None):

        if np.abs(atoms.cell[0, 0]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        if (np.abs(atoms.cell[0, 0]) < 1e-12) | (np.abs(atoms.cell[1, 1]) < 1e-12):
            raise RuntimeError('atoms has no width')

        self._atoms = atoms.copy()
        self._atoms.wrap()

        if origin is None:
            self._origin = np.array([0., 0.])
        else:
            self._origin = origin

        if extent is None:
            extent = np.diag(atoms.cell)[:2]

        self._num_slices = num_slices

        super().__init__(extent=extent, gpts=gpts, sampling=sampling)

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
        if i >= self.num_slices:
            raise RuntimeError()

    def slice_thickness(self, i):
        raise NotImplementedError()

    def precalculate(self):
        array = np.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]))
        slice_thicknesses = np.zeros(self.num_slices)

        for i in tqdm(range(self.num_slices)):
            array[i] = self.get_slice(i)
            slice_thicknesses[i] = self.slice_thickness(i)

        return PrecalculatedPotential(array, slice_thicknesses, self.extent)


class Potential(PotentialBase, HasCache):

    def __init__(self, atoms, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5, num_slices=None,
                 parametrization='lobato', method='interpolation', tolerance=1e-3):

        self._tolerance = tolerance
        self._method = method

        if isinstance(parametrization, str):
            if parametrization == 'lobato':
                self._potential_func = lobato
                self._projected_func = lobato_projected_finite
                self._diff_func = dvdr_lobato
                self._convert_param_func = convert_lobato
                self._parameters = load_parameters('data/lobato.txt')

            elif parametrization == 'kirkland':
                self._potential_func = kirkland
                self._projected_func = kirkland_projected_finite
                self._diff_func = dvdr_kirkland
                self._convert_param_func = convert_kirkland
                self._parameters = load_parameters('data/kirkland.txt')

            else:
                raise RuntimeError()

        if num_slices is None:
            if slice_thickness is None:
                raise RuntimeError()

            num_slices = int(np.floor(atoms.cell[2, 2] / slice_thickness))

        super().__init__(atoms=atoms.copy(), origin=origin, extent=extent, gpts=gpts, sampling=sampling,
                         num_slices=num_slices)

    def slice_thickness(self, i):
        return self.thickness / self.num_slices

    @property
    def tolerance(self):
        return self._tolerance

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

    def get_atoms(self, cutoff=0.):
        return make_orthogonal_atoms(self.atoms, self._origin, self.extent, cutoff=cutoff)

    @cached_method()
    def _prepare_interpolation(self):
        unique_atomic_numbers = np.unique(self._atoms.get_atomic_numbers())

        data = {}
        max_cutoff = 0.
        for atomic_number in unique_atomic_numbers:
            data[atomic_number] = {}
            data[atomic_number]['parameters'] = self._convert_param_func(self.parameters[atomic_number])

            func = lambda x: self._potential_func(x, *data[atomic_number]['parameters']) - self.tolerance
            data[atomic_number]['r_cut'] = brentq(func, 1e-7, 1000)
            max_cutoff = max(data[atomic_number]['r_cut'], max_cutoff)

            data[atomic_number]['v_cut'] = self._potential_func(data[atomic_number]['r_cut'],
                                                                *data[atomic_number]['parameters'])
            data[atomic_number]['dvdr_cut'] = self._diff_func(data[atomic_number]['r_cut'],
                                                              *data[atomic_number]['parameters'])

        atoms = self.get_atoms(cutoff=max_cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()

        for atomic_number in unique_atomic_numbers:
            data[atomic_number]['positions'] = positions[np.where(atomic_numbers == atomic_number)]

        v = np.zeros(self.gpts)
        return v, data

    def _evaluate_interpolation(self, i, num_spline_nodes=200, num_integration_samples=100):
        v, data_dict = self._prepare_interpolation()
        v[:, :] = 0.

        for atomic_number, data in data_dict.items():
            positions = data['positions']
            parameters = data['parameters']
            r_cut = data['r_cut']
            v_cut = data['v_cut']
            dvdr_cut = data['dvdr_cut']

            r = np.linspace(min(self.sampling) / 2., r_cut, num_spline_nodes)

            block_margin = int(r_cut / min(self.sampling))
            block_size = 2 * block_margin + 1

            positions = positions[np.abs(i * self.slice_thickness(i) + self.slice_thickness(i) / 2 - positions[:, 2]) <
                                  (r_cut + self.slice_thickness(i) / 2)]

            corner_positions = np.round(positions[:, :2] / self.sampling).astype(np.int) - block_margin
            block_positions = positions[:, :2] - self.sampling * corner_positions

            x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size)
            y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size)

            z0 = i * self.slice_thickness(i) - positions[:, 2]
            z1 = (i + 1) * self.slice_thickness(i) - positions[:, 2]

            vr = self._projected_func(r, r_cut, v_cut, dvdr_cut, z0, z1, num_integration_samples, *parameters)

            interpolation_kernel_parallel(v, r, vr, corner_positions, block_positions, x, y)

        return v / kappa

    def get_slice(self, i):
        if self._method == 'interpolation':
            return self._evaluate_interpolation(i)
        elif self._method == 'fourier':
            raise RuntimeError()
        else:
            raise NotImplementedError('method {} not implemented')


def import_potential(path):
    npzfile = np.load(path)
    return PrecalculatedPotential(npzfile['array'], npzfile['slice_thicknesses'], npzfile['extent'])


class PrecalculatedPotential(ArrayWithGrid):

    def __init__(self, array, slice_thicknesses, extent=None, sampling=None):

        if isinstance(slice_thicknesses, numbers.Number):
            slice_thicknesses = np.full(array.shape[0], slice_thicknesses, dtype=np.float)

        self._slice_thicknesses = slice_thicknesses

        super().__init__(array=array, array_dimensions=3, spatial_dimensions=2, extent=extent, sampling=sampling,
                         space='direct')

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
