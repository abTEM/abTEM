import os

import numpy as np
from ase import units
from scipy.optimize import brentq

from abtem.bases import Grid, HasCache, cached_method, ArrayWithGrid
from abtem.interpolation import interpolation_kernel, thread_safe_coloring
from abtem.parametrizations import convert_kirkland, kirkland, kirkland_projected_finite, dvdr_kirkland, load_parameters
from abtem.parametrizations import convert_lobato, lobato, lobato_projected_finite, dvdr_lobato
from abtem.transform import make_orthogonal_atoms

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


class PotentialBase(Grid):

    def __init__(self, extent=None, gpts=None, sampling=None):
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)

    @property
    def num_slices(self):
        raise NotImplementedError()

    @property
    def thickness(self):
        raise NotImplementedError()

    @property
    def slice_thickness(self):
        return self.thickness / self.num_slices

    def get_slice(self, i):
        raise NotImplementedError

    def slice_entrance(self, i):
        return i * self.slice_thickness

    def slice_exit(self, i):
        return (i + 1) * self.slice_thickness

    def precalculate(self):
        array = np.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]))

        for i in range(self.num_slices):
            array[i] = self.get_slice(i)

        return PrecalculatedPotential(array, self.thickness, self.extent)


class Potential(HasCache, PotentialBase):

    def __init__(self, atoms, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5, num_slices=None,
                 parametrization='lobato', periodic=True, method='interpolation', tolerance=1e-3,
                 truncation='derivative'):

        if np.abs(atoms.cell[0, 0]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        if (np.abs(atoms.cell[0, 0]) < 1e-12) | (np.abs(atoms.cell[1, 1]) < 1e-12):
            raise RuntimeError('atoms has no width')

        # if np.any(np.abs(atoms.cell[~np.eye(3, dtype=bool)]) > 1e-12):
        #    raise RuntimeError('non-diagonal unit cell not supported')

        if periodic is False:
            raise NotImplementedError()

        self._atoms = atoms.copy()
        self._atoms.wrap()

        if origin is None:
            self._origin = np.array([0., 0.])
        else:
            self._origin = origin

        if num_slices is None:
            if slice_thickness is None:
                raise RuntimeError()

            self._num_slices = int(np.floor(atoms.cell[2, 2] / slice_thickness))
        else:
            self._num_slices = num_slices

        self._tolerance = tolerance
        self._periodic = periodic
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

        if extent is None:
            extent = np.diag(atoms.cell)[:2]

        HasCache.__init__(self)
        PotentialBase.__init__(self, extent=extent, gpts=gpts, sampling=sampling)

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

    @cached_method
    def _prepare_interpolation(self):
        atoms = make_orthogonal_atoms(self.atoms, self._origin, self.extent)

        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)

        data = {}
        max_cutoff = 0.
        for atomic_number in unique_atomic_numbers:
            data[atomic_number] = {}
            data[atomic_number]['positions'] = positions[np.where(atomic_numbers == atomic_number)]
            data[atomic_number]['parameters'] = self._convert_param_func(self.parameters[atomic_number])

            func = lambda x: self._potential_func(x, *data[atomic_number]['parameters']) - self.tolerance
            data[atomic_number]['r_cut'] = np.float32(brentq(func, 1e-7, 1000))
            max_cutoff = max(data[atomic_number]['r_cut'], max_cutoff)

            data[atomic_number]['v_cut'] = np.float32(self._potential_func(data[atomic_number]['r_cut'],
                                                                           *data[atomic_number]['parameters']))
            data[atomic_number]['dvdr_cut'] = np.float32(self._diff_func(data[atomic_number]['r_cut'],
                                                                         *data[atomic_number]['parameters']))

        margin = int(np.ceil(max_cutoff / min(self.sampling)))
        padded_gpts = self.gpts + 2 * margin
        v = np.zeros((padded_gpts[0], padded_gpts[1]), dtype=np.float32)
        return v, margin, data

    def _evaluate_interpolation(self, i, num_spline_nodes=200, num_integration_samples=100):
        v, margin, data_dict = self._prepare_interpolation()
        v[:, :] = 0.

        for atomic_number, data in data_dict.items():
            positions = data['positions']
            parameters = data['parameters']
            r_cut = data['r_cut']
            v_cut = data['v_cut']
            dvdr_cut = data['dvdr_cut']

            r = np.linspace(min(self.sampling) / 2., r_cut, num_spline_nodes, dtype=np.float32)

            block_margin = int(r_cut / min(self.sampling))
            block_size = 2 * block_margin + 1

            positions = positions[np.abs(self.slice_entrance(i) + self.slice_thickness / 2 - positions[:, 2]) <
                                  (r_cut + self.slice_thickness / 2)]

            positions = positions.astype(np.float32)

            corner_positions = np.round(positions[:, :2] / self.sampling).astype(np.int32) - block_margin + margin
            block_positions = positions[:, :2] + self.sampling * margin - self.sampling * corner_positions.astype(
                np.float32)

            x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size, dtype=np.float32)
            y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size, dtype=np.float32)

            z0 = np.float32(self.slice_entrance(i) - positions[:, 2])
            z1 = np.float32(self.slice_exit(i) - positions[:, 2])

            vr = self._projected_func(r, r_cut, v_cut, dvdr_cut, z0, z1, num_integration_samples, *parameters)
            colors = thread_safe_coloring(corner_positions, block_size)

            interpolation_kernel(v, r, vr, r_cut, corner_positions, block_positions, x, y, colors)

        v[margin:2 * margin] += v[-margin:]
        v[-2 * margin:-margin] += v[:margin]
        v[:, margin:2 * margin] += v[:, -margin:]
        v[:, -2 * margin:-margin] += v[:, :margin]
        v = v[margin:-margin, margin:-margin]

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
    return PrecalculatedPotential(npzfile['array'], npzfile['thickness'], npzfile['extent'])


class PrecalculatedPotential(ArrayWithGrid):

    def __init__(self, array, thickness, extent=None, sampling=None):
        ArrayWithGrid.__init__(self, array, 3, 2, extent=extent, sampling=sampling, space='direct')

        self._thickness = thickness

    @property
    def thickness(self):
        return self._thickness

    @property
    def num_slices(self):
        return self._array.shape[0]

    @property
    def slice_thickness(self):
        return self.thickness / self.num_slices

    def repeat(self, multiples):
        assert len(multiples) == 2
        self._array = np.tile(self._array, (1,) + multiples)
        self.extent = multiples * self.extent

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

        if first <= last:
            raise RuntimeError()

        thickness = self.slice_thickness * (last - first)
        return self.__class__(array=self.array[first:last], thickness=thickness, extent=self.extent)

    def export(self, path, overwrite=False):
        if (os.path.isfile(path) | os.path.isfile(path + '.npz')) & (not overwrite):
            raise RuntimeError('file {} already exists')

        np.savez(path, array=self.array, thickness=self.thickness, extent=self.extent)
