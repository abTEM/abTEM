from collections import Iterable
from copy import copy

import h5py
import imageio
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

from abtem.bases import Grid
from abtem.device import asnumpy
from abtem.plot import show_image, show_line


class Calibration:

    def __init__(self, offset, sampling, units, name=''):
        self.offset = offset
        self.sampling = sampling
        self.units = units
        self.name = name

    def copy(self):
        return self.__class__(self.offset, self.sampling, self.units, self.name)


def fourier_space_offset(sampling, gpts):
    offset = np.zeros(len(sampling))
    for i, (d, p) in enumerate(zip(sampling, gpts)):
        if p % 2 == 0:
            offset[i] = -1 / (2 * d)
        else:
            offset[i] = -1 / (2 * d) + 1 / (2 * d * p)
    return offset


def calibrations_from_grid(grid: Grid, names=None, units=None, fourier_space=False, scale_factor=1):
    if names is None:
        names = ('',) * len(grid)
    elif len(names) != len(grid):
        raise RuntimeError()

    if units is None:
        if fourier_space:
            units = '1 / Å'
        else:
            units = 'Å'

    calibrations = []
    if fourier_space:
        for name, extent, offset in zip(names, grid.extent, fourier_space_offset(grid.sampling, grid.gpts)):
            calibrations.append(Calibration(offset * scale_factor, 1 / extent * scale_factor, units, name))
    else:
        for name, sampling in zip(names, grid.sampling):
            calibrations.append(Calibration(0., sampling * scale_factor, units, name))

    return calibrations


class Measurement:

    def __init__(self, array, calibrations, units='', name=''):
        if len(calibrations) != len(array.shape):
            raise RuntimeError()

        self._array = array
        self._calibrations = calibrations
        self._units = units
        self._name = name

    def __getitem__(self, args):
        if isinstance(args, Iterable):
            args += (slice(None),) * (len(self.array.shape) - len(args))
        else:
            args = (args,) + (slice(None),) * (len(self.array.shape) - 1)

        new_array = self.array[args]
        new_calibrations = []
        for i, (arg, calibration) in enumerate(zip(args, self.calibrations)):
            if isinstance(arg, slice):
                if arg.start is None:
                    offset = calibration.offset
                else:
                    offset = arg.start * calibration.sampling + calibration.offset

                new_calibrations.append(Calibration(offset=offset,
                                                    sampling=calibration.sampling,
                                                    units=calibration.units, name=calibration.name))
            elif not isinstance(arg, int):
                raise TypeError('indices must be integers or slices, not float')

        return self.__class__(new_array, new_calibrations)

    @property
    def units(self):
        return self._units

    @property
    def name(self):
        return self._name

    @property
    def dimensions(self):
        return len(self.array.shape)

    def dpc_difference(self, positive_indices, negative_indices):
        if self.dimensions < 3:
            raise RuntimeError()

        array = self.array.reshape(self.array.shape[:self.dimensions - 2] + (-1,))
        calibrations = self.calibrations[:self.dimensions - 2]

        differences = array[..., positive_indices] - array[..., negative_indices]
        if len(differences.shape) == len(array.shape):
            differences = differences.sum(-1)

        return self.__class__(differences, calibrations)

    def dpc_sum(self, indices=None):
        if self.dimensions < 3:
            raise RuntimeError()

        array = self.array.reshape(self.array.shape[:self.dimensions - 2] + (-1,))
        calibrations = self.calibrations[:self.dimensions - 2]

        if indices is None:
            sums = array.sum(-1)
        else:
            sums = array[..., indices]
            if len(sums.shape) == len(array.shape):
                sums = sums.sum(-1)

        return self.__class__(sums, calibrations)

    def blur(self, amount, **kwargs):
        calibrations = [calibration.copy() for calibration in self.calibrations]
        return self.__class__(gaussian_filter(self.array, amount, **kwargs), calibrations, units=self.units,
                              name=self.name)

    def mean(self, axes):
        if not isinstance(axes, Iterable):
            axes = (axes,)

        array = self.array.mean(axes)
        calibrations = [calibration for i, calibration in enumerate(self.calibrations) if i not in axes]

        return self.__class__(array, calibrations)

    def center_of_mass(self):
        shape = self.array.shape[2:]
        center = np.array(shape) / 2 - [.5 * (shape[0] % 2), .5 * (shape[1] % 2)]
        com = np.zeros(self.array.shape[:2] + (2,))
        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                com[i, j] = scipy.ndimage.measurements.center_of_mass(self.array[i, j])
        com = com - center[None, None]
        com[..., 0] = com[..., 0] * self.calibrations[2].sampling
        com[..., 1] = com[..., 1] * self.calibrations[3].sampling
        return (self.__class__(com[..., 0], self.calibrations[:2], units='mrad.', name='com_x'),
                self.__class__(com[..., 1], self.calibrations[:2], units='mrad.', name='com_y'))

    @property
    def calibrations(self):
        return self._calibrations

    def interpolate(self, new_sampling):

        scale_factors = [calibration.sampling / new_sampling for calibration in self.calibrations]
        new_array = zoom(self.array, scale_factors, mode='wrap')

        calibrations = []
        for calibration in self.calibrations:
            calibrations.append(copy(calibration))
            calibrations[-1].sampling = new_sampling

        return self.__class__(new_array, calibrations, units=self.units)

    def tile(self, repeats):
        new_array = np.tile(self._array, repeats)
        return self.__class__(new_array, self.calibrations)

    def poisson_noise(self, dose):
        pixel_area = np.product([calibration.sampling for calibration in self.calibrations])
        new_copy = copy(self)
        array = new_copy.array
        array[:] = array / np.sum(array) * dose * pixel_area * np.prod(array.shape)
        array[:] = np.random.poisson(array).astype(np.float)
        return new_copy

    @property
    def array(self):
        return self._array

    @classmethod
    def read(cls, path):
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        calibrations = []
        for i in range(len(datasets['offset'])):
            calibrations.append(Calibration(offset=datasets['offset'][i],
                                            sampling=datasets['sampling'][i],
                                            units=datasets['units'][i].decode('utf-8'),
                                            name=datasets['name'][i].decode('utf-8')))

        return cls(datasets['array'], calibrations)

    def write(self, path, mode='w'):
        with h5py.File(path, mode) as f:
            f.create_dataset('array', data=self.array)
            f.create_dataset('offset', data=[calibration.offset for calibration in self.calibrations])
            f.create_dataset('sampling', data=[calibration.sampling for calibration in self.calibrations])
            units = [calibration.units.encode('utf-8') for calibration in self.calibrations]
            f.create_dataset('units', (len(units),), 'S10', units)
            names = [calibration.name.encode('utf-8') for calibration in self.calibrations]
            f.create_dataset('name', (len(names),), 'S10', names)

        return path

    def save_as_image(self, path):
        array = (self.array - self.array.min()) / self.array.ptp() * np.iinfo(np.uint16).max
        array = array.astype(np.uint16)
        imageio.imwrite(path, array.T)

    def __copy__(self):
        calibrations = []
        for calibration in self.calibrations:
            calibrations.append(copy(calibration))
        return self.__class__(self._array.copy(), calibrations=calibrations)

    def show(self, **kwargs):
        calibrations = [calib for calib, num_elem in zip(self.calibrations, self.array.shape) if num_elem > 1]
        array = np.squeeze(asnumpy(self.array))

        dims = len(array.shape)
        cbar_label = self._name + ' [' + self._units + ']'
        if dims == 1:
            return show_line(array, calibrations[0], **kwargs)
        elif dims == 2:
            return show_image(array, calibrations, cbar_label=cbar_label, **kwargs)
        else:
            raise RuntimeError('plotting not supported for {}d measurement, use reduction operation first'.format(dims))
