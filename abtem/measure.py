from collections import Iterable
from copy import copy

import h5py
import imageio
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

from abtem.device import asnumpy
from abtem.plot import show_image, show_line


class Calibration:

    def __init__(self, offset, sampling, units, name=''):
        self.offset = offset
        self.sampling = sampling
        self.units = units
        self.name = name

    def __eq__(self, other):
        return ((self.offset == other.offset) &
                (self.sampling == other.sampling) &
                (self.units == other.units) &
                (self.name == other.name))

    def copy(self):
        return self.__class__(self.offset, self.sampling, self.units, self.name)


def fourier_space_offset(n, d):
    if n % 2 == 0:
        return -1 / (2 * d)
    else:
        return -1 / (2 * d) + 1 / (2 * d * n)


def calibrations_from_grid(gpts, sampling, names=None, units=None, fourier_space=False, scale_factor=1):
    if names is None:
        names = ('',) * len(gpts)
    elif len(names) != len(gpts):
        raise RuntimeError()

    if units is None:
        if fourier_space:
            units = '1 / Å'
        else:
            units = 'Å'

    calibrations = ()
    if fourier_space:
        for name, n, d in zip(names, gpts, sampling):
            l = n * d
            offset = fourier_space_offset(n, d)
            calibrations += (Calibration(offset * scale_factor, 1 / l * scale_factor, units, name),)
    else:
        for name, d in zip(names, sampling):
            calibrations += (Calibration(0., d * scale_factor, units, name),)

    return calibrations


def fwhm(y):
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    left = np.argmin(np.abs(y[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(y[peak_idx:] - peak_value / 2))
    return right - left


def center_of_mass(measurement):
    shape = measurement.array.shape[2:]
    center = np.array(shape) / 2 - [.5 * (shape[0] % 2), .5 * (shape[1] % 2)]
    com = np.zeros(measurement.array.shape[:2] + (2,))
    for i in range(measurement.array.shape[0]):
        for j in range(measurement.array.shape[1]):
            com[i, j] = scipy.ndimage.measurements.center_of_mass(measurement.array[i, j])
    com = com - center[None, None]
    com[..., 0] = com[..., 0] * measurement.calibrations[2].sampling
    com[..., 1] = com[..., 1] * measurement.calibrations[3].sampling
    return (Measurement(com[..., 0], measurement.calibrations[:2], units='mrad.', name='com_x'),
            Measurement(com[..., 1], measurement.calibrations[:2], units='mrad.', name='com_y'))


class Measurement:

    def __init__(self, array, calibrations, units='', name=''):
        if len(calibrations) != len(array.shape):
            raise RuntimeError()

        self._array = asnumpy(array)
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
            elif isinstance(arg, Iterable):
                new_calibrations.append(None)

            elif not isinstance(arg, int):
                raise TypeError('Indices must be integers or slices, not float')

        return self.__class__(new_array, new_calibrations)

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        return self._array.shape

    @property
    def units(self):
        return self._units

    @property
    def name(self):
        return self._name

    @property
    def dimensions(self):
        return len(self.array.shape)

    def __sub__(self, other):
        assert isinstance(other, self.__class__)

        for calibration, other_calibration in zip(self.calibrations, other.calibrations):
            if not calibration == other_calibration:
                raise ValueError()

        difference = self.array - other.array
        return self.__class__(difference, calibrations=self.calibrations, units=self.units, name=self.name)

    def mean(self, axis):
        if not isinstance(axis, Iterable):
            axis = (axis,)

        array = np.mean(self.array, axis=axis)

        axis = [d % len(self.calibrations) for d in axis]
        calibrations = [self.calibrations[i] for i in range(len(self.calibrations)) if i not in axis]

        return self.__class__(array, calibrations)

    @property
    def calibrations(self):
        return self._calibrations

    def interpolate(self, new_sampling):
        import warnings
        warnings.filterwarnings('ignore', '.*output shape of zoom.*')

        scale_factors = [calibration.sampling / new_sampling for calibration in self.calibrations]
        new_array = zoom(self.array, scale_factors, mode='wrap')

        calibrations = []
        for calibration in self.calibrations:
            calibrations.append(copy(calibration))
            calibrations[-1].sampling = new_sampling

        return self.__class__(new_array, calibrations, name=self.name, units=self.units)

    def tile(self, repeats):
        new_array = np.tile(self._array, repeats)
        return self.__class__(new_array, self.calibrations, name=self.name, units=self.units)

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
            raise RuntimeError('Plotting not supported for {}d measurement, use reduction operation first'.format(dims))
