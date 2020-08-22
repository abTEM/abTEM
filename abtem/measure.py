"""Module to describe the detection of scattered electron waves."""
from collections.abc import Iterable
from copy import copy
from typing import Sequence

import h5py
import imageio
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.ndimage import zoom

from abtem.device import asnumpy
from abtem.plot import show_image, show_line


class Calibration:
    """
    Calibration object

    The calibration object represents the sampling of a uniformly sampled Measurement.

    Parameters
    ----------
    offset: float
        The lower bound of the sampling points.
    sampling: float
        The distance between sampling points.
    units: str
        The units of the calibration shown in plots.
    name:
        The name of this calibration to be shown in plots.
    """

    def __init__(self, offset: float, sampling: float, units: str, name: str = ''):
        self.offset = offset
        self.sampling = sampling
        self.units = units
        self.name = name

    def __eq__(self, other):
        return ((self.offset == other.offset) &
                (self.sampling == other.sampling) &
                (self.units == other.units) &
                (self.name == other.name))

    def __copy__(self):
        return self.__class__(self.offset, self.sampling, self.units, self.name)

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


def fourier_space_offset(n, d):
    """
    Fourier-space offset function

    Function for calculating an offset in Fourier space.

    Parameters
    ----------
    n : int
        JM
    d : int
        JM
    """

    if n % 2 == 0:
        return -1 / (2 * d)
    else:
        return -1 / (2 * d) + 1 / (2 * d * n)


def calibrations_from_grid(gpts: Sequence[int],
                           sampling: Sequence[float],
                           names: Sequence[str] = None,
                           units: str = None,
                           fourier_space: bool = False,
                           scale_factor: float = 1.0) -> Sequence[Calibration]:
    """
    Returns the spatial calibrations for a given computational grid and sampling.

    Parameters
    ----------
    gpts: list of int
        Number of grid points in the x and y directions.
    sampling: list of float
        Sampling of the potential in Å.
    names: list of str, optional
        The name of this calibration.
    units: str, optional
        Units for the calibration.
    fourier_space: bool, optional
        Setting for calibrating either in the reciprocal or real space. Default is False.
    scale_factor: float, optional
        Scaling factor for the calibration. Default is 1.0.
    """

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
            r = n * d
            offset = fourier_space_offset(n, d)
            calibrations += (Calibration(offset * scale_factor, 1 / r * scale_factor, units, name),)
    else:
        for name, d in zip(names, sampling):
            calibrations += (Calibration(0., d * scale_factor, units, name),)

    return calibrations


class Measurement:
    """
    Measurement object.

    Parameters
    ----------
    array: ndarray
        The array representing the measurements. The array can be any dimension.
    calibrations: list of Calibration objects
        The calibration for each dimension of the measurement array.
    units: str
        The units of the array values to be displayed in plots.
    name: str
        The name of the array values to be displayed in plots.
    """

    def __init__(self, array, calibrations, units='', name=''):

        if len(calibrations) != len(array.shape):
            raise RuntimeError(
                'The number of calibrations must equal the number of array dimensions. For undefined calibrations use None.')

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
        """
        The shape of the measurement array.
        """
        return self._array.shape

    @property
    def units(self):
        """
        The units of the array values to be displayed in plots.
        """
        return self._units

    @property
    def name(self):
        """
        The name of the array values to be displayed in plots.
        """
        return self._name

    @property
    def dimensions(self):
        """

        Returns
        -------

        """
        return len(self.array.shape)

    def __sub__(self, other):
        assert isinstance(other, self.__class__)

        for calibration, other_calibration in zip(self.calibrations, other.calibrations):
            if not calibration == other_calibration:
                raise ValueError()

        difference = self.array - other.array
        return self.__class__(difference, calibrations=self.calibrations, units=self.units, name=self.name)

    def __add__(self, other):
        assert isinstance(other, self.__class__)

        for calibration, other_calibration in zip(self.calibrations, other.calibrations):
            if not calibration == other_calibration:
                raise ValueError()

        difference = self.array - other.array
        return self.__class__(difference, calibrations=self.calibrations, units=self.units, name=self.name)

    @property
    def calibrations(self):
        return self._calibrations

    def _reduction(self, reduction_function, axis):
        if not isinstance(axis, Iterable):
            axis = (axis,)

        array = reduction_function(self.array, axis=axis)

        axis = [d % len(self.calibrations) for d in axis]
        calibrations = [self.calibrations[i] for i in range(len(self.calibrations)) if i not in axis]

        return self.__class__(array, calibrations)

    def sum(self, axis):
        return self._reduction(np.mean, axis)

    def mean(self, axis):
        return self._reduction(np.mean, axis)

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

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)

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
            raise RuntimeError('Plotting not supported for {}D measurement, use reduction operation first'.format(dims))


def fwhm(y):
    """Function for calculating the full width at half maximum value for a 1D function."""
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    left = np.argmin(np.abs(y[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(y[peak_idx:] - peak_value / 2))
    return right - left


def center_of_mass(measurement: Measurement):
    """Function for estimating the intensity center-of-mass for a given measurement."""

    if (measurement.dimensions != 3) or (measurement.dimensions != 4):
        raise RuntimeError()

    shape = measurement.array.shape[-2:]
    center = np.array(shape) / 2 - np.array([.5 * (shape[-2] % 2), .5 * (shape[-1] % 2)])
    com = np.zeros(measurement.array.shape[:-2] + (2,))
    for i in range(measurement.array.shape[0]):
        for j in range(measurement.array.shape[1]):
            com[i, j] = scipy.ndimage.measurements.center_of_mass(measurement.array[i, j])
    com = com - center[None, None]
    com[..., 0] = com[..., 0] * measurement.calibrations[-2].sampling
    com[..., 1] = com[..., 1] * measurement.calibrations[-1].sampling
    return (Measurement(com[..., 0], measurement.calibrations[:-2], units='mrad', name='com_x'),
            Measurement(com[..., 1], measurement.calibrations[:-2], units='mrad', name='com_y'))
