"""Module to describe the detection of scattered electron waves."""
from collections.abc import Iterable
from abc import ABCMeta
from copy import copy
from typing import Sequence, Tuple

import h5py
import imageio
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.interpolate import interpn
from scipy.ndimage import zoom

from abtem.device import asnumpy
from abtem.visualize.mpl import show_measurement_2d, show_measurement_1d


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
    name: str
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


def _fourier_space_offset(n: int, d: float):
    """
    Calculate the Fourier space offset.

    Parameters
    ----------
    n : int
        Number of sampling points.
    d : float
        Real space sampling density.
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
                           scale_factor: float = 1.0) -> Tuple[Calibration]:
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

    Returns
    -------
    calibrations: Tuple of Calibrations
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
            offset = _fourier_space_offset(n, d)
            calibrations += (Calibration(offset * scale_factor, 1 / r * scale_factor, units, name),)
    else:
        for name, d in zip(names, sampling):
            calibrations += (Calibration(0., d * scale_factor, units, name),)

    return calibrations


class Measurement:  # (metaclass=ABCMeta):
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

    def __init__(self, array, calibrations=None, units='', name=''):

        if issubclass(array.__class__, self.__class__):
            measurement = array

            array = measurement.array
            calibrations = measurement.calibrations
            units = measurement.array
            name = measurement.name

        if not isinstance(calibrations, Iterable):
            calibrations = [calibrations] * len(array.shape)

        if len(calibrations) != len(array.shape):
            raise RuntimeError(
                'The number of calibrations must equal the number of array dimensions. For undefined use None.')

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

    @property
    def extent(self):
        extent = ()
        for i, calibration in enumerate(self.calibrations):
            if calibration is None:
                extent += (None, None)
            else:
                extent += (calibration.offset, calibration.offset + self.array.shape[i] * calibration.sampling)
        return extent

    def __len__(self):
        return self.shape[0]

    @property
    def array(self):
        """
        Array of measurements.
        """
        return self._array

    # @array.setter
    # def array(self, array):
    #     """
    #     Array of measurements.
    #     """
    #     self._array[:] = array

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
        The measurement dimensions.
        """
        return len(self.array.shape)

    @property
    def calibrations(self):
        """
        The measurement calibrations.
        """
        return self._calibrations

    def check_match_calibrations(self, other):
        for calibration, other_calibration in zip(self.calibrations, other.calibrations):
            if not calibration == other_calibration:
                raise ValueError('Calibration mismatch, operation not possible.')

    def __isub__(self, other):
        if isinstance(other, self.__class__):
            self.check_match_calibrations(other)
            self._array -= other.array
        else:
            self._array -= asnumpy(other)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            self.check_match_calibrations(other)
            new_array = self.array - other.array
        else:
            new_array = self._array - asnumpy(other)
        return self.__class__(new_array, calibrations=self.calibrations, units=self.units, name=self.name)

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self.check_match_calibrations(other)
            self._array += other.array
        else:
            self._array += asnumpy(other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            self.check_match_calibrations(other)
            new_array = self.array + other.array
        else:
            new_array = self._array + asnumpy(other)
        return self.__class__(new_array, calibrations=self.calibrations, units=self.units, name=self.name)

    def _reduction(self, reduction_function, axis):
        if not isinstance(axis, Iterable):
            axis = (axis,)

        array = reduction_function(self.array, axis=axis)

        axis = [d % len(self.calibrations) for d in axis]
        calibrations = [self.calibrations[i] for i in range(len(self.calibrations)) if i not in axis]

        return self.__class__(array, calibrations)

    # def integrate(self, axes, start=0, end=None, use_units=True):
    #
    #     if end is None:
    #         end = ()
    #         for axis in axes:
    #             end += (self.array.shape[axis],)
    #
    #     else:
    #         end = list(end)
    #         for axis in axes:
    #             end[axis]
    #
    #     if not isinstance(axes, Iterable):
    #         axes = (axes,)
    #
    #     if not isinstance(start, Iterable):
    #         start = (start,) * len(axes)
    #
    #     if not isinstance(end, Iterable):
    #         end = (end,) * len(axes)

    def sum(self, axis):
        """
        Sum of measurment elements over a given axis.

        Parameters
        ----------
        axis: int or tuple of ints
            Axis or axes along which a sum is performed. If axis is negative it counts from the last to the first axis.

        Returns
        -------
        Measurement
            A measurement with the same shape, but with the specified axis removed.
        """
        return self._reduction(np.mean, axis)

    def mean(self, axis):
        """
        Mean of measurment elements over a given axis.

        Parameters
        ----------
        axis: int or tuple of ints
            Axis or axes along which a sum is performed. If axis is negative it counts from the last to the first axis.

        Returns
        -------
        Measurement object
            A measurement with the same shape, but with the specified axis removed.
        """
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

    def tile(self, multiples):
        """
        Tile the measurement.

        Parameters
        ----------
        multiples: two int
            The number of repetitions of the measurement along each axis.

        Returns
        -------
        Measurement object
            The tiled potential.
        """
        new_array = np.tile(self._array, multiples)
        return self.__class__(new_array, self.calibrations, name=self.name, units=self.units)

    @classmethod
    def read(cls, path):
        """
        Read measurement from a hdf5 file.

        path: str
            The path to read the file.
        """

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

        # print(datasets['array'].shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(datasets['array'])
        # plt.show()

        return cls(datasets['array'], calibrations)

    def write(self, path, mode='w'):
        """
        Write measurement to a hdf5 file.

        path: str
            The path to write the file.
        """

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
        """
        Write the measurement array to an image file. The array will be normalized and converted to 16-bit integers.

        path: str
            The path to write the file.
        """

        if self.dimensions != 2:
            raise RuntimeError('Only 2d measurements can be saved as an image.')

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

    def squeeze(self):
        new_meaurement = self.copy()
        calibrations = [calib for calib, num_elem in zip(self.calibrations, self.array.shape) if num_elem > 1]
        new_meaurement._calibrations = calibrations
        new_meaurement._array = np.squeeze(asnumpy(new_meaurement.array))
        return new_meaurement

    def interpolate_line(self, start, end, gpts=None, sampling=None):
        from abtem.scan import LineScan

        if not (self.dimensions == 2):
            raise RuntimeError()

        if (self.calibrations[0] is None) or (self.calibrations[1] is None):
            raise RuntimeError()

        if self.calibrations[0].units != self.calibrations[1].units:
            raise RuntimeError()

        if (gpts is None) & (sampling is None):
            sampling = (self.calibrations[0].sampling + self.calibrations[1].sampling) / 2.

        x = np.linspace(self.calibrations[0].offset, self.shape[0] * self.calibrations[0].sampling, self.shape[0])
        y = np.linspace(self.calibrations[1].offset, self.shape[1] * self.calibrations[1].sampling, self.shape[1])

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling)

        interpolated_array = interpn((x, y), self.array, scan.get_positions())

        calibration = Calibration(offset=0, sampling=scan.sampling[0],
                                  units=self.calibrations[0].units,
                                  name=self.calibrations[0].name)
        return Measurement(interpolated_array, calibration)

    def show(self, ax=None, **kwargs):
        """
        Show the measurement.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments for the abtem.plot.show_image function.
        """
        if self.dimensions == 1:
            return show_measurement_1d(self, ax=ax, **kwargs)
        else:
            return show_measurement_2d(self, ax=ax, **kwargs)


class FlexibleAnnularMeasurement(Measurement):

    def __init__(self, array, spatial_sampling, angular_sampling, angular_offset=0.):
        if not isinstance(spatial_sampling, Iterable):
            spatial_sampling = (spatial_sampling,) * (len(array.shape) - 1)

        calibrations = [Calibration(0., d, 'Å', name) for d, name in zip(spatial_sampling, ('x', 'y'))]
        calibrations += [Calibration(angular_offset, angular_sampling, 'mrad', 'alpha')]

        super().__init__(array, calibrations)

    def integrate(self, start, end):
        offset = self.calibrations[-1].offset
        sampling = self.calibrations[-1].sampling

        start = int((start - offset) / sampling)
        stop = int((end - offset) / sampling)

        array = self.array[..., start:stop].sum(-1)
        return Measurement(array, calibrations=self.calibrations[:-1])


class DiffractionPatterns(Measurement):

    def __init__(self, array, spatial_sampling, angular_sampling):
        pass


def probe_profile(probe_measurement: Measurement, angle: float = 0.) -> Measurement:
    calibrations = probe_measurement.calibrations
    extent = (calibrations[-2].sampling * probe_measurement.array.shape[-2],
              calibrations[-1].sampling * probe_measurement.array.shape[-1])

    point0 = np.array((extent[0] / 2, extent[1] / 2))
    point1 = point0 + np.array([np.cos(np.pi * angle / 180), np.sin(np.pi * angle / 180)])
    point0, point1 = _line_intersect_rectangle(point0, point1, (0., 0.), extent)
    line_profile = interpolate_line(probe_measurement, point0, point1)
    return line_profile


def block_zeroth_order_spot(diffraction_pattern: Measurement):
    shape = diffraction_pattern.shape
    diffraction_pattern._array[..., shape[-2] // 2, shape[-1] // 2] = diffraction_pattern.array.min()
    return diffraction_pattern


def _line_intersect_rectangle(point0, point1, lower_corner, upper_corner):
    if point0[0] == point1[0]:
        return (point0[0], lower_corner[1]), (point0[0], upper_corner[1])

    m = (point1[1] - point0[1]) / (point1[0] - point0[0])

    def y(x):
        return m * (x - point0[0]) + point0[1]

    def x(y):
        return (y - point0[1]) / m + point0[0]

    if y(0) < lower_corner[1]:
        intersect0 = (x(lower_corner[1]), y(x(lower_corner[1])))
    else:
        intersect0 = (0, y(lower_corner[0]))

    if y(upper_corner[0]) > upper_corner[1]:
        intersect1 = (x(upper_corner[1]), y(x(upper_corner[1])))
    else:
        intersect1 = (upper_corner[0], y(upper_corner[0]))

    return intersect0, intersect1


def interpolate_line(measurement: Measurement, start, end, gpts=None, sampling=None):
    from abtem.scan import LineScan

    array = np.squeeze(measurement.array)

    if not (len(array.shape) == 2):
        raise RuntimeError()

    if (measurement.calibrations[-1] is None) or (measurement.calibrations[-2] is None):
        raise RuntimeError()

    if measurement.calibrations[-2].units != measurement.calibrations[-1].units:
        raise RuntimeError()

    if (gpts is None) & (sampling is None):
        sampling = (measurement.calibrations[-2].sampling + measurement.calibrations[-1].sampling) / 2.

    x = np.linspace(measurement.calibrations[-2].offset,
                    measurement.shape[-2] * measurement.calibrations[-2].sampling,
                    measurement.shape[-2])
    y = np.linspace(measurement.calibrations[1].offset,
                    measurement.shape[-1] * measurement.calibrations[-1].sampling,
                    measurement.shape[-1])

    line_scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling)

    interpolated_array = interpn((x, y), array, line_scan.get_positions())

    return Measurement(interpolated_array,
                       Calibration(offset=0,
                                   sampling=line_scan.sampling[0],
                                   units=measurement.calibrations[-2].units,
                                   name=measurement.calibrations[-2].name))


def calculate_fwhm(measurement: Measurement):
    """Function for calculating the full width at half maximum value for a 1D function."""

    array = measurement.array
    peak_idx = np.argmax(array)
    peak_value = array[peak_idx]
    left = np.argmin(np.abs(array[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(array[peak_idx:] - peak_value / 2))

    fwhm = right - left
    if measurement.calibrations[0] is not None:
        fwhm = fwhm * measurement.calibrations[0].sampling

    return fwhm


def center_of_mass(measurement: Measurement):
    """Function for estimating the intensity center-of-mass for a given measurement."""

    if (measurement.dimensions != 3) and (measurement.dimensions != 4):
        raise RuntimeError()

    if not (measurement.calibrations[-1].units == measurement.calibrations[-2].units):
        raise RuntimeError()

    shape = measurement.array.shape[-2:]
    center = np.array(shape) / 2 - np.array([.5 * (shape[-2] % 2), .5 * (shape[-1] % 2)])
    com = np.zeros(measurement.array.shape[:-2] + (2,))

    if measurement.dimensions == 3:
        for i in range(measurement.array.shape[0]):
            com[i] = scipy.ndimage.measurements.center_of_mass(measurement.array[i])
        com = com - center[None]
    else:
        for i in range(measurement.array.shape[0]):
            for j in range(measurement.array.shape[1]):
                com[i, j] = scipy.ndimage.measurements.center_of_mass(measurement.array[i, j])
        com = com - center[None, None]

    com[..., 0] = com[..., 0] * measurement.calibrations[-2].sampling
    com[..., 1] = com[..., 1] * measurement.calibrations[-1].sampling

    return (Measurement(com[..., 0], measurement.calibrations[:-2], units='mrad', name='com_x'),
            Measurement(com[..., 1], measurement.calibrations[:-2], units='mrad', name='com_y'))
