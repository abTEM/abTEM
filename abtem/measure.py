"""Module to describe the detection of scattered electron waves."""
from collections.abc import Iterable, Callable
from copy import copy
from typing import Sequence, Tuple, Union

import h5py
import imageio
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy.interpolate import interpn

from abtem.base_classes import Grid
from abtem.device import asnumpy
from abtem.utils import periodic_crop, energy2wavelength
from abtem.visualize.mpl import show_measurement_2d, show_measurement_1d
from ase import units


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

    def __init__(self, offset: float, sampling: float, units: str, name: str = '', endpoint=True):
        self.offset = offset
        self.sampling = sampling
        self.units = units
        self.name = name
        self.endpoint = endpoint

    def __eq__(self, other):
        return ((self.offset == other.offset) &
                (self.sampling == other.sampling) &
                (self.units == other.units) &
                (self.name == other.name))

    def coordinates(self, n):
        return np.linspace(self.offset, n * self.sampling + self.offset, n, endpoint=False)

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


def grid_from_calibration(calibration, extent=None, gpts=None):
    if (extent is None) and (gpts is None):
        raise RuntimeError

    sampling = calibration.sampling
    return Grid(extent=extent, gpts=gpts, sampling=sampling)


class Measurement:  # (metaclass=ABCMeta):
    """
    Measurement object.

    The measurement object is used for representing the output of a TEM simulation. For example a line profile, an image
    or a collection of diffraction patterns.

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

    def __init__(self,
                 array: np.ndarray,
                 calibrations: Union[Calibration, Sequence[Calibration]] = None,
                 units: str = '',
                 name: str = ''):

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
    def array(self) -> np.ndarray:
        """
        Array of measurements.
        """
        return self._array

    @property
    def dimension(self):
        return len(self.shape)

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

    def _reduction(self, reduction_function: Callable, axis: Union[int, Sequence[int]]):
        if not isinstance(axis, Iterable):
            axis = (axis,)

        array = reduction_function(self.array, axis=axis)

        axis = [d % len(self.calibrations) for d in axis]
        calibrations = [self.calibrations[i] for i in range(len(self.calibrations)) if i not in axis]

        return self.__class__(array, calibrations)

    def sum(self, axis):
        """
        Sum of measurement elements over a given axis.

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

    def mean(self, axis) -> 'Measurement':
        """
        Mean of measurement elements over a given axis.

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

    def diffractograms(self, axes: Tuple[int] = None):
        """
        Calculate the diffractograms of this measurement.

        Parameters
        ----------
        axes : list of int
            The axes to Fourier transform.

        Returns
        -------
        Measurement
        """

        if axes is None:
            if self.dimensions >= 2:
                axes = (-1, -2)
            else:
                axes = (-1,)

        array = np.fft.fftn(self.array, axes=axes)

        sampling = []
        gpts = []
        for i in axes:
            sampling += [self.calibrations[i].sampling]
            gpts += [self.array.shape[i]]

        calibrations = calibrations_from_grid(gpts=gpts, sampling=sampling)

        array = np.fft.fftshift(np.abs(array) ** 2, axes=axes)

        return self.__class__(array=array, calibrations=calibrations)

    def interpolate(self,
                    new_sampling: Union[float, Sequence[float]],
                    padding: str = 'wrap',
                    kind: str = 'quintic') -> 'Measurement':
        """
        Interpolate a 2d measurement.

        Parameters
        ----------
        new_sampling : float
            Target measurement sampling. Same units as measurement calibrations.
        padding : str
            The padding mode as used by numpy.pad.
        kind : str
            The kind of spline interpolation to use. Default is 'quintic'.

        Returns
        -------
        Measurement object
            Interpolated measurement
        """
        if self.dimensions != 2:
            raise RuntimeError('interpolate only implemented for 2d measurements')

        if not (self.calibrations[-1].units == self.calibrations[-2].units):
            raise RuntimeError('the units of the interpolation dimensions must match')

        endpoint = tuple([calibration.endpoint for calibration in self.calibrations])
        sampling = tuple([calibration.sampling for calibration in self.calibrations])
        offset = tuple([calibration.offset for calibration in self.calibrations])

        extent = (sampling[0] * (self.array.shape[0] - endpoint[0]),
                  sampling[1] * (self.array.shape[1] - endpoint[1]))

        new_grid = Grid(extent=extent, sampling=new_sampling, endpoint=endpoint)
        array = np.pad(self.array, ((5,) * 2,) * 2, mode=padding)

        x = self.calibrations[0].coordinates(array.shape[0]) - 5 * self.calibrations[0].sampling
        y = self.calibrations[1].coordinates(array.shape[1]) - 5 * self.calibrations[1].sampling

        interpolator = interp2d(x, y, array.T, kind=kind)

        x = np.linspace(offset[0], offset[0] + extent[0], new_grid.gpts[0], endpoint=endpoint[0])
        y = np.linspace(offset[1], offset[1] + extent[1], new_grid.gpts[1], endpoint=endpoint[1])
        new_array = interpolator(x, y).T

        calibrations = []
        for calibration, d in zip(self.calibrations, new_grid.sampling):
            calibrations.append(copy(calibration))
            calibrations[-1].sampling = d

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
            if not datasets['is_none'][i]:
                calibrations.append(Calibration(offset=datasets['offset'][i],
                                                sampling=datasets['sampling'][i],
                                                units=datasets['units'][i].decode('utf-8'),
                                                name=datasets['name'][i].decode('utf-8')))
            else:
                calibrations.append(None)

        return cls(datasets['array'], calibrations)

    def write(self, path, mode='w'):
        """
        Write measurement to a hdf5 file.

        path: str
            The path to write the file.
        """

        with h5py.File(path, mode) as f:
            f.create_dataset('array', data=self.array)

            is_none = []
            offsets = []
            sampling = []
            units = []
            names = []
            for calibration in self.calibrations:
                if calibration is None:
                    offsets += [0.]
                    sampling += [0.]
                    units += ['']
                    names += ['']
                    is_none += [True]
                else:
                    offsets += [calibration.offset]
                    sampling += [calibration.sampling]
                    units += [calibration.units.encode('utf-8')]
                    names += [calibration.name.encode('utf-8')]
                    is_none += [False]

            f.create_dataset('offset', data=offsets)
            f.create_dataset('sampling', data=sampling)
            f.create_dataset('units', (len(units),), 'S10', units)
            f.create_dataset('name', (len(names),), 'S10', names)
            f.create_dataset('is_none', data=is_none)

        return path

    def save_as_image(self, path: str):
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

    def __init__(self, array: np.ndarray,
                 spatial_sampling: Union[float, Sequence[float]],
                 angular_sampling: float,
                 angular_offset: float = 0.):
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
        calibrations = [Calibration(0., d, 'Å', name) for d, name in zip(spatial_sampling, ('x', 'y'))]

        super().__init__(array, )


def probe_profile(probe_measurement: Measurement, angle: float = 0.) -> Measurement:
    calibrations = probe_measurement.calibrations
    extent = (calibrations[-2].sampling * probe_measurement.array.shape[-2],
              calibrations[-1].sampling * probe_measurement.array.shape[-1])

    point0 = np.array((extent[0] / 2, extent[1] / 2))
    point1 = point0 + np.array([np.cos(np.pi * angle / 180), np.sin(np.pi * angle / 180)])
    point0, point1 = _line_intersect_rectangle(point0, point1, (0., 0.), extent)
    line_profile = interpolate_line(probe_measurement, point0, point1)
    return line_profile


def block_zeroth_order_spot(diffraction_pattern: Measurement, angular_radius=1):
    alpha_x = diffraction_pattern.calibrations[-2].coordinates(diffraction_pattern.array.shape[-2])
    alpha_y = diffraction_pattern.calibrations[-1].coordinates(diffraction_pattern.array.shape[-1])

    alpha_x, alpha_y = np.meshgrid(alpha_x, alpha_y, indexing='ij')

    alpha = alpha_x ** 2 + alpha_y ** 2
    block = alpha > angular_radius ** 2

    diffraction_pattern._array *= block
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


def interpolate_line(measurement: Measurement, start: Sequence[float], end: Sequence[float],
                     sampling: Sequence[float] = None):
    from abtem.scan import LineScan

    array = np.squeeze(measurement.array)

    if not (len(array.shape) == 2):
        raise RuntimeError()

    if (measurement.calibrations[-1] is None) or (measurement.calibrations[-2] is None):
        raise RuntimeError()

    if measurement.calibrations[-2].units != measurement.calibrations[-1].units:
        raise RuntimeError()

    if (sampling is None):
        sampling = (measurement.calibrations[-2].sampling + measurement.calibrations[-1].sampling) / 2.

    x = np.linspace(measurement.calibrations[-2].offset,
                    measurement.shape[-2] * measurement.calibrations[-2].sampling,
                    measurement.shape[-2])
    y = np.linspace(measurement.calibrations[1].offset,
                    measurement.shape[-1] * measurement.calibrations[-1].sampling,
                    measurement.shape[-1])

    line_scan = LineScan(start=start, end=end, sampling=sampling)

    interpolated_array = interpn((x, y), array, line_scan.get_positions())

    return Measurement(interpolated_array,
                       Calibration(offset=0,
                                   sampling=line_scan.sampling[0],
                                   units=measurement.calibrations[-2].units,
                                   name=measurement.calibrations[-2].name))


def calculate_fwhm(probe_profile: Measurement):
    """Function for calculating the full width at half maximum value for a 1D function."""

    array = probe_profile.array
    peak_idx = np.argmax(array)
    peak_value = array[peak_idx]
    left = np.argmin(np.abs(array[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(array[peak_idx:] - peak_value / 2))

    fwhm = right - left
    if probe_profile.calibrations[0] is not None:
        fwhm = fwhm * probe_profile.calibrations[0].sampling

    return fwhm


def intgrad2d(gradient, sampling=None):
    """
    Perform Fourier space integration of gradient.

    Parameters
    ----------
    gradient : two np.ndarrays
        The x- and y-components of the gradient.
    sampling : two float
        Lateral sampling of the gradients. Default is 1.0.

    Returns
    -------
    np.ndarray
        Integrated center of mass measurement
    """
    gx, gy = gradient
    (ny, nx) = gx.shape
    ikx = np.fft.fftfreq(nx, d=sampling[0])
    iky = np.fft.fftfreq(ny, d=sampling[1])
    grid_ikx, grid_iky = np.meshgrid(ikx, iky, indexing='ij')
    k = grid_ikx ** 2 + grid_iky ** 2
    k[k == 0] = 1e-12
    That = (np.fft.fft2(gx) * grid_ikx + np.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
    T = np.real(np.fft.ifft2(That))
    return T


def center_of_mass(measurement: Measurement, return_icom=False):
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

    if return_icom:
        if measurement.dimensions != 4:
            raise RuntimeError()

        sampling = (measurement.calibrations[0].sampling, measurement.calibrations[1].sampling)
        icom = intgrad2d((com[..., 0], com[..., 1]), sampling)
        return Measurement(icom, measurement.calibrations[:-2])
    else:
        return (Measurement(com[..., 0], measurement.calibrations[:-2], units='mrad', name='com_x'),
                Measurement(com[..., 1], measurement.calibrations[:-2], units='mrad', name='com_y'))


def rotational_average(measurement: Measurement):
    array = np.squeeze(measurement.array)

    n = min(array.shape[-2:])

    r = np.arange(0, n, 1)[:, None]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)[None]
    p = np.array([(np.cos(angles) * r).ravel(), (np.sin(angles) * r).ravel()])
    p += np.array([array.shape[-2] // 2, array.shape[-1] // 2])[:, None]

    unrolled = ndimage.map_coordinates(array, p, order=1)
    unrolled = unrolled.reshape((r.shape[0], angles.shape[1]))
    unrolled = unrolled.mean(1)
    return Measurement(unrolled, Calibration(offset=0,
                                             sampling=measurement.calibrations[-2].sampling,
                                             units=measurement.calibrations[-2].units,
                                             name=measurement.calibrations[-2].name))


def integrate_disc(image: Measurement, position: np.ndarray, radius: float, return_mean: bool = True,
                   border: str = 'wrap', interpolate: Union[float, bool] = 0.01) -> float:
    """
    Integrate the values of a 2d measurement on a disc-shaped region.

    Parameters
    ----------
    position : two floats
        Center of disc-shaped integration region
    measurement : 2d measurement
        The measurement to integrate
    radius : float
        Radius of disc-shaped integration region
    return_mean : bool
        If true return the mean, otherwise return the sum.
    border : str
        Specify how to treat integration regions that cross the image border. The valid values and their behaviour is:
        'wrap'
            The measurement is extended by wrapping around to the opposite edge.
        'raise'
            Raise an error if the integration region crosses the measurement border.
    interpolate: float or False
        The image will be interpolated to this sampling. Units of Angstrom.

    Returns
    -------
    float
        Integral value
    """

    if interpolate:
        image = image.interpolate(interpolate)

    calibrations = image.calibrations
    offset = [calibration.offset for calibration in calibrations]
    position = np.array(position) - offset

    new_shape = (int(np.ceil(2 * radius / calibrations[0].sampling)),
                 int(np.ceil(2 * radius / calibrations[1].sampling)))

    corner = (int(np.floor(position[0] / calibrations[0].sampling)) - new_shape[0] // 2,
              int(np.floor(position[1] / calibrations[1].sampling)) - new_shape[1] // 2)

    if border == 'wrap':
        cropped = periodic_crop(image.array, corner, new_shape)
    elif border == 'raise':
        if ((np.any(np.array(corner) < 0)) |
                (corner[0] + new_shape[0] > image.array.shape[0]) |
                (corner[1] + new_shape[1] > image.array.shape[1])):
            raise RuntimeError('The integration region is outside the image.')

        cropped = periodic_crop(image.array, corner, new_shape)
    else:
        raise RuntimeError('border must be one of "wrap" or "raise"')

    x = np.linspace(0., cropped.shape[0] * calibrations[0].sampling, cropped.shape[0],
                    endpoint=calibrations[0].endpoint)
    y = np.linspace(0., cropped.shape[1] * calibrations[1].sampling, cropped.shape[1],
                    endpoint=calibrations[0].endpoint)
    x, y = np.meshgrid(x, y, indexing='ij')

    cropped_position = np.array(position)[:2] - (corner[0] * calibrations[0].sampling,
                                                 corner[1] * calibrations[1].sampling)

    r = np.sqrt((x - cropped_position[0]) ** 2 + (y - cropped_position[1]) ** 2)

    mean_sampling = (calibrations[0].sampling + calibrations[1].sampling) / 2

    mask = 1 - np.clip((r - radius + mean_sampling / 2) / mean_sampling, 0, 1)

    if return_mean:
        return (cropped * mask).sum() / mask.sum()
    else:
        return (cropped * mask).sum()
