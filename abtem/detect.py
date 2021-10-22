"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Tuple, List, Any, Union, Sequence

import numpy as np

from abtem.base_classes import Cache, Event, watched_property, cached_method
from abtem.device import get_array_module, get_device_function
from abtem.measure import Calibration, calibrations_from_grid, Measurement
from abtem.scan import AbstractScan
from abtem.utils import spatial_frequencies
from abtem.visualize.mpl import show_measurement_2d


def _polar_regions(gpts: Tuple[int, int], angular_sampling: Tuple[float, float], inner: float, outer: float,
                   nbins_radial: int, nbins_azimuthal: int, rotation=0.):
    """
    Create an array of labels for the regions of a given detector geometry.

    Parameters
    ----------
    gpts : two int
        Number of grid points describing the detector regions.
    angular_sampling : two float
        Angular sampling of the discretized dete ctor regions in radians.
    inner : float
        Inner boundary of the detector regions [rad].
    outer : float
        Outer boundary of the detector regions [rad].
    nbins_radial : int
        Number of radial detector bins.
    nbins_azimuthal
        Number of azimuthal detector bins.

    Returns
    -------
    2d array
        Array of integer labels representing the detector regions.
    """
    """Create the polar segmentation of a detector."""

    sampling = (1 / angular_sampling[0] / gpts[0], 1 / angular_sampling[1] / gpts[1])
    kx, ky = spatial_frequencies(gpts, sampling)

    alpha_x = np.asarray(kx)
    alpha_y = np.asarray(ky)

    alpha = np.sqrt(alpha_x.reshape((-1, 1)) ** 2 + alpha_y.reshape((1, -1)) ** 2)

    radial_bins = -np.ones(gpts, dtype=int)
    valid = (alpha >= inner) & (alpha <= outer)

    radial_bins[valid] = nbins_radial * (alpha[valid] - inner) / (outer - inner)

    angles = (np.arctan2(alpha_x[:, None], alpha_y[None]) + rotation) % (2 * np.pi)

    angular_bins = np.floor(nbins_azimuthal * (angles / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_azimuthal - 1).astype(np.int)

    bins = -np.ones(gpts, dtype=int)
    bins[valid] = angular_bins[valid] + radial_bins[valid] * nbins_azimuthal
    return bins


def check_max_angle_exceeded(waves, max_angle):
    if (max_angle is not None) and (not isinstance(max_angle, str)):
        if max_angle > min(waves.cutoff_scattering_angles):
            raise RuntimeError('Detector max angle exceeds the cutoff scattering angle.')


class AbstractDetector(metaclass=ABCMeta):
    """Abstract base class for all detectors."""

    def __init__(self, max_detected_angle=None, save_file: str = None):
        if save_file is not None:
            save_file = str(save_file)

            if not save_file.endswith('.hdf5'):
                self._save_file = save_file + '.hdf5'
            else:
                self._save_file = save_file

        else:
            self._save_file = None

        self._max_detected_angle = max_detected_angle

    @property
    def save_file(self) -> str:
        """The path to the file for saving the detector output."""
        return self._save_file

    @abstractmethod
    def detect(self, waves) -> Any:
        pass

    @abstractmethod
    def allocate_measurement(self, waves, scan) -> Measurement:
        pass


class _PolarDetector(AbstractDetector):
    """Class to define a polar detector, forming the basis of annular and segmented detectors."""

    def __init__(self,
                 inner: float = None,
                 outer: float = None,
                 radial_steps: float = 1.,
                 azimuthal_steps: float = None,
                 offset: Tuple[float, float] = None,
                 rotation: float = 0.,
                 save_file: str = None):

        self._inner = inner
        self._outer = outer

        self._radial_steps = radial_steps

        if azimuthal_steps is None:
            azimuthal_steps = 2 * np.pi

        self._azimuthal_steps = azimuthal_steps

        self._rotation = rotation
        self._offset = offset

        self.cache = Cache(1)
        self.changed = Event()
        super().__init__(max_detected_angle=outer, save_file=save_file)

    @classmethod
    def _label_to_index(cls, labels):
        xp = get_array_module(labels)
        labels = labels.flatten()
        labels_order = labels.argsort()
        sorted_labels = labels[labels_order]
        indices = xp.arange(0, len(labels) + 1)[labels_order]
        index = xp.arange(0, np.max(labels) + 1)
        lo = xp.searchsorted(sorted_labels, index, side='left')
        hi = xp.searchsorted(sorted_labels, index, side='right')
        for i, (l, h) in enumerate(zip(lo, hi)):
            yield indices[l:h]

    def _get_bins(self, cutoff_scattering_angle=None):
        if self._inner is None:
            inner = 0.
        else:
            inner = self._inner

        if self._outer is None:
            if cutoff_scattering_angle is None:
                raise RuntimeError('The outer integration angle is not set.')

            outer = cutoff_scattering_angle
            outer = np.floor(outer / self._radial_steps) * self._radial_steps
        else:
            outer = self._outer

        nbins_radial = int(np.ceil((outer - inner) / self._radial_steps))

        nbins_azimuthal = int(np.ceil(2 * np.pi / self._azimuthal_steps))

        return nbins_radial, nbins_azimuthal, inner, outer

    @cached_method('cache')
    def _get_regions(self,
                     gpts: Tuple[int, int],
                     angular_sampling: Tuple[float, float],
                     cutoff_scattering_angle: float = None,
                     xp=np) -> List[np.ndarray]:

        nbins_radial, nbins_azimuthal, inner, outer = self._get_bins(cutoff_scattering_angle)

        region_labels = _polar_regions(gpts,
                                       (angular_sampling[0] / 1e3, angular_sampling[1] / 1e3),
                                       inner / 1e3,
                                       outer / 1e3,
                                       nbins_radial,
                                       nbins_azimuthal,
                                       rotation=self._rotation)

        if self._offset is not None:
            offset = (int(round(self._offset[0] / angular_sampling[0])),
                      int(round(self._offset[1] / angular_sampling[1])))

            if (abs(offset[0]) > region_labels.shape[0]) or (abs(offset[1]) > region_labels.shape[1]):
                raise RuntimeError('Detector offset exceeds maximum detected angle.')

            region_labels = np.roll(region_labels, offset, (0, 1))

        region_labels = xp.asarray(region_labels)

        if np.all(region_labels == -1):
            raise RuntimeError('Zero-sized detector region.')

        region_indices = []
        for indices in self._label_to_index(region_labels):
            region_indices.append(indices)

        return region_indices

    def allocate_measurement(self, waves, scan: AbstractScan = None) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves object
            An example of the
        scan : Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        waves.accelerator.check_is_defined()

        if scan is None:
            shape = ()
            calibrations = ()
        else:
            shape = scan.shape
            calibrations = scan.calibrations

        nbins_radial, nbins_azimuthal, inner, _ = self._get_bins(min(waves.cutoff_scattering_angles))

        if nbins_radial > 1:
            shape += (nbins_radial,)
            calibrations += (Calibration(offset=inner, sampling=self._radial_steps, units='mrad'),)

        if nbins_azimuthal > 1:
            shape += (nbins_azimuthal,)
            calibrations += (Calibration(offset=0, sampling=self._azimuthal_steps, units='rad'),)

        array = np.zeros(shape, dtype=np.float32)

        measurement = Measurement(array, calibrations=calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def show(self, waves, **kwargs):
        """
        Visualize the detector region(s) of the detector as applied to a specified wave function.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function the visualization will be created to match
        kwargs :
            Additional keyword arguments for abtem.visualize.mpl.show_measurement_2d.
        """

        waves.grid.check_is_defined()
        array = np.full(waves.gpts, -1, dtype=np.int)

        for i, indices in enumerate(self._get_regions(waves.gpts,
                                                      waves.angular_sampling,
                                                      min(waves.cutoff_scattering_angles))):
            array.ravel()[indices] = i

        calibrations = calibrations_from_grid(waves.gpts,
                                              waves.sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=waves.wavelength * 1e3,
                                              fourier_space=True)

        array = np.fft.fftshift(array, axes=(-1, -2))

        measurement = Measurement(array, calibrations=calibrations, name='Detector regions')

        return show_measurement_2d(measurement, discrete_cmap=True, **kwargs)


class AnnularDetector(_PolarDetector):
    """
    Annular detector object.

    The annular detector integrates the intensity of the detected wave functions between an inner and outer integration
    limit.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float
        Outer integration limit [mrad].
    offset: two float, optional
        Center offset of integration region [mrad].
    save_file: str, optional
        The path to the file for saving the detector output.
    """

    def __init__(self, inner: float, outer: float, offset: Tuple[float, float] = None, save_file: str = None):
        super().__init__(inner=inner, outer=outer, offset=offset, radial_steps=outer - inner, save_file=save_file)

    @property
    def inner(self) -> float:
        """Inner integration limit [mrad]."""
        return self._inner

    @inner.setter
    @watched_property('changed')
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        """Outer integration limit [mrad]."""
        return self._outer

    @outer.setter
    @watched_property('changed')
    def outer(self, value: float):
        self._max_detected_angle = value
        self._outer = value

    def _integrate_array(self, array: np.ndarray, angular_sampling: Tuple[float, float],
                         cutoff_scattering_angle: float = None):

        xp = get_array_module(array)
        indices = self._get_regions(array.shape[-2:], angular_sampling, cutoff_scattering_angle)[0]
        indexed = array.reshape(array.shape[:-2] + (-1,))[..., indices]
        values = xp.sum(indexed, axis=-1)
        return values

    def integrate(self, diffraction_patterns: Measurement) -> Measurement:
        """
        Integrate diffraction pattern measurements on the detector region.

        Parameters
        ----------
        diffraction_patterns : 2d, 3d or 4d Measurement object
            The collection diffraction patterns to be integrated.

        Returns
        -------
        Measurement
        """

        if diffraction_patterns.dimensions < 2:
            raise ValueError()

        if not (diffraction_patterns.calibrations[-1].units == diffraction_patterns.calibrations[-2].units):
            raise ValueError()

        sampling = (diffraction_patterns.calibrations[-2].sampling, diffraction_patterns.calibrations[-1].sampling)

        calibrations = diffraction_patterns.calibrations[:-2]
        array = np.fft.ifftshift(diffraction_patterns.array, axes=(-2, -1))

        cutoff_scattering_angle = min(diffraction_patterns.calibrations[-2].sampling *
                                      (diffraction_patterns.array.shape[-2] // 2),
                                      diffraction_patterns.calibrations[-1].sampling *
                                      (diffraction_patterns.array.shape[-1] // 2), )

        if cutoff_scattering_angle < self.outer:
            raise RuntimeError('Outer integration limit exceeds the maximum measurement scattering angle '
                               f'({cutoff_scattering_angle} mrad)')

        return Measurement(self._integrate_array(array, sampling, cutoff_scattering_angle), calibrations=calibrations)

    def detect(self, waves) -> np.ndarray:
        """
        Integrate the intensity of a the wave functions over the detector range.

        Parameters
        ----------
        waves : Waves object
            The batch of wave functions to detect.

        Returns
        -------
        1d array
            Detected values as a 1D array. The array has the same length as the batch size of the wave functions.
        """

        xp = get_array_module(waves.array)
        fft2 = get_device_function(xp, 'fft2')
        abs2 = get_device_function(xp, 'abs2')

        intensity = abs2(fft2(waves.array, overwrite_x=False))
        return self._integrate_array(intensity, waves.angular_sampling, min(waves.cutoff_scattering_angles))

    def __copy__(self) -> 'AnnularDetector':
        return self.__class__(self.inner, self.outer, save_file=self.save_file)

    def copy(self) -> 'AnnularDetector':
        """Make a copy."""
        return copy(self)


class FlexibleAnnularDetector(_PolarDetector):
    """
    Flexible annular detector object.

    The FlexibleAnnularDetector object allows choosing the integration limits after running the simulation by radially
    binning the intensity.

    Parameters
    ----------
    step_size: float
        The radial separation between integration regions [mrad].
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self, step_size: float = 1., save_file: str = None):
        super().__init__(radial_steps=step_size, save_file=save_file)

    @property
    def step_size(self) -> float:
        """
        Step size [mrad].
        """
        return self._radial_steps

    @step_size.setter
    @watched_property('changed')
    def step_size(self, value: float):
        self._radial_steps = value

    def detect(self, waves) -> np.ndarray:
        """
        Integrate the intensity of a the wave functions over the detector range.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
        2d array
            Detected values. The array has shape of (batch size, number of bins).
        """

        xp = get_array_module(waves.array)
        fft2 = get_device_function(xp, 'fft2')
        abs2 = get_device_function(xp, 'abs2')
        sum_run_length_encoded = get_device_function(xp, 'sum_run_length_encoded')

        intensity = abs2(fft2(waves.array, overwrite_x=False))
        indices = self._get_regions(waves.gpts, waves.angular_sampling, min(waves.cutoff_scattering_angles), xp)

        separators = xp.concatenate((xp.array([0]), xp.cumsum(xp.array([len(ring) for ring in indices]))))
        intensity = intensity.reshape((intensity.shape[0], -1))[:, xp.concatenate(indices)]
        result = xp.zeros((len(intensity), len(indices)), dtype=xp.float32)

        sum_run_length_encoded(intensity, result, separators)

        return result

    def __copy__(self) -> 'FlexibleAnnularDetector':
        return self.__class__(self.step_size, save_file=self.save_file)

    def copy(self) -> 'FlexibleAnnularDetector':
        """
        Make a copy.
        """
        return copy(self)


class SegmentedDetector(_PolarDetector):
    """
    Segmented detector object.

    The segmented detector covers an annular angular range, and is partitioned into several integration regions divided
    to radial and angular segments. This can be used for simulating differential phase contrast (DPC) imaging.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float
        Outer integration limit [mrad].
    nbins_radial: int
        Number of radial bins.
    nbins_angular: int
        Number of angular bins.
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self, inner: float, outer: float, nbins_radial: int, nbins_angular: int, rotation: float = 0.,
                 save_file: str = None):
        radial_steps = (outer - inner) / nbins_radial
        azimuthal_steps = 2 * np.pi / nbins_angular
        super().__init__(inner=inner, outer=outer, radial_steps=radial_steps, azimuthal_steps=azimuthal_steps,
                         rotation=rotation, save_file=save_file)

    @property
    def inner(self) -> float:
        """Inner integration limit [mrad]."""
        return self._inner

    @inner.setter
    @watched_property('changed')
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        """Outer integration limit [mrad]."""
        return self._outer

    @outer.setter
    @watched_property('changed')
    def outer(self, value: float):
        self._outer = value

    @property
    def nbins_radial(self) -> int:
        """Number of radial bins."""
        return int((self.outer - self.inner) / self._radial_steps)

    @nbins_radial.setter
    @watched_property('changed')
    def nbins_radial(self, value: int):
        self._radial_steps = (self.outer - self.inner) / value

    @property
    def nbins_angular(self) -> int:
        """Number of angular bins."""
        return int(2 * np.pi / self._azimuthal_steps)

    @nbins_angular.setter
    @watched_property('changed')
    def nbins_angular(self, value: float):
        self._azimuthal_steps = 2 * np.pi / value

    def detect(self, waves) -> np.ndarray:
        """
        Integrate the intensity of a the wave functions over the detector range.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
        3d array
            Detected values. The first dimension indexes the batch size, the second and third indexes the radial and
            angular bins, respectively.
        """

        xp = get_array_module(waves.array)
        fft2 = get_device_function(xp, 'fft2')
        abs2 = get_device_function(xp, 'abs2')
        sum_run_length_encoded = get_device_function(xp, 'sum_run_length_encoded')
        intensity = abs2(fft2(waves.array, overwrite_x=False))

        indices = self._get_regions(waves.gpts, waves.angular_sampling, min(waves.cutoff_scattering_angles), xp)

        separators = xp.concatenate((xp.array([0]), xp.cumsum(xp.array([len(ring) for ring in indices]))))
        intensity = intensity.reshape((intensity.shape[0], -1))[:, xp.concatenate(indices)]
        result = xp.zeros((len(intensity), len(separators) - 1), dtype=xp.float32)
        sum_run_length_encoded(intensity, result, separators)

        shape = (-1,)
        if self.nbins_radial > 1:
            shape += (self.nbins_radial,)

        if self.nbins_angular > 1:
            shape += (self.nbins_angular,)

        return result.reshape(shape)

    def __copy__(self) -> 'SegmentedDetector':
        return self.__class__(inner=self.inner, outer=self.outer, nbins_radial=self.nbins_radial,
                              nbins_angular=self.nbins_angular, save_file=self.save_file)

    def copy(self) -> 'SegmentedDetector':
        """Make a copy."""
        return copy(self)


class PixelatedDetector(AbstractDetector):
    """
    Pixelated detector object.

    The pixelated detector records the intensity of the Fourier-transformed exit wavefunction. This may be used for
    example for simulating 4D-STEM.

    Parameters
    ----------
    max_angle : str or float or None
        The diffraction patterns will be detected up to this angle. If set to a string it must be 'limit' or 'valid'
    resample : 'uniform' or False
        If 'uniform', the diffraction patterns from rectangular cells will be downsampled to a uniform angular sampling.
    mode : 'intensity' or 'complex'
    save_file : str
        The path to the file used for saving the detector output.
    """

    def __init__(self,
                 max_angle: Union[str, float] = 'valid',
                 resample: Union[str, float] = False,
                 mode='intensity',
                 save_file: str = None):

        self._max_angle = max_angle
        self._resample = resample
        self._mode = mode

        super().__init__(save_file=save_file)

    @property
    def max_angle(self):
        return self._max_angle

    @property
    def resample(self):
        return self._resample

    def _bilinear_nodes_and_weight(self, old_shape, new_shape, old_angular_sampling, new_angular_sampling, xp):
        nodes = []
        weights = []

        old_sampling = (1 / old_angular_sampling[0] / old_shape[0],
                        1 / old_angular_sampling[1] / old_shape[1])

        new_sampling = (1 / new_angular_sampling[0] / new_shape[0],
                        1 / new_angular_sampling[1] / new_shape[1])

        for n, m, r, d in zip(old_shape, new_shape, old_sampling, new_sampling):
            k = xp.fft.fftshift(xp.fft.fftfreq(n, r).astype(xp.float32))
            k_new = xp.fft.fftshift(xp.fft.fftfreq(m, d).astype(xp.float32))

            distances = k_new[None] - k[:, None]
            distances[distances < 0.] = np.inf

            w = distances.min(0) / (k[1] - k[0])
            w[w == np.inf] = 0.

            nodes.append(distances.argmin(0))
            weights.append(w)

        v, u = nodes
        vw, uw = weights
        v, u, vw, uw = xp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])
        return v, u, vw, uw

    def _resampled_gpts(self, gpts, angular_sampling):
        if self._resample is False:
            return gpts, angular_sampling

        if self._resample == 'uniform':
            scale_factor = (angular_sampling[0] / max(angular_sampling),
                            angular_sampling[1] / max(angular_sampling))

        else:
            scale_factor = (angular_sampling[0] / self._resample[0],
                            angular_sampling[1] / self._resample[1])

        new_gpts = (int(np.ceil(gpts[0] * scale_factor[0])),
                    int(np.ceil(gpts[1] * scale_factor[1])))

        if np.abs(new_gpts[0] - new_gpts[1]) <= 2:
            new_gpts = (min(new_gpts),) * 2

        new_angular_sampling = (angular_sampling[0] / scale_factor[0],
                                angular_sampling[1] / scale_factor[1])
        return new_gpts, new_angular_sampling

    def _interpolate(self, array, angular_sampling):
        xp = get_array_module(array)
        interpolate_bilinear = get_device_function(xp, 'interpolate_bilinear')

        new_gpts, new_angular_sampling = self._resampled_gpts(array.shape[-2:], angular_sampling)
        v, u, vw, uw = self._bilinear_nodes_and_weight(array.shape[-2:],
                                                       new_gpts,
                                                       angular_sampling,
                                                       new_angular_sampling,
                                                       xp)

        return interpolate_bilinear(array, v, u, vw, uw)

    def allocate_measurement(self, waves, scan: AbstractScan = None) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function that will define the shape of the diffraction patterns.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        waves.accelerator.check_is_defined()
        check_max_angle_exceeded(waves, self.max_angle)

        gpts = waves.downsampled_gpts(self.max_angle)
        gpts, new_angular_sampling = self._resampled_gpts(gpts, angular_sampling=waves.angular_sampling)

        sampling = (1 / new_angular_sampling[0] / gpts[0] * waves.wavelength * 1000,
                    1 / new_angular_sampling[1] / gpts[1] * waves.wavelength * 1000)

        calibrations = calibrations_from_grid(gpts,
                                              sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=waves.wavelength * 1000,
                                              fourier_space=True)

        if scan is None:
            scan_shape = ()
            scan_calibrations = ()
        elif isinstance(scan, tuple):
            scan_shape = scan
            scan_calibrations = (None,) * len(scan)
        else:
            scan_shape = scan.shape
            scan_calibrations = scan.calibrations

        if self._mode == 'intensity':
            array = np.zeros(scan_shape + gpts, dtype=np.float32)
        elif self._mode == 'complex':
            array = np.zeros(scan_shape + gpts, dtype=np.complex64)
        else:
            raise ValueError()

        measurement = Measurement(array, calibrations=scan_calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves) -> np.ndarray:
        """
        Calculate the far field intensity of the wave functions. The output is cropped to include the non-suppressed
        frequencies from the antialiased 2D fourier spectrum.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
            Detected values. The first dimension indexes the batch size, the second and third indexes the two components
            of the spatial frequency.
        """

        xp = get_array_module(waves.array)
        abs2 = get_device_function(xp, 'abs2')

        waves = waves.far_field(max_angle=self.max_angle)

        if self._mode == 'intensity':
            array = abs2(waves.array)
        elif self._mode == 'complex':
            array = waves.array
        else:
            raise ValueError()

        array = xp.fft.fftshift(array, axes=(-2, -1))

        if self._resample:
            array = self._interpolate(array, waves.angular_sampling)
        return array


class WavefunctionDetector(AbstractDetector):
    """
    Wave function detector object

    The wave function detector records the raw exit wave functions.

    Parameters
    ----------
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self, save_file: str = None):
        super().__init__(max_detected_angle=None, save_file=save_file)

    def allocate_measurement(self, waves, scan: AbstractScan) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function that will define the shape of the diffraction patterns.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        calibrations = calibrations_from_grid(waves.gpts, waves.sampling, names=['x', 'y'], units='Å')

        array = np.zeros(scan.shape + waves.gpts, dtype=np.complex64)
        measurement = Measurement(array, calibrations=scan.calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves) -> np.ndarray:
        """
        Detect the complex wave function.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
        3d complex array
            The arrays of the Waves object.
        """
        return waves.array
