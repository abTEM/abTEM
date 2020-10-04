"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Tuple, List, Any

import numpy as np

from abtem.base_classes import Cache, Event, watched_property, cached_method, Grid, AntialiasFilter
from abtem.device import get_array_module, get_device_function
from abtem.measure import Calibration, calibrations_from_grid, Measurement
from abtem.plot import show_image
from abtem.scan import AbstractScan
from abtem.utils import spatial_frequencies


def _crop_to_center(array: np.ndarray):
    """Crop an array around its center to remove the suppressed frequencies from an antialiased 2D fourier spectrum."""
    shape = array.shape
    w = shape[-2] // 2
    h = shape[-1] // 2
    left = w - w // 2
    right = w + (w - w // 2)
    top = h - h // 2
    bottom = h + (h - h // 2)
    return array[..., left:right, top:bottom]


def _calculate_far_field_intensity(waves, overwrite: bool = False):
    """Calculate the far-field intensity of a wave."""
    xp = get_array_module(waves.array)
    fft2 = get_device_function(xp, 'fft2')
    abs2 = get_device_function(xp, 'abs2')
    array = fft2(waves.array, overwrite_x=overwrite)
    intensity = _crop_to_center(xp.fft.fftshift(array, axes=(-2, -1)))
    return abs2(intensity)


def _polar_regions(gpts, sampling, wavelength, inner, outer, nbins_radial, nbins_azimuthal):
    """Create the polar segmentation of a detector."""
    kx, ky = spatial_frequencies(gpts, sampling)

    alpha_x = np.asarray(kx) * wavelength
    alpha_y = np.asarray(ky) * wavelength

    alpha = np.sqrt(alpha_x.reshape((-1, 1)) ** 2 + alpha_y.reshape((1, -1)) ** 2)

    radial_bins = -np.ones(gpts, dtype=int)
    valid = (alpha >= inner) & (alpha < outer)
    radial_bins[valid] = nbins_radial * (alpha[valid] - inner) / (outer - inner)

    angles = np.arctan2(alpha_x[:, None], alpha_y[None]) % (2 * np.pi)

    angular_bins = np.floor(nbins_azimuthal * (angles / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_azimuthal - 1).astype(np.int)

    bins = -np.ones(gpts, dtype=int)
    bins[valid] = angular_bins[valid] + radial_bins[valid] * nbins_azimuthal
    return bins


class AbstractDetector(metaclass=ABCMeta):
    """Abstract base class for all detectors."""

    def __init__(self, save_file: str = None):
        if save_file is not None:
            save_file = str(save_file)

            if not save_file.endswith('.hdf5'):
                self._save_file = save_file + '.hdf5'
            else:
                self._save_file = save_file

        else:
            self._save_file = None

    @property
    def save_file(self) -> str:
        """The path to the file for saving the detector output."""
        return self._save_file

    @abstractmethod
    def detect(self, waves) -> Any:
        pass

    @abstractmethod
    def allocate_measurement(self, grid, wavelength, scan) -> Measurement:
        pass


class _PolarDetector(AbstractDetector):
    """Class to define a polar detector, forming the basis of annular and segmented detectors."""

    def __init__(self, inner: float = None, outer: float = None, radial_steps: float = 1.,
                 azimuthal_steps: float = None, save_file: str = None):

        self._inner = inner
        self._outer = outer

        self._radial_steps = radial_steps

        if azimuthal_steps is None:
            azimuthal_steps = 2 * np.pi

        self._azimuthal_steps = azimuthal_steps

        self.cache = Cache(1)
        self.changed = Event()
        super().__init__(save_file=save_file)

    def _get_bins(self, sampling: Tuple[float], wavelength: float) -> Tuple[float, float, int, int]:
        if self._inner is None:
            inner = 0
        else:
            inner = self._inner

        max_angle = 1 / np.max(sampling) * wavelength / 2 * 1000

        if self._outer is None:
            outer = max_angle
        else:
            outer = self._outer

        if outer > max_angle:
            raise RuntimeError('Maximum detector angle exceeds maximum simulated scattering angle.')

        nbins_radial = int(np.ceil((outer - inner) / self._radial_steps))
        nbins_azimuthal = int(np.ceil(2 * np.pi / self._azimuthal_steps))
        return inner, outer, nbins_radial, nbins_azimuthal

    @classmethod
    def _label_to_index(cls, labels):
        labels = labels.flatten()
        labels_order = labels.argsort()
        sorted_labels = labels[labels_order]
        indices = np.arange(0, len(labels) + 1)[labels_order]
        index = np.arange(0, np.max(labels) + 1)
        lo = np.searchsorted(sorted_labels, index, side='left')
        hi = np.searchsorted(sorted_labels, index, side='right')
        for i, (l, h) in enumerate(zip(lo, hi)):
            yield indices[l:h]

    @cached_method('cache')
    def _get_regions(self, gpts: Tuple[int], sampling: Tuple[float], wavelength: float) -> List[np.ndarray]:
        inner, outer, nbins_radial, nbins_azimuthal = self._get_bins(sampling, wavelength)

        region_labels = _polar_regions(gpts,
                                       sampling,
                                       wavelength,
                                       inner / 1000,
                                       outer / 1000,
                                       nbins_radial,
                                       nbins_azimuthal)

        if np.all(region_labels == -1):
            raise RuntimeError('Zero-sized detector region.')

        region_indices = []
        for indices in self._label_to_index(region_labels):
            region_indices.append(indices)
        return region_indices

    def allocate_measurement(self, grid: Grid, wavelength: float, scan: AbstractScan) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        grid : Grid object
            The grid of the Waves objects that will be detected.
        wavelength : float
            The wavelength of the Waves objects that will be detected.
        scan : Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        inner, outer, nbins_radial, nbins_azimuthal = self._get_bins(grid.antialiased_sampling, wavelength)

        shape = scan.shape
        calibrations = scan.calibrations

        if nbins_radial > 1:
            shape += (nbins_radial,)
            calibrations += (Calibration(offset=inner, sampling=self._radial_steps, units='mrad'),)

        if nbins_azimuthal > 1:
            shape += (nbins_azimuthal,)
            calibrations += (
                Calibration(offset=0, sampling=self._azimuthal_steps, units='rad'),)  # JM verify rad/change to mrad

        array = np.zeros(shape, dtype=np.float32)
        measurement = Measurement(array, calibrations=calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def show(self, grid: Grid, wavelength: float, cbar_label: str = 'Detector regions', **kwargs):
        """
        Visualize the detector region(s) of the detector.

        Parameters
        ----------
        grid : Grid
            The grid of the Waves objects that will be detected.
        wavelength : float
            The wavelength of the Waves objects that will be detected.
        cbar_label : str
            Label for the colorbar. Default is 'Detector regions'.
        kwargs :
            Additional keyword arguments for abtem.plot.show_image.
        """

        grid.check_is_defined()

        array = np.full(grid.antialiased_gpts, -1, dtype=np.int)
        for i, indices in enumerate(self._get_regions(grid.antialiased_gpts, grid.antialiased_sampling, wavelength)):
            array.ravel()[indices] = i

        calibrations = calibrations_from_grid(grid.antialiased_gpts, grid.antialiased_sampling,
                                              names=['alpha_x', 'alpha_y'], units='mrad',
                                              scale_factor=wavelength * 1000, fourier_space=True)
        return show_image(array, calibrations, cbar_label=cbar_label, discrete=True, **kwargs)


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
    save_file: str
        The path to the file for saving the detector output.
    """

    def __init__(self, inner: float, outer: float, save_file: str = None):
        super().__init__(inner=inner, outer=outer, radial_steps=outer - inner, save_file=save_file)

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

    def _integrate(self, array, sampling, wavelength, normalize=True):
        xp = get_array_module(array)

        indices = self._get_regions(array.shape[-2:], sampling, wavelength)[0]

        values = xp.sum(array.reshape(array.shape[:-2] + (-1,))[..., indices], axis=-1)

        if normalize:
            return values / xp.sum(array, axis=(-2, -1))
        else:
            return values

    def detect(self, waves, normalize=True) -> np.ndarray:
        """
        Integrate the intensity of a the wave functions over the detector range.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
        1d array
            Detected values as a 1D array. The array has the same length as the batch size of the wave functions.
        """

        xp = get_array_module(waves.array)
        fft2 = get_device_function(xp, 'fft2')
        abs2 = get_device_function(xp, 'abs2')

        # TODO : crop before integrating.
        intensity = abs2(fft2(waves.array, overwrite_x=True))
        # antialias_filter = AntialiasFilter()
        # antialias_filter.grid.match(waves)

        return self._integrate(intensity, waves.sampling, waves.wavelength, normalize)

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
        intensity = _calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)

        total = xp.sum(intensity, axis=(-2, -1))
        result = np.zeros((len(intensity), len(indices)), dtype=np.float32)
        for i, indices in enumerate(indices):
            result[:, i] = xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total
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

    def __init__(self, inner: float, outer: float, nbins_radial: int, nbins_angular: int, save_file: str = None):
        radial_steps = (outer - inner) / nbins_radial
        azimuthal_steps = 2 * np.pi / nbins_angular
        super().__init__(inner=inner, outer=outer, radial_steps=radial_steps, azimuthal_steps=azimuthal_steps,
                         save_file=save_file)

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
        intensity = _calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)

        total = xp.sum(intensity, axis=(-2, -1))
        result = np.zeros((len(intensity), len(indices)), dtype=np.float32)
        for i, indices in enumerate(indices):
            result[:, i] = xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total
        return result.reshape((-1, self.nbins_radial, self.nbins_angular))

    def __copy__(self) -> 'SegmentedDetector':
        return self.__class__(inner=self.inner, outer=self.outer, nbins_radial=self.nbins_radial,
                              nbins_angular=self.nbins_angular, save_file=self.save_file)

    def copy(self) -> 'SegmentedDetector':
        """Make a copy."""
        return copy(self)


class PixelatedDetector(AbstractDetector):
    """
    Pixelated detector object.

    The pixelated detector records the intensity of the Fourier-transformed exit wave function. This may be used for
    simulating 4D-STEM.

    Parameters
    ----------
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self, save_file: str = None):
        super().__init__(save_file=save_file)

    def allocate_measurement(self, grid: Grid, wavelength: float, scan: AbstractScan) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        grid: Grid object
            The grid of the Waves objects that will be detected.
        wavelength: float
            The wavelength of the Waves objects that will be detected.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        grid.check_is_defined()
        shape = (grid.gpts[0] // 2, grid.gpts[1] // 2)

        calibrations = calibrations_from_grid(grid.antialiased_gpts,
                                              grid.antialiased_sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=wavelength * 1000,
                                              fourier_space=True)

        array = np.zeros(scan.shape + shape)
        measurement = Measurement(array, calibrations=scan.calibrations + calibrations)
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
        fft2 = get_device_function(xp, 'fft2')

        intensity = abs2(fft2(waves.array, overwrite_x=False))
        intensity = xp.fft.fftshift(intensity, axes=(-1, -2))
        intensity = _crop_to_center(intensity)
        return intensity


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
        super().__init__(save_file=save_file)

    def allocate_measurement(self, grid: Grid, wavelength: float, scan: AbstractScan) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        grid: Grid
            The grid of the Waves objects that will be detected.
        wavelength: float
            The wavelength of the Waves objects that will be detected.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        grid.check_is_defined()
        calibrations = calibrations_from_grid(grid.gpts, grid.sampling, names=['x', 'y'], units='Å')

        array = np.zeros(scan.shape + grid.gpts, dtype=np.complex64)
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
