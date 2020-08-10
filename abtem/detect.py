from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, List

import h5py
import numpy as np

from abtem.bases import Cache, Event, watched_property, cached_method, Grid
from abtem.device import get_array_module, get_device_function
from abtem.measure import Calibration, calibrations_from_grid, fourier_space_offset, Measurement
from abtem.plot import show_image
from abtem.scan import AbstractScan
from abtem.utils import label_to_index_generator, spatial_frequencies


def crop_to_center(array: np.ndarray):
    shape = array.shape
    w = shape[-2] // 2
    h = shape[-1] // 2
    left = w - w // 2
    right = w + (w - w // 2)
    top = h - h // 2
    bottom = h + (h - h // 2)
    return array[..., left:right, top:bottom]


def calculate_far_field_intensity(waves, overwrite: bool = False):
    xp = get_array_module(waves.array)
    fft2 = get_device_function(xp, 'fft2')
    abs2 = get_device_function(xp, 'abs2')
    array = fft2(waves.array, overwrite_x=overwrite)
    intensity = crop_to_center(xp.fft.fftshift(array, axes=(-2, -1)))
    return abs2(intensity)


def polar_regions(gpts, sampling, wavelength, inner, outer, nbins_radial, nbins_azimuthal):
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
    return np.fft.fftshift(bins)


class AbstractDetector(metaclass=ABCMeta):

    def __init__(self, save_file: str = None):
        if save_file is not None:
            if not save_file.endswith('.hdf5'):
                self._save_file = save_file + '.hdf5'
            else:
                self._save_file = save_file

        else:
            self._save_file = None

    @property
    def save_file(self) -> str:
        return self._save_file

    @abstractmethod
    def detect(self, waves) -> np.ndarray:
        pass

    @abstractmethod
    def allocate_measurement(self, grid, wavelength, scan) -> Measurement:
        pass


class _PolarDetector(AbstractDetector):

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
            raise RuntimeError()

        nbins_radial = int(np.ceil((outer - inner) / self._radial_steps))
        nbins_azimuthal = int(np.ceil(2 * np.pi / self._azimuthal_steps))
        return inner, outer, nbins_radial, nbins_azimuthal

    @cached_method('cache')
    def _get_regions(self, gpts: Tuple[int], sampling: Tuple[float], wavelength: float) -> List[np.ndarray]:
        inner, outer, nbins_radial, nbins_azimuthal = self._get_bins(sampling, wavelength)

        region_labels = polar_regions(gpts, sampling, wavelength, inner / 1000, outer / 1000, nbins_radial,
                                      nbins_azimuthal)
        region_indices = []
        for indices in label_to_index_generator(region_labels):
            region_indices.append(indices)
        return region_indices

    def allocate_measurement(self, grid: Grid, wavelength: float, scan: AbstractScan) -> Measurement:
        inner, outer, nbins_radial, nbins_azimuthal = self._get_bins(grid.antialiased_sampling, wavelength)

        shape = scan.shape
        calibrations = scan.calibrations

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

    def show(self, grid: Grid, wavelength: float, cbar_label: str = 'Detector regions', **kwargs):
        grid.check_is_defined()

        array = np.full(grid.antialiased_gpts, -1, dtype=np.int)
        for i, indices in enumerate(self._get_regions(grid.antialiased_gpts, grid.antialiased_sampling, wavelength)):
            array.ravel()[indices] = i

        calibrations = calibrations_from_grid(grid.antialiased_gpts, grid.antialiased_sampling,
                                              names=['alpha_x', 'alpha_y'], units='mrad.',
                                              scale_factor=wavelength * 1000, fourier_space=True)
        return show_image(array, calibrations, cbar_label=cbar_label, discrete=True, **kwargs)


class AnnularDetector(_PolarDetector):

    def __init__(self, inner: float, outer: float, save_file: str = None):
        super().__init__(inner=inner, outer=outer, radial_steps=outer - inner, save_file=save_file)

    @property
    def inner(self) -> float:
        return self._inner

    @inner.setter
    @watched_property('changed')
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        return self._outer

    @outer.setter
    @watched_property('changed')
    def outer(self, value: float):
        self._outer = value

    def detect(self, waves) -> np.ndarray:
        xp = get_array_module(waves.array)
        intensity = calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)[0]
        total = xp.sum(intensity, axis=(-2, -1))
        return xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total

    def copy(self) -> 'AnnularDetector':
        return self.__class__(self.inner, self.outer, save_file=self.save_file)


class FlexibleAnnularDetector(_PolarDetector):

    def __init__(self, step_size: float = 1., save_file: str = None):
        super().__init__(radial_steps=step_size, save_file=save_file)

    def detect(self, waves) -> np.ndarray:
        xp = get_array_module(waves.array)
        intensity = calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)

        total = xp.sum(intensity, axis=(-2, -1))
        result = np.zeros((len(intensity), len(indices)), dtype=np.float32)
        for i, indices in enumerate(indices):
            result[:, i] = xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total
        return result


class SegmentedDetector(_PolarDetector):

    def __init__(self, inner: float, outer: float, nbins_radial: int, nbins_angular: int, save_file: str = None):
        radial_steps = (outer - inner) / nbins_radial
        azimuthal_steps = 2 * np.pi / nbins_angular
        super().__init__(inner=inner, outer=outer, radial_steps=radial_steps, azimuthal_steps=azimuthal_steps,
                         save_file=save_file)

    @property
    def inner(self) -> float:
        return self._inner

    @inner.setter
    @watched_property('changed')
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        return self._outer

    @outer.setter
    @watched_property('changed')
    def outer(self, value: float):
        self._outer = value

    @property
    def nbins_radial(self) -> int:
        return int((self.outer - self.inner) / self._radial_steps)

    @nbins_radial.setter
    @watched_property('changed')
    def nbins_radial(self, value: int):
        self._radial_steps = (self.outer - self.inner) / value

    @property
    def nbins_angular(self) -> float:
        return int(2 * np.pi / self._azimuthal_steps)

    @nbins_angular.setter
    @watched_property('changed')
    def nbins_angular(self, value: float):
        self._azimuthal_steps = 2 * np.pi / value

    def detect(self, waves) -> np.ndarray:
        xp = get_array_module(waves.array)
        intensity = calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)

        total = xp.sum(intensity, axis=(-2, -1))
        result = np.zeros((len(intensity), len(indices)), dtype=np.float32)
        for i, indices in enumerate(indices):
            result[:, i] = xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total
        return result.reshape((-1, self.nbins_radial, self.nbins_angular))


class PixelatedDetector(AbstractDetector):

    def __init__(self, save_file: str = None):
        super().__init__(save_file=save_file)

    def allocate_measurement(self, grid: Grid, wavelength: float, scan: AbstractScan) -> Measurement:
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
        xp = get_array_module(waves.array)
        abs2 = get_device_function(xp, 'abs2')
        fft2 = get_device_function(xp, 'fft2')

        intensity = abs2(fft2(waves.array, overwrite_x=False))
        intensity = xp.fft.fftshift(intensity, axes=(-1, -2))
        intensity = crop_to_center(intensity)
        return intensity


class WavefunctionDetector(AbstractDetector):

    def __init__(self, save_file: str = None):
        super().__init__(save_file=save_file)

    def allocate_measurement(self, grid: Grid, wavelength: float, scan: AbstractScan) -> Measurement:
        grid.check_is_defined()
        calibrations = calibrations_from_grid(grid.gpts, grid.sampling, names=['x', 'y'], units='Å')

        array = np.zeros(scan.shape + grid.gpts, dtype=np.complex64)
        measurement = Measurement(array, calibrations=scan.calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves) -> np.ndarray:
        return waves.array
