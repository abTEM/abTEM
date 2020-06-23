from abc import ABCMeta, abstractmethod

import cupy as cp
import numpy as np

from abtem.bases import Grid, cached_method, Cache, DeviceManager, Event, cache_clear_callback, watched_property, \
    Accelerator
from abtem.measure import Calibration, calibrations_from_grid, fourier_space_offset
from abtem.plot import show_image
from abtem.utils import create_fftw_objects
from abtem.config import DTYPE
from abtem.cpu_kernels import abs2
from copy import copy


class AbstractDetector(metaclass=ABCMeta):

    def __init__(self, save_file=None, device=None):

        if save_file is not None:
            if not save_file.endswith('.hdf5'):
                self._save_file = save_file + '.hdf5'
            else:
                self._save_file = save_file

        else:
            self._save_file = None

        self.device_manager = DeviceManager(device)

    @property
    def save_file(self) -> str:
        return self._save_file

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def calibrations(self) -> tuple:
        pass

    @abstractmethod
    def detect(self, waves):
        pass


def cosine_window(x, cutoff, rolloff, attenuate='high'):
    xp = cp.get_array_module(x)

    rolloff *= cutoff
    if attenuate == 'high':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff - rolloff) / rolloff))
        array[x < cutoff] = 0.
        array = xp.where(x < cutoff + rolloff, array, xp.ones_like(x, dtype=DTYPE))
    elif attenuate == 'low':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff + rolloff) / rolloff))
        array[x > cutoff] = 0.
        array = xp.where(x > cutoff - rolloff, array, xp.ones_like(x, dtype=DTYPE))
    else:
        raise RuntimeError('attenuate must be "high" or "low"')

    return array


class AnnularDetector(AbstractDetector):

    def __init__(self, inner, outer, rolloff=0., save_file=None, device=None):

        super().__init__(save_file=save_file, device=device)

        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff
        self.cache = Cache(1)
        self.changed = Event()
        self.changed.register(cache_clear_callback(self.cache))

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
    def rolloff(self) -> float:
        return self._rolloff

    @rolloff.setter
    @watched_property('changed')
    def rolloff(self, value: float):
        self._rolloff = value

    @property
    def shape(self) -> tuple:
        return ()

    @property
    def calibrations(self):
        return ()

    @cached_method('cache')
    def get_integration_region(self, grid, wavelength):
        xp = self.device_manager.get_array_library()

        kx, ky = grid.spatial_frequencies()

        alpha_x = xp.asarray(kx) * wavelength
        alpha_y = xp.asarray(ky) * wavelength

        alpha = xp.sqrt(alpha_x.reshape((-1, 1)) ** 2 + alpha_y.reshape((1, -1)) ** 2)

        if self.rolloff > 0.:
            array = cosine_window(alpha, self.outer, self.rolloff) * cosine_window(alpha, self.inner, self.rolloff)
        else:
            array = (alpha >= self._inner) & (alpha <= self._outer)
        return array

    def detect(self, waves):
        xp = self.device_manager.get_array_library()

        array = waves.array.copy()
        integration_region = self.get_integration_region(waves.grid, waves.wavelength)

        if self.device_manager.is_cuda:
            intensity = abs2(cp.fft.fft2(array))
        else:
            fftw_forward, _ = create_fftw_objects(array)
            fftw_forward()
            intensity = abs2(array)

        result = xp.sum(intensity * integration_region, axis=(1, 2)) / xp.sum(intensity, axis=(1, 2))
        return result

    def copy(self) -> 'AnnularDetector':
        return self.__class__(self.inner, self.outer, rolloff=self.rolloff, save_file=self.save_file)

    def show(self, grid, wavelength, **kwargs):
        array = np.fft.fftshift(self.get_integration_region(grid, wavelength))
        calibrations = calibrations_from_grid(grid, names=['alpha_x', 'alpha_y'], units='mrad.',
                                              scale_factor=wavelength, fourier_space=True)
        return show_image(array, calibrations, **kwargs)


def polar_labels(shape, inner=1, outer=None, nbins_angular=1, nbins_radial=None):
    if outer is None:
        outer = min(shape) // 2
    if nbins_radial is None:
        nbins_radial = outer - inner
    sx, sy = shape
    X, Y = np.ogrid[0:sx, 0:sy]

    r = np.hypot(X - sx / 2, Y - sy / 2)
    radial_bins = -np.ones(shape, dtype=int)
    valid = (r > inner) & (r < outer)
    radial_bins[valid] = nbins_radial * (r[valid] - inner) / (outer - inner)

    angles = np.arctan2(X - sx // 2, Y - sy // 2) % (2 * np.pi)

    angular_bins = np.floor(nbins_angular * (angles / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_angular - 1).astype(np.int)

    bins = np.zeros(shape, dtype=int)
    bins[valid] = radial_bins[valid] * nbins_angular + angular_bins[valid]
    return bins


def label_to_index_generator(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


class SegmentedDetector(AbstractDetector):

    def __init__(self, inner, outer, nbins_radial, nbins_angular=1, save_file=None, device=None):
        super().__init__(save_file=save_file, device=device)
        self._inner = inner
        self._outer = outer
        self._nbins_radial = nbins_radial
        self._nbins_angular = nbins_angular
        self.cache = Cache(1)
        self.changed = Event()
        self.changed.register(cache_clear_callback(self.cache))

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
    def nbins_radial(self) -> float:
        return self._nbins_radial

    @nbins_radial.setter
    @watched_property('changed')
    def nbins_radial(self, value: float):
        self._nbins_radial = value

    @property
    def nbins_angular(self) -> float:
        return self._nbins_angular

    @nbins_angular.setter
    @watched_property('changed')
    def nbins_angular(self, value: float):
        self._nbins_angular = value

    @property
    def shape(self) -> tuple:
        return (self.nbins_radial, self.nbins_angular)

    @property
    def calibrations(self):
        radial_sampling = (self.outer - self.inner) / self.nbins_radial * 1000
        angular_sampling = 2 * np.pi / self.nbins_angular
        return (Calibration(offset=self.inner * 1000, sampling=radial_sampling, units='mrad'),
                Calibration(offset=0, sampling=angular_sampling, units='rad'))

    @cached_method('cache')
    def get_integration_region(self, grid, wavelength):
        xp = self.device_manager.get_array_library()
        kx, ky = grid.spatial_frequencies()

        alpha_x = xp.asarray(kx) * wavelength
        alpha_y = xp.asarray(ky) * wavelength

        alpha = xp.sqrt(alpha_x.reshape((-1, 1)) ** 2 + alpha_y.reshape((1, -1)) ** 2)

        radial_bins = -xp.ones(grid.gpts, dtype=int)
        valid = (alpha > self.inner) & (alpha < self.outer)
        radial_bins[valid] = self.nbins_radial * (alpha[valid] - self.inner) / (self.outer - self.inner)

        angles = xp.arctan2(alpha_x[:, None], alpha_y[None]) % (2 * np.pi)

        angular_bins = xp.floor(self.nbins_angular * (angles / (2 * np.pi)))
        angular_bins = xp.clip(angular_bins, 0, self.nbins_angular - 1).astype(np.int)

        bins = -xp.ones(grid.gpts, dtype=int)
        bins[valid] = angular_bins[valid] + radial_bins[valid] * self.nbins_angular

        return bins

    def detect(self, waves):
        labels = self.get_integration_region(waves.grid, waves.wavelength)
        xp = self.device_manager.get_array_library()
        array = waves.array.copy()

        if self.device_manager.is_cuda:
            intensity = abs2(cp.fft.fft2(array))
        else:
            fftw_forward, _ = create_fftw_objects(array)
            fftw_forward()
            intensity = abs2(array)

        intensity = intensity.reshape((len(intensity), -1))
        result = xp.zeros((len(intensity),) + self.shape, dtype=np.float32)

        total_intensity = xp.sum(intensity, axis=1)
        for i, indices in enumerate(label_to_index_generator(labels)):
            j = i % self.nbins_angular
            i = i // self.nbins_angular
            result[:, i, j] = xp.sum(intensity[:, indices], axis=1) / total_intensity

        return result

    def show(self, grid, wavelength, **kwargs):
        array = np.fft.fftshift(self.get_integration_region(grid, wavelength))
        calibrations = calibrations_from_grid(grid, names=['alpha_x', 'alpha_y'], units='mrad.',
                                              scale_factor=wavelength, fourier_space=True)
        return show_image(array, calibrations, discrete=True, **kwargs)


class PixelatedDetector(AbstractDetector):

    def __init__(self, max_semiangle=None, save_file=None, device=None):
        super().__init__(save_file=save_file, device=device)
        self._max_semiangle = max_semiangle
        self._shape = None
        self._calibrations = None
        self.device_manager = DeviceManager(device)

    def adapt_to_waves(self, waves):

        waves.grid.check_is_defined()
        waves.accelerator.check_is_defined()

        if self.max_semiangle is None:
            self._shape = tuple(waves.grid.gpts)
        else:
            angular_extent = waves.grid.gpts / waves.grid.extent * waves.accelerator.wavelength / 2
            self._shape = tuple(np.floor(self.max_semiangle / angular_extent * waves.grid.gpts / 2.).astype(int) * 2)

        samplings = 1 / waves.grid.extent * waves.accelerator.wavelength * 1000
        offsets = fourier_space_offset(waves.grid.extent / np.array(self.shape),
                                       self.shape) * waves.accelerator.wavelength * 1000
        self._calibrations = (Calibration(offset=offsets[0], sampling=samplings[0], units='mrad.', name='alpha_x'),
                              Calibration(offset=offsets[1], sampling=samplings[1], units='mrad.', name='alpha_y'))

    @property
    def calibrations(self) -> tuple:
        return self._calibrations

    @property
    def max_semiangle(self):
        return self._max_semiangle

    @property
    def shape(self):
        return self._shape

    def detect(self, waves):
        xp = self.device_manager.get_array_library()
        self.adapt_to_waves(waves)

        array = waves.array.copy()

        if self.device_manager.is_cuda:
            intensity = abs2(cp.fft.fft2(array))
        else:
            fftw_forward, _ = create_fftw_objects(array)
            fftw_forward()
            intensity = abs2(array)

        intensity = xp.fft.fftshift(intensity, axes=(-1, -2))
        crop = ((intensity.shape[1] - self.shape[0]) // 2, (intensity.shape[2] - self.shape[1]) // 2)
        intensity = intensity[:, crop[0]:crop[0] + self.shape[0], crop[1]:crop[1] + self.shape[1]]
        return intensity


class WavefunctionDetector:

    def __init__(self, save_file=None, device=None):
        super().__init__(save_file=save_file, device=device)
        self._shape = None
        self._calibrations = None
        self.device_manager = DeviceManager(device)

    def adapt_to_waves(self, waves):
        waves.grid.check_is_defined()

        self._shape = waves.gpts

        samplings = 1 / waves.grid.extent * waves.accelerator.wavelength * 1000
        offsets = fourier_space_offset(waves.grid.extent / np.array(self.shape),
                                       self.shape) * waves.accelerator.wavelength * 1000
        self._calibrations = (Calibration(offset=offsets[0], sampling=samplings[0], units='mrad.', name='alpha_x'),
                              Calibration(offset=offsets[1], sampling=samplings[1], units='mrad.', name='alpha_y'))

    @property
    def calibrations(self) -> tuple:
        return self._calibrations

    @property
    def shape(self):
        return self._shape

    def detect(self, waves):
        return waves.array
