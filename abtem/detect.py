from abc import ABCMeta, abstractmethod

import cupy as cp
import mkl_fft
import numpy as np

from abtem.bases import cached_method, Cache, Event, cache_clear_callback, watched_method
from abtem.cpu_kernels import abs2
from abtem.device import get_array_module, get_device_function
from abtem.measure import Calibration, calibrations_from_grid, fourier_space_offset, Measurement
from abtem.plot import show_image
import h5py


def crop_to_center(array):
    shape = array.shape
    w = shape[-2] // 2
    h = shape[-1] // 2
    left = w - w // 2
    right = w + (w - w // 2)
    top = h - h // 2
    bottom = h + (h - h // 2)
    return array[..., left:right, top:bottom]


def unravel_slice_2d(start, end, shape):
    slices = []
    rows = []
    slices_1d = []
    n = 0
    n_accum = 0
    for index in range(start, end):
        index_in_row = index % shape[-1]
        n += 1
        if index_in_row == shape[-1] - 1:
            slices_1d.append(slice(n_accum, n_accum + n))
            slices.append(slice(index_in_row - n + 1, index_in_row + 1))
            rows.append(index // shape[-1])
            n_accum += n
            n = 0
    if n > 0:
        slices_1d.append(slice(n_accum, n_accum + n))
        slices.append(slice(index_in_row - n + 1, index_in_row + 1))
        rows.append(index // shape[-1])
    return rows, slices, slices_1d


class AbstractDetector(metaclass=ABCMeta):

    def __init__(self, save_file=None):
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
    def detect(self, waves):
        pass

    @abstractmethod
    def allocate_measurement(self, grid, wavelength, scan):
        pass


def cosine_window(x, cutoff, rolloff, attenuate='high'):
    xp = get_array_module(x)

    rolloff *= cutoff
    if attenuate == 'high':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff - rolloff) / rolloff))
        array[x < cutoff] = 0.
        array = xp.where(x < cutoff + rolloff, array, xp.ones_like(x, dtype=xp.float32))
    elif attenuate == 'low':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff + rolloff) / rolloff))
        array[x > cutoff] = 0.
        array = xp.where(x > cutoff - rolloff, array, xp.ones_like(x, dtype=xp.float32))
    else:
        raise RuntimeError('attenuate must be "high" or "low"')

    return array


class AnnularDetector(AbstractDetector):

    def __init__(self, inner, outer, rolloff=0., save_file=None, device=None):

        super().__init__(save_file=save_file)

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
    @watched_method('changed')
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        return self._outer

    @outer.setter
    @watched_method('changed')
    def outer(self, value: float):
        self._outer = value

    @property
    def rolloff(self) -> float:
        return self._rolloff

    @rolloff.setter
    @watched_method('changed')
    def rolloff(self, value: float):
        self._rolloff = value

    @property
    def calibrations(self):
        return ()

    @cached_method('cache')
    def get_integration_region(self, grid, wavelength, xp=np):
        kx, ky = grid.spatial_frequencies()

        alpha_x = xp.asarray(kx) * wavelength
        alpha_y = xp.asarray(ky) * wavelength

        alpha = xp.sqrt(alpha_x.reshape((-1, 1)) ** 2 + alpha_y.reshape((1, -1)) ** 2)

        if self.rolloff > 0.:
            array = cosine_window(alpha, self.outer, self.rolloff) * cosine_window(alpha, self.inner, self.rolloff)
        else:
            array = (alpha >= self._inner) & (alpha <= self._outer)
        return array

    def detect(self, waves, overwrite_x=False):
        xp = get_array_module(waves.array)
        fft2 = get_device_function(xp, 'fft2')
        abs2 = get_device_function(xp, 'abs2')
        # return np.ones(len(waves.array))
        integration_region = self.get_integration_region(waves.grid, waves.wavelength, xp)
        intensity = abs2(fft2(waves.array, overwrite_x=overwrite_x))
        return xp.sum(intensity * integration_region, axis=(1, 2)) / xp.sum(intensity, axis=(1, 2))

    def allocate_measurement(self, grid, wavelength, scan):
        array = np.zeros(scan.shape)
        measurement = Measurement(array, calibrations=scan.calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

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
    @watched_method('changed')
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        return self._outer

    @outer.setter
    @watched_method('changed')
    def outer(self, value: float):
        self._outer = value

    @property
    def nbins_radial(self) -> float:
        return self._nbins_radial

    @nbins_radial.setter
    @watched_method('changed')
    def nbins_radial(self, value: float):
        self._nbins_radial = value

    @property
    def nbins_angular(self) -> float:
        return self._nbins_angular

    @nbins_angular.setter
    @watched_method('changed')
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

    def __init__(self, save_file=None):
        super().__init__(save_file=save_file)

    def allocate_measurement(self, grid, wavelength, scan):
        grid.check_is_defined()
        shape = (grid.gpts[0] // 2, grid.gpts[1] // 2)

        samplings = 1 / grid.extent * wavelength * 1000
        offsets = fourier_space_offset(grid.extent / np.array(shape), shape) * wavelength * 1000
        calibrations = (Calibration(offset=offsets[0], sampling=samplings[0], units='mrad.', name='alpha_x'),
                        Calibration(offset=offsets[1], sampling=samplings[1], units='mrad.', name='alpha_y'))

        array = np.zeros(scan.shape + shape)
        measurement = Measurement(array, calibrations=scan.calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves):
        xp = get_array_module(waves.array)
        abs2 = get_device_function(xp, 'abs2')
        fft2 = get_device_function(xp, 'fft2')

        intensity = abs2(fft2(waves.array, overwrite_x=False))
        intensity = xp.fft.fftshift(intensity, axes=(-1, -2))
        intensity = crop_to_center(intensity)
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
