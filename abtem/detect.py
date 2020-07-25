from abc import ABCMeta, abstractmethod

import numpy as np

from abtem.bases import Cache, Event, cache_clear_callback, watched_property, Grid, cached_method
from abtem.cpu_kernels import abs2
from abtem.device import get_array_module, get_device_function
from abtem.measure import Calibration, calibrations_from_grid, fourier_space_offset, Measurement
from abtem.plot import show_image
from abtem.utils import label_to_index_generator, spatial_frequencies
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


def calculate_far_field_intensity(waves, overwrite=False):
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


class _PolarDetector(AbstractDetector):

    def __init__(self, inner=None, outer=None, radial_steps=.001, azimuthal_steps=None, save_file=None):
        self._inner = inner
        self._outer = outer

        self._radial_steps = radial_steps

        if azimuthal_steps is None:
            azimuthal_steps = 2 * np.pi

        self._azimuthal_steps = azimuthal_steps

        self.cache = Cache(1)
        self.changed = Event()
        super().__init__(save_file=save_file)

    def _get_bins(self, sampling, wavelength):
        if self._inner is None:
            inner = 0
        else:
            inner = self._inner

        max_angle = 1 / np.max(sampling) * wavelength / 2

        if self._outer is None:
            outer = max_angle
        elif self._outer < max_angle:
            outer = self._outer
        else:
            raise RuntimeError()

        nbins_radial = int(np.ceil((outer - inner) / self._radial_steps))
        nbins_azimuthal = int(np.ceil(2 * np.pi / self._azimuthal_steps))
        return inner, outer, nbins_radial, nbins_azimuthal

    @cached_method('cache')
    def _get_regions(self, gpts, sampling, wavelength):
        inner, outer, nbins_radial, nbins_azimuthal = self._get_bins(sampling, wavelength)

        region_labels = polar_regions(gpts, sampling, wavelength, inner, outer, nbins_radial, nbins_azimuthal)
        region_indices = []
        for indices in label_to_index_generator(region_labels):
            region_indices.append(indices)

        return region_indices

    def allocate_measurement(self, grid, wavelength, scan):
        inner, outer, nbins_radial, nbins_azimuthal = self._get_bins(grid.antialiased_sampling, wavelength)

        shape = scan.shape
        calibrations = scan.calibrations

        if nbins_radial > 1:
            shape += (nbins_radial,)
            calibrations += (Calibration(offset=inner * 1000, sampling=self._radial_steps, units='mrad'),)

        if nbins_azimuthal > 1:
            shape += (nbins_azimuthal,)
            calibrations += (Calibration(offset=0, sampling=self._azimuthal_steps, units='rad'),)

        array = np.zeros(shape, dtype=np.float32)
        measurement = Measurement(array, calibrations=calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def show(self, grid, wavelength, **kwargs):
        array = np.full(grid.antialiased_gpts, -1, dtype=np.int)
        for i, indices in enumerate(self._get_regions(grid.antialiased_gpts, grid.antialiased_sampling, wavelength)):
            array.ravel()[indices] = i

        calibrations = calibrations_from_grid(grid.antialiased_gpts, grid.antialiased_sampling,
                                              names=['alpha_x', 'alpha_y'], units='mrad.',
                                              scale_factor=wavelength, fourier_space=True)
        return show_image(array, calibrations, **kwargs)


class AnnularDetector(_PolarDetector):

    def __init__(self, inner, outer, save_file=None):
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

    def detect(self, waves):
        xp = get_array_module(waves.array)
        intensity = calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)[0]
        total = xp.sum(intensity, axis=(-2, -1))
        return xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total

    def copy(self) -> 'AnnularDetector':
        return self.__class__(self.inner, self.outer, save_file=self.save_file)


class AdjustableAnnularDetector(_PolarDetector):

    def __init__(self, step_size=.001, save_file=None):
        super().__init__(radial_steps=step_size, save_file=save_file)

    def detect(self, waves):
        xp = get_array_module(waves.array)
        intensity = calculate_far_field_intensity(waves, overwrite=False)
        indices = self._get_regions(waves.grid.antialiased_gpts, waves.grid.antialiased_sampling, waves.wavelength)

        total = xp.sum(intensity, axis=(-2, -1))
        result = np.zeros((len(intensity), len(indices)), dtype=np.float32)
        for i, indices in enumerate(indices):
            result[:, i] = xp.sum(intensity.reshape((intensity.shape[0], -1))[:, indices], axis=-1) / total
        return result


class SegmentedDetector(AbstractDetector):

    def __init__(self, inner, outer, nbins_radial, nbins_angular=1, save_file=None):

        self._inner = inner
        self._outer = outer
        self._nbins_radial = nbins_radial
        self._nbins_angular = nbins_angular
        super().__init__(save_file=save_file)

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


class WavefunctionDetector(AbstractDetector):

    def __init__(self, save_file=None, fourier_space=False):
        super().__init__(save_file=save_file)
        self._fourier_space = fourier_space

    def allocate_measurement(self, grid, wavelength, scan):
        grid.check_is_defined()
        shape = (grid.gpts[0], grid.gpts[1])

        if self._fourier_space:
            offsets = (fourier_space_offset(grid.extent[0] / shape[0], shape[0]) * wavelength * 1000,
                       fourier_space_offset(grid.extent[1] / shape[1], shape[1]) * wavelength * 1000)
            shape = (shape[0] // 2, shape[1] // 2)
            sampling = (1 / grid.extent[0] * wavelength * 1000, 1 / grid.extent[1] * wavelength * 1000)
        else:
            offsets = (0, 0)
            sampling = grid.sampling

        calibrations = (Calibration(offset=offsets[0], sampling=sampling[0], units='mrad.', name='alpha_x'),
                        Calibration(offset=offsets[1], sampling=sampling[1], units='mrad.', name='alpha_y'))

        # array = np.zeros(scan.shape + shape, dtype=np.complex64)
        # measurement = Measurement(array, calibrations=scan.calibrations + calibrations)
        if isinstance(self.save_file, str):
            with h5py.File(self.save_file, 'w') as f:
                f.create_dataset('array', scan.shape + shape, dtype=np.complex64)
                f.create_dataset('offset', data=[calibration.offset for calibration in calibrations])
                f.create_dataset('sampling', data=[calibration.sampling for calibration in calibrations])
                units = [calibration.units.encode('utf-8') for calibration in calibrations]
                f.create_dataset('units', (len(units),), 'S10', units)
                names = [calibration.name.encode('utf-8') for calibration in calibrations]
                f.create_dataset('name', (len(names),), 'S10', names)

            # measurement = measurement.write(self.save_file)
        return self.save_file

    def detect(self, waves):
        if self._fourier_space:
            xp = get_array_module(waves.array)
            fft2 = get_device_function(xp, 'fft2')
            return crop_to_center(fft2(waves.array))
        else:
            return waves.array
