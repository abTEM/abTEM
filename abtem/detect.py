import cupy as cp
import numpy as np
import pyfftw as fftw

from abtem.bases import Grid, Energy, cached_method, Cache, notify, ArrayWithGridAndEnergy2D, Buildable
from abtem.config import FFTW_THREADS, COMPLEX_DTYPE
from abtem.utils import squared_norm, cosine_window, abs2


class DetectorBase:

    def __init__(self, output=None, **kwargs):

        if output is not None:
            if not output.endswith('.hdf5'):
                self._output = output + '.hdf5'
            else:
                self._output = output

        else:
            self._output = None

        super().__init__(**kwargs)

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self) -> tuple:
        raise NotImplementedError()

    def detect(self, waves):
        raise NotImplementedError()


class AnnularDetector(DetectorBase, Energy, Grid, Buildable, Cache):

    def __init__(self, inner, outer, rolloff=0., extent=None, gpts=None, sampling=None, energy=None,
                 output=None, build_on_gpu=False):

        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, output=output,
                         build_on_gpu=build_on_gpu)

        self.register_observer(self)

    @property
    def inner(self) -> float:
        return self._inner

    @inner.setter
    @notify
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        return self._outer

    @outer.setter
    @notify
    def outer(self, value: float):
        self._outer = value

    @property
    def rolloff(self) -> float:
        return self._rolloff

    @rolloff.setter
    @notify
    def rolloff(self, value: float):
        self._rolloff = value

    @property
    def output_shape(self) -> tuple:
        return ()

    @cached_method()
    def get_integration_region(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        xp = self._array_module()

        kx, ky = self.fftfreq()

        kx = xp.asarray(kx * self.wavelength)
        ky = xp.asarray(ky * self.wavelength)

        alpha = xp.sqrt(squared_norm(kx, ky))

        if self.rolloff > 0.:
            array = cosine_window(alpha, self.outer, self.rolloff) * cosine_window(alpha, self.inner, self.rolloff)

        else:
            array = (alpha >= self._inner) & (alpha <= self._outer)

        return ArrayWithGridAndEnergy2D(array, extent=self.extent, energy=self.energy, fourier_space=True)

    def detect(self, waves, in_place=False):
        self.match_grid(waves)
        self.match_energy(waves)

        xp = self._array_module()

        array = waves.array

        if in_place:
            out_array = array
        else:
            out_array = xp.zeros(array.shape, dtype=COMPLEX_DTYPE)

        if self._build_on_gpu:
            intensity = abs2(cp.fft.fft2(array))
        else:
            fftw.FFTW(array, out_array, axes=(1, 2), threads=FFTW_THREADS, flags=('FFTW_ESTIMATE',))()
            intensity = abs2(out_array)

        result = xp.sum(intensity * self.get_integration_region().array, axis=(1, 2)) / xp.sum(intensity, axis=(1, 2))
        return result

    def copy(self) -> 'AnnularDetector':
        return self.__class__(self.inner, self.outer, rolloff=self.rolloff, extent=self.extent, gpts=self.gpts,
                              energy=self.energy, export=self.output)


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

    bins = -np.ones(shape, dtype=int)
    bins[valid] = angular_bins[valid] * nbins_radial + radial_bins[valid]
    return bins




class SegmentedDetector(Energy, Grid, DetectorBase, Cache):

    def __init__(self, inner, outer, nbins_radial, nbins_angular=1, extent=None, gpts=None, sampling=None,
                 energy=None, output=None, build_on_gpu=False):
        self._inner = inner
        self._outer = outer
        self._nbins_radial = nbins_radial
        self._nbins_angular = nbins_angular
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, output=output)

    @property
    def inner(self) -> float:
        return self._inner

    @inner.setter
    @notify
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        return self._outer

    @outer.setter
    @notify
    def outer(self, value: float):
        self._outer = value

    @property
    def nbins_radial(self) -> float:
        return self._nbins_radial

    @nbins_radial.setter
    @notify
    def nbins_radial(self, value: float):
        self._nbins_radial = value

    @property
    def nbins_angular(self) -> float:
        return self._nbins_angular

    @nbins_angular.setter
    @notify
    def nbins_angular(self, value: float):
        self._nbins_angular = value

    def out_shape(self) -> tuple:
        return (self.nbins_radial, self.nbins_angular)

    @cached_method()
    def get_integration_region(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()

        kx, ky = self.fftfreq()
        alpha_x = kx * self.wavelength
        alpha_y = ky * self.wavelength

        alpha = np.sqrt(squared_norm(alpha_x, alpha_y))

        radial_bins = -np.ones(self.gpts, dtype=int)
        valid = (alpha > self.inner) & (alpha < self.outer)
        radial_bins[valid] = self.nbins_radial * (alpha[valid] - self.inner) / (self.outer - self.inner)

        angles = np.arctan2(alpha_x[:, None], alpha_y[None]) % (2 * np.pi)

        angular_bins = np.floor(self.nbins_angular * (angles / (2 * np.pi)))
        angular_bins = np.clip(angular_bins, 0, self.nbins_angular - 1).astype(np.int)

        bins = -np.ones(self.gpts, dtype=int)
        bins[valid] = angular_bins[valid] * self.nbins_radial + radial_bins[valid]

        return ArrayWithGridAndEnergy2D(bins, extent=self.extent, energy=self.energy, fourier_space=True)


class PixelatedDetector(Energy, Grid, DetectorBase):

    def __init__(self, max_angle=None, extent=None, gpts=None, sampling=None, energy=None, export=None):
        self.max_angle = max_angle
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, export=export)

    @property
    def out_shape(self):
        if self.max_angle:
            self.check_is_grid_defined()
            self.check_is_energy_defined()

            angular_extent = self.gpts / self.extent * self.wavelength / 2
            out_shape = tuple(np.ceil(self.max_angle / angular_extent * self.gpts / 2.).astype(int) * 2)

            return out_shape
        else:
            return tuple(self.gpts)

    def detect(self, waves):
        intensity = np.fft.fftshift(np.abs(np.fft.fft2(waves.array)) ** 2, axes=(1, 2))

        if self.max_angle:
            self.check_is_grid_defined()
            self.check_is_energy_defined()

            out_shape = self.out_shape
            crop = ((intensity.shape[1] - out_shape[0]) // 2, (intensity.shape[2] - out_shape[1]) // 2)
            intensity = intensity[:, crop[0]:crop[0] + out_shape[0], crop[1]:crop[1] + out_shape[1]]

        return intensity
