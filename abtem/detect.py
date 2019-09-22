import numpy as np

from abtem.bases import Grid, Energy, HasCache, SelfObservable, notifying_property, cached_method
from abtem.utils import squared_norm, semiangles
from skimage.transform import resize


class DetectorBase:

    def __init__(self, export=None, **kwargs):
        if export is not None:
            if not export.endswith('.hdf5'):
                self._export = export + '.hdf5'
            else:
                self._export = export

        else:
            self._export = None

        super().__init__(**kwargs)

    @property
    def export(self):
        return self._export

    def out_shape(self):
        raise NotImplementedError()

    def detect(self, wave):
        raise NotImplementedError()


class PtychographyDetector(Energy, Grid, DetectorBase):

    def __init__(self, max_angle=None, resize_isotropic=False, extent=None, gpts=None, sampling=None, energy=None,
                 export=None):
        self._resize_isotropic = resize_isotropic
        self._crop_to_angle = max_angle

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, export=export)

    @property
    def export(self):
        return self._export

    @property
    def out_shape(self):
        if self._crop_to_angle:
            angular_extent = self.gpts / self.extent * self.wavelength / 2
            out_shape = tuple(np.ceil(self._crop_to_angle / angular_extent * self.gpts / 2.).astype(int) * 2)
            if self._resize_isotropic:
                return (min(out_shape), min(out_shape))
            else:
                return out_shape
        elif self._resize_isotropic:
            return (np.min(self.gpts), np.min(self.gpts))
        else:
            return tuple(self.gpts)

    def detect(self, wave):
        self.match_grid(wave)

        intensity = np.fft.fftshift(np.abs(np.fft.fft2(wave.array)) ** 2, axes=(1, 2))

        if self._resize_isotropic:
            new_size = (np.min(self.gpts), np.min(self.gpts))
            resized_intensity = np.zeros((intensity.shape[0],) + new_size)
            for i in range(intensity.shape[0]):
                resized_intensity[i] = resize(intensity[i], new_size, order=0)
            intensity = resized_intensity

        if self._crop_to_angle:
            out_shape = self.out_shape
            crop = ((intensity.shape[1] - out_shape[0]) // 2, (intensity.shape[2] - out_shape[1]) // 2)
            intensity = intensity[:, crop[0]:crop[0] + out_shape[0], crop[1]:crop[1] + out_shape[1]]

        #     resized_intensity = np.zeros((intensity.shape[0],) + self.out_shape)
        #     for i in range(intensity.shape[0]):
        #         resized_intensity[i] = resize(intensity[i], self.out_shape)
        #     return resized_intensity
        # else:
        return intensity


class RingDetector(DetectorBase, Energy, Grid, HasCache, SelfObservable):

    def __init__(self, inner, outer, rolloff=0., integrate=True, extent=None, gpts=None, sampling=None, energy=None,
                 export=None):

        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff
        self._integrate = integrate

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, export=export)

    inner = notifying_property('_inner')
    outer = notifying_property('_outer')
    rolloff = notifying_property('_rolloff')

    @property
    def out_shape(self):
        return (1,)

    @cached_method()
    def get_efficiency(self):

        self.check_is_grid_defined()
        self.check_is_energy_defined()

        alpha2 = squared_norm(*semiangles(self))
        alpha = np.sqrt(alpha2)

        if self.rolloff > 0.:
            outer = .5 * (1 + np.cos(np.pi * (alpha - self.outer) / self.rolloff))
            outer *= alpha < self.outer + self.rolloff
            outer = np.where(alpha > self.outer, outer, np.ones(alpha.shape))

            inner = .5 * (1 + np.cos(np.pi * (self.inner - alpha) / self.rolloff))
            inner *= alpha > self.inner - self.rolloff
            inner = np.where(alpha < self.inner, inner, np.ones(alpha.shape))
            array = inner * outer

        else:
            array = (alpha >= self._inner) & (alpha <= self._outer)

        return array

    def detect(self, wave):
        self.match_grid(wave)
        self.match_energy(wave)

        intensity = np.abs(np.fft.fft2(wave.array)) ** 2
        efficiency = self.get_efficiency()

        return np.sum(intensity * efficiency.reshape((1,) + efficiency.shape), axis=(1, 2)) / np.sum(intensity,
                                                                                                     axis=(1, 2))
