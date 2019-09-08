import numpy as np

from abtem.bases import Grid, Energy, HasCache, Observable, notifying_property, cached_method
from abtem.utils import squared_norm, semiangles
from skimage.transform import resize


class DetectorBase(object):

    # def __init__(self, extent=None, gpts=None, sampling=None):
    #    Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)

    def out_shape(self):
        raise NotImplementedError()

    def detect(self, wave):
        raise NotImplementedError()


class PtychographyDetector(DetectorBase, Grid):

    def __init__(self, extent=None, gpts=None, sampling=None, resize_isotropic=False):
        self._resize_isotropic = resize_isotropic
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)

    @property
    def out_shape(self):
        if self._resize_isotropic:
            return (np.min(self.gpts), np.min(self.gpts))
        else:
            return tuple(self.gpts)

    def detect(self, wave):
        self.match_grid(wave)

        intensity = np.fft.fftshift(np.abs(np.fft.fft2(wave.array)) ** 2, axes=(1, 2))

        if self._resize_isotropic:
            resized_intensity = np.zeros((intensity.shape[0],) + self.out_shape)
            for i in range(intensity.shape[0]):
                resized_intensity[i] = resize(intensity[i], self.out_shape)
            return resized_intensity
        else:
            return intensity


class RingDetector(DetectorBase, Energy, Grid, HasCache, Observable):

    def __init__(self, inner, outer, rolloff=0., integrate=True, extent=None, gpts=None, sampling=None, energy=None):

        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff
        self._integrate = integrate

        Energy.__init__(self, energy=energy)
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        HasCache.__init__(self)
        Observable.__init__(self)
        DetectorBase.__init__(self)

        self.register_observer(self)

    inner = notifying_property('_inner')
    outer = notifying_property('_outer')
    rolloff = notifying_property('_rolloff')

    @property
    def out_shape(self):
        return (1,)

    @cached_method
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
