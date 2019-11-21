import h5py
import numpy as np

from abtem.bases import Grid, Energy, cached_method, Cache, notify, ArrayWithGridAndEnergy
from abtem.utils import squared_norm, semiangles


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

    @property
    def out_shape(self) -> tuple:
        raise NotImplementedError()

    def detect(self, wave):
        raise NotImplementedError()


class RealSpaceDetector:
    pass


class FourierSpaceDetector(Energy, Grid, DetectorBase):

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


class RingDetector(DetectorBase, Energy, Grid, Cache):

    def __init__(self, inner, outer, rolloff=0., extent=None, gpts=None, sampling=None, energy=None,
                 export=None):

        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, export=export)

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

        return ArrayWithGridAndEnergy(array, spatial_dimensions=2, extent=self.extent, energy=self.energy)

    def detect(self, waves):
        self.extent = waves.extent
        self.gpts = waves.gpts
        self.energy = waves.energy

        intensity = np.abs(np.fft.fft2(waves.array)) ** 2

        return np.sum(intensity * self.get_efficiency().array, axis=(1, 2)) / np.sum(intensity, axis=(1, 2))

    def copy(self) -> 'RingDetector':
        return self.__class__(self.inner, self.outer, rolloff=self.rolloff, extent=self.extent, gpts=self.gpts,
                              energy=self.energy, export=self.export)
