import numpy as np

from abtem.bases import Grid, Energy, cached_method, Cache, notify, ArrayWithGridAndEnergy
from abtem.utils import squared_norm
from abtem.bases import semiangles
import functools

def polar_labels(shape, inner=1, outer=None, nbins_angular=32, nbins_radial=None):
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


def generate_indices(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


@functools.lru_cache(maxsize=1)
def polar_indices(shape, inner, outer, nbins_angular):
    labels = polar_labels(shape, inner=inner, outer=outer, nbins_angular=nbins_angular)

    indices = np.zeros((labels.max() + 1, nbins_angular), dtype=np.int)
    weights = np.zeros((labels.max() + 1, nbins_angular), dtype=np.float32)
    lengths = np.zeros((labels.max() + 1,), dtype=np.int)

    for j, i in enumerate(generate_indices(labels, first_label=0)):
        if len(i) > 0:
            indices[j, :len(i)] = i
            weights[j, :len(i)] = 1 / len(i)
            lengths[j] = len(i)

    indices = indices.reshape((nbins_angular, -1, nbins_angular))
    weights = weights.reshape((nbins_angular, -1, nbins_angular))
    lengths = lengths.reshape((nbins_angular, -1))
    nans = lengths == 0

    for i in range(indices.shape[0]):
        k = np.where(nans[:, i] == 0)[0]
        for j in np.where(nans[:, i])[0]:
            idx = bisect_left(k, j)
            idx = idx % len(k)

            l1 = lengths[k[idx - 1], i]
            l2 = lengths[k[idx], i]

            indices[j, i, :l1] = indices[k[idx - 1], i, :l1]
            indices[j, i, l1:l1 + l2] = indices[k[idx], i, :l2]

            d1 = min(abs(k[idx - 1] - j), abs((nbins_angular - k[idx - 1] + j)))
            d2 = min(abs(k[idx] - j), abs((-nbins_angular - k[idx] + j)))

            weights[j, i, :l1] = 1 / d1
            weights[j, i, l1:l1 + l2] = 1 / d2
            weights[j, i, :l1 + l2] /= weights[j, i, :l1 + l2].sum()

    indices = indices[:, :, :np.max(lengths)]
    weights = weights[:, :, :np.max(lengths)]

    return indices, weights


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

    def match_grid_and_energy(self, waves):
        self.extent = waves.extent
        self.gpts = waves.gpts
        self.energy = waves.energy


class PolarDetector:
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

    @property
    def array(self) -> np.ndarray:
        return self.get_efficiency()

    def detect(self, waves):
        self.extent = waves.extent
        self.gpts = waves.gpts
        self.energy = waves.energy

        intensity = np.abs(np.fft.fft2(waves.array)) ** 2

        return np.sum(intensity * self.get_efficiency().array, axis=(1, 2)) / np.sum(intensity, axis=(1, 2))

    def copy(self) -> 'RingDetector':
        return self.__class__(self.inner, self.outer, rolloff=self.rolloff, extent=self.extent, gpts=self.gpts,
                              energy=self.energy, export=self.export)
