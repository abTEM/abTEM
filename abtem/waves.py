from collections import Iterable

import h5py
import numpy as np
import pyfftw as fftw
from ase import Atoms
from numba import jit, prange

from abtem.bases import cached_method, HasCache, Grid, Energy, notifying_property, cached_method_with_args, \
    SelfObservable, Image, ArrayWithGridAndEnergy
from abtem.potentials import Potential
from abtem.scan import GridScan, LineScan, CustomScan
from abtem.transfer import CTF, CTFBase
from abtem.utils import complex_exponential, fftfreq, BatchGenerator, convert_complex


def get_fft_plans(shape, threads=16):
    temp_1 = fftw.empty_aligned(shape, dtype='complex128')
    temp_2 = fftw.empty_aligned(shape, dtype='complex128')
    fft_object_forward = fftw.FFTW(temp_1, temp_2, axes=(1, 2), threads=threads)
    fft_object_backward = fftw.FFTW(temp_2, temp_1, axes=(1, 2), direction='FFTW_BACKWARD', threads=threads)
    return temp_1, temp_2, fft_object_forward, fft_object_backward


class Propagator(Grid, Energy, HasCache):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        super().__init__(gpts=gpts, extent=extent, sampling=sampling, energy=energy)

    @cached_method_with_args(('gpts', 'extent', 'sampling', 'energy'))
    def get_array(self, dz):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        kx = np.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = np.fft.fftfreq(self.gpts[1], self.sampling[1])
        k = (kx ** 2)[:, None] + (ky ** 2)[None]
        return complex_exponential(-k * np.pi * self.wavelength * dz)[None]


def multislice(waves, potential, in_place=False, show_progress=False):
    if not in_place:
        waves = waves.copy()
    else:
        waves = waves

    if isinstance(potential, Atoms):
        potential = Potential(atoms=potential)

    temp_1, temp_2, fft_object_forward, fft_object_backward = get_fft_plans(waves.array.shape)

    propagator = Propagator(extent=waves.extent, gpts=waves.gpts, energy=waves.energy)

    for i in range(potential.num_slices):
        temp_1[:] = waves.array * complex_exponential(waves.sigma * potential.get_slice(i))
        temp_2[:] = fft_object_forward() * propagator.get_array(potential.slice_thickness(i))
        waves.array[:] = fft_object_backward()

    return waves


class Waves(ArrayWithGridAndEnergy):

    def __init__(self, array, extent=None, sampling=None, energy=None):
        array = np.array(array, dtype=np.complex)
        super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling, energy=energy)

    def apply_ctf(self, ctf=None, in_place=False, **kwargs):
        if not in_place:
            waves = self.copy()
        else:
            waves = self

        if ctf is None:
            ctf = CTF(**kwargs)

        ctf.match_grid(waves)
        ctf.match_energy(waves)

        waves.check_is_grid_defined()
        waves.check_is_energy_defined()

        temp_1, temp_2, fft_object_forward, fft_object_backward = get_fft_plans(self.array.shape)
        temp_1[:] = waves.array
        temp_2[:] = fft_object_forward() * ctf.get_array()
        array = fft_object_backward()

        return self.__class__(array, extent=self.extent, energy=self.energy)

    def multislice(self, potential, in_place=False, show_progress=True):
        return multislice(self, potential, in_place=in_place, show_progress=show_progress)

    def get_image(self, i=0, convert='intensity'):
        return Image(convert_complex(self._array[i], convert), extent=self.extent, space='direct')

    def copy(self):
        new = self.__class__(array=self.array.copy(), extent=self.extent.copy(), energy=self.energy)
        return new


class PlaneWaves(Grid, Energy, HasCache):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None):
        """
        Plane waves object



        :param num_waves: number of waves
        :type num_waves: int
        :param extent:
        :type extent:
        :param gpts:
        :type gpts:
        :param sampling:
        :type sampling:
        :param energy:
        :type energy:
        """

        self._num_waves = num_waves
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, dimensions=2, energy=energy)

    num_waves = notifying_property('_num_waves')

    def multislice(self, potential):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        self.match_grid(potential)
        return self.build().multislice(potential, in_place=True)

    @cached_method()
    def build(self):
        array = np.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=np.complex64)
        return Waves(array, extent=self.extent, energy=self.energy)

    def copy(self):
        new = self.__class__(num_waves=self.num_waves, extent=self.extent.copy(), energy=self.energy)
        return new


class Scanable:

    def __init__(self):
        super().__init__()

    def generate_scan_waves(self, scan, potential, max_batch, show_progress):
        raise NotImplementedError()

    def scan(self, scan, potential, detectors, max_batch, show_progress=True):
        if isinstance(potential, Atoms):
            potential = Potential(potential)

        if not isinstance(detectors, Iterable):
            detectors = [detectors]

        for detector in detectors:
            detector.match_grid(self)
            detector.match_energy(self)

            data_shape = (int(np.prod(scan.gpts)),) + tuple(n for n in detector.out_shape if n > 1)

            if detector.export is not None:
                f = h5py.File(detector.export, 'w')

                measurement_gpts = f.create_dataset('gpts', (2,), dtype=np.int)
                measurement_extent = f.create_dataset('extent', (2,), dtype=np.float)
                measurement_endpoint = f.create_dataset('endpoint', (1,), dtype=np.bool)

                measurement_gpts[:] = scan.gpts
                measurement_extent[:] = scan.extent
                measurement_endpoint[:] = scan.endpoint

                f.create_dataset('data', data_shape, dtype=np.float32)
                f.close()
            else:
                scan.measurements[detector] = np.zeros(data_shape)

        for start, stop, waves in self.generate_scan_waves(scan, potential, max_batch, show_progress):

            for detector in detectors:
                if detector.export is not None:
                    with h5py.File(detector.export, 'a') as f:
                        f['data'][start:start + stop] = detector.detect(waves)

                else:
                    scan.measurements[detector][start:start + stop] = detector.detect(waves)

        for detector in detectors:
            if not detector.export:
                scan.measurements[detector] = scan.measurements[detector].reshape(tuple(scan.gpts) + detector.out_shape)
                scan.measurements[detector] = np.squeeze(scan.measurements[detector])

        return scan

    def custom_scan(self, potential, detectors, positions, max_batch=1, show_progress=True):
        scan = CustomScan(positions=positions)
        return self.scan(scan=scan, potential=potential, detectors=detectors, max_batch=max_batch,
                         show_progress=show_progress)

    def linescan(self, potential, detectors, start, end, gpts=None, sampling=None, endpoint=True, max_batch=1,
                 show_progress=True):

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, potential=potential, detectors=detectors, max_batch=max_batch,
                         show_progress=show_progress)

    def gridscan(self, potential, detectors, start=None, end=None, gpts=None, sampling=None, endpoint=False,
                 max_batch=1, show_progress=True):

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        if start is None:
            start = potential.origin

        if end is None:
            end = potential.extent

        scan = GridScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, potential=potential, max_batch=max_batch, detectors=detectors,
                         show_progress=show_progress)


def translate(positions, kx, ky):
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))
    return complex_exponential(2 * np.pi * (kx * x + ky * y))


class ProbeWaves(CTF, Scanable):

    def __init__(self, cutoff=np.inf, rolloff=0., focal_spread=0., extent=None, gpts=None, sampling=None, energy=None,
                 parameters=None, normalize=False, **kwargs):

        self._normalize = normalize

        super().__init__(cutoff=cutoff, rolloff=rolloff, focal_spread=focal_spread, extent=extent, gpts=gpts,
                         sampling=sampling, energy=energy, parameters=parameters, **kwargs)

    normalize = notifying_property('_normalize')

    def build_at(self, positions):
        positions = np.array(positions)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        kx, ky = fftfreq(self)

        temp_1 = fftw.empty_aligned((len(positions), self.gpts[0], self.gpts[1]), dtype='complex64')
        temp_2 = fftw.empty_aligned((len(positions), self.gpts[0], self.gpts[1]), dtype='complex64')

        fft_object = fftw.FFTW(temp_1, temp_2, axes=(1, 2))

        temp_1[:] = self.get_array() * translate(positions, kx, ky)
        fft_object()

        return Waves(temp_2, extent=self.extent, energy=self.energy)

    def generate_scan_waves(self, scan, potential, max_batch, show_progress=True):
        self.match_grid(potential)

        for start, stop, positions in scan.generate_positions(max_batch, show_progress=show_progress):
            waves = self.build_at(positions)

            waves.multislice(potential=potential, in_place=True, show_progress=False)

            yield start, stop, waves

    @property
    def array(self):
        return self.build().array

    def build_ctf(self):
        return super().build()

    def build(self):
        array = np.fft.fftshift(np.fft.fft2(self.get_array()))

        if self.normalize:
            array[:] = array / np.sum(np.abs(array) ** 2, axis=(1, 2)) * np.prod(array.shape[1:])[None]

        return Waves(array, extent=self.extent, energy=self.energy)


class PrismWaves(Grid, Energy, HasCache, SelfObservable):

    def __init__(self, interpolation=1., cutoff=.03, extent=None, gpts=None, sampling=None, energy=None):
        self._interpolation = interpolation
        self._cutoff = cutoff

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    cutoff = notifying_property('_cutoff')
    interpolation = notifying_property('_interpolation')

    @cached_method()
    def get_spatial_frequencies(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()

        n_max = np.ceil(self.cutoff / (self.wavelength / self.extent[0] * self.interpolation))
        m_max = np.ceil(self.cutoff / (self.wavelength / self.extent[1] * self.interpolation))

        kx = np.arange(-n_max, n_max + 1) / self.extent[0] * self.interpolation
        ky = np.arange(-m_max, m_max + 1) / self.extent[1] * self.interpolation

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.cutoff / self.wavelength) ** 2
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        return kx[mask], ky[mask]

    def generate_expansion(self, max_batch_size=None):
        kx, ky = self.get_spatial_frequencies()
        x = np.linspace(0, self.extent[0], self.gpts[0], endpoint=self.endpoint)
        y = np.linspace(0, self.extent[1], self.gpts[1], endpoint=self.endpoint)

        if max_batch_size is None:
            max_batch_size = len(kx)

        batch_generator = BatchGenerator(len(kx), max_batch_size)

        for start, length in batch_generator.generate():
            kx_batch = kx[start:start + length]
            ky_batch = ky[start:start + length]
            yield ScatteringMatrix(complex_exponential(-2 * np.pi *
                                                       (kx_batch[:, None, None] * x[None, :, None] +
                                                        ky_batch[:, None, None] * y[None, None, :])),
                                   interpolation=self.interpolation, cutoff=self.cutoff, extent=self.extent,
                                   energy=self.energy, kx=kx_batch, ky=ky_batch)

    def build(self):
        return next(self.generate_expansion())

    def multislice(self, potential):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        self.match_grid(potential)

        for S in self.generate_expansion():
            S.multislice(potential, in_place=True)

        return S


def prism_translate(positions, kx, ky):
    return complex_exponential(2 * np.pi * (kx[None] * positions[:, 0, None] + ky[None] * positions[:, 1, None]))


class ScatteringMatrix(CTFBase, Grid):

    def __init__(self, array, interpolation, cutoff, kx, ky, extent=None, sampling=None, energy=None):
        self._interpolation = interpolation
        self._cutoff = cutoff
        self._kx = kx
        self._ky = ky
        self._positions = np.array([[0., 0.]])
        super().__init__(array=array, extent=extent, sampling=sampling, energy=energy)

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        value = np.array(value)
        assert value.shape[1] == 2
        change = np.any(self.positions != value)
        self._positions = np.array(value)
        self.notify_observers({'name': 'position', 'old': self.positions, 'new': value, 'change': change})

    # @cached_method('any')
    def get_array(self):
        coefficients = super().get_array()[0] * prism_translate(self.positions, self.kx, self.ky)

        window_shape = (len(self.positions), self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)
        window = np.zeros(window_shape, dtype=np.complex)

        corners = np.round(self.positions / self.sampling - np.floor_divide(window.shape[1:], 2)).astype(np.int)
        corners = np.remainder(corners, self.gpts)

        window_and_collapse(self.array, window, corners, coefficients)

        return window

    def generate_scan_waves(self, scan, potential=None, max_batch=1, show_progress=True):
        for start, stop, positions in scan.generate_positions(max_batch, show_progress=show_progress):
            self.positions = positions
            yield start, stop, self.build()

    def build(self):
        return Waves(self.get_array(), extent=self.extent, energy=self.energy)

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_alpha(self):
        return np.sqrt(self._kx ** 2 + self._ky ** 2) * self.wavelength

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_phi(self):
        return np.arctan2(self._kx, self._ky)
