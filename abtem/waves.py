from collections import Iterable

import h5py
import numpy as np
from ase import Atoms

from abtem.bases import cached_method, HasCache, ArrayWithGrid, Grid, Energy, notifying_property, \
    cached_method_with_args
from abtem.potentials import Potential
from abtem.scan import GridScan, LineScan
from abtem.transfer import CTF
from abtem.utils import complex_exponential, fourier_propagator, fftfreq, squared_norm, BatchGenerator

USE_FFTW = True

if USE_FFTW:
    import pyfftw as fftw


class FourierPropagator(Energy, Grid, HasCache):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    @cached_method_with_args()
    def get_array(self, dz):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        kx = np.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = np.fft.fftfreq(self.gpts[1], self.sampling[1])
        k = (kx ** 2)[:, None] + (ky ** 2)[None]
        return complex_exponential(-k * np.pi * self.wavelength * dz)[None]


class Waves(ArrayWithGrid, Energy, HasCache):

    def __init__(self, array, extent=None, sampling=None, energy=None):
        array = np.array(array, dtype=np.complex)

        if len(array.shape) == 2:
            array = array[None]

        if len(array.shape) != 3:
            raise RuntimeError('array must be 2d or 3d')

        super().__init__(array=array, array_dimensions=3, spatial_dimensions=2, extent=extent, sampling=sampling,
                         space='direct', energy=energy)

    def apply_ctf(self, ctf=None, in_place=False, **kwargs):
        if ctf is None:
            ctf = CTF(**kwargs)

        return self._apply_ctf(ctf, in_place=in_place)

    def _apply_ctf(self, ctf, in_place):

        ctf.match_grid(self)
        ctf.match_energy(self)

        self.check_is_grid_defined()
        self.check_is_energy_defined()

        ctf_array = np.expand_dims(ctf.get_array(), axis=0)
        array = np.fft.ifft2(np.fft.fft2(self.array) * ctf_array)

        return self.__class__(array, extent=self.extent, energy=self.energy)

    def multislice(self, potential, propagator=None, in_place=False, show_progress=False):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        if in_place:
            wave = self
        else:
            wave = self.copy()

        wave.match_grid(potential)
        wave.check_is_grid_defined()
        wave.check_is_energy_defined()

        if propagator is None:
            propagator = FourierPropagator(extent=self.extent, gpts=self.gpts, energy=self.energy)

        if USE_FFTW:
            temp_1 = fftw.empty_aligned(wave._array.shape, dtype='complex64')
            temp_2 = fftw.empty_aligned(wave._array.shape, dtype='complex64')
            fft_object_forward = fftw.FFTW(temp_1, temp_2, axes=(1, 2))
            fft_object_backward = fftw.FFTW(temp_2, temp_1, axes=(1, 2), direction='FFTW_BACKWARD')

        for i in range(potential.num_slices):
            potential_slice = potential.get_slice(i)

            if USE_FFTW:
                temp_1[:] = wave._array * complex_exponential(wave.sigma * potential_slice)
                temp_2[:] = fft_object_forward() * propagator.get_array(potential.slice_thickness)
                wave._array[:] = fft_object_backward()

            else:
                wave._array[:] = np.fft.ifft2(
                    np.fft.fft2(wave._array * complex_exponential(wave.sigma * potential_slice)) *
                    propagator.get_array(potential.slice_thickness))

        return wave

    def get_intensity(self):
        return np.abs(self.array) ** 2

    def get_diffractogram(self):
        return np.abs(np.fft.fftshift(np.fft.fft2(self.array))) ** 2

    def copy(self):
        new = self.__class__(array=self.array.copy(), extent=self.extent.copy(), energy=self.energy)
        return new


class PlaneWaves(Grid, Energy, HasCache):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None):
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


class Probebase:

    def __init__(self):
        super().__init__()

    def get_profile(self):
        raise NotImplementedError()

    def scan(self, *args, **kwargs):
        raise RuntimeError()

    def linescan(self, potential, detectors, start, end, gpts=None, sampling=None, endpoint=True, max_batch=1,
                 return_scan=False, show_progress=True):

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, potential=potential, detectors=detectors, max_batch=max_batch,
                         return_scan=return_scan, show_progress=show_progress)

    def gridscan(self, potential, detectors, start=None, end=None, gpts=None, sampling=None, endpoint=False,
                 max_batch=1, return_scan=False, show_progress=True):

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        if start is None:
            start = potential.origin

        if end is None:
            end = potential.extent

        scan = GridScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, potential=potential, max_batch=max_batch, detectors=detectors,
                         return_scan=return_scan, show_progress=show_progress)


def translate(positions, kx, ky):
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))
    return complex_exponential(2 * np.pi * (kx * x + ky * y))


class ProbeWaves(CTF, Probebase):

    def __init__(self, cutoff=np.inf, rolloff=0., focal_spread=0., extent=None, gpts=None, sampling=None, energy=None,
                 parameters=None, **kwargs):

        super().__init__(cutoff=cutoff, rolloff=rolloff, focal_spread=focal_spread, extent=extent, gpts=gpts,
                         sampling=sampling, energy=energy, parameters=parameters, **kwargs)

    def build_at(self, positions, wrap=True):
        positions = np.array(positions)
        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        kx, ky = fftfreq(self)

        if USE_FFTW:
            a = fftw.empty_aligned((len(positions), self.gpts[0], self.gpts[1]), dtype='complex64')
            b = fftw.empty_aligned((len(positions), self.gpts[0], self.gpts[1]), dtype='complex64')

            fft_object = fftw.FFTW(a, b, axes=(1, 2))

            a[:] = self.get_array() * translate(positions, kx, ky)
            array = fft_object()

        else:
            array = np.fft.fft2(self.get_array() * translate(positions, kx, ky))

        if wrap:
            return Waves(array, extent=self.extent, energy=self.energy)

        else:
            return array

    @property
    def array(self):
        return self.build().array

    def build(self, wrap=True):
        array = np.fft.fftshift(np.fft.fft2(self.get_array()))
        if wrap:
            return Waves(array, extent=self.extent, energy=self.energy)
        else:
            return array

    # def get_intensity_profile(self):
    #    return np.abs(self.build(wrap=False)[self.gpts[0] // 2]) ** 2

    def get_fwhm(self):
        profile = np.abs(self.build(wrap=False)[self.gpts[0] // 2]) ** 2
        peak_idx = np.argmax(profile)
        peak_value = profile[peak_idx]
        left = np.argmin(np.abs(profile[:peak_idx] - peak_value / 2))
        right = peak_idx + np.argmin(np.abs(profile[peak_idx:] - peak_value / 2))
        return (right - left) * self.sampling[0]

    def scan(self, scan, potential, detectors, max_batch, return_scan=False, show_progress=True):
        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self.match_grid(potential)

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

        propagator = FourierPropagator(extent=self.extent, gpts=self.gpts, energy=self.energy)

        for start, stop, positions in scan.generate_positions(max_batch, show_progress=show_progress):
            waves = self.build_at(positions)
            waves = waves.multislice(potential, propagator=propagator, in_place=True, show_progress=False)

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


class PrismWaves(CTF):

    def __init__(self, interpolation=1., cutoff=.03, rolloff=0., focal_spread=0., extent=None, gpts=None, sampling=None,
                 energy=None, parameters=None, **kwargs):
        self._interpolation = interpolation

        super().__init__(cutoff=cutoff, rolloff=rolloff, focal_spread=focal_spread, extent=extent, gpts=gpts,
                         sampling=sampling, energy=energy, parameters=parameters, **kwargs)

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

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_alpha(self):
        kx, ky = self.get_spatial_frequencies()
        return np.sqrt(kx ** 2 + ky ** 2)

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_phi(self):
        kx, ky = self.get_spatial_frequencies()
        phi = np.arctan2(kx, ky)
        return phi

    def generate_expansion(self, max_batch_size=None):
        x = np.linspace(0, self.extent[0], self.gpts[0], endpoint=self.endpoint)
        y = np.linspace(0, self.extent[1], self.gpts[1], endpoint=self.endpoint)
        kx, ky = self.get_spatial_frequencies()

        if max_batch_size is None:
            max_batch_size = len(kx)

        batch_generator = BatchGenerator(len(kx), max_batch_size)

        for start, length in batch_generator.generate():
            yield complex_exponential(-2 * np.pi * (kx[start:start + length, None, None] * x[None, :, None] +
                                                    ky[start:start + length, None, None] * y[None, None, :]))

    def multislice(self):

        propagator = FourierPropagator(extent=self.extent, gpts=self.gpts, energy=self.energy)

        for waves in self.generate_expansion():
            pass
