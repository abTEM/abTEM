from collections import Iterable

import numpy as np
from ase import Atoms
import sys
from abtem.bases import cached_method, HasCache, ArrayWithGrid, Grid, Energy, notifying_property
from abtem.potentials import Potential
from abtem.scan import GridScan, LineScan
from abtem.transfer import CTF
from abtem.utils import complex_exponential, fourier_propagator, fftfreq

USE_FFTW = False

if USE_FFTW:
    import pyfftw as fftw


class WavesBase(object):

    @property
    def array(self):
        raise NotImplementedError()

    @property
    def intensity_as_array(self):
        return np.abs(self.array) ** 2


class Waves(Energy, ArrayWithGrid, HasCache, WavesBase):

    def __init__(self, array, extent=None, sampling=None, energy=None):
        array = np.array(array, dtype=np.complex64)

        if len(array.shape) == 2:
            array = array[None]

        if len(array.shape) != 3:
            raise RuntimeError('array must be 2d or 3d')

        HasCache.__init__(self)
        ArrayWithGrid.__init__(self, array=array, array_dimensions=3, spatial_dimensions=2, extent=extent,
                               sampling=sampling, space='direct')
        Energy.__init__(self, energy=energy)

    @cached_method
    def get_fourier_propagator(self, dz):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        kx = np.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = np.fft.fftfreq(self.gpts[1], self.sampling[1])
        return fourier_propagator((kx ** 2)[:, None] + (ky ** 2)[None], dz, self.wavelength)[None]

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

    def multislice(self, potential, in_place=False, show_progress=False):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        if in_place:
            wave = self
        else:
            wave = self.copy()

        wave.match_grid(potential)
        wave.check_is_grid_defined()
        wave.check_is_energy_defined()

        propagator = self.get_fourier_propagator(potential.slice_thickness)

        if USE_FFTW:
            temp_1 = fftw.empty_aligned(wave._array.shape, dtype='complex64')
            temp_2 = fftw.empty_aligned(wave._array.shape, dtype='complex64')
            fft_object_forward = fftw.FFTW(temp_1, temp_2, axes=(1, 2))
            fft_object_backward = fftw.FFTW(temp_2, temp_1, axes=(1, 2), direction='FFTW_BACKWARD')

        for i in range(potential.num_slices):
            potential_slice = potential.get_slice(i)

            if USE_FFTW:
                temp_1[:] = wave._array * complex_exponential(wave.sigma * potential_slice)
                temp_2[:] = fft_object_forward() * propagator
                wave._array[:] = fft_object_backward()

            else:
                wave._array[:] = np.fft.ifft2(
                    np.fft.fft2(wave._array * complex_exponential(wave.sigma * potential_slice)) * propagator)

        return wave

    def copy(self):
        new = self.__class__(array=self.array.copy(), extent=self.extent.copy(), energy=self.energy)
        return new


class PlaneWaves(Grid, Energy, HasCache, WavesBase):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None):
        HasCache.__init__(self)
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=2)
        Energy.__init__(self, energy=energy)

        self._num_waves = num_waves

    num_waves = notifying_property('_num_waves')

    def multislice(self, potential):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match_grid(potential)

        return self.build().multislice(potential, in_place=True)

    @cached_method
    def build(self):
        array = np.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=np.complex64)
        return Waves(array, extent=self.extent, energy=self.energy)


class ProbeBase(object):

    def scan(self):
        raise NotImplementedError()

    def linescan(self):
        raise NotImplementedError()

    def gridscan(self):
        raise NotImplementedError()

    def get_profile(self):
        raise NotImplementedError()


def translate(positions, kx, ky):
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))
    return complex_exponential(2 * np.pi * (kx * x + ky * y))


class A(object):

    def __init__(self):
        pass


class ProbeWaves(CTF, WavesBase):

    def __init__(self, cutoff=np.inf, rolloff=0., focal_spread=0., normalize=False, extent=None, gpts=None,
                 sampling=None, energy=None, parametrization='polar', parameters=None, **kwargs):
        CTF.__init__(self, cutoff=cutoff, rolloff=rolloff, focal_spread=focal_spread, extent=extent, gpts=gpts,
                     sampling=sampling, energy=energy, parametrization=parametrization, parameters=parameters, **kwargs)

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

    def get_a(self):
        return A()

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

    def scan(self, scan, potential, detectors, max_batch, show_progress=True):
        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self.match_grid(potential)

        if not isinstance(detectors, Iterable):
            detectors = [detectors]

        measurements = {}
        for detector in detectors:
            detector.match_grid(self)
            detector.match_energy(self)

            if not detector.export:
                measurements[detector] = np.zeros((int(np.prod(scan.gpts)),) + detector.out_shape)
                measurements[detector] = np.squeeze(measurements[detector])

        for start, stop, positions in scan.generate_positions(max_batch, show_progress=show_progress):

            waves = self.build_at(positions)

            waves = waves.multislice(potential, in_place=True, show_progress=False)

            for detector in detectors:
                if detector.export:
                    np.save(
                        detector.export + '_{}-{}in{}x{}.npy'.format(start, start + stop, scan.gpts[0], scan.gpts[1]),
                        detector.detect(waves))
                else:
                    measurements[detector][start:start + stop] = detector.detect(waves)

        for detector in detectors:
            if not detector.export:
                measurements[detector] = measurements[detector].reshape(
                    (scan.gpts[0], scan.gpts[1]) + detector.out_shape)
                measurements[detector] = np.squeeze(measurements[detector])

        return measurements

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
