import numexpr as ne
import numpy as np
from ase import Atoms

from .bases import cached_method, HasCache, ArrayWithGrid, Grid, Energy, notifying_property
from .potentials import Potential


def complex_exponential(x):
    return ne.evaluate('exp(1.j * x)')


def fourier_propagator(k, dz, wavelength):
    return complex_exponential(-k * np.pi * wavelength * dz)


class Waves(Energy, ArrayWithGrid, HasCache):

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

    def check_is_grid_and_energy_defined(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()

    def match_grid_and_energy(self, other):
        self.match_grid(other)
        self.match_energy(other)

    @cached_method
    def get_fourier_propagator(self, dz):
        self.check_is_grid_and_energy_defined()
        kx = np.fft.fftfreq(self.gpts[1], self.sampling[1])
        ky = np.fft.fftfreq(self.gpts[0], self.sampling[0])
        return fourier_propagator((kx ** 2)[None] + (ky ** 2)[:, None], dz, self.wavelength)[None]

    def multislice(self, potential, in_place=False):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        if in_place:
            wave = self
        else:
            wave = self.copy()

        wave.match_grid(potential)

        wave.check_is_grid_and_energy_defined()
        potential.check_is_grid_defined()

        propagator = self.get_fourier_propagator(potential.slice_thickness)

        for i in range(potential.num_slices):
            potential_slice = potential.get_slice(i)

            wave._array[:, :, :] = np.fft.ifft2(
                np.fft.fft2(wave._array * complex_exponential(wave.sigma * potential_slice)) * propagator)

        return wave

    def copy(self):
        new = self.__class__(array=self.array.copy(), extent=self.extent.copy(), energy=self.energy)
        return new


class WaveFactory(Grid, Energy, HasCache):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        HasCache.__init__(self)
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=2)
        Energy.__init__(self, energy=energy)

    def check_is_grid_and_energy_defined(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()

    def match_grid_and_energy(self, other):
        self.match_grid(other)
        self.match_energy(other)

    def build(self):
        raise NotImplementedError()

    def multislice(self, potential):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match_grid_and_energy(potential)

        return self.build().multislice(potential, in_place=True)


class PlaneWaves(WaveFactory):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)

        self._num_waves = num_waves

    num_waves = notifying_property('_num_waves')

    @cached_method
    def build(self):
        array = np.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=np.complex64)
        return Waves(array, extent=self.extent, energy=self.energy)
