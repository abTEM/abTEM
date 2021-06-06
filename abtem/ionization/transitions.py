import contextlib
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Tuple

import numpy as np
from ase import units
from ase.data import chemical_symbols
from gpaw.atom.all_electron import AllElectron
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, sph_harm
from sympy.physics.wigner import wigner_3j

from abtem.base_classes import HasAcceleratorMixin, HasGridMixin, Grid, Accelerator, Cache, cached_method
from abtem.device import get_array_module, get_device_function
from abtem.ionization.utils import check_valid_quantum_number, config_str_to_config_tuples, \
    remove_electron_from_config_str, load_electronic_configurations
from abtem.measure import Measurement, calibrations_from_grid
from abtem.utils import energy2wavelength, spatial_frequencies, polar_coordinates, \
    relativistic_mass_correction, fourier_translation_operator


def get_gpaw_bound_wave(Z, n, ell, xc='PBE'):
    # _, a = get_fac_bound_wave(Z,n, ell)

    check_valid_quantum_number(Z, n, ell)
    config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[Z]])

    i = [shell[:2] for shell in config_tuples].index((n, ell))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ae = AllElectron(chemical_symbols[Z], xcname=xc, gpernode=500)
        ae.run()

    return ae.e_j[i] * units.Hartree, interp1d(ae.r, ae.u_j[i], kind='cubic', fill_value=0, bounds_error=False)


def schroedinger_derivative(y, r, l, e, vr):
    (u, up) = y
    return np.array([up, (l * (l + 1) / r ** 2 - 2 * vr(r) / r - e) * u])


def solve_schroedinger_continuum(ell, epsilon, r, vr):
    ur = integrate.odeint(schroedinger_derivative, [0.0, 1.], r, args=(ell, epsilon, vr))
    return ur[:, 0]


def get_gpaw_continuum_wave(Z, n, ell, ellprime, epsilon, xc='PBE'):
    check_valid_quantum_number(Z, n, ell)

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ae = AllElectron(chemical_symbols[Z], corehole=(n, ell, 1.), xcname=xc, gpernode=500)
        ae.run()

    vr = ae.vr
    vr = interp1d(ae.r, -vr, fill_value='extrapolate', bounds_error=False)

    r = np.geomspace(1e-7, 200, 1000000)
    ur = integrate.odeint(schroedinger_derivative, [0.0, 1.], r, args=(ellprime, epsilon, vr))
    ur = ur[:, 0]

    ke = 1 / (2 * epsilon / units.Hartree * (1 + units.alpha ** 2 * epsilon / units.Hartree / 2)) ** .25
    ur = ur / ur.max() / epsilon ** .5 * units.Rydberg ** 0.25 / ke / np.sqrt(np.pi)

    return interp1d(r, ur, kind='cubic', fill_value=0, bounds_error=False)


class AbstractTransitionCollection(metaclass=ABCMeta):

    @abstractmethod
    def get_transition_potentials(self):
        pass


class SubshellTransitions(AbstractTransitionCollection):

    def __init__(self, Z, n, ell, order=1, min_contrast=.9, epsilon=1):
        check_valid_quantum_number(Z, n, ell)
        self._Z = Z
        self._n = n
        self._ell = ell
        self.order = order
        self.min_contrast = min_contrast
        self.epsilon = epsilon

    @property
    def Z(self):
        return self._Z

    @property
    def n(self):
        return self._n

    @property
    def ell(self):
        return self._ell

    def __len__(self):
        return len(self.get_transition_quantum_numbers())

    @property
    def bound_configuration(self):
        return load_electronic_configurations()[chemical_symbols[self.Z]]

    @property
    def excited_configuration(self):
        return remove_electron_from_config_str(self.bound_configuration, self.n, self.ell)

    def get_transition_quantum_numbers(self):
        min_new_ell = max(self.ell - self.order, 0)
        transitions = []
        for ml in np.arange(-self.ell, self.ell + 1):
            for new_ell in np.arange(min_new_ell, self.ell + self.order + 1):
                for new_ml in np.arange(-new_ell, new_ell + 1):
                    transitions.append([(self.ell, ml), (new_ell, new_ml)])
        return transitions

    def get_transition_potentials(self,
                                  extent: Union[float, Sequence[float]] = None,
                                  gpts: Union[float, Sequence[float]] = None,
                                  sampling: Union[float, Sequence[float]] = None,
                                  energy: float = None):
        transitions = []
        for bound_state, continuum_state in self.get_transition_quantum_numbers():
            transition = TransitionPotential(self.Z,
                                             self.n,
                                             bound_state,
                                             continuum_state,
                                             self.epsilon,
                                             extent=extent,
                                             gpts=gpts,
                                             sampling=sampling,
                                             energy=energy)
            transitions += [transition]

        return transitions


class TransitionPotential(HasAcceleratorMixin, HasGridMixin):

    def __init__(self,
                 Z: int,
                 n: int,
                 bound_state: Tuple[int, int],
                 continuum_state: Tuple[int, int],
                 epsilon: float = 1.,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 wave_calculator='fac_data'):

        self._Z = Z
        self._n = n

        if (bound_state[1] > bound_state[0]) or (continuum_state[1] > continuum_state[0]):
            raise ValueError()

        self._bound_state = bound_state
        self._continuum_state = continuum_state
        self._epsilon = epsilon

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

        # self._bound_energy, self._bound_wave = get_fac_bound_wave(self.Z, self.n, self._bound_state[0])
        # self._continuum_wave = get_fac_continuum_wave(self.Z, self.n, self._bound_state[0], self._continuum_state[0])

        self._bound_energy, self._bound_wave = get_gpaw_bound_wave(self.Z, self.n, self._bound_state[0])
        self._continuum_wave = get_gpaw_continuum_wave(self.Z, self.n, self._bound_state[0], self._continuum_state[0],
                                                       epsilon)

        self._continuum_energy = epsilon
        # self._load_bound_fac_data()
        # self._load_continuum_fac_data()

        self._cache = Cache(1)

    def __str__(self):
        return (f'{self._bound_state} -> {self._continuum_state}')

    @property
    def Z(self):
        return self._Z

    @property
    def n(self):
        return self._n

    @property
    def epsilon(self):
        return self._epsilon

    def energy_loss(self):
        return self._bound_energy - self._continuum_energy

    def momentum_transfer(self, energy):
        k0 = 1 / energy2wavelength(energy)
        kn = 1 / energy2wavelength(energy + self.energy_loss())
        return k0 - kn

    # def _load_bound_fac_data(self):
    #     symbol = chemical_symbols[self.Z]
    #     ell, ml = self._bound_state
    #     fname = f'fac/{symbol}/{symbol}_bound_n={self.n}_l={ell}.txt'
    #     header_data, table = load_fac_data(fname)
    #     self._bound_energy = header_data['energy']
    #     self._bound_wave = (table[:, 1], table[:, 4])
    #
    # def _load_continuum_fac_data(self):
    #     symbol = chemical_symbols[self.Z]
    #     ell, ml = self._bound_state
    #     ellprime, ml = self._continuum_state
    #
    #     fname = f'fac/{symbol}/{symbol}_excited_n={self.n}_l={ell}_lprime={ellprime}.txt'
    #     header_data, table = load_fac_data(fname)
    #     ilast = header_data['ilast']
    #     self._continuum_energy = header_data['energy']
    #
    #     epsilon = self._continuum_energy
    #
    #     # Normalization from Flexible Atomic Code
    #     norm_fac = 1 / (2 * (epsilon / units.Hartree) * (1 + units.alpha ** 2 * (epsilon / units.Hartree) / 2)) ** .25
    #
    #     # Normalization from Manson 1972
    #     norm_manson = 1 / np.sqrt(np.pi) / (epsilon / units.Rydberg) ** 0.25
    #
    #     continuum_wave = np.zeros_like(table[:, 1])
    #     continuum_wave[:ilast + 1] = table[:ilast + 1, 4] / norm_fac * norm_manson
    #     amplitude = table[ilast + 1:, 2] / norm_fac * norm_manson
    #     phase = table[ilast + 1:, 3]
    #
    #     continuum_wave[ilast + 1:] = amplitude * np.sin(phase)
    #     self._continuum_wave = (table[:, 1], continuum_wave)

    def evaluate_bound_wave(self, r):
        return self._bound_wave(r)

    def evaluate_continuum_wave(self, r):

        # return interp1d(*self._continuum_wave, kind='cubic', fill_value=0, bounds_error=False)(r)
        return self._continuum_wave(r)

    def overlap_integral(self, k, ellprimeprime):
        rmax = self._bound_wave.x[-1]
        rmax = 20
        # rmax = max(self._bound_wave[1])
        grid = 2 * np.pi * k * units.Bohr
        r = np.linspace(0, rmax, 1000)
        values = (self.evaluate_bound_wave(r) *
                  spherical_jn(ellprimeprime, grid[:, None] * r[None]) *
                  self.evaluate_continuum_wave(r))
        return np.trapz(values, r, axis=1)

    def _fourier_translation_operator(self, positions):
        return fourier_translation_operator(positions, self.gpts, self.sampling)

    def build(self, positions=None):
        if positions is None:
            positions = np.zeros((1, 2), dtype=np.float32)
        else:
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        positions /= self.sampling

        # potential = np.fft.fft2(self._evaluate_potential() * self._fourier_translation_operator(positions))
        return self._evaluate_potential()
        return potential

    @cached_method('_cache')
    def _evaluate_potential(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        potential = np.zeros(self.gpts, dtype=np.complex64)

        k0 = 1 / energy2wavelength(self.energy)
        kn = 1 / energy2wavelength(self.energy + self.energy_loss())

        kx, ky = spatial_frequencies(self.gpts, self.sampling)
        kz = self.momentum_transfer(self.energy)

        kt, phi = polar_coordinates(kx, ky)
        k = np.sqrt(kt ** 2 + kz ** 2)
        theta = np.pi - np.arctan(kt / kz)

        kmax = np.max(k)
        radial_grid = np.arange(0, kmax * 1.05, 1 / max(self.extent))

        ell, ml = self._bound_state
        ellprime, mlprime = self._continuum_state

        for ellprimeprime in range(max(ell - ellprime, 0), np.abs(ell + ellprime) + 1):
            prefactor1 = (np.sqrt(4 * np.pi) * ((-1.j) ** ellprimeprime) *
                          np.sqrt((2 * ellprime + 1) * (2 * ellprimeprime + 1) * (2 * ell + 1)))
            jk = None

            for mlprimeprime in range(-ellprimeprime, ellprimeprime + 1):
                # Check second selection rule
                # (http://mathworld.wolfram.com/Wigner3j-Symbol.html)
                if ml - mlprime - mlprimeprime != 0:
                    continue

                # evaluate Eq. (14) from Dwyer Ultramicroscopy 104 (2005) 141-151
                prefactor2 = ((-1.0) ** (mlprime + mlprimeprime)
                              * np.float(wigner_3j(ellprime, ellprimeprime, ell, 0, 0, 0))
                              * np.float(wigner_3j(ellprime, ellprimeprime, ell, -mlprime, -mlprimeprime, ml)))

                if np.abs(prefactor2) > 1e-12:
                    if jk is None:
                        jk = interp1d(radial_grid, self.overlap_integral(radial_grid, ellprimeprime))(k)

                    Ylm = sph_harm(mlprimeprime, ellprimeprime, phi, theta)
                    potential += prefactor1 * prefactor2 * jk * Ylm

        potential *= np.prod(self.gpts) / np.prod(self.extent)

        # Multiply by orbital filling
        # if orbital_filling_factor:
        potential *= np.sqrt(4 * ell + 2)

        # Apply constants:
        # sqrt(Rdyberg) to convert to 1/sqrt(eV) units
        # 1 / (2 pi**2 a0 kn) as as per paper
        # Relativistic mass correction to go from a0 to relativistically corrected a0*
        # divide by q**2
        potential *= relativistic_mass_correction(self.energy) / (
                2 * units.Bohr * np.pi ** 2 * np.sqrt(units.Rydberg) * kn * k ** 2
        )
        # potential = np.fft.ifft2(potential)
        return potential

    def show(self, **kwargs):
        array = np.fft.fftshift(np.fft.ifft2(self._evaluate_potential()))
        calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        abs2 = get_device_function(get_array_module(array), 'abs2')
        Measurement(abs2(array), calibrations, name=str(self)).show(**kwargs)
