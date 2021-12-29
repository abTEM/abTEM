import contextlib
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Tuple, List

import numpy as np
from ase import Atoms
from ase import units
from ase.data import chemical_symbols
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, sph_harm

from abtem.core.complex import abs2
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2wavelength, relativistic_mass_correction
from abtem.core.fft import fft_shift_kernel
from abtem.core.fft import ifft2
from abtem.core.grid import HasGridMixin, Grid, polar_spatial_frequencies
from abtem.ionization.electron_configurations import electron_configurations
from abtem.ionization.utils import check_valid_quantum_number, config_str_to_config_tuples, \
    remove_electron_from_config_str
from abtem.measure.measure import Images


class AbstractTransitionCollection(metaclass=ABCMeta):

    def __init__(self, Z):
        self._Z = Z

    @property
    def Z(self):
        return self._Z

    @abstractmethod
    def get_transition_potentials(self):
        pass


class SubshellTransitions(AbstractTransitionCollection):

    def __init__(self, Z: int, n: int, l: int, order: int = 1, min_contrast: float = 1., epsilon: float = 1.,
                 xc: str = 'PBE'):
        check_valid_quantum_number(Z, n, l)
        self._n = n
        self._l = l
        self._order = order
        self._min_contrast = min_contrast
        self._epsilon = epsilon
        self._xc = xc
        super().__init__(Z)

    @property
    def order(self):
        return self._order

    @property
    def min_contrast(self):
        return self._min_contrast

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def xc(self):
        return self._xc

    @property
    def n(self):
        return self._n

    @property
    def l(self):
        return self._l

    @property
    def lprimes(self):
        min_new_l = max(self.l - self.order, 0)
        return np.arange(min_new_l, self.l + self.order + 1)

    def __len__(self):
        return len(self.get_transition_quantum_numbers())

    @property
    def ionization_energy(self):
        atomic_energy, _ = self._calculate_bound()
        ionic_energy, _ = self._calculate_continuum()
        return ionic_energy - atomic_energy

    @property
    def energy_loss(self):
        atomic_energy, _ = self._calculate_bound()
        ionic_energy, _ = self._calculate_continuum()
        return self.ionization_energy + self.epsilon

    @property
    def bound_configuration(self):
        return electron_configurations[chemical_symbols[self.Z]]

    @property
    def excited_configuration(self):
        return remove_electron_from_config_str(self.bound_configuration, self.n, self.l)

    def _calculate_bound(self):
        from gpaw.atom.all_electron import AllElectron

        check_valid_quantum_number(self.Z, self.n, self.l)
        config_tuples = config_str_to_config_tuples(electron_configurations[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
            ae.run()

        return ae.ETotal * units.Hartree, (ae.r * units.Bohr, ae.u_j[subshell_index])

    def _calculate_continuum(self):
        from gpaw.atom.all_electron import AllElectron

        check_valid_quantum_number(self.Z, self.n, self.l)
        config_tuples = config_str_to_config_tuples(electron_configurations[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
            ae.f_j[subshell_index] -= 1.
            ae.run()

        vr = interp1d(ae.r, -ae.vr, fill_value='extrapolate', bounds_error=False)

        def schroedinger_derivative(y, r, l, e, vr):
            (u, up) = y
            return np.array([up, (l * (l + 1) / r ** 2 - 2 * vr(r) / r - e) * u])

        r = np.geomspace(1e-7, 200, 1000000)
        continuum_waves = {}
        for lprime in self.lprimes:
            ur = integrate.odeint(schroedinger_derivative, [0.0, 1.], r, args=(lprime, self.epsilon, vr))

            sqrt_k = self.epsilon ** .5 / (
                    self.epsilon * (1 + units.alpha ** 2 * self.epsilon / units.Rydberg / 4)) ** .25

            ur = ur[:, 0] / ur[:, 0].max() / sqrt_k / np.sqrt(np.pi)

            continuum_waves[lprime] = r * units.Bohr, ur
        return ae.ETotal * units.Hartree, continuum_waves

    def get_bound_wave(self):
        return self._calculate_bound()[1]

    def get_continuum_waves(self):
        return self._calculate_continuum()[1]

    def get_transition_quantum_numbers(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        transitions = []
        for ml in np.arange(-self.l, self.l + 1):
            for new_l in self.lprimes:
                for new_ml in np.arange(-new_l, new_l + 1):
                    if not abs(new_l - self.l) == 1:
                        continue

                    if not (abs(ml - new_ml) < 2):
                        continue

                    transitions.append(((self.l, ml), (new_l, new_ml)))
        return transitions

    def as_arrays(self):
        _, bound_wave = self._calculate_bound()
        _, continuum_waves = self._calculate_continuum()

        bound_state = self.get_transition_quantum_numbers()[0][0]
        continuum_states = [state[1] for state in self.get_transition_quantum_numbers()]
        _, continuum_waves = self._calculate_continuum()

        arrays = SubshellTransitionsArrays(Z=self.Z,
                                           bound_wave=bound_wave,
                                           continuum_waves=continuum_waves,
                                           bound_state=bound_state,
                                           continuum_states=continuum_states,
                                           energy_loss=self.energy_loss,
                                           )

        return arrays

    def get_transition_potentials(self,
                                  extent: Union[float, Sequence[float]] = None,
                                  gpts: Union[float, Sequence[float]] = None,
                                  sampling: Union[float, Sequence[float]] = None,
                                  energy: float = None):

        _, bound_wave = self._calculate_bound()
        _, continuum_waves = self._calculate_continuum()
        energy_loss = self.energy_loss

        quantum_numbers = self.get_transition_quantum_numbers()

        transition_potential = SubshellTransitionPotentials(Z=self.Z,
                                                            bound_wave=bound_wave,
                                                            continuum_waves=continuum_waves,
                                                            quantum_numbers=quantum_numbers,
                                                            energy_loss=energy_loss,
                                                            extent=extent,
                                                            gpts=gpts,
                                                            sampling=sampling,
                                                            energy=energy
                                                            )

        return transition_potential


class AbstractTransitionPotential(HasAcceleratorMixin, HasGridMixin):

    def __init__(self, Z, extent, gpts, sampling, energy):
        self._Z = Z
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

    @property
    def Z(self):
        return self._Z


class SubshellTransitionPotentials(AbstractTransitionPotential):

    def __init__(self,
                 Z: int,
                 bound_wave: callable,
                 continuum_waves: dict,
                 quantum_numbers: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                 energy_loss: float = 1.,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None):

        self._bound_wave = bound_wave
        self._continuum_waves = continuum_waves
        self._quantum_numbers = quantum_numbers
        self._energy_loss = energy_loss
        super().__init__(Z, extent, gpts, sampling, energy)

        self._array = None

        def clear_data(*args):
            self._array = None

        self.grid.observe(clear_data, ('sampling', 'gpts', 'extent'))
        self.accelerator.observe(clear_data, ('energy',))

    @property
    def energy_loss(self):
        return self._energy_loss

    @property
    def bound_wave(self):
        return self._bound_wave

    @property
    def continuum_waves(self):
        return self._continuum_waves

    @property
    def quantum_numbers(self):
        return self._quantum_numbers

    # def __str__(self):
    #    return f'{self._bound_state} -> {self._continuum_state}'

    @property
    def array(self):
        if self._array is None:
            self._array = self._calculate_array()

        return self._array

    @property
    def total_intensity(self):
        return (np.abs(self.array) ** 2).sum()

    @property
    def momentum_transfer(self):
        k0 = 1 / energy2wavelength(self.energy)
        kn = 1 / energy2wavelength(self.energy + self.energy_loss)
        return k0 - kn

    def _validate_sites(self, sites):
        if isinstance(sites, Atoms):
            sites = sites[sites.numbers == self.Z].positions[:, :2]

        sites = np.array(sites, dtype=np.float32)
        return sites

    def scatter(self, waves, sites):
        self.grid.match(waves)
        self.accelerator.match(waves)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        positions = self._validate_sites(sites)

        # if positions is None:
        #     positions = np.zeros((1, 2), dtype=np.float32)
        # else:
        #     positions = np.array(positions, dtype=np.float32)
        #
        # if len(positions.shape) == 1:
        #     positions = np.expand_dims(positions, axis=0)

        array = self.array[None]

        positions /= self.sampling
        array = ifft2(array * fft_shift_kernel(positions, self.gpts)[:, None])

        array = array.reshape((-1,) + array.shape[-2:])
        # sss
        d = waves._copy_as_dict(copy_array=False)
        d['array'] = array * waves.array
        d['extra_axes_metadata'] = [{'type': 'ensemble', 'label': 'core ionization'}]
        return waves.__class__(**d)

    def generate_scatter(self, waves, sites, chunks):
        pass

    def _calculate_overlap_integral(self, lprime, lprimeprime, k):
        radial_grid = np.arange(0, np.max(k) * 1.05, 1 / max(self.extent))
        integration_grid = np.linspace(0, self._bound_wave[0][-1], 10000)

        bound_wave = interp1d(*self._bound_wave, kind='cubic', fill_value='extrapolate', bounds_error=False)
        continuum_wave = interp1d(*self._continuum_waves[lprime], kind='cubic', fill_value='extrapolate',
                                  bounds_error=False)

        values = (bound_wave(integration_grid) *
                  spherical_jn(lprimeprime, radial_grid[:, None] * integration_grid[None]) *
                  continuum_wave(integration_grid))

        integral = np.trapz(values, integration_grid, axis=1)

        return interp1d(radial_grid, integral)(k)

    def _calculate_array(self):
        from sympy.physics.wigner import wigner_3j

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        array = np.zeros((len(self.quantum_numbers),) + self.gpts, dtype=np.complex64)
        kz = self.momentum_transfer

        kt, phi = polar_spatial_frequencies(self.gpts, self.sampling)
        k = np.sqrt(kt ** 2 + kz ** 2) * 2 * np.pi
        theta = np.pi - np.arctan(kt / kz)

        for i, ((l, ml), (lprime, mlprime)) in enumerate(self.quantum_numbers):

            for lprimeprime in range(max(l - lprime, 0), abs(l + lprime) + 1):
                overlap_integral = self._calculate_overlap_integral(lprime, lprimeprime, k)

                for mlprimeprime in range(-lprimeprime, lprimeprime + 1):

                    if ml - mlprime - mlprimeprime != 0:
                        continue

                    prefactor = (np.sqrt(4 * np.pi) * ((-1.j) ** lprimeprime) *
                                 np.sqrt((2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * l + 1)))

                    prefactor *= ((-1.0) ** (mlprime + mlprimeprime)
                                  * float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))
                                  * float(wigner_3j(lprime, lprimeprime, l, -mlprime, -mlprimeprime, ml)))

                    if np.abs(prefactor) < 1e-12:
                        continue

                    Ylm = sph_harm(mlprimeprime, lprimeprime, phi, theta)
                    array[i] += prefactor * overlap_integral * Ylm

            array[i] *= np.sqrt(4 * l + 2)

        array *= np.prod(self.gpts) / np.prod(self.extent)

        kn = 1 / energy2wavelength(self.energy + self.energy_loss)

        y = relativistic_mass_correction(self.energy)

        # matrix_element_const = 4 * np.pi * units._e ** 2 / k ** 2
        # interaction_constant = y * units._me / (2 * np.pi * units._hbar ** 2 * kn)
        # print(interaction_constant)
        # bohr = 4 * np.pi * units._eps0 * units._hbar / (units._me * units._e ** 2)
        # rydberg = units._me * units._e ** 4 / (8 * units._eps0 ** 2 * units._hplanck ** 2) / units._e
        # rydberg = 1 / 2 * units._e ** 2 / (4 * np.pi * units._eps0 * units.Bohr)
        # bohr ** 2 * rydberg = 4 * np.pi
        # 1 / (2 * units._me) * (units._hbar * 1e10)**2 / units._e
        # inter = relativistic_mass_correction(self.energy) *

        array *= y * np.sqrt(units.Rydberg) / (2 * units.Bohr ** 2 * units.Rydberg * kn * k ** 2 / 4)

        return array

    def image(self):
        array = np.fft.fftshift(ifft2(self.array))
        array = abs2(array)
        return Images(array, sampling=self.sampling)

    def show(self, **kwargs):
        self.image().show(**kwargs)
