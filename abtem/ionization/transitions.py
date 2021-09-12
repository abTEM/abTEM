import contextlib
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Tuple

import numpy as np
from ase import units
from ase.data import chemical_symbols

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, sph_harm

from abtem.base_classes import HasAcceleratorMixin, HasGridMixin, Grid, Accelerator, Cache, cached_method
from abtem.device import get_array_module, get_device_function
from abtem.ionization.utils import check_valid_quantum_number, config_str_to_config_tuples, \
    remove_electron_from_config_str, load_electronic_configurations
from abtem.measure import Measurement, calibrations_from_grid
from abtem.utils import energy2wavelength, spatial_frequencies, polar_coordinates, \
    relativistic_mass_correction, fourier_translation_operator
from abtem.utils import ProgressBar
from abtem.structures import SlicedAtoms


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

    def __init__(self, Z, n, l, order=1, min_contrast=1., epsilon=1, xc='PBE'):
        check_valid_quantum_number(Z, n, l)
        self._n = n
        self._l = l
        self._order = order
        self._min_contrast = min_contrast
        self._epsilon = epsilon
        self._xc = xc

        self._bound_cache = Cache(1)
        self._continuum_cache = Cache(1)
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
        return load_electronic_configurations()[chemical_symbols[self.Z]]

    @property
    def excited_configuration(self):
        return remove_electron_from_config_str(self.bound_configuration, self.n, self.l)

    @cached_method('_bound_cache')
    def _calculate_bound(self):
        from gpaw.atom.all_electron import AllElectron

        check_valid_quantum_number(self.Z, self.n, self.l)
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
            ae.run()

        wave = interp1d(ae.r, ae.u_j[subshell_index], kind='cubic', fill_value='extrapolate', bounds_error=False)
        return ae.ETotal * units.Hartree, wave

    @cached_method('_continuum_cache')
    def _calculate_continuum(self):
        from gpaw.atom.all_electron import AllElectron

        check_valid_quantum_number(self.Z, self.n, self.l)
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
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

            sqrt_k = 1 / (2 * self.epsilon / units.Hartree * (
                    1 + units.alpha ** 2 * self.epsilon / units.Hartree / 2)) ** .25
            ur = ur[:, 0] / ur[:, 0].max() / self.epsilon ** .5 * units.Rydberg ** 0.25 / sqrt_k / np.sqrt(np.pi)

            continuum_waves[lprime] = interp1d(r, ur, kind='cubic', fill_value='extrapolate', bounds_error=False)
        return ae.ETotal * units.Hartree, continuum_waves

    def get_bound_wave(self):
        return self._calculate_bound()[1]

    def get_continuum_waves(self):
        return self._calculate_continuum()[1]

    def get_transition_quantum_numbers(self):
        transitions = []
        for ml in np.arange(-self.l, self.l + 1):
            for new_l in self.lprimes:
                for new_ml in np.arange(-new_l, new_l + 1):
                    if not abs(new_l - self.l) == 1:
                        continue

                    if not (abs(ml - new_ml) < 2):
                        continue

                    transitions.append([(self.l, ml), (new_l, new_ml)])
        return transitions

    def get_transition_potentials(self,
                                  extent: Union[float, Sequence[float]] = None,
                                  gpts: Union[float, Sequence[float]] = None,
                                  sampling: Union[float, Sequence[float]] = None,
                                  energy: float = None,
                                  pbar=True):

        transitions = []
        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Transitions', disable=(not pbar))

        _, bound_wave = self._calculate_bound()
        _, continuum_waves = self._calculate_continuum()
        energy_loss = self.energy_loss

        for bound_state, continuum_state in self.get_transition_quantum_numbers():
            continuum_wave = continuum_waves[continuum_state[0]]

            transition = ProjectedAtomicTransition(Z=self.Z,
                                                   bound_wave=bound_wave,
                                                   continuum_wave=continuum_wave,
                                                   bound_state=bound_state,
                                                   continuum_state=continuum_state,
                                                   energy_loss=energy_loss,
                                                   extent=extent,
                                                   gpts=gpts,
                                                   sampling=sampling,
                                                   energy=energy
                                                   )
            transitions += [transition]
            pbar.update(1)

        pbar.refresh()
        pbar.close()
        return transitions


class AbstractProjectedAtomicTransition(HasAcceleratorMixin, HasGridMixin):

    def __init__(self, Z, extent, gpts, sampling, energy):
        self._Z = Z
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)


class ProjectedAtomicTransition(AbstractProjectedAtomicTransition):

    def __init__(self,
                 Z: int,
                 bound_wave: callable,
                 continuum_wave: callable,
                 bound_state: Tuple[int, int],
                 continuum_state: Tuple[int, int],
                 energy_loss: float = 1.,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None):

        self._bound_wave = bound_wave
        self._continuum_wave = continuum_wave
        self._bound_state = bound_state
        self._continuum_state = continuum_state
        self._energy_loss = energy_loss

        self._cache = Cache(1)

        super().__init__(Z, extent, gpts, sampling, energy)

    def __str__(self):
        return (f'{self._bound_state} -> {self._continuum_state}')

    @property
    def energy_loss(self):
        return self._energy_loss

    @property
    def momentum_transfer(self):
        k0 = 1 / energy2wavelength(self.energy)
        kn = 1 / energy2wavelength(self.energy + self.energy_loss)
        return k0 - kn

    def _fourier_translation_operator(self, positions):
        return fourier_translation_operator(positions, self.gpts)

    def build(self, positions=None):
        if positions is None:
            positions = np.zeros((1, 2), dtype=np.float32)
        else:
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        positions /= self.sampling
        potential = np.fft.ifft2(self._evaluate_potential() * self._fourier_translation_operator(positions))
        return potential

    def calculate_total_intensity(self):
        return (np.abs(self.build()) ** 2).sum()

    def overlap_integral(self, k, lprimeprime):
        rmax = self._bound_wave.x[-1]
        rmax = 20
        # rmax = max(self._bound_wave[1])
        grid = 2 * np.pi * k * units.Bohr
        r = np.linspace(0, rmax, 10000)

        values = (self._bound_wave(r) *
                  spherical_jn(lprimeprime, grid[:, None] * r[None]) *
                  self._continuum_wave(r))
        return np.trapz(values, r, axis=1)

    @cached_method('_cache')
    def _evaluate_potential(self):
        from sympy.physics.wigner import wigner_3j

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        potential = np.zeros(self.gpts, dtype=np.complex64)

        kx, ky = spatial_frequencies(self.gpts, self.sampling)
        kz = self.momentum_transfer

        kt, phi = polar_coordinates(kx, ky)
        k = np.sqrt(kt ** 2 + kz ** 2)
        theta = np.pi - np.arctan(kt / kz)

        radial_grid = np.arange(0, np.max(k) * 1.05, 1 / max(self.extent))

        l, ml = self._bound_state
        lprime, mlprime = self._continuum_state

        for lprimeprime in range(max(l - lprime, 0), np.abs(l + lprime) + 1):
            prefactor1 = (np.sqrt(4 * np.pi) * ((-1.j) ** lprimeprime) *
                          np.sqrt((2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * l + 1)))
            jk = None

            for mlprimeprime in range(-lprimeprime, lprimeprime + 1):

                if ml - mlprime - mlprimeprime != 0:  # Wigner3j selection rule
                    continue

                # Evaluate Eq. (14) from Dwyer Ultramicroscopy 104 (2005) 141-151
                prefactor2 = ((-1.0) ** (mlprime + mlprimeprime)
                              * float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))
                              * float(wigner_3j(lprime, lprimeprime, l, -mlprime, -mlprimeprime, ml)))

                if np.abs(prefactor2) < 1e-12:
                    continue

                if jk is None:
                    jk = interp1d(radial_grid, self.overlap_integral(radial_grid, lprimeprime))(k)

                Ylm = sph_harm(mlprimeprime, lprimeprime, phi, theta)
                potential += prefactor1 * prefactor2 * jk * Ylm

        potential *= np.prod(self.gpts) / np.prod(self.extent)

        # Multiply by orbital filling
        # if orbital_filling_factor:
        potential *= np.sqrt(4 * l + 2)

        # Apply constants:
        # sqrt(Rdyberg) to convert to 1/sqrt(eV) units
        # 1 / (2 pi**2 a0 kn) as as per paper
        # Relativistic mass correction to go from a0 to relativistically corrected a0*
        # divide by q**2

        kn = 1 / energy2wavelength(self.energy + self.energy_loss)

        potential *= relativistic_mass_correction(self.energy) / (
                2 * units.Bohr * np.pi ** 2 * np.sqrt(units.Rydberg) * kn * k ** 2
        )
        return potential

    def measure(self):
        array = np.fft.fftshift(self.build())[0]
        calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        abs2 = get_device_function(get_array_module(array), 'abs2')
        return Measurement(array, calibrations, name=str(self))

    def show(self, ax, **kwargs):
        # array = np.fft.fftshift(self.build())[0]
        # calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        # abs2 = get_device_function(get_array_module(array), 'abs2')
        self.measure().show(ax=ax)
        # Measurement(abs2(array), calibrations, name=str(self)).show(**kwargs)


class TransitionPotential(HasAcceleratorMixin, HasGridMixin):

    def __init__(self,
                 transitions,
                 atoms=None,
                 slice_thickness=None,
                 gpts: Union[int, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 min_contrast=.95):

        if isinstance(transitions, SubshellTransitions):
            transitions = [transitions]

        self._slice_thickness = slice_thickness

        self._grid = Grid(gpts=gpts, sampling=sampling)

        self.atoms = atoms
        self._transitions = transitions

        self._accelerator = Accelerator(energy=energy)

        self._sliced_atoms = SlicedAtoms(atoms, slice_thicknesses=self._slice_thickness)

        self._potentials_cache = Cache(1)

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms

        if atoms is not None:
            self.extent = np.diag(atoms.cell)[:2]
            self._sliced_atoms = SlicedAtoms(atoms, slice_thicknesses=self._slice_thickness)
        else:
            self._sliced_atoms = None

    @property
    def num_edges(self):
        return len(self._transitions)

    @property
    def num_slices(self):
        return self._sliced_atoms.num_slices

    @cached_method('_potentials_cache')
    def _calculate_potentials(self, transitions_idx):
        transitions = self._transitions[transitions_idx]
        return transitions.get_transition_potentials(extent=self.extent, gpts=self.gpts, energy=self.energy, pbar=False)

    def _generate_slice_transition_potentials(self, slice_idx, transitions_idx):
        transitions = self._transitions[transitions_idx]
        Z = transitions.Z

        atoms_slice = self._sliced_atoms.get_subsliced_atoms(slice_idx, atomic_number=Z).atoms

        for transition in self._calculate_potentials(transitions_idx):
            for atom in atoms_slice:
                t = np.asarray(transition.build(atom.position[:2]))
                yield t

    def show(self, transitions_idx=0):
        intensity = None

        if self._sliced_atoms.slice_thicknesses is None:
            none_slice_thickess = True
            self._sliced_atoms.slice_thicknesses = self._sliced_atoms.atoms.cell[2, 2]
        else:
            none_slice_thickess = False

        for slice_idx in range(self.num_slices):
            for t in self._generate_slice_transition_potentials(slice_idx, transitions_idx):
                if intensity is None:
                    intensity = np.abs(t) ** 2
                else:
                    intensity += np.abs(t) ** 2

        if none_slice_thickess:
            self._sliced_atoms.slice_thicknesses = None

        calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        Measurement(intensity[0], calibrations, name=str(self)).show()
