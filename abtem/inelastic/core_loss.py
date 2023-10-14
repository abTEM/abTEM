from __future__ import annotations
import contextlib
import itertools
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Tuple, List

from ase import Atom
from ase import units
from numba import jit
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, sph_harm
import numpy as np
from abtem.core.axes import OrdinalAxis
from abtem.core.energy import (
    HasAcceleratorMixin,
    Accelerator,
    energy2wavelength,
    relativistic_mass_correction,
)
from abtem.core.fft import fft_shift_kernel
from abtem.core.fft import ifft2
from abtem.core.grid import HasGridMixin, Grid, polar_spatial_frequencies

from abtem.core.chunks import generate_chunks

from ase.data import chemical_symbols

from abtem.core.electron_configurations import electron_configurations

azimuthal_number = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
azimuthal_letter = {value: key for key, value in azimuthal_number.items()}


def config_str_to_config_tuples(config_str):
    config_tuples = []
    for subshell_string in config_str.split(" "):
        config_tuples.append(
            (
                int(subshell_string[0]),
                azimuthal_number[subshell_string[1]],
                int(subshell_string[2]),
            )
        )
    return config_tuples


def config_tuples_to_config_str(config_tuples):
    config_str = []
    for n, ell, occ in config_tuples:
        config_str.append(str(n) + azimuthal_letter[ell] + str(occ))
    return " ".join(config_str)


def remove_electron_from_config_str(config_str, n, ell):
    config_tuples = []
    for shell in config_str_to_config_tuples(config_str):
        if shell[:2] == (n, ell):
            config_tuples.append(shell[:2] + (shell[2] - 1,))
        else:
            config_tuples.append(shell)
    return config_tuples_to_config_str(config_tuples)


def check_valid_quantum_number(Z, n, ell):
    symbol = chemical_symbols[Z]
    config_tuple = config_str_to_config_tuples(electron_configurations[symbol])

    if not any([shell[:2] == (n, ell) for shell in config_tuple]):
        raise RuntimeError(
            f"Quantum numbers (n, ell) = ({n}, {ell}) not valid for element {symbol}"
        )


class RadialWavefunction:
    def __init__(
        self,
        n: int,
        l: int,
        energy: float,
        radial_grid: np.ndarray,
        radial_values: np.ndarray,
    ):
        self._n = n
        self._l = l
        self._energy = energy
        self._radial_grid = radial_grid
        self._radial_values = radial_values

    def __call__(self, r):
        f = interp1d(
            self._radial_grid,
            self._radial_values,
            kind=2,
            fill_value="extrapolate",
        )
        return f(r)

    @property
    def bound(self):
        return self._n > 0

    @property
    def energy(self):
        return self._energy

    @property
    def radial_grid(self):
        return self._radial_grid

    @property
    def n(self):
        return self._n

    @property
    def l(self):
        return self._l


@jit(nopython=True)
def numerov(f, x0, dx, dh):
    """Given precomputed function f(x), solves for x(t), which satisfies:
    x''(t) = f(t) x(t)
    """
    x = np.zeros(len(f))
    x[0] = x0
    x[1] = x0 + dh * dx
    h2 = dh**2
    h12 = h2 / 12.0
    w0 = x0 * (1 - h12 * f[0])
    w1 = x[1] * (1 - h12 * f[1])
    xi = x[1]
    fi = f[1]
    for i in range(2, f.size):
        w2 = 2 * w1 - w0 + h2 * fi * xi  # here fi=f1
        fi = f[i]  # fi=f2
        xi = w2 / (1 - h12 * fi)
        x[i] = xi
        w0 = w1
        w1 = w2
    return x


def calculate_bound_radial_wavefunction(Z, n, l, xc="PBE"):
    from gpaw.atom.all_electron import AllElectron

    check_valid_quantum_number(Z, n, l)
    config_tuples = config_str_to_config_tuples(
        electron_configurations[chemical_symbols[Z]]
    )
    subshell_index = [shell[:2] for shell in config_tuples].index((n, l))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ae = AllElectron(chemical_symbols[Z], xcname=xc)
        ae.run()

    energy = ae.e_j[subshell_index] * units.Hartree

    return RadialWavefunction(
        n=n,
        l=l,
        energy=energy,
        radial_grid=ae.r,
        radial_values=ae.u_j[subshell_index],
    )


def radial_schroedinger_equation(ef, l, r, vr):
    return l * (l + 1) / r**2 - vr(r) / r - ef


def calculate_continuum_radial_wavefunction(Z, n, l, lprime, epsilon, xc="PBE"):
    from gpaw.atom.all_electron import AllElectron

    check_valid_quantum_number(Z, n, l)
    config_tuples = config_str_to_config_tuples(
        electron_configurations[chemical_symbols[Z]]
    )
    subshell_index = [shell[:2] for shell in config_tuples].index((n, l))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ae = AllElectron(chemical_symbols[Z], xcname=xc)
        ae.f_j[subshell_index] -= 1.0
        ae.run()

    vr = interp1d(ae.r, -2 * ae.vr, fill_value="extrapolate", bounds_error=False)

    ef = epsilon / units.Rydberg

    r = np.linspace(1e-6, 100, 1000000)
    f = radial_schroedinger_equation(ef, lprime, r, vr)

    ur = numerov(f, 0.0, 1e-7, r[1] - r[0])
    ur = ur / ur.max() / (np.sqrt(np.pi) * ef ** (1 / 4))

    return RadialWavefunction(
        n=0,
        l=lprime,
        energy=epsilon,
        radial_grid=r,
        radial_values=ur,
    )


class BaseTransitionCollection:
    def __init__(self, Z):
        self._Z = Z

    @property
    def Z(self):
        return self._Z

    @abstractmethod
    def get_transition_potential(self):
        pass


class SubshellTransitions(BaseTransitionCollection):
    def __init__(
        self,
        Z: int,
        n: int,
        l: int,
        order: int = 1,
        only_dipole: bool = False,
        min_contrast: float = 1.0,
        epsilon: float = 1.0,
        xc: str = "PBE",
    ):
        # check_valid_quantum_number(Z, n, l)
        self._n = n
        self._l = l
        self._order = order
        self._min_contrast = min_contrast
        self._only_dipole = only_dipole
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

    def get_bound_wavefunction(self):
        return calculate_bound_radial_wavefunction(
            Z=self.Z, n=self.n, l=self.l, xc=self.xc
        )

    def get_excited_wavefunctions(self):
        continuum = [
            calculate_continuum_radial_wavefunction(
                Z=self.Z, n=self.n, l=self.l, lprime=lprime, epsilon=self.epsilon
            )
            for lprime in self.lprimes
        ]
        return continuum

    def get_transition_quantum_numbers(self):
        return [
            (
                (bound[0].n, bound[0].l, bound[1]),
                (excited[0].energy, excited[0].l, excited[1]),
            )
            for (bound, excited) in self.get_transitions()
        ]

    def get_transitions(self):
        bound = self.get_bound_wavefunction()
        bound_states = [(bound, ml) for ml in np.arange(-bound.l, bound.l + 1)]

        excited = self.get_excited_wavefunctions()
        excited_states = [
            (wave_function, ml)
            for wave_function in excited
            for ml in np.arange(-wave_function.l, wave_function.l + 1)
        ]

        transitions = []
        for bound_state, excited_state in itertools.product(
            bound_states, excited_states
        ):

            if self._only_dipole and (
                not (abs(bound_state[0].l - excited_state[0].l) == 1)
            ):
                continue

            if self._only_dipole and (not (abs(bound_state[1] - excited_state[1]) < 2)):
                continue

            transitions.append((bound_state, excited_state))

        return transitions

    def get_transition_potentials(
        self,
        extent: float | tuple[float, float] = None,
        gpts: float | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        energy: float = None,
    ):
        transitions = self.get_transitions()
        return SubshellTransitionPotentials(
            self.Z,
            transitions,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
            energy=energy,
        )


class BaseTransitionPotential(HasAcceleratorMixin, HasGridMixin):
    def __init__(self, Z, extent, gpts, sampling, energy):
        self._Z = Z
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

    @property
    def Z(self):
        return self._Z


class SubshellTransitionPotentials(BaseTransitionPotential):
    def __init__(
        self,
        Z: int,
        transitions,
        orbital_filling_factor: bool = True,
        extent: float | tuple[float, float] = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        energy: float = None,
    ):
        self._orbital_filling_factor = orbital_filling_factor
        self._transitions = transitions
        super().__init__(Z, extent, gpts, sampling, energy)

    @property
    def ensemble_shape(self):
        return (len(self._transitions),)

    @property
    def ensemble_axes_metadata(self):
        values = [
            f"{bound} -> {excited}"
            for (bound, excited) in self.transition_quantum_numbers
        ]
        return OrdinalAxis(label="(n, l, ml) -> (l', ml')", values=values)

    @property
    def transitions(self):
        return self._transitions

    @property
    def transition_quantum_numbers(self):
        return [
            (bound.quantum_numbers, excited.quantum_numbers)
            for (bound, excited) in self._transitions
        ]

    def overlap_integral(self, lprimeprime, bound, excited, k):
        radial_grid = np.arange(0, np.max(k) * 1.01, 1 / max(self.extent))
        integration_grid = np.linspace(0, bound.radial_grid[-1], 400000)

        values = (
            bound(integration_grid)
            * spherical_jn(
                lprimeprime,
                2 * np.pi * units.Bohr * radial_grid[:, None] * integration_grid[None],
            )
            * excited(integration_grid)
        )

        integral = np.trapz(values, integration_grid, axis=1)

        return interp1d(radial_grid, integral)(k)

    def form_factor(self, bound, excited, k, phi, theta):
        from sympy.physics.wigner import wigner_3j

        Hn0 = np.zeros_like(k, dtype=complex)
        l = bound[0].l
        lprime = excited[0].l
        ml = bound[1]  # .ml
        mlprime = excited[1]  # .ml

        for lprimeprime in range(abs(l - lprime), abs(l + lprime) + 1):
            jq = self.overlap_integral(lprimeprime, bound[0], excited[0], k)

            for mlprimeprime in range(-lprimeprime, lprimeprime + 1):
                if ml - mlprime - mlprimeprime != 0:
                    continue

                prefactor = (
                    np.sqrt(4 * np.pi)
                    * ((-1j) ** lprimeprime)
                    * np.sqrt((2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * l + 1))
                    * (-1.0) ** (mlprime + mlprimeprime)
                    * float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))
                    * float(
                        wigner_3j(lprime, lprimeprime, l, -mlprime, -mlprimeprime, ml)
                    )
                )

                if np.abs(prefactor) < 1e-12:
                    continue

                Ylm = sph_harm(mlprimeprime, lprimeprime, phi, theta)
                Hn0 += prefactor * jq * Ylm

        return Hn0

    def build(self):
        array = np.zeros((len(self._transitions),) + self.gpts, dtype=complex)
        k0 = 1 / energy2wavelength(self.energy)

        for i, (bound, excited) in enumerate(self._transitions):
            energy_loss = bound[0].energy - excited[0].energy
            kn = 1 / energy2wavelength(self.energy + energy_loss)
            kz = k0 - kn

            kt, phi = polar_spatial_frequencies(self.gpts, self.sampling, dtype=float)
            k = np.sqrt(kt**2 + kz**2)
            theta = np.pi - np.arctan(kt / kz)

            array[i] = self.form_factor(bound, excited, k, phi, theta)

            if self._orbital_filling_factor:
                array[i] *= np.sqrt(4 * bound[0].l + 2)

            array[i] *= relativistic_mass_correction(self.energy) / (
                2 * units.Bohr * np.pi**2 * np.sqrt(units.Rydberg) * kn * k**2
            )

        array = np.fft.ifft2(array) / np.prod(self.sampling)
        return array


#
# class SubshellTransitions(AbstractTransitionCollection):
#     def __init__(
#         self,
#         Z: int,
#         n: int,
#         l: int,
#         order: int = 1,
#         min_contrast: float = 1.0,
#         epsilon: float = 1.0,
#         xc: str = "PBE",
#     ):
#         check_valid_quantum_number(Z, n, l)
#         self._n = n
#         self._l = l
#         self._order = order
#         self._min_contrast = min_contrast
#         self._epsilon = epsilon
#         self._xc = xc
#         super().__init__(Z)
#
#     @property
#     def order(self):
#         return self._order
#
#     @property
#     def min_contrast(self):
#         return self._min_contrast
#
#     @property
#     def epsilon(self):
#         return self._epsilon
#
#     @property
#     def xc(self):
#         return self._xc
#
#     @property
#     def n(self):
#         return self._n
#
#     @property
#     def l(self):
#         return self._l
#
#     @property
#     def lprimes(self):
#         min_new_l = max(self.l - self.order, 0)
#         return np.arange(min_new_l, self.l + self.order + 1)
#
#     def __len__(self):
#         return len(self.get_transition_quantum_numbers())
#
#     @property
#     def ionization_energy(self):
#         atomic_energy, _ = self._calculate_bound()
#         ionic_energy, _ = self._calculate_continuum()
#         return ionic_energy - atomic_energy
#
#     @property
#     def energy_loss(self):
#         atomic_energy, _ = self._calculate_bound()
#         ionic_energy, _ = self._calculate_continuum()
#         return self.ionization_energy + self.epsilon
#
#     @property
#     def bound_configuration(self):
#         return electron_configurations[chemical_symbols[self.Z]]
#
#     @property
#     def excited_configuration(self):
#         return remove_electron_from_config_str(self.bound_configuration, self.n, self.l)
#
#     def _calculate_bound(self):
#         from gpaw.atom.all_electron import AllElectron
#
#         check_valid_quantum_number(self.Z, self.n, self.l)
#         config_tuples = config_str_to_config_tuples(
#             electron_configurations[chemical_symbols[self.Z]]
#         )
#         subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))
#
#         with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#             ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
#             ae.run()
#
#         return ae.ETotal * units.Hartree, (ae.r, ae.u_j[subshell_index])
#
#     def _calculate_continuum(self):
#         from gpaw.atom.all_electron import AllElectron
#
#         check_valid_quantum_number(self.Z, self.n, self.l)
#         config_tuples = config_str_to_config_tuples(
#             electron_configurations[chemical_symbols[self.Z]]
#         )
#         subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))
#
#         with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#             ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
#             ae.f_j[subshell_index] -= 1.0
#             ae.run()
#
#         vr = interp1d(ae.r, -2 * ae.vr, fill_value="extrapolate", bounds_error=False)
#
#         # def schroedinger_derivative(y, r, l, k2, vr):
#         #     (u, up) = y
#         #     return np.array([up, (l * (l + 1) / r**2 - 2 * vr(r) / r - k2) * u])
#         #
#
#         def radial_schroedinger_equation(ef, l, r, vr):
#             return l * (l + 1) / r**2 - vr(r) / r - ef
#
#         ef = self.epsilon / units.Rydberg
#         norm = 1 / (np.sqrt(np.pi) * ef ** (1/4))
#
#         r = np.linspace(1e-6, 100, 10000000)
#         continuum_waves = {}
#         for lprime in self.lprimes:
#             # ur = integrate.odeint(
#             #     schroedinger_derivative, [0.0, 1.0], r, args=(lprime, k2, vr)
#             # )
#
#             f = radial_schroedinger_equation(ef, lprime, r, vr)
#             ur = Numerovc(f, 0.0, 1e-7, r[1] - r[0])
#
#             ur = ur / ur.max()
#
#             continuum_waves[lprime] = r, ur * norm
#         return ae.ETotal * units.Hartree, continuum_waves
#
#     def get_bound_wave(self):
#         return self._calculate_bound()[1]
#
#     def get_continuum_waves(self):
#         return self._calculate_continuum()[1]
#
#     def get_transition_quantum_numbers(
#         self,
#     ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
#         transitions = []
#         for ml in np.arange(-self.l, self.l + 1):
#             for new_l in self.lprimes:
#                 for new_ml in np.arange(-new_l, new_l + 1):
#
#                     # if not abs(new_l - self.l) == 1:
#                     #   continue
#
#                     # if not (abs(ml - new_ml) < 2):
#                     #     continue
#
#                     transitions.append(((self.l, ml), (new_l, new_ml)))
#         return transitions
#
#     def as_arrays(self):
#         _, bound_wave = self._calculate_bound()
#         _, continuum_waves = self._calculate_continuum()
#
#         bound_state = self.get_transition_quantum_numbers()[0][0]
#         continuum_states = [state[1] for state in self.get_transition_quantum_numbers()]
#         _, continuum_waves = self._calculate_continuum()
#
#         arrays = SubshellTransitionsArrays(
#             Z=self.Z,
#             bound_wave=bound_wave,
#             continuum_waves=continuum_waves,
#             bound_state=bound_state,
#             continuum_states=continuum_states,
#             energy_loss=self.energy_loss,
#         )
#
#         return arrays
#
#     def get_transition_potentials(
#         self,
#         extent: Union[float, Sequence[float]] = None,
#         gpts: Union[float, Sequence[float]] = None,
#         sampling: Union[float, Sequence[float]] = None,
#         energy: float = None,
#         quantum_numbers=None,
#     ):
#
#         _, bound_wave = self._calculate_bound()
#         _, continuum_waves = self._calculate_continuum()
#         energy_loss = self.energy_loss
#
#         if quantum_numbers is None:
#             quantum_numbers = self.get_transition_quantum_numbers()
#
#         transition_potential = SubshellTransitionPotentials(
#             Z=self.Z,
#             bound_wave=bound_wave,
#             continuum_waves=continuum_waves,
#             quantum_numbers=quantum_numbers,
#             energy_loss=energy_loss,
#             extent=extent,
#             gpts=gpts,
#             sampling=sampling,
#             energy=energy,
#         )
#
#         return transition_potential
#
#
# class BaseTransitionPotential(HasAcceleratorMixin, HasGridMixin):
#     def __init__(self, Z, extent, gpts, sampling, energy):
#         self._Z = Z
#         self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
#         self._accelerator = Accelerator(energy=energy)
#
#     @property
#     def Z(self):
#         return self._Z
#
#
# class SubshellTransitionPotentials(BaseTransitionPotential):
#     def __init__(
#         self,
#         Z: int,
#         bound_wave: callable,
#         continuum_waves: dict,
#         quantum_numbers: List[Tuple[Tuple[int, int], Tuple[int, int]]],
#         energy_loss: float = 1.0,
#         extent: Union[float, Tuple[float, float]] = None,
#         gpts: Union[int, Tuple[float, float]] = None,
#         sampling: Union[float, Tuple[float, float]] = None,
#         energy: float = None,
#     ):
#
#         self._bound_wave = bound_wave
#         self._continuum_waves = continuum_waves
#         self._quantum_numbers = quantum_numbers
#         self._energy_loss = energy_loss
#         super().__init__(Z, extent, gpts, sampling, energy)
#
#         self._array = None
#
#     @property
#     def energy_loss(self):
#         return self._energy_loss
#
#     @property
#     def bound_wave(self):
#         return self._bound_wave
#
#     @property
#     def continuum_waves(self):
#         return self._continuum_waves
#
#     @property
#     def quantum_numbers(self):
#         return self._quantum_numbers
#
#     # def __str__(self):
#     #    return f'{self._bound_state} -> {self._continuum_state}'
#
#     @property
#     def array(self):
#         if self._array is None:
#             self._array = self._calculate_array()
#
#         return self._array
#
#     def filter_low_intensity_potentials(self, min_total_intensity):
#         intensities = (np.abs(self.array) ** 2).sum((1, 2))
#         fractions = intensities / intensities.sum()
#         order = np.argsort(fractions)[::-1]
#
#         running_total = fractions[0]
#         included = [order[0]]
#         for i, j in enumerate(order[1:]):
#
#             if running_total + fractions[j] > min_total_intensity:
#                 if not np.isclose(fractions[order[i]], fractions[j]):
#                     break
#
#             included.append(j)
#             running_total += fractions[j]
#
#     @property
#     def total_intensity(self):
#         return (np.abs(self.array) ** 2).sum()
#
#     @property
#     def momentum_transfer(self):
#         k0 = 1 / energy2wavelength(self.energy)
#         kn = 1 / energy2wavelength(self.energy + self.energy_loss)
#         return k0 - kn
#
#     def validate_sites(self, sites):
#         if isinstance(sites, Atoms):
#             sites = sites[sites.numbers == self.Z].positions[:, :2]
#         elif isinstance(sites, Atom):
#             if sites.number == self.Z:
#                 sites = sites.position[:2]
#             else:
#                 sites = np.zeros((0, 2), dtype=np.float32)
#
#         if len(sites.shape) == 1:
#             sites = sites[None]
#
#         sites = np.array(sites, dtype=np.float32)
#         return sites
#
#     def scatter(self, waves, sites):
#         self.grid.match(waves)
#         self.accelerator.match(waves)
#         self.grid.check_is_defined()
#         self.accelerator.check_is_defined()
#
#         positions = self.validate_sites(sites)
#         positions /= self.sampling
#
#         self._array = copy_to_device(self.array, waves.array)
#         positions = copy_to_device(positions, waves.array)
#
#         array = ifft2(
#             self.array[None] * fft_shift_kernel(positions, self.gpts)[:, None]
#         )
#
#         array = array.reshape((-1,) + (1,) * (len(waves.shape) - 2) + array.shape[-2:])
#
#         d = waves._copy_as_dict(copy_array=False)
#         d["array"] = array * waves.array[None]
#         d["ensemble_axes_metadata"] = [
#             {"type": "ensemble", "label": "core ionization"}
#         ] + d["ensemble_axes_metadata"]
#         return waves.__class__(**d)
#
#     def generate_scattered_waves(self, waves, sites, chunks=1):
#         sites = self.validate_sites(sites)
#
#         for start, end in generate_chunks(len(sites), chunks=chunks):
#             if end - start == 0:
#                 break
#
#             scattered_waves = self.scatter(waves, sites[start:end])
#             yield sites[start:end], scattered_waves
#
#     def _calculate_overlap_integral(self, lprime, lprimeprime, k):
#         radial_grid = np.arange(0, np.max(k) * 1.05, 1 / max(self.extent))
#         integration_grid = np.linspace(0, self._bound_wave[0][-1], 10000)
#
#         bound_wave = interp1d(
#             *self._bound_wave,
#             kind="cubic",
#             fill_value="extrapolate",
#             bounds_error=False,
#         )
#         continuum_wave = interp1d(
#             *self._continuum_waves[lprime],
#             kind="cubic",
#             fill_value="extrapolate",
#             bounds_error=False,
#         )
#
#         integration_grid = np.linspace(0, self._bound_wave[0][-1], 10000)
#
#         values = (
#             bound_wave(integration_grid)
#             * spherical_jn(lprimeprime, radial_grid[:, None] * integration_grid[None])
#             * continuum_wave(integration_grid)
#         )
#
#         integral = np.trapz(values, integration_grid, axis=1)
#
#         return interp1d(radial_grid, integral)(k)
#
#     def _calculate_array(self):
#         from sympy.physics.wigner import wigner_3j
#
#         self.grid.check_is_defined()
#         self.accelerator.check_is_defined()
#
#         array = np.zeros((len(self.quantum_numbers),) + self.gpts, dtype=np.complex64)
#         kz = self.momentum_transfer
#
#         kt, phi = polar_spatial_frequencies(self.gpts, self.sampling)
#         k = np.sqrt(kt**2 + kz**2) * 2 * np.pi
#         theta = np.pi - np.arctan(kt / kz)
#
#         for i, ((l, ml), (lprime, mlprime)) in enumerate(self.quantum_numbers):
#
#             for lprimeprime in range(max(l - lprime, 0), abs(l + lprime) + 1):
#                 overlap_integral = self._calculate_overlap_integral(
#                     lprime, lprimeprime, k
#                 )
#
#                 for mlprimeprime in range(-lprimeprime, lprimeprime + 1):
#
#                     if ml - mlprime - mlprimeprime != 0:
#                         continue
#
#                     prefactor = (
#                         np.sqrt(4 * np.pi)
#                         * ((-1.0j) ** lprimeprime)
#                         * np.sqrt(
#                             (2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * l + 1)
#                         )
#                     )
#
#                     prefactor *= (
#                         (-1.0) ** (mlprime + mlprimeprime)
#                         * float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))
#                         * float(
#                             wigner_3j(
#                                 lprime, lprimeprime, l, -mlprime, -mlprimeprime, ml
#                             )
#                         )
#                     )
#
#                     if np.abs(prefactor) < 1e-12:
#                         continue
#
#                     Ylm = sph_harm(mlprimeprime, lprimeprime, phi, theta)
#                     array[i] += prefactor * overlap_integral * Ylm
#
#             # array[i] *= np.sqrt(4 * l + 2)
#
#             # import matplotlib.pyplot as plt
#             #
#             # plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(array[i]))))
#             # plt.show()
#
#         array *= np.prod(self.gpts) / np.prod(self.extent)
#
#         # kn = 1 / energy2wavelength(self.energy + self.energy_loss)
#
#         # y = relativistic_mass_correction(self.energy)
#
#         # matrix_element_const = 4 * np.pi * units._e ** 2 / k ** 2
#         # interaction_constant = y * units._me / (2 * np.pi * units._hbar ** 2 * kn)
#         # print(interaction_constant)
#         # bohr = 4 * np.pi * units._eps0 * units._hbar / (units._me * units._e ** 2)
#         # rydberg = units._me * units._e ** 4 / (8 * units._eps0 ** 2 * units._hplanck ** 2) / units._e
#         # rydberg = 1 / 2 * units._e ** 2 / (4 * np.pi * units._eps0 * units.Bohr)
#         # bohr ** 2 * rydberg = 4 * np.pi
#         # 1 / (2 * units._me) * (units._hbar * 1e10)**2 / units._e
#         # inter = relativistic_mass_correction(self.energy) *
#
#         # array *= (
#         #     y
#         #     * np.sqrt(units.Rydberg)
#         #     / (2 * units.Bohr**2 * units.Rydberg * kn * k**2 / 4)
#         # )
#
#         return array
#
#     def to_images(self):
#         array = np.fft.fftshift(ifft2(self.array), axes=(-2, -1))
#         ensemble_axes_metadata = [OrdinalAxis(values=[1] * 4)]
#         return Images(
#             array, sampling=self.sampling, ensemble_axes_metadata=ensemble_axes_metadata
#         )
#
#     def show(self, **kwargs):
#         self.to_images().show(**kwargs)
#
#
# from typing import TYPE_CHECKING
#
# import numpy as np
# from ase import Atoms
#
# from abtem.antialias import AntialiasAperture
# from abtem.core.backend import get_array_module, copy_to_device
# from abtem.core.complex import complex_exponential
# from abtem.measurements import Images
# from abtem.potentials.iam import _validate_potential
# from abtem.slicing import SliceIndexedAtoms
# from abtem.multislice import (
#     FresnelPropagator,
#     multislice_step,
#     allocate_multislice_measurements,
# )
#
# if TYPE_CHECKING:
#     from abtem.prism.s_matrix import SMatrix
#
#
# def validate_sites(potential=None, sites=None):
#     if sites is None:
#         if hasattr(potential, "frozen_phonons"):
#             sites = potential.frozen_phonons.atoms
#         else:
#             raise RuntimeError(
#                 f"transition sites cannot be inferred from potential of type {type(potential)}"
#             )
#
#     if isinstance(sites, SliceIndexedAtoms):
#         if len(potential) != len(sites):
#             raise RuntimeError(
#                 f"transition sites slices({len(sites)}) != potential slices({len(potential)})"
#             )
#     elif isinstance(sites, Atoms):
#         sites = SliceIndexedAtoms(sites, slice_thickness=potential.slice_thickness)
#
#     else:
#         raise RuntimeError(
#             f"transition sites must be Atoms or SliceIndexedAtoms, received {type(sites)}"
#         )
#
#     return sites
#
#
# def transition_potential_multislice_and_detect(
#     waves,
#     potential,
#     detectors,
#     transition_potentials,
#     ctf=None,
#     keep_ensemble_dims=False,
# ):
#     # print(potential.num_frozen_phonons)
#     potential = _validate_potential(potential)
#     # sites = validate_sites(potential, sites)
#
#     transition_potentials.grid.match(waves)
#     transition_potentials.accelerator.match(waves)
#
#     antialias_aperture = AntialiasAperture(
#         device=get_array_module(waves.array)
#     ).match_grid(waves)
#     propagator = FresnelPropagator(device=get_array_module(waves.array)).match_waves(
#         waves
#     )
#
#     transmission_function = potential.build(lazy=waves.is_lazy).transmission_function(
#         energy=waves.energy
#     )
#     transmission_function = antialias_aperture.bandlimit(transmission_function)
#
#     sites = potential.sliced_atoms
#
#     measurements = allocate_multislice_measurements(waves, potential, detectors)
#
#     for scattering_index, (transmission_function_slice, sites_slice) in enumerate(
#         zip(transmission_function, sites)
#     ):
#         sites_slice = transition_potentials.validate_sites(sites_slice)
#
#         for _, scattered_waves in transition_potentials.generate_scattered_waves(
#             waves, sites_slice
#         ):
#
#             slice_generator = transmission_function._generate_slices(
#                 first_slice=scattering_index
#             )
#             current_slice_index = scattering_index
#
#             first_exit_slice = np.searchsorted(
#                 potential.exit_planes, current_slice_index
#             )
#
#             for detect_index, exit_slice in enumerate(
#                 potential.exit_planes[first_exit_slice:], first_exit_slice
#             ):
#
#                 while exit_slice != current_slice_index:
#                     potential_slice = next(slice_generator)
#                     scattered_waves = multislice_step(
#                         scattered_waves, potential_slice, propagator, antialias_aperture
#                     )
#                     current_slice_index += 1
#
#                 if ctf is not None:
#                     scattered_waves = scattered_waves.apply_ctf(ctf)
#
#                 for detector, measurement in zip(detectors, measurements):
#                     new_measurement = detector.detect(scattered_waves).mean(0)
#                     measurements[detector].array[
#                         (0, detect_index)
#                     ] += new_measurement.array
#
#         propagator.thickness = transmission_function_slice.thickness
#         waves = transmission_function_slice.transmit(waves)
#         waves = propagator.propagate(waves)
#
#     measurements = tuple(measurements.values())
#
#     if not keep_ensemble_dims:
#         measurements = tuple(measurement[0, 0] for measurement in measurements)
#
#     return measurements
#
#
# def linear_scaling_transition_multislice(
#     S1: "SMatrix", S2: "SMatrix", scan, transition_potentials, reverse_multislice=False
# ):
#     xp = get_array_module(S1._device)
#     from tqdm.auto import tqdm
#
#     positions = scan.get_positions(lazy=False).reshape((-1, 2))
#
#     prism_region = (
#         S1.extent[0] / S1.interpolation[0] / 2,
#         S1.extent[1] / S1.interpolation[1] / 2,
#     )
#
#     positions = xp.asarray(positions, dtype=np.float32)
#
#     wave_vectors = xp.asarray(S1.wave_vectors)
#     coefficients = complex_exponential(
#         -2.0 * xp.float32(xp.pi) * positions[:, 0, None] * wave_vectors[None, :, 0]
#     )
#     coefficients *= complex_exponential(
#         -2.0 * xp.float32(np.pi) * positions[:, 1, None] * wave_vectors[None, :, 1]
#     )
#     coefficients = coefficients / xp.sqrt(coefficients.shape[1]).astype(np.float32)
#
#     potential = S1.potential
#
#     sites = validate_sites(potential, sites=None)
#     chunks = S1.chunks
#     stream = S1._device == "gpu" and S1._store_on_host
#
#     S1 = S1.build(lazy=False, stop=0)
#
#     if reverse_multislice:
#         S2_multislice = S2.build(lazy=False, start=len(potential), stop=0)
#     else:
#         S2_multislice = S2
#
#     images = np.zeros(len(positions), dtype=np.float32)
#     for i in tqdm(range(len(potential))):
#
#         # if stream:
#         #     S1 = S1.streaming_multislice(potential, chunks=chunks, start=max(i - 1, 0), stop=i)
#         #     # if hasattr(S2_multislice, 'build'):
#         #     S2 = S2.streaming_multislice(potential, chunks=chunks, start=max(i - 1, 0), stop=i)
#         #     # else:
#         #     #    S2_multislice = S2_multislice.build(potential, chunks=chunks, start=max(i - 1, 0),
#         #     #                                                       stop=i)
#         # else:
#         S1 = S1.multislice(potential, chunks=chunks, start=max(i - 1, 0), stop=i)
#         # S2_multislice = S2.build(start=len(potential), stop=i, lazy=False)
#
#         if reverse_multislice:
#             S2_multislice = S2_multislice.multislice(
#                 potential, chunks=chunks, start=max(i - 1, 0), stop=i, conjugate=True
#             )
#         else:
#             S2_multislice = S2.build(lazy=False, start=len(potential), stop=i)
#
#         sites_slice = transition_potentials.validate_sites(sites[i])
#
#         for site in sites_slice:
#             # S2_crop = S2.crop_to_positions(site)
#             S2_crop = S2_multislice.crop_to_positions(site)
#             scattered_S1 = S1.crop_to_positions(site)
#
#             if stream:
#                 S2_crop = S2_crop.copy("gpu")
#                 scattered_S1 = scattered_S1.copy("gpu")
#
#             shifted_site = site - np.array(scattered_S1.crop_offset) * np.array(
#                 scattered_S1.sampling
#             )
#             scattered_S1 = transition_potentials.scatter(scattered_S1, shifted_site)
#
#             if S1.interpolation == (1, 1):
#                 cropped_coefficients = coefficients
#                 mask = None
#             else:
#                 mask = xp.ones(len(coefficients), dtype=bool)
#                 if S1.interpolation[0] > 1:
#                     mask *= (
#                         xp.abs(positions[:, 0] - site[0])
#                         % (S1.extent[0] - prism_region[0])
#                     ) <= prism_region[0]
#                 if S1.interpolation[1] > 1:
#                     mask *= (
#                         xp.abs(positions[:, 1] - site[1])
#                         % (S1.extent[1] - prism_region[1])
#                     ) <= prism_region[1]
#
#                 cropped_coefficients = coefficients[mask]
#
#             a = S2_crop.array.reshape((1, len(S2), -1))
#             b = xp.swapaxes(
#                 scattered_S1.array.reshape((len(scattered_S1.array), len(S1), -1)), 1, 2
#             )
#
#             SHn0 = xp.dot(a, b)
#             SHn0 = xp.swapaxes(SHn0[0], 0, 1)
#
#             new_values = copy_to_device(
#                 (xp.abs(xp.dot(SHn0, cropped_coefficients.T[None])) ** 2).sum(
#                     (0, 1, 2)
#                 ),
#                 np,
#             )
#             if mask is not None:
#                 images[mask] += new_values
#             else:
#                 images += new_values
#
#     images *= np.prod(S1.interpolation).astype(np.float32) ** 2
#
#     images = Images(images.reshape(scan.gpts), sampling=scan.sampling)
#     return images
