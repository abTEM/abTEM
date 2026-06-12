from __future__ import annotations

import contextlib
import itertools
import os
from abc import ABCMeta, abstractmethod
from bisect import bisect_left
from typing import TYPE_CHECKING

import numpy as np
from ase import Atom, Atoms, units
from ase.data import chemical_symbols
from numba import jit
from scipy.interpolate import interp1d

# ---------
# This is a version check for scipy >=v1.17.0
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version
try:
    import scipy
    _SCIPY_VERSION = Version(version("scipy"))
except PackageNotFoundError:
    scipy = None
    _SCIPY_VERSION = None

if _SCIPY_VERSION is not None and _SCIPY_VERSION >= Version("1.17.0"):
    from scipy.special import spherical_jn, sph_harm_y
else:
    from scipy.special import spherical_jn, sph_harm
    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)
# ---------


from abtem.array import ArrayObject
from abtem.core.axes import AxisMetadata, OrdinalAxis
from abtem.core.backend import copy_to_device, get_array_module
from abtem.core.chunks import validate_chunks
from abtem.core.complex import abs2, complex_exponential
from abtem.core.electron_configurations import electron_configurations
from abtem.core.energy import (
    Accelerator,
    HasAcceleratorMixin,
    energy2sigma,
    energy2wavelength,
    relativistic_mass_correction,
)
from abtem.core.fft import fft2, fft2_convolve, fft_shift_kernel, ifft2
from abtem.core.grid import Grid, HasGrid2DMixin, polar_spatial_frequencies
from abtem.core.utils import CopyMixin
from abtem.measurements import Images, RealSpaceLineProfiles, _polar_detector_bins

if TYPE_CHECKING:
    from abtem.prism.s_matrix import SMatrix
    from abtem.waves import Waves

azimuthal_number = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
azimuthal_letter = {value: key for key, value in azimuthal_number.items()}


def config_str_to_config_tuples(config_str):
    """Parse an electron configuration string (e.g. "1s2 2s2 2p6") into a list of
    (n, l, occupancy) tuples."""
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
    """Convert a list of (n, l, occupancy) tuples back to an electron configuration
    string (e.g. "1s2 2s2 2p6")."""
    config_str = []
    for n, ell, occ in config_tuples:
        config_str.append(str(n) + azimuthal_letter[ell] + str(occ))
    return " ".join(config_str)


def remove_electron_from_config_str(config_str, n, ell):
    """Remove one electron from the (n, l) subshell in the given configuration string
    and return the updated configuration string."""
    config_tuples = []
    for shell in config_str_to_config_tuples(config_str):
        if shell[:2] == (n, ell):
            config_tuples.append(shell[:2] + (shell[2] - 1,))
        else:
            config_tuples.append(shell)
    return config_tuples_to_config_str(config_tuples)


def check_valid_quantum_number(Z, n, ell):
    """Validate that the quantum numbers (n, l) correspond to an occupied subshell
    for element with atomic number Z. Raises RuntimeError if invalid."""
    symbol = chemical_symbols[Z]
    config_tuple = config_str_to_config_tuples(electron_configurations[symbol])

    if not any([shell[:2] == (n, ell) for shell in config_tuple]):
        raise RuntimeError(
            f"Quantum numbers (n, ell) = ({n}, {ell}) not valid for element {symbol}"
        )


def _validate_transition_potentials(transition_potentials):
    if hasattr(transition_potentials, "scatter"):
        transition_potentials = [transition_potentials]
    return transition_potentials


class RadialWavefunction:
    def __init__(
        self,
        n: int | None,
        l: int,
        energy: float,
        radial_grid: np.ndarray,
        radial_values: np.ndarray,
    ):
        self._n = n
        self._l = l
        self._energy = energy

        if energy >= 0.0:
            if n is not None:
                raise ValueError()
        else:
            if n is None:
                raise ValueError()

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
        return self.n > 0

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

    def to_lineprofiles(self, sampling=0.01):
        r = np.arange(0, self._radial_grid[-1], sampling)
        return RealSpaceLineProfiles(self(r), sampling=sampling)

    def show(self, **kwargs):
        return self.to_lineprofiles().show(**kwargs)


class AtomicWaveFunction:
    def __init__(self, radial_wavefunction, ml):
        self._radial_wavefunction = radial_wavefunction
        self._ml = ml

    def __call__(self, r):
        return self._radial_wavefunction(r)

    @property
    def bound(self):
        return self._radial_wavefunction.bound

    @property
    def energy(self):
        return self._radial_wavefunction.energy

    @property
    def radial_grid(self):
        return self._radial_wavefunction.radial_grid

    @property
    def n(self):
        return self._radial_wavefunction.n

    @property
    def l(self):
        return self._radial_wavefunction.l

    @property
    def ml(self):
        return self._ml

    @property
    def quantum_numbers(self):
        return self.n, self.l, self.ml


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
    return (l * (l + 1) / r**2 - vr(r) / r) * 1.02 - ef


def calculate_continuum_radial_wavefunction(Z, n, l, lprime, epsilon, xc="PBE"):
    # from gpaw.atom.all_electron import AllElectron
    from gpaw.atom.aeatom import AllElectronAtom

    def f(self, *args, **kwargs):
        pass

    AllElectronAtom.log = f

    check_valid_quantum_number(Z, n, l)
    # config_tuples = config_str_to_config_tuples(
    #     electron_configurations[chemical_symbols[Z]]
    # )
    # subshell_index = [shell[:2] for shell in config_tuples].index((n, l))

    ae = AllElectronAtom(chemical_symbols[Z], xc=xc)
    # ae.f_j[subshell_index] -= 0.0
    ae.run()
    ae.scalar_relativistic = True
    ae.refine()

    vr = interp1d(
        ae.rgd.r_g, -2 * ae.vr_sg[0], fill_value="extrapolate", bounds_error=False
    )

    ef = epsilon / units.Rydberg

    r = np.linspace(1e-12, 20, 1000000)
    f = radial_schroedinger_equation(ef, lprime, r, vr)

    ur = numerov(f, 0.0, 1e-12, r[1] - r[0])
    ur = ur / ur.max() / (np.sqrt(np.pi) * ef ** (1 / 4))

    return RadialWavefunction(
        n=None,
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
        min_contrast: float = 1.0,
        epsilon: float = 1.0,
        xc: str = "PBE",
    ):
        check_valid_quantum_number(Z, n, l)
        self._n = n
        self._l = l
        self._order = order
        self._min_contrast = min_contrast
        self._epsilon = epsilon
        self._xc = xc
        super().__init__(Z)

    def __len__(self):
        return len(self.get_transition_quantum_numbers())

    @property
    def bound_configuration(self):
        return electron_configurations[chemical_symbols[self.Z]]

    @property
    def excited_configuration(self):
        return remove_electron_from_config_str(
            electron_configurations[chemical_symbols[self.Z]], self.n, self.l
        )

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
        return range(min_new_l, self.l + self.order + 1)

    def get_bound_wave_function(self):
        wave_functions = calculate_bound_radial_wavefunction(
            Z=self.Z, n=self.n, l=self.l, xc=self.xc
        )
        return wave_functions

    def get_excited_wave_functions(self):
        wave_functions = [
            calculate_continuum_radial_wavefunction(
                Z=self.Z, n=self.n, l=self.l, lprime=lprime, epsilon=self.epsilon
            )
            for lprime in self.lprimes
        ]
        return wave_functions

    def get_transition_quantum_numbers(self):
        bound_states = [(self.n, self.l, ml) for ml in range(-self.l, self.l + 1)]

        excited_states = []
        for lprime in self.lprimes:
            for mlprime in range(-lprime, lprime + 1):
                excited_states.append((None, lprime, mlprime))

        transitions = []
        for bound_state, excited_state in itertools.product(
            bound_states, excited_states
        ):
            transitions.append((bound_state, excited_state))

        return transitions

    def get_transitions(self):
        bound_state = self.get_bound_wave_function()
        bound_states = [
            AtomicWaveFunction(bound_state, ml)
            for ml in range(-bound_state.l, bound_state.l + 1)
        ]

        excited_states = self.get_excited_wave_functions()
        excited_states = [
            AtomicWaveFunction(radial, ml)
            for radial in excited_states
            for ml in range(-radial.l, radial.l + 1)
        ]

        transitions = []
        for bound_state, excited_state in itertools.product(
            bound_states, excited_states
        ):
            transitions.append((bound_state, excited_state))

        return transitions

    def get_transition_potentials(
        self,
        extent: float | tuple[float, float] = None,
        gpts: float | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        energy: float = None,
        double_channel: bool = True,
    ):
        transitions = self.get_transitions()
        return TransitionPotential(
            self.Z,
            transitions,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
            energy=energy,
            double_channel=double_channel,
        )


class BaseTransitionPotential(
    HasAcceleratorMixin, HasGrid2DMixin, CopyMixin, metaclass=ABCMeta
):
    def __init__(
        self,
        Z: int,
        extent: float | tuple[float, float],
        gpts: int | tuple[int, int],
        sampling: float | tuple[float, float],
        energy: float,
        double_channel: bool = True,
        **kwargs,
    ):
        self._Z = Z
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._double_channel = double_channel
        super().__init__(**kwargs)

    @property
    def double_channel(self) -> bool:
        return self._double_channel

    @property
    def Z(self) -> int:
        return self._Z

    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass


class TransitionPotential(BaseTransitionPotential):
    def __init__(
        self,
        Z: int,
        transitions,
        orbital_filling_factor: bool = True,
        extent: float | tuple[float, float] = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        energy: float = None,
        double_channel: bool = True,
    ):
        self._Z = Z
        self._orbital_filling_factor = orbital_filling_factor
        self._transitions = transitions
        super().__init__(Z, extent, gpts, sampling, energy, double_channel)

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def orbital_filling_factor(self) -> bool:
        return self._orbital_filling_factor

    @property
    def double_channel(self) -> bool:
        return self._double_channel

    @property
    def Z(self) -> int:
        return self._Z

    @property
    def ensemble_shape(self) -> tuple[int]:
        return (len(self._transitions),)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        values = [
            f"{bound[1:]} → {excited[1:]}"
            for (bound, excited) in self.transition_quantum_numbers
        ]

        return [
            OrdinalAxis(
                values=values,
                label="(l,ml)→(l',ml')",
                tex_label=r"$(\ell, m_l) → (\ell', m_l')$",
            )
        ]

    @property
    def metadata(self) -> dict:
        bound = self.transition_quantum_numbers[0][0]
        return {"Z": self.Z, "n": bound[0], "l": bound[1]}

    @property
    def transitions(self):
        return self._transitions

    @property
    def transition_quantum_numbers(self):
        return [
            (bound.quantum_numbers, excited.quantum_numbers)
            for (bound, excited) in self._transitions
        ]

    def _calculate_overlap_integral(self, lprimeprime, bound, excited, k):
        radial_grid = np.arange(0, np.max(k) * 1.05, 1 / max(self.extent))
        integration_grid = np.linspace(0, bound.radial_grid[-1], 20000)

        values = (
            bound(integration_grid)
            * spherical_jn(
                lprimeprime,
                2 * np.pi * units.Bohr * radial_grid[:, None] * integration_grid[None],
            )
            * excited(integration_grid)
        )

        integral = np.trapezoid(values, integration_grid, axis=1) / (
            units.Bohr * np.sqrt(units.Rydberg)
        )

        return interp1d(radial_grid, integral)(k)

    def _calculate_form_factor(self, bound, excited, k, phi, theta):
        from sympy.physics.wigner import wigner_3j

        Hn0 = np.zeros_like(k, dtype=complex)
        l = bound.l
        lprime = excited.l
        ml = bound.ml
        mlprime = excited.ml

        mask = k <= np.max(k) * 2 / 3

        for lprimeprime in range(abs(l - lprime), abs(l + lprime) + 1):
            jq = self._calculate_overlap_integral(lprimeprime, bound, excited, k)

            for mlprimeprime in range(-lprimeprime, lprimeprime + 1):
                if ml - mlprime - mlprimeprime != 0:
                    continue

                lprime = int(lprime)
                lprimeprime = int(lprimeprime)
                l = int(l)
                mlprime = int(mlprime)
                mlprimeprime = int(mlprimeprime)
                ml = int(ml)

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

                Ylm = sph_harm_y(lprimeprime, mlprimeprime, theta, phi)
                Hn0[mask] += prefactor * (jq * Ylm)[mask]

        return Hn0

    def integrated_intensities(self):
        intensities = self.build().to_images().intensity()
        return intensities.array.sum((-2, -1)) * np.prod(self.sampling)

    def filter_by_intensity(self, threshold: float) -> TransitionPotential:
        integrated_intensities = self.integrated_intensities()
        order = np.argsort(-integrated_intensities)
        integrated_intensities = integrated_intensities[order]

        cumulative = np.cumsum(integrated_intensities / integrated_intensities.sum())

        n = np.searchsorted(cumulative, threshold) + 1
        transitions = self.transitions[:n]

        if not len(transitions) > 0:
            raise RuntimeError()

        kwargs = self._copy_kwargs(exclude=("transitions",))
        kwargs["transitions"] = transitions

        return self.__class__(**kwargs)

    def build(self) -> TransitionPotentialArray:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        array = np.zeros((len(self._transitions),) + self.gpts, dtype=np.complex64)
        k0 = 1 / energy2wavelength(self.energy)

        for i, (bound, excited) in enumerate(self._transitions):
            energy_loss = bound.energy - excited.energy

            kn = 1 / energy2wavelength(self.energy + energy_loss)

            kz = k0 - kn

            kxy, phi = polar_spatial_frequencies(self.gpts, self.sampling)
            k = np.sqrt(kxy**2 + kz**2)
            theta = np.pi - np.arctan(kxy / kz)

            array[i] = self._calculate_form_factor(bound, excited, k, phi, theta)

            if self._orbital_filling_factor:
                array[i] *= np.sqrt(4 * bound.l + 2)

            array[i] *= relativistic_mass_correction(self.energy) / (
                2 * np.pi**2 * kn * k**2 * energy2sigma(self.energy)
            )

        array = array / np.prod(self.sampling)

        # array = array.astype(xp.complex64)

        return TransitionPotentialArray(
            self.Z,
            array,
            energy=self.energy,
            extent=self.extent,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def scatter(self, waves: Waves, sites: Atoms | Atom | np.ndarray) -> Waves:
        self.grid.match(waves)
        self.accelerator.match(waves)

        return self.build().scatter(waves, sites)

    def show(self, **kwargs):
        return self.build().to_images().show(**kwargs)


# @njit(fastmath=True)
def fast_roll(array, shifts):
    xp = get_array_module(array)
    output = xp.empty((len(shifts),) + array.shape, dtype=array.dtype)

    for i in range(len(shifts)):
        if np.all(shifts[i] > 0):
            output[i, : shifts[i, 0], : shifts[i, 1]] = array[
                -shifts[i, 0] :, -shifts[i, 1] :
            ]
            output[i, : shifts[i, 0], shifts[i, 1] :] = array[
                -shifts[i, 0] :, : -shifts[i, 1]
            ]
            output[i, shifts[i, 0] :, : shifts[i, 1]] = array[
                : -shifts[i, 0], -shifts[i, 1] :
            ]
            output[i, shifts[i, 0] :, shifts[i, 1] :] = array[
                : -shifts[i, 0], : -shifts[i, 1]
            ]
        elif shifts[i, 1] > 0:
            output[i, :, : shifts[i, 1]] = array[:, -shifts[i, 1] :]
            output[i, :, shifts[i, 1] :] = array[:, : -shifts[i, 1] :]
        elif shifts[i, 0] > 0:
            output[i, : shifts[i, 0], :] = array[-shifts[i, 0] :, :]
            output[i, shifts[i, 0] :, :] = array[: -shifts[i, 0] :, :]
        elif (shifts[i, 0] == 0) and (shifts[i, 1] == 0):
            output[i] = array
        else:
            raise RuntimeError()

    return output


class TransitionPotentialArray(ArrayObject, BaseTransitionPotential):
    _base_dims = 2

    def __init__(
        self,
        Z: int,
        array: np.ndarray,
        energy: float = None,
        extent: float | tuple[float, float] = None,
        sampling: float | tuple[float, float] = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        super().__init__(
            Z=Z,
            extent=extent,
            gpts=array.shape[-2:],
            sampling=sampling,
            energy=energy,
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

        self._local_potential = self.local_potential(space="real").sum(0)
        self._threshold = None

    def from_array_and_metadata(self, array, metadata):
        raise NotImplementedError

    def set_threshold(self, wave, threshold):
        local_potentials = self.local_potential(space="real")
        local_potential = local_potentials.sum(0)

        c = np.fft.irfft2(np.fft.rfft2(local_potential) * np.fft.rfft2(wave.array))
        c = np.sort(c.ravel())[::-1]

    def local_potential(self, max_angle=None, space="reciprocal"):
        """
        Parameters
        ----------
        max_angle : float
            Maximum angle (in degrees) for the local potential calculation.
        space : str, optional
            Specifies the coordinate space in which the potential is calculated.
            Default is "reciprocal". Possible values are "reciprocal" and "real".

        Returns
        -------
        array : ndarray
            The calculated local potential.

        """
        self.accelerator.check_is_defined()
        fourier_space_sampling = self.reciprocal_space_sampling

        angular_sampling = (
            fourier_space_sampling[0] * self.wavelength * 1e3,
            fourier_space_sampling[1] * self.wavelength * 1e3,
        )

        array = self.array

        if max_angle is not None:
            region = _polar_detector_bins(
                gpts=self.gpts,
                sampling=angular_sampling,
                inner=0.0,
                outer=max_angle,
                nbins_radial=1,
                nbins_azimuthal=1,
                fftshift=False,
                rotation=0.0,
                # offset=self.offset,
                return_indices=False,
            )
            region = region >= 0.0
            array = array * region

        if space == "reciprocal":
            array = abs2(array)
        elif space == "real":
            array = abs2(ifft2(array))
        else:
            raise ValueError(
                "The 'space' parameter is invalid. Accepted values are 'reciprocal' or"
                " 'real'."
            )

        return array

    def integrated_intensities(self, max_angle: float, space: str = "reciprocal"):
        array = self.local_potential(max_angle, space)
        intensity = array.sum((-2, -1)) * np.prod(self.sampling)
        return intensity

    def filter_by_intensity(
        self, threshold: float, max_angle: float
    ) -> TransitionPotential:
        intensities = self.integrated_intensities(max_angle)
        order = np.argsort(-intensities)
        intensities = intensities[order]
        cumulative = np.cumsum(intensities / intensities.sum())
        n = np.searchsorted(cumulative, threshold) + 1
        included = order[:n]
        return self[included]

    def absolute_threshold(self, waves: Waves, threshold: float = 1.0):
        if threshold >= 1.0:
            return 0.0

        if hasattr(waves, "build"):
            waves = waves.build(lazy=False)

        local_potential = self.local_potential(space="real").sum(0)
        array = abs2(waves.array)

        local_potential = copy_to_device(local_potential, array)

        overlap = fft2_convolve(
            local_potential[(None,) * (len(array.shape) - 2)].astype(np.complex64),
            fft2(array.astype(np.complex64)),
        ).real

        overlap = copy_to_device(overlap, "cpu")

        overlap = np.sort(overlap.ravel())[::-1]

        cumulative = np.cumsum(overlap) / overlap.sum()

        return overlap[np.searchsorted(cumulative, threshold, side="left") - 1]

    def validate_sites(self, sites: Atoms | Atom) -> np.ndarray:
        if isinstance(sites, Atoms):
            sites = sites[sites.numbers == self.Z].positions[:, :2]
        elif isinstance(sites, Atom):
            if sites.number == self.Z:
                sites = sites.position[:2]
            else:
                sites = np.zeros((0, 2), dtype=np.float32)
        else:
            sites = np.array(sites)

        if len(sites.shape) == 1:
            sites = sites[None]

        sites = np.array(sites, dtype=np.float32)
        return sites

    def filter_sites(self, waves, sites, threshold):
        if hasattr(waves, "build"):
            waves = waves.build(lazy=False)

        validated_sites = self.validate_sites(sites)

        if threshold is not None and threshold > 0.0:
            xp = get_array_module(waves.array)
            validated_sites = copy_to_device(validated_sites, waves.array)

            rounded_sites = xp.round(
                (validated_sites / xp.array(self.sampling))
            ).astype(int)

            local_potential = copy_to_device(self._local_potential, waves.array)

            shifted_local_potential = fast_roll(local_potential, rounded_sites)

            shifted_local_potential = shifted_local_potential.reshape(
                (len(validated_sites),)
                + (1,) * (len(waves.shape) - 2)
                + shifted_local_potential.shape[-2:]
            )

            overlaps = shifted_local_potential * abs2(waves.array[None])
            overlaps = overlaps.sum(axis=(-2, -1))

            mask = overlaps > threshold

            mask = mask.any(tuple(range(1, len(overlaps.shape))))

            mask = copy_to_device(mask, "cpu")
            # if np.any(mask):
            #     print(shifted_local_potential.shape, waves.shape)
            #
            #     plt.imshow(
            #         shifted_local_potential[0, 0, 0]
            #         / shifted_local_potential[0, 0, 0].max()
            #         + abs2(waves.array).sum((0,1)) / abs2(waves.array[0, 0]).max()
            #     )
            #     plt.title("include")
            #     # plt.show()
            #     # plt.imshow(abs2(waves.array[0, 0]))
            #     plt.show()
            # else:
            #     plt.imshow(
            #         shifted_local_potential[0, 0, 0]
            #         / shifted_local_potential[0, 0, 0].max()
            #         + abs2(waves.array).sum((0,1)) / abs2(waves.array[0, 0]).max()
            #     )
            #     plt.title("skip")
            #     # plt.show()
            #     # plt.imshow(abs2(waves.array[0, 0]))
            #     plt.show()

            # print(type(mask), type(sites))

            sites = sites[mask]

        return sites

    def scatter(
        self, waves: Waves, sites: Atoms | Atom | np.ndarray, threshold: float = None
    ) -> Waves:
        self.grid.match(waves)
        self.accelerator.match(waves)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = get_array_module(waves.array)

        sites = self.validate_sites(sites)
        sites = self.filter_sites(waves, sites, threshold=threshold)

        if len(sites) == 0:
            array = waves.array[None][[False]]

        else:
            self._array = copy_to_device(self.array, waves.array)
            sites = copy_to_device(sites, waves.array)

            sites = sites / xp.array(self.sampling, dtype=xp.float32)

            array = ifft2(
                self.array[None]
                * fft_shift_kernel(sites, self.gpts)[:, None]
                * energy2sigma(self.energy)
            )

            array = array.reshape(
                (
                    len(sites),
                    len(self),
                )
                + (1,) * (len(waves.shape) - 2)
                + array.shape[-2:]
            )

            array = array * waves.array[None, None]

            array = array.reshape((-1,) + array.shape[2:])

        d = waves._copy_kwargs(exclude=("array",))
        d["array"] = array

        ensemble_axes_metadata = [AxisMetadata(label="sites")]

        d["ensemble_axes_metadata"] = (
            ensemble_axes_metadata + d["ensemble_axes_metadata"]
        )
        return waves.__class__(**d)

    def generate_scattered_waves(
        self,
        waves: Waves,
        sites: Atoms | Atom | np.ndarray,
        max_batch: int = "auto",
        threshold=None,
    ):
        sites = self.validate_sites(sites)

        if isinstance(max_batch, int):
            limit = int(max_batch * np.prod(waves.shape) * len(self))
        else:
            limit = max_batch

        chunks = validate_chunks(
            shape=(len(sites),) + waves.shape,
            chunks=(max_batch,) + (-1,) * len(waves.shape),
            max_elements=limit,
            dtype=waves.dtype,
            device=self.device,
        )[0]

        start = 0
        for chunk in chunks:
            end = start + chunk
            if end - start == 0:
                break

            sites_chunk = sites[start:end]
            start = end

            scattered_waves = self.scatter(waves, sites_chunk, threshold=threshold)
            yield sites_chunk, scattered_waves

    def to_images(self):
        array = np.fft.fftshift(ifft2(self.array), axes=(-2, -1))
        return Images(
            array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
        )

    def show(self, **kwargs):
        self.to_images().show(**kwargs)


def prism_transition_potential_scan(
    s_matrix: "SMatrix",
    transition_potentials,
    scan,
    detectors=None,
    sites=None,
    double_channel: bool = False,
):
    """PRISM-EELS driver following Brown et al. (Phys. Rev. Research 1,
    033186, 2019).

    Supports any ``interpolation`` factor, ``downsample`` setting, and both
    single- and double-channel modes. The scatter and (optionally)
    double-channel propagation operate on a cropped grid at full resolution
    centered at each scattering site (Sec. IV B); when ``downsample`` is
    enabled the scattered result is Fourier-cropped to the downsampled
    resolution before per-position reduction.

    At ``interpolation=(1,1)`` with ``downsample=False`` the output is
    bit-equivalent (to float32 noise) to ``Probe.transition_potential_scan``
    at the matching ``double_channel`` setting.

    ``double_channel=True`` propagates the scattered state through the
    remaining potential slices to the exit before reducing per-position;
    ``double_channel=False`` (default) detects immediately at the scatter
    slice — Brown's single-channel approximation.

    Frozen-phonon ensemble averaging is handled at the ``SMatrix`` level
    (see ``SMatrix.transition_potential_scan``); Dask lazy evaluation is
    supported via the ``lazy`` parameter on that method.

    Parameters
    ----------
    s_matrix : SMatrix
        S-matrix specification (any ``interpolation``).
    transition_potentials : BaseTransitionPotential
        Atomic transition potential.
    scan : BaseScan or tuple
        Scan positions.
    detectors : BaseDetector or list, optional
        Detectors. Defaults to ``FlexibleAnnularDetector()``.
    sites : Atoms or SliceIndexedAtoms, optional
        Scattering sites. Auto-extracted from the potential if not given,
        following the same logic as
        ``transition_potential_multislice_and_detect``.

    Returns
    -------
    BaseMeasurements or list of BaseMeasurements
        One measurement per detector.
    """
    from abtem.antialias import AntialiasAperture
    from abtem.detectors import FlexibleAnnularDetector, validate_detectors
    from abtem.multislice import (
        FresnelPropagator,
        _potential_ensemble_shape_and_metadata,
        allocate_multislice_measurements,
        conventional_multislice_step,
    )
    from abtem.prism.utils import (
        batch_crop_2d,
        minimum_crop,
        plane_waves,
        wrapped_crop_2d,
    )
    from abtem.scan import BaseScan, validate_scan
    from abtem.slicing import SliceIndexedAtoms
    from abtem.waves import Waves

    if isinstance(transition_potentials, (list, tuple)):
        if len(transition_potentials) != 1:
            raise NotImplementedError(
                "PRISM-EELS supports a single transition potential."
            )
        transition_potential = transition_potentials[0]
    else:
        transition_potential = transition_potentials

    if isinstance(transition_potential, TransitionPotential):
        transition_potential = transition_potential.build()

    from abtem.core.utils import get_dtype

    potential = s_matrix.potential
    energy = s_matrix.energy
    extent = s_matrix.extent
    gpts = s_matrix.gpts
    xp = get_array_module(s_matrix.device)
    complex_dtype = get_dtype(complex=True)
    real_dtype = get_dtype(complex=False)

    # Normalise scan -> BaseScan with squeeze metadata applied to a single
    # tuple input, matching what ``Probe.transition_potential_scan`` does
    # internally via the same helper.
    scan = validate_scan(scan)

    # Detector list
    if detectors is None:
        detectors = [FlexibleAnnularDetector()]
    detectors = validate_detectors(detectors)

    # --- Build initial plane-wave state ---
    wave_vectors = s_matrix.wave_vectors  # (n_k, 2)
    wave_vectors_np = np.array(
        wave_vectors.get() if hasattr(wave_vectors, "get") else wave_vectors
    )
    n_k = len(wave_vectors_np)

    s_array = plane_waves(xp.asarray(wave_vectors_np, dtype=np.float32), extent, gpts)
    # Mirror the normalisation in _build_s_matrix (s_matrix.py:1727)
    s_array = s_array * (np.prod(s_matrix.interpolation) / np.prod(s_array.shape[-2:]))

    s_waves = Waves(
        s_array,
        energy=energy,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_k)))],
    )

    # --- Multislice machinery ---
    antialias_aperture = AntialiasAperture()
    propagator = FresnelPropagator()

    def _step(waves, transmission):
        return conventional_multislice_step(
            waves,
            potential_slice=transmission,
            propagator=propagator,
            antialias_aperture=antialias_aperture,
        )

    transmissions = [
        antialias_aperture.bandlimit(
            s.transmission_function(energy=energy), in_place=False
        )
        for s in potential.generate_slices()
    ]
    n_slices = len(transmissions)

    # --- Transition potential setup ---
    transition_potential.grid.match(s_waves)
    transition_potential.accelerator.match(s_waves)
    transition_potential = transition_potential.copy_to_device(s_matrix.device)
    Z = transition_potential.Z

    # --- Site extraction (mirror transition_potential_multislice_and_detect) ---
    if sites is None and hasattr(potential, "get_sliced_atoms"):
        sites = potential.get_sliced_atoms()
    elif sites is None and hasattr(potential, "atoms"):
        sites = potential.atoms
    elif sites is None and hasattr(potential, "potential_unit"):
        if hasattr(potential.potential_unit, "get_transformed_atoms"):
            unit_atoms = potential.potential_unit.get_transformed_atoms()
            sites = unit_atoms * potential.repetitions

    if isinstance(sites, Atoms):
        sites = SliceIndexedAtoms(sites, slice_thickness=potential.slice_thickness)
    elif not isinstance(sites, SliceIndexedAtoms):
        raise ValueError(
            "Could not derive scattering sites from the potential "
            f"({type(potential).__name__}). Pass ``sites=`` explicitly."
        )

    # --- Position + CTF coefficients ---
    # Mirror what the elastic SMatrix.scan path does in
    # ``_batch_reduce_to_measurements`` (s_matrix.py:1067-1118): bare
    # position phase ramps × CTF coefficients. The CTF normalisation
    # handles the per-k aperture weighting (not just a uniform 1/sqrt(N_k));
    # _calculate_ctf_coefficients (s_matrix.py:1055-1065) normalises by
    # sqrt(sum_k |ctf(k)|^2) so the reduced wave-function amplitude lines up
    # with what Probe.build / Probe.multislice produce.
    from abtem.transfer import CTF
    positions_np = np.asarray(scan.get_positions()).reshape((-1, 2))
    positions = xp.asarray(positions_np, dtype=np.float32)
    n_positions = positions.shape[0]
    wave_vectors_xp = xp.asarray(wave_vectors_np, dtype=np.float32)

    position_coefficients = complex_exponential(
        -2.0 * np.float32(np.pi) * positions[:, 0:1] * wave_vectors_xp[None, :, 0]
    ) * complex_exponential(
        -2.0 * np.float32(np.pi) * positions[:, 1:2] * wave_vectors_xp[None, :, 1]
    )

    ctf = CTF(semiangle_cutoff=s_matrix.semiangle_cutoff, energy=energy)
    ctf.grid.match(s_matrix.dummy_probes())
    alpha = (
        xp.sqrt(wave_vectors_xp[:, 0] ** 2 + wave_vectors_xp[:, 1] ** 2)
        * np.float32(ctf.wavelength)
    )
    phi = xp.arctan2(wave_vectors_xp[:, 1], wave_vectors_xp[:, 0])
    ctf_array = ctf._evaluate_from_angular_grid(alpha, phi)
    ctf_array = ctf_array / xp.sqrt((ctf_array**2).sum(axis=-1, keepdims=True))
    coefficients = (position_coefficients * ctf_array[None, :]).astype(complex_dtype)
    # coefficients: (n_positions, n_k)

    # --- Window properties ---
    # Scatter happens at full gpts resolution; the result is optionally
    # Fourier-cropped to the downsampled resolution before reduction.
    from abtem.core.utils import safe_ceiling_int
    from abtem.core.fft import fft_interpolate

    interpolation = s_matrix.interpolation
    ds_gpts = s_matrix.downsampled_gpts
    full_sampling = (extent[0] / gpts[0], extent[1] / gpts[1])
    ds_sampling = (extent[0] / ds_gpts[0], extent[1] / ds_gpts[1])
    needs_downsample = ds_gpts != gpts

    scatter_window_gpts = (
        safe_ceiling_int(gpts[0] / interpolation[0]),
        safe_ceiling_int(gpts[1] / interpolation[1]),
    )
    output_window_gpts = (
        safe_ceiling_int(ds_gpts[0] / interpolation[0]),
        safe_ceiling_int(ds_gpts[1] / interpolation[1]),
    )
    scatter_window_extent = (
        scatter_window_gpts[0] * full_sampling[0],
        scatter_window_gpts[1] * full_sampling[1],
    )
    output_window_extent = (
        output_window_gpts[0] * ds_sampling[0],
        output_window_gpts[1] * ds_sampling[1],
    )

    # --- Pre-compute windowed TP (Brown et al. Sec. IV B) ---
    # Scatter and double-channel propagation operate on a
    # scatter_window_gpts-sized grid centered at each site.
    _tp_real_origin = ifft2(
        transition_potential.array * energy2sigma(energy)
    )
    _tp_crop_corner = (
        -scatter_window_gpts[0] // 2,
        -scatter_window_gpts[1] // 2,
    )
    _tp_window_real = wrapped_crop_2d(
        _tp_real_origin, _tp_crop_corner, scatter_window_gpts
    )
    _tp_window_k = fft2(_tp_window_real)
    _window_propagator = FresnelPropagator()

    # Reduction helpers operate in the downsampled grid.
    pixel_positions = positions / xp.asarray(ds_sampling, dtype=np.float32)
    reduce_crop_corner, reduce_size, reduce_corners = minimum_crop(
        pixel_positions, output_window_gpts
    )

    # --- Exit planes ---
    exit_planes = potential.exit_planes
    n_exit = len(exit_planes)
    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    # --- Allocate measurements with the scan shape ---
    scan_axes_metadata = scan.ensemble_axes_metadata
    scan_shape = scan.shape
    dummy_scan_waves = Waves(
        xp.zeros(scan_shape + output_window_gpts, dtype=complex_dtype),
        energy=energy,
        extent=output_window_extent,
        ensemble_axes_metadata=scan_axes_metadata,
    )
    measurements = allocate_multislice_measurements(
        dummy_scan_waves,
        detectors,
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    )

    # --- Reduce, detect, accumulate helper ---
    def _reduce_and_record(scattered_window, site_xy, exit_idx):
        ds_sampling_arr = np.array(ds_sampling, dtype=np.float32)
        site_pixel_ds = site_xy / ds_sampling_arr
        site_pixel_int_ds = np.rint(site_pixel_ds).astype(int)
        site_crop_corner_ds = (
            int(site_pixel_int_ds[0]) - output_window_gpts[0] // 2,
            int(site_pixel_int_ds[1]) - output_window_gpts[1] // 2,
        )
        site_in_bbox = (
            site_crop_corner_ds[0] - reduce_crop_corner[0],
            site_crop_corner_ds[1] - reduce_crop_corner[1],
        )
        bbox_scattered = xp.zeros(
            scattered_window.shape[:-2] + tuple(reduce_size),
            dtype=complex_dtype,
        )
        for _n0 in range(-1, 2):
            for _n1 in range(-1, 2):
                _r0 = site_in_bbox[0] + _n0 * ds_gpts[0]
                _r1 = site_in_bbox[1] + _n1 * ds_gpts[1]
                _s0 = max(0, -_r0)
                _s1 = max(0, -_r1)
                _d0 = max(0, _r0)
                _d1 = max(0, _r1)
                _e0 = min(reduce_size[0], _r0 + output_window_gpts[0])
                _e1 = min(reduce_size[1], _r1 + output_window_gpts[1])
                if _d0 >= _e0 or _d1 >= _e1:
                    continue
                bbox_scattered[
                    ..., _d0:_e0, _d1:_e1
                ] = scattered_window[
                    ...,
                    _s0 : _s0 + (_e0 - _d0),
                    _s1 : _s1 + (_e1 - _d1),
                ]

        reduced = xp.tensordot(
            coefficients, bbox_scattered, axes=[-1, -3]
        )
        reduced = xp.moveaxis(reduced, 1, 0)
        waves_at_positions = batch_crop_2d(
            reduced, reduce_corners, output_window_gpts
        )

        position_waves_shape = (
            waves_at_positions.shape[:-3]
            + scan_shape
            + waves_at_positions.shape[-2:]
        )
        waves_at_positions = waves_at_positions.reshape(
            position_waves_shape
        )

        n_T = waves_at_positions.shape[0]
        position_waves = Waves(
            waves_at_positions,
            energy=energy,
            extent=output_window_extent,
            ensemble_axes_metadata=[
                OrdinalAxis(values=tuple(range(n_T)))
            ]
            + list(scan_axes_metadata),
        )

        for det_idx, detector in enumerate(detectors):
            m = detector.detect(position_waves)
            m = m.sum((0,))
            if isinstance(exit_idx, int):
                idx = () if n_exit == 1 else (exit_idx,)
                measurements[det_idx].array[idx] += m.array
            else:
                measurements[det_idx].array[exit_idx] += (
                    m.array[(None,) * len(exit_idx)]
                )

    # --- Main loop ---
    for slice_index, transmission in enumerate(transmissions):
        s_waves = _step(s_waves, transmission)

        sites_this_slice = sites.get_atoms_in_slices(
            slice_index, atomic_number=Z
        )
        if len(sites_this_slice) == 0:
            continue

        for atom in sites_this_slice:
            site_xy = np.array(
                [atom.position[0], atom.position[1]], dtype=np.float32
            )

            # --- Windowed scatter (Brown et al. Sec IV B) ---
            full_sampling_arr = np.array(full_sampling, dtype=np.float32)
            site_pixel = site_xy / full_sampling_arr
            site_pixel_int = np.rint(site_pixel).astype(int)
            sub_pixel = xp.asarray(
                (site_pixel - site_pixel_int).reshape(1, 2),
                dtype=np.float32,
            )
            site_crop_corner = (
                int(site_pixel_int[0]) - scatter_window_gpts[0] // 2,
                int(site_pixel_int[1]) - scatter_window_gpts[1] // 2,
            )

            s_cropped = wrapped_crop_2d(
                s_waves.array, site_crop_corner, scatter_window_gpts
            )

            shift_k = fft_shift_kernel(sub_pixel, scatter_window_gpts)
            tp_shifted = ifft2(_tp_window_k * shift_k)

            scattered_window = tp_shifted[:, None] * s_cropped[None, :]

            if double_channel:
                n_T_val = scattered_window.shape[0]
                sw_flat = scattered_window.reshape(
                    (-1,) + tuple(scatter_window_gpts)
                )
                sw_waves = Waves(
                    sw_flat, energy=energy, extent=scatter_window_extent,
                    ensemble_axes_metadata=[
                        OrdinalAxis(
                            values=tuple(range(sw_flat.shape[0]))
                        )
                    ],
                )

                if slice_index in exit_planes:
                    sw_out = sw_waves.array.reshape(
                        (n_T_val, n_k) + tuple(scatter_window_gpts)
                    )
                    if needs_downsample:
                        sw_out = fft_interpolate(
                            sw_out, output_window_gpts,
                            normalization="intensity",
                        )
                    _reduce_and_record(
                        sw_out, site_xy,
                        exit_planes.index(slice_index),
                    )

                for inner_idx, inner_transmission in enumerate(
                    transmissions[slice_index + 1 :]
                ):
                    inner_t_arr = wrapped_crop_2d(
                        inner_transmission.array,
                        site_crop_corner,
                        scatter_window_gpts,
                    )
                    sw_waves._array = sw_waves._array * inner_t_arr
                    sw_waves = _window_propagator.propagate(
                        sw_waves,
                        thickness=inner_transmission.slice_thickness[0],
                        in_place=True,
                    )
                    abs_inner = slice_index + 1 + inner_idx
                    if abs_inner in exit_planes:
                        sw_out = sw_waves.array.reshape(
                            (n_T_val, n_k) + tuple(scatter_window_gpts)
                        )
                        if needs_downsample:
                            sw_out = fft_interpolate(
                                sw_out, output_window_gpts,
                                normalization="intensity",
                            )
                        _reduce_and_record(
                            sw_out, site_xy,
                            exit_planes.index(abs_inner),
                        )

            else:
                if needs_downsample:
                    scattered_window = fft_interpolate(
                        scattered_window, output_window_gpts,
                        normalization="intensity",
                    )

                ep_start = bisect_left(exit_planes, slice_index)
                exit_idx = () if n_exit == 1 else (
                    slice(ep_start, n_exit),
                )
                _reduce_and_record(
                    scattered_window, site_xy, exit_idx
                )

    # Squeeze out single-point-scan axes the same way the multislice path
    # does (via reduce_ensemble inside Waves.transition_potential_multislice
    # — see waves.py:1075). This is what makes ``scan=(0, 0)`` return a bare
    # detector-shaped measurement instead of a ``(1, *detector_shape)``
    # array with a singleton scan axis.
    from abtem.waves import reduce_ensemble
    measurements = [reduce_ensemble(m) for m in measurements]

    if len(measurements) == 1:
        return measurements[0]
    return measurements


def linear_scaling_transition_multislice(
    S1: SMatrix, S2: SMatrix, scan, transition_potentials, reverse_multislice=False
):
    xp = get_array_module(S1._device)
    from tqdm.auto import tqdm

    positions = scan.get_positions(lazy=False).reshape((-1, 2))

    prism_region = (
        S1.extent[0] / S1.interpolation[0] / 2,
        S1.extent[1] / S1.interpolation[1] / 2,
    )

    positions = xp.asarray(positions, dtype=np.float32)

    wave_vectors = xp.asarray(S1.wave_vectors)
    coefficients = complex_exponential(
        -2.0 * xp.float32(xp.pi) * positions[:, 0, None] * wave_vectors[None, :, 0]
    )
    coefficients *= complex_exponential(
        -2.0 * xp.float32(np.pi) * positions[:, 1, None] * wave_vectors[None, :, 1]
    )
    coefficients = coefficients / xp.sqrt(coefficients.shape[1]).astype(np.float32)

    potential = S1.potential

    sites = transition_potentials.validate_sites(potential.atoms)

    chunks = S1.chunks
    stream = S1._device == "gpu" and S1._store_on_host

    S1 = S1.build(lazy=False, stop=0)

    if reverse_multislice:
        S2_multislice = S2.build(lazy=False, start=len(potential), stop=0)
    else:
        S2_multislice = S2

    images = np.zeros(len(positions), dtype=np.float32)
    for i in tqdm(range(len(potential)), delay=0.5):
        if stream:
            S1 = S1.streaming_multislice(
                potential, chunks=chunks, start=max(i - 1, 0), stop=i
            )
            if hasattr(S2_multislice, "build"):
                S2 = S2.streaming_multislice(
                    potential, chunks=chunks, start=max(i - 1, 0), stop=i
                )
            else:
                S2_multislice = S2_multislice.build(
                    potential, chunks=chunks, start=max(i - 1, 0), stop=i
                )
        else:
            S1 = S1.multislice(potential, chunks=chunks, start=max(i - 1, 0), stop=i)
        # S2_multislice = S2.build(start=len(potential), stop=i, lazy=False)

        if reverse_multislice:
            S2_multislice = S2_multislice.multislice(
                potential, chunks=chunks, start=max(i - 1, 0), stop=i, conjugate=True
            )
        else:
            S2_multislice = S2.build(lazy=False, start=len(potential), stop=i)

        sites_slice = transition_potentials.validate_sites(sites[i])

        for site in sites_slice:
            # S2_crop = S2.crop_to_positions(site)
            S2_crop = S2_multislice.crop_to_positions(site)
            scattered_S1 = S1.crop_to_positions(site)

            if stream:
                S2_crop = S2_crop.copy("gpu")
                scattered_S1 = scattered_S1.copy("gpu")

            shifted_site = site - np.array(scattered_S1.crop_offset) * np.array(
                scattered_S1.sampling
            )
            scattered_S1 = transition_potentials.scatter(scattered_S1, shifted_site)

            if S1.interpolation == (1, 1):
                cropped_coefficients = coefficients
                mask = None
            else:
                mask = xp.ones(len(coefficients), dtype=bool)
                if S1.interpolation[0] > 1:
                    mask *= (
                        xp.abs(positions[:, 0] - site[0])
                        % (S1.extent[0] - prism_region[0])
                    ) <= prism_region[0]
                if S1.interpolation[1] > 1:
                    mask *= (
                        xp.abs(positions[:, 1] - site[1])
                        % (S1.extent[1] - prism_region[1])
                    ) <= prism_region[1]

                cropped_coefficients = coefficients[mask]

            a = S2_crop.array.reshape((1, len(S2), -1))
            b = xp.swapaxes(
                scattered_S1.array.reshape((len(scattered_S1.array), len(S1), -1)), 1, 2
            )

            SHn0 = xp.dot(a, b)
            SHn0 = xp.swapaxes(SHn0[0], 0, 1)

            new_values = copy_to_device(
                (xp.abs(xp.dot(SHn0, cropped_coefficients.T[None])) ** 2).sum(
                    (0, 1, 2)
                ),
                np,
            )
            if mask is not None:
                images[mask] += new_values
            else:
                images += new_values

    images *= np.prod(S1.interpolation).astype(np.float32) ** 2

    images = Images(images.reshape(scan.gpts), sampling=scan.sampling)
    return images
