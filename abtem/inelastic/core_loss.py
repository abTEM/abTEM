from __future__ import annotations

import io
import itertools
import warnings
from abc import ABCMeta, abstractmethod
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
from abtem.core.utils import CopyMixin, get_dtype
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


def _eager_waves(waves):
    """The waves as an eager object; a lazy one is computed into a copy.

    For the methods that reduce the whole wave intensity to a host-side result
    (a threshold, a site subset): a lazy array would otherwise meet in-place
    NumPy/CuPy operations it cannot take part in, or be recomputed piecewise.
    abTEM's compute() works in place, so the caller's waves are copied first.
    """
    if waves.is_lazy:
        return waves.copy().compute(progress_bar=False)
    return waves


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
        self._interpolator = None

    def __call__(self, r):
        # Built once and reused: the overlap integral evaluates each radial
        # function several times, and the continuum grid can hold millions of
        # points at low continuum energy.
        if self._interpolator is None:
            self._interpolator = interp1d(
                self._radial_grid,
                self._radial_values,
                kind=2,
                fill_value="extrapolate",
            )
        return self._interpolator(r)

    @property
    def bound(self):
        # Continuum states carry n=None, so comparing n to 0 raised TypeError.
        return self._n is not None

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
    # f.copy() rather than np.zeros(len(f)): some numba/numpy pairings
    # (observed: numba 0.64.0 + numpy 2.4.3) fail to type numba's internal
    # np.zeros -> np.empty lowering inside @jit, while ndarray.copy() is
    # unaffected. Every element of x is overwritten below before being
    # read, so the borrowed initial values from f are never used.
    x = f.copy()
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

    with AllElectron(chemical_symbols[Z], xcname=xc, txt=None) as ae:
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


# Radial step of the continuum integration grid [Bohr], and the largest grid we
# are willing to build before giving up on resolving the asymptotic region.
_CONTINUUM_STEP = 20.0 / (1000000 - 1)
_CONTINUUM_MAX_RADIUS = 150.0

# The outer region must span this many oscillations, and the centrifugal term
# throughout it must be this small a fraction of the kinetic energy, for the
# wave to be free-particle-like enough to read off its amplitude. The envelope
# of a free wave is exactly constant, so one oscillation is enough to sample it;
# the spread of the envelope is checked separately as a diagnostic.
_CONTINUUM_PERIODS = 1.0
_CONTINUUM_CENTRIFUGAL_TOLERANCE = 0.02

# Fraction of the grid, measured from the outer edge, used to read the amplitude.
_ASYMPTOTIC_REGION_FRACTION = 0.25


def _continuum_radial_grid(ef: float, lprime: int) -> np.ndarray:
    """
    Radial grid whose outer quarter is asymptotically free.

    The amplitude of the continuum wave can only be read off where the
    centrifugal barrier is negligible and the wave has completed several
    oscillations. Both conditions are hardest to satisfy at low continuum
    energy and high angular momentum, so the grid is extended as needed.

    Parameters
    ----------
    ef : float
        Continuum energy [Rydberg].
    lprime : int
        Angular momentum of the continuum state.

    Returns
    -------
    r : np.ndarray
        Radial grid [Bohr].
    """
    k = np.sqrt(ef)

    fraction = _ASYMPTOTIC_REGION_FRACTION

    # The outer region spans `fraction` of the grid and must contain enough
    # oscillations to sample the envelope.
    radius = max(20.0, _CONTINUUM_PERIODS * 2 * np.pi / (k * fraction))

    if lprime > 0:
        # The centrifugal term is largest at the inner edge of the outer region.
        free_radius = np.sqrt(
            lprime * (lprime + 1) / (_CONTINUUM_CENTRIFUGAL_TOLERANCE * ef)
        )
        radius = max(radius, free_radius / (1.0 - fraction))

    if radius > _CONTINUUM_MAX_RADIUS:
        warnings.warn(
            f"the continuum state (epsilon={ef * units.Rydberg:.3g} eV, "
            f"l'={lprime}) does not reach its asymptotic form within "
            f"{_CONTINUUM_MAX_RADIUS} Bohr; its normalisation, and hence the "
            "absolute scale of the transition potential, may be inaccurate",
            RuntimeWarning,
            stacklevel=3,
        )
        radius = _CONTINUUM_MAX_RADIUS

    return np.linspace(1e-12, radius, int(round(radius / _CONTINUUM_STEP)) + 1)


def _asymptotic_amplitude(r: np.ndarray, u: np.ndarray, k: float) -> float:
    """
    Amplitude of the asymptotic oscillation ``u(r) -> A sin(k r + delta)``.

    Taken from the envelope ``sqrt(u^2 + (u'/k)^2)`` over the outer region,
    which is constant wherever the wave is free.

    Using ``max(u)`` instead, as earlier versions of abTEM did, is wrong in two
    regimes: at low continuum energy the grid spans less than one oscillation,
    and for ``l' >= 2`` the envelope overshoots its asymptotic value near the
    turning point of the centrifugal barrier, so the maximum is attained well
    inside the atom.

    Parameters
    ----------
    r : np.ndarray
        Radial grid [Bohr].
    u : np.ndarray
        Radial wavefunction on that grid.
    k : float
        Asymptotic wavenumber [1/Bohr].

    Returns
    -------
    amplitude : float
    """
    outer = r > (1.0 - _ASYMPTOTIC_REGION_FRACTION) * r[-1]

    du = np.gradient(u, r)
    envelope = np.sqrt(u[outer] ** 2 + (du[outer] / k) ** 2)

    mean = float(np.mean(envelope))
    if mean <= 0.0:
        raise RuntimeError(
            "the continuum wavefunction vanishes in the asymptotic region"
        )

    spread = float(np.std(envelope)) / mean
    if spread > 0.05:
        warnings.warn(
            f"the continuum wavefunction amplitude varies by {spread:.1%} over "
            "the asymptotic region, so it has not reached its free-particle "
            "form; its normalisation may be inaccurate",
            RuntimeWarning,
            stacklevel=3,
        )

    return float(np.median(envelope))


# GPAW's xc kernels clamp the density from below at this value (measured: an
# LDA/PBE atom's vxc is exactly constant wherever n < 1e-10).
_GPAW_XC_DENSITY_FLOOR = 1e-10


def _atomic_rv(Z, xc="PBE"):
    """r*V(r) of the neutral ground-state atom [Rydberg * Bohr], from a scalar-
    relativistic GPAW all-electron calculation. The potential the continuum
    states are solved in.

    GPAW evaluates the xc potential at a floored density, so in the vacuum
    beyond ~17 Bohr (LDA) / ~27 Bohr (PBE) its vxc sits at a constant
    -8.5e-4 Hartree instead of vanishing -- the same for every element, and
    however far the radial grid extends. That constant made r*V grow
    linearly to the edge of the grid and, extrapolated beyond it, shifted
    the continuum's asymptotic wavenumber (q/k = 1.012 at 1 eV). Below the
    floor vxc is scaled as n^(1/3), the low-density limit of LDA exchange and
    correlation, which is continuous at the floor and vanishes in vacuum; r*V
    of the neutral atom then goes to ~0 by the edge of the grid and is held
    constant beyond it.
    """
    # from gpaw.atom.all_electron import AllElectron
    from gpaw.atom.aeatom import AllElectronAtom

    ae = AllElectronAtom(chemical_symbols[Z], xc=xc, log=io.StringIO())
    # ae.f_j[subshell_index] -= 0.0
    ae.run()
    ae.scalar_relativistic = True
    ae.refine()

    r = ae.rgd.r_g
    n = ae.n_sg.sum(0)
    vr = ae.vr_sg[0].copy()  # r * (vxc + vH) - Z [Hartree * Bohr]
    below_floor = n < _GPAW_XC_DENSITY_FLOOR
    vxc = ae.vxc_sg[0][below_floor]
    vr[below_floor] -= r[below_floor] * vxc * (
        1 - np.cbrt(n[below_floor] / _GPAW_XC_DENSITY_FLOOR)
    )

    # -2: to the sign and Rydberg units of radial_schroedinger_equation.
    rv = -2 * vr

    return interp1d(r, rv, fill_value=(rv[0], rv[-1]), bounds_error=False)


def calculate_continuum_radial_wavefunction(Z, n, l, lprime, epsilon, xc="PBE"):
    check_valid_quantum_number(Z, n, l)
    # config_tuples = config_str_to_config_tuples(
    #     electron_configurations[chemical_symbols[Z]]
    # )
    # subshell_index = [shell[:2] for shell in config_tuples].index((n, l))

    vr = _atomic_rv(Z, xc=xc)

    ef = epsilon / units.Rydberg

    r = _continuum_radial_grid(ef, lprime)
    f = radial_schroedinger_equation(ef, lprime, r, vr)

    ur = numerov(f, 0.0, 1e-12, r[1] - r[0])

    # Energy normalisation, <eps|eps'> = delta(eps - eps') with eps in Rydberg:
    # u(r) -> sin(k r + delta) / sqrt(pi k), so the asymptotic amplitude is
    # 1 / sqrt(pi k) with k = sqrt(ef).
    ur = ur / _asymptotic_amplitude(r, ur, k=np.sqrt(ef))
    ur = ur / (np.sqrt(np.pi) * ef ** (1 / 4))

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
        # np.arange stops short of its end by up to one step, so with only the 5 %
        # margin a coarse grid (max(k) * extent below ~20, e.g. 16 pixels) ended
        # below max(k) and interp1d refused the largest k. Reach at least one step
        # past max(k); grids that already did are unchanged.
        step = 1 / max(self.extent)
        k_max = np.max(k)
        radial_grid = np.arange(0, max(k_max * 1.05, k_max + step), step)
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
        try:
            from sympy.physics.wigner import wigner_3j
        except ImportError as e:
            raise ImportError(
                "Calculating core-loss EELS form factors requires sympy. "
                "Install it with `pip install abtem[core-loss]` or "
                "`pip install sympy`."
            ) from e

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

        # Keep the n strongest transitions. Earlier versions sliced the
        # unsorted list instead, which kept an arbitrary subset; the sibling
        # TransitionPotentialArray.filter_by_intensity has always used `order`.
        transitions = [self.transitions[i] for i in order[:n]]

        if not len(transitions) > 0:
            raise RuntimeError()

        kwargs = self._copy_kwargs(exclude=("transitions",))
        kwargs["transitions"] = transitions

        return self.__class__(**kwargs)

    def build(self) -> TransitionPotentialArray:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        array = np.zeros(
            (len(self._transitions),) + self.gpts, dtype=get_dtype(complex=True)
        )
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
                # 4*l+2 is the full subshell's electron count: spin (2) times
                # orbital degeneracy (2*l+1). The orbital part is already
                # realised explicitly -- SubshellTransitions.get_transitions
                # builds one bound state per ml and this array is summed
                # incoherently over all of them -- so only the spin factor
                # belongs here. Applying 4*l+2 per ml double-counts the
                # orbital degeneracy by (2*l+1); invisible for l=0, where
                # 4*l+2 reduces to the spin-only factor of 2.
                array[i] *= np.sqrt(2)

            array[i] *= relativistic_mass_correction(self.energy) / (
                2 * np.pi**2 * kn * k**2 * energy2sigma(self.energy)
            )

        # In place: `array / np.prod(...)` divides by a float64 numpy scalar,
        # which under NEP 50 promotes the whole array to complex128 and silently
        # discards the dtype the array was allocated with.
        array /= np.prod(self.sampling)

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


def fast_roll(array, shifts):
    """Batched 2D circular roll: ``out[i] == xp.roll(array, shifts[i], axis=(0, 1))``.

    On CPU the per-site quadrant-copy loop is already very fast — each slice is
    a memmove — and beats both ``xp.roll`` in a loop and a full advanced-indexing
    gather. On GPU the advanced-indexing form wins because the per-site loop
    serialises kernel launches; we dispatch on the backend.

    Shifts are first reduced modulo ``H`` / ``W`` so negative and out-of-range
    values are handled correctly (the previous version raised RuntimeError on
    negative shifts).
    """
    xp = get_array_module(array)
    H, W = array.shape[-2:]
    shifts = shifts.copy()
    shifts[:, 0] %= H
    shifts[:, 1] %= W

    if xp is not np:
        # GPU path: batched gather. CuPy's advanced indexing launches one
        # kernel for the whole batch instead of one per site.
        rows = (xp.arange(H)[None, :] - shifts[:, 0:1]) % H
        cols = (xp.arange(W)[None, :] - shifts[:, 1:2]) % W
        return array[rows[:, :, None], cols[:, None, :]]

    # CPU path: per-site quadrant copy. Memmove inside each branch is faster
    # than any vectorised numpy alternative we benchmarked.
    output = xp.empty((len(shifts),) + array.shape, dtype=array.dtype)
    for i in range(len(shifts)):
        s0, s1 = int(shifts[i, 0]), int(shifts[i, 1])
        if s0 > 0 and s1 > 0:
            output[i, :s0, :s1] = array[-s0:, -s1:]
            output[i, :s0, s1:] = array[-s0:, :-s1]
            output[i, s0:, :s1] = array[:-s0, -s1:]
            output[i, s0:, s1:] = array[:-s0, :-s1]
        elif s1 > 0:
            output[i, :, :s1] = array[:, -s1:]
            output[i, :, s1:] = array[:, :-s1]
        elif s0 > 0:
            output[i, :s0, :] = array[-s0:, :]
            output[i, s0:, :] = array[:-s0, :]
        else:
            output[i] = array

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
        """The overlap cut that ``filter_sites`` applies for a relative
        ``threshold``: ranking the overlap of a site placed at every pixel
        with ``waves``, the most-overlapping pixels that together carry the
        fraction ``threshold`` of the total overlap all lie above the cut.
        Evaluate it on the waves that the sites are then filtered against
        -- applied to a different (e.g. further propagated) wave, a group of
        symmetry-tied sites at the cut falls below it all at once.

        Returns 0.0, which disables filtering, for ``threshold >= 1``.
        """
        if threshold >= 1.0:
            return 0.0

        if hasattr(waves, "build"):
            waves = waves.build(lazy=False)
        waves = _eager_waves(waves).ensure_real_space()

        local_potential = self.local_potential(space="real").sum(0)
        array = abs2(waves.array)

        local_potential = copy_to_device(local_potential, array)

        complex_dtype = get_dtype(complex=True)
        overlap = fft2_convolve(
            local_potential[(None,) * (len(array.shape) - 2)].astype(complex_dtype),
            fft2(array.astype(complex_dtype)),
        ).real

        overlap = copy_to_device(overlap, "cpu")

        overlap = np.sort(overlap.ravel())[::-1]

        cumulative = np.cumsum(overlap) / overlap.sum()

        # The pixels ranked 0..index together carry at least the fraction
        # ``threshold`` of the overlap, so the cut is the overlap of pixel
        # ``index`` itself -- not of pixel ``index - 1``, which stopped the
        # retained set short of the requested fraction (and, for a threshold
        # below the largest single-pixel fraction, wrapped to -1, i.e. the
        # *smallest* overlap, keeping almost every site).
        index = min(
            int(np.searchsorted(cumulative, threshold, side="left")),
            len(overlap) - 1,
        )

        # filter_sites keeps sites whose overlap is strictly greater than the
        # returned value, recomputing it by direct summation, which differs
        # from this FFT convolution by float round-off (~1 ulp of the largest
        # overlap). Lower the cut by a few such ulps so that the pixel at the
        # cut -- and any group of (symmetry-)tied pixels with it -- is kept
        # whole rather than partly lost to round-off.
        tolerance = 8 * np.finfo(overlap.dtype).eps * float(overlap[0])
        return max(float(overlap[index]) - tolerance, 0.0)

    def validate_sites(self, sites: Atoms | Atom) -> np.ndarray:
        if isinstance(sites, Atoms):
            sites = sites[sites.numbers == self.Z].positions[:, :2]
        elif isinstance(sites, Atom):
            if sites.number == self.Z:
                sites = sites.position[:2]
            else:
                sites = np.zeros((0, 2), dtype=get_dtype())
        else:
            sites = np.array(sites)

        if len(sites.shape) == 1:
            sites = sites[None]

        sites = np.array(sites, dtype=get_dtype())
        return sites

    def filter_sites(self, waves, sites, threshold):
        """The sites at which the transition potential overlaps the waves.

        Parameters
        ----------
        waves : Waves or Probe
            The waves that are scattered.
        sites : Atoms, Atom or numpy.ndarray
            The candidate sites, as atoms or as an (n, 2) array of positions.
        threshold : float, optional
            The absolute overlap threshold; sites with an overlap above it are kept.
            Without a threshold all sites are kept.

        Returns
        -------
        Atoms or numpy.ndarray
            The sites that are kept, in the form they were given: the atoms of the
            transition potential's element for `Atoms`, positions otherwise. Without
            a threshold nothing is filtered and an `Atoms` or `Atom` is returned as
            it is; with a threshold a single `Atom` gives an array of positions.
        """
        if hasattr(waves, "build"):
            waves = waves.build(lazy=False)
        waves = _eager_waves(waves).ensure_real_space()

        # The mask below is computed over the validated array, which subsets an
        # Atoms input to this element: apply it to that subset, not to the whole
        # of the caller's atoms, or the two lengths disagree.
        original_sites = sites
        validated_sites = self.validate_sites(sites)
        kept = validated_sites

        if threshold is not None and threshold > 0.0:
            xp = get_array_module(waves.array)
            validated_sites = copy_to_device(validated_sites, waves.array)

            rounded_sites = xp.round(
                (validated_sites / xp.array(self.sampling))
            ).astype(int)

            local_potential = copy_to_device(self._local_potential, waves.array)

            # Stream the overlap reduction over sites in chunks. The full
            # (n_sites, *waves_shape, H, W) tensor that the naive computation
            # would build can dwarf available memory for big scans / many sites;
            # by reducing each chunk to a per-site sum before moving on, peak
            # transient is O(chunk_size) instead of O(n_sites).
            #
            # The chunk size targets the same byte budget as the rest of abTEM
            # (dask.chunk-size / dask.chunk-size-gpu) via validate_chunks, so
            # users who already tuned that knob for a memory-constrained
            # workstation get the tighter behaviour here automatically.
            abs2_waves = abs2(waves.array)
            n_sites = len(validated_sites)
            reduce_axes_offset = len(waves.shape) - 2  # broadcast dims per site

            chunks = validate_chunks(
                shape=(n_sites,) + waves.shape,
                chunks=("auto",) + (-1,) * len(waves.shape),
                max_elements="auto",
                dtype=waves.dtype,
                device=self.device,
            )[0]

            mask = xp.zeros(n_sites, dtype=bool)
            start = 0
            for chunk_size in chunks:
                end = start + chunk_size
                shifted = fast_roll(local_potential, rounded_sites[start:end])
                shifted = shifted.reshape(
                    (end - start,)
                    + (1,) * reduce_axes_offset
                    + shifted.shape[-2:]
                )
                overlaps = (shifted * abs2_waves[None]).sum(axis=(-2, -1))
                chunk_mask = overlaps > threshold
                if chunk_mask.ndim > 1:
                    chunk_mask = chunk_mask.any(
                        tuple(range(1, chunk_mask.ndim))
                    )
                mask[start:end] = chunk_mask
                start = end

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

            kept = kept[mask]  # on the host: validated_sites was moved to the device
            if isinstance(original_sites, Atoms):
                return original_sites[original_sites.numbers == self.Z][mask]

            return kept

        # without a threshold nothing is filtered: the sites are returned as given
        if isinstance(original_sites, (Atoms, Atom)):
            return original_sites

        return kept

    def scatter(
        self, waves: Waves, sites: Atoms | Atom | np.ndarray, threshold: float = None
    ) -> Waves:
        """Scatter ``waves`` at ``sites``. Waves in reciprocal space are converted to
        real space, and the scattered waves are in real space."""
        waves = waves.ensure_real_space()
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

            sites = sites / xp.array(self.sampling, dtype=get_dtype())

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
        """Yield ``(sites, scattered_waves)`` for chunks of ``sites``. Waves in
        reciprocal space are converted to real space once, and the scattered waves
        are in real space."""
        waves = waves.ensure_real_space()
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
        return self.to_images().show(**kwargs)


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
