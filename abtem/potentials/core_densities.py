"""Frozen-core electron densities derived from atomic structure alone.

:class:`.ChargeDensityPotential` is given a valence-only electron density and adds
each atom's full nuclear charge `Z`. The core electrons that separate the two are
not represented anywhere, which leaves the cell carrying a net `+Nc` per atom. This
module supplies those core electrons without requiring an auxiliary DFT calculation
of any kind -- no GPAW, no VASP `POTCAR` -- so that the charge density handed to the
Poisson solve is neutral.

The core is modelled with Slater-screened hydrogenic orbitals. Core electrons are
tightly bound and very nearly transferable between chemical environments, so an
analytic atomic model is a good approximation for them in a way it would not be for
the valence density (which the caller supplies from a real calculation).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from ase.data import chemical_symbols

#: Bohr radius in Ångström.
_BOHR = 0.5291772105638411

#: Closed-shell (noble gas) electron counts.
_NOBLE_GAS_ELECTRONS = (0, 2, 10, 18, 36, 54, 86)

#: Subshells in *shell* order -- by principal quantum number, then angular momentum.
#: Frozen cores are filled this way rather than in Aufbau energy order: strontium's
#: 28-electron core is `[Ar]3d10`, whereas Aufbau ordering puts 4s before 3d and
#: would give `[Ar]4s2 3d8` instead.
_SUBSHELLS = tuple((n, l) for n in range(1, 8) for l in range(n))

#: Effective principal quantum numbers used by Slater's rules.
_N_STAR = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2, 7: 4.3}


def conventional_core_electrons(symbol: str | int) -> int:
    """
    The number of core electrons a frozen-core pseudopotential conventionally uses
    for an element: the largest closed noble-gas shell strictly smaller than `Z`.

    This is only a default. Pseudopotentials that keep semicore states in the
    valence use a smaller core -- VASP's `Sr_sv` and `Ti_sv` have 28 and 12 core
    electrons, not the 36 and 18 this returns -- so a caller must always check the
    result against the valence-electron count the density itself integrates to, and
    fall back to an explicit value when they disagree.

    Parameters
    ----------
    symbol : str or int
        Chemical symbol or atomic number.

    Returns
    -------
    core_electrons : int
    """
    number = symbol if isinstance(symbol, (int, np.integer)) else chemical_symbols.index(symbol)
    return max(n for n in _NOBLE_GAS_ELECTRONS if n < number)


def core_configuration(core_electrons: int) -> tuple[tuple[int, int, int], ...]:
    """
    Fill subshells in shell order until `core_electrons` electrons are placed.

    Returns
    -------
    configuration : tuple of (n, l, occupancy)
    """
    configuration, remaining = [], int(core_electrons)
    for n, l in _SUBSHELLS:
        if remaining <= 0:
            break
        occupancy = min(2 * (2 * l + 1), remaining)
        configuration.append((n, l, occupancy))
        remaining -= occupancy
    if remaining > 0:
        raise ValueError(f"cannot place {core_electrons} electrons in the known subshells")
    return tuple(configuration)


def _slater_exponent(number: int, configuration, index: int) -> float:
    """Slater's rules screening constant for one subshell of `configuration`."""
    n, l, _ = configuration[index]
    screening = 0.0
    for other, (n_other, l_other, occupancy) in enumerate(configuration):
        count = occupancy - 1 if other == index else occupancy
        if count <= 0:
            continue
        if l <= 1:
            if n_other == n and l_other <= 1:
                screening += count * (0.30 if n == 1 else 0.35)
            elif n_other == n - 1:
                screening += count * 0.85
            elif n_other < n - 1:
                screening += count * 1.00
            elif n_other == n and l_other >= 2:
                screening += count * 1.00
        else:
            # A d or f electron is screened fully by everything further in, and by
            # 0.35 per electron in its own subshell. Electrons in OUTER subshells
            # must contribute nothing -- including them over-screens the core.
            if n_other == n and l_other == l:
                screening += count * 0.35
            elif n_other < n or (n_other == n and l_other < l):
                screening += count * 1.00
    return max(number - screening, 0.3) / _N_STAR[n]


@lru_cache(maxsize=256)
def _radial_core_density(number: int, core_electrons: int):
    """`4 pi r^2 n(r)` for the frozen core, on a logarithmic grid in Ångström."""
    radius = np.geomspace(1e-5, 40.0, 5000)  # Bohr
    whole_atom = core_configuration(number)
    density = np.zeros_like(radius)
    for n, l, occupancy in core_configuration(core_electrons):
        index = next(
            i for i, (n_i, l_i, _) in enumerate(whole_atom) if (n_i, l_i) == (n, l)
        )
        u = radius ** _N_STAR[n] * np.exp(-_slater_exponent(number, whole_atom, index) * radius)
        u /= np.sqrt(np.trapezoid(u**2, radius))
        density += occupancy * u**2
    return radius * _BOHR, density


def slater_core_form_factor(symbol: str | int, core_electrons: int, G) -> np.ndarray:
    """
    The `l = 0` spherical Fourier transform of an element's frozen-core electron
    density, from Slater-screened hydrogenic orbitals.

    Parameters
    ----------
    symbol : str or int
        Chemical symbol or atomic number.
    core_electrons : int
        Number of core electrons, `Z - ZVAL`.
    G : numpy.ndarray
        Angular wavenumbers [1 / Å] at which to evaluate the transform.

    Returns
    -------
    form_factor : numpy.ndarray
        In electrons, equal to `core_electrons` at `G = 0`.
    """
    number = symbol if isinstance(symbol, (int, np.integer)) else chemical_symbols.index(symbol)
    G = np.asarray(G, dtype=float)
    if core_electrons <= 0:
        return np.zeros_like(G)

    radius, density = _radial_core_density(int(number), int(core_electrons))
    Gr = np.multiply.outer(G, radius)
    form_factor = np.trapezoid(np.sinc(Gr / np.pi) * density, radius, axis=-1)
    # the radial quadrature sets the overall scale only approximately; the transform
    # must equal the core electron count at G = 0 by construction
    return form_factor * (core_electrons / form_factor.flat[0]) if form_factor.flat[0] else form_factor
