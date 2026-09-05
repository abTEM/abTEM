"""Atomic X-ray emission data for EDX simulations.

This module is a thin adapter over `xraydb <https://github.com/xraypy/XrayDB>`_,
which bundles the tabulation of Elam, Ravel and Sieber (2002). It exposes the
quantities needed to convert an ionisation probability computed by abTEM's
transition-potential machinery into emitted X-ray photons:

- fluorescence yields, per subshell,
- emission-line energies and radiative branching ratios,
- Coster-Kronig transition probabilities,
- natural (core-hole lifetime) line widths.

abTEM's :class:`~abtem.inelastic.core_loss.SubshellTransitions` is
non-relativistic and is labelled by ``(n, l)``, whereas the X-ray data is
resolved by total angular momentum ``j`` (L1/L2/L3, M1...M5). The functions here
handle that mapping: a vacancy created in an ``(n, l)`` subshell is distributed
over the corresponding ``j``-levels with statistical weights ``2j + 1``, and is
then optionally redistributed by Coster-Kronig transitions before it radiates.

Coster-Kronig transitions are applied across the whole shell, not only within
the ionised subshell. Ionising 2s puts a vacancy in L1, and for iron 87 % of it
migrates to L2 and L3, which then emit the L-alpha and L-beta lines; treating
the subshell in isolation puts the fluorescence yield of an L1 edge five times
too low. Because the cascade is linear in the vacancy population, each subshell
still carries its own independent coefficient, so several edges are combined by
a weighted sum -- see :meth:`~abtem.XrayDetector.to_counts_from_subshells`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

__all__ = [
    "EmissionLine",
    "subshell_levels",
    "shell_levels",
    "statistical_weights",
    "vacancy_distribution",
    "fluorescence_yield",
    "emission_lines",
    "line_families",
    "natural_width",
    "absorption_edge",
]


# The IUPAC index of a subshell within its shell, e.g. l = 1 gives L2 and L3.
# The K shell is written without an index by convention.
_SHELL_LETTERS = "KLMNOPQ"

_INSTALL_HINT = (
    "X-ray emission data requires xraydb. Install it with "
    "`pip install abtem[gpaw]` or `pip install xraydb`."
)


def _xraydb():
    """Import xraydb, raising an informative error if it is not installed."""
    try:
        import xraydb  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return xraydb


def _symbol(element: int | str) -> str:
    if isinstance(element, str):
        if element not in atomic_numbers:
            raise ValueError(f"'{element}' is not a chemical symbol")
        return element
    return chemical_symbols[int(element)]


@dataclass(frozen=True)
class EmissionLine:
    """A characteristic X-ray emission line.

    Parameters
    ----------
    name : str
        Siegbahn name of the line, e.g. ``"Ka1"``.
    energy : float
        Photon energy [eV].
    intensity : float
        Photons emitted per ionisation of the parent ``(n, l)`` subshell. This
        already includes the statistical splitting over ``j``-levels, any
        Coster-Kronig redistribution, the fluorescence yield and the radiative
        branching ratio.
    initial_level : str
        IUPAC name of the level holding the vacancy, e.g. ``"K"`` or ``"L3"``.
    final_level : str
        IUPAC name of the level the electron falls from.
    """

    name: str
    energy: float
    intensity: float
    initial_level: str
    final_level: str

    @property
    def family(self) -> str:
        """Line family, e.g. ``"Ka"`` for ``"Ka1"`` and ``"Lb"`` for ``"Lb2,15"``."""
        match = re.match(r"^([A-Za-z]+?)[\d,]*$", self.name)
        return match.group(1) if match else self.name


def subshell_levels(n: int, l: int) -> tuple[str, ...]:
    """
    IUPAC level names for a non-relativistic ``(n, l)`` subshell.

    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.

    Returns
    -------
    levels : tuple of str
        One name for ``l = 0``, otherwise the two spin-orbit split levels in
        order of increasing ``j``, e.g. ``("L2", "L3")`` for ``n = 2, l = 1``.

    Examples
    --------
    >>> subshell_levels(1, 0)
    ('K',)
    >>> subshell_levels(2, 1)
    ('L2', 'L3')
    >>> subshell_levels(3, 2)
    ('M4', 'M5')
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 <= l < n:
        raise ValueError(f"l must satisfy 0 <= l < n, got l={l} for n={n}")
    if n > len(_SHELL_LETTERS):
        raise ValueError(f"no IUPAC level names defined for n={n}")

    letter = _SHELL_LETTERS[n - 1]

    if n == 1:
        return (letter,)

    if l == 0:
        return (f"{letter}1",)

    # Subshells are numbered 1: s, 2: p1/2, 3: p3/2, 4: d3/2, 5: d5/2, ...
    return (f"{letter}{2 * l}", f"{letter}{2 * l + 1}")


def statistical_weights(l: int) -> tuple[float, ...]:
    """
    Fractional occupancy of the spin-orbit split levels of a subshell.

    The weights are ``(2j + 1) / (4l + 2)``, so for ``l = 1`` they are
    ``(1/3, 2/3)`` for L2 and L3 respectively.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.

    Returns
    -------
    weights : tuple of float
        Weights summing to one, ordered as :func:`subshell_levels`.
    """
    if l < 0:
        raise ValueError(f"l must be non-negative, got {l}")

    if l == 0:
        return (1.0,)

    total = 4 * l + 2
    return (2 * l / total, (2 * l + 2) / total)


@lru_cache(maxsize=None)
def shell_levels(n: int) -> tuple[str, ...]:
    """
    Every IUPAC level of a shell, in order of decreasing binding energy.

    Parameters
    ----------
    n : int
        Principal quantum number.

    Returns
    -------
    levels : tuple of str
        ``("K",)`` for n = 1, ``("L1", "L2", "L3")`` for n = 2, and so on.

    Examples
    --------
    >>> shell_levels(2)
    ('L1', 'L2', 'L3')
    """
    levels: list[str] = []
    for l in range(n):
        levels.extend(subshell_levels(n, l))
    return tuple(levels)


def _coster_kronig_cascade(
    symbol: str, vacancies: dict[str, float], levels: tuple[str, ...]
) -> dict[str, float]:
    """
    Redistribute vacancies within a shell by Coster-Kronig transitions.

    Levels are processed in order of decreasing binding energy, so a vacancy
    transferred into an intermediate level goes on to cascade from there. The
    *direct* rates are therefore the right input; xraydb's default
    ``total=True`` already folds the cascade in, and using it here would double
    count. The identity ``f13_total = f13_direct + f12 * f23`` holds exactly in
    the tabulation, which is a useful check.
    """
    xraydb = _xraydb()

    vacancies = dict(vacancies)
    for i, upper in enumerate(levels):
        if vacancies.get(upper, 0.0) == 0.0:
            continue

        transfers = {}
        for lower in levels[i + 1 :]:
            rate = xraydb.ck_probability(symbol, upper, lower, total=False)
            if rate:
                transfers[lower] = vacancies[upper] * rate

        outgoing = sum(transfers.values())
        if outgoing > vacancies[upper]:
            raise RuntimeError(
                f"Coster-Kronig rates out of {symbol} {upper} sum to more than "
                "one; the tabulation cannot be applied as a cascade"
            )

        vacancies[upper] -= outgoing
        for lower, amount in transfers.items():
            vacancies[lower] = vacancies.get(lower, 0.0) + amount

    return vacancies


def _vacancy_distribution(
    symbol: str, n: int, l: int, coster_kronig: bool
) -> tuple[tuple[str, float], ...]:
    levels = shell_levels(n)

    vacancies = {level: 0.0 for level in levels}
    for level, weight in zip(subshell_levels(n, l), statistical_weights(l)):
        vacancies[level] = weight

    if coster_kronig:
        vacancies = _coster_kronig_cascade(symbol, vacancies, levels)

    # Drop levels that never receive a vacancy, so a K edge stays {"K": 1.0}
    # and a 2p edge stays {"L2": ..., "L3": ...}.
    return tuple((level, v) for level, v in vacancies.items() if v != 0.0)


def vacancy_distribution(
    element: int | str, n: int, l: int, coster_kronig: bool = True
) -> dict[str, float]:
    """
    Distribution of a single ``(n, l)`` vacancy over the ``j``-resolved levels.

    The vacancy is created with statistical weights, then optionally
    redistributed by Coster-Kronig transitions across the **whole shell**, not
    only within the ionised subshell. That matters: ionising 2s puts a vacancy
    in L1, and most of it migrates to L2 and L3, which then emit the L-alpha and
    L-beta lines. Treating the subshell in isolation misses that entirely.

    Vacancies are conserved, so the returned fractions sum to one.

    Parameters
    ----------
    element : int or str
        Atomic number or chemical symbol.
    n, l : int
        Quantum numbers of the ionised subshell.
    coster_kronig : bool, optional
        Apply the intra-subshell Coster-Kronig transfer. Default is True.

    Returns
    -------
    distribution : dict
        Mapping of IUPAC level name to vacancy fraction.

    Examples
    --------
    >>> vacancy_distribution("Fe", 2, 1)  # doctest: +SKIP
    {'L2': 0.1933..., 'L3': 0.8066...}
    >>> vacancy_distribution("Fe", 2, 0)  # doctest: +SKIP
    {'L1': 0.13, 'L2': 0.3, 'L3': 0.57}
    """
    # A fresh dict per call, so that a caller mutating the result cannot poison
    # the cache behind it.
    return dict(_vacancy_distribution(_symbol(element), n, l, coster_kronig))


def fluorescence_yield(
    element: int | str, n: int, l: int, coster_kronig: bool = True
) -> float:
    """
    Number of X-ray photons emitted per ionisation of an ``(n, l)`` subshell.

    This is the vacancy-weighted average of the ``j``-resolved fluorescence
    yields. For a K edge it is simply the tabulated omega_K.

    Parameters
    ----------
    element : int or str
        Atomic number or chemical symbol.
    n, l : int
        Quantum numbers of the ionised subshell.
    coster_kronig : bool, optional
        Apply the intra-subshell Coster-Kronig transfer. Default is True.

    Returns
    -------
    yield : float
        Photons per ionisation, summed over all emission lines.

    Examples
    --------
    >>> round(fluorescence_yield("Cu", 1, 0), 4)  # doctest: +SKIP
    0.4411
    """
    xraydb = _xraydb()
    symbol = _symbol(element)
    edges = xraydb.xray_edges(symbol)

    distribution = vacancy_distribution(symbol, n, l, coster_kronig)

    total = 0.0
    for level, fraction in distribution.items():
        if level not in edges:
            continue
        total += fraction * edges[level].fyield

    return total


def emission_lines(
    element: int | str,
    n: int,
    l: int,
    coster_kronig: bool = True,
    min_intensity: float = 0.0,
) -> dict[str, EmissionLine]:
    """
    Emission lines produced by ionising an ``(n, l)`` subshell.

    Parameters
    ----------
    element : int or str
        Atomic number or chemical symbol.
    n, l : int
        Quantum numbers of the ionised subshell.
    coster_kronig : bool, optional
        Apply the intra-subshell Coster-Kronig transfer. Default is True.
    min_intensity : float, optional
        Discard lines emitting fewer photons per ionisation than this. Default
        is 0.0, which keeps every tabulated line.

    Returns
    -------
    lines : dict
        Mapping of Siegbahn name to :class:`EmissionLine`, sorted by decreasing
        intensity. The intensities sum to :func:`fluorescence_yield`.

    Examples
    --------
    >>> lines = emission_lines("Cu", 1, 0)  # doctest: +SKIP
    >>> round(lines["Ka1"].energy, 1)  # doctest: +SKIP
    8046.3
    """
    xraydb = _xraydb()
    symbol = _symbol(element)
    edges = xraydb.xray_edges(symbol)

    distribution = vacancy_distribution(symbol, n, l, coster_kronig)

    collected: dict[str, EmissionLine] = {}
    for level, fraction in distribution.items():
        if level not in edges or fraction == 0.0:
            continue

        omega = edges[level].fyield
        level_lines = xraydb.xray_lines(symbol, level)

        # The tabulated branching ratios of a level sum to one only to within
        # rounding. Renormalising makes the sum rule exact, so that the line
        # intensities always add up to the fluorescence yield. A large deviation
        # would mean the tabulated lines are incomplete, in which case
        # rescaling would wrongly inflate the lines that are present.
        #
        # xraydb's M4 line list is a known exception: for most elements with a
        # tabulated M4 line (Z ~ 56-92) it lists only the dominant Mb line at
        # 0.997068, about 0.3% short of unity, consistently across elements --
        # a gap in the underlying Elam tabulation rather than element-specific
        # noise. K, L1-L3, M3 and M5 all close to within 1e-6. The tolerance is
        # set just above the M4 gap so it still catches genuinely incomplete
        # tabulations elsewhere.
        branching_sum = sum(line.intensity for line in level_lines.values())
        if branching_sum <= 0.0:
            continue
        if abs(branching_sum - 1.0) > 5e-3:
            raise RuntimeError(
                f"branching ratios of {symbol} {level} sum to {branching_sum}, "
                "which suggests the tabulated emission lines are incomplete"
            )

        for name, line in level_lines.items():
            intensity = fraction * omega * line.intensity / branching_sum

            if name in collected:
                # Defensive: the same Siegbahn name should not be produced by
                # two levels of one subshell, but sum rather than overwrite.
                previous = collected[name]
                intensity += previous.intensity

            collected[name] = EmissionLine(
                name=name,
                energy=line.energy,
                intensity=intensity,
                initial_level=level,
                final_level=line.final_level,
            )

    lines = {
        name: line
        for name, line in collected.items()
        if line.intensity >= min_intensity
    }

    return dict(sorted(lines.items(), key=lambda kv: -kv[1].intensity))


def line_families(lines: dict[str, EmissionLine]) -> dict[str, list[EmissionLine]]:
    """
    Group emission lines into families, e.g. ``Ka1`` and ``Ka2`` into ``Ka``.

    Parameters
    ----------
    lines : dict
        Mapping of name to :class:`EmissionLine`, as returned by
        :func:`emission_lines`.

    Returns
    -------
    families : dict
        Mapping of family name to the lines belonging to it, sorted by
        decreasing total intensity.
    """
    families: dict[str, list[EmissionLine]] = {}
    for line in lines.values():
        families.setdefault(line.family, []).append(line)

    for members in families.values():
        members.sort(key=lambda line: -line.intensity)

    return dict(
        sorted(
            families.items(),
            key=lambda kv: -sum(line.intensity for line in kv[1]),
        )
    )


def natural_width(element: int | str, n: int, l: int) -> float:
    """
    Core-hole lifetime broadening of an ``(n, l)`` subshell [eV].

    The statistically weighted average of the tabulated core-hole widths of the
    ``j``-resolved levels. Coster-Kronig redistribution is deliberately not
    applied: the width belongs to the level in which the hole is created.

    Parameters
    ----------
    element : int or str
        Atomic number or chemical symbol.
    n, l : int
        Quantum numbers of the ionised subshell.

    Returns
    -------
    width : float
        Full width at half maximum [eV].
    """
    xraydb = _xraydb()
    symbol = _symbol(element)

    levels = subshell_levels(n, l)
    weights = statistical_weights(l)

    total = 0.0
    for level, weight in zip(levels, weights):
        width = xraydb.core_width(symbol, level)
        if width is None:
            continue
        total += weight * float(np.atleast_1d(width)[0])

    return total


def absorption_edge(element: int | str, n: int, l: int) -> float:
    """
    Ionisation threshold of an ``(n, l)`` subshell [eV].

    The statistically weighted average of the ``j``-resolved edge energies.

    Parameters
    ----------
    element : int or str
        Atomic number or chemical symbol.
    n, l : int
        Quantum numbers of the ionised subshell.

    Returns
    -------
    energy : float
        Edge energy [eV].
    """
    xraydb = _xraydb()
    symbol = _symbol(element)
    edges = xraydb.xray_edges(symbol)

    levels = subshell_levels(n, l)
    weights = statistical_weights(l)

    total = 0.0
    norm = 0.0
    for level, weight in zip(levels, weights):
        if level not in edges:
            continue
        total += weight * edges[level].energy
        norm += weight

    if norm == 0.0:
        raise ValueError(
            f"no tabulated absorption edge for {symbol} (n={n}, l={l})"
        )

    return total / norm
