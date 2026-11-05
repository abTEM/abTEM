"""Choosing the multislice sampling.

The sampling has to resolve the atoms' projected potential: the antialias
aperture passes spatial frequencies up to a third of the inverse sampling, and
the potential's scattering beyond it is lost. The collected directions put no
condition on the grid, since the reciprocity waves are carried as periodic
envelopes (see :mod:`abtem.ebsd.reciprocity`).
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from ase import Atoms

from abtem.potentials.iam import Potential

__all__ = ["AntialiasLossWarning", "potential_sampling", "scattering_power_lost"]

#: Fraction of the scattering power a sampling may lose before it is warned about.
_UNDERSAMPLED = 0.02


class AntialiasLossWarning(UserWarning):
    """The sampling leaves part of the atoms' scattering outside the aperture.

    A separate class, so that a builder running many calculations can silence
    the individual warnings and warn once for the whole run.
    """


@lru_cache(maxsize=32)
def _scattering_power(number: int) -> tuple[np.ndarray, np.ndarray]:
    """An atom's projected scattering power, accumulated outwards in frequency.

    Returns
    -------
    frequencies, fraction : np.ndarray
        Spatial frequencies [1 / Å] in increasing order, and the fraction of
        the power at or below each.
    """
    # The Rutherford tail carries a noticeable part of the power, so the grid
    # has to reach well past the cutoff: 1024 points over 8 Å converge it to
    # about 1.5% for light elements and 3% for gold.
    extent, gpts, thickness = 8.0, 1024, 2.0

    atoms = Atoms(
        numbers=[number],
        positions=[(extent / 2, extent / 2, thickness / 2)],
        cell=(extent, extent, thickness),
    )
    # on the host, whatever device abTEM is configured for: a one-off grid,
    # cached per element
    projected = (
        Potential(
            atoms,
            gpts=gpts,
            slice_thickness=thickness,
            projection="finite",
            device="cpu",
        )
        .build(lazy=False)
        .array[0]
    )

    power = np.abs(np.fft.fft2(np.asarray(projected))) ** 2
    frequencies = np.fft.fftfreq(gpts, d=extent / gpts)
    radial = np.hypot(*np.meshgrid(frequencies, frequencies, indexing="ij"))

    order = np.argsort(radial.ravel())
    cumulative = np.cumsum(power.ravel()[order])
    return radial.ravel()[order], cumulative / cumulative[-1]


def _scattering_power_cutoff(number: int, tolerance: float) -> float:
    """Spatial frequency holding all but `tolerance` of an atom's scattering."""
    frequencies, fraction = _scattering_power(number)
    return float(frequencies[np.searchsorted(fraction, 1.0 - tolerance)])


def scattering_power_lost(atoms: Atoms, sampling: float) -> float:
    """Fraction of the atoms' scattering power that a sampling leaves out.

    The fraction outside the antialias aperture, for the most demanding element
    present.

    Parameters
    ----------
    atoms : ase.Atoms
        The specimen. Only the elements present matter.
    sampling : float
        Real-space sampling [Å].

    Returns
    -------
    fraction : float
    """
    aperture = 1.0 / (3.0 * float(sampling))
    lost = 0.0
    for number in np.unique(atoms.numbers):
        frequencies, fraction = _scattering_power(int(number))
        index = (
            min(np.searchsorted(frequencies, aperture, side="right"), len(fraction)) - 1
        )
        lost = max(lost, 1.0 - float(fraction[max(index, 0)]))
    return lost


def _warn_if_undersampled(atoms: Atoms, sampling: float, what: str) -> None:
    """Warn when `sampling` loses more than `_UNDERSAMPLED` of the scattering."""
    lost = scattering_power_lost(atoms, sampling)
    if lost > _UNDERSAMPLED:
        warnings.warn(
            f"{what} sampling of {sampling:.3f} Å leaves {lost:.1%} of the atoms' "
            f"scattering power outside the antialias aperture; "
            f"{potential_sampling(atoms):.3f} Å or finer resolves them",
            AntialiasLossWarning,
        )


def potential_sampling(atoms: Atoms, tolerance: float = 0.01) -> float:
    """Sampling that resolves the projected potential of `atoms`.

    Keeps all but `tolerance` of an isolated atom's projected scattering power
    inside the antialias aperture, for the most demanding element present. The
    error in the pattern is of the order of the power lost, so the default
    targets about a percent; treat the result as an estimate and check
    convergence for production runs.

    The yield of static atoms converges slowly with the sampling, since a finer
    grid resolves the channelled waves at the emitting atoms ever more sharply:
    along [001] in silicon it changes by 2% between 0.063 and 0.032 Å. With
    :class:`~abtem.FrozenPhonons` it changes by 0.3%.

    Parameters
    ----------
    atoms : ase.Atoms
        The specimen. Only the elements present matter.
    tolerance : float, optional
        Fraction of the scattering power allowed outside the antialias aperture
        (default 0.01).

    Returns
    -------
    sampling : float
        Recommended sampling [Å].
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must be between 0 and 1, got {tolerance}")

    if len(atoms) == 0:
        raise ValueError("cannot estimate a sampling for an empty cell")

    cutoff = max(
        _scattering_power_cutoff(int(number), tolerance)
        for number in np.unique(atoms.numbers)
    )
    # the aperture passes |k| < 1 / (3 * sampling)
    return float(1.0 / (3.0 * cutoff))
