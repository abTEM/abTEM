"""The outgoing directions collected by a reciprocity EBSD calculation."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from abtem.core.energy import energy2wavelength
from abtem.core.utils import CopyMixin, EqualityMixin

__all__ = ["BackscatterDetector", "small_angle_error"]


def small_angle_error(angle: float) -> float:
    """Relative error of the small-angle approximation at a given angle.

    Parameters
    ----------
    angle : float
        Scattering angle [mrad].

    Returns
    -------
    error : float
        ``|sin(a) - a| / sin(a)``, the relative error made by treating the
        transverse wavenumber as proportional to the angle.
    """
    radians = angle * 1e-3
    sine = np.sin(radians)
    if sine == 0.0:
        return 0.0
    return float(np.abs(sine - radians) / sine)


class BackscatterDetector(CopyMixin, EqualityMixin):
    """The directions collected by an :class:`~abtem.ebsd.EBSD` calculation.

    Each direction is that of a reciprocity plane wave sent into the slab; it
    gives the intensity of the electrons leaving back through the slab's top
    surface in the opposite direction. The intensity is not a function of an
    exit wave, so this is not a :class:`~abtem.detectors.BaseDetector`.

    The directions are stored as unit vectors, which do not depend on the
    energy, so one detector serves every energy.

    Construct either from a regular grid of angles, with `max_angle` and
    `gpts`, or from explicit `directions`.

    Parameters
    ----------
    max_angle : float, optional
        Half-width of the detector grid along each axis [mrad], measured as a
        scattering angle from the beam direction. The corner of the grid
        reaches a larger angle than this by up to a factor ``sqrt(2)``.
        Give together with `gpts`.
    gpts : int, optional
        Number of detector pixels along each axis of the grid. Give together
        with `max_angle`.
    directions : np.ndarray, optional
        Explicit unit vectors of shape ``(N, 3)`` in the frame of the slab,
        with ``z`` along the beam. Mutually exclusive with `max_angle` and
        `gpts`.
    """

    def __init__(
        self,
        max_angle: Optional[float] = None,
        gpts: Optional[int] = None,
        directions: Optional[np.ndarray] = None,
    ):
        grid_given = max_angle is not None or gpts is not None

        if grid_given and directions is not None:
            raise ValueError(
                "give either 'directions' or 'max_angle' and 'gpts', not both"
            )

        if grid_given:
            if max_angle is None or gpts is None:
                raise ValueError("'max_angle' and 'gpts' must be given together")
            if gpts < 1:
                raise ValueError(f"gpts must be at least 1, got {gpts}")
            if not 0.0 < max_angle < np.pi / 2 * 1e3:
                raise ValueError(
                    f"max_angle must be between 0 and {np.pi / 2 * 1e3:.0f} mrad, "
                    f"got {max_angle}"
                )
            self._gpts: Optional[int] = int(gpts)
            self._max_angle: Optional[float] = float(max_angle)
            self._directions = self._grid_directions(float(max_angle), int(gpts))
        elif directions is not None:
            directions = np.asarray(directions, dtype=float)
            if directions.ndim != 2 or directions.shape[1] != 3:
                raise ValueError(
                    f"directions must have shape (N, 3), got {directions.shape}"
                )
            if np.any(directions[:, 2] <= 0.0):
                raise ValueError(
                    "every direction must have a positive z component; the "
                    "reciprocity plane waves travel into the crystal"
                )
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            self._gpts = None
            self._max_angle = None
            self._directions = directions / norms
        else:
            raise ValueError("give either 'directions' or 'max_angle' and 'gpts'")

        # only a grid's angular calibration assumes small angles
        if self.is_grid:
            error = small_angle_error(self.max_scattering_angle * 1e3)
            if error > 0.05:
                warnings.warn(
                    f"the largest collected angle is "
                    f"{self.max_scattering_angle * 1e3:.0f} mrad, where the "
                    f"small-angle approximation is in error by {error:.1%}; "
                    f"the angular axis calibration assumes it holds"
                )

    @staticmethod
    def _grid_directions(max_angle: float, gpts: int) -> np.ndarray:
        # laid out like the frequencies of an fftshifted diffraction pattern
        sine_max = np.sin(max_angle * 1e-3)
        step = 2.0 * sine_max / gpts
        axis = (np.arange(gpts) - gpts // 2) * step

        dx, dy = np.meshgrid(axis, axis, indexing="ij")
        dx, dy = dx.ravel(), dy.ravel()

        transverse = dx**2 + dy**2
        if np.any(transverse >= 1.0):
            raise ValueError(
                "the detector grid reaches beyond 90 degrees; reduce max_angle"
            )

        return np.stack([dx, dy, np.sqrt(1.0 - transverse)], axis=1)

    @property
    def directions(self) -> np.ndarray:
        """The collected directions, as unit vectors of shape ``(N, 3)``."""
        return self._directions

    @property
    def gpts(self) -> Optional[int]:
        """Detector pixels per axis, or None if built from explicit directions."""
        return self._gpts

    @property
    def max_angle(self) -> Optional[float]:
        """Half-width of the grid [mrad], or None for explicit directions."""
        return self._max_angle

    @property
    def is_grid(self) -> bool:
        """Whether the directions form a regular grid."""
        return self._gpts is not None

    @property
    def base_shape(self) -> tuple[int, ...]:
        """Shape the collected intensities take in a measurement."""
        if self._gpts is None:
            return (len(self._directions),)
        return (self._gpts, self._gpts)

    @property
    def max_scattering_angle(self) -> float:
        """The largest collected scattering angle [rad]."""
        return float(np.arccos(np.min(self._directions[:, 2])))

    def __len__(self) -> int:
        return len(self._directions)

    def transverse_wave_vectors(self, energy: float) -> np.ndarray:
        """Transverse wavevectors of the reciprocity plane waves.

        Parameters
        ----------
        energy : float
            Electron energy [eV].

        Returns
        -------
        wave_vectors : np.ndarray
            Array of shape ``(N, 2)`` holding ``kx`` and ``ky`` [1 / Å].
        """
        wavenumber = 1.0 / energy2wavelength(energy)
        return wavenumber * self._directions[:, :2]

    def reciprocal_sampling(self, energy: float) -> float:
        """Reciprocal-space sampling of the detector grid [1 / Å].

        Parameters
        ----------
        energy : float
            Electron energy [eV].

        Returns
        -------
        sampling : float

        Raises
        ------
        RuntimeError
            If the detector was built from explicit directions, which have no
            regular sampling.
        """
        if self._gpts is None or self._max_angle is None:
            raise RuntimeError(
                "a detector built from explicit directions has no regular sampling"
            )

        wavenumber = 1.0 / energy2wavelength(energy)
        return 2.0 * np.sin(self._max_angle * 1e-3) * wavenumber / self._gpts
