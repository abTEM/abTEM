"""Module for simulating beam tilt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from abtem.core.axes import AxisAlignedTiltAxis, AxisMetadata, TiltAxis
from abtem.core.backend import get_array_module
from abtem.distributions import (
    BaseDistribution,
    EnsembleFromDistributions,
    MultidimensionalDistribution,
    from_values,
    validate_distribution,
)
from abtem.transform import ArrayObjectTransform, CompositeArrayObjectTransform

if TYPE_CHECKING:
    from abtem.waves import Waves


def _validate_tilt(
    tilt: BaseDistribution | tuple[float, float] | np.ndarray,
) -> BeamTilt | CompositeArrayObjectTransform:
    """Validate that the given tilt is correctly defined."""
    if isinstance(tilt, MultidimensionalDistribution):
        raise NotImplementedError

    if isinstance(tilt, BaseDistribution):
        return BeamTilt(tilt)

    elif isinstance(tilt, (tuple, list)):
        assert len(tilt) == 2

        transforms = []
        for tilt_component, direction in zip(tilt, ("x", "y")):
            transforms.append(
                AxisAlignedBeamTilt(tilt=tilt_component, direction=direction)
            )

        tilt = CompositeArrayObjectTransform(transforms)
    elif isinstance(tilt, np.ndarray):
        return BeamTilt(tilt)

    return tilt


def _get_tilt_axes(waves) -> tuple[TiltAxis | AxisAlignedTiltAxis, ...]:
    return tuple(axis for axis in waves.ensemble_axes_metadata if hasattr(axis, "tilt"))


def precession_tilts(
    precession_angle: float,
    num_samples: int,
    min_azimuth: float = 0.0,
    max_azimuth: float = 2 * np.pi,
    endpoint: bool = False,
):
    """
    Tilts for electron precession at a given precession angle.

    Parameters
    ----------
    precession_angle : float
        Precession angle [mrad].
    num_samples : int
        Number of tilt samples.
    min_azimuth : float, optional
        Minimum azimuthal angle [rad]. Default is 0.
    max_azimuth : float, optional
        Maximum azimuthal angle [rad]. Default is $2 \\pi$.
    endpoint
        If True, end is `max_azimuth`. Otherwise, it is not included. Default is False.

    Returns
    -------
    array_of_tilts : 2D array
        Array of xy-tilt angles [mrad].
    """
    azimuthal_angles = np.linspace(
        min_azimuth, max_azimuth, num=num_samples, endpoint=endpoint
    )

    tilt_x = precession_angle * np.cos(azimuthal_angles)
    tilt_y = precession_angle * np.sin(azimuthal_angles)

    return np.array([tilt_x, tilt_y], dtype=float).T


def array_tilts(
        polar_angles: list[float],
        nums_azimuthal: list[int],
):
    """
    An array of tilt values at different polar tilts for e.g. tilt-corrected DPC.

    Parameters
    ----------
    polar_angles : list[float]
        List of polar angles [mrad].
    nums_azimuthal : list[int]
        List of the number of azimuthal tilts per polar angle.

    Returns
    -------
    array_of_tilts : 2D array
        Array of xy-tilt angles [mrad].
    """

    assert len(polar_angles) == len(nums_azimuthal), \
        "The number of azimuthal angles needs to be provided for each polar angle!"

    # Doing this manually with lists as the number of tilts per radius may vary.
    tilts_x = []
    tilts_y = []

    # Creates a list of lists of x and y tilts.
    for i in range(len(polar_angles)):
        azimuthal_angles = np.linspace(
            0, 2 * np.pi, num=nums_azimuthal[i], endpoint=False
        )
        tilts_x.append(polar_angles[i] * np.cos(azimuthal_angles))
        tilts_y.append(polar_angles[i] * np.sin(azimuthal_angles))

    # Flattens the list of lists of tilts to one long list for each x and y.
    tilts_x = [x for xs in tilts_x for x in xs]
    tilts_y = [y for ys in tilts_y for y in ys]

    return np.array([tilts_x, tilts_y], dtype=float).T


class BaseBeamTilt(EnsembleFromDistributions, ArrayObjectTransform):
    def _out_metadata(self, waves):
        metadata = super()._out_metadata(waves)[0]

        if "base_tilt_x" in waves.metadata:
            metadata["base_tilt_x"] += waves.metadata["base_tilt_x"]

        if "base_tilt_y" in waves.metadata:
            metadata["base_tilt_y"] += waves.metadata["base_tilt_y"]

        return (metadata,)

    def _calculate_new_array(self, waves: Waves) -> np.ndarray | tuple[np.ndarray, ...]:
        xp = get_array_module(waves.device)
        array = waves.array[(None,) * len(self.ensemble_shape)]
        array = xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))
        return array

    def apply(self, waves: Waves) -> Waves:
        """
        Apply tilt(s) to (an ensemble of) wave function(s).

        Parameters
        ----------
        waves : Waves
            The waves to transform.
        in_place: bool, optional
            If True, the array representing the waves may be modified in-place.

        Returns
        -------
        waves_with_tilt : Waves
        """

        return self._apply(waves)


class BeamTilt(BaseBeamTilt):
    """
    Class describing beam tilt.

    Parameters
    ----------
    tilt : tuple of float
        Tilt along the `x` and `y` axes [mrad] with an optional spread of values.
    """

    def __init__(self, tilt: tuple[float, float] | BaseDistribution | np.ndarray):
        if isinstance(tilt, np.ndarray):
            tilt = from_values(tilt)

        self._tilt = tilt
        super().__init__(distributions=("tilt",))

    @property
    def tilt(self) -> tuple[float, float] | BaseDistribution:
        """Beam tilt angle [mrad]."""
        return self._tilt

    @property
    def metadata(self):
        """Metadata describing the tilt."""
        if isinstance(self.tilt, BaseDistribution):
            return {"base_tilt_x": 0.0, "base_tilt_y": 0.0}
        else:
            return {"base_tilt_x": self.tilt[0], "base_tilt_y": self.tilt[1]}

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """Metadata describing (an ensemble of) tilted wave function(s)."""
        if isinstance(self.tilt, BaseDistribution):
            return [
                TiltAxis(
                    label="tilt",
                    values=tuple(tuple(value) for value in self.tilt.values),
                    units="mrad",
                    _ensemble_mean=self.tilt.ensemble_mean,
                )
            ]
        else:
            return []


class AxisAlignedBeamTilt(BaseBeamTilt):
    """
    Class describing tilt(s) aligned with an axis.

    Parameters
    ----------
    tilt : array of BeamTilt
        Tilt along the given direction with an optional spread of values.
    direction : str
        Cartesian axis, should be either 'x' or 'y'.
    """

    def __init__(self, tilt: float | BaseDistribution = 0.0, direction: str = "x"):
        if isinstance(tilt, (np.ndarray, list, tuple)):
            tilt = validate_distribution(tilt)

        if not isinstance(tilt, BaseDistribution):
            tilt = float(tilt)

        self._tilt = tilt
        self._direction = direction
        super().__init__(distributions=("tilt",))

    @property
    def direction(self) -> str:
        """Tilt direction."""
        return self._direction

    @property
    def tilt(self) -> float | BaseDistribution:
        """Beam tilt [mrad]."""
        return self._tilt

    @property
    def metadata(self):
        if isinstance(self.tilt, BaseDistribution):
            return {f"base_tilt_{self._direction}": 0.0}
        else:
            return {f"base_tilt_{self._direction}": self._tilt}

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        if isinstance(self.tilt, BaseDistribution):
            return [
                AxisAlignedTiltAxis(
                    label=f"tilt_{self._direction}",
                    values=tuple(self.tilt.values),
                    direction=self._direction,
                    units="mrad",
                    _ensemble_mean=self.tilt.ensemble_mean,
                )
            ]
        else:
            return []
