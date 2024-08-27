"""Module for simulating beam tilt."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abtem.core.axes import AxisMetadata, TiltAxis, AxisAlignedTiltAxis
from abtem.core.backend import get_array_module
from abtem.transform import CompositeArrayObjectTransform, ArrayObjectTransform
from abtem.distributions import (
    BaseDistribution,
    MultidimensionalDistribution,
    EnsembleFromDistributions,
    from_values,
    validate_distribution,
)

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


def _get_tilt_axes(waves):
    return tuple(
        i
        for i, axis in enumerate(waves.ensemble_axes_metadata)
        if hasattr(axis, "tilt")
    )


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
        Precession angle [rad].
    num_samples : int
        Number of tilt samples.
    min_azimuth : float, optional
        Minmimum azimuthal angle [rad]. Default is 0.
    max_azimuth : float, optional
        Maximum azimuthal angle [rad]. Default is $2 \\pi$.
    endpoint
        If True, end is `max_azimuth`. Otherwise, it is not included. Default is False.

    Returns
    -------
    array_of_tilts : 2D array
        Array of xy-tilt angles [rad].
    """
    azimuthal_angles = np.linspace(
        min_azimuth, max_azimuth, num=num_samples, endpoint=endpoint
    )

    tilt_x = precession_angle * np.cos(azimuthal_angles)
    tilt_y = precession_angle * np.sin(azimuthal_angles)

    return np.array([tilt_x, tilt_y], dtype=float).T


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

        # kwargs = waves._copy_kwargs(exclude=("array",))
        # kwargs["array"] = array
        # kwargs["metadata"] = self._out_metadata(waves)
        # kwargs["ensemble_axes_metadata"] = (
        #     self.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
        # )

        # return waves.__class__(**kwargs)


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
