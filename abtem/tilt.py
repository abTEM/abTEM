"""Module for simulating beam tilt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from abtem import distributions
from abtem.core.axes import AxisAlignedTiltAxis, AxisMetadata, TiltAxis
from abtem.core.backend import get_array_module
from abtem.distributions import (
    BaseDistribution,
    DistributionFromValues,
    MultidimensionalDistribution,
    validate_distribution,
)
from abtem.transform import WavesToWavesTransform

if TYPE_CHECKING:
    from abtem.waves import Waves


TiltType = float | BaseDistribution | np.ndarray
TiltType2D = tuple[TiltType, TiltType] | BaseDistribution | np.ndarray


def validate_tilt(
    tilt: TiltType2D,
) -> BeamTilt | BeamTilt2D | AxisAlignedBeamTilt:
    """Validate that the given tilt is correctly defined."""
    if isinstance(tilt, MultidimensionalDistribution):
        raise NotImplementedError

    if isinstance(tilt, BaseBeamTilt):
        return tilt

    # tilt_shape = np.array(tilt, dtype=float).shape

    # if len(tilt_shape) == 2 and not tilt_shape[1] == 2:
    #    raise ValueError("Tilt should be a 1D or Nx2 array.")

    # if len(tilt_shape) == 1 and not len(tilt) == 2:
    #    raise ValueError("Tilt should be a 1D or Nx2 array.")

    validated_tilt: BeamTilt | BeamTilt2D | AxisAlignedBeamTilt
    if isinstance(tilt, BaseDistribution):
        validated_tilt = BeamTilt(tilt)

    elif isinstance(tilt, (tuple, list)):
        assert len(tilt) == 2

        # tilt_x = AxisAlignedBeamTilt(tilt[0], direction="x")
        # tilt_y = AxisAlignedBeamTilt(tilt[1], direction="y")

        validated_tilt = BeamTilt2D(tilt_x=tilt[0], tilt_y=tilt[1])
    elif isinstance(tilt, np.ndarray):
        validated_tilt = BeamTilt(tilt)
    else:
        raise ValueError("Tilt should be a BaseDistribution, tuple, or numpy array.")

    return validated_tilt


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


class BaseBeamTilt(WavesToWavesTransform):
    def _out_metadata(self, waves):
        metadata = super()._out_metadata(waves)[0]

        if "base_tilt_x" in waves.metadata:
            metadata["base_tilt_x"] += waves.metadata["base_tilt_x"]

        if "base_tilt_y" in waves.metadata:
            metadata["base_tilt_y"] += waves.metadata["base_tilt_y"]

        return (metadata,)

    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        xp = get_array_module(waves.device)
        array = waves.array[(None,) * len(self.ensemble_shape)]
        array = xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))
        return array

    def apply(self, waves: Waves, max_batch: int | str = "auto") -> Waves:
        """
        Apply tilt(s) to (an ensemble of) wave function(s).

        Parameters
        ----------
        waves : Waves
            The wave function(s) to apply the tilt to.
        max_batch : int or str, optional
            The maximum batch size used in the calculation. If 'auto', the batch size is
            automatically determined. Default is 'auto'.

        Returns
        -------
        waves_with_tilt : Waves
        """
        return super().apply(waves, max_batch=max_batch)


class BeamTilt(BaseBeamTilt):
    """
    Class describing beam tilt.

    Parameters
    ----------
    tilt : tuple of float
        Tilt along the `x` and `y` axes [mrad] with an optional spread of values.
    """

    def __init__(self, tilt: tuple[float, float] | BaseDistribution | np.ndarray):
        if isinstance(tilt, BaseDistribution):
            tilt = tilt
        elif isinstance(tilt, (np.ndarray, list, tuple)):
            tilt_shape = np.array(tilt, dtype=float).shape
            if len(tilt_shape) == 2:
                if not tilt_shape[1] == 2:
                    raise ValueError("Tilt should be a 1D or Nx2 array.")

                if isinstance(tilt, np.ndarray):
                    tilt = distributions.from_values(tilt)
            elif len(tilt_shape) == 1:
                assert len(tilt) == 2
                tilt = tuple(tilt)
            else:
                raise ValueError("Tilt should be a 1D or Nx2 array.")

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


class AxisAlignedBeamTilt(DistributionFromValues):
    """
    Class describing tilt(s) aligned with an axis.

    Parameters
    ----------
    tilt : array of BeamTilt
        Tilt along the given direction with an optional spread of values.
    direction : str
        Cartesian axis, should be either 'x' or 'y'.
    """

    def __init__(self, tilt: TiltType = 0.0, direction: str = "x"):
        if isinstance(tilt, (np.ndarray, list, tuple)):
            tilt = validate_distribution(tilt)

        if not isinstance(tilt, BaseDistribution):
            tilt = float(tilt)

        self._tilt = tilt
        self._direction = direction

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


class BeamTilt2D(BaseBeamTilt):
    """
    Class describing 2D beam tilt.

    Parameters
    ----------
    tilt : array of BeamTilt
        Tilt along the `x` and `y` axes [mrad] with an optional spread of values.
    """

    def __init__(self, tilt_x: BaseDistribution, tilt_y: BaseDistribution):
        tilt_x = validate_distribution(tilt_x)
        tilt_y = validate_distribution(tilt_y)

        self._tilt_x = tilt_x
        self._tilt_y = tilt_y
        super().__init__(distributions=("tilt_x", "tilt_y"))

    @property
    def ensemble_shape(self) -> tuple[int, int]:
        if isinstance(self.tilt_x, BaseDistribution):
            ensemble_shape_x = self.tilt_x.shape
        else:
            ensemble_shape_x = ()

        if isinstance(self.tilt_y, BaseDistribution):
            ensemble_shape_y = self.tilt_y.shape
        else:
            ensemble_shape_y = ()

        return ensemble_shape_x + ensemble_shape_y

    @property
    def shape(self) -> tuple[int, int]:
        return self.ensemble_shape

    @property
    def tilt_x(self) -> BaseDistribution | float:
        """Beam tilt along the x-axis."""
        return self._tilt_x

    @property
    def tilt_y(self) -> BaseDistribution | float:
        """Beam tilt along the y-axis."""
        return self._tilt_y

    @property
    def metadata(self):
        metadata = {}
        if isinstance(self.tilt_x, BaseDistribution):
            metadata["base_tilt_x"] = 0.0
        else:
            metadata["base_tilt_x"] = self.tilt_x

        if isinstance(self.tilt_y, BaseDistribution):
            metadata["base_tilt_y"] = 0.0
        else:
            metadata["base_tilt_y"] = self.tilt_y

        return metadata

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata: list[AxisMetadata] = []

        if isinstance(self.tilt_x, BaseDistribution):
            ensemble_axes_metadata.append(
                AxisAlignedTiltAxis(
                    label="tilt_x",
                    values=tuple(self.tilt_x.values),
                    units="mrad",
                    direction="x",
                    _ensemble_mean=self.tilt_x.ensemble_mean,
                )
            )

        if isinstance(self.tilt_y, BaseDistribution):
            ensemble_axes_metadata.append(
                AxisAlignedTiltAxis(
                    label="tilt_y",
                    values=tuple(self.tilt_y.values),
                    units="mrad",
                    direction="y",
                    _ensemble_mean=self.tilt_y.ensemble_mean,
                )
            )

        return ensemble_axes_metadata
