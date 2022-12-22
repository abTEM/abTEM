"""Module for simulating beam tilt."""
from typing import Union, TYPE_CHECKING, Tuple, List

import dask.array as da
import numpy as np

from abtem.core.axes import AxisMetadata, TiltAxis, AxisAlignedTiltAxis
from abtem.core.backend import get_array_module
from abtem.core.transform import CompositeWaveTransform, WaveTransform
from abtem.distributions import (
    BaseDistribution,
    _AxisAlignedDistributionND,
    _EnsembleFromDistributionsMixin,
    _DistributionFromValues,
    from_values,
)

if TYPE_CHECKING:
    from abtem.waves import Waves


def validate_tilt(tilt):
    """Validate that the given tilt is correctly defined."""
    if isinstance(tilt, _AxisAlignedDistributionND):
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

        tilt = CompositeWaveTransform(transforms)
    elif isinstance(tilt, np.ndarray):

        return BeamTilt(tilt)

    return tilt


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
        Precession angle.
    num_samples : int

    min_azimuth
    max_azimuth
    endpoint

    Returns
    -------

    """
    azimuthal_angles = np.linspace(
        min_azimuth, max_azimuth, num=num_samples, endpoint=endpoint
    )

    tilt_x = precession_angle * np.cos(azimuthal_angles)
    tilt_y = precession_angle * np.sin(azimuthal_angles)

    return np.array([tilt_x, tilt_y], dtype=float).T


class BeamTilt(_EnsembleFromDistributionsMixin, WaveTransform):
    """
    Class describing beam tilt.

    Parameters
    ----------
    tilt : tuple of float
        Tilt along the `x` and `y` axes [mrad] with an optional spread of values.
    """

    def __init__(self, tilt: Union[Tuple[float, float], BaseDistribution, np.ndarray]):

        if isinstance(tilt, np.ndarray):
            tilt = from_values(tilt)

        self._tilt = tilt
        super().__init__(distributions=("tilt",))

    @property
    def tilt(self) -> Union[Tuple[float, float], BaseDistribution]:
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
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        """Metadata describing (an ensemble of) tilted wave function(s)."""
        if isinstance(self.tilt, BaseDistribution):
            return [
                TiltAxis(
                    label=f"tilt",
                    values=tuple(tuple(value) for value in self.tilt.values),
                    units="mrad",
                    _ensemble_mean=self.tilt.ensemble_mean,
                )
            ]

    def apply(self, waves: "Waves", overwrite_x: bool = False) -> "Waves":
        """Apply tilt(s) to (an ensamble of) wave function(s)."""
        xp = get_array_module(waves.device)

        array = waves.array[(None,) * len(self.ensemble_shape)]

        if waves.is_lazy:
            array = da.tile(array, self.ensemble_shape + (1,) * len(waves.shape))
        else:
            array = xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))

        kwargs = waves._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["metadata"] = {**kwargs["metadata"], **self.metadata}
        kwargs["ensemble_axes_metadata"] = (
                self.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
        )
        return waves.__class__(**kwargs)


class AxisAlignedBeamTilt(_EnsembleFromDistributionsMixin, WaveTransform):
    """
    Class describing tilt(s) aligned with an axis.

    Parameters
    ----------
    tilt : array of BeamTilt
        Tilt along the given direction with an optional spread of values.
    direction : str
        Cartesian axis, should be either 'x' or 'y'.
    """

    def __init__(
            self, tilt: Union[float, BaseDistribution] = 0.0, direction: str = "x"
    ):
        if not isinstance(tilt, BaseDistribution):
            tilt = float(tilt)
        self._tilt = tilt
        self._direction = direction
        super().__init__(distributions=("tilt",))

    @property
    def direction(self):
        return self._direction

    @property
    def tilt(self) -> Union[float, BaseDistribution]:
        """Beam tilt [mrad]."""
        return self._tilt

    @property
    def metadata(self):
        if isinstance(self.tilt, BaseDistribution):
            return {f"base_tilt_{self._direction}": 0.0}
        else:
            return {f"base_tilt_{self._direction}": self._tilt}

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
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

    def apply(self, waves: "Waves", overwrite_x: bool = False) -> "Waves":
        """Apply tilt(s) to (an ensamble of) wave function(s)."""
        xp = get_array_module(waves.device)

        if self.tilt == 0.:
            return waves

        array = waves.array[(None,) * len(self.ensemble_shape)]

        if waves.is_lazy:
            array = da.tile(array, self.ensemble_shape + (1,) * len(waves.shape))
        else:
            array = xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))

        kwargs = waves._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["metadata"] = {**kwargs["metadata"], **self.metadata}
        kwargs["ensemble_axes_metadata"] = (
                self.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
        )
        return waves.__class__(**kwargs)
