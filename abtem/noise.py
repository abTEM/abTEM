"""Module for applying noise to measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore

from abtem.core.axes import NonLinearAxis, SampleAxis
from abtem.core.utils import get_dtype
from abtem.distributions import BaseDistribution, validate_distribution
from abtem.inelastic.phonons import validate_seeds
from abtem.transform import EnsembleTransform

if TYPE_CHECKING:
    from abtem.array import ArrayObject
    from abtem.core.axes import AxisMetadata


class NoiseTransform(EnsembleTransform):
    def __init__(
        self,
        dose: float | np.ndarray | BaseDistribution,
        samples: Optional[int] = None,
        seeds: Optional[int | tuple[int, ...]] = None,
    ):
        self._dose = validate_distribution(dose)

        seeds_distribution: None | int | BaseDistribution
        if (isinstance(seeds, int) or seeds is None) and (
            samples is None or samples == 1
        ):
            seeds_distribution = seeds

        elif seeds is not None or samples is not None:
            seeds = validate_seeds(seeds, samples)
            seeds_distribution = validate_distribution(seeds)

        else:
            seeds_distribution = None

        self._seeds = seeds_distribution

        super().__init__(
            distributions=(
                "dose",
                "seeds",
            )
        )

    @property
    def dose(self) -> float | np.ndarray | BaseDistribution:
        return self._dose

    @property
    def seeds(self) -> Optional[BaseDistribution | int]:
        return self._seeds

    @property
    def samples(self) -> int:
        if isinstance(self.seeds, BaseDistribution):
            return len(self.seeds.values)
        else:
            return 1

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata: list[AxisMetadata] = []

        if isinstance(self.dose, BaseDistribution):
            ensemble_axes_metadata += [
                NonLinearAxis(label="Dose", values=tuple(self.dose.values), units="e")
            ]

        if isinstance(self.seeds, BaseDistribution):
            ensemble_axes_metadata += [SampleAxis()]

        return ensemble_axes_metadata

    @property
    def metadata(self) -> dict:
        return {"units": "", "label": "electron counts"}

    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        array = array_object._eager_array

        if isinstance(self.seeds, BaseDistribution):
            array = np.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        if isinstance(self.dose, BaseDistribution):
            dose = np.array(self.dose.values, dtype=get_dtype())
            array = array[None] * np.expand_dims(
                dose, tuple(range(1, len(array.shape) + 1))
            )
        else:
            array = array * np.array(self.dose, dtype=get_dtype())

        if isinstance(self.seeds, BaseDistribution):
            seed = sum(self.seeds.values)
        else:
            seed = self.seeds

        seed_rng = np.random.default_rng(seed=seed)

        randomized_seed = int(seed_rng.integers(np.iinfo(np.int32).max))

        poisson_rng = np.random.RandomState(seed=randomized_seed)

        array = np.clip(array, a_min=0.0, a_max=None)

        array = poisson_rng.poisson(array).astype(get_dtype())

        return array

    def apply(
        self, array_object: ArrayObject, max_batch: int | str = "auto"
    ) -> ArrayObject:
        new_array_object = array_object.apply_transform(self)
        if TYPE_CHECKING:
            assert isinstance(new_array_object, self.__class__)
        return new_array_object


def _pixel_times(
    dwell_time: float, flyback_time: float, shape: tuple[int, int]
) -> np.ndarray:
    """
    Pixel times internal function

    Function for calculating scan pixel times.

    Parameters
    ----------
    dwell_time : float
        Dwell time on a single pixel in s.
    flyback_time : float
        Flyback time for the scanning probe at the end of each scan line in s.
    shape : two ints
        Dimensions of a scan in pixels.
    """

    line_time = (dwell_time * shape[0]) + flyback_time
    slow_time = np.tile(
        np.linspace(line_time, shape[1] * line_time, shape[1]), (shape[0], 1)
    )

    fast_time = np.tile(
        np.linspace(
            (line_time - flyback_time) / shape[1], line_time - flyback_time, shape[0]
        )[:, None],
        (1, shape[1]),
    )
    return slow_time + fast_time


def _single_axis_distortion(
    time: np.ndarray,
    max_frequency: float,
    num_components: int,
    seed: Optional[int] = None,
):
    """
    Single axis distortion internal function

    Function for emulating a scan distortion along a single axis.

    Parameters
    ----------
    time : np.ndarray
        Time constant for the distortion in s.
    max_frequency : float
        Maximum noise frequency in 1 / s.
    num_components: int
        Number of frequency components.
    """

    rng = np.random.RandomState(seed=seed)
    frequencies = rng.rand(num_components, 1, 1) * max_frequency
    amplitudes = rng.rand(num_components, 1, 1) / np.sqrt(frequencies)
    displacements = rng.rand(num_components, 1, 1) / frequencies
    return (amplitudes * np.sin(2 * np.pi * (time + displacements) * frequencies)).sum(
        axis=0
    )


def _make_displacement_field(
    time: np.ndarray,
    max_frequency: float,
    num_components: int,
    rms_power: float,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Displacement field creation internal function

    Function to create a displacement field to emulate 2D scan distortion.

    Parameters
    ----------
    time : np.ndarray
       Time constant for the distortion in s.
    max_frequency : float
       Maximum noise frequency in 1 / s.
    num_components : int
       Number of frequency components.
    rms_power : float
       Root-mean-square power of the distortion.
    """

    profile_x = _single_axis_distortion(time, max_frequency, num_components, seed=seed)
    profile_y = _single_axis_distortion(time, max_frequency, num_components, seed=seed)

    x_mag_deviation = np.gradient(profile_x, axis=1)
    y_mag_deviation = np.gradient(profile_y, axis=0)

    frame_mag_deviation = (1 + x_mag_deviation) * (1 + y_mag_deviation) - 1
    frame_mag_deviation = np.sqrt(np.mean(frame_mag_deviation**2))

    # 235.5 = 2.355 * 100 %; 2.355 converts from 1/e width to FWHM

    profile_x *= rms_power / (2.355 * 100 * frame_mag_deviation)
    profile_y *= rms_power / (2.355 * 100 * frame_mag_deviation)

    return profile_x, profile_y


def _apply_displacement_field(
    image: np.ndarray, distortion_x: np.ndarray, distortion_y: np.ndarray
) -> np.ndarray:
    """
    Displacement field applying function

    Function to apply a displacement field to an image.

    Parameters
    ----------
    image : ndarray
        Image array.
    distortion_x : ndarray
        Displacement field along the x-axis.
    distortion_y : ndarray
        Displacement field along the y-axis.
    """

    x = np.arange(0, image.shape[0])
    y = np.arange(0, image.shape[1])

    interpolating_function = RegularGridInterpolator([x, y], image, fill_value=None)

    y, x = np.meshgrid(y, x)
    p = np.array([(x + distortion_x).ravel(), (y + distortion_y).ravel()]).T

    # p[:, 0] = np.clip(p[:, 0], 0, x.max())
    p[:, 0] = p[:, 0] % x.max()
    # p[:, 1] = np.clip(p[:, 1], 0, y.max())
    p[:, 1] = p[:, 1] % y.max()

    warped = interpolating_function(p)
    return warped.reshape(image.shape)


# def apply_scan_noise(
#     measurement: np.ndarray,
#     dwell_time: float,
#     flyback_time: float,
#     max_frequency: float,
#     rms_power: float,
#     num_components: int = 200,
# ):
#     """
#     Add scan noise to a measurement.

#     Parameters
#     ----------
#     measurement: Measurement object or 2d array
#         The measurement to add noise to.
#     dwell_time: float
#         Dwell time on a single pixel in s.
#     flyback_time: float
#         Flyback time for the scanning probe at the end of each scan line in s.
#     max_frequency: float
#         Maximum noise frequency in 1 / s.
#     rms_power: float
#         Root-mean-square power of the distortion in unit of percent.
#     num_components: int, optional
#         Number of frequency components. More components will be more 'white' but will
#         take longer.

#     Returns
#     -------
#     measurement: Measurement object
#         The noisy measurement.
#     """

#     time = _pixel_times(dwell_time, flyback_time, array.T.shape)
#     displacement_x, displacement_y = _make_displacement_field(
#         time, max_frequency, num_components, rms_power
#     )

#     array = _apply_displacement_field(array.T, displacement_x, displacement_y)
#     return array.T


class ScanNoiseTransform(EnsembleTransform):
    def __init__(
        self,
        rms_power: float | np.ndarray | BaseDistribution,
        dwell_time: float,
        flyback_time: float,
        samples: Optional[int] = None,
        max_frequency: float = 500,
        num_components: int = 1000,
        seeds: Optional[int | tuple[int, ...]] = None,
    ):
        self._rms_power = validate_distribution(rms_power)
        self._dwell_time = dwell_time
        self._flyback_time = flyback_time
        self._max_frequency = max_frequency
        self._num_components = num_components

        if samples is None and seeds is None:
            samples = 1

        if seeds is not None:
            seeds_distribution = validate_distribution(validate_seeds(seeds, samples))
        else:
            seeds_distribution = None

        self._seeds = seeds_distribution

        super().__init__(
            distributions=(
                "dose",
                "seeds",
            )
        )

    @property
    def rms_power(self) -> float | np.ndarray | BaseDistribution:
        return self._rms_power

    @property
    def dwell_time(self) -> float:
        return self._dwell_time

    @property
    def flyback_time(self) -> float:
        return self._flyback_time

    @property
    def max_frequency(self) -> float:
        return self._max_frequency

    @property
    def num_components(self) -> int:
        return self._num_components

    @property
    def seeds(self) -> Optional[BaseDistribution]:
        return self._seeds

    @property
    def samples(self) -> int:
        if self.seeds is not None:
            return len(self.seeds.values)
        else:
            return 1

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata: list[AxisMetadata] = []
        if isinstance(self.rms_power, BaseDistribution):
            ensemble_axes_metadata += [
                NonLinearAxis(
                    label="RMS power",
                    values=tuple(self.rms_power.values),
                    units=r"\%",
                )
            ]

        if isinstance(self.seeds, BaseDistribution):
            ensemble_axes_metadata += [SampleAxis()]

        return ensemble_axes_metadata

    @property
    def metadata(self) -> dict:
        return {"units": "", "label": "electron counts"}

    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        array = array_object._eager_array
        base_shape = array_object.base_shape
        assert len(base_shape) == 2

        if isinstance(self.seeds, BaseDistribution):
            array = np.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        time = _pixel_times(self.dwell_time, self.flyback_time, base_shape)

        if isinstance(self.rms_power, BaseDistribution):
            rms_powers = np.array(self.rms_power.values, dtype=get_dtype())
        else:
            rms_powers = np.array([self.rms_power], dtype=get_dtype())

        if self.seeds is not None:
            seed = sum(self.seeds.values)
        else:
            seed = None

        arrays = []
        for rms_power in rms_powers:
            inner_array = np.zeros_like(array)
            for i in np.ndindex(array.shape[:-2]):
                displacement_x, displacement_y = _make_displacement_field(
                    time,
                    self.max_frequency,
                    self.num_components,
                    rms_power,
                    seed=seed,
                )

                inner_array[i] = _apply_displacement_field(
                    array[i], displacement_x, displacement_y
                )

            arrays.append(inner_array)

        if isinstance(self.rms_power, BaseDistribution):
            array = np.stack(arrays, axis=0)
        else:
            array = arrays[0]

        return array
