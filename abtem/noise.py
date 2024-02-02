"""Module for applying noise to measurements."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from abtem.core.axes import (
    NonLinearAxis,
    SampleAxis,
)
from abtem.core.backend import get_array_module
from abtem.distributions import validate_distribution, BaseDistribution
from abtem.inelastic.phonons import _validate_seeds
from abtem.transform import EnsembleTransform


class NoiseTransform(EnsembleTransform):
    def __init__(
        self,
        dose: float | np.ndarray | BaseDistribution,
        samples: int = None,
        seeds: int | tuple[int, ...] = None,
    ):
        self._dose = validate_distribution(dose)

        if samples is None and seeds is None:
            samples = 1

        if seeds is None and samples > 1:
            seeds = _validate_seeds(seeds, samples)
            seeds = validate_distribution(seeds)

        self._seeds = seeds

        super().__init__(
            distributions=(
                "dose",
                "seeds",
            )
        )

    @property
    def dose(self):
        return self._dose

    @property
    def seeds(self):
        return self._seeds

    @property
    def samples(self):
        if hasattr(self.seeds, "__len__"):
            return len(self.seeds)
        else:
            return 1

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = []

        if isinstance(self.dose, BaseDistribution):
            ensemble_axes_metadata += [
                NonLinearAxis(label="Dose", values=tuple(self.dose.values), units="e")
            ]

        if isinstance(self.seeds, BaseDistribution):
            ensemble_axes_metadata += [SampleAxis()]

        return ensemble_axes_metadata

    @property
    def metadata(self):
        return {"units": "", "label": "electron counts"}

    def _calculate_new_array(self, array_object) -> np.ndarray | tuple[np.ndarray, ...]:
        array = array_object.array
        xp = get_array_module(array)

        if isinstance(self.seeds, BaseDistribution):
            array = xp.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        if isinstance(self.dose, BaseDistribution):
            dose = xp.array(self.dose.values, dtype=xp.float32)
            array = array[None] * xp.expand_dims(
                dose, tuple(range(1, len(array.shape) + 1))
            )
        else:
            array = array * xp.array(self.dose, dtype=xp.float32)

        if isinstance(self.seeds, BaseDistribution):
            seed = sum(self.seeds.values)
        else:
            seed = self.seeds

        rng = xp.random.default_rng(seed=seed)

        randomized_seed = int(
            rng.integers(np.iinfo(np.int32).max)
        )  # fixes strange cupy bug

        rng = xp.random.RandomState(seed=randomized_seed)

        array = xp.clip(array, a_min=0.0, a_max=None)

        array = rng.poisson(array).astype(xp.float32)

        return array


def _pixel_times(dwell_time, flyback_time, shape):
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


def _single_axis_distortion(time, max_frequency, num_components):
    """
    Single axis distortion internal function

    Function for emulating a scan distortion along a single axis.

    Parameters
    ----------
    time : float
        Time constant for the distortion in s.
    max_frequency : float
        Maximum noise frequency in 1 / s.
    num_components: int
        Number of frequency components.
    """

    frequencies = np.random.rand(num_components, 1, 1) * max_frequency
    amplitudes = np.random.rand(num_components, 1, 1) / np.sqrt(frequencies)
    displacements = np.random.rand(num_components, 1, 1) / frequencies
    return (amplitudes * np.sin(2 * np.pi * (time + displacements) * frequencies)).sum(
        axis=0
    )


def _make_displacement_field(time, max_frequency, num_components, rms_power):
    """
    Displacement field creation internal function

    Function to create a displacement field to emulate 2D scan distortion.

    Parameters
    ----------
    time : float
       Time constant for the distortion in s.
    max_frequency : float
       Maximum noise frequency in 1 / s.
    num_components : int
       Number of frequency components.
    rms_power : float
       Root-mean-square power of the distortion.
    """

    profile_x = _single_axis_distortion(time, max_frequency, num_components)
    profile_y = _single_axis_distortion(time, max_frequency, num_components)

    x_mag_deviation = np.gradient(profile_x, axis=1)
    y_mag_deviation = np.gradient(profile_y, axis=0)

    frame_mag_deviation = (1 + x_mag_deviation) * (1 + y_mag_deviation) - 1
    frame_mag_deviation = np.sqrt(np.mean(frame_mag_deviation**2))

    # 235.5 = 2.355 * 100 %; 2.355 converts from 1/e width to FWHM

    profile_x *= rms_power / (2.355 * 100 * frame_mag_deviation)
    profile_y *= rms_power / (2.355 * 100 * frame_mag_deviation)

    return profile_x, profile_y


def _apply_displacement_field(image, distortion_x, distortion_y):
    """
    Displacement field applying function

    Function to apply a displacement field to an image.

    Parameters
    ----------
    image : ndarray
        Image array.
    distortion_x : ndarray
        Displacement field along the x axis.
    distortion_y : ndarray
        Displacement field along the y axis.
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


def apply_scan_noise(
    measurement,
    dwell_time: float,
    flyback_time: float,
    max_frequency: float,
    rms_power: float,
    num_components: int = 200,
):
    """
    Add scan noise to a measurement.

    Parameters
    ----------
    measurement: Measurement object or 2d array
        The measurement to add noise to.
    dwell_time: float
        Dwell time on a single pixel in s.
    flyback_time: float
        Flyback time for the scanning probe at the end of each scan line in s.
    max_frequency: float
        Maximum noise frequency in 1 / s.
    rms_power: float
        Root-mean-square power of the distortion in unit of percent.
    num_components: int, optional
        Number of frequency components. More components will be more 'white' but will take longer.

    Returns
    -------
    measurement: Measurement object
        The noisy measurement.
    """

    measurement = measurement.copy()
    if hasattr(measurement, "array"):
        array = measurement.array
    else:
        array = measurement

    time = _pixel_times(dwell_time, flyback_time, array.T.shape)
    displacement_x, displacement_y = _make_displacement_field(
        time, max_frequency, num_components, rms_power
    )

    array = _apply_displacement_field(array[:].T, displacement_x, displacement_y)
    measurement.array[:] = array.T
    return measurement


class ScanNoiseTransform(EnsembleTransform):
    def __init__(
        self,
        rms_power: float | np.ndarray | BaseDistribution,
        dwell_time: float,
        flyback_time: float,
        samples: int = None,
        max_frequency: float = 500,
        num_components: int = 1000,
        seeds: int | tuple[int, ...] = None,
    ):
        self._rms_power = validate_distribution(rms_power)
        self._dwell_time = dwell_time
        self._flyback_time = flyback_time
        self._max_frequency = max_frequency
        self._num_components = num_components

        if samples is None and seeds is None:
            samples = 1

        if seeds is None and samples > 1:
            seeds = _validate_seeds(seeds, samples)
            seeds = validate_distribution(seeds)

        self._seeds = seeds

        super().__init__(
            distributions=(
                "dose",
                "seeds",
            )
        )

    @property
    def rms_power(self):
        return self._rms_power

    @property
    def dwell_time(self):
        return self._dwell_time

    @property
    def flyback_time(self):
        return self._flyback_time

    @property
    def max_frequency(self):
        return self._max_frequency

    @property
    def num_components(self):
        return self._num_components

    @property
    def seeds(self):
        return self._seeds

    @property
    def samples(self):
        if hasattr(self.seeds, "__len__"):
            return len(self.seeds)
        else:
            return 1

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = []

        if isinstance(self.rms_power, BaseDistribution):
            ensemble_axes_metadata += [
                NonLinearAxis(
                    label="RMS power",
                    values=tuple(self.rms_power.values),
                    units="\%",
                )
            ]

        if isinstance(self.seeds, BaseDistribution):
            ensemble_axes_metadata += [SampleAxis()]

        return ensemble_axes_metadata

    @property
    def metadata(self):
        return {"units": "", "label": "electron counts"}

    def _calculate_new_array(self, array_object) -> np.ndarray | tuple[np.ndarray, ...]:
        array = array_object.array
        xp = get_array_module(array)

        if isinstance(self.seeds, BaseDistribution):
            array = xp.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        time = _pixel_times(self.dwell_time, self.flyback_time, array.shape[-2:])

        if isinstance(self.rms_power, BaseDistribution):
            rms_powers = xp.array(self.rms_power.values, dtype=xp.float32)
        else:
            rms_powers = xp.array([self.rms_power], dtype=xp.float32)

        arrays = []
        for rms_power in rms_powers:
            inner_array = np.zeros_like(array)
            for i in np.ndindex(array.shape[:-2]):
                displacement_x, displacement_y = _make_displacement_field(
                    time, self.max_frequency, self.num_components, rms_power
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
