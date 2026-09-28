"""Module for applying noise to measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore

from abtem.core.axes import NonLinearAxis, SampleAxis
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.utils import get_dtype
from abtem.distributions import BaseDistribution, validate_distribution
from abtem.inelastic.phonons import validate_seeds
from abtem.transform import EnsembleTransform

if TYPE_CHECKING:
    from abtem.array import ArrayObject
    from abtem.core.axes import AxisMetadata


def _poisson_sample(
    expected: np.ndarray,
    base_dims: int,
    offset: tuple[int, ...],
    seeds: int | tuple[int, ...],
    sample_axis: Optional[int],
) -> np.ndarray:
    """
    Draw Poisson counts, one base array (image, diffraction pattern, ...) at a
    time, each from its own random generator keyed on its global ensemble
    index. The draw of a base array therefore does not depend on how the full
    array is chunked, so a lazy result equals the eager one for the same seeds.

    Parameters
    ----------
    expected : np.ndarray
        Expected counts; a block of the full array when evaluated lazily.
    base_dims : int
        Number of trailing base dimensions, sampled whole.
    offset : tuple of int
        Global index of the first entry of the block along each ensemble axis.
    seeds : int or tuple of int
        A single seed, or one seed per entry of the sample axis.
    sample_axis : int, optional
        The ensemble axis indexing `seeds`, if there is one.
    """
    xp = get_array_module(expected)
    # Poisson sampling requires CPU arrays; move back to GPU afterwards
    expected = np.clip(asnumpy(expected), a_min=0.0, a_max=None)
    counts = np.empty(expected.shape, dtype=get_dtype())

    for index in np.ndindex(expected.shape[: expected.ndim - base_dims]):
        global_index = tuple(i + o for i, o in zip(index, offset))
        seed = seeds if sample_axis is None else seeds[global_index[sample_axis]]
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=global_index)
        )
        counts[index] = rng.poisson(expected[index])

    return xp.asarray(counts)


def _poisson_sample_block(block, block_info=None, **kwargs):
    base_dims = kwargs["base_dims"]
    location = block_info[None]["array-location"]
    offset = tuple(start for start, _ in location[: len(location) - base_dims])
    return _poisson_sample(block, offset=offset, **kwargs)


class NoiseTransform(EnsembleTransform):
    # `samples` is implied by `seeds` (one seed per sample), so it is not passed
    # on when the transform is rebuilt for a chunk: a chunk receives a sub-block
    # of the seeds, which would not match the full sample count
    _exclude_from_copy = ("samples",)

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
        return {"units": "electrons", "label": "Counts"}

    def _expected_counts(self, array: np.ndarray) -> np.ndarray:
        xp = get_array_module(array)

        if isinstance(self.seeds, BaseDistribution):
            array = xp.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        if isinstance(self.dose, BaseDistribution):
            dose = xp.array(self.dose.values, dtype=get_dtype())
            array = array[None] * xp.expand_dims(
                dose, tuple(range(1, len(array.shape) + 1))
            )
        else:
            array = array * xp.asarray(self.dose, dtype=get_dtype())

        return array

    def _sampling_kwargs(self, base_dims: int) -> dict:
        if isinstance(self.seeds, BaseDistribution):
            # the sample axis follows the dose axis, see _expected_counts
            seeds = tuple(int(seed) for seed in self.seeds.values)
            sample_axis = 1 if isinstance(self.dose, BaseDistribution) else 0
        else:
            # unseeded: draw fresh entropy once, shared by all blocks, so that
            # blocks are not correlated with each other
            seeds = (
                np.random.SeedSequence().entropy if self.seeds is None else self.seeds
            )
            sample_axis = None

        return {"base_dims": base_dims, "seeds": seeds, "sample_axis": sample_axis}

    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        # called on the whole (eager) array, which starts at the global origin;
        # lazy arrays are sampled per block in `apply`
        expected = self._expected_counts(array_object._eager_array)
        base_dims = len(array_object.base_shape)
        return _poisson_sample(
            expected,
            offset=(0,) * (expected.ndim - base_dims),
            **self._sampling_kwargs(base_dims),
        )

    def apply(
        self, array_object: ArrayObject, max_batch: int | str = "auto"
    ) -> ArrayObject:
        if not array_object.is_lazy:
            new_array_object = array_object.apply_transform(self)
            if TYPE_CHECKING:
                assert isinstance(new_array_object, self.__class__)
            return new_array_object

        # A block of a lazy array does not know its global position, which the
        # chunk-independent draw needs: build the expected counts through the
        # ensemble machinery, then sample them with dask, which does.
        expected = array_object.apply_transform(
            _PoissonExpectedCounts(**self._copy_kwargs()), max_batch=max_batch
        )
        array = expected.array
        base_dims = len(expected.base_shape)
        array = array.rechunk(
            array.chunks[: array.ndim - base_dims]
            + tuple((n,) for n in array.shape[array.ndim - base_dims :])
        )
        xp = get_array_module(array)
        array = array.map_blocks(
            _poisson_sample_block,
            dtype=get_dtype(),
            meta=xp.array((), dtype=get_dtype()),
            **self._sampling_kwargs(base_dims),
        )

        new_array_object = expected.__class__.from_array_and_metadata(
            array, axes_metadata=expected.axes_metadata, metadata=expected.metadata
        )
        if expected.device == "gpu":
            new_array_object._device = "gpu"
        return new_array_object


class _PoissonExpectedCounts(NoiseTransform):
    """The deterministic part of `NoiseTransform`: sample tiling and dose
    scaling, without the Poisson draw. `NoiseTransform.apply` uses it for lazy
    arrays."""

    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        return self._expected_counts(array_object._eager_array)


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
        Dimensions of a scan in pixels. The first axis (x) is the fast scan axis,
        i.e. a scan line runs along axis 0, and the second axis (y) is the slow
        axis indexing the scan lines.

    Returns
    -------
    times : np.ndarray
        Time at each pixel. Consecutive pixels along a line are separated by
        `dwell_time`, and consecutive lines by `shape[0] * dwell_time +
        flyback_time`.
    """

    line_time = (dwell_time * shape[0]) + flyback_time
    slow_time = np.tile(
        np.linspace(line_time, shape[1] * line_time, shape[1]), (shape[0], 1)
    )

    fast_time = np.tile(
        np.linspace(dwell_time, line_time - flyback_time, shape[0])[:, None],
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
    time : numpy.ndarray
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
    time : numpy.ndarray
       Time constant for the distortion in s.
    max_frequency : float
       Maximum noise frequency in 1 / s.
    num_components : int
       Number of frequency components.
    rms_power : float
       Root-mean-square power of the distortion.
    seed : int, optional
       Seed for the random distortions. The x and y distortions are drawn
       independently of each other.

    Returns
    -------
    profile_x, profile_y : np.ndarray
       Displacements in pixels along axis 0 (x) and axis 1 (y).
    """

    if seed is None:
        seed_x = seed_y = None
    else:
        seed_x, seed_y = (
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(seed).spawn(2)
        )

    profile_x = _single_axis_distortion(
        time, max_frequency, num_components, seed=seed_x
    )
    profile_y = _single_axis_distortion(
        time, max_frequency, num_components, seed=seed_y
    )

    # profile_x (profile_y) displaces the image along axis 0 (axis 1), see
    # _apply_displacement_field, so its magnification deviation is the
    # derivative along that same axis
    x_mag_deviation = np.gradient(profile_x, axis=0)
    y_mag_deviation = np.gradient(profile_y, axis=1)

    frame_mag_deviation = (1 + x_mag_deviation) * (1 + y_mag_deviation) - 1
    frame_mag_deviation = np.sqrt(np.mean(frame_mag_deviation**2))

    # 235.5 = 2.355 * 100 %; 2.355 converts a standard deviation to a FWHM

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


class ScanNoiseTransform(EnsembleTransform):
    # `samples` is implied by `seeds` (one seed per sample), so it is not passed
    # on when the transform is rebuilt for a chunk: a chunk receives a sub-block
    # of the seeds, which would not match the full sample count
    _exclude_from_copy = ("samples",)

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

        # one seed per sample; unseeded samples > 1 get random per-sample seeds
        if seeds is not None or samples > 1:
            seeds_distribution = validate_distribution(validate_seeds(seeds, samples))
        else:
            seeds_distribution = None

        self._seeds = seeds_distribution

        super().__init__(
            distributions=(
                "rms_power",
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
        return {"units": "electrons", "label": "Counts"}

    def apply(
        self, array_object: ArrayObject, max_batch: int | str = "auto"
    ) -> ArrayObject:
        return array_object.apply_transform(self, max_batch=max_batch)

    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        array = array_object._eager_array
        xp = get_array_module(array)
        # the distortion is interpolated with scipy, which requires CPU arrays;
        # move back to the original device afterwards
        array = asnumpy(array)
        base_shape = array_object.base_shape
        assert len(base_shape) == 2

        if isinstance(self.seeds, BaseDistribution):
            array = np.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        time = _pixel_times(self.dwell_time, self.flyback_time, base_shape)

        if isinstance(self.rms_power, BaseDistribution):
            rms_powers = np.array(self.rms_power.values, dtype=get_dtype())
        else:
            rms_powers = np.array([self.rms_power], dtype=get_dtype())

        arrays = []
        for rms_power in rms_powers:
            inner_array = np.zeros_like(array)
            for i in np.ndindex(array.shape[:-2]):
                # the leading axis is the sample axis; each sample uses its own seed
                if self.seeds is not None:
                    seed = int(self.seeds.values[i[0]])
                else:
                    seed = None

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

        return xp.asarray(array)
