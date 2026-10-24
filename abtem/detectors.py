"""Module for describing the detection of transmitted waves and different detector
types."""

from __future__ import annotations

from abc import abstractmethod
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Type, TypeVar

import numpy as np

from abtem.core.axes import (
    AxisMetadata,
    EnergyAxis,
    LinearAxis,
    RealSpaceAxis,
    ReciprocalSpaceAxis,
)
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.chunks import Chunks
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength
from abtem.core.ensemble import _wrap_with_array
from abtem.core.fft import fft_interpolate
from abtem.core.utils import cos_sin_deg, get_dtype, safe_floor_int
from abtem.measurements import (
    BaseMeasurements,
    DiffractionPatterns,
    Images,
    MeasurementsEnsemble,
    PolarMeasurements,
    RealSpaceLineProfiles,
    _diffraction_pattern_resampling_gpts,
    _image_resampling_gpts,
    _needs_the_diffuse_part,
    _polar_detector_bins,
    _radial_weight_map,
    _scan_axes,
    _scan_shape,
    _scanned_measurement_type,
    _validate_frozen_phonon_components,
    _validate_reduction_dtype,
)
from abtem.transform import ArrayObjectTransform, WavesType
from abtem.visualize.visualizations import discrete_cmap

if TYPE_CHECKING:
    from abtem.array import ArrayObject, ArrayObjectType
    from abtem.waves import BaseWaves, Waves
else:
    Waves = object
    ArrayObject = object
    ArrayObjectType = TypeVar("ArrayObjectType", bound="ArrayObject")


def _energy_from_waves(waves) -> Optional[float]:
    """Return a scalar electron energy [eV] from *waves*, or ``None`` for a
    full, un-indexed multi-energy ensemble. Uses the same resolution order as
    ``Waves._valid_energy`` (see :func:`abtem.core.energy.resolve_energy`),
    but returns ``None`` instead of raising when unresolved."""
    from abtem.core.energy import resolve_energy

    return resolve_energy(waves.energy, waves.metadata, waves.ensemble_axes_metadata)


def _gpts_and_sampling_from_obj(obj):
    """Extract grid parameters from waves *or* a DiffractionPatterns object.

    Returns
    -------
    gpts : tuple[int, int]
    angular_sampling : tuple[float, float]   [mrad]
    reciprocal_space_sampling : tuple[float, float]   [1/Å]
    energy : float or None   [eV]
    """
    from abtem.measurements import DiffractionPatterns

    if isinstance(obj, DiffractionPatterns):
        gpts = obj.shape[-2:]
        angular_sampling = obj.angular_sampling
        reciprocal_space_sampling = obj.sampling
        energy = obj.metadata.get("energy")
    else:
        # BaseWaves
        gpts = obj._gpts_within_angle("cutoff")
        angular_sampling = obj.angular_sampling
        reciprocal_space_sampling = obj.reciprocal_space_sampling
        energy = _energy_from_waves(obj)
    return gpts, angular_sampling, reciprocal_space_sampling, energy


class RadialSensitivity:
    """
    A radially varying detector sensitivity (gain) as a function of the radial
    scattering angle.

    This describes how the recorded signal of an annular detector is weighted as a
    function of scattering angle, replacing the idealized assumption of a detector that
    is uniformly sensitive between its inner and outer integration limits. Real annular
    detectors typically have a non-uniform angular response, e.g. a gradual roll-off
    towards the edges or a measured gain curve.

    The angle is measured from the detector centre, i.e. from the optical axis displaced
    by the detector's ``offset``, so that the sensitivity moves with an offset detector
    and ``inner``/``outer`` refer to the same radii as the sensitivity.

    The sensitivity is a weight in the range ``[0, 1]``, where 1 is full sensitivity;
    weights greater than 1 are unphysical and rejected. It may be specified in two ways:

    1. As a callable mapping the radial scattering angle [mrad] to a sensitivity weight.
       The callable must accept and return array-like input (it is evaluated on a grid of
       angles), e.g. ``lambda alpha: alpha / 100.0``. It is sampled across the detector's
       integration range and must return finite weights within ``[0, 1]``.
    2. As a tuple ``(angles, values)`` of one-dimensional array-likes giving a measured
       sensitivity curve. The angles [mrad] must be strictly increasing and the values
       must lie within ``[0, 1]``. The values are linearly interpolated onto the detector
       grid; angles outside the measured range are clamped to the first and last value.

    Parameters
    ----------
    sensitivity : callable or tuple of array-like
        The radial sensitivity, given either as a callable ``w(alpha)`` or as a tuple
        ``(angles, values)`` describing a measured sensitivity curve. If an existing
        ``RadialSensitivity`` is given, it is returned unchanged by
        :func:`validate_radial_sensitivity`.

    Examples
    --------
    >>> # Linearly increasing sensitivity towards higher angles.
    >>> sens = RadialSensitivity(lambda alpha: alpha / 80.0)
    >>> # A measured response curve.
    >>> sens = RadialSensitivity(([0, 40, 80, 120], [0.0, 1.0, 0.9, 0.2]))
    """

    def __init__(
        self,
        sensitivity: Callable | tuple[Any, Any],
    ):
        if callable(sensitivity):
            self._func: Optional[Callable] = sensitivity
            self._angles: Optional[np.ndarray] = None
            self._values: Optional[np.ndarray] = None
        else:
            try:
                angles, values = sensitivity
            except (TypeError, ValueError):
                raise ValueError(
                    "'sensitivity' must be a callable or a tuple (angles, values)"
                )

            angles = np.asarray(angles, dtype=float)
            values = np.asarray(values, dtype=float)

            if angles.ndim != 1 or angles.shape != values.shape:
                raise ValueError(
                    "'angles' and 'values' must be one-dimensional arrays of equal length"
                )

            if angles.size < 2:
                raise ValueError("'angles' must contain at least two points")

            if np.any(np.diff(angles) <= 0):
                raise ValueError("'angles' must be strictly increasing")

            if np.any(values < 0) or np.any(values > 1):
                raise ValueError(
                    "sensitivity 'values' must lie within [0, 1] (a value of 1 is full "
                    "sensitivity); values greater than 1 are unphysical"
                )

            self._func = None
            self._angles = angles
            self._values = values

    @property
    def angles(self) -> Optional[np.ndarray]:
        """The angles of the measured sensitivity curve [mrad], or None if callable."""
        return self._angles

    @property
    def values(self) -> Optional[np.ndarray]:
        """The values of the measured sensitivity curve, or None if callable."""
        return self._values

    @property
    def func(self) -> Optional[Callable]:
        """The sensitivity callable, or None if a measured curve was given."""
        return self._func

    def __call__(self, alpha):
        """
        Evaluate the sensitivity at the given radial scattering angles.

        Parameters
        ----------
        alpha : array-like
            Radial scattering angles [mrad].

        Returns
        -------
        weights : array-like
            The sensitivity weights, with the same shape as ``alpha``.
        """
        if self._func is not None:
            return self._func(alpha)

        xp = get_array_module(alpha)
        weights = np.interp(asnumpy(alpha), self._angles, self._values)
        return xp.asarray(weights)

    def _check_range(self, inner: float, outer: float) -> None:
        """
        Validate the sensitivity against a detector's integration range [inner, outer].

        For a measured ``(angles, values)`` curve, all provided angles must lie within
        ``[inner, outer]`` so that no detector angle relies on extrapolation (the values
        are already validated to lie within ``[0, 1]`` at construction). For a callable,
        the domain cannot be introspected, so instead the function is sampled across
        ``[inner, outer]`` and checked to return finite weights within ``[0, 1]``.

        Parameters
        ----------
        inner : float
            Inner integration limit of the detector [mrad].
        outer : float
            Outer integration limit of the detector [mrad].

        Raises
        ------
        ValueError
            If the measured angles fall outside ``[inner, outer]``, or if a callable
            produces non-finite weights or weights outside ``[0, 1]`` over that range.
        """
        tol = 1e-6

        if self._angles is not None:
            low, high = float(self._angles[0]), float(self._angles[-1])
            if low < inner - tol or high > outer + tol:
                raise ValueError(
                    f"RadialSensitivity angles span [{low:.3g}, {high:.3g}] mrad, which "
                    f"lies outside the detector integration range "
                    f"[{inner:.3g}, {outer:.3g}] mrad. Provide sensitivity angles within "
                    f"the detector's inner and outer limits."
                )
        else:
            sample = np.linspace(inner, outer, 128)
            weights = np.asarray(self._func(sample), dtype=float)
            if not np.all(np.isfinite(weights)):
                raise ValueError(
                    f"RadialSensitivity callable produced non-finite weights over the "
                    f"detector range [{inner:.3g}, {outer:.3g}] mrad."
                )
            if np.any(weights < -tol):
                raise ValueError(
                    f"RadialSensitivity callable produced negative weights over the "
                    f"detector range [{inner:.3g}, {outer:.3g}] mrad; sensitivity "
                    f"weights must be non-negative."
                )
            if np.any(weights > 1 + tol):
                raise ValueError(
                    f"RadialSensitivity callable produced weights greater than 1 over "
                    f"the detector range [{inner:.3g}, {outer:.3g}] mrad; sensitivity "
                    f"weights must lie within [0, 1] (a weight of 1 is full sensitivity)."
                )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RadialSensitivity):
            return False
        if self._func is not None or other._func is not None:
            return self._func is other._func
        return bool(
            np.array_equal(self._angles, other._angles)
            and np.array_equal(self._values, other._values)
        )

    def __hash__(self) -> int:
        if self._func is not None:
            return hash(self._func)
        return hash((tuple(self._angles.tolist()), tuple(self._values.tolist())))


def validate_radial_sensitivity(
    sensitivity: Optional[RadialSensitivity | Callable | tuple[Any, Any]],
) -> Optional[RadialSensitivity]:
    """
    Normalize a radial sensitivity specification to a ``RadialSensitivity`` or None.

    Parameters
    ----------
    sensitivity : RadialSensitivity, callable, tuple of array-like or None
        The radial sensitivity specification. If None, None is returned.

    Returns
    -------
    sensitivity : RadialSensitivity or None
    """
    if sensitivity is None or isinstance(sensitivity, RadialSensitivity):
        return sensitivity
    return RadialSensitivity(sensitivity)


def validate_detectors(
    detectors: Optional[BaseDetector | list[BaseDetector]] = None,
    waves: Optional[BaseWaves] = None,
) -> list[BaseDetector]:
    """
    Validate that a variable is a list of detectors.

    Parameters
    ----------
    detectors : BaseDetector or list of BaseDetector
        The detectors to validate.
    waves : Waves, optional
        The waves to match the detectors to.

    Returns
    -------
    list of BaseDetector
        A list of validated detectors. With `waves`, every detector that matches
        itself to the waves (e.g. by auto-sizing its outer angle) is returned as a
        matched copy, and the detectors that were passed in are not modified.

    Raises
    ------
    TypeError
        If `detectors` is not a BaseDetector or a list of BaseDetector.
    """
    if isinstance(detectors, BaseDetector):
        detectors = [detectors]

    elif detectors is None:
        detectors = [WavesDetector()]

    elif not (
        isinstance(detectors, list)
        and all(hasattr(detector, "detect") for detector in detectors)
    ):
        raise RuntimeError("Detectors must be BaseDetector or list of BaseDetector.")

    if waves is not None:
        detectors = [
            detector._matched(waves) if hasattr(detector, "_matched") else detector
            for detector in detectors
        ]

    return detectors


class BaseDetector(ArrayObjectTransform[Waves, BaseMeasurements | Waves]):
    """
    Base detector class.

    Parameters
    ----------
    to_cpu : bool, optional
       If True, copy the measurement data from the calculation device to CPU memory
       after applying the detector, otherwise the data stays on the respective devices.
       Default is True.
    url : str, optional
       If this parameter is set the measurement data is saved at the specified location,
       typically a path to a local file. A URL can also include a protocol specifier
       like s3:// for remote data. If not set (default) the data stays in memory.
    """

    _splits_energy_ensembles = True

    def __init__(self, to_cpu: bool = True, url: Optional[str] = None):
        self._to_cpu = to_cpu
        self._url = url

    @property
    def url(self) -> Optional[str]:
        """The storage location of the measurement data."""
        return self._url

    @property
    def to_cpu(self) -> bool:
        """The measurements are copied to host memory."""
        return self._to_cpu

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ()

    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple[Any, ...]:
        return ()

    @classmethod
    def _from_partition_args_func(cls, **kwargs):
        detector = cls(**kwargs)
        return _wrap_with_array(detector)

    def _from_partitioned_args(self) -> Callable:
        kwargs = self._copy_kwargs()
        return partial(self._from_partition_args_func, **kwargs)

    def _out_type(self, waves: Waves) -> tuple[Type[BaseMeasurements] | Type[Waves]]:
        raise NotImplementedError

    def _out_meta(self, waves: Waves) -> tuple[np.ndarray, ...]:
        """
        The meta describing the measurement array created when detecting the given
        waves.

        Parameters
        ----------
        waves : Waves
            The waves to derive the measurement meta from.

        Returns
        -------
        meta : array-like
            Empty array.
        """
        if self.to_cpu:
            return (np.array((), dtype=self._out_dtype(waves)[0]),)
        else:
            xp = get_array_module(waves.device)
            return (xp.array((), dtype=self._out_dtype(waves)[0]),)

    def detect(self, waves: Waves) -> BaseMeasurements | Waves:
        """
        Detect the given waves producing a measurement.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : BaseMeasurements
        """

        return self.apply(waves, max_batch="auto")

    def apply(
        self, waves: Waves, max_batch: int | str = "auto"
    ) -> BaseMeasurements | Waves:
        measurements = waves.apply_transform(self)
        assert isinstance(measurements, (BaseMeasurements, Waves))
        return measurements

class _AbstractRadialDetector(BaseDetector):
    # Whether an automatic outer angle is sized for each energy of a multi-energy
    # ensemble on its own. A detector with radial bins has one radial axis for the
    # whole ensemble, which no single outer angle fits.
    _sizes_outer_per_energy = False

    def __init__(
        self,
        inner: float,
        outer: Optional[float] = None,
        rotation: float = 0.0,
        offset: tuple[float, float] = (0.0, 0.0),
        sensitivity: Optional[RadialSensitivity | Callable | tuple[Any, Any]] = None,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        if inner < 0.0:
            raise ValueError(f"'inner' must be non-negative, got {inner}")
        if outer is not None and outer < 0.0:
            raise ValueError(f"'outer' must be non-negative, got {outer}")

        self._inner = inner
        self._outer = outer
        # Whether `outer` was given explicitly (constructor or the setter
        # below), as opposed to sized from the detected waves by `_matched`.
        # `outer is None` alone cannot tell these apart on a matched copy, which
        # holds its sized value but must be sized again by the next waves it
        # detects (see `_matched`).
        self._outer_is_explicit = outer is not None
        self._rotation = rotation
        self._offset = offset
        self._sensitivity = validate_radial_sensitivity(sensitivity)

        # Validate the sensitivity against the integration range as soon as it is known.
        # If 'outer' is None it is resolved from the waves at detection time, where the
        # check is repeated.
        if self._sensitivity is not None and outer is not None:
            self._sensitivity._check_range(inner, outer)

        super().__init__(to_cpu=to_cpu, url=url)

    def _copy_kwargs(self, exclude: tuple[str, ...] = (), cls=None) -> dict:
        kwargs = super()._copy_kwargs(exclude=exclude, cls=cls)
        # A sized outer angle stays automatic in a detector rebuilt for a lazy
        # `detect`, so it is sized again for the waves the rebuilt one detects.
        if not self._outer_is_explicit:
            kwargs["outer"] = None
        return kwargs

    @property
    def sensitivity(self) -> Optional[RadialSensitivity]:
        """The radially varying detector sensitivity, or None if uniform."""
        return self._sensitivity

    @property
    def inner(self) -> float:
        """Inner integration limit [mrad]."""
        return self._inner

    @inner.setter
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> Optional[float]:
        """Outer integration limit [mrad]."""
        return self._outer

    @outer.setter
    def outer(self, value: float):
        self._outer = value
        self._outer_is_explicit = value is not None

    @property
    def rotation(self):
        """Rotation of the bins around the origin [rad]."""
        return self._rotation

    @property
    def offset(self) -> tuple[float, float]:
        """Offset of the detector centre from the origin in `x` and `y` [mrad]."""
        return self._offset

    @property
    @abstractmethod
    def radial_sampling(self):
        """Spacing between the radial detector bins [mrad]."""

    @property
    @abstractmethod
    def azimuthal_sampling(self):
        """Spacing between the azimuthal detector bins [mrad]."""

    @property
    @abstractmethod
    def nbins_radial(self):
        """Spacing between the azimuthal detector bins [mrad]."""

    @property
    @abstractmethod
    def nbins_azimuthal(self):
        """Spacing between the azimuthal detector bins [mrad]."""

    def _out_dtype(self, waves: WavesType) -> tuple[np.dtype]:
        return (np.finfo(waves.dtype).dtype,)

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        matched = self._matched(waves)
        return ((matched.nbins_radial, matched.nbins_azimuthal),)

    def _out_type(self, waves: WavesType) -> tuple[Type[PolarMeasurements]]:
        return (PolarMeasurements,)

    def _out_metadata(self, waves: WavesType) -> tuple[dict]:
        metadata = super()._out_metadata(waves)[0]
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return (metadata,)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        matched = self._matched(waves)
        return (
            [
                LinearAxis(
                    label="Radial scattering angle",
                    offset=self.inner,
                    sampling=matched.radial_sampling,
                    _concatenate=False,
                    units="mrad",
                ),
                LinearAxis(
                    label="Azimuthal scattering angle",
                    offset=self.rotation,
                    sampling=matched.azimuthal_sampling,
                    _concatenate=False,
                    units="rad",
                ),
            ],
        )

    def angular_limits(self, waves: WavesType) -> tuple[float, float]:
        outer = self._outer_for(waves)

        # The sensitivity is checked against the requested outer angle, as in the
        # constructor, not the outer edge of the outermost bin.
        if self.sensitivity is not None:
            self.sensitivity._check_range(self.inner, outer)

        return self.inner, self._binned_outer(outer)

    def _binned_outer(self, outer: float) -> float:
        """Outer edge of the outermost bin for a requested ``outer`` [mrad]."""
        return outer

    def _nbins_within(self, outer: float) -> int:
        """Number of radial bins for a requested ``outer`` [mrad]."""
        return self.nbins_radial

    def _outer_for(self, waves: WavesType) -> float:
        """The ``outer`` used to detect ``waves`` [mrad]: the given one, or
        else the antialias cutoff angle of ``waves`` (see `_matched`)."""
        if self._outer_is_explicit:
            return self.outer

        if not self._sizes_outer_per_energy and any(
            isinstance(axis, EnergyAxis) and len(axis.values) > 1
            for axis in waves.ensemble_axes_metadata
        ):
            raise RuntimeError(
                f"{type(self).__name__} cannot auto-size its outer angle for "
                "a multi-energy ensemble: each energy has its own antialias "
                "cutoff angle (it scales with wavelength at fixed grid), so "
                "no single radial axis fits every member. Pass an explicit "
                "outer= (a value valid for every member), or run one energy "
                "at a time and combine the results yourself."
            )

        return min(waves.cutoff_angles)

    def _calculate_new_array(self, waves: WavesType) -> np.ndarray:
        """
        Detect the given waves producing polar measurements.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : PolarMeasurements
        """
        detector = self._matched(waves)
        inner, outer = detector.angular_limits(waves)

        # The pattern is cropped about k=0 and polar_binning then shifts the
        # bins by the offset, so the crop reaches `outer` beyond the offset
        # centre, plus a pixel for the nearest-pixel rounding of the offset.
        # A detector reaching past the grid uses the full pattern: the bins
        # shifted beyond the Nyquist frequency are dropped, not wrapped round.
        max_angle: float | str = outer
        if np.any(np.array(self._offset) != 0.0):
            max_angle = (
                outer + float(np.hypot(*self._offset)) + max(waves.angular_sampling)
            )
            gpts = waves._gpts_within_angle(max_angle, parity="same")
            if any(g >= n for g, n in zip(gpts, waves._valid_gpts)):
                max_angle = "full"

        measurement = waves.diffraction_patterns(max_angle=max_angle, parity="same")

        measurement = measurement.polar_binning(
            nbins_radial=detector.nbins_radial,
            nbins_azimuthal=detector.nbins_azimuthal,
            inner=inner,
            outer=outer,
            rotation=self._rotation,
            offset=self._offset,
            sensitivity=self.sensitivity,
        )

        if self.to_cpu:
            measurement = measurement.to_cpu()

        return measurement._eager_array

    def _match_ensemble(self, waves: WavesType) -> _AbstractRadialDetector:
        """This detector sized for ``waves`` (see `_matched`), before the ensemble
        is split into energies or lazy blocks. An automatic outer angle is
        refused for a multi-energy ensemble."""
        return self._matched(waves)

    def _matched(self, waves: WavesType) -> _AbstractRadialDetector:
        """This detector with its outer angle sized for ``waves``.

        An explicit ``outer`` is returned as it is, as the detector itself. Else the
        outer angle is the antialias cutoff angle of ``waves``, held by a copy:
        sizing never writes onto the detector the caller holds, which another run
        (other energy, grid or algorithm) would otherwise reuse with the first
        run's angle. The copy is not explicit, so it is sized again by the next
        waves it is matched with, as the S-matrix reduction does twice.

        A detector with radial bins (`FlexibleAnnularDetector`,
        `SegmentedDetector`) cannot be sized for a multi-energy ensemble: each
        energy has its own cutoff angle (`waves.cutoff_angles`, which scales with
        wavelength at a fixed grid; the semiangle cutoff of the probe does not
        enter it), so no single radial axis fits all of them. Picking one energy's
        cutoff would silently depend on the order of the energies, so this raises.
        `AnnularDetector` has no radial bins and sizes each energy on its own.
        """
        if self._outer_is_explicit:
            return self

        matched = self.copy()
        matched._outer = self._outer_for(waves)
        if matched.sensitivity is not None:
            matched.sensitivity._check_range(matched.inner, matched._outer)
        return matched

    def detect(self, waves: WavesType) -> PolarMeasurements:
        """
        Detect the given waves producing polar measurements.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : PolarMeasurements
        """
        measurements = super().detect(waves)
        assert isinstance(measurements, PolarMeasurements)
        return measurements

    def _region_limits(self, waves: Optional[BaseWaves] = None):
        """(inner, outer, nbins_radial) of the bins that detecting ``waves``
        uses, or that the detector's own ``outer`` gives without waves."""
        if waves is not None:
            inner, outer = self.angular_limits(waves)
        elif self.outer is None:
            raise ValueError("provide the waves or the outer limit of the detector")
        else:
            inner, outer = self.inner, self._binned_outer(self.outer)
        return inner, outer, self._nbins_within(outer)

    def get_detector_regions(self, waves: Optional[BaseWaves] = None):
        """
        Get the polar detector regions as a polar measurement.

        By default each region is labelled by its integer index. If the detector has a
        radial sensitivity, the regions are instead filled with the detector efficiency
        [%] at the center of each radial bin, so that the sensitivity profile is
        reflected when the regions are shown.

        The regions are labelled on polar axes about the detector centre; a
        detector ``offset`` is not represented (``show`` draws it).

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the polar detector regions from.

        Returns
        -------
        detector_region : PolarMeasurements
        """
        inner, outer, nbins_radial = self._region_limits(waves)

        if waves is not None:
            metadata = copy(waves.metadata)
        else:
            metadata = {}

        if self.sensitivity is not None:
            radial_sampling = (outer - inner) / nbins_radial
            radial_centers = inner + (np.arange(nbins_radial) + 0.5) * radial_sampling
            weights = np.asarray(self.sensitivity(radial_centers), dtype=float)
            # A sensitivity callable is not required to preserve the input shape (e.g. a
            # constant weight returns a scalar); broadcast explicitly so a scalar result
            # doesn't crash the [:, None] indexing below.
            weights = np.broadcast_to(weights, radial_centers.shape) * 100.0
            bins = np.repeat(weights[:, None], self.nbins_azimuthal, axis=1)
            metadata.update({"label": "Detector efficiency", "units": "%"})
        else:
            bins = np.arange(0, nbins_radial * self.nbins_azimuthal)
            bins = bins.reshape((nbins_radial, self.nbins_azimuthal))
            metadata.update({"label": "detector regions", "units": ""})

        polar_measurements = PolarMeasurements(
            bins,
            radial_sampling=(outer - inner) / nbins_radial,
            azimuthal_sampling=self.azimuthal_sampling,
            radial_offset=inner,
            metadata=metadata,
            azimuthal_offset=self._rotation,
        )

        return polar_measurements

    def _raise_region_arity_error(self, has: str, use: str, use_is_for: str) -> None:
        raise NotImplementedError(
            f"'{type(self).__name__}' {has}; use '{use}' to obtain {use_is_for}."
        )

    def get_detector_region(
        self, waves: Optional[BaseWaves] = None, fftshift: bool = True
    ):
        """
        Not available for multi-region detectors.

        Use :meth:`get_detector_regions` (plural) to obtain the polar regions of a
        ``FlexibleAnnularDetector`` or ``SegmentedDetector``. The singular
        ``get_detector_region`` is only defined for the single-region
        ``AnnularDetector``, which overrides this method.
        """
        self._raise_region_arity_error(
            has="has multiple regions",
            use="get_detector_regions",
            use_is_for="its polar regions (plural); the singular 'get_detector_region' "
            "is only for the single-region AnnularDetector",
        )

    def show(
        self,
        waves: Optional[BaseWaves] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
        **kwargs,
    ):
        """
        Show the detector regions as a polar plot.

        The discrete detector regions are shown with a categorical colormap. If the
        detector has a radial sensitivity, the per-pixel detector efficiency [%] is shown
        instead with a continuous colormap, so the sensitivity profile is visible.

        The regions are drawn about the detector centre, including any
        ``offset``, out to the bins that detecting the waves would use.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the detector regions from.
        gpts : two int, optional
            Number of grid points describing the wave functions to be detected.
        sampling : two float, optional
            Lateral sampling of the wave functions to be detected [Å]. If not
            given, the drawn grid extends 10 % beyond the detector.
        energy : float, optional
            Electron energy of the wave functions to be detected [eV].
        kwargs :
            Optional keyword arguments for DiffractionPatterns.show.

        Returns
        -------
        visualization : Visualization
        """

        if waves is not None:
            if gpts is not None or sampling is not None or energy is not None:
                raise ValueError(
                    "provide either waves or 'gpts', 'sampling' and 'energy'"
                )
            energy = _energy_from_waves(waves)
            if energy is None:
                raise ValueError(
                    "cannot show the detector for a multi-energy ensemble; "
                    "select a single energy"
                )
            gpts = tuple(waves.gpts)
        elif energy is None:
            raise ValueError("provide the waves or the energy of waves")
        elif gpts is None:
            gpts = 1024

        if not isinstance(gpts, tuple):
            gpts = (int(gpts),) * 2

        inner, outer, nbins_radial = self._region_limits(waves)
        offset = self._offset if self._offset is not None else (0.0, 0.0)
        wavelength = energy2wavelength(energy) * 1e3

        if sampling is None:
            # the grid reaches 10 % beyond the outermost angle the region
            # covers, about its (possibly offset) centre
            reach = 1.1 * (outer + float(np.hypot(*offset)))
            angular_sampling = (2 * reach / gpts[0], 2 * reach / gpts[1])
            reciprocal_space_sampling = (
                angular_sampling[0] / wavelength,
                angular_sampling[1] / wavelength,
            )
        else:
            if not isinstance(sampling, tuple):
                sampling = (float(sampling),) * 2

            reciprocal_space_sampling = (
                1 / (gpts[0] * sampling[0]),
                1 / (gpts[1] * sampling[1]),
            )
            angular_sampling = (
                reciprocal_space_sampling[0] * wavelength,
                reciprocal_space_sampling[1] * wavelength,
            )

        regions = _polar_detector_bins(
            gpts=gpts,
            sampling=angular_sampling,
            inner=inner,
            outer=outer,
            nbins_radial=nbins_radial,
            nbins_azimuthal=self.nbins_azimuthal,
            fftshift=True,
            rotation=self.rotation,
            offset=offset,
            return_indices=False,
        )
        assert isinstance(regions, np.ndarray)
        inside = regions >= 0

        num_colors = nbins_radial * self.nbins_azimuthal

        if self.sensitivity is not None:
            # Show the continuous per-pixel efficiency [%] so the radial sensitivity is
            # visible, rather than the (here uninformative) discrete region labels.
            # The weights are a function of the angle from the detector centre
            # (including any offset), as in `polar_binning`.
            weights = _radial_weight_map(
                gpts=tuple(gpts),
                sampling=angular_sampling,
                sensitivity=self.sensitivity,
                fftshift=True,
                offset=offset,
            )
            array = np.where(inside, weights * 100.0, np.nan).astype(
                get_dtype(complex=False)
            )
            metadata = {
                "energy": energy,
                "label": "Detector efficiency",
                "units": "%",
            }
            diffraction_patterns = DiffractionPatterns(
                array,
                sampling=reciprocal_space_sampling,
                fftshift=True,
                metadata=metadata,
            )
            # Scale the colormap to the actual efficiency range within the detector so
            # the radial sensitivity gradient is visible (rather than washed out over a
            # fixed 0-100 % range).
            finite = array[np.isfinite(array)]
            low = float(finite.min()) if finite.size else 0.0
            high = float(finite.max()) if finite.size else 100.0
            if high <= low:
                low, high = 0.0, max(high, 1e-6)
            kwargs.setdefault("cmap", "viridis")
            kwargs.setdefault("vmin", low)
            kwargs.setdefault("vmax", high)
        else:
            array = regions.astype(get_dtype(complex=False))
            array[..., ~inside] = np.nan
            diffraction_patterns = DiffractionPatterns(
                array,
                sampling=reciprocal_space_sampling,
                fftshift=True,
                metadata={"energy": energy},
            )

            if "cmap" not in kwargs:
                kwargs["cmap"] = "tab10" if num_colors <= 10 else "tab20"
            kwargs["cmap"] = discrete_cmap(
                num_colors=num_colors, base_cmap=kwargs["cmap"]
            )
            kwargs.setdefault("vmin", -0.5)
            kwargs.setdefault("vmax", num_colors - 0.5)

        if "units" not in kwargs:
            kwargs["units"] = "mrad"

        diffraction_patterns.metadata["energy"] = energy

        return diffraction_patterns.show(**kwargs)


class AnnularDetector(_AbstractRadialDetector):
    """
    The annular detector integrates the intensity of the detected wave functions between
    an inner and outer radial integration limits, i.e. over an annulus.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float, optional
        Outer integration limit [mrad]. If None, the antialias cutoff angle of the
        detected waves.
    offset: two float, optional
        Center offset of the annular integration region [mrad].
    sensitivity : RadialSensitivity, callable or tuple of array-like, optional
        A radially varying detector sensitivity. If given, the diffraction patterns are
        weighted by this factor as a function of the radial scattering angle [mrad]
        before integration, simulating a detector that is not uniformly sensitive within
        its integration limits. May be given as a :class:`RadialSensitivity`, a callable
        ``w(alpha)``, or a tuple ``(angles, values)`` describing a measured sensitivity
        curve. By default the detector is uniformly sensitive.
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory
        after applying the detector, otherwise the data stays on the respective devices.
        Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified
        location, typically a path to a local file. A URL can also include a protocol
        specifier like s3:// for remote data. If not set (default) the data stays in
        memory.
    """

    def __init__(
        self,
        inner: float = 0.0,
        outer: Optional[float] = None,
        offset: tuple[float, float] = (0.0, 0.0),
        sensitivity: Optional[RadialSensitivity | Callable | tuple[Any, Any]] = None,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._inner = inner
        self._outer = outer
        self._offset = offset

        super().__init__(
            inner=inner,
            outer=outer,
            rotation=0.0,  # Rotation is meaningless for standard annular detector
            offset=offset,
            sensitivity=sensitivity,
            to_cpu=to_cpu,
            url=url,
        )

    @property
    def inner(self) -> float:
        """Inner integration limit in mrad."""
        return self._inner

    @inner.setter
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float | None:
        """Outer integration limit in mrad."""
        return self._outer

    @outer.setter
    def outer(self, value: float):
        self._outer = value
        self._outer_is_explicit = value is not None

    @property
    def nbins_radial(self):
        return 1

    @property
    def nbins_azimuthal(self):
        return 1

    @property
    def radial_sampling(self) -> float:
        if self._outer is None:
            raise RuntimeError(
                "radial_sampling is not defined when outer angle is None"
            )
        return self._outer - self._inner

    @property
    def azimuthal_sampling(self) -> float:
        return 2 * np.pi

    def _out_metadata(self, array_object: WavesType) -> tuple[dict]:
        metadata = super()._out_metadata(array_object)[0]
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return (metadata,)

    def _out_ensemble_axes_metadata(
        self, waves: WavesType
    ) -> tuple[list[AxisMetadata]]:
        source = _scan_axes(waves)
        scan_axes_metadata = [waves.ensemble_axes_metadata[i] for i in source]
        ensemble_axes_metadata = [
            m for i, m in enumerate(waves.ensemble_axes_metadata) if i not in source
        ]
        return (ensemble_axes_metadata + scan_axes_metadata,)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        return ([],)

    def _out_ensemble_shape(self, waves: WavesType) -> tuple[tuple[int, ...], ...]:
        ensemble_shapes = super()._out_ensemble_shape(waves)

        source = _scan_axes(waves)
        if not source:
            return ensemble_shapes  # No 2D scan axes: keep PositionsAxis in ensemble as-is

        # Drop exactly the axes _scan_axes identifies, by position -- not the
        # last two entries, which need not be the scan axes (e.g. a GridScan
        # probe's own energy ensemble trails its two ScanAxis entries).
        drop = {i + len(self.ensemble_shape) for i in source}
        return tuple(
            tuple(s for i, s in enumerate(ensemble_shape) if i not in drop)
            for ensemble_shape in ensemble_shapes
        )

    def _out_ensemble_source(
        self, waves: WavesType
    ) -> tuple[tuple[int, ...], ...]:
        source = _scan_axes(waves)
        if not source:
            return super()._out_ensemble_source(waves)
        kept = [i for i in range(len(waves.ensemble_shape)) if i not in source]
        return (tuple(kept + list(source)),)

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, ...]]:
        return (_scan_shape(waves),)

    # An auto-sized outer angle follows each energy's cutoff, as a separate run of
    # each energy would; the result has no radial axis to share. (The metadata of
    # an AnnularDetector result never records an outer angle.)
    _sizes_outer_per_energy = True

    def _out_type(
        self, waves: WavesType
    ) -> tuple[Type[RealSpaceLineProfiles] | Type[Images] | Type[MeasurementsEnsemble]]:
        return (_scanned_measurement_type(waves),)

    def _calculate_new_array(self, waves: WavesType) -> np.ndarray:
        """
        Detect the given waves producing diffraction patterns.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : DiffractionPatterns
        """
        inner, outer = self.angular_limits(waves)

        diffraction_patterns = waves.diffraction_patterns(
            max_angle="full", parity="same", fftshift=False
        )
        diffraction_patterns._check_integration_limits(inner, outer)
        offset = self.offset if self.offset is not None else (0.0, 0.0)

        # Integrated over the pattern axes only, so the result keeps the waves'
        # own axis order, the one every _calculate_new_array returns;
        # ArrayObject.apply_transform moves the scan axes to the end, as
        # _out_ensemble_source declares, once for eager and lazy results alike.
        intensity = DiffractionPatterns._integrate_fourier_space(
            diffraction_patterns._eager_array,
            sampling=diffraction_patterns.angular_sampling,
            inner=inner,
            outer=outer,
            fftshift=False,
            offset=offset,
            sensitivity=self.sensitivity,
        )

        if self.to_cpu and hasattr(intensity, "get"):
            intensity = intensity.get()

        return intensity

    def detect(
        self, waves: WavesType
    ) -> Images | RealSpaceLineProfiles | MeasurementsEnsemble:
        """
        Detect the given waves producing images.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : Images or RealSpaceLineProfiles
        """
        measurements = self.apply(waves)
        assert isinstance(
            measurements, (RealSpaceLineProfiles, Images, MeasurementsEnsemble)
        )
        return measurements

    def _get_detector_region_array(
        self, waves, fftshift: bool = True
    ) -> np.ndarray:
        inner, outer = self.angular_limits(waves)
        gpts, angular_sampling, _, _ = _gpts_and_sampling_from_obj(waves)

        array = _polar_detector_bins(
            gpts=gpts,
            sampling=angular_sampling,
            inner=inner,
            outer=outer,
            nbins_radial=1,
            nbins_azimuthal=1,
            fftshift=fftshift,
            rotation=0.0,
            offset=self.offset,
            return_indices=False,
        )
        assert isinstance(array, np.ndarray)
        inside = array >= 0

        if self.sensitivity is not None:
            weights = _radial_weight_map(
                gpts=inside.shape,
                sampling=angular_sampling,
                sensitivity=self.sensitivity,
                fftshift=fftshift,
                offset=self.offset,
            )
            # Select (rather than multiply) by the weight: a sensitivity callable that is
            # undefined (e.g. NaN) outside [inner, outer] must not corrupt pixels that are
            # already excluded by `inside`.
            region = np.where(inside, weights, 0.0).astype(get_dtype(complex=False))
        else:
            region = inside.astype(get_dtype(complex=False))

        return region * 100.0

    def get_detector_region(
        self, waves, fftshift: bool = True
    ) -> DiffractionPatterns:
        """
        Get the annular detector region as a diffraction pattern.

        The region is the detector efficiency [%] as a function of scattering angle: a
        binary mask (0 or 100 %) for a uniform detector, or the sensitivity-weighted
        efficiency if a radial sensitivity is set.

        Parameters
        ----------
        waves : BaseWaves or DiffractionPatterns
            The waves or diffraction patterns used to derive grid calibration.
        fftshift : bool, optional
            If True, the zero-frequency of the detector region is shifted to the center
            of the array, otherwise the center is at (0, 0).

        Returns
        -------
        detector_region : DiffractionPatterns
        """

        array = self._get_detector_region_array(waves, fftshift=fftshift)
        _, _, reciprocal_space_sampling, energy = _gpts_and_sampling_from_obj(waves)
        metadata = {
            "energy": energy,
            "label": "Detector efficiency",
            "units": "%",
        }
        diffraction_patterns = DiffractionPatterns(
            array,
            metadata=metadata,
            sampling=reciprocal_space_sampling,
            fftshift=fftshift,
        )
        return diffraction_patterns


def _slit_detector_mask(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    origin: tuple[float, float],
    angle: float,
    q_min: float,
    q_max: float,
    width: float,
    fftshift: bool = False,
    xp=np,
) -> np.ndarray:
    """Boolean mask for a rectangular slit in reciprocal space.

    The slit's long axis starts at the sweep *origin* and points along
    ``d = (cos(angle), sin(angle))``; a pixel at ``k`` is inside when

        q_min <= (k - origin) . d < q_max   and
        -width / 2 <= (k - origin) . n < width / 2,   n = (-sin, cos).

    Membership is tested in this frame, relative to the origin, rather than
    to the slit centre: a pixel at the origin has local coordinates exactly
    zero, so with ``q_min=0`` the q = 0 pixel (the direct beam, for the
    default origin) is always inside. In the centre frame it sat on the
    ``-extent / 2`` edge up to rounding, and was dropped at some angles.
    Testing the rotated frame, not an axis-aligned bounding box, keeps the
    mask correct for any *angle*.

    Parameters
    ----------
    gpts : (int, int)
        Grid points.
    sampling : (float, float)
        Angular sampling [mrad/pixel].
    origin : (kx, ky)
        Origin of the q-axis sweep [mrad].
    angle : float
        Rotation of the long axis [degrees, CCW from kx].
    q_min, q_max : float
        Range of the long axis from the origin [mrad].
    width : float
        Full width of the slit perpendicular to its long axis [mrad].
    fftshift : bool
        If True, zero frequency is at the centre of the array.
    xp : array module
    """
    from abtem.core.grid import spatial_frequencies

    kx, ky = spatial_frequencies(
        gpts,
        (1 / sampling[0] / gpts[0], 1 / sampling[1] / gpts[1]),
        False,
        xp,
    )
    dx = kx[:, None] - origin[0]
    dy = ky[None, :] - origin[1]

    cos_a, sin_a = cos_sin_deg(angle)
    along = dx * cos_a + dy * sin_a
    across = -dx * sin_a + dy * cos_a

    half_width = width / 2.0
    mask = (
        (along >= q_min)
        & (along < q_max)
        & (across >= -half_width)
        & (across < half_width)
    )

    if fftshift:
        mask = xp.fft.fftshift(mask)

    return mask


def _corners_from_slit_params(
    offset: tuple[float, float],
    angle: float,
    extent: float,
    width: float,
) -> tuple[float, float, float, float]:
    """Convert slit geometry parameters to axis-aligned corners after rotation.

    The slit is centred at *offset*, has its long axis along *angle* (degrees,
    CCW from the kx axis), full length *extent* and full width *width*.

    Returns the rotated corners as ``(kx_min, kx_max, ky_min, ky_max)`` in the
    *rotated* frame — the mask function works in this frame after rotating the
    coordinate grid by ``-angle``.
    """
    half_e = extent / 2.0
    half_w = width / 2.0
    # corners in the rotated frame, centred at origin
    corners_local = np.array(
        [[-half_e, -half_w], [-half_e, half_w], [half_e, -half_w], [half_e, half_w]]
    )
    cos_a, sin_a = cos_sin_deg(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners_world = corners_local @ R.T + np.array(offset)
    kx_min, ky_min = corners_world.min(axis=0)
    kx_max, ky_max = corners_world.max(axis=0)
    return float(kx_min), float(kx_max), float(ky_min), float(ky_max)


class SpectralSlitDetector(BaseDetector):
    """
    A rectangular slit detector in reciprocal (diffraction) space.

    The slit can be defined in two ways:

    **Geometry mode** — specify size, q-range and orientation:

    Parameters
    ----------
    width : float
        **Full** width of the slit perpendicular to its long axis [mrad].
        This is the full integration aperture, *not* the half-width.  For
        equivalent integration coverage perpendicular to the q-scan direction
        as a :class:`SpectralAnnularDetector` with acceptance radius
        ``outer=r``, use ``width = 2 * r`` (the disk diameter, not the
        radius).
    q_min : float, optional
        Start of the q-axis [mrad].  Default is 0, which includes q=0 (the
        direct beam direction) as the first point of the spectrum.  Set to a
        positive value to exclude the low-q / direct-beam region, e.g.
        ``q_min=10`` to start at 10 mrad.  Directly comparable to the
        ``q_min`` parameter of :class:`SpectralAnnularDetector`.
    q_max : float
        Maximum scattering vector along the slit's long axis [mrad].  Directly
        comparable to the ``q_max`` parameter of
        :class:`SpectralAnnularDetector`.
    angle : float, optional
        Rotation of the long axis of the slit [degrees, CCW from kx axis].
        Default is 0.
    offset : two floats, optional
        Origin of the q-axis sweep ``(kx, ky)`` [mrad].  The q-axis starts
        here (at ``q_min``) and extends in the direction given by ``angle``.
        Default is ``(0, 0)``, i.e. the sweep starts from the diffraction
        pattern centre.
    q_sampling : float, optional
        Desired q-axis bin size [mrad].  If None (default) the native
        pixel sampling of the diffraction pattern is used.  Setting a
        larger value bins adjacent line samples together, producing fewer
        q-points and a faster spectrum.

    **Corner mode** — specify the four sides directly:

    Parameters
    ----------
    corners : (kx_min, kx_max, ky_min, ky_max)
        Axis-aligned bounds of the rectangle [mrad], with signs measured from
        the diffraction-pattern origin.  Incompatible with *offset*, *angle*,
        *q_min*, *q_max* and *width*.  The q-axis origin is taken as
        ``(kx_min, (ky_min+ky_max)/2)``, so ``q=0`` maps to the left edge of
        the rectangle.

    Common parameters
    -----------------
    to_cpu : bool, optional
        Copy result to CPU after detection.  Default is True.
    url : str, optional
        Save path for the measurement.

    Notes
    -----
    **Comparing slit and annular detectors**

    Both detector types share the same ``q_min``/``q_max`` convention — the
    same numerical value gives the same scattering-vector range in the output
    spectrum.  The perpendicular acceptance differs: the slit integrates a
    rectangle of full width ``width``, while the annular detector integrates a
    disk of radius ``outer``.

    ===========================  =================================
    SpectralSlitDetector         SpectralAnnularDetector
    ===========================  =================================
    ``width`` — full slit width  ``outer`` — acceptance **radius**
    ``q_min`` — start q (≥ 0)    ``q_min`` — start q (≥ 0)
    ``q_max`` — max q            ``q_max`` — max q
    ``angle`` — sweep direction  ``angle`` — sweep direction
    ===========================  =================================

    For equivalent perpendicular acceptance and the same q-range::

        SpectralSlitDetector(width=2*r, q_min=Q0, q_max=Q)
        SpectralAnnularDetector(outer=r, q_min=Q0, q_max=Q)

    Note that ``width = 2 * outer``: the slit ``width`` is the full aperture
    diameter, whereas ``outer`` is the acceptance *radius*.
    """

    def __init__(
        self,
        width: Optional[float] = None,
        q_min: float = 0.0,
        q_max: Optional[float] = None,
        angle: float = 0.0,
        offset: tuple[float, float] = (0.0, 0.0),
        corners: Optional[tuple[float, float, float, float]] = None,
        q_sampling: Optional[float] = None,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._q_sampling = float(q_sampling) if q_sampling is not None else None
        if corners is not None:
            if (
                q_max is not None
                or width is not None
                or angle != 0.0
                or offset != (0.0, 0.0)
                or q_min != 0.0
            ):
                raise ValueError(
                    "Provide either 'corners' or 'offset'/'angle'/'q_min'/'q_max'/'width', not both."
                )
            if len(corners) != 4:
                raise ValueError("'corners' must be a sequence of four values (kx_min, kx_max, ky_min, ky_max).")
            self._from_corners = True
            self._corners = tuple(float(c) for c in corners)
            # offset = start of q-sweep (left edge, ky-centre), consistent with
            # geometry mode where offset is the q=0 origin.
            self._offset = (
                float(corners[0]),
                (corners[2] + corners[3]) / 2.0,
            )
            self._angle = 0.0
            self._extent = float(corners[1] - corners[0])
            self._width = float(corners[3] - corners[2])
            self._q_min = 0.0
            self._q_max = self._extent
        else:
            if q_max is None or width is None:
                raise ValueError("Provide both 'q_max' and 'width' when not using 'corners'.")
            q_min = float(q_min)
            q_max = float(q_max)
            if q_min < 0 or q_min >= q_max:
                raise ValueError(f"q_min must satisfy 0 <= q_min < q_max, got q_min={q_min}, q_max={q_max}.")
            self._from_corners = False
            self._q_min = q_min
            self._q_max = q_max
            self._offset = tuple(float(v) for v in offset)
            self._angle = float(angle)
            # Physical slit extent and centre: spans from q_min to q_max along
            # the slit direction, centred at offset + (q_min+q_max)/2 * direction.
            cos_a, sin_a = cos_sin_deg(float(angle))
            slit_center = (
                offset[0] + (q_min + q_max) / 2.0 * cos_a,
                offset[1] + (q_min + q_max) / 2.0 * sin_a,
            )
            self._extent = q_max - q_min
            self._width = float(width)
            # AABB retained only for introspection/display via the `corners`
            # property; the detector mask itself tests the rotated frame
            # (see _slit_detector_mask) so it is correct for any angle.
            self._corners = _corners_from_slit_params(
                slit_center, self._angle, self._extent, self._width
            )
        super().__init__(to_cpu=to_cpu, url=url)

    @property
    def offset(self) -> tuple[float, float]:
        """Origin of the q-axis sweep (kx, ky) [mrad].  The q-axis starts here."""
        return self._offset

    @property
    def angle(self) -> float:
        """Long-axis rotation angle [degrees]."""
        return self._angle

    @property
    def q_min(self) -> float:
        """Start of the q-axis [mrad]."""
        return self._q_min

    @property
    def q_max(self) -> float:
        """Maximum scattering vector along the slit's long axis [mrad] (= q_min + extent)."""
        return self._q_min + self._extent

    @property
    def extent(self) -> float:
        """Physical length of the slit along its long axis [mrad] (= q_max - q_min)."""
        return self._extent

    @property
    def q_sampling(self) -> Optional[float]:
        """q-axis bin size [mrad], or None for native DP sampling."""
        return self._q_sampling

    @property
    def width(self) -> float:
        """Full width perpendicular to the long axis [mrad]."""
        return self._width

    @property
    def corners(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding rectangle (kx_min, kx_max, ky_min, ky_max) [mrad]."""
        return self._corners

    def _copy_kwargs(self, exclude: tuple[str, ...] = (), cls=None) -> dict:
        # The constructor takes the geometry either as corners or as the slit
        # parameters, not both, so a copy (and every lazy block) is rebuilt from
        # the form it was given in.
        if self._from_corners:
            exclude = exclude + ("width", "q_min", "q_max", "angle", "offset")
        else:
            exclude = exclude + ("corners",)
        kwargs = super()._copy_kwargs(exclude=exclude, cls=cls)
        if "q_max" in kwargs:
            # as given, not q_min + extent, which can differ in the last bit
            kwargs["q_max"] = self._q_max
        return kwargs

    def angular_limits(self, waves: WavesType) -> tuple[float, float]:
        """Radial bounds [mrad] of the acceptance region, for grid-sufficiency
        checks. The slit has no rotationally-symmetric inner exclusion, so
        the inner bound is 0; the outer bound is the farthest distance from
        the origin reached by the bounding rectangle's corners."""
        kx_min, kx_max, ky_min, ky_max = self.corners
        outer = max(
            float(np.hypot(kx, ky))
            for kx in (kx_min, kx_max)
            for ky in (ky_min, ky_max)
        )
        return 0.0, outer

    def _out_metadata(self, array_object: WavesType) -> tuple[dict]:
        metadata = super()._out_metadata(array_object)[0]
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return (metadata,)

    def _out_ensemble_axes_metadata(
        self, waves: WavesType
    ) -> tuple[list[AxisMetadata]]:
        source = _scan_axes(waves)
        scan_axes_metadata = [waves.ensemble_axes_metadata[i] for i in source]
        ensemble_axes_metadata = [
            m for i, m in enumerate(waves.ensemble_axes_metadata) if i not in source
        ]
        return (ensemble_axes_metadata + scan_axes_metadata,)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        return ([],)

    def _out_ensemble_shape(self, waves: WavesType) -> tuple[tuple[int, ...], ...]:
        ensemble_shapes = super()._out_ensemble_shape(waves)
        source = _scan_axes(waves)
        if not source:
            return ensemble_shapes
        # Drop exactly the axes _scan_axes identifies, by position -- not the
        # last two entries, which need not be the scan axes (e.g. a GridScan
        # probe's own energy ensemble trails its two ScanAxis entries).
        drop = {i + len(self.ensemble_shape) for i in source}
        return tuple(
            tuple(s for i, s in enumerate(ensemble_shape) if i not in drop)
            for ensemble_shape in ensemble_shapes
        )

    def _out_ensemble_source(
        self, waves: WavesType
    ) -> tuple[tuple[int, ...], ...]:
        source = _scan_axes(waves)
        if not source:
            return super()._out_ensemble_source(waves)
        kept = [i for i in range(len(waves.ensemble_shape)) if i not in source]
        return (tuple(kept + list(source)),)

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, ...]]:
        return (_scan_shape(waves),)

    def _out_dtype(self, waves: WavesType) -> tuple[np.dtype]:
        return (np.finfo(waves.dtype).dtype,)

    def _out_type(
        self, waves: WavesType
    ) -> tuple[Type[RealSpaceLineProfiles] | Type[Images] | Type[MeasurementsEnsemble]]:
        return (_scanned_measurement_type(waves),)

    def _mask(self, gpts, sampling, fftshift: bool = False, xp=np) -> np.ndarray:
        return _slit_detector_mask(
            gpts=gpts,
            sampling=sampling,
            origin=self._offset,
            angle=self._angle,
            q_min=self.q_min,
            # as given, like _copy_kwargs: q_min + extent can differ in the last bit
            q_max=self._q_max,
            width=self._width,
            fftshift=fftshift,
            xp=xp,
        )

    def _get_detector_region_array(
        self, waves, fftshift: bool = True
    ) -> np.ndarray:
        gpts, angular_sampling, _, _ = _gpts_and_sampling_from_obj(waves)
        return self._mask(gpts, angular_sampling, fftshift=fftshift)

    def get_detector_region(self, waves, fftshift: bool = True):
        """
        Get the slit detector region as a DiffractionPatterns object.

        Parameters
        ----------
        waves : BaseWaves or DiffractionPatterns
            The waves or diffraction patterns used to derive grid calibration.
        fftshift : bool, optional
            If True, the zero-frequency component is shifted to the centre.
        """
        array = self._get_detector_region_array(waves, fftshift=fftshift)
        _, _, reciprocal_space_sampling, energy = _gpts_and_sampling_from_obj(waves)
        metadata = {
            "energy": energy,
            "label": "detector efficiency",
            "units": "%",
        }
        return DiffractionPatterns(
            array,
            metadata=metadata,
            sampling=reciprocal_space_sampling,
            fftshift=fftshift,
        )

    @staticmethod
    def _show_pattern_bg(ax, waves, power):
        """Render the summed DP as a grayscale imshow background."""
        from abtem.measurements import DiffractionPatterns

        if not isinstance(waves, DiffractionPatterns):
            raise ValueError(
                "show_pattern=True requires a DiffractionPatterns object"
            )
        arr = np.array(
            waves.array.compute() if hasattr(waves.array, "compute") else waves.array
        )
        if arr.ndim > 2:
            arr = arr.sum(axis=tuple(range(arr.ndim - 2)))
        if not getattr(waves, "fftshift", True):
            arr = np.fft.fftshift(arr)
        if power != 1.0:
            arr = np.abs(arr) ** power
        mx, my = waves.max_angles
        from abtem.core import config

        cmap = config.get("visualize.cmap", "viridis")
        ax.imshow(
            arr,
            extent=[-mx, mx, -my, my],
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        return mx, my

    def show(
        self,
        waves,
        show_pattern: bool = False,
        power: float = 0.5,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Show the slit detector region as a polygon patch.

        Parameters
        ----------
        waves : BaseWaves or DiffractionPatterns
            Provides grid calibration.  When *show_pattern* is True, the
            diffraction pattern (summed over all ensemble axes) is shown as a
            grayscale background and *waves* must be a
            :class:`~abtem.measurements.DiffractionPatterns`.
        show_pattern : bool, optional
            Overlay the patch on the summed diffraction pattern.  Requires a
            :class:`~abtem.measurements.DiffractionPatterns` as *waves*.
        power : float, optional
            Exponent applied to the pattern before display (default 0.5 →
            square-root stretch).  Ignored when *show_pattern* is False.
        ax : matplotlib Axes, optional
        figsize : tuple, optional
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (6, 6))

        mx = my = None
        if show_pattern:
            mx, my = self._show_pattern_bg(ax, waves, power)

        # Compute world-space corners of the rotated rectangle.
        cos_a, sin_a = cos_sin_deg(self._angle)
        half_e = self._extent / 2.0
        half_w = self._width / 2.0
        # Centre of the slit rectangle in world coords
        center = np.array([
            self._offset[0] + (self._q_min + half_e) * cos_a,
            self._offset[1] + (self._q_min + half_e) * sin_a,
        ])
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local = np.array([
            [-half_e, -half_w],
            [half_e, -half_w],
            [half_e, half_w],
            [-half_e, half_w],
        ])
        world = local @ R.T + center

        patch = MplPolygon(
            world,
            closed=True,
            facecolor="red",
            alpha=0.25,
            edgecolor="red",
            linewidth=1.5,
        )
        ax.add_patch(patch)

        ax.set_aspect("equal")
        ax.set_xlabel("kx [mrad]")
        ax.set_ylabel("ky [mrad]")
        if not show_pattern:
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        if mx is not None:
            ax.set_xlim(-mx, mx)
            ax.set_ylim(-my, my)
        else:
            lim = (self.q_max + self._width) * 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.set_title(
            f"SpectralSlitDetector  width={self._width} mrad  angle={self._angle}°"
        )
        return ax

    def _calculate_new_array(self, waves: WavesType) -> np.ndarray:
        xp = get_array_module(waves.array)

        diffraction_patterns = waves.diffraction_patterns(
            max_angle="full", parity="same", fftshift=False
        )
        gpts = diffraction_patterns.shape[-2:]
        sampling = diffraction_patterns.angular_sampling

        mask = self._mask(gpts, sampling, xp=xp)
        intensity = xp.sum(
            diffraction_patterns._eager_array * mask, axis=(-2, -1)
        )

        if self.to_cpu and hasattr(intensity, "get"):
            intensity = intensity.get()

        return intensity

    def detect(
        self, waves: WavesType
    ) -> Images | RealSpaceLineProfiles | MeasurementsEnsemble:
        """
        Detect the given waves producing images.

        Parameters
        ----------
        waves : Waves

        Returns
        -------
        measurement : Images or RealSpaceLineProfiles
        """
        measurements = self.apply(waves)
        assert isinstance(
            measurements, (RealSpaceLineProfiles, Images, MeasurementsEnsemble)
        )
        return measurements


class SpectralAnnularDetector(AnnularDetector):
    """
    Sweeps an offset circular acceptance region over q to build S(q, E).

    The acceptance disk (radius ``outer``, inner always 0) is centred at
    ``(q·cos(angle), q·sin(angle))`` for each q in ``[q_min, q_max)``.
    Pass to :func:`abtem.momentum_resolved_spectrum` together with
    energy-resolved diffraction patterns to obtain a
    :class:`~abtem.measurements.MomentumResolvedSpectrum`.

    Parameters
    ----------
    outer : float
        Acceptance **radius** [mrad] of the integration disk at each q-point.
        The full disk diameter is ``2 * outer``.  The q-axis in the resulting
        :class:`~abtem.measurements.MomentumResolvedSpectrum` runs from
        ``q_min`` to ``q_max`` in approximately ``outer``-sized steps.  For
        equivalent perpendicular
        acceptance as a :class:`SpectralSlitDetector` with ``width=w``, use
        ``outer = w / 2``.
    q_min : float, optional
        Start of the q sweep [mrad].  Default is 0.
    q_max : float, optional
        End of the q sweep [mrad].  If None (default), the diffraction-pattern
        cutoff angle is used at call time.  To cover the same q-range as a
        :class:`SpectralSlitDetector` with ``q_max=Q``, use the same
        ``q_max=Q``.
    angle : float, optional
        Direction of the q sweep [degrees, CCW from kx].  Default is 0.
    q_sampling : float, optional
        Step between q-points [mrad].  If None (default) the step equals
        ``outer`` (one disk-radius per step).  Setting a larger value
        produces fewer q-points and a faster spectrum.
    to_cpu : bool, optional
    url : str, optional

    Notes
    -----
    **Comparing annular and slit detectors**

    =========================  ====================================
    SpectralAnnularDetector    SpectralSlitDetector
    =========================  ====================================
    ``outer`` — disk radius    ``width/2`` — half-width
    ``q_max`` — max q          ``q_max`` — max q
    =========================  ====================================

    For equivalent perpendicular acceptance and the same q-range::

        SpectralAnnularDetector(outer=r, q_max=Q)
        SpectralSlitDetector(q_max=Q, width=2*r)
    """

    def __init__(
        self,
        outer: float,
        q_min: float = 0.0,
        q_max: Optional[float] = None,
        angle: float = 0.0,
        q_sampling: Optional[float] = None,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._q_min = float(q_min)
        self._q_max = q_max
        self._sweep_angle = float(angle)
        self._q_sampling = float(q_sampling) if q_sampling is not None else None
        super().__init__(
            inner=0.0, outer=outer, offset=(0.0, 0.0), to_cpu=to_cpu, url=url
        )

    def _copy_kwargs(self, exclude: tuple[str, ...] = (), cls=None) -> dict:
        # The constructor's `angle` is the sweep angle; a copy (and every lazy
        # block) is rebuilt from it.
        kwargs = super()._copy_kwargs(exclude=exclude + ("angle",), cls=cls)
        if "angle" not in exclude:
            kwargs["angle"] = self.sweep_angle
        return kwargs

    @property
    def q_min(self) -> float:
        """Start of the q sweep [mrad]."""
        return self._q_min

    @property
    def q_max(self) -> Optional[float]:
        """End of the q sweep [mrad], or None to use the DP cutoff angle."""
        return self._q_max

    @property
    def q_sampling(self) -> Optional[float]:
        """Step between q-points [mrad], or None to use ``outer``."""
        return self._q_sampling

    @property
    def sweep_angle(self) -> float:
        """Direction of the q sweep [degrees, CCW from kx]."""
        return self._sweep_angle

    def show(
        self,
        waves,
        show_pattern: bool = False,
        power: float = 0.5,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Show all acceptance-disk positions along the q-sweep.

        Each disk (radius ``outer``) is drawn at the q-position it would be
        centred on when computing a spectrum, so the full sweep from ``q_min``
        to ``q_max`` is visible at once.

        Parameters
        ----------
        waves : BaseWaves or DiffractionPatterns
            Provides grid calibration and, when *show_pattern* is True, the
            diffraction data.  Must be a
            :class:`~abtem.measurements.DiffractionPatterns` when
            *show_pattern* is True.
        show_pattern : bool, optional
            Overlay the disks on the summed diffraction pattern shown as a
            grayscale background.
        power : float, optional
            Exponent applied to the pattern before display (default 0.5 →
            square-root stretch).  Ignored when *show_pattern* is False.
        ax : matplotlib Axes, optional
        figsize : tuple, optional
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        from abtem.measurements import DiffractionPatterns

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (6, 6))

        mx = my = None
        if show_pattern:
            mx, my = SpectralSlitDetector._show_pattern_bg(ax, waves, power)

        # q-values that will be swept
        if isinstance(waves, DiffractionPatterns):
            q_max_dp = min(waves.max_angles)
        else:
            q_max_dp = min(waves.cutoff_angles)
        q_max = self.q_max if self.q_max is not None else q_max_dp
        step = self.q_sampling if self.q_sampling is not None else self.outer
        n_steps = max(2, round((q_max - self.q_min) / step) + 1)
        q_vals = np.linspace(self.q_min, q_max, n_steps)

        cos_a, sin_a = cos_sin_deg(self._sweep_angle)

        for q in q_vals:
            cx, cy = q * cos_a, q * sin_a
            ax.add_patch(
                Circle(
                    (cx, cy),
                    self.outer,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.8,
                    alpha=0.6,
                )
            )

        ax.set_aspect("equal")
        ax.set_xlabel("kx [mrad]")
        ax.set_ylabel("ky [mrad]")
        if not show_pattern:
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        if mx is not None:
            ax.set_xlim(-mx, mx)
            ax.set_ylim(-my, my)
        else:
            lim = (q_max + self.outer) * 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.set_title(
            f"SpectralAnnularDetector  outer={self.outer} mrad"
            f"  angle={self._sweep_angle}°"
        )
        return ax


class FlexibleAnnularDetector(_AbstractRadialDetector):
    """
    The flexible annular detector allows choosing the integration limits after running
    the simulation by binning the intensity in annular integration regions.

    Parameters
    ----------
    step_size : float, optional
        Radial extent of the bins [mrad] (default is 1).
    inner : float, optional
        Inner integration limit of the bins [mrad].
    outer : float, optional
        Outer integration limit of the bins [mrad]. Every bin is ``step_size``
        wide, so if ``outer - inner`` is not a multiple of ``step_size`` the
        trailing partial step is dropped and the last bin ends at
        ``inner + n * step_size``. If not given, the antialias cutoff angle of
        the detected waves is used.
    sensitivity : RadialSensitivity, callable or tuple of array-like, optional
        A radially varying detector sensitivity. If given, the diffraction patterns are
        weighted by this factor as a function of the radial scattering angle [mrad]
        before binning, simulating a detector that is not uniformly sensitive across the
        bins. May be given as a :class:`RadialSensitivity`, a callable ``w(alpha)``, or a
        tuple ``(angles, values)`` describing a measured sensitivity curve. By default
        the bins are uniformly sensitive. Note that integrating the resulting polar
        measurement reproduces the signal of a sensitivity-weighted annular detector.
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory
        after applying the detector, otherwise the data stays on the respective
        devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified
        location, typically a path to a local file. A URL can also include a
        protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        inner: float = 0.0,
        outer: Optional[float] = None,
        sensitivity: Optional[RadialSensitivity | Callable | tuple[Any, Any]] = None,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._step_size = step_size
        super().__init__(
            inner=inner,
            outer=outer,
            rotation=0.0,
            offset=(0.0, 0.0),
            sensitivity=sensitivity,
            to_cpu=to_cpu,
            url=url,
        )

    def _nbins_within(self, outer: float) -> int:
        # Whole steps only: a trailing partial step is not binned.
        return safe_floor_int((outer - self.inner) / self.step_size)

    @property
    def nbins_radial(self):
        return self._nbins_within(self.outer)

    @property
    def nbins_azimuthal(self):
        return 1

    def _binned_outer(self, outer: float) -> float:
        # The binned range ends at the last whole step, so every bin is exactly
        # step_size wide: bin i is [inner + i * step, inner + (i + 1) * step).
        return self.inner + self._nbins_within(outer) * self.step_size

    @property
    def step_size(self) -> float:
        """Step size [mrad]."""
        return self._step_size

    @step_size.setter
    def step_size(self, value: float):
        self._step_size = value

    @property
    def radial_sampling(self) -> float:
        return self.step_size

    @property
    def azimuthal_sampling(self) -> float:
        return 2 * np.pi


class SegmentedDetector(_AbstractRadialDetector):
    """
    The segmented detector covers an annular angular range, and is partitioned into
    several integration regions divided to radial and angular segments. This can be
    used for simulating differential phase contrast (DPC) imaging.

    Parameters
    ----------
    nbins_radial : int
        Number of radial bins.
    nbins_azimuthal : int
        Number of angular bins.
    inner : float
        Inner integration limit of the bins [mrad].
    outer : float
        Outer integration limit of the bins [mrad]. If None, the antialias cutoff angle
        of the detected waves.
    rotation : float
        Rotation of the bins around the origin [rad].
    offset : two float
        Offset of the bins from the origin in `x` and `y` [mrad].
    sensitivity : RadialSensitivity, callable or tuple of array-like, optional
        A radially varying detector sensitivity. If given, the diffraction patterns are
        weighted by this factor as a function of the radial scattering angle [mrad]
        before binning. May be given as a :class:`RadialSensitivity`, a callable
        ``w(alpha)``, or a tuple ``(angles, values)`` describing a measured sensitivity
        curve. By default the bins are uniformly sensitive.
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory
        after applying the detector, otherwise the data stays on the respective devices.
        Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified
        location,typically a path to a local file. A URL can also include a protocol
        specifier like s3:// for remote data. If not set (default) the data stays in
        memory.
    """

    def __init__(
        self,
        nbins_radial: int,
        nbins_azimuthal: int,
        inner: float,
        outer: Optional[float],
        rotation: float = 0.0,
        offset: tuple[float, float] = (0.0, 0.0),
        sensitivity: Optional[RadialSensitivity | Callable | tuple[Any, Any]] = None,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._nbins_radial = nbins_radial
        self._nbins_azimuthal = nbins_azimuthal
        super().__init__(
            inner=inner,
            outer=outer,
            rotation=rotation,
            offset=offset,
            sensitivity=sensitivity,
            to_cpu=to_cpu,
            url=url,
        )

    @property
    def rotation(self):
        return self._rotation

    @property
    def radial_sampling(self):
        return (self.outer - self.inner) / self.nbins_radial

    @property
    def azimuthal_sampling(self):
        return 2 * np.pi / self.nbins_azimuthal

    @property
    def nbins_radial(self) -> int:
        """Number of radial bins."""
        return self._nbins_radial

    @nbins_radial.setter
    def nbins_radial(self, value: int):
        self._nbins_radial = value

    @property
    def nbins_azimuthal(self) -> int:
        """Number of angular bins."""
        return self._nbins_azimuthal

    @nbins_azimuthal.setter
    def nbins_azimuthal(self, value: int):
        self._nbins_azimuthal = value


def _crop_margin(waves: WavesType, margin: float | tuple[float, float]) -> WavesType:
    """Crop `margin` [Å] off each side of `waves`, or return it unchanged if
    `margin` is zero. Factored out of `Waves.window` so a detector can get the
    post-margin shape (for `_out_base_shape` and friends) without also paying
    for -- or needing -- the windowed array itself."""
    if isinstance(margin, (int, float)):
        margin = (float(margin), float(margin))

    if margin[0] == 0.0 and margin[1] == 0.0:
        return waves

    return waves.crop(
        extent=(waves.extent[0] - 2 * margin[0], waves.extent[1] - 2 * margin[1]),
        centered=True,
    )


def _window_power_gain(window: str | tuple, shape: tuple[int, int]) -> float:
    """`1 / mean(taper**2)` for the separable window `window` over `shape`.

    A window function's mean squared value is below one almost everywhere, so
    tapering a signal with one lowers its total power; multiplying detected
    intensities by this factor undoes that loss, so windowed and unwindowed
    detections stay on a comparable intensity scale. For a square Hann window
    this reduces to the `(8/3)**2` constant the 3DED project's own analysis
    tools apply by hand.
    """
    from scipy.signal import windows as scipy_windows

    window_x = scipy_windows.get_window(window, shape[0])
    window_y = scipy_windows.get_window(window, shape[1])
    taper = window_x[:, None] * window_y[None, :]
    return float(1.0 / np.mean(taper**2))


class PixelatedDetector(BaseDetector):
    """
    The pixelated detector records the intensity of the Fourier-transformed exit wave
    function, i.e. the diffraction patterns. This may be used for example for simulating
    4D-STEM.

    Parameters
    ----------
    max_angle : float or {'cutoff', 'valid', 'full'}
        The diffraction patterns will be detected up to this angle [mrad].
        If str, it must be one of:

        ``cutoff``
            The maximum scattering angle will be the cutoff of the antialiasing
            aperture.
        ``valid``
            The maximum scattering angle will be the largest rectangle that fits
            inside the circular antialiasing aperture (default).
        ``full``
            Diffraction patterns will not be cropped and will include angles outside
            the antialiasing aperture.
        For waves with several energies, every energy is cropped to the same
        number of pixels, those of the highest energy, as
        `Waves.diffraction_patterns` crops such waves.
    resample : str or False
        If 'uniform', the diffraction patterns from rectangular cells will be
        downsampled to a uniform angular sampling.
    reciprocal_space : bool, optional
        If True (default), the diffraction pattern intensities are detected, otherwise
        the probe intensities are
        detected as images.
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory
        after applying the detector,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified
        location, typically a path to a local file. A URL can also include a protocol
        specifier like s3:// for remote data. If not set (default) the data stays in
        memory.
    components : {'all', 'total', 'elastic', 'diffuse'} or sequence of str, optional
        Detect the elastic, diffuse and total intensity of a frozen-phonon ensemble
        instead of the intensity of each configuration, as
        :func:`abtem.measurements.elastic_diffuse_diffraction_patterns` forms them
        from the exit waves of every configuration. Each multislice task keeps the
        diffracted amplitude ψ(k) and its intensity |ψ(k)|² cropped to `max_angle`;
        the ensemble mean over the frozen-phonon axis averages both as the tasks
        finish, so the exit waves of the configurations are never kept. The
        elastic intensity |<ψ>|² and the diffuse intensity <|ψ|²> - |<ψ>|² are
        formed after that mean, and any other ensemble-mean axis is averaged
        after the split, i.e. incoherently. A single name returns that component,
        ``'all'`` or a sequence stack them in the order given along a leading
        ``OrdinalAxis(label='component')`` (``'all'`` is
        ``('total', 'elastic', 'diffuse')``). Requires a ``FrozenPhononsAxis``
        with ``ensemble_mean=True`` among the ensemble axes, ``reciprocal_space``
        and no `resample`. The components are formed when the ensemble is reduced,
        as in the result of a multislice simulation; a detector applied to waves
        directly returns the unreduced amplitudes and intensities, to be reduced
        with :func:`abtem.waves.reduce_ensemble`. Supported by multislice and by
        the PRISM scan (:meth:`abtem.prism.SMatrix.scan`), except with
        ``upsample=True``, which raises ``NotImplementedError``. With several
        energies, every energy's components are cropped to the pixel count of the
        highest energy, as for `max_angle` in mrad, so they stack along the energy
        axis. Default is None, which detects the intensity of every configuration.
    unbiased : bool, optional
        Correct the (N - 1)/N bias of the diffuse intensity, see
        :func:`abtem.measurements.elastic_diffuse_diffraction_patterns`. Only with
        `components`. Default is False.
    reduction_dtype : {None, 'float32', 'float64'}, optional
        Precision of the amplitudes and intensities that are averaged, see
        :func:`abtem.measurements.elastic_diffuse_diffraction_patterns`. Only with
        `components`. Default is None, the precision of the exit waves. The diffuse
        intensity is the difference of the total and the elastic intensity, so
        where it is far below the Bragg peaks (weak thermal displacements, a plane
        wave or a large probe) it is limited by the rounding of the large
        intensities: in single precision it can be off by as much as the diffuse
        intensity itself, and even negative. Use ``reduction_dtype='float64'`` for
        a weak diffuse signal.
    margin : float or tuple of float, optional
        Margin cropped from each side of the exit wave [Å] before detecting, and
        before applying `window_func` (default is 0.0, i.e. no cropping). Useful
        for a finite, non-periodic structure (e.g. from :func:`abtem.atoms.cut_disk`
        or :func:`abtem.atoms.cut_ball`), where the simulation cell is padded
        beyond the structure itself. See :meth:`abtem.waves.Waves.crop`.
    window_func : str or tuple, optional
        Real-space window function applied before detecting, as accepted by
        :func:`scipy.signal.windows.get_window` (default is `None`, i.e. no
        windowing). Tapers the (optionally margin-cropped) exit wave to
        (near-)zero at the edges, suppressing the truncation-rod streaks a
        finite, non-periodic structure would otherwise produce in its
        diffraction pattern. The detected intensity is automatically rescaled
        by the window's power gain (`1 / mean(taper**2)`) so windowed and
        unwindowed detections stay on a comparable scale. See
        :meth:`abtem.waves.Waves.window`.
    """

    # Cropping and resampling work on pixels and 1/Å, the same for every energy
    # once the crop is fixed for the whole ensemble (see _match_ensemble), so a
    # multi-energy ensemble is detected at once.
    _splits_energy_ensembles = False

    def __init__(
        self,
        max_angle: str | float = "valid",
        resample: str | tuple[float, float] | bool = False,
        reciprocal_space: bool = True,
        to_cpu: bool = True,
        url: Optional[str] = None,
        components: Optional[str | Sequence[str]] = None,
        unbiased: bool = False,
        reduction_dtype: Optional[str | np.dtype] = None,
        margin: float | tuple[float, float] = 0.0,
        window_func: Optional[str | tuple] = None,
        _ensemble_gpts: Optional[tuple[int, int]] = None,
    ):
        if not reciprocal_space and isinstance(resample, str):
            raise ValueError(
                f"resample={resample!r} applies to diffraction patterns only; "
                "in real space give the sampling of the images [Å]."
            )
        if not isinstance(unbiased, (bool, np.bool_)):
            raise TypeError(f"unbiased must be a bool, got {type(unbiased).__name__}")
        reduction_dtype = _validate_reduction_dtype(reduction_dtype)
        if components is None:
            if unbiased or reduction_dtype is not None:
                raise ValueError("unbiased and reduction_dtype apply to components")
            self._component_names = ()
            self._components_stacked = False
        else:
            if not reciprocal_space or resample:
                # The amplitude and the intensity of a pattern do not resample
                # alike, and the intensity of a real-space image has no amplitude.
                raise ValueError(
                    "components needs diffraction patterns without resampling: "
                    "reciprocal_space=True and resample=False."
                )
            self._component_names, self._components_stacked = (
                _validate_frozen_phonon_components(components)
            )
        self._components = components
        self._unbiased = bool(unbiased)
        self._reduction_dtype = reduction_dtype
        self._resample = resample
        self._max_angle = max_angle
        self._reciprocal_space = reciprocal_space
        self._margin = margin
        self._window_func = window_func
        # The crop of a multi-energy ensemble before any resampling, set by
        # _match_ensemble and kept through the constructor so that lazy blocks
        # rebuild it.
        self._ensemble_gpts = _ensemble_gpts
        super().__init__(to_cpu=to_cpu, url=url)

    def _match_ensemble(self, waves: WavesType) -> PixelatedDetector:
        """Fix the crop size of a multi-energy ensemble.

        Cropped to a `max_angle` in mrad, each energy has its own number of pixels,
        since the angular sampling scales with the wavelength. Every energy is cropped
        to the pixel count of the whole ensemble instead, before any resampling,
        as `Waves.diffraction_patterns` crops an ensemble at once, so the members
        stack, and their axes are labelled with the ensemble's sampling.
        """
        from abtem.array import _multi_energy_axis

        # A string max_angle crops every energy to the same pixels already.
        if (
            self._ensemble_gpts is not None
            or not self.reciprocal_space
            or isinstance(self.max_angle, str)
            or _multi_energy_axis(waves) is None
        ):
            return self

        # The highest energy of the axis decides. It is counted in pixels of the
        # waves actually detected, i.e. after the margin crop, which changes the
        # reciprocal sampling.
        matched = self.copy()
        matched._ensemble_gpts = self._crop_gpts(_crop_margin(waves, self.margin))
        return matched

    def _crop_gpts(self, waves: WavesType) -> tuple[int, int]:
        """The number of pixels the diffraction patterns are cropped to before any
        resampling: those within `max_angle`, or the ensemble's (see
        `_match_ensemble`)."""
        if self._ensemble_gpts is not None:
            return self._ensemble_gpts
        if self.max_angle:
            return waves._gpts_within_angle(self.max_angle)
        return waves._valid_gpts

    @property
    def max_angle(self) -> str | float:
        """Maximum detected scattering angle."""
        return self._max_angle

    @property
    def reciprocal_space(self) -> bool:
        """Detect the exit wave functions in real or reciprocal space."""
        return self._reciprocal_space

    @property
    def resample(self) -> str | bool | tuple[float, float]:
        """How to resample the detected diffraction patterns."""
        return self._resample

    @property
    def components(self) -> Optional[str | Sequence[str]]:
        """The frozen-phonon components detected, None for the intensities of the
        configurations."""
        return self._components

    @property
    def unbiased(self) -> bool:
        """Correct the bias of the diffuse intensity of a frozen-phonon ensemble."""
        return self._unbiased

    @property
    def reduction_dtype(self) -> Optional[np.dtype]:
        """Precision of the amplitudes and intensities averaged over the frozen
        phonons."""
        return self._reduction_dtype

    @property
    def _moment_channels(self) -> tuple[str, ...]:
        """The real quantities detected for `components`, stacked along the first
        base axis, from which the components are formed after the ensemble mean:
        the real and imaginary part of the amplitude ψ(k) for the elastic
        intensity |<ψ>|² and the intensity |ψ(k)|² for the total <|ψ|²>; the
        diffuse intensity (and the unbiased elastic one) needs both."""
        names = self._component_names
        both = _needs_the_diffuse_part(names, self.unbiased)
        return (
            *(("real", "imag") if both or "elastic" in names else ()),
            *(("intensity",) if both or "total" in names else ()),
        )

    @property
    def margin(self) -> float | tuple[float, float]:
        """Margin cropped from each side of the exit wave [Å] before detecting."""
        return self._margin

    @property
    def window_func(self) -> Optional[str | tuple]:
        """Real-space window function applied before detecting."""
        return self._window_func

    def angular_limits(self, waves: Waves) -> tuple[float, float]:
        waves = _crop_margin(waves, self.margin)
        if isinstance(self.max_angle, str):
            if self.max_angle == "valid":
                cutoff = waves.rectangle_cutoff_angles
            elif self.max_angle == "cutoff":
                cutoff = waves.cutoff_angles
            elif self.max_angle == "full":
                cutoff = waves.full_cutoff_angles
            else:
                raise RuntimeError()
        else:
            cutoff = waves.cutoff_angles

        return 0.0, min(cutoff)

    def _new_sampling_and_gpts(self, waves: WavesType):
        """
        Calculate the reciprocal-space sampling and grid points for the detector output.

        Determines the output shape of the diffraction pattern after optional resampling
        and max_angle cropping. The returned values must be consistent with the actual
        array produced by ``_calculate_new_array``, since they are used to pre-allocate
        measurement arrays during multislice simulations.

        Parameters
        ----------
        waves : WavesType
            The input waves used to determine reciprocal-space sampling and grid size.

        Returns
        -------
        sampling : tuple[float, float]
            Reciprocal-space sampling in each dimension (Å⁻¹ or mrad).
        gpts : tuple[int, int]
            Number of grid points in each dimension for the detector output.
        """
        waves = _crop_margin(waves, self.margin)

        if self.resample:
            sampling = waves.reciprocal_space_sampling
            gpts = self._crop_gpts(waves)

            gpts, sampling = _diffraction_pattern_resampling_gpts(
                old_sampling=sampling,
                old_gpts=gpts,
                sampling=self.resample,
                gpts=None,
                adjust_sampling=False,
            )

            if self.max_angle:
                gpts = tuple(
                    min(g, g_max) for g, g_max in zip(gpts, self._crop_gpts(waves))
                )
        else:
            sampling = waves.reciprocal_space_sampling
            gpts = self._crop_gpts(waves)

        return sampling, gpts

    def _real_space_sampling_and_gpts(self, waves: WavesType):
        """The sampling [Å] and grid points of the intensity images detected in real
        space: the waves' own grid, or the grid `Images.interpolate` gives with
        `resample` as its sampling. The grid is that of the margin-cropped waves,
        which `_calculate_new_array` detects."""
        waves = _crop_margin(waves, self.margin)
        sampling, gpts = waves._valid_sampling, waves._valid_gpts
        if self.resample:
            extent = tuple(d * n for d, n in zip(sampling, gpts))
            gpts = _image_resampling_gpts(extent, self.resample)
            sampling = tuple(e / n for e, n in zip(extent, gpts))
        return sampling, gpts

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        if self.components is not None:
            # The quantities of `_moment_channels` side by side along the first
            # base axis, until the ensemble is reduced.
            gpts = self._new_sampling_and_gpts(waves)[1]
            return ((len(self._moment_channels) * gpts[0], gpts[1]),)
        if self.reciprocal_space:
            return (self._new_sampling_and_gpts(waves)[1],)
        return (self._real_space_sampling_and_gpts(waves)[1],)

    def _out_dtype(self, waves: WavesType) -> tuple[np.dtype]:
        if self.reduction_dtype is not None:
            return (
                np.finfo(np.result_type(self.reduction_dtype, np.complex64)).dtype,
            )
        if self.resample and not self.reciprocal_space:
            # Images.interpolate resamples in the configured precision
            return (get_dtype(complex=False),)
        return (np.finfo(waves.dtype).dtype,)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        if self.reciprocal_space:
            sampling, gpts = self._new_sampling_and_gpts(waves)

            return (
                [
                    ReciprocalSpaceAxis(
                        sampling=sampling[0],
                        offset=-(gpts[0] // 2) * sampling[0],
                        label="kx",
                        units="1/Å",
                        fftshift=True,
                        tex_label="$k_x$",
                    ),
                    ReciprocalSpaceAxis(
                        sampling=sampling[1],
                        offset=-(gpts[1] // 2) * sampling[1],
                        label="ky",
                        units="1/Å",
                        fftshift=True,
                        tex_label="$k_y$",
                    ),
                ],
            )
        else:
            sampling = self._real_space_sampling_and_gpts(waves)[0]
            return (
                [
                    RealSpaceAxis(label="x", sampling=sampling[0], units="Å"),
                    RealSpaceAxis(label="y", sampling=sampling[1], units="Å"),
                ],
            )

    def _out_type(self, waves: WavesType) -> tuple[Type[DiffractionPatterns | Images]]:
        if self.reciprocal_space:
            return (DiffractionPatterns,)
        else:
            return (Images,)

    def _out_metadata(self, waves: WavesType) -> tuple[dict]:
        metadata = super()._out_metadata(waves)[0]
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        if self.components is not None:
            # Read by `BaseMeasurements.reduce_ensemble`, which forms the components
            # from the averaged quantities.
            metadata["frozen_phonon_moments"] = {
                "channels": list(self._moment_channels),
                "components": list(self._component_names),
                "stacked": self._components_stacked,
                "unbiased": self.unbiased,
            }
        return (metadata,)

    def _calculate_moments(self, waves: WavesType) -> np.ndarray:
        """The quantities of `_moment_channels` of the diffraction patterns of the
        given waves, stacked along the first base axis."""
        if self.reduction_dtype is not None:
            dtype = np.result_type(self.reduction_dtype, np.complex64)
            if waves.dtype != dtype:
                kwargs = waves._copy_kwargs(exclude=("array",))
                kwargs["array"] = waves.array.astype(dtype)
                waves = waves.__class__(**kwargs)

        # After the cast, so a wider reduction_dtype windows in that precision.
        waves = waves.window(window=self.window_func, margin=self.margin)

        if self._ensemble_gpts is not None:
            patterns = waves._diffraction_patterns(
                self._ensemble_gpts, return_complex=True
            )
        else:
            patterns = waves.diffraction_patterns(
                max_angle=self.max_angle, parity="same", return_complex=True
            )

        amplitude = patterns.array
        if self.window_func is not None:
            # The amplitude carries the square root of the power gain, so that the
            # intensity |ψ(k)|², and the elastic intensity |<ψ>|² formed from it,
            # carry the gain itself, as in the detection of the intensity.
            renorm = _window_power_gain(self.window_func, waves.base_shape) ** 0.5
            amplitude = amplitude * renorm
        xp = get_array_module(amplitude)
        quantities = {
            "real": amplitude.real,
            "imag": amplitude.imag,
            "intensity": abs2(amplitude),
        }
        array = xp.concatenate(
            [quantities[name] for name in self._moment_channels], axis=-2
        )

        if self.to_cpu:
            array = asnumpy(array)

        return array

    def _calculate_new_array(self, waves: WavesType) -> np.ndarray:
        """
        Detect the given waves producing diffraction patterns.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : DiffractionPatterns
        """
        measurements: Images | DiffractionPatterns

        if self.components is not None:
            return self._calculate_moments(waves)

        waves = waves.window(window=self.window_func, margin=self.margin)

        if self.reciprocal_space and self._ensemble_gpts is not None:
            measurements = waves._diffraction_patterns(self._ensemble_gpts)
        elif self.reciprocal_space:
            measurements = waves.diffraction_patterns(
                max_angle=self.max_angle, parity="same"
            )

        else:
            measurements = waves.intensity()

        if self.resample:
            measurements = measurements.interpolate(sampling=self.resample)

        if self.to_cpu:
            measurements = measurements.to_cpu()

        array = measurements._eager_array

        if self.window_func is not None:
            renorm = _window_power_gain(self.window_func, waves.base_shape)
            array = array * renorm  # a Python float keeps the dtype (NEP 50)

        return array

    def detect(self, waves: WavesType) -> DiffractionPatterns | Images:
        """
        Detect the given waves producing diffraction patterns.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : DiffractionPatterns
        """
        measurements = super().detect(waves)
        assert isinstance(measurements, (DiffractionPatterns, Images))
        return measurements


class WindowedPixelatedDetector(PixelatedDetector):
    """Alias for :class:`PixelatedDetector` with `margin`/`window_func`, kept so
    code written against the 3DED project's py3DED package
    (github.com/3DED/py3DED, `py3DED.detector.WindowedPixelatedDetector`) runs
    unmodified against abTEM directly. New code should use `PixelatedDetector`."""


class WavesDetector(BaseDetector):
    """
    Detect the complex wave functions.

    Parameters
    ----------
    gpts : two int, optional
       Number of grid points of the detected wave functions. The waves are
       Fourier-interpolated onto `gpts` points over their unchanged extent, as
       :meth:`abtem.waves.Waves.downsample` does. If not given (default), the waves
       keep their grid.
    to_cpu : bool, optional
       If True, copy the measurement data from the calculation device to CPU memory
       after applying the detector, otherwise the data stays on the respective devices.
       Default is False: unlike the other detectors, this one returns the (large)
       wave functions themselves, and it is the implicit detector used when no
       detectors are given, so the data is left on the calculation device rather
       than forcing a device-to-host copy of every exit wave.
    url : str, optional
       If this parameter is set the measurement data is saved at the specified location,
       typically a path to a local file. A URL can also include a protocol specifier
       like s3:// for remote data. If not set (default) the data stays in memory.
    average_frozen_phonons : bool, optional
       If True, return the configuration-averaged wave function <ψ> of a
       frozen-phonon ensemble, the coherent (elastic) wave, instead of the wave
       function of every configuration: the ``FrozenPhononsAxis`` is averaged when
       the ensemble is reduced, as the multislice tasks finish, so the exit waves
       of the configurations are not all kept. Other ensemble axes are kept as
       they are. Requires a ``FrozenPhononsAxis`` with ``ensemble_mean=True``
       among the ensemble axes; the average is formed when the ensemble is reduced,
       as in the result of a multislice simulation, and a detector applied to
       waves directly returns the waves of every configuration, to be reduced with
       :func:`abtem.waves.reduce_ensemble`. For the elastic and diffuse
       intensities in reciprocal space, use ``PixelatedDetector(components=...)``.
       Default is False, which detects the wave function of every configuration.
    reduction_dtype : {None, 'float32', 'float64'}, optional
       Precision of the wave functions that are averaged: with ``'float64'`` the
       detected waves are complex128 and so is their average. Only with
       `average_frozen_phonons`. Default is None, the precision of the waves.
    margin : float or tuple of float, optional
        Margin cropped from each side of the exit wave [Å] before detecting, and
        before applying `window_func` (default is 0.0, i.e. no cropping). See
        :meth:`abtem.waves.Waves.crop`.
    window_func : str or tuple, optional
        Real-space window function applied before detecting, as accepted by
        :func:`scipy.signal.windows.get_window` (default is `None`, i.e. no
        windowing). See :meth:`abtem.waves.Waves.window` and
        :class:`PixelatedDetector`. The returned wave functions' amplitude is
        automatically rescaled by the square root of the window's power gain
        (`1 / mean(taper**2)`), so that their intensity (`abs(array)**2`)
        stays on the same scale as an unwindowed detection would give.
    """

    # The wave functions do not depend on the wavelength once computed, so a
    # multi-energy ensemble is passed on at once, without a copy.
    _splits_energy_ensembles = False

    def __init__(
        self,
        gpts: Optional[tuple[int, int]] = None,
        to_cpu: bool = False,
        url: Optional[str] = None,
        average_frozen_phonons: bool = False,
        reduction_dtype: Optional[str | np.dtype] = None,
        margin: float | tuple[float, float] = 0.0,
        window_func: Optional[str | tuple] = None,
    ):
        if not isinstance(average_frozen_phonons, (bool, np.bool_)):
            raise TypeError(
                "average_frozen_phonons must be a bool, got "
                f"{type(average_frozen_phonons).__name__}"
            )
        reduction_dtype = _validate_reduction_dtype(reduction_dtype)
        if reduction_dtype is not None and not average_frozen_phonons:
            raise ValueError("reduction_dtype applies to average_frozen_phonons")
        self._gpts = gpts
        self._average_frozen_phonons = bool(average_frozen_phonons)
        self._reduction_dtype = reduction_dtype
        self._margin = margin
        self._window_func = window_func
        super().__init__(to_cpu=to_cpu, url=url)

    @property
    def average_frozen_phonons(self) -> bool:
        """The wave function is averaged over the frozen phonons."""
        return self._average_frozen_phonons

    @property
    def reduction_dtype(self) -> Optional[np.dtype]:
        """Precision of the wave functions averaged over the frozen phonons."""
        return self._reduction_dtype

    @property
    def gpts(self) -> Optional[tuple[int, int]]:
        """Number of grid points of the detected wave functions."""
        return self._gpts

    @property
    def margin(self) -> float | tuple[float, float]:
        """Margin cropped from each side of the exit wave [Å] before detecting."""
        return self._margin

    @property
    def window_func(self) -> Optional[str | tuple]:
        """Real-space window function applied before detecting."""
        return self._window_func

    def _out_dtype(self, waves: Waves) -> tuple[np.dtype]:
        if self.reduction_dtype is not None:
            return (np.result_type(self.reduction_dtype, np.complex64),)
        # An empty gpts asks for no resampling, as None does (the other _out_*
        # methods and _calculate_new_array test it the same way).
        if self._gpts:
            # fft_interpolate works in the configured precision
            return (get_dtype(complex=True),)
        return (waves.dtype,)

    def _out_type(self, waves: Waves) -> tuple[Type[Waves]]:
        from abtem.waves import Waves

        return (Waves,)

    def _out_metadata(self, waves: Waves) -> tuple[dict]:
        metadata = super()._out_metadata(array_object=waves)[0]
        metadata["reciprocal_space"] = False
        if self.average_frozen_phonons:
            # Read by `reduce_ensemble`, which averages the frozen phonons.
            metadata["frozen_phonon_average"] = True
        if self._gpts:
            # as `Waves.downsample` records it: the resampled waves keep the
            # band limit of the waves they were resampled from
            metadata["adjusted_antialias_cutoff_gpts"] = waves.antialias_cutoff_gpts
        return (metadata,)

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        if self._gpts:
            return (tuple(self._gpts),)
        return (_crop_margin(waves, self.margin).base_shape,)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        if not self._gpts:
            # cropping the margin keeps the sampling of the waves
            return super()._out_base_axes_metadata(waves)
        # `gpts` points over the extent of the (margin-cropped) waves, as
        # `Waves.downsample` gives
        extent = _crop_margin(waves, self.margin)._valid_extent
        sampling = tuple(length / n for length, n in zip(extent, self._gpts))
        return (
            [
                RealSpaceAxis(
                    label="x", sampling=sampling[0], units="Å", endpoint=False
                ),
                RealSpaceAxis(
                    label="y", sampling=sampling[1], units="Å", endpoint=False
                ),
            ],
        )

    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        waves = waves.ensure_real_space()
        waves = waves.window(window=self.window_func, margin=self.margin)

        if self.to_cpu:
            waves = waves.to_cpu()

        if self._gpts:
            array = fft_interpolate(waves._eager_array, new_shape=tuple(self._gpts))
        else:
            array = waves.array

        if self.reduction_dtype is not None:
            array = array.astype(self._out_dtype(waves)[0])

        if self.window_func is not None:
            renorm = _window_power_gain(self.window_func, waves.base_shape) ** 0.5
            array = array * renorm  # a Python float keeps the dtype (NEP 50)

        return array

    def detect(self, waves: WavesType) -> Waves:
        """
        Detect the given waves directly as complex waves.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : Waves
        """
        measurements = super().detect(waves)
        assert isinstance(measurements, Waves)
        return measurements

    def angular_limits(self, waves: BaseWaves) -> tuple[float, float]:
        return 0.0, min(waves.full_cutoff_angles)
