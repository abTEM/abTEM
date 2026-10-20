"""Module for describing the detection of transmitted waves and different detector
types."""

from __future__ import annotations

from abc import abstractmethod
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar

import numpy as np

from abtem.core.axes import AxisMetadata, LinearAxis, RealSpaceAxis, ReciprocalSpaceAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks
from abtem.core.energy import energy2wavelength
from abtem.core.ensemble import _wrap_with_array
from abtem.core.fft import fft_interpolate
from abtem.core.utils import get_dtype, safe_floor_int
from abtem.measurements import (
    BaseMeasurements,
    DiffractionPatterns,
    Images,
    MeasurementsEnsemble,
    PolarMeasurements,
    RealSpaceLineProfiles,
    _diffraction_pattern_resampling_gpts,
    _polar_detector_bins,
    _scan_axes,
    _scan_shape,
    _scanned_measurement_type,
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
        matched = []
        for detector in detectors:
            if hasattr(detector, "_match_waves"):
                # Matching writes the waves' cutoff angle onto the detector. Done on
                # a copy, the caller's detector keeps no outer angle that a later,
                # unrelated run (another energy, grid or algorithm) would reuse.
                detector = detector.copy()
                detector._match_waves(waves)
            matched.append(detector)
        detectors = matched

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

    @abstractmethod
    def angular_limits(self, waves: Waves) -> tuple[float, float]:
        """
        The outer limits of the detected scattering angles in x and y [mrad] for the
        given waves.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the detector limits from.

        Returns
        -------
        limits : tuple of float
        """


class _AbstractRadialDetector(BaseDetector):
    def __init__(
        self,
        inner: float,
        outer: Optional[float] = None,
        rotation: float = 0.0,
        offset: tuple[float, float] = (0.0, 0.0),
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._inner = inner
        self._outer = outer
        # Whether `outer` was ever given explicitly (constructor or setter), as
        # opposed to auto-matched from waves in `_match_waves`. `outer is None`
        # alone cannot tell these apart once a value has been auto-matched, which
        # made `_match_waves` latch onto the first waves it ever saw.
        self._outer_is_explicit = outer is not None
        self._rotation = rotation
        self._offset = offset
        super().__init__(to_cpu=to_cpu, url=url)

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
        return (get_dtype(complex=False),)

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        self._match_waves(waves)
        return ((self.nbins_radial, self.nbins_azimuthal),)

    def _out_type(self, waves: WavesType) -> tuple[Type[PolarMeasurements]]:
        return (PolarMeasurements,)

    def _out_metadata(self, waves: WavesType) -> tuple[dict]:
        metadata = super()._out_metadata(waves)[0]
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return (metadata,)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        return (
            [
                LinearAxis(
                    label="Radial scattering angle",
                    offset=self.inner,
                    sampling=self.radial_sampling,
                    _concatenate=False,
                    units="mrad",
                ),
                LinearAxis(
                    label="Azimuthal scattering angle",
                    offset=self.rotation,
                    sampling=self.azimuthal_sampling,
                    _concatenate=False,
                    units="rad",
                ),
            ],
        )

    def angular_limits(self, waves: WavesType) -> tuple[float, float]:
        return self.inner, self._binned_outer(self._outer_for(waves))

    def _binned_outer(self, outer: float) -> float:
        """Outer edge of the outermost bin for a requested ``outer`` [mrad]."""
        return outer

    def _nbins_within(self, outer: float) -> int:
        """Number of radial bins for a requested ``outer`` [mrad]."""
        return self.nbins_radial

    def _outer_for(self, waves: WavesType) -> float:
        """The ``outer`` used to detect ``waves`` [mrad]: the given one, or
        else the antialias cutoff angle of ``waves`` (see `_match_waves`)."""
        if self.outer is not None:
            return self.outer

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
        inner, outer = self.angular_limits(waves)

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
            nbins_radial=self.nbins_radial,
            nbins_azimuthal=self.nbins_azimuthal,
            inner=inner,
            outer=outer,
            rotation=self._rotation,
            offset=self._offset,
        )

        if self.to_cpu:
            measurement = measurement.to_cpu()

        return measurement._eager_array

    def _match_waves(self, waves: WavesType) -> None:
        """Auto-size ``outer`` from ``waves``, unless the user gave one.

        This used to be guarded by ``if self.outer is None``, which conflated "the
        user never gave an outer" with "not matched yet": once matched, the detector
        kept the angle of the first waves it saw, also for a later, unrelated call
        (for example the same instance reused for a scan at another energy).
        Matching on every call makes each call consistent with its own waves.
        """
        if not self._outer_is_explicit:
            self._outer = min(waves.cutoff_angles)

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

        bins = np.arange(0, nbins_radial * self.nbins_azimuthal)
        bins = bins.reshape((nbins_radial, self.nbins_azimuthal))

        if waves is not None:
            metadata = copy(waves.metadata)
        else:
            metadata = {}

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

    def show(
        self,
        waves: Optional[BaseWaves] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
        **kwargs,
    ):
        """
        Show the segmented detector regions as a polar plot.

        The regions are drawn about the detector centre, including any
        ``offset``, out to the bins that detecting the waves would use.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the segmented detector regions from.
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
            energy = waves.energy
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

        regions = regions.astype(get_dtype(complex=False))
        regions[..., regions < 0] = np.nan

        diffraction_patterns = DiffractionPatterns(
            regions,
            sampling=reciprocal_space_sampling,
            fftshift=True,
            metadata={"energy": energy},
        )

        n_bins_radial = nbins_radial
        n_bins_azimuthal = self.nbins_azimuthal
        num_colors = n_bins_radial * n_bins_azimuthal

        if "cmap" not in kwargs:
            if num_colors <= 10:
                kwargs["cmap"] = "tab10"
            else:
                kwargs["cmap"] = "tab20"

        kwargs["cmap"] = discrete_cmap(num_colors=num_colors, base_cmap=kwargs["cmap"])

        if "vmin" not in kwargs:
            kwargs["vmin"] = -0.5

        if "vmax" not in kwargs:
            kwargs["vmax"] = num_colors - 0.5

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
    outer: float
        Outer integration limit [mrad].
    offset: two float, optional
        Center offset of the annular integration region [mrad].
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
    def offset(self) -> tuple[float, float]:
        """Center offset of the annular integration region [mrad]."""
        return self._offset

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

    def angular_limits(self, waves: BaseWaves) -> tuple[float, float]:
        inner = self.inner

        if self.outer is not None:
            outer = self.outer
        else:
            outer = min(waves.cutoff_angles)

        return inner, outer

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

        # Drop exactly the axes _scan_axes identifies, by position: a LineScan has
        # only one, so the last two entries need not all be scan axes.
        drop = {i + len(self.ensemble_shape) for i in source}
        return tuple(
            tuple(s for i, s in enumerate(ensemble_shape) if i not in drop)
            for ensemble_shape in ensemble_shapes
        )

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, ...]]:
        return (_scan_shape(waves),)

    def _out_dtype(self, waves: WavesType) -> tuple[np.dtype]:
        return (get_dtype(complex=False),)

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
        if self.outer is None:
            outer = np.floor(min(waves.cutoff_angles))
        else:
            outer = self.outer

        diffraction_patterns = waves.diffraction_patterns(
            max_angle="full", parity="same", fftshift=False
        )
        offset = self.offset if self.offset is not None else (0.0, 0.0)
        measurement = diffraction_patterns.integrate_radial(
            inner=self.inner, outer=outer, offset=offset
        )

        if self.to_cpu and hasattr(measurement, "to_cpu"):
            measurement = measurement.to_cpu()
        return measurement._eager_array

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
        self, waves: BaseWaves, fftshift: bool = True
    ) -> np.ndarray:
        inner, outer = self.angular_limits(waves)

        array = _polar_detector_bins(
            gpts=waves._gpts_within_angle("cutoff"),
            sampling=waves.angular_sampling,
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
        # A detector efficiency in %: 100 where the detector is sensitive.
        return (array >= 0).astype(get_dtype(complex=False)) * 100.0

    def get_detector_region(self, waves: BaseWaves, fftshift: bool = True):
        """
        Get the annular detector region as a diffraction pattern.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the annular detector region from.
        fftshift : bool, optional
            If True, the zero-frequency of the detector region if shifted to the center
            of the array, otherwise the center is at (0, 0).

        Returns
        -------
        detector_region : DiffractionPatterns
        """

        array = self._get_detector_region_array(waves, fftshift=fftshift)
        metadata = {
            "energy": waves.energy,
            "label": "detector efficiency",
            "units": "%",
        }
        diffraction_patterns = DiffractionPatterns(
            array,
            metadata=metadata,
            sampling=waves.reciprocal_space_sampling,
            fftshift=fftshift,
        )
        return diffraction_patterns


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
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        self._step_size = step_size
        super().__init__(
            inner=inner,
            outer=outer,
            rotation=0.0,
            offset=(0.0, 0.0),
            to_cpu=to_cpu,
            url=url,
        )

    def _nbins_within(self, outer: float) -> int:
        # The number of whole steps, floored after the division: a trailing
        # partial step (always present for the default auto outer) is dropped,
        # so the bins are as wide as the step_size reported as radial_sampling.
        return safe_floor_int((outer - self.inner) / self.step_size)

    @property
    def nbins_radial(self):
        return self._nbins_within(self.outer)

    def _binned_outer(self, outer: float) -> float:
        # The binned range ends at the last whole step, so every bin is exactly
        # step_size wide: bin i is [inner + i * step, inner + (i + 1) * step).
        return self.inner + self._nbins_within(outer) * self.step_size

    @property
    def nbins_azimuthal(self):
        return 1

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

    def detect(self, waves: Waves) -> PolarMeasurements:
        self._match_waves(waves)
        return super().detect(waves)


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
        Outer integration limit of the bins [mrad].
    rotation : float
        Rotation of the bins around the origin [mrad].
    offset : two float
        Offset of the bins from the origin in `x` and `y` [mrad].
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
        outer: float,
        rotation: float = 0.0,
        offset: tuple[float, float] = (0.0, 0.0),
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
    """

    def __init__(
        self,
        max_angle: str | float = "valid",
        resample: str | tuple[float, float] | bool = False,
        reciprocal_space: bool = True,
        to_cpu: bool = True,
        url: Optional[str] = None,
    ):
        if not reciprocal_space and isinstance(resample, str):
            raise ValueError(
                f"resample={resample!r} applies to diffraction patterns only; "
                "in real space give the sampling of the images [Å]."
            )
        self._resample = resample
        self._max_angle = max_angle
        self._reciprocal_space = reciprocal_space
        super().__init__(to_cpu=to_cpu, url=url)

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

    def angular_limits(self, waves: Waves) -> tuple[float, float]:
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
        if self.resample:
            sampling = waves.reciprocal_space_sampling
            gpts = waves._gpts_within_angle(self.max_angle)

            gpts, sampling = _diffraction_pattern_resampling_gpts(
                old_sampling=sampling,
                old_gpts=gpts,
                sampling=self.resample,
                gpts=None,
                adjust_sampling=False,
            )

            if self.max_angle:
                gpts = tuple(
                    min(g, g_max)
                    for g, g_max in zip(
                        gpts, waves._gpts_within_angle(self.max_angle)
                    )
                )
        elif self.max_angle and not self.resample:
            gpts = waves._gpts_within_angle(self.max_angle)
            sampling = waves.reciprocal_space_sampling
        else:
            sampling = waves.reciprocal_space_sampling
            gpts = waves._valid_gpts

        return sampling, gpts

    def _real_space_sampling_and_gpts(self, waves: WavesType):
        """The sampling [Å] and grid points of the intensity images detected in real
        space: the waves' own grid, or the grid `Images.interpolate` gives with
        `resample` as its sampling."""
        sampling, gpts = waves._valid_sampling, waves._valid_gpts
        if self.resample:
            extent = tuple(d * n for d, n in zip(sampling, gpts))
            resample = (
                (self.resample,) * 2 if np.isscalar(self.resample) else self.resample
            )
            gpts = tuple(int(np.ceil(e / d)) for e, d in zip(extent, resample))
            sampling = tuple(e / n for e, n in zip(extent, gpts))
        return sampling, gpts

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        if self.reciprocal_space:
            return (self._new_sampling_and_gpts(waves)[1],)
        return (self._real_space_sampling_and_gpts(waves)[1],)

    def _out_dtype(self, waves: WavesType) -> tuple[np.dtype]:
        return (get_dtype(complex=False),)

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
        return (metadata,)

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

        if self.reciprocal_space:
            measurements = waves.diffraction_patterns(
                max_angle=self.max_angle, parity="same"
            )

        else:
            measurements = waves.intensity()

        resample = self.resample
        if resample:
            if isinstance(measurements, Images):
                assert not isinstance(resample, str)
                measurements = measurements.interpolate(sampling=resample)
            else:
                measurements = measurements.interpolate(sampling=resample)

        if self.to_cpu:
            measurements = measurements.to_cpu()

        return measurements._eager_array

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
    """

    def __init__(
        self,
        gpts: Optional[tuple[int, int]] = None,
        to_cpu: bool = False,
        url: Optional[str] = None,
    ):
        self._gpts = gpts
        super().__init__(to_cpu=to_cpu, url=url)

    @property
    def gpts(self) -> Optional[tuple[int, int]]:
        """Number of grid points of the detected wave functions."""
        return self._gpts

    def _out_type(self, waves: Waves) -> tuple[Type[Waves]]:
        from abtem.waves import Waves

        return (Waves,)

    def _out_metadata(self, waves: Waves) -> tuple[dict]:
        metadata = super()._out_metadata(array_object=waves)[0]
        metadata["reciprocal_space"] = False
        if self._gpts:
            # as `Waves.downsample` records it: the resampled waves keep the
            # band limit of the waves they were resampled from
            metadata["adjusted_antialias_cutoff_gpts"] = waves.antialias_cutoff_gpts
        return (metadata,)

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        if self._gpts:
            return (tuple(self._gpts),)
        return super()._out_base_shape(waves)

    def _out_base_axes_metadata(self, waves: WavesType) -> tuple[list[AxisMetadata]]:
        if not self._gpts:
            return super()._out_base_axes_metadata(waves)
        # `gpts` points over the extent of the waves, as `Waves.downsample` gives
        sampling = tuple(
            length / n for length, n in zip(waves._valid_extent, self._gpts)
        )
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

        if self.to_cpu:
            waves = waves.to_cpu()

        if self._gpts:
            array = fft_interpolate(waves._eager_array, new_shape=tuple(self._gpts))
        else:
            array = waves.array

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
