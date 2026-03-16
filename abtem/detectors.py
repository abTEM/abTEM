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
from abtem.core.units import units_type
from abtem.core.utils import get_dtype
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
        A list of validated detectors.

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
        for detector in detectors:
            if hasattr(detector, "_match_waves"):
                detector._match_waves(waves)

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

    @property
    def rotation(self):
        """Rotation of the bins around the origin [rad]."""
        return self._rotation

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
        inner = self.inner

        if self.outer is not None:
            outer = self.outer
        else:
            outer = np.floor(min(waves.cutoff_angles))

        return inner, outer

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

        measurement = waves.diffraction_patterns(max_angle=outer, parity="same")

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
        if self.outer is None:
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

    def get_detector_regions(self, waves: Optional[BaseWaves] = None):
        """
        Get the polar detector regions as a polar measurement.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the polar detector regions from.

        Returns
        -------
        detector_region : PolarMeasurements
        """

        bins = np.arange(0, self.nbins_radial * self.nbins_azimuthal)
        bins = bins.reshape((self.nbins_radial, self.nbins_azimuthal))

        if waves is not None:
            metadata = copy(waves.metadata)
        else:
            metadata = {}

        metadata.update({"label": "detector regions", "units": ""})

        polar_measurements = PolarMeasurements(
            bins,
            radial_sampling=self.radial_sampling,
            azimuthal_sampling=self.azimuthal_sampling,
            radial_offset=self.inner,
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

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the segmented detector regions from.
        gpts : two int, optional
            Number of grid points describing the wave functions to be detected.
        sampling : two float, optional
            Lateral sampling of the wave functions to be detected [1 / Å].
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
            segmented_regions = self.get_detector_regions(waves)
            diffraction_patterns = segmented_regions.to_diffraction_patterns(waves.gpts)
            energy = waves.energy
        elif energy is None:
            raise ValueError("provide the waves or the energy of waves")
        else:
            if units_type[kwargs["units"]] == "reciprocal_space":
                if energy is None:
                    raise ValueError(
                        "energy or waves must be provided when using real space units"
                    )
            if gpts is None:
                gpts = 1024

            if not isinstance(gpts, tuple):
                assert isinstance(gpts, int)
                gpts = (gpts,) * 2

            if sampling is None:
                assert isinstance(self.outer, float)
                angular_sampling = (
                    self.outer / float(gpts[0] * 2 * 1.1),
                    self.outer / float(gpts[1] * 2 * 1.1),
                )
                reciprocal_space_sampling = (
                    angular_sampling[0] / (energy2wavelength(energy) * 1e3),
                    angular_sampling[1] / (energy2wavelength(energy) * 1e3),
                )
            else:
                if not isinstance(sampling, tuple):
                    assert isinstance(sampling, float)
                    sampling = (sampling,) * 2

                reciprocal_space_sampling = (
                    1 / (gpts[0] * sampling[0]),
                    1 / (gpts[0] * sampling[0]),
                )
                angular_sampling = (
                    reciprocal_space_sampling[0] * energy2wavelength(energy) * 1e3,
                    reciprocal_space_sampling[1] * energy2wavelength(energy) * 1e3,
                )

            if self.outer is None:
                raise ValueError("provide the outer limit of the detector")

            regions = _polar_detector_bins(
                gpts=gpts,
                sampling=angular_sampling,
                inner=self.inner,
                outer=self.outer,
                nbins_radial=self.nbins_radial,
                nbins_azimuthal=self.nbins_azimuthal,
                fftshift=True,
                rotation=self.rotation,
                offset=(0.0, 0.0),
                return_indices=False,
            )
            assert isinstance(regions, np.ndarray)

            regions = regions.astype(get_dtype(complex=False))
            regions[..., regions < 0] = np.nan

            diffraction_patterns = DiffractionPatterns(
                regions, sampling=reciprocal_space_sampling, metadata={"energy": energy}
            )

        n_bins_radial = self.nbins_radial
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

        if len(_scan_shape(waves)) == 0:
            return ((),)
            # raise RuntimeError("annular detector requires a scan axis")

        return tuple(ensemble_shape[:-2] for ensemble_shape in ensemble_shapes)

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
        measurement = diffraction_patterns.integrate_radial(
            inner=self.inner, outer=outer
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
        return array >= 0

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
            array, metadata=metadata, sampling=waves.reciprocal_space_sampling
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
        Outer integration limit of the bins [mrad].
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

    @property
    def nbins_radial(self):
        return int(np.floor(self.outer - self.inner) / self.step_size)

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
        to_cpu: bool = False,
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
            ``cutoff`` :
                The maximum scattering angle will be the cutoff of the antialiasing
                aperture.
            ``valid`` :
                The maximum scattering angle will be the largest rectangle that fits
                inside the circular antialiasing aperture (default).
            ``full`` :
                Diffraction patterns will not be cropped and will include angles outside
                the antialiasing aperture.
    resample : str or False
        If 'uniform', the diffraction patterns from rectangular cells will be ¨
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
        if self.resample:
            sampling = waves.reciprocal_space_sampling
            gpts = waves._gpts_within_angle(self.max_angle)

            _, sampling = _diffraction_pattern_resampling_gpts(
                old_sampling=sampling,
                old_gpts=gpts,
                sampling=self.resample,
                gpts=None,
                adjust_sampling=False,
            )

            if self.max_angle and sampling[0] == sampling[1]:
                gpts = (min(gpts), min(gpts))
        elif self.max_angle and not self.resample:
            gpts = waves._gpts_within_angle(self.max_angle)
            sampling = waves.reciprocal_space_sampling
        else:
            sampling = waves.reciprocal_space_sampling
            gpts = waves._valid_gpts

        return sampling, gpts

    def _out_base_shape(self, waves: WavesType) -> tuple[tuple[int, int]]:
        return (self._new_sampling_and_gpts(waves)[1],)

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
            return (
                [
                    RealSpaceAxis(
                        label="x", sampling=waves._valid_sampling[0], units="Å"
                    ),
                    RealSpaceAxis(
                        label="y", sampling=waves._valid_sampling[1], units="Å"
                    ),
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
    to_cpu : bool, optional
       If True, copy the measurement data from the calculation device to CPU memory
       after applying the detector, otherwise the data stays on the respective devices.
       Default is True.
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

    def _out_type(self, waves: Waves) -> tuple[Type[Waves]]:
        from abtem.waves import Waves

        return (Waves,)

    def _out_metadata(self, waves: Waves) -> tuple[dict]:
        metadata = super()._out_metadata(array_object=waves)[0]
        metadata["reciprocal_space"] = False
        return (metadata,)

    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        waves = waves.ensure_real_space()

        if self.to_cpu:
            waves = waves.to_cpu()

        if self._gpts is not None:
            array = fft_interpolate(
                waves._eager_array, new_shape=waves.shape[:-2] + self._gpts
            )
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
