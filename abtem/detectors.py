"""Module for describing the detection of transmitted waves and different detector types."""
from __future__ import annotations

from abc import abstractmethod
from copy import copy
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from abtem.core.axes import ReciprocalSpaceAxis, RealSpaceAxis, LinearAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks
from abtem.measurements import (
    DiffractionPatterns,
    PolarMeasurements,
    Images,
    RealSpaceLineProfiles,
    _scanned_measurement_type,
    _polar_detector_bins,
)
from abtem.array import T
from abtem.transform import ArrayObjectTransform
from abtem.visualize import discrete_cmap


if TYPE_CHECKING:
    from abtem.waves import BaseWaves, Waves
    from abtem.measurements import BaseMeasurements
    from abtem.visualize import MeasurementVisualization2D


def _validate_detectors(
    detectors: BaseDetector | list[BaseDetector],
) -> list[BaseDetector]:
    if hasattr(detectors, "detect"):
        detectors = [detectors]

    elif detectors is None:
        detectors = [WavesDetector()]

    elif not (
        isinstance(detectors, list)
        and all(hasattr(detector, "detect") for detector in detectors)
    ):

        raise RuntimeError(
            "Detectors must be AbstractDetector or list of AbstractDetector."
        )

    return detectors


class BaseDetector(ArrayObjectTransform):
    """
    Base detector class.

    Parameters
    ----------
    to_cpu : bool, optional
       If True, copy the measurement data from the calculation device to CPU memory after applying the detector,
       otherwise the data stays on the respective devices. Default is True.
    url : str, optional
       If this parameter is set the measurement data is saved at the specified location, typically a path to a
       local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
       the data stays in memory.
    """

    def __init__(self, to_cpu: bool = True, url: str = None):
        self._to_cpu = to_cpu
        self._url = url

    @property
    def url(self):
        """The storage location of the measurement data."""
        return self._url

    @property
    def to_cpu(self):
        """The measurements are copied to host memory."""
        return self._to_cpu

    @property
    def _default_ensemble_chunks(self):
        return ()

    def _partition_args(self, chunks: Chunks = None, lazy: bool = True):
        return ()

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs()
        return partial(self.__class__, **kwargs)

    def _out_meta(self, waves: Waves, index=0) -> np.ndarray:
        """
        The meta describing the measurement array created when detecting the given waves.

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
            return np.array((), dtype=self._out_dtype(waves))
        else:
            xp = get_array_module(waves.device)

            return xp.array((), dtype=self._out_dtype(waves))

    def detect(self, waves: T) -> T:
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
        return self.apply(waves)

    @abstractmethod
    def angular_limits(self, waves: BaseWaves) -> tuple[float, float]:
        """
        The outer limits of the detected scattering angles in x and y [mrad] for the given waves.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the detector limits from.

        Returns
        -------
        limits : tuple of float
        """
        pass


class AnnularDetector(BaseDetector):
    """
    The annular detector integrates the intensity of the detected wave functions between an inner and outer radial
    integration limits, i.e. over an annulus.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float
        Outer integration limit [mrad].
    offset: two float, optional
        Center offset of the annular integration region [mrad].
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory after applying the detector,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified location, typically a path to a
        local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
    """

    def __init__(
        self,
        inner: float,
        outer: float,
        offset: tuple[float, float] = (0.0, 0.0),
        to_cpu: bool = True,
        url: str = None,
    ):

        self._inner = inner
        self._outer = outer
        self._offset = offset
        super().__init__(to_cpu=to_cpu, url=url)

    @property
    def inner(self) -> float:
        """Inner integration limit in mrad."""
        return self._inner

    @inner.setter
    def inner(self, value: float):
        self._inner = value

    @property
    def outer(self) -> float:
        """Outer integration limit in mrad."""
        return self._outer

    @outer.setter
    def outer(self, value: float):
        self._outer = value

    @property
    def offset(self) -> tuple[float, float]:
        """Center offset of the annular integration region [mrad]."""
        return self._offset

    def _out_metadata(self, array_object, index=0):
        metadata = super()._out_metadata(array_object)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return metadata

    def angular_limits(self, waves: BaseWaves) -> tuple[float, float]:
        if self.inner is not None:
            inner = self.inner
        else:
            inner = 0.0

        if self.outer is not None:
            outer = self.outer
        else:
            outer = min(waves.cutoff_angles)

        return inner, outer

    def _out_base_axes_metadata(
        self, waves: BaseWaves, index: int = 0
    ) -> list[AxisMetadata]:
        return []

    def _out_base_shape(self, waves: BaseWaves, index: int = 0) -> tuple:
        return ()

    def _out_dtype(self, array_object, index: int = 0) -> np.dtype.base:
        return np.float32

    def _out_type(
        self, waves: BaseWaves, index: int = 0
    ) -> type(RealSpaceLineProfiles) | type(Images):
        return _scanned_measurement_type(waves)

    def _calculate_new_array(self, waves):
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

        return measurement.array

    def detect(self, waves: Waves) -> Images:
        """
        Detect the given waves producing images.

        Parameters
        ----------
        waves : Waves
            The waves to detect.

        Returns
        -------
        measurement : Images
        """
        return self.apply(waves)

    def _get_detector_region_array(
        self, waves: BaseWaves, fftshift: bool = True
    ) -> np.ndarray:

        array = _polar_detector_bins(
            gpts=waves.gpts,
            sampling=waves.angular_sampling,
            inner=self.inner,
            outer=self.outer,
            nbins_radial=1,
            nbins_azimuthal=1,
            fftshift=fftshift,
            rotation=0.0,
            offset=self.offset,
            return_indices=False,
        )
        return array >= 0

    def get_detector_region(self, waves: BaseWaves, fftshift: bool = True):
        """
        Get the annular detector region as a diffraction pattern.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the annular detector region from.
        fftshift : bool, optional
            If True, the zero-frequency of the detector region if shifted to the center of the array, otherwise the
            center is at (0, 0).

        Returns
        -------
        detector_region : DiffractionPatterns
        """

        array = self._get_detector_region_array(waves, fftshift=fftshift)
        metadata = {"energy": waves.energy, "label": "detector efficiency"}
        diffraction_patterns = DiffractionPatterns(
            array, metadata=metadata, sampling=waves.reciprocal_space_sampling
        )
        return diffraction_patterns


class _AbstractRadialDetector(BaseDetector):
    def __init__(
        self,
        inner: float,
        outer: float,
        rotation: float,
        offset: tuple[float, float],
        to_cpu: bool = True,
        url: str = None,
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
    def outer(self) -> float:
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
        pass

    @property
    @abstractmethod
    def azimuthal_sampling(self):
        """Spacing between the azimuthal detector bins [mrad]."""
        pass

    @abstractmethod
    def _calculate_nbins_radial(self, waves: BaseWaves):
        pass

    @abstractmethod
    def _calculate_nbins_azimuthal(self, waves: BaseWaves):
        pass

    def _out_dtype(self, waves: Waves, index: bool = 0):
        return np.float32

    def _out_base_shape(self, waves: Waves, index: bool = 0):
        shape = (
            self._calculate_nbins_radial(waves),
            self._calculate_nbins_azimuthal(waves),
        )

        return shape

    def _out_type(self, waves: Waves, index: bool = 0):
        return PolarMeasurements

    def _out_metadata(self, waves: Waves, index: bool = 0) -> dict:
        metadata = super()._out_metadata(waves)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return metadata

    def _out_base_axes_metadata(self, waves: Waves, index: bool = 0):
        return [
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
        ]

    def angular_limits(self, waves: BaseWaves) -> tuple[float, float]:
        if self.inner is not None:
            inner = self.inner
        else:
            inner = 0.0

        if self.outer is not None:
            outer = self.outer
        else:
            outer = np.floor(min(waves.cutoff_angles))

        return inner, outer

    def _calculate_new_array(self, waves: Waves):
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

        measurement = waves.diffraction_patterns(max_angle="cutoff", parity="same")

        measurement = measurement.polar_binning(
            nbins_radial=self._calculate_nbins_radial(waves),
            nbins_azimuthal=self._calculate_nbins_azimuthal(waves),
            inner=inner,
            outer=outer,
            rotation=self._rotation,
            offset=self._offset,
        )

        if self.to_cpu:
            measurement = measurement.to_cpu()

        return measurement.array

    def detect(self, waves: Waves):
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
        return self.apply(waves)

    def get_detector_regions(self, waves: BaseWaves = None):
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

        bins = np.arange(
            0,
            self._calculate_nbins_radial(waves)
            * self._calculate_nbins_azimuthal(waves),
        )
        bins = bins.reshape(
            (
                self._calculate_nbins_radial(waves),
                self._calculate_nbins_azimuthal(waves),
            )
        )

        metadata = copy(waves.metadata)
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

    def show(self, waves: BaseWaves = None, **kwargs):
        """
        Show the segmented detector regions as a polar plot.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the segmented detector regions from.
        kwargs :
            Optional keyword arguments for PolarMeasurements.show.

        Returns
        -------
        visualization : MeasurementVisualization2D
        """

        num_colors = self._calculate_nbins_radial(
            waves
        ) * self._calculate_nbins_azimuthal(waves)

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

        segmented_regions = self.get_detector_regions(waves).to_diffraction_patterns(
            waves.gpts
        )

        return segmented_regions.show(**kwargs)


class FlexibleAnnularDetector(_AbstractRadialDetector):
    """
    The flexible annular detector allows choosing the integration limits after running the simulation by binning the
    intensity in annular integration regions.

    Parameters
    ----------
    step_size : float, optional
        Radial extent of the bins [mrad] (default is 1).
    inner : float, optional
        Inner integration limit of the bins [mrad].
    outer : float, optional
        Outer integration limit of the bins [mrad].
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory after applying the detector,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified location, typically a path to a
        local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        inner: float = 0.0,
        outer: float = None,
        to_cpu: bool = True,
        url: str = None,
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

    def _calculate_nbins_radial(self, waves: Waves) -> int:
        if self.outer is None:
            outer = min(waves.cutoff_angles)
        else:
            outer = self.outer

        return int(np.floor(outer - self.inner) / self.step_size)

    def _calculate_nbins_azimuthal(self, waves: Waves) -> int:
        return 1


class SegmentedDetector(_AbstractRadialDetector):
    """
    The segmented detector covers an annular angular range, and is partitioned into several integration regions
    divided to radial and angular segments. This can be used for simulating differential phase contrast (DPC)
    imaging.

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
        If True, copy the measurement data from the calculation device to CPU memory after applying the detector,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified location, typically a path to a
        local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
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
        url: str = None,
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
    def nbins_azimuthal(self, value: float):
        self._nbins_azimuthal = value

    def _calculate_nbins_radial(self, waves: Waves = None):
        return self.nbins_radial

    def _calculate_nbins_azimuthal(self, waves: Waves = None):
        return self.nbins_azimuthal


class PixelatedDetector(BaseDetector):
    """
    The pixelated detector records the intensity of the Fourier-transformed exit wave function, i.e. the diffraction
    patterns. This may be used for example for simulating 4D-STEM.

    Parameters
    ----------
    max_angle : float or {'cutoff', 'valid', 'full'}
        The diffraction patterns will be detected up to this angle [mrad]. If str, it must be one of:
            ``cutoff`` :
                The maximum scattering angle will be the cutoff of the antialiasing aperture.
            ``valid`` :
                The maximum scattering angle will be the largest rectangle that fits inside the circular antialiasing
                aperture (default).
            ``full`` :
                Diffraction patterns will not be cropped and will include angles outside the antialiasing aperture.
    resample : str or False
        If 'uniform', the diffraction patterns from rectangular cells will be downsampled to a uniform angular
        sampling.
    reciprocal_space : bool, optional
        If True (default), the diffraction pattern intensities are detected, otherwise the probe intensities are
        detected as images.
    to_cpu : bool, optional
        If True, copy the measurement data from the calculation device to CPU memory after applying the detector,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified location, typically a path to a
        local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
    """

    def __init__(
        self,
        max_angle: str | float = "valid",
        resample: bool = False,
        reciprocal_space: bool = True,
        to_cpu: bool = True,
        url: str = None,
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
    def resample(self) -> str | bool:
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

    def _out_base_shape(self, waves: Waves, index: int = 0) -> tuple[int, int]:
        if self.reciprocal_space:
            shape = waves._gpts_within_angle(self.max_angle)
        else:
            shape = waves.gpts

        return shape

    def _out_dtype(self, array_object, index=0) -> np.dtype.base:
        return np.float32

    def _out_base_axes_metadata(self, waves: Waves, index=0) -> list[AxisMetadata]:
        if self.reciprocal_space:
            sampling = waves.reciprocal_space_sampling
            gpts = waves._gpts_within_angle(self.max_angle)

            return [
                ReciprocalSpaceAxis(
                    sampling=sampling[0],
                    offset=-(gpts[0] // 2) * sampling[0],
                    label="kx",
                    units="1/Å",
                    fftshift=True,
                    _tex_label="$k_x$",
                ),
                ReciprocalSpaceAxis(
                    sampling=sampling[1],
                    offset=-(gpts[1] // 2) * sampling[1],
                    label="ky",
                    units="1/Å",
                    fftshift=True,
                    _tex_label="$k_y$",
                ),
            ]
        else:
            return [
                RealSpaceAxis(label="x", sampling=waves.sampling[0], units="Å"),
                RealSpaceAxis(label="y", sampling=waves.sampling[1], units="Å"),
            ]

    def _out_type(self, waves: Waves, index=0):
        if self.reciprocal_space:
            return DiffractionPatterns
        else:
            return Images

    def _out_metadata(self, waves: BaseWaves, index=0) -> dict:
        metadata = super()._out_metadata(waves, index=0)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return metadata

    def _calculate_new_array(self, waves):
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
        if self.reciprocal_space:
            measurements = waves.diffraction_patterns(
                max_angle=self.max_angle, parity="same"
            )

        else:
            measurements = waves.intensity()

        if self.to_cpu:
            measurements = measurements.to_cpu()

        return measurements.array

    def detect(self, waves: Waves) -> DiffractionPatterns:
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
        return self.apply(waves)


class WavesDetector(BaseDetector):
    """
    Detect the complex wave functions.

    Parameters
    ----------
    to_cpu : bool, optional
       If True, copy the measurement data from the calculation device to CPU memory after applying the detector,
       otherwise the data stays on the respective devices. Default is True.
    url : str, optional
       If this parameter is set the measurement data is saved at the specified location, typically a path to a
       local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
       the data stays in memory.
    """

    def __init__(self, to_cpu: bool = False, url: str = None):
        super().__init__(to_cpu=to_cpu, url=url)

    def _out_type(self, array_object, index: bool = 0):
        from abtem.waves import Waves

        return Waves

    def _calculate_new_array(self, waves):

        waves = waves.ensure_real_space()

        if self.to_cpu:
            waves = waves.to_cpu()

        return waves.array

    def detect(self, waves: Waves) -> Waves:
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
        return self.apply(waves)

    def angular_limits(self, waves: Waves) -> tuple[float, float]:
        return 0.0, min(waves.full_cutoff_angles)
