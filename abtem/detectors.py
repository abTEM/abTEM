"""Module for describing the detection of transmitted waves and different detector types."""

from __future__ import annotations

from abc import abstractmethod
from copy import copy
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from abtem.array import T
from abtem.core.axes import ReciprocalSpaceAxis, RealSpaceAxis, LinearAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks
from abtem.core.energy import energy2wavelength
from abtem.core.ensemble import _wrap_with_array
from abtem.core.units import units_type
from abtem.measurements import (
    DiffractionPatterns,
    PolarMeasurements,
    Images,
    RealSpaceLineProfiles,
    _scanned_measurement_type,
    _polar_detector_bins,
    _scan_shape,
    _scan_axes,
)
from abtem.transform import ArrayObjectTransform
from abtem.visualize.visualizations import discrete_cmap

if TYPE_CHECKING:
    from abtem.waves import BaseWaves, Waves
    from abtem.measurements import BaseMeasurements


def _validate_detectors(
    detectors: BaseDetector | list[BaseDetector], waves=None
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

    if waves is not None:
        for detector in detectors:
            if hasattr(detector, "_match_waves"):
                detector._match_waves(waves)

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

    @classmethod
    def _from_partition_args_func(cls, *args, **kwargs):
        detector = cls(**kwargs)
        return _wrap_with_array(detector)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs()
        return partial(self._from_partition_args_func, **kwargs)

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

    def _out_ensemble_axes_metadata(
        self, waves: BaseWaves, index: int = 0
    ) -> list[AxisMetadata]:
        source = _scan_axes(waves)
        scan_axes_metadata = [waves.ensemble_axes_metadata[i] for i in source]
        ensemble_axes_metadata = [
            m for i, m in enumerate(waves.ensemble_axes_metadata) if i not in source
        ]
        return ensemble_axes_metadata + scan_axes_metadata

    def _out_base_axes_metadata(
        self, waves: BaseWaves, index: int = 0
    ) -> list[AxisMetadata]:
        # source = _scan_axes(waves)
        # scan_axes_metadata = [waves.ensemble_axes_metadata[i] for i in source]
        return []

    def _out_ensemble_shape(self, waves: BaseWaves, index: int = 0) -> tuple:
        ensemble_shape = super()._out_ensemble_shape(waves, index)  # noqa
        return ensemble_shape[: -len(_scan_shape(waves))]

    def _out_base_shape(self, waves: BaseWaves, index: int = 0) -> tuple:
        return _scan_shape(waves)

    def _out_dtype(self, array_object, index: int = 0) -> np.dtype.base:
        return np.float32

    def _out_type(
        self, waves: BaseWaves, index: int = 0
    ) -> RealSpaceLineProfiles | Images:
        return _scanned_measurement_type(waves)

    def _calculate_new_array(self, waves: Waves):
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
        return self.apply(waves)  # noqa

    def _get_detector_region_array(
        self, waves: BaseWaves, fftshift: bool = True
    ) -> np.ndarray:
        array = _polar_detector_bins(
            gpts=waves._gpts_within_angle("cutoff"),
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

    @property
    @abstractmethod
    def nbins_radial(self):
        """Spacing between the azimuthal detector bins [mrad]."""
        pass

    @property
    @abstractmethod
    def nbins_azimuthal(self):
        """Spacing between the azimuthal detector bins [mrad]."""
        pass

    def _out_dtype(self, waves: Waves, index: bool = 0):
        return np.float32

    def _out_base_shape(self, waves: Waves, index: bool = 0):
        self._match_waves(waves)
        return self.nbins_radial, self.nbins_azimuthal

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

        return measurement.array

    def _match_waves(self, waves):
        if self.outer is None:
            self._outer = min(waves.cutoff_angles)

    def detect(self, waves: Waves) -> PolarMeasurements:
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
        return self.apply(waves)  # noqa

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

        bins = np.arange(0, self.nbins_radial * self.nbins_azimuthal)
        bins = bins.reshape((self.nbins_radial, self.nbins_azimuthal))

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

    def show(
        self,
        waves: BaseWaves = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[int, int] = None,
        energy: float = None,
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
            elif energy is None:
                energy = 100e3

            if gpts is None:
                gpts = 1024

            if not hasattr(gpts, "__len__"):
                gpts = (gpts,) * 2

            if sampling is None:
                angular_sampling = (
                    self.outer / gpts[0] * 2 * 1.1,
                    self.outer / gpts[1] * 2 * 1.1,
                )
                reciprocal_space_sampling = (
                    angular_sampling[0] / (energy2wavelength(energy) * 1e3),
                    angular_sampling[1] / (energy2wavelength(energy) * 1e3),
                )
            else:
                if not hasattr(sampling, "__len__"):
                    sampling = (sampling,) * 2

                reciprocal_space_sampling = 1 / (gpts[0] * sampling[0]), 1 / (
                    gpts[0] * sampling[0]
                )
                angular_sampling = (
                    reciprocal_space_sampling[0] * energy2wavelength(energy) * 1e3,
                    reciprocal_space_sampling[1] * energy2wavelength(energy) * 1e3,
                )

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
            
            regions = regions.astype(np.float32)
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

    def _out_metadata(self, waves: Waves, index: int = 0) -> dict:
        metadata = super()._out_metadata(waves, index=0)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return metadata

    def _calculate_new_array(self, waves: Waves):
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
        return self.apply(waves)  # noqa


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

    def _out_metadata(self, waves: Waves, index=0) -> dict:
        metadata = super()._out_metadata(array_object=waves, index=index)
        metadata["reciprocal_space"] = False
        return metadata

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
