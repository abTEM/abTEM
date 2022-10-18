"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Any, Union, List, TYPE_CHECKING, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from abtem.core.axes import FourierSpaceAxis, RealSpaceAxis, LinearAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.utils import CopyMixin
from abtem.measurements import (
    DiffractionPatterns,
    PolarMeasurements,
    Images,
    RealSpaceLineProfiles,
    _scanned_measurement_type,
)

if TYPE_CHECKING:
    from abtem.waves import BaseWaves
    from abtem.waves import Waves


def _validate_detectors(
    detectors: Union["BaseDetector", List["BaseDetector"]]
) -> List["BaseDetector"]:
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


class BaseDetector(CopyMixin, metaclass=ABCMeta):
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
        return self._url

    @property
    def to_cpu(self):
        return self._to_cpu

    def measurement_meta(self, waves: "BaseWaves"):
        if self.to_cpu:
            return np.array((), dtype=self.measurement_dtype)
        else:
            xp = get_array_module(waves.device)
            return xp.array((), dtype=self.measurement_dtype)

    def measurement_metadata(self, waves: "BaseWaves") -> dict:
        return waves.metadata

    @property
    @abstractmethod
    def measurement_dtype(self):
        pass

    @abstractmethod
    def measurement_shape(self, waves: "BaseWaves") -> Tuple:
        pass

    @abstractmethod
    def measurement_type(self, waves: "BaseWaves"):
        pass

    @abstractmethod
    def measurement_axes_metadata(self, waves: "BaseWaves"):
        pass

    @abstractmethod
    def detect(self, waves: "Waves") -> Any:
        pass

    @abstractmethod
    def angular_limits(self, waves: "BaseWaves") -> Tuple[float, float]:
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
        offset: Tuple[float, float] = None,
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
    def offset(self) -> Tuple[float, float]:
        return self._offset

    def measurement_metadata(self, waves: "BaseWaves") -> dict:
        metadata = super().measurement_metadata(waves)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return metadata

    def angular_limits(self, waves: "BaseWaves") -> Tuple[float, float]:
        if self.inner is not None:
            inner = self.inner
        else:
            inner = 0.0

        if self.outer is not None:
            outer = self.outer
        else:
            outer = min(waves.cutoff_angles)

        return inner, outer

    def measurement_axes_metadata(self, waves: "BaseWaves"):
        return []

    def measurement_shape(self, waves: "BaseWaves") -> Tuple:
        return ()

    @property
    def measurement_dtype(self) -> np.dtype.base:
        return np.float32

    def measurement_type(
        self, waves: "BaseWaves"
    ) -> Union[type(RealSpaceLineProfiles), type(Images)]:
        return _scanned_measurement_type(waves)

    def detect(self, waves: "Waves") -> Images:

        if self.outer is None:
            outer = np.floor(min(waves.cutoff_angles))
        else:
            outer = self.outer

        diffraction_patterns = waves.diffraction_patterns(
            max_angle="cutoff", parity="same"
        )
        measurement = diffraction_patterns.integrate_radial(
            inner=self.inner, outer=outer
        )

        if self.to_cpu and hasattr(measurement, "to_cpu"):
            measurement = measurement.to_cpu()

        return measurement

    def show(self, ax: Axes = None):
        bins = np.arange(0, 2).reshape((2, 1))
        bins[1, 0] = -1

        vmin = -0.5
        vmax = np.max(bins) + 0.5
        cmap = plt.get_cmap("tab20", np.nanmax(bins) + 1)
        cmap.set_under(color="white")

        polar_measurements = PolarMeasurements(
            bins,
            radial_sampling=self.outer - self.inner,
            azimuthal_sampling=2 * np.pi,
            radial_offset=self.inner,
            azimuthal_offset=0.0,
        )

        ax, im = polar_measurements.show(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, extend="min", label="Detector region")
        return ax, im


class _AbstractRadialDetector(BaseDetector):
    def __init__(
        self,
        inner: float,
        outer: float,
        rotation: float,
        offset: Tuple[float, float],
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
        return self._rotation

    @property
    @abstractmethod
    def radial_sampling(self):
        pass

    @property
    @abstractmethod
    def azimuthal_sampling(self):
        pass

    @abstractmethod
    def _calculate_nbins_radial(self, waves):
        pass

    @abstractmethod
    def _calculate_nbins_azimuthal(self, waves):
        pass

    @property
    def measurement_dtype(self):
        return np.float32

    def measurement_shape(self, waves: "Waves"):
        shape = (
            self._calculate_nbins_radial(waves),
            self._calculate_nbins_azimuthal(waves),
        )
        return shape

    def measurement_type(self, waves: "Waves"):
        return PolarMeasurements

    def measurement_metadata(self, waves: "BaseWaves") -> dict:
        metadata = super().measurement_metadata(waves)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"
        return metadata

    def measurement_axes_metadata(self, waves: "Waves"):
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

    def angular_limits(self, waves: "Waves") -> Tuple[float, float]:
        if self.inner is not None:
            inner = self.inner
        else:
            inner = 0.0

        if self.outer is not None:
            outer = self.outer
        else:
            outer = np.floor(min(waves.cutoff_angles))

        return inner, outer

    def detect(self, waves: "Waves"):
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

        return measurement

    def show(self, waves=None, ax=None, cmap="tab20", figsize=None, **kwargs):
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

        vmin = -0.5
        vmax = np.max(bins) + 0.5
        cmap = plt.get_cmap(cmap, np.nanmax(bins) + 1)
        cmap.set_under(color="white")
        polar_measurements = PolarMeasurements(
            bins,
            radial_sampling=self.radial_sampling,
            azimuthal_sampling=self.azimuthal_sampling,
            radial_offset=self.inner,
            azimuthal_offset=self._rotation,
        )

        ax, im = polar_measurements.show(
            ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, figsize=figsize, **kwargs
        )

        plt.colorbar(im, extend="min", label="Detector region")
        # ax.set_rlim([-0, min(waves.cutoff_angles) * 1.1])
        return ax, im


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

    def _calculate_nbins_radial(self, waves: "Waves") -> int:
        return int(np.floor(min(waves.cutoff_angles)) / self.step_size)

    def _calculate_nbins_azimuthal(self, waves: "Waves") -> int:
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
        offset: Tuple[float, float] = (0.0, 0.0),
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

    def _calculate_nbins_radial(self, waves: "Waves" = None):
        return self.nbins_radial

    def _calculate_nbins_azimuthal(self, waves: "Waves" = None):
        return self.nbins_azimuthal


class PixelatedDetector(BaseDetector):
    """
    The pixelated detector records the intensity of the Fourier-transformed exit wave function. This may be used for
    example for simulating 4D-STEM.

    Parameters
    ----------
    max_angle : str or float
        The diffraction patterns will be detected up to this angle [mrad]. If str, it must be one of:
        ``cutoff`` :
        The maximum scattering angle will be the cutoff of the antialiasing aperture.
        ``valid`` :
        The maximum scattering angle will be the largest rectangle that fits inside the circular antialiasing aperture
        (default).
        ``full`` :
        Diffraction patterns will not be cropped and will include angles outside the antialiasing aperture.
    resample : str or False
        If 'uniform', the diffraction patterns from rectangular cells will be downsampled to a uniform angular
        sampling.
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
        max_angle: Union[str, float] = "valid",
        resample: bool = False,
        fourier_space: bool = True,
        to_cpu: bool = True,
        url: str = None,
    ):
        self._resample = resample
        self._max_angle = max_angle
        self._fourier_space = fourier_space
        super().__init__(to_cpu=to_cpu, url=url)

    @property
    def max_angle(self) -> Union[str, float]:
        return self._max_angle

    @property
    def fourier_space(self) -> bool:
        return self._fourier_space

    @property
    def resample(self) -> Union[str, bool]:
        return self._resample

    def angular_limits(self, waves: "Waves") -> Tuple[float, float]:
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

    def measurement_shape(self, waves: "Waves") -> Tuple[int, int]:
        if self.fourier_space:
            shape = waves._gpts_within_angle(self.max_angle)
        else:
            shape = waves.gpts

        return shape

    @property
    def measurement_dtype(self) -> np.dtype.base:
        return np.float32

    def measurement_axes_metadata(self, waves: "Waves") -> List[AxisMetadata]:
        if self.fourier_space:
            sampling = waves.fourier_space_sampling
            gpts = waves._gpts_within_angle(self.max_angle)

            return [
                FourierSpaceAxis(
                    sampling=sampling[0],
                    offset=-(gpts[0] // 2) * sampling[0],
                    label="kx",
                    units="1 / Å",
                    fftshift=True,
                ),
                FourierSpaceAxis(
                    sampling=sampling[1],
                    offset=-(gpts[1] // 2) * sampling[1],
                    label="ky",
                    units="1 / Å",
                    fftshift=True,
                ),
            ]
        else:
            return [
                RealSpaceAxis(label="x", sampling=waves.sampling[0], units="Å"),
                RealSpaceAxis(label="y", sampling=waves.sampling[1], units="Å"),
            ]

    def measurement_type(self, waves: "Waves"):
        if self.fourier_space:
            return DiffractionPatterns
        else:
            return Images

    def detect(self, waves: "Waves") -> "DiffractionPatterns":
        """
        Calculate the far-field intensity of the wave functions. The output is cropped to include the non-suppressed
        frequencies from the antialiased 2D Fourier spectrum.

        Parameters
        ----------
        waves: Waves
            The batch of wave functions to detect.

        Returns
        -------
        values : DiffractionPatterns
            Detected values. The first dimension indexes the batch size, the second and third indexes the two components
            of the spatial frequency.
        """
        if self.fourier_space:
            measurements = waves.diffraction_patterns(
                max_angle=self.max_angle, parity="same"
            )

        else:
            measurements = waves.intensity()

        if self.to_cpu:
            measurements = measurements.to_cpu()

        return measurements


class WavesDetector(BaseDetector):
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

    def __init__(self, to_cpu: bool = False, url: str = None):
        super().__init__(to_cpu=to_cpu, url=url)

    def detect(self, waves: "Waves") -> "Waves":
        if self.to_cpu:
            waves = waves.to_cpu()
        return waves

    def angular_limits(self, waves: "Waves") -> Tuple[float, float]:
        return 0.0, min(waves.full_cutoff_angles)

    @property
    def measurement_dtype(self) -> np.dtype.base:
        return np.complex64

    def measurement_shape(self, waves: "Waves") -> Tuple[int, int]:
        return waves.gpts

    def measurement_type(self, waves: "Waves") -> Type["Waves"]:
        from abtem.waves import Waves

        return Waves

    def measurement_axes_metadata(self, waves: "Waves") -> List[AxisMetadata]:
        return waves.base_axes_metadata
