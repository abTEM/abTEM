"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy
from typing import Tuple, Any, Union, List, TYPE_CHECKING

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from abtem.core.axes import ScanAxis, FrozenPhononsAxis, OrdinalAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.measure.measure import DiffractionPatterns, PolarMeasurements, Images, LineProfiles

if TYPE_CHECKING:
    from abtem.waves.waves import Waves
    from abtem.potentials.potentials import AbstractPotential


def stack_measurements(measurements, axes_metadata):
    array = np.stack([measurement.array for measurement in measurements])
    cls = measurements[0].__class__
    d = measurements[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + d['extra_axes_metadata']
    return cls(**d)


def allocate_measurements(waves, scan, detectors, potential: 'AbstractPotential'):
    measurements = []
    for detector in detectors:

        extra_axes_shape = ()
        extra_axes_metadata = []

        if potential.num_frozen_phonons > 1 and not potential.ensemble_mean:
            extra_axes_shape = (potential.num_frozen_phonons,)
            extra_axes_metadata = [FrozenPhononsAxis()]

        if detector.detect_every:
            extra_axes_shape = (detector.num_detections(potential),)
            extra_axes_metadata = [OrdinalAxis()]

        measurement = detector.allocate_measurement(waves,
                                                    scan,
                                                    extra_axes_shape=extra_axes_shape,
                                                    extra_axes_metadata=extra_axes_metadata)

        measurements.append(measurement)

    return measurements


def validate_detectors(detectors: Union['AbstractDetector', List['AbstractDetector']]) -> List['AbstractDetector']:
    if hasattr(detectors, 'detect'):
        detectors = [detectors]

    elif detectors is None:
        detectors = [WavesDetector()]

    elif not (isinstance(detectors, list) and all(hasattr(detector, 'detect') for detector in detectors)):

        raise RuntimeError('detectors must be AbstractDetector or list of AbstractDetector')

    return detectors


class AbstractDetector(metaclass=ABCMeta):

    def __init__(self, to_cpu: bool = True, url: str = None, detect_every: int = None):
        """
        Abstract base detector class

        Parameters
        ----------
        to_cpu : bool, optional
            If True, copy the measurement data after applying the detector from the calculation device to cpu memory,
            otherwise the data stays on the respective devices. Default is True.
        url : str, optional
            If this parameter is set the measurement data is saved at the specified location, typically a path to a
            local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
            the data stays in memory.
        detect_every : int, optional
            If this parameter is set the 'detect' method of the detector is applied after every 'detect_every' step of
            the multislice algorithm, in addition to after the last step.
            If not set (default), the detector is only applied after the last step.
        """
        self._to_cpu = to_cpu
        self._url = url
        self._detect_every = detect_every

    def num_detections(self, potential):

        if self.detect_every:
            num_detect_thicknesses = len(potential) // self.detect_every

            if len(potential) % self.detect_every != 0:
                num_detect_thicknesses += 1

            return num_detect_thicknesses

        return 1

    @property
    def url(self):
        return self._url

    @property
    def detect_every(self):
        return self._detect_every

    @property
    def to_cpu(self):
        return self._to_cpu

    def allocate_measurement(self,
                             waves,
                             extra_axes_shape: Tuple[int, ...] = None,
                             extra_axes_metadata: List[AxisMetadata] = None,
                             lazy: bool = False):

        d = self.measurement_kwargs(waves)
        shape = self.measurement_shape(waves)

        if extra_axes_shape:
            assert len(extra_axes_metadata) == len(extra_axes_shape)
            shape = extra_axes_shape + shape
            d['extra_axes_metadata'] = extra_axes_metadata + d['extra_axes_metadata']

        if self.to_cpu:
            xp = np
        else:
            xp = get_array_module(waves.device)

        if lazy:
            d['array'] = da.zeros(shape, dtype=self.measurement_dtype)
        else:
            d['array'] = xp.zeros(shape, dtype=self.measurement_dtype)

        return self.measurement_type(waves)(**d)

    @property
    @abstractmethod
    def measurement_dtype(self):
        pass

    @abstractmethod
    def measurement_shape(self, waves) -> Tuple:
        pass

    @abstractmethod
    def measurement_type(self, waves):
        pass

    @abstractmethod
    def measurement_kwargs(self, waves):
        pass

    @abstractmethod
    def detect(self, waves) -> Any:
        pass

    @abstractmethod
    def angular_limits(self, waves) -> Tuple[float, float]:
        pass

    def copy(self):
        """Make a copy."""
        return deepcopy(self)


class AnnularDetector(AbstractDetector):
    """
    Annular detector.

    The annular detector integrates the intensity of the detected wave functions between an inner and outer radial
    integration limit, i.e. over an annulus.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float
        Outer integration limit [mrad].
    offset: two float, optional
        Center offset of the annular integration region [mrad].
    to_cpu : bool, optional
        If True, copy the measurement data after applying the detector from the calculation device to cpu memory,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified location, typically a path to a
        local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
    detect_every : int, optional
        If this parameter is set the 'detect' method of the detector is applied after every 'detect_every' step of
        the multislice algorithm, in addition to after the last step.
        If not set (default), the detector is only applied after the last step.
    """

    def __init__(self,
                 inner: float,
                 outer: float,
                 offset: Tuple[float, float] = None,
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):

        self._inner = inner
        self._outer = outer
        self._offset = offset
        super().__init__(to_cpu=to_cpu, url=url, detect_every=detect_every)

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
    def offset(self):
        return self._offset

    def angular_limits(self, waves) -> Tuple[float, float]:
        if self.inner is not None:
            inner = self.inner
        else:
            inner = 0.

        if self.outer is not None:
            outer = self.outer
        else:
            outer = min(waves.cutoff_angles)

        return inner, outer

    def measurement_shape(self, waves: 'Waves') -> Tuple:
        return waves.extra_axes_shape

    @property
    def measurement_dtype(self) -> type(np.float32):
        return np.float32

    def measurement_type(self, waves: 'Waves') -> Union[type(LineProfiles), type(Images)]:
        num_scan_axes = waves.num_scan_axes

        if num_scan_axes == 1:
            return LineProfiles
        elif num_scan_axes == 2:
            return Images
        else:
            raise RuntimeError(f'no measurement type for AnnularDetector and Waves with {num_scan_axes} scan axes')

    def measurement_kwargs(self, waves: 'Waves') -> dict:
        extra_axes_metadata = copy(waves.extra_axes_metadata)

        sampling = ()
        n = len(extra_axes_metadata) - 1
        for i, axes_metadata in enumerate(extra_axes_metadata[::-1]):
            if not isinstance(axes_metadata, ScanAxis):
                continue
            del extra_axes_metadata[n - i]
            sampling += (axes_metadata.sampling,)
            if len(sampling) == 2:
                break

        if len(sampling) == 1:
            sampling = sampling[0]

        measurement_type = self.measurement_type(waves)

        if measurement_type is LineProfiles:
            return {'sampling': sampling,
                    'start': waves.scan_axes_metadata[0].start,
                    'end': waves.scan_axes_metadata[0].end,
                    'extra_axes_metadata': extra_axes_metadata,
                    'metadata': waves.metadata}

        elif measurement_type is Images:
            return {'sampling': sampling,
                    'extra_axes_metadata': extra_axes_metadata,
                    'metadata': waves.metadata}
        else:
            raise RuntimeError()

    def detect(self, waves: 'Waves') -> Images:

        if self.outer is None:
            outer = np.floor(min(waves.cutoff_angles))
        else:
            outer = self.outer

        diffraction_patterns = waves.diffraction_patterns(max_angle='cutoff')
        measurement = diffraction_patterns.integrate_radial(inner=self.inner, outer=outer)

        if self.to_cpu:
            measurement = measurement.to_cpu()

        return measurement

    def show(self, ax=None):
        bins = np.arange(0, 2).reshape((2, 1))
        bins[1, 0] = -1

        vmin = -.5
        vmax = np.max(bins) + .5
        cmap = plt.get_cmap('tab20', np.nanmax(bins) + 1)
        cmap.set_under(color='white')

        polar_measurements = PolarMeasurements(bins,
                                               radial_sampling=self.outer - self.inner,
                                               azimuthal_sampling=2 * np.pi,
                                               radial_offset=self.inner,
                                               azimuthal_offset=0.)

        ax, im = polar_measurements.show(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, extend='min', label='Detector region')
        return ax, im

    def __copy__(self) -> 'AnnularDetector':
        return self.__class__(self.inner, self.outer)


class AbstractRadialDetector(AbstractDetector):

    def __init__(self,
                 inner: float,
                 outer: float,
                 rotation: float,
                 offset: Tuple[float, float],
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):
        self._inner = inner
        self._outer = outer
        self._rotation = rotation
        self._offset = offset
        super().__init__(to_cpu=to_cpu, url=url, detect_every=detect_every)

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

    def measurement_shape(self, waves):

        shape = self._calculate_nbins_radial(waves), self._calculate_nbins_azimuthal(waves)

        shape = waves.shape[:-len(waves.base_axes)] + shape

        return shape

    def measurement_kwargs(self, waves):
        extra_axes_metadata = waves.extra_axes_metadata

        d = {'radial_sampling': self.radial_sampling,
             'azimuthal_sampling': self.azimuthal_sampling,
             'radial_offset': self.inner,
             'azimuthal_offset': self._rotation,
             'extra_axes_metadata': extra_axes_metadata,
             }
        return d

    def angular_limits(self, waves) -> Tuple[float, float]:
        if self.inner is not None:
            inner = self.inner
        else:
            inner = 0.

        if self.outer is not None:
            outer = self.outer
        else:
            outer = np.floor(min(waves.cutoff_angles))

        return inner, outer

    def detect(self, waves):
        inner, outer = self.angular_limits(waves)

        measurement = waves.diffraction_patterns(max_angle='cutoff')

        measurement = measurement.polar_binning(nbins_radial=self._calculate_nbins_radial(waves),
                                                nbins_azimuthal=self._calculate_nbins_azimuthal(waves),
                                                inner=inner,
                                                outer=outer,
                                                rotation=self._rotation,
                                                offset=self._offset)

        if self.to_cpu:
            measurement = measurement.to_cpu()

        return measurement

    def measurement_type(self, waves):
        return PolarMeasurements

    def show(self, waves=None, ax=None, cmap='prism'):
        bins = np.arange(0, self._calculate_nbins_radial(waves) * self._calculate_nbins_azimuthal(waves))
        bins = bins.reshape((self._calculate_nbins_radial(waves), self._calculate_nbins_azimuthal(waves)))

        vmin = -.5
        vmax = np.max(bins) + .5
        cmap = plt.get_cmap(cmap, np.nanmax(bins) + 1)
        cmap.set_under(color='white')
        polar_measurements = PolarMeasurements(bins,
                                               radial_sampling=self.radial_sampling,
                                               azimuthal_sampling=self.azimuthal_sampling,
                                               radial_offset=self.inner,
                                               azimuthal_offset=self._rotation)

        ax, im = polar_measurements.show(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, extend='min', label='Detector region')
        # ax.set_rlim([-0, min(waves.cutoff_angles) * 1.1])
        return ax, im


class FlexibleAnnularDetector(AbstractRadialDetector):
    """
    Flexible annular detector.

    The FlexibleAnnularDetector allows choosing the integration limits after running the simulation by binning the
    intensity in annular integration regions.

    Parameters
    ----------
    step_size : float, optional
        Radial extent of the bins [mrad]. Default is 1.
    inner : float, optional
        Inner integration limit of the bins [mrad].
    outer : float, optional
        Outer integration limit of the bins [mrad].
    to_cpu : bool, optional
        If True, copy the measurement data after applying the detector from the calculation device to cpu memory,
        otherwise the data stays on the respective devices. Default is True.
    url : str, optional
        If this parameter is set the measurement data is saved at the specified location, typically a path to a
        local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
        the data stays in memory.
    detect_every : int, optional
        If this parameter is set the 'detect' method of the detector is applied after every 'detect_every' step of
        the multislice algorithm, in addition to after the last step.
        If not set (default), the detector is only applied after the last step.
    """

    def __init__(self,
                 step_size: float = 1.,
                 inner: float = 0.,
                 outer: float = None,
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):
        self._step_size = step_size
        super().__init__(inner=inner, outer=outer, rotation=0., offset=(0., 0.), to_cpu=to_cpu, url=url,
                         detect_every=detect_every)

    @property
    def step_size(self) -> float:
        """ Step size [mrad]. """
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

    def _calculate_nbins_radial(self, waves: 'Waves') -> int:
        return int(np.floor(min(waves.cutoff_angles)) / self.step_size)

    def _calculate_nbins_azimuthal(self, waves: 'Waves') -> int:
        return 1


class SegmentedDetector(AbstractRadialDetector):

    def __init__(self,
                 nbins_radial: int,
                 nbins_azimuthal: int,
                 inner: float,
                 outer: float,
                 rotation: float = 0.,
                 offset: Tuple[float, float] = (0., 0.),
                 to_cpu: bool = False,
                 url: str = None,
                 detect_every: int = None):
        """
        Segmented detector.

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
            Offset of the bins from the origin in x and y [mrad].
        to_cpu : bool, optional
            If True, copy the measurement data after applying the detector from the calculation device to cpu memory,
            otherwise the data stays on the respective devices. Default is True.
        url : str, optional
            If this parameter is set the measurement data is saved at the specified location, typically a path to a
            local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
            the data stays in memory.
        detect_every : int, optional
            If this parameter is set the 'detect' method of the detector is applied after every 'detect_every' step of
            the multislice algorithm, in addition to after the last step.
            If not set (default), the detector is only applied after the last step.
        """

        self._nbins_radial = nbins_radial
        self._nbins_azimuthal = nbins_azimuthal
        super().__init__(inner=inner, outer=outer, rotation=rotation, offset=offset, to_cpu=to_cpu, url=url,
                         detect_every=detect_every)

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

    def _calculate_nbins_radial(self, waves=None):
        return self.nbins_radial

    def _calculate_nbins_azimuthal(self, waves=None):
        return self.nbins_azimuthal


class PixelatedDetector(AbstractDetector):

    def __init__(self,
                 max_angle: Union[str, float] = 'valid',
                 resample: bool = False,
                 fourier_space: bool = True,
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):
        """
        Pixelated detector.

        The pixelated detector records the intensity of the Fourier-transformed exit wave function. This may be used for
        example for simulating 4D-STEM.

        Parameters
        ----------
        max_angle : str or float or None
            The diffraction patterns will be detected up to this angle. If set to a string it must be 'limit' or 'valid'
        resample : 'uniform' or False
            If 'uniform', the diffraction patterns from rectangular cells will be downsampled to a uniform angular sampling.
        max_angle : {'cutoff', 'valid'} or float
            Maximum detected scattering angle of the diffraction patterns. If str, it must be one of:
            ``cutoff`` :
            The maximum scattering angle will be the cutoff of the antialias aperture.
            ``valid`` :
            The maximum scattering angle will be the largest rectangle that fits inside the circular antialias aperture
            (default).

        to_cpu : bool, optional
            If True, copy the measurement data after applying the detector from the calculation device to cpu memory,
            otherwise the data stays on the respective devices. Default is True.
        url : str, optional
            If this parameter is set the measurement data is saved at the specified location, typically a path to a
            local file. A URL can also include a protocol specifier like s3:// for remote data. If not set (default)
            the data stays in memory.
        detect_every : int, optional
            If this parameter is set the 'detect' method of the detector is applied after every 'detect_every' step of
            the multislice algorithm, in addition to after the last step.
            If not set (default), the detector is only applied after the last step.
        """
        self._resample = resample
        self._max_angle = max_angle
        self._fourier_space = fourier_space
        super().__init__(to_cpu=to_cpu, url=url, detect_every=detect_every)

    @property
    def max_angle(self):
        return self._max_angle

    @property
    def fourier_space(self):
        return self._fourier_space

    @property
    def resample(self):
        return self._resample

    def angular_limits(self, waves) -> Tuple[float, float]:

        if isinstance(self.max_angle, str):
            if self.max_angle == 'valid':
                outer = min(waves.rectangle_cutoff_angles)
            elif self.max_angle == 'cutoff':
                outer = min(waves.cutoff_angles)
            elif self.max_angle == 'full':
                outer = min(waves.full_cutoff_angles)
            else:
                raise RuntimeError()
        else:
            outer = min(waves.cutoff_angles)

        return 0., outer

    def measurement_shape(self, waves):
        if self.fourier_space:
            shape = waves._gpts_within_angle(self.max_angle)
        else:
            shape = waves.gpts

        shape = waves.extra_axes_shape + shape
        return shape

    @property
    def measurement_dtype(self):
        return np.float32

    def measurement_kwargs(self, waves):
        extra_axes_metadata = deepcopy(waves.extra_axes_metadata)

        if self.fourier_space:
            return {'sampling': waves.fourier_space_sampling,
                    'fftshift': True,
                    'extra_axes_metadata': extra_axes_metadata,
                    'metadata': waves.metadata}
        else:
            return {'sampling': waves.sampling,
                    'extra_axes_metadata': extra_axes_metadata,
                    'metadata': waves.metadata}

    def measurement_type(self, waves, scan=None):
        if self.fourier_space:
            return DiffractionPatterns
        else:
            return Images

    def detect(self, waves) -> np.ndarray:
        # """
        # Calculate the far field intensity of the wave functions. The output is cropped to include the non-suppressed
        # frequencies from the antialiased 2D fourier spectrum.
        #
        # Parameters
        # ----------
        # waves: Waves object
        #     The batch of wave functions to detect.
        #
        # Returns
        # -------
        #     Detected values. The first dimension indexes the batch size, the second and third indexes the two components
        #     of the spatial frequency.
        # """

        if self.fourier_space:
            measurements = waves.diffraction_patterns(max_angle=self.max_angle)

        else:
            measurements = waves.intensity()

        if self.to_cpu:
            measurements = measurements.to_cpu()

        return measurements


class WavesDetector(AbstractDetector):

    def __init__(self, to_cpu: bool = False, url: str = None, detect_every: int = None):
        super().__init__(to_cpu=to_cpu, url=url, detect_every=detect_every)

    def detect(self, waves):
        if self.to_cpu:
            waves = waves.copy(device='cpu')
        return waves

    def angular_limits(self, waves) -> Tuple[float, float]:
        return 0., min(waves.full_cutoff_angles)

    @property
    def measurement_dtype(self):
        return np.complex64

    def measurement_shape(self, waves) -> Tuple:
        shape = waves.extra_axes_shape + waves.gpts
        return shape

    def measurement_type(self, waves):
        from abtem.waves import Waves
        return Waves

    def measurement_kwargs(self, waves):
        extra_axes_metadata = waves.extra_axes_metadata
        return {'sampling': waves.sampling,
                'energy': waves.energy,
                'antialias_cutoff_gpts': waves.antialias_cutoff_gpts,
                'tilt': waves.tilt,
                'extra_axes_metadata': deepcopy(extra_axes_metadata),
                'metadata': waves.metadata}
