"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy
from typing import Tuple, Any, Union, List

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from abtem.core.axes import ScanAxis, AxisMetadata, FrozenPhononsAxis, OrdinalAxis
from abtem.core.backend import get_array_module
from abtem.core.dask import ComputableList
from abtem.measure.measure import DiffractionPatterns, PolarMeasurements, Images, LineProfiles


def stack_waves(waves, axes_metadata):
    if len(waves) == 0:
        return waves[0]
    array = np.stack([waves.array for waves in waves], axis=0)
    d = waves[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + waves[0].extra_axes_metadata
    return waves[0].__class__(**d)


def stack_measurements(measurements, axes_metadata):
    array = np.stack([measurement.array for measurement in measurements])
    cls = measurements[0].__class__
    d = measurements[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + d['extra_axes_metadata']
    return cls(**d)


def stack_measurement_ensembles(detectors, measurements):
    for i, (detector, output) in enumerate(zip(detectors, measurements)):
        if not detector.ensemble_mean or isinstance(output, list):
            measurements[i] = stack_measurements(output, axes_metadata=FrozenPhononsAxis())

            if detector.ensemble_mean:
                measurements[i] = measurements[i].mean(0)

        measurements[i] = measurements[i].squeeze()

    if len(measurements) == 1:
        return measurements[0]
    else:
        return ComputableList(measurements)


def allocate_measurements(waves, scan, detectors, potential):
    measurements = []
    for detector in detectors:

        extra_axes_shape = ()
        extra_axes_metadata = []

        if potential.num_configurations > 1 and not detector.ensemble_mean:
            extra_axes_shape = (potential.num_configurations,)
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


def simple_repr(obj, keys):
    keys = ', '.join([f'{key}={getattr(obj, key)}' for key in keys])
    return f'{obj.__class__.__name__}({keys})'


def check_cutoff_angle(waves, angle):
    if (angle is not None) and (not isinstance(angle, str)):
        if (angle > waves.cutoff_angles[0]) or (angle > waves.cutoff_angles[1]):
            raise RuntimeError('Detector max angle exceeds the cutoff scattering angle.')


def validate_detectors(detectors):
    if hasattr(detectors, 'detect'):
        detectors = [detectors]
    elif detectors is None:
        detectors = [WavesDetector()]
    elif isinstance(detectors, (list, tuple)):
        pass
    else:
        raise RuntimeError()
    return detectors


class AbstractDetector(metaclass=ABCMeta):
    """Abstract base class for all detectors."""

    def __init__(self, ensemble_mean: bool = True, to_cpu: bool = True, url: str = None, detect_every: int = None):
        self._ensemble_mean = ensemble_mean
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
    def ensemble_mean(self):
        return self._ensemble_mean

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

    @property
    def url(self):
        return self._url

    def copy(self):
        """Make a copy."""
        return deepcopy(self)


class AnnularDetector(AbstractDetector):
    """
    Annular detector object.

    The annular detector integrates the intensity of the detected wave functions between an inner and outer integration
    limit.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float
        Outer integration limit [mrad].
    offset: two float, optional
        Center offset of integration region [mrad].
    path: str, optional
        The path to the file for saving the detector output.
    """

    def __init__(self,
                 inner: float,
                 outer: float,
                 offset: Tuple[float, float] = None,
                 ensemble_mean: bool = True,
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):
        self._inner = inner
        self._outer = outer
        self._offset = offset
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url, detect_every=detect_every)

    def stack_measurements(self, measurements, axes_metadata):
        return stack_measurements(measurements, axes_metadata)

    def __repr__(self):
        return simple_repr(self, ('inner', 'outer', 'ensemble_mean', 'to_cpu', 'url'))

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

    def measurement_shape(self, waves) -> Tuple:
        return waves.extra_axes_shape

    @property
    def measurement_dtype(self) -> type:
        return np.float32

    def measurement_type(self, waves):
        num_scan_axes = waves.num_scan_axes

        if num_scan_axes == 1:
            return LineProfiles
        elif num_scan_axes == 2:
            return Images
        else:
            raise RuntimeError(f'no measurement type for AnnularDetector and Waves with {num_scan_axes} scan axes')

    def measurement_kwargs(self, waves):
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

    def detect(self, waves) -> Images:

        if self.outer is None:
            outer = np.floor(min(waves.cutoff_angles))
        else:
            outer = self.outer

        diffraction_patterns = waves.diffraction_patterns(max_angle='cutoff')
        measurement = diffraction_patterns.integrate_radial(inner=self.inner, outer=outer)

        if self._to_cpu:
            measurement = measurement.to_cpu()

        # if probes.num_ensemble_axes > 0 and self.ensemble_mean:
        #    measurement = measurement.mean(probes.ensemble_axes)

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

    def __init__(self, inner, outer, rotation, ensemble_mean: bool = True, to_cpu: bool = True, url: str = None,
                 detect_every: int = None):
        self._inner = inner
        self._outer = outer
        self._rotation = rotation
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url, detect_every=detect_every)

    def stack_measurements(self, measurements, axes_metadata):
        return stack_measurements(measurements, axes_metadata)

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
                                                rotation=self._rotation)

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
    Flexible annular detector object.

    The FlexibleAnnularDetector object allows choosing the integration limits after running the simulation by radially
    binning the intensity.

    Parameters
    ----------
    step_size: float
        The radial separation between integration regions [mrad].
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self,
                 step_size: float = 1.,
                 inner: float = 0.,
                 outer: float = None,
                 ensemble_mean: bool = True,
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):
        self._step_size = step_size
        super().__init__(inner=inner, outer=outer, rotation=0., ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url,
                         detect_every=detect_every)

    def __repr__(self):
        return simple_repr(self, ('inner', 'outer', 'step_size', 'detect_every'))

    @property
    def step_size(self) -> float:
        """
        Step size [mrad].
        """
        return self._step_size

    @step_size.setter
    def step_size(self, value: float):
        self._step_size = value

    @property
    def radial_sampling(self):
        return self.step_size

    @property
    def azimuthal_sampling(self):
        return 2 * np.pi

    def _calculate_nbins_radial(self, waves) -> int:
        # if self.step_size is not None:
        #     # print()
        #     # if self.step_size <= min(waves.angular_sampling):
        #     #     raise RuntimeError(
        #     #         (f'step_size ({self.step_size} mrad) of FlexibleAnnularDetector smaller than simulated angular'
        #     #          f'sampling ({min(waves.angular_sampling)} mrad)'))
        #
        #     step_size = self.step_size
        # else:
        #     step_size = min(waves.angular_sampling)
        # print(waves.cutoff_angles)
        return int(np.floor(min(waves.cutoff_angles)) / self.step_size)

    def _calculate_nbins_azimuthal(self, waves):
        return 1


class SegmentedDetector(AbstractRadialDetector):
    """
    Segmented detector object.

    The segmented detector covers an annular angular range, and is partitioned into several integration regions divided
    to radial and angular segments. This can be used for simulating differential phase contrast (DPC) imaging.

    Parameters
    ----------
    inner: float
        Inner integration limit [mrad].
    outer: float
        Outer integration limit [mrad].
    nbins_radial: int
        Number of radial bins.
    nbins_angular: int
        Number of angular bins.
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self,
                 inner: float,
                 outer: float,
                 nbins_radial: int,
                 nbins_azimuthal: int,
                 rotation: float = 0.,
                 ensemble_mean: bool = True,
                 to_cpu: bool = False,
                 url: str = None,
                 detect_every: int = None):
        self._nbins_radial = nbins_radial
        self._nbins_azimuthal = nbins_azimuthal
        super().__init__(inner=inner, outer=outer, rotation=rotation, ensemble_mean=ensemble_mean, to_cpu=to_cpu,
                         url=url, detect_every=detect_every)

    def __repr__(self):
        return simple_repr(self, ('inner', 'outer', 'nbins_radial', 'nbins_azimuthal', 'rotation'))

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
    """
    Pixelated detector object.

    The pixelated detector records the intensity of the Fourier-transformed exit wavefunction. This may be used for
    example for simulating 4D-STEM.

    Parameters
    ----------
    max_angle : str or float or None
        The diffraction patterns will be detected up to this angle. If set to a string it must be 'limit' or 'valid'
    resample : 'uniform' or False
        If 'uniform', the diffraction patterns from rectangular cells will be downsampled to a uniform angular sampling.
    mode : 'intensity' or 'complex'
    url : str
        The path to the file used for saving the detector output.
    """

    def __init__(self,
                 max_angle: Union[str, float] = 'valid',
                 resample: bool = False,
                 ensemble_mean: bool = True,
                 fourier_space: bool = True,
                 to_cpu: bool = True,
                 url: str = None,
                 detect_every: int = None):

        self._resample = resample
        self._max_angle = max_angle
        self._fourier_space = fourier_space
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url, detect_every=detect_every)

    def stack_measurements(self, measurements, axes_metadata):
        return stack_measurements(measurements, axes_metadata)

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
        super().__init__(ensemble_mean=False, to_cpu=to_cpu, url=url, detect_every=detect_every)

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

    def stack_measurements(self, measurements, axes_metadata):
        return stack_waves(measurements, axes_metadata)
