"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Tuple, Any, Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from abtem.core.axes import ScanAxis
from abtem.core.backend import get_array_module
from abtem.measure.measure import DiffractionPatterns, PolarMeasurements, Images, LineProfiles
from typing import Iterable


def check_cutoff_angle(waves, angle):
    if (angle is not None) and (not isinstance(angle, str)):
        if (angle > waves.cutoff_angles[0]) or (angle > waves.cutoff_angles[1]):
            raise RuntimeError('Detector max angle exceeds the cutoff scattering angle.')


def validate_detectors(detectors):
    if isinstance(detectors, AbstractDetector):
        detectors = [detectors]
    elif detectors is None:
        detectors = [WavesDetector()]
    elif isinstance(detectors, (list, tuple)):
        pass
    else:
        raise RuntimeError()
    return detectors


def apply_detector_func(self, func, detectors, scan=None, **kwargs):
    detectors = validate_detectors(detectors)

    new_cls = [detector.measurement_type(self, scan) for detector in detectors]
    new_cls_kwargs = [detector.measurement_kwargs(self, scan) for detector in detectors]

    signatures = []
    output_sizes = {}
    meta = []
    i = 2
    for detector in detectors:
        shape = detector.measurement_shape(self)
        signatures.append(f'({",".join([str(i) for i in range(i, i + len(shape))])})')
        output_sizes.update({str(index): n for index, n in zip(range(i, i + len(shape)), shape)})
        meta.append(np.array((), dtype=detector.measurement_dtype))
        i += len(shape)

    signature = '(0,1)->' + ','.join(signatures)

    measurements = self.apply_gufunc(func,
                                     detectors=detectors,
                                     new_cls=new_cls,
                                     new_cls_kwargs=new_cls_kwargs,
                                     signature=signature,
                                     output_sizes=output_sizes,
                                     meta=meta,
                                     **kwargs)

    return measurements


class AbstractDetector(metaclass=ABCMeta):
    """Abstract base class for all detectors."""

    def __init__(self, ensemble_mean=True, to_cpu=True, url: str = None):
        self._ensemble_mean = ensemble_mean
        self._to_cpu = to_cpu
        self._url = url

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    def to_cpu(self):
        return self._to_cpu

    def allocate_measurement(self, waves, scan=None):
        d = self.measurement_kwargs(waves, scan)
        d['array'] = np.zeros(self.measurement_shape(waves, scan), dtype=self.measurement_dtype)
        return self.measurement_type(waves, scan)(**d)

    @property
    @abstractmethod
    def measurement_dtype(self):
        pass

    @abstractmethod
    def measurement_shape(self, waves, scan) -> Tuple:
        pass

    @abstractmethod
    def measurement_type(self, waves, scan):
        pass

    @abstractmethod
    def measurement_kwargs(self, waves, scan):
        pass

    @abstractmethod
    def detect(self, waves) -> Any:
        pass

    @property
    def url(self):
        return self._url

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """Make a copy."""
        return copy(self)


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
                 url: str = None):

        self._inner = inner
        self._outer = outer
        self._offset = offset
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)

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

    def measurement_shape(self, waves, scan=None) -> Tuple:
        shape = waves.extra_axes_shape
        if scan is not None:
            shape = scan.gpts + shape
        return shape

    @property
    def measurement_dtype(self) -> type:
        return np.float32

    def measurement_type(self, waves, scan=None):
        num_scan_axes = waves.num_scan_axes
        if scan is not None:
            num_scan_axes += len(scan.shape)

        if num_scan_axes == 1:
            return LineProfiles
        elif num_scan_axes == 2:
            return Images
        else:
            raise RuntimeError(f'no measurement type for AnnularDetector and Waves with {num_scan_axes} scan axes')

    def measurement_kwargs(self, waves, scan=None):
        extra_axes_metadata = copy(waves.extra_axes_metadata)
        if scan is not None:
            extra_axes_metadata = scan.axes_metadata + extra_axes_metadata

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

        measurement_type = self.measurement_type(waves, scan)
        if measurement_type is LineProfiles:

            if scan is not None:
                start = scan.start
                end = scan.end
            else:
                start = waves.scan_axes_metadata[0].start
                end = waves.scan_axes_metadata[0].end

            return {'sampling': sampling,
                    'start': start,
                    'end': end,
                    'extra_axes_metadata': extra_axes_metadata,
                    'metadata': waves.metadata}
        elif measurement_type is Images:

            return {'sampling': sampling,
                    'extra_axes_metadata': extra_axes_metadata,
                    'metadata': waves.metadata}
        else:
            raise RuntimeError()

    def create_measurement(self, array=None, scan=None, waves=None, extra_axes_metadata=None) \
            -> Union[LineProfiles, Images]:

        if extra_axes_metadata is None:
            extra_axes_metadata = []

        if array is None:
            array = np.zeros(scan.shape + self.detected_shape(waves), dtype=np.float32)

        if len(scan.shape) == 1:
            sampling = scan.axes_metadata[0]['sampling']
            measurement = LineProfiles(array, sampling=sampling, axes_metadata=extra_axes_metadata)
        elif len(scan.shape) == 2:
            sampling = (scan.axes_metadata[0]['sampling'], scan.axes_metadata[1]['sampling'])
            measurement = Images(array, sampling=sampling, extra_axes_metadata=extra_axes_metadata)
        else:
            raise NotImplementedError

        if self._url is not None:
            return measurement.to_zarr(self.url, overwrite=True, compute=False)
        else:
            return measurement

    def detect(self, probes) -> Images:
        measurement = probes.diffraction_patterns().integrate_radial(inner=self.inner, outer=self.outer)

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


class FlexibleAnnularDetector(AbstractDetector):
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

    def __init__(self, step_size: float = 1., ensemble_mean: bool = True, to_cpu: bool = True, url: str = None):
        self.step_size = step_size
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)

    @property
    def step_size(self) -> float:
        """
        Step size [mrad].
        """
        return self._radial_steps

    @step_size.setter
    def step_size(self, value: float):
        self._radial_steps = value

    def nbins_radial(self, waves) -> int:
        return int(np.floor(min(waves.cutoff_angles)) / (self._radial_steps))

    def detected_shape(self, waves) -> Tuple[int, int]:
        return self.nbins_radial(waves), 1

    @property
    def detected_dtype(self):
        return np.float32

    def measurement_from_array(self, array, scan=None, waves=None, axes_metadata=None):

        if axes_metadata is None:
            axes_metadata = []

        if hasattr(waves, 'ensemble_axes'):
            axes_metadata += waves.ensemble_axes

        axes_metadata += scan.axes_metadata

        return PolarMeasurements(array,
                                 radial_sampling=self.step_size,
                                 azimuthal_sampling=2 * np.pi,
                                 radial_offset=0.,
                                 azimuthal_offset=0.,
                                 axes_metadata=axes_metadata)

    def detect(self, waves) -> np.ndarray:
        """
        Integrate the intensity of a the wave functions over the detector range.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
        2d array
            Detected values. The array has shape of (batch size, number of bins).
        """

        diffraction_patterns = waves.diffraction_patterns(max_angle='cutoff')

        measurements = diffraction_patterns.polar_binning(inner=0., outer=np.floor(min(waves.cutoff_angles)),
                                                          nbins_radial=self.nbins_radial(waves), nbins_azimuthal=1)

        if (waves.num_ensemble_axes > 0) and self.ensemble_mean:
            measurements = measurements.mean(measurements.ensemble_axes)

        return measurements

    def show(self, waves, ax=None):
        bins = np.arange(0., self.nbins_radial(waves))[:, None]

        cmap = plt.get_cmap('prism', np.nanmax(bins) + 1)
        cmap.set_under(color='white')

        polar_measurements = PolarMeasurements(bins,
                                               radial_sampling=self._radial_steps,
                                               azimuthal_sampling=2 * np.pi,
                                               radial_offset=0,
                                               azimuthal_offset=0.)

        ax, im = polar_measurements.show(ax=ax, cmap=cmap, vmin=-.5, vmax=np.max(bins) + .5)

        plt.colorbar(im, extend='min', label='Detector region')
        ax.set_rlim([-0, min(waves.cutoff_angles) * 1.1])

        return ax, im

    def __copy__(self) -> 'FlexibleAnnularDetector':
        return self.__class__(self.step_size, ensemble_mean=self.ensemble_mean)

    def copy(self) -> 'FlexibleAnnularDetector':
        """
        Make a copy.
        """
        return copy(self)


class SegmentedDetector(AbstractDetector):
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

    def __init__(self, inner: float, outer: float, nbins_radial: int, nbins_azimuthal: int, rotation: float = 0.,
                 ensemble_mean: bool = True, to_cpu: bool = False, url: str = None):
        self._inner = inner
        self._outer = outer
        self._nbins_radial = nbins_radial
        self._nbins_azimuthal = nbins_azimuthal
        self._rotation = rotation
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)

    @property
    def inner(self) -> float:
        """Inner integration limit [mrad]."""
        return self._inner

    @inner.setter
    def inner(self, value: float):
        self._inner = value

    @property
    def radial_sampling(self):
        return (self.outer - self.inner) / self.nbins_radial

    @property
    def azimuthal_sampling(self):
        return 2 * np.pi / self.nbins_azimuthal

    @property
    def outer(self) -> float:
        """Outer integration limit [mrad]."""
        return self._outer

    @outer.setter
    def outer(self, value: float):
        self._outer = value

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

    def detected_shape(self, waves):
        return (self.nbins_radial, self.nbins_azimuthal)

    @property
    def detected_dtype(self):
        return np.float32

    def measurement_from_array(self, array, scan=None, waves=None, axes_metadata=None):

        if axes_metadata is None:
            axes_metadata = []

        if hasattr(waves, 'ensemble_axes'):
            axes_metadata += waves.ensemble_axes

        axes_metadata += scan.axes_metadata

        return PolarMeasurements(array,
                                 radial_sampling=self.radial_sampling,
                                 azimuthal_sampling=self.azimuthal_sampling,
                                 radial_offset=self.inner,
                                 azimuthal_offset=self.outer,
                                 axes_metadata=axes_metadata)

    def detect(self, probes):
        measurement = probes.diffraction_patterns().polar_binning(nbins_radial=self.nbins_radial,
                                                                  nbins_azimuthal=self.nbins_azimuthal,
                                                                  inner=self.inner, outer=self.outer,
                                                                  rotation=self._rotation)

        if probes.num_ensemble_axes > 0:
            measurement = measurement.mean(probes.ensemble_axes)

        return measurement

    def show(self, ax=None):
        bins = np.arange(0, self.nbins_radial * self.nbins_azimuthal).reshape((self.nbins_radial, self.nbins_azimuthal))

        vmin = -.5
        vmax = np.max(bins) + .5
        cmap = plt.get_cmap('tab20', np.nanmax(bins) + 1)
        cmap.set_under(color='white')
        polar_measurements = PolarMeasurements(bins,
                                               radial_sampling=self.radial_sampling,
                                               azimuthal_sampling=self.azimuthal_sampling,
                                               radial_offset=self.inner,
                                               azimuthal_offset=self._rotation)

        ax, im = polar_measurements.show(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, extend='min', label='Detector region')
        return ax, im

    def __copy__(self) -> 'SegmentedDetector':
        return self.__class__(inner=self.inner, outer=self.outer, nbins_radial=self.nbins_radial,
                              nbins_azimuthal=self.nbins_azimuthal, ensemble_mean=self.ensemble_mean)

    def copy(self) -> 'SegmentedDetector':
        """Make a copy."""
        return copy(self)


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
                 url: str = None):

        self._resample = resample
        self._max_angle = max_angle
        self._fourier_space = fourier_space
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)

    @property
    def max_angle(self):
        return self._max_angle

    @property
    def fourier_space(self):
        return self._fourier_space

    @property
    def resample(self):
        return self._resample

    def measurement_shape(self, waves, scan=None):
        if self.fourier_space:
            shape = waves._gpts_within_angle(self.max_angle)
        else:
            shape = waves.gpts

        shape = waves.extra_axes_shape + shape
        if scan is not None:
            shape = scan.shape + shape
        return shape

    @property
    def measurement_dtype(self):
        return np.float32

    def measurement_kwargs(self, waves, scan=None):
        extra_axes_metadata = waves.extra_axes_metadata
        if scan is not None:
            extra_axes_metadata = scan.axes_metadata + extra_axes_metadata
        if self.fourier_space:
            return {'angular_sampling': waves.angular_sampling,
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

    def _bilinear_nodes_and_weight(self, old_shape, new_shape, old_angular_sampling, new_angular_sampling, xp):
        nodes = []
        weights = []

        old_sampling = (1 / old_angular_sampling[0] / old_shape[0],
                        1 / old_angular_sampling[1] / old_shape[1])

        new_sampling = (1 / new_angular_sampling[0] / new_shape[0],
                        1 / new_angular_sampling[1] / new_shape[1])

        for n, m, r, d in zip(old_shape, new_shape, old_sampling, new_sampling):
            k = xp.fft.fftshift(xp.fft.fftfreq(n, r).astype(xp.float32))
            k_new = xp.fft.fftshift(xp.fft.fftfreq(m, d).astype(xp.float32))

            distances = k_new[None] - k[:, None]
            distances[distances < 0.] = np.inf

            w = distances.min(0) / (k[1] - k[0])
            w[w == np.inf] = 0.

            nodes.append(distances.argmin(0))
            weights.append(w)

        v, u = nodes
        vw, uw = weights
        v, u, vw, uw = xp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])
        return v, u, vw, uw

    def _resampled_gpts(self, gpts, angular_sampling):
        if self._resample is False:
            return gpts, angular_sampling

        if self._resample == 'uniform':
            scale_factor = (angular_sampling[0] / max(angular_sampling),
                            angular_sampling[1] / max(angular_sampling))

            new_gpts = (int(np.ceil(gpts[0] * scale_factor[0])),
                        int(np.ceil(gpts[1] * scale_factor[1])))

            if np.abs(new_gpts[0] - new_gpts[1]) <= 2:
                new_gpts = (min(new_gpts),) * 2

            new_angular_sampling = (angular_sampling[0] / scale_factor[0],
                                    angular_sampling[1] / scale_factor[1])

        else:
            raise RuntimeError('')

        return new_gpts, new_angular_sampling

    def _interpolate(self, array, angular_sampling):
        xp = get_array_module(array)
        # interpolate_bilinear = get_device_function(xp, 'interpolate_bilinear')

        new_gpts, new_angular_sampling = self._resampled_gpts(array.shape[-2:], angular_sampling)
        v, u, vw, uw = self._bilinear_nodes_and_weight(array.shape[-2:],
                                                       new_gpts,
                                                       angular_sampling,
                                                       new_angular_sampling,
                                                       xp)

        # return interpolate_bilinear(array, v, u, vw, uw)

    def measurement_from_array(self, array, scan=None, waves=None, extra_axes_metadata=None, **kwargs):

        if extra_axes_metadata is None:
            extra_axes_metadata = []

        extra_axes_metadata = extra_axes_metadata + scan.axes_metadata

        return DiffractionPatterns(array, angular_sampling=waves.angular_sampling, axes_metadata=extra_axes_metadata,
                                   fftshift=True)

    def detect(self, waves) -> np.ndarray:

        if self.fourier_space:
            measurements = waves.diffraction_patterns(max_angle=self.max_angle)
        else:
            measurements = waves.intensity()

        # if (waves.num_ensemble_axes > 0) & self.ensemble_mean:
        #    measurements = measurements.mean(waves.ensemble_axes)

        if self.to_cpu:
            measurements = measurements.to_cpu()

        return measurements

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
        #
        # xp = get_array_module(waves.array)
        # abs2 = get_device_function(xp, 'abs2')
        #
        # waves = waves.far_field(max_angle=self.max_angle)
        #
        # if self._mode == 'intensity':
        #     array = abs2(waves.array)
        # elif self._mode == 'complex':
        #     array = waves.array
        # else:
        #     raise ValueError()
        #
        # array = xp.fft.fftshift(array, axes=(-2, -1))
        #
        # if self._resample:
        #     array = self._interpolate(array, waves.angular_sampling)
        # return array

    def __copy__(self) -> 'PixelatedDetector':
        return self.__class__(max_angle=self.max_angle,
                              resample=self.resample,
                              ensemble_mean=self.ensemble_mean,
                              to_cpu=self.to_cpu,
                              url=self.url)


class WavesDetector(AbstractDetector):

    def __init__(self, ensemble_mean: bool = True, to_cpu: bool = True, url: str = None):
        super().__init__(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)

    def detect(self, waves):
        images = waves.intensity()

        if self.to_cpu:
            images = images.to_cpu()

        return images

    @property
    def measurement_dtype(self):
        return np.float32

    def measurement_shape(self, waves, scan=None) -> Tuple:
        shape = waves.extra_axes_shape + waves.gpts
        if scan is not None:
            shape = scan.shape + shape
        return shape

    def measurement_type(self, waves, scan=None):
        return Images

    def measurement_kwargs(self, waves, scan=None):

        extra_axes_metadata = waves.extra_axes_metadata
        if scan is not None:
            extra_axes_metadata = scan.axes_metadata + extra_axes_metadata

        return {'sampling': waves.sampling,
                'extra_axes_metadata': extra_axes_metadata,
                'metadata': waves.metadata}

    def __copy__(self):
        pass
