"""Module for describing the detection of transmitted waves and different detector types."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Tuple, List, Any, Union

import matplotlib.pyplot as plt
import numpy as np

from abtem.device import get_array_module, get_device_function
from abtem.measure.measure import DiffractionPatterns, PolarMeasurements
from abtem.measure.old_measure import Calibration, calibrations_from_grid, Measurement
from abtem.measure.utils import polar_detector_bins
from abtem.visualize.mpl import show_measurement_2d
from abtem.waves.scan import AbstractScan


# if (nbins_radial == 1) & (nbins_azimuthal == 1):
#    return np.where(bins == 0)

# region_indices = []
# for indices in self._label_to_index(region_labels):
#     region_indices.append(indices)
# return region_indices
# return bins


def check_cutoff_angle(waves, angle):
    if (angle is not None) and (not isinstance(angle, str)):
        if (angle > waves.cutoff_angles[0]) or (angle > waves.cutoff_angles[1]):
            raise RuntimeError('Detector max angle exceeds the cutoff scattering angle.')


class AbstractDetector(metaclass=ABCMeta):
    """Abstract base class for all detectors."""

    def __init__(self, ensemble_mean=True):
        self._ensemble_mean = ensemble_mean

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @abstractmethod
    def detect(self, waves) -> Any:
        pass

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """Make a copy."""
        return copy(self)


class _PolarDetector(AbstractDetector):
    """Class to define a polar detector, forming the basis of annular and segmented detectors."""

    def __init__(self,
                 inner: float = None,
                 outer: float = None,
                 radial_steps: float = 1.,
                 azimuthal_steps: float = None,
                 offset: Tuple[float, float] = None,
                 rotation: float = 0.,
                 save_file: str = None):

        self._inner = inner
        self._outer = outer

        self._radial_steps = radial_steps

        if azimuthal_steps is None:
            azimuthal_steps = 2 * np.pi

        self._azimuthal_steps = azimuthal_steps

        self._rotation = rotation
        self._offset = offset

        super().__init__(max_detected_angle=outer, save_file=save_file)

    def _get_bins(self, cutoff_scattering_angle=None):
        if self._inner is None:
            inner = 0.
        else:
            inner = self._inner

        if self._outer is None:
            if cutoff_scattering_angle is None:
                raise RuntimeError('The outer integration angle is not set.')

            outer = cutoff_scattering_angle
            outer = np.floor(outer / self._radial_steps) * self._radial_steps
        else:
            outer = self._outer

        nbins_radial = int(np.ceil((outer - inner) / self._radial_steps))

        nbins_azimuthal = int(np.ceil(2 * np.pi / self._azimuthal_steps))

        return nbins_radial, nbins_azimuthal, inner, outer

    def _get_regions(self,
                     gpts: Tuple[int, int],
                     angular_sampling: Tuple[float, float],
                     cutoff_scattering_angle: float = None,
                     xp=np) -> List[np.ndarray]:

        nbins_radial, nbins_azimuthal, inner, outer = self._get_bins(cutoff_scattering_angle)

        region_labels = polar_detector_regions(gpts,
                                               (angular_sampling[0] / 1e3, angular_sampling[1] / 1e3),
                                               inner / 1e3,
                                               outer / 1e3,
                                               nbins_radial,
                                               nbins_azimuthal,
                                               rotation=self._rotation)

        if self._offset is not None:
            offset = (int(round(self._offset[0] / angular_sampling[0])),
                      int(round(self._offset[1] / angular_sampling[1])))

            if (abs(offset[0]) > region_labels.shape[0]) or (abs(offset[1]) > region_labels.shape[1]):
                raise RuntimeError('Detector offset exceeds maximum detected angle.')

            region_labels = np.roll(region_labels, offset, (0, 1))

        region_labels = xp.asarray(region_labels)

        if np.all(region_labels == -1):
            raise RuntimeError('Zero-sized detector region.')

        region_indices = []
        for indices in self._label_to_index(region_labels):
            region_indices.append(indices)
        return region_indices

    def measurement_shape(self, waves):
        nbins_radial, nbins_azimuthal, inner, _ = self._get_bins(min(waves.cutoff_scattering_angles))

        shape = ()

        if nbins_radial > 1:
            shape += (nbins_radial,)

        if nbins_azimuthal > 1:
            shape += (nbins_azimuthal,)

        return shape

    def measurement_calibrations(self, waves):
        nbins_radial, nbins_azimuthal, inner, _ = self._get_bins(min(waves.cutoff_scattering_angles))

        calibrations = ()

        if nbins_radial > 1:
            calibrations += (Calibration(offset=inner, sampling=self._radial_steps, units='mrad'),)

        if nbins_azimuthal > 1:
            calibrations += (Calibration(offset=0, sampling=self._azimuthal_steps, units='rad'),)

        return calibrations

    def allocate_measurement(self, waves, scan: AbstractScan = None) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves object
            An example of the
        scan : Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        waves.accelerator.check_is_defined()

        if scan is None:
            shape = ()
            calibrations = ()
        else:
            shape = scan.shape
            calibrations = scan.calibrations

        nbins_radial, nbins_azimuthal, inner, _ = self._get_bins(min(waves.cutoff_scattering_angles))

        if nbins_radial > 1:
            shape += (nbins_radial,)
            calibrations += (Calibration(offset=inner, sampling=self._radial_steps, units='mrad'),)

        if nbins_azimuthal > 1:
            shape += (nbins_azimuthal,)
            calibrations += (Calibration(offset=0, sampling=self._azimuthal_steps, units='rad'),)

        array = np.zeros(shape, dtype=np.float32)
        measurement = Measurement(array, calibrations=calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def show(self, waves, **kwargs):
        """
        Visualize the detector region(s) of the detector as applied to a specified wave function.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function the visualization will be created to match
        kwargs :
            Additional keyword arguments for abtem.visualize.mpl.show_measurement_2d.
        """

        waves.grid.check_is_defined()
        array = np.full(waves.gpts, -1, dtype=np.int)

        for i, indices in enumerate(self._get_regions(waves.gpts,
                                                      waves.angular_sampling,
                                                      min(waves.cutoff_scattering_angles))):
            array.ravel()[indices] = i

        calibrations = calibrations_from_grid(waves.gpts,
                                              waves.sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=waves.wavelength * 1e3,
                                              fourier_space=True)

        array = np.fft.fftshift(array, axes=(-1, -2))

        measurement = Measurement(array, calibrations=calibrations, name='Detector regions')

        return show_measurement_2d(measurement, discrete_cmap=True, **kwargs)


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

    def __init__(self, inner: float, outer: float, offset: Tuple[float, float] = None, ensemble_mean=True):
        self._inner = inner
        self._outer = outer
        self._offset = offset
        super().__init__(ensemble_mean=ensemble_mean)

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

    def show(self, probes, ax=None):
        bins = polar_detector_bins(gpts=probes.gpts,
                                   sampling=probes.angular_sampling,
                                   inner=self.inner,
                                   outer=self.outer,
                                   nbins_radial=1,
                                   nbins_azimuthal=1,
                                   fftshift=True)

        image = DiffractionPatterns(bins, angular_sampling=probes.angular_sampling,
                                    axes_metadata=probes._fourier_space_axes_metadata)
        return image.show(ax=ax)

    def detect(self, probes):
        measurement = probes.diffraction_patterns().integrate_annular_disc(inner=self.inner, outer=self.outer)

        if probes.num_ensemble_axes > 0 and self.ensemble_mean:
            measurement = measurement.mean(probes.ensemble_axes)

        return measurement

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

    def __init__(self, step_size: float = 1., ensemble_mean: bool = True):
        self.step_size = step_size
        super().__init__(ensemble_mean=ensemble_mean)

    @property
    def step_size(self) -> float:
        """
        Step size [mrad].
        """
        return self._radial_steps

    @step_size.setter
    def step_size(self, value: float):
        self._radial_steps = value

    def nbins_radial(self, waves):
        return int(min(waves.cutoff_angles) / self._radial_steps)

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

        measurements = waves.diffraction_patterns(max_angle='cutoff').polar_binning(inner=0.,
                                                                                    outer=min(waves.cutoff_angles),
                                                                                    nbins_radial=self.nbins_radial(
                                                                                        waves),
                                                                                    nbins_azimuthal=1)

        if waves.num_ensemble_axes > 0:
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
        return self.__class__(self.step_size, url=self.url, ensemble_mean=self.ensemble_mean)

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
                 ensemble_mean: bool = True):
        self._inner = inner
        self._outer = outer
        self._nbins_radial = nbins_radial
        self._nbins_azimuthal = nbins_azimuthal
        self._rotation = rotation
        super().__init__(ensemble_mean=ensemble_mean)

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

        radial_sampling = (self.outer - self.inner) / self.nbins_radial
        azimuthal_sampling = 2 * np.pi / self.nbins_azimuthal

        vmin = -.5
        vmax = np.max(bins) + .5
        cmap = plt.get_cmap('tab20', np.nanmax(bins) + 1)
        cmap.set_under(color='white')
        polar_measurements = PolarMeasurements(bins,
                                               radial_sampling=radial_sampling,
                                               azimuthal_sampling=azimuthal_sampling,
                                               radial_offset=self.inner,
                                               azimuthal_offset=self._rotation)

        ax, im = polar_measurements.show(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, extend='min', label='Detector region')
        return ax, im

    def __copy__(self) -> 'SegmentedDetector':
        return self.__class__(inner=self.inner, outer=self.outer, nbins_radial=self.nbins_radial,
                              nbins_azimuthal=self.nbins_azimuthal, url=self.url, ensemble_mean=self.ensemble_mean)

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
    save_file : str
        The path to the file used for saving the detector output.
    """

    def __init__(self,
                 max_angle: Union[str, float] = 'valid',
                 resample: Union[str, float] = False,
                 mode='intensity',
                 ensemble_mean: bool = True):

        self._max_angle = max_angle
        self._resample = resample
        self._mode = mode

        super().__init__(ensemble_mean=ensemble_mean)

    @property
    def max_angle(self):
        return self._max_angle

    @property
    def resample(self):
        return self._resample

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
        interpolate_bilinear = get_device_function(xp, 'interpolate_bilinear')

        new_gpts, new_angular_sampling = self._resampled_gpts(array.shape[-2:], angular_sampling)
        v, u, vw, uw = self._bilinear_nodes_and_weight(array.shape[-2:],
                                                       new_gpts,
                                                       angular_sampling,
                                                       new_angular_sampling,
                                                       xp)

        return interpolate_bilinear(array, v, u, vw, uw)

    def detect(self, waves) -> np.ndarray:
        measurements = waves.diffraction_patterns(max_angle=self.max_angle)

        if waves.num_ensemble_axes > 0:
            measurements = measurements.mean(waves.ensemble_axes)

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

    def __copy__(self) -> 'SegmentedDetector':
        return self.__class__(inner=self.inner, outer=self.outer, nbins_radial=self.nbins_radial,
                              nbins_azimuthal=self.nbins_azimuthal, url=self.url, ensemble_mean=self.ensemble_mean)

    def copy(self) -> 'SegmentedDetector':
        """Make a copy."""
        return copy(self)


class WavefunctionDetector(AbstractDetector):
    """
    Wave function detector object

    The wave function detector records the raw exit wave functions.

    Parameters
    ----------
    save_file: str
        The path to the file used for saving the detector output.
    """

    def __init__(self, save_file: str = None):
        super().__init__(max_detected_angle=None, save_file=save_file)

    def allocate_measurement(self, waves, scan: AbstractScan) -> Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function that will define the shape of the diffraction patterns.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        calibrations = calibrations_from_grid(waves.gpts, waves.sampling, names=['x', 'y'], units='Å')

        array = np.zeros(scan.shape + waves.gpts, dtype=np.complex64)
        measurement = Measurement(array, calibrations=scan.calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves) -> np.ndarray:
        """
        Detect the complex wave function.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
        3d complex array
            The arrays of the Waves object.
        """
        return waves.array
