"""Module for handling measurements."""
import copy
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple, TypeVar, Dict, List, Sequence, Type, TYPE_CHECKING

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from ase import Atom
from ase.cell import Cell
from matplotlib.axes import Axes
from numba import prange, jit

from abtem.core.array import HasArray
from abtem.core.axes import (
    HasAxes,
    RealSpaceAxis,
    AxisMetadata,
    FourierSpaceAxis,
    LinearAxis,
    NonLinearAxis,
    SampleAxis,
    ScanAxis,
)
from abtem.core.backend import cp, get_array_module, get_ndimage_module
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength
from abtem.core.fft import fft_interpolate, fft_crop
from abtem.core.grid import adjusted_gpts, polar_spatial_frequencies, spatial_frequencies
from abtem.indexing import IndexedDiffractionPatterns
from abtem.core.interpolate import interpolate_bilinear
from abtem.core.utils import CopyMixin, EqualityMixin, label_to_index
from abtem.inelastic.phonons import _validate_seeds
from abtem.visualize import show_measurement_2d, show_measurements_1d, _make_cbar_label

# Enables CuPy-accelerated functions if it is available.
if cp is not None:
    from abtem.core.cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
    from abtem.core.cuda import interpolate_bilinear as interpolate_bilinear_cuda
else:
    sum_run_length_encoded_cuda = None
    interpolate_bilinear_cuda = None

# Avoids circular imports.
if TYPE_CHECKING:
    from abtem.waves import BaseWaves

# Ensures that `Measurement` objects created by `Measurement` objects retain their type (e.g. `Images`).
T = TypeVar("T", bound="BaseMeasurement")

# Options for displaying alternative units.
angular_units = ("angular", "mrad")
bins = ("bins",)
reciprocal_units = ("reciprocal", "1/Å")


def _to_hyperspy_axes_metadata(axes_metadata, shape):
    hyperspy_axes = []

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    for metadata, n in zip(axes_metadata, shape):
        hyperspy_axes.append({"size": n})

        axes_mapping = {
            "sampling": "scale",
            "units": "units",
            "label": "name",
            "offset": "offset",
        }

        if isinstance(metadata, NonLinearAxis):
            # TODO : when hyperspy supports arbitrary (non-uniform) DataAxis this should be updated

            if len(metadata.values) > 1:
                sampling = metadata.values[1] - metadata.values[0]
            else:
                sampling = 1.0

            metadata = LinearAxis(
                label=metadata.label,
                units=metadata.units,
                sampling=sampling,
                offset=metadata.values[0],
            )

        for attr, mapped_attr in axes_mapping.items():
            if hasattr(metadata, attr):
                hyperspy_axes[-1][mapped_attr] = getattr(metadata, attr)

    return hyperspy_axes


def _scanned_measurement_type(
    measurement: Union["BaseMeasurement", "BaseWaves"]
) -> Type["BaseMeasurement"]:
    if len(_scan_shape(measurement)) == 0:
        return _SinglePointMeasurement

    elif len(_scan_shape(measurement)) == 1:
        return RealSpaceLineProfiles

    elif len(_scan_shape(measurement)) == 2:
        return Images

    else:
        raise RuntimeError(
            f"no measurement type for {measurement.__class__} with {len(_scan_shape(measurement))} scan "
            f"axes"
        )


def _reduced_scanned_images_or_line_profiles(
    new_array,
    old_measurement,
    metadata=None,
) -> Union["RealSpaceLineProfiles", "Images", np.ndarray]:
    if metadata is None:
        metadata = {}

    metadata = {**old_measurement.metadata, **metadata}

    if _scanned_measurement_type(old_measurement) is RealSpaceLineProfiles:
        sampling = old_measurement.ensemble_axes_metadata[-1].sampling

        ensemble_axes_metadata = old_measurement.ensemble_axes_metadata[:-1]

        return RealSpaceLineProfiles(
            new_array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    elif _scanned_measurement_type(old_measurement) is Images:

        ensemble_axes_metadata = old_measurement.ensemble_axes_metadata[:-2]

        sampling = (
            old_measurement.ensemble_axes_metadata[-2].sampling,
            old_measurement.ensemble_axes_metadata[-1].sampling,
        )

        images = Images(
            new_array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

        return images
    else:
        return new_array


def _scan_axes(self):
    num_trailing_scan_axes = 0
    for axis in reversed(self.ensemble_axes_metadata):
        if not isinstance(axis, ScanAxis) or num_trailing_scan_axes == 2:
            break

        num_trailing_scan_axes += 1

    return tuple(
        range(
            len(self.ensemble_shape) - num_trailing_scan_axes,
            len(self.ensemble_shape),
        )
    )


def _scan_sampling(measurements):
    return tuple(
        measurements.axes_metadata[i].sampling for i in _scan_axes(measurements)
    )


def _scan_axes_metadata(measurements):
    return [measurements.axes_metadata[i] for i in _scan_axes(measurements)]


def _scan_shape(measurements):
    return tuple(measurements.shape[i] for i in _scan_axes(measurements))


def _scan_area_per_pixel(measurements):
    if len(_scan_sampling(measurements)) == 2:
        return np.prod(_scan_sampling(measurements))
    else:
        raise RuntimeError("Cannot infer pixel area from axes metadata.")


def _scan_extent(measurement):
    extent = ()
    for n, metadata in zip(_scan_shape(measurement), _scan_axes_metadata(measurement)):
        extent += (metadata.sampling * n,)
    return extent


def _annular_detector_mask(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    inner: float,
    outer: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    fftshift: bool = False,
    xp=np,
) -> Union[np.ndarray, List[np.ndarray]]:

    kx, ky = spatial_frequencies(gpts, (1 / sampling[0] / gpts[0], 1 / sampling[1] / gpts[1]), False, xp)

    k2 = kx[:, None] ** 2 + ky[None] ** 2

    bins = (k2 >= inner ** 2) & (k2 < outer ** 2)

    if np.any(np.array(offset) != 0.0):
        offset = (
            int(round(offset[0] / sampling[0])),
            int(round(offset[1] / sampling[1])),
        )

        # if (abs(offset[0]) > bins[0]) or (abs(offset[1]) > bins[1]):
        #     raise RuntimeError("Detector offset exceeds maximum detected angle.")

        bins = np.roll(bins, offset, (0, 1))

    if fftshift:
        bins = xp.fft.fftshift(bins)

    return bins


def _polar_detector_bins(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    inner: float,
    outer: float,
    nbins_radial: int,
    nbins_azimuthal: int,
    rotation: float = 0.0,
    offset: Tuple[float, float] = (0.0, 0.0),
    fftshift: bool = False,
    return_indices: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    alpha, phi = polar_spatial_frequencies(
        gpts, (1 / sampling[0] / gpts[0], 1 / sampling[1] / gpts[1])
    )
    phi = (phi + rotation) % (2 * np.pi)

    radial_bins = -np.ones(gpts, dtype=int)
    valid = (alpha >= inner) & (alpha < outer)

    radial_bins[valid] = nbins_radial * (alpha[valid] - inner) / (outer - inner)

    angular_bins = np.floor(nbins_azimuthal * (phi / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_azimuthal - 1).astype(int)

    bins = -np.ones(gpts, dtype=int)
    bins[valid] = angular_bins[valid] + radial_bins[valid] * nbins_azimuthal

    if np.any(np.array(offset) != 0.0):
        offset = (
            int(round(offset[0] / sampling[0])),
            int(round(offset[1] / sampling[1])),
        )

        # if (abs(offset[0]) > bins[0]) or (abs(offset[1]) > bins[1]):
        #     raise RuntimeError("Detector offset exceeds maximum detected angle.")

        bins = np.roll(bins, offset, (0, 1))

    if fftshift:
        bins = np.fft.fftshift(bins)

    if return_indices:
        indices = []
        for i in label_to_index(bins, nbins_radial * nbins_azimuthal - 1):
            indices.append(i)
        return indices
    else:
        return bins


@jit(nopython=True, nogil=True, fastmath=True)
def _sum_run_length_encoded(array, result, separators):
    for x in prange(result.shape[1]):
        for i in range(result.shape[0]):
            for j in range(separators[x], separators[x + 1]):
                result[i, x] += array[i, j]


def _interpolate_stack(array, positions, mode, order, **kwargs):
    map_coordinates = get_ndimage_module(array).map_coordinates
    xp = get_array_module(array)

    positions_shape = positions.shape
    positions = positions.reshape((-1, 2))

    old_shape = array.shape
    array = array.reshape((-1,) + array.shape[-2:])
    array = xp.pad(array, ((0, 0), (2 * order,) * 2, (2 * order,) * 2), mode=mode)

    positions = positions + 2 * order
    output = xp.zeros((array.shape[0], positions.shape[0]), dtype=np.float32)

    for i in range(array.shape[0]):
        map_coordinates(array[i], positions.T, output=output[i], order=order, **kwargs)

    output = output.reshape(old_shape[:-2] + positions_shape[:-1])
    return output


class BaseMeasurement(HasArray, HasAxes, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    Parent class common to all measurement types.

    Parameters
    ----------
    array : ndarray
        Array containing data of type `float` or ´complex´.
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    allow_base_axis_chunks : bool
        Sets whether the measurement is allowed to be chunked along the base axis (e.g. `Images` are allowed, but
        `DiffractionPatterns` are not).
    """

    def __init__(
        self,
        array,
        ensemble_axes_metadata,
        metadata,
        allow_base_axis_chunks=False,
    ):

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._metadata = metadata

        self._array = array

        self._check_axes_metadata()

        if not allow_base_axis_chunks:
            if self.is_lazy and (
                not all(len(chunks) == 1 for chunks in array.chunks[-2:])
            ):
                raise RuntimeError(
                    f"Chunks not allowed in base axes of {self.__class__}."
                )

    # TODO: should be removed and the more basic version in superclass used instead.
    def iterate_ensemble(self, keep_dims: bool = False):
        for i in np.ndindex(*self.ensemble_shape):
            yield i, self.get_items(i, keep_dims=keep_dims)

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def energy(self):
        if not "energy" in self.metadata.keys():
            raise RuntimeError("Energy not in measurement metadata.")
        return self.metadata["energy"]

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    def scan_positions(self):
        positions = ()
        for n, metadata in zip(_scan_shape(self), _scan_axes_metadata(self)):
            positions += (
                np.linspace(
                    metadata.offset,
                    metadata.offset + metadata.sampling * n,
                    n,
                    endpoint=metadata.endpoint,
                ),
            )
        return positions

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list:
        pass

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def dimensions(self) -> int:
        return len(self._array.shape)

    def _check_is_complex(self):
        if not np.iscomplexobj(self.array):
            raise RuntimeError("Function not implemented for non-complex measurements.")

    def real(self) -> T:
        """Returns the real part of a complex-valued measurement."""
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).real)

    def imag(self) -> T:
        """Returns the imaginary part of a complex-valued measurement."""
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).imag)

    def phase(self) -> T:
        """Calculates the phase of a complex-valued measurement."""
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).angle)

    def abs(self) -> T:
        """Calculates the absolute value of a complex-valued measurement."""
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).abs)

    def intensity(self) -> T:
        """Calculates the squared norm of a complex-valued measurement."""
        self._check_is_complex()
        return self._apply_element_wise_func(abs2)

    def relative_difference(
        self, other: "BaseMeasurement", min_relative_tol: float = 0.0
    ):
        """
        Calculates the relative difference with respect to another compatible measurement.

        Parameters
        ----------
        other : BaseMeasurement
            Measurement to which the difference is calculated.
        min_relative_tol : float
            Avoids division by zero errors by defining a minimum value of the divisor in the relative difference.

        Returns
        -------
        difference : BaseMeasurement
            The relative difference as a measurement of the same type.
        """
        difference = self - other

        xp = get_array_module(self.array)

        valid = xp.abs(self.array) >= min_relative_tol * self.array.max()
        difference._array[valid] /= self.array[valid]
        difference._array[valid == 0] = np.nan
        difference._array *= 100.0

        difference.metadata["label"] = "Relative difference"
        difference.metadata["units"] = "%"

        return difference

    def power(self, number):
        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = self.array**number
        return self.__class__(**kwargs)

    @abstractmethod
    def from_array_and_metadata(
        self, array: np.ndarray, axes_metadata: List[AxisMetadata], metadata: dict
    ) -> "T":
        """Documented in the subclasses."""
        pass

    def reduce_ensemble(self) -> "T":
        """Takes the mean of an ensemble measurement (e.g. of frozen phonon configurations)."""
        axes = tuple(
            i
            for i, axis in enumerate(self.axes_metadata)
            if hasattr(axis, "_ensemble_mean") and axis._ensemble_mean
        )
        return self.mean(axes=axes)

    def _apply_element_wise_func(self, func: callable) -> "T":
        d = self._copy_kwargs(exclude=("array",))
        d["array"] = func(self.array)
        return self.__class__(**d)

    @property
    @abstractmethod
    def _area_per_pixel(self):
        pass

    @staticmethod
    def _add_poisson_noise(array, seed, total_dose, block_info=None):
        xp = get_array_module(array)

        if block_info is not None:
            chunk_index = np.ravel_multi_index(
                block_info[0]["chunk-location"], block_info[0]["num-chunks"]
            )
        else:
            chunk_index = 0

        rng = xp.random.default_rng(seed + chunk_index)
        randomized_seed = int(
            rng.integers(np.iinfo(np.int32).max)
        )  # fixes strange cupy bug

        rng = xp.random.RandomState(seed=randomized_seed)

        return rng.poisson(xp.clip(array, a_min=0.0, a_max=None) * total_dose).astype(
            xp.float32
        )

    def poisson_noise(
        self,
        dose_per_area: float = None,
        total_dose: float = None,
        samples: int = 1,
        seed: int = None,
    ):
        """
        Add Poisson noise (i.e. shot noise) to a measurement corresponding to the provided 'total_dose' (per measurement
        if applied to an ensemble) or 'dose_per_area' (not applicable for single measurements).

        Parameters
        ----------
        dose_per_area : float, optional
            The irradiation dose [electrons per Å:sup:`2`].
        total_dose : float, optional
            The irradiation dose per diffraction pattern.
        samples : int, optional
            The number of samples to draw from a Poisson distribution. If this is greater than 1, an additional
            ensemble axis will be added to the measurement.
        seed : int, optional
            Seed the random number generator.

        Returns
        -------
        noisy_measurement : BaseMeasurement
            The noisy measurement.
        """

        wrong_dose_error = RuntimeError(
            "Provide one of 'dose_per_area' or 'total_dose'."
        )

        if dose_per_area is not None:
            if total_dose is not None:
                raise wrong_dose_error

            total_dose = self._area_per_pixel * dose_per_area

        elif total_dose is not None:
            if dose_per_area is not None:
                raise wrong_dose_error

        else:
            raise wrong_dose_error

        xp = get_array_module(self.array)

        seeds = _validate_seeds(seed, samples)

        arrays = []
        for seed in seeds:
            if self.is_lazy:

                arrays.append(
                    self.array.map_blocks(
                        self._add_poisson_noise,
                        total_dose=total_dose,
                        seed=seed,
                        meta=xp.array((), dtype=xp.float32),
                    )
                )
            else:
                arrays.append(
                    self._add_poisson_noise(
                        self.array, total_dose=total_dose, seed=seed
                    )
                )

        if len(seeds) > 1:
            if self.is_lazy:
                arrays = da.stack(arrays)
            else:
                arrays = xp.stack(arrays)
            axes_metadata = [SampleAxis(label="sample")]
        else:
            arrays = arrays[0]
            axes_metadata = []

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = arrays
        kwargs["ensemble_axes_metadata"] = (
            axes_metadata + kwargs["ensemble_axes_metadata"]
        )
        return self.__class__(**kwargs)

    def to_hyperspy(self):
        """Convert measurement to a Hyperspy signal."""

        try:
            import hyperspy.api as hs
        except ImportError:
            raise ImportError(
                "This functionality of abTEM requires Hyperspy, see https://hyperspy.org/."
            )

        if self._base_dims == 1:
            signal_type = hs.signals.Signal1D
        elif self._base_dims == 2:
            signal_type = hs.signals.Signal2D
        else:
            raise RuntimeError()

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.ensemble_axes_metadata,
            self.ensemble_shape,
        )

        array = np.transpose(
            self.to_cpu().array, self.ensemble_axes[::-1] + self.base_axes[::-1]
        )

        s = signal_type(array, axes=axes_extra[::-1] + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s


class Images(BaseMeasurement):
    """
    A collection of 2D measurements such as HRTEM or STEM-ADF images. May be used to represent a reconstructed phase.

    Parameters
    ----------
    array : np.ndarray
        2D or greater array containing data of type `float` or ´complex´. The second-to-last and last
        dimensions are the image `y`- and `x`-axis, respectively.
    sampling : two float
        Lateral sampling of images in `x` and `y` [Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    """

    _base_dims = 2  # Images are assumed to be 2D

    def __init__(
        self,
        array: Union[da.core.Array, np.array],
        sampling: Union[float, Tuple[float, float]],
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: Dict = None,
    ):

        if np.isscalar(sampling):
            sampling = (float(sampling),) * 2
        else:
            sampling = float(sampling[0]), float(sampling[1])

        self._sampling = sampling

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
            allow_base_axis_chunks=True,
        )

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None) -> "Images":
        """
        Creates an image from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more 2D wave functions. The second-to-last and last dimensions are the
            `y`- and `x`-axis.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
            axes must be RealSpaceAxis.
        metadata : dict
            A dictionary defining the measurement metadata.

        Returns
        -------
        images : Images
            Images from the array and metadata.
        """
        real_space_axes = tuple(
            i for i, axis in enumerate(axes_metadata) if isinstance(axis, RealSpaceAxis)
        )

        if len(real_space_axes) < 2:
            raise RuntimeError()

        scan_axes_metadata = [axes_metadata[i] for i in real_space_axes[-2:]]

        other_axes_metadata = [
            axes_metadata[i]
            for i, metadata in enumerate(axes_metadata)
            if i not in real_space_axes[-2:]
        ]

        sampling = (scan_axes_metadata[-2].sampling, scan_axes_metadata[-1].sampling)

        return cls(
            array,
            sampling=sampling,
            ensemble_axes_metadata=other_axes_metadata,
            metadata=metadata,
        )

    @property
    def _area_per_pixel(self):
        return np.prod(self.sampling)

    @property
    def sampling(self) -> Tuple[float, float]:
        """Sampling of images in `x` and `y` [Å]."""
        return self._sampling

    @property
    def extent(self) -> Tuple[float, float]:
        """Extent of images in `x` and `y` [Å]."""
        return (
            self.sampling[0] * self.base_shape[0],
            self.sampling[1] * self.base_shape[1],
        )

    @property
    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Coordinates of pixels in `x` and `y` [Å]."""
        x = np.linspace(0.0, self.shape[-2] * self.sampling[0], self.shape[-2])
        y = np.linspace(0.0, self.shape[-1] * self.sampling[1], self.shape[-1])
        return x, y

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [
            RealSpaceAxis(label="x", sampling=self.sampling[0], units="Å"),
            RealSpaceAxis(label="y", sampling=self.sampling[1], units="Å"),
        ]

    def integrate_gradient(self):
        """
        Calculate integrated gradients. Requires complex images whose real and imaginary parts represent the `x` and `y`
        components of a gradient.

        Returns
        -------
        integrated_gradient : Images
            The integrated gradient.
        """
        self._check_is_complex()
        if self.is_lazy:
            xp = get_array_module(self.array)
            array = self.array.rechunk(
                self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],))
            )
            array = array.map_blocks(
                _integrate_gradient_2d,
                sampling=self.sampling,
                meta=xp.array((), dtype=np.float32),
            )
        else:
            array = _integrate_gradient_2d(self.array, sampling=self.sampling)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def crop(
        self, extent: Tuple[float, float], offset: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Crop images to a smaller extent.

        Parameters
        ----------
        extent : tuple of float
            Extent of rectangular cropping region in `x` and `y` [Å].
        offset : tuple of float
            Lower corner of cropping region in `x` and `y` [Å] (default is (0,0)).

        Returns
        -------
        cropped_images : Images
            The cropped images.
        """

        offset = (
            int(np.round(self.base_shape[0] * offset[0] / self.extent[0])),
            int(np.round(self.base_shape[1] * offset[1] / self.extent[1])),
        )
        new_shape = (
            int(np.round(self.base_shape[0] * extent[0] / self.extent[0])),
            int(np.round(self.base_shape[1] * extent[1] / self.extent[1])),
        )

        array = self.array[
            ...,
            offset[0] : offset[0] + new_shape[0],
            offset[1] : offset[1] + new_shape[1],
        ]

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def interpolate(
        self,
        sampling: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        method: str = "fft",
        boundary: str = "periodic",
        order: int = 3,
        normalization: str = "values",
        cval: float = 0.0,
    ) -> "Images":
        """
        Interpolate images producing equivalent images with a different sampling. Either 'sampling' or 'gpts' must be
        provided (but not both).

        Parameters
        ----------
        sampling : float or two float
            Sampling of images after interpolation in `x` and `y` [Å].
        gpts : int or two int
            Number of grid points of images after interpolation in `x` and `y`. Do not use if 'sampling' is used.
        method : {'fft', 'spline'}
            The interpolation method.

                ``fft`` :
                    Interpolate by cropping or zero-padding in reciprocal space. This method should be preferred for
                    periodic images.

                ``spline`` :
                    Interpolate using spline interpolation. This method should be preferred for non-periodic images.

        boundary : {'periodic', 'reflect', 'constant'}
            The boundary parameter determines how the input array is extended beyond its boundaries for spline
            interpolation.

                ``periodic`` :
                    The images are extended by wrapping around to the opposite edge. Use this mode for periodic images
                    (default).

                ``reflect`` :
                    The images are extended by reflecting about the edge of the last pixel.

                ``constant`` :
                    The images are extended by filling all values beyond the edge with the same constant value, defined
                    by the 'cval' parameter.

        order : int
            The order of the spline interpolation (default is 3). The order has to be in the range 0-5.
        normalization : {'values', 'amplitude'}
            The normalization parameter determines which quantity is preserved after normalization.

                ``values`` :
                    The pixel-wise values of the images are preserved.

                ``intensity`` :
                    The total intensity of the images is preserved.

        cval : scalar, optional
            Value to fill past edges in spline interpolation input if boundary is 'constant' (default is 0.0).

        Returns
        -------
        interpolated_images : Images
            The interpolated images.
        """
        if method == "fft" and boundary != "periodic":
            raise ValueError(
                "Only periodic boundaries available for FFT interpolation."
            )

        if sampling is None and gpts is None:
            raise ValueError()

        if gpts is None and sampling is not None:
            if np.isscalar(sampling):
                sampling = (sampling,) * 2
            gpts = tuple(int(np.ceil(l / d)) for d, l in zip(sampling, self.extent))

        elif gpts is not None:
            if np.isscalar(gpts):
                gpts = (gpts,) * 2
        else:
            raise ValueError()

        xp = get_array_module(self.array)

        sampling = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])

        def interpolate_spline(array, old_gpts, new_gpts, pad_mode, order, cval):
            xp = get_array_module(array)
            x = xp.linspace(0.0, old_gpts[0], new_gpts[0], endpoint=False)
            y = xp.linspace(0.0, old_gpts[1], new_gpts[1], endpoint=False)
            positions = xp.meshgrid(x, y, indexing="ij")
            positions = xp.stack(positions, axis=-1)
            return _interpolate_stack(
                array, positions, pad_mode, order=order, cval=cval
            )

        if boundary == "periodic":
            boundary = "wrap"

        array = None
        if self.is_lazy:
            array = self.array.rechunk(
                chunks=self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],))
            )
            if method == "fft":
                array = array.map_blocks(
                    fft_interpolate,
                    new_shape=gpts,
                    normalization=normalization,
                    chunks=self.array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
                    meta=xp.array((), dtype=self.array.dtype),
                )

            elif method == "spline":
                array = array.map_blocks(
                    interpolate_spline,
                    old_gpts=self.shape[-2:],
                    new_gpts=gpts,
                    order=order,
                    cval=cval,
                    pad_mode=boundary,
                    chunks=self.array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
                    meta=xp.array((), dtype=self.array.dtype),
                )

        else:
            if method == "fft":
                array = fft_interpolate(self.array, gpts, normalization=normalization)
            elif method == "spline":
                array = interpolate_spline(
                    self.array,
                    old_gpts=self.shape[-2:],
                    new_gpts=gpts,
                    pad_mode=boundary,
                    order=order,
                    cval=cval,
                )

        if array is None:
            raise RuntimeError()

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["sampling"] = sampling
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def interpolate_line_at_position(
        self,
        center: Union[Tuple[float, float], Atom],
        angle: float,
        extent: float,
        gpts: int = None,
        sampling: float = None,
        width: float = 0.0,
        order: int = 3,
        endpoint: bool = True,
    ):

        from abtem.scan import LineScan

        scan = LineScan.at_position(position=center, extent=extent, angle=angle)

        return self.interpolate_line(
            scan.start,
            scan.end,
            gpts=gpts,
            sampling=sampling,
            width=width,
            order=order,
            endpoint=endpoint,
        )

    def interpolate_line(
        self,
        start: Union[Tuple[float, float], Atom] = None,
        end: Union[Tuple[float, float], Atom] = None,
        sampling: float = None,
        gpts: int = None,
        width: float = 0.0,
        margin: float = 0.0,
        order: int = 3,
        endpoint: bool = False,
    ) -> "RealSpaceLineProfiles":
        """
        Interpolate image(s) along a given line. Either 'sampling' or 'gpts' must be provided.

        Parameters
        ----------
        start : two float, Atom, optional
            Starting position of the line [Å] (alternatively taken from a selected atom).
        end : two float, Atom, optional
            Ending position of the line [Å] (alternatively taken from a selected atom).
        sampling : float
            Sampling of grid points along the line [1 / Å].
        gpts : int
            Number of grid points along the line.
        width : float, optional
            The interpolation will be averaged across a perpendicular distance equal to this width.
        margin : float or tuple of float, optional
            Add margin [Å] to the start and end interpolated line.
        order : int, optional
            The spline interpolation order.
        endpoint : bool
            Sets whether the ending position is included or not.

        Returns
        -------
        line_profiles : RealSpaceLineProfiles
            The interpolated line(s).
        """

        from abtem.scan import LineScan

        if self.is_complex:
            raise NotImplementedError

        if (sampling is None) and (gpts is None):
            sampling = min(self.sampling)

        xp = get_array_module(self.array)

        if start is None:
            start = (0.0, 0.0)

        if end is None:
            end = (0.0, self.extent[0])

        scan = LineScan(
            start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint
        )

        if margin != 0.0:
            scan.add_margin(margin)

        positions = xp.asarray(scan.get_positions(lazy=False) / self.sampling)

        if width:
            direction = xp.array(scan.end) - xp.array(scan.start)
            direction = direction / xp.linalg.norm(direction)
            perpendicular_direction = xp.array([-direction[1], direction[0]])
            n = xp.floor(width / min(self.sampling) / 2) * 2 + 1
            perpendicular_positions = (
                xp.linspace(-n / 2, n / 2, int(n))[:, None]
                * perpendicular_direction[None]
            )
            positions = perpendicular_positions[None, :] + positions[:, None]

        if self.is_lazy:
            array = self.array.map_blocks(
                _interpolate_stack,
                positions=positions,
                mode="wrap",
                order=order,
                drop_axis=self.base_axes,
                new_axis=self.base_axes[0],
                chunks=self.array.chunks[:-2] + (positions.shape[0],),
                meta=xp.array((), dtype=np.float32),
            )
        else:
            array = _interpolate_stack(self.array, positions, mode="wrap", order=order)

        if width:
            array = array.mean(-1)

        metadata = copy.copy(self.metadata)
        metadata.update(scan.metadata)

        return RealSpaceLineProfiles(
            array=array,
            sampling=scan.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

    def tile(self, repetitions: Tuple[int, int]) -> "Images":
        """
        Tile image(s).

        Parameters
        ----------
        repetitions : tuple of int
            The number of repetitions of the images along the `x`- and `y`-axis, respectively.

        Returns
        -------
        tiled_images : Images
            The tiled image(s).
        """
        if len(repetitions) != 2:
            raise RuntimeError()
        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = np.tile(
            self.array, (1,) * (len(self.array.shape) - 2) + repetitions
        )
        return self.__class__(**kwargs)

    def gaussian_filter(
        self,
        sigma: Union[float, Tuple[float, float]],
        boundary: str = "periodic",
        cval: float = 0.0,
    ):
        """
        Apply 2D gaussian filter to image(s).

        Parameters
        ----------
        sigma : float or two float
            Standard deviation for the Gaussian kernel in the `x` and `y`-direction. If given as a single number, the
            standard deviation is equal for both axes.

        boundary : {'periodic', 'reflect', 'constant'}
            The boundary parameter determines how the images are extended beyond their boundaries when the filter
            overlaps with a border.

                ``periodic`` :
                    The images are extended by wrapping around to the opposite edge. Use this mode for periodic
                    (default).

                ``reflect`` :
                    The images are extended by reflecting about the edge of the last pixel.

                ``constant`` :
                    The images are extended by filling all values beyond the edge with the same constant value, defined
                    by the 'cval' parameter.
        cval : scalar, optional
            Value to fill past edges in spline interpolation input if boundary is 'constant' (default is 0.0).

        Returns
        -------
        filtered_images : Images
            The filtered image(s).
        """
        xp = get_array_module(self.array)
        gaussian_filter = get_ndimage_module(self.array).gaussian_filter

        if boundary == "periodic":
            mode = "wrap"
        elif boundary in ("reflect", "constant"):
            mode = boundary
        else:
            raise ValueError()

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        sigma = (0,) * (len(self.shape) - 2) + tuple(
            s / d for s, d in zip(sigma, self.sampling)
        )

        if self.is_lazy:
            depth = tuple(
                min(int(np.ceil(4.0 * s)), n) for s, n in zip(sigma, self.base_shape)
            )
            array = self.array.map_overlap(
                gaussian_filter,
                sigma=sigma,
                boundary=boundary,
                mode=mode,
                cval=cval,
                depth=(0,) * (len(self.shape) - 2) + depth,
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = gaussian_filter(self.array, sigma=sigma, mode=mode, cval=cval)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def diffractograms(self) -> "DiffractionPatterns":
        """
        Calculate diffractograms (i.e. power spectra) from image(s).

        Returns
        -------
        diffractograms : DiffractionPatterns
            Diffractograms of image(s).
        """
        xp = get_array_module(self.array)

        def diffractograms(array):
            array = xp.fft.fft2(array)
            return xp.fft.fftshift(xp.abs(array), axes=(-2, -1))

        if self.is_lazy:
            array = self.array.rechunk(
                chunks=self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],))
            )
            array = array.map_blocks(
                diffractograms, meta=xp.array((), dtype=xp.float32)
            )
        else:
            array = diffractograms(self.array)

        sampling = 1 / self.extent[0], 1 / self.extent[1]
        return DiffractionPatterns(
            array=array,
            sampling=sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def show(
        self,
        cmap: str = "viridis",
        explode: bool = False,
        ax: Axes = None,
        figsize: Tuple[int, int] = None,
        title: Union[bool, str] = True,
        panel_titles: Union[bool, List[str]] = True,
        x_ticks: bool = True,
        y_ticks: bool = True,
        x_label: Union[bool, str] = True,
        y_label: Union[bool, str] = True,
        row_super_label: Union[bool, str] = False,
        col_super_label: Union[bool, str] = False,
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        common_color_scale=False,
        cbar: bool = False,
        cbar_labels: str = None,
        sizebar: bool = False,
        float_formatting: str = ".2f",
        panel_labels: dict = None,
        image_grid_kwargs: dict = None,
        imshow_kwargs: dict = None,
        anchored_text_kwargs: dict = None,
        complex_coloring_kwargs: dict = None,
    ) -> Axes:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. Ignored if image array is complex.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown.
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for image grids.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Add a title to the figure. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the metadata dictionary, if this item exists.
        panel_titles : bool or list of str, optional
            Add titles to each panel. If True a title will be created from the axis metadata. If given as a list of
            strings an item must exist for each panel.
        x_ticks : bool or list, optional
            If False, the ticks on the `x`-axis will be removed.
        y_ticks : bool or list, optional
            If False, the ticks on the `y`-axis will be removed.
        x_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        y_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will created from the corresponding axis
            metadata. A string may be given to override this.
        row_super_label : bool or str, optional
            Add super label to the rows of an image grid. If True the label will be created from the corresponding axis
            metadata. A string may be given to override this. The default is no super label.
        col_super_label : bool or str, optional
            Add super label to the columns of an image grid. If True the label will be created from the corresponding
            axis metadata. A string may be given to override this. The default is no super label.
        power : float
            Show image on a power scale.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The position and size of the colorbar(s) may be controlled by passing
            keyword arguments to `mpl_toolkits.axes_grid1.axes_grid.ImageGrid` through `image_grid_kwargs`.
        cbar_labels : str or list of str
            Label(s) for the colorbar(s).
        sizebar : bool, optional,
            Add a size bar to the image(s).
        float_formatting : str, optional
            A formatting string used for formatting the floats of the panel titles.
        panel_labels : list of str
            A list of labels for each panel of a grid of images.
        image_grid_kwargs : dict
            Additional keyword arguments passed to `mpl_toolkits.axes_grid1.axes_grid.ImageGrid`.
        imshow_kwargs : dict
            Additional keyword arguments passed to `matplotlib.axes.Axes.imshow`.
        anchored_text_kwargs : dict
            Additional keyword arguments passed to `matplotlib.offsetbox.AnchoredText`. This is used for creating panel
            labels.

        Returns
        -------
        Figure, matplotlib.axes.Axes
        """
        if not explode:
            measurements = self[(0,) * len(self.ensemble_shape)]
        else:
            if ax is not None:
                raise NotImplementedError(
                    "`ax` not implemented for with `explode = True`."
                )
            measurements = self

        return show_measurement_2d(
            measurements=measurements,
            cmap=cmap,
            figsize=figsize,
            super_title=title,
            sub_title=panel_titles,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label=x_label,
            y_label=y_label,
            row_super_label=row_super_label,
            col_super_label=col_super_label,
            power=power,
            vmin=vmin,
            vmax=vmax,
            common_color_scale=common_color_scale,
            cbar=cbar,
            cbar_labels=cbar_labels,
            sizebar=sizebar,
            float_formatting=float_formatting,
            panel_labels=panel_labels,
            image_grid_kwargs=image_grid_kwargs,
            imshow_kwargs=imshow_kwargs,
            anchored_text_kwargs=anchored_text_kwargs,
            complex_coloring_kwargs=complex_coloring_kwargs,
            axes=ax,
        )


class _SinglePointMeasurement(BaseMeasurement):
    _base_dims = 0

    def __init__(self, array, ensemble_axes_metadata, metadata):
        super().__init__(array, ensemble_axes_metadata, metadata)


class _AbstractMeasurement1d(BaseMeasurement):
    _base_dims = 1

    def __init__(
        self,
        array: np.ndarray,
        sampling: float = None,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: dict = None,
    ):

        self._sampling = sampling

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
            allow_base_axis_chunks=True,
        )

    @property
    def _area_per_pixel(self):
        raise RuntimeError("Cannot infer pixel area from metadata.")

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None) -> "T":
        """
        Creates line profile(s) from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more 1D line profiles.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
            axes must be RealSpaceAxis.
        metadata : dict, optional
            A dictionary defining the measurement metadata.

        Returns
        -------
        line_profiles : RealSpaceLineProfiles
            Line profiles from the array and metadata.
        """
        sampling = axes_metadata[-1].sampling
        axes_metadata = axes_metadata[:-1]
        return cls(
            array,
            sampling=sampling,
            ensemble_axes_metadata=axes_metadata,
            metadata=metadata,
        )

    @property
    def extent(self) -> float:
        return self.sampling * self.shape[-1]

    @property
    def sampling(self) -> float:
        return self._sampling

    @property
    @abstractmethod
    def base_axes_metadata(self) -> List[Union[RealSpaceAxis, FourierSpaceAxis]]:
        pass

    def width(self, height: float = 0.5):
        """
        Calculate the width of line(s) at a given height, e.g. full width at half maximum (the default).

        Parameters
        ----------
        height : float
            Fractional height at which the width is calculated.

        Returns
        -------
        width : float
            The calculated width.
        """

        def calculate_widths(array, sampling, height):
            xp = get_array_module(array)
            array = array - xp.max(array, axis=-1, keepdims=True) * height

            widths = xp.zeros(array.shape[:-1], dtype=np.float32)
            for i in np.ndindex(array.shape[:-1]):
                zero_crossings = xp.where(xp.diff(xp.sign(array[i]), axis=-1))[0]
                left, right = zero_crossings[0], zero_crossings[-1]
                widths[i] = (right - left) * sampling

            return widths

        return self.array.map_blocks(
            calculate_widths,
            drop_axis=(len(self.array.shape) - 1,),
            dtype=np.float32,
            sampling=self.sampling,
            height=height,
        )

    def interpolate(
        self,
        sampling: float = None,
        gpts: int = None,
        order: int = 3,
        endpoint: bool = False,
    ) -> "T":
        """
        Interpolate line profile(s) producing equivalent line profile(s) with a different sampling. Either 'sampling' or
        'gpts' must be provided (but not both).

        Parameters
        ----------
        sampling : float
            Sampling of line profiles after interpolation [Å].
        gpts : int
            Number of grid points of line profiles after interpolation. Do not use if 'sampling' is used.
        order : int
            The order of the spline interpolation (default is 3). The order has to be in the range 0-5.

        Returns
        -------
        interpolated_profiles : RealSpaceLineProfiles
            The interpolated line profile(s).
        """
        map_coordinates = get_ndimage_module(self.array).map_coordinates
        xp = get_array_module(self.array)

        if (gpts is not None) and (sampling is not None):
            raise RuntimeError()

        if sampling is None and gpts is None:
            sampling = self.sampling

        if gpts is None:
            gpts = int(np.ceil(self.extent / sampling))

        if sampling is None:
            sampling = self.extent / gpts

        def interpolate(array, gpts, endpoint, order):
            old_shape = array.shape
            array = array.reshape((-1, array.shape[-1]))

            array = xp.pad(array, ((0,) * 2, (3,) * 2), mode="wrap")
            new_points = xp.linspace(
                3.0, array.shape[-1] - 3.0, gpts, endpoint=endpoint
            )[None]

            new_array = xp.zeros(array.shape[:-1] + (gpts,), dtype=xp.float32)
            for i in range(len(array)):
                map_coordinates(array[i], new_points, new_array[i], order=order)

            return new_array.reshape(old_shape[:-1] + (gpts,))

        if self.is_lazy:
            array = self.array.rechunk(self.array.chunks[:-1] + ((self.shape[-1],),))
            array = array.map_blocks(
                interpolate,
                gpts=gpts,
                endpoint=endpoint,
                order=order,
                chunks=self.array.chunks[:-1] + ((gpts,)),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = interpolate(self.array, gpts, endpoint, order)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["sampling"] = sampling
        return self.__class__(**kwargs)


class RealSpaceLineProfiles(_AbstractMeasurement1d):
    """
    A collection of real-space line profile(s).

    Parameters
    ----------
    array : np.ndarray
        1D or greater array containing data of type `float` or ´complex´.
    sampling : float
        Sampling of line profiles [Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        sampling: float = None,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: dict = None,
    ):

        super().__init__(
            array=array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def base_axes_metadata(self) -> List[RealSpaceAxis]:
        return [RealSpaceAxis(label="r", sampling=self.sampling, units="Å")]

    def tile(self, reps: int) -> "RealSpaceLineProfiles":
        kwargs = self._copy_kwargs(exclude=("array",))
        xp = get_array_module(self.array)
        reps = (1,) * (len(self.array.shape) - 1) + (reps,)

        if self.is_lazy:
            kwargs["array"] = da.tile(self.array, reps)
        else:
            kwargs["array"] = xp.tile(self.array, reps)

        return self.__class__(**kwargs)

    def add_to_plot(self, *args, **kwargs):
        if not all(key in self.metadata for key in ("start", "end")):
            raise RuntimeError(
                "The metadata does not contain the keys 'start' and 'end'"
            )

        start, end = self.metadata["start"], self.metadata["end"]
        from abtem.scan import LineScan

        return LineScan(start=start, end=end).add_to_plot(*args, **kwargs)

    def show(
        self,
        ax: Axes = None,
        figsize: Tuple[int, int] = None,
        title: str = None,
        x_label: str = None,
        y_label=None,  # TODO: needs to be implemented!
        float_formatting: str = ".2f",
        **kwargs,
    ):
        """
        Show the line profile(s) using matplotlib.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If given the plots are added to the Axes. This is not available for image grids.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Add a title to the figure. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the metadata dictionary, if this item exists.
        x_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        y_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        float_formatting : str, optional
            A formatting string used for formatting the floats of the panel titles.

        Returns
        -------
        matplotlib Axes
        """
        extent = [0, self.extent]
        return show_measurements_1d(
            self,
            ax=ax,
            figsize=figsize,
            title=title,
            x_label=x_label,
            extent=extent,
            float_formatting=float_formatting,
            **kwargs,
        )


class ReciprocalSpaceLineProfiles(_AbstractMeasurement1d):
    """
    A collection of reciprocal-space line profile(s).

    Parameters
    ----------
    array : np.ndarray
        1D or greater array containing data of type `float` or ´complex´.
    sampling : float
        Sampling of line profiles [1 / Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        sampling: float = None,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: dict = None,
    ):
        super().__init__(
            array=array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [FourierSpaceAxis(label="k", sampling=self.sampling, units="1 / Å")]

    @property
    def angular_extent(self):
        return self.extent * self.wavelength * 1e3

    def show(
        self,
        ax: Axes = None,
        figsize: Tuple[int, int] = None,
        title: str = None,
        x_label: str = None,
        y_label=None,  # TODO: needs to be implemented
        units: str = "reciprocal",
        float_formatting: str = ".2f",
        **kwargs,
    ):
        """
        Show the reciprocal-space line profile(s) using matplotlib.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If given the plots are added to the Axes. This is not available for image grids.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
        title : bool or str, optional
            Add a title to the figure. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the metadata dictionary, if this item exists.
        x_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        y_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        units : str, optional
            The units of the reciprocal line profile can be either 'reciprocal' (resulting in [1 / Å]), or 'mrad'.
        float_formatting : str, optional
            A formatting string used for formatting the floats of the panel titles.

        Returns
        -------
        matplotlib Axes
        """
        if units in angular_units:
            extent = [0, self.angular_extent]

            if x_label is None:
                x_label = "Scattering angle [mrad]"
        elif units in reciprocal_units:
            extent = [0, self.extent]
        else:
            raise RuntimeError()

        return show_measurements_1d(
            self,
            ax=ax,
            figsize=figsize,
            title=title,
            x_label=x_label,
            float_formatting=float_formatting,
            extent=extent,
            **kwargs,
        )


def _integrate_gradient_2d(gradient, sampling):
    xp = get_array_module(gradient)
    gx, gy = gradient.real, gradient.imag
    (nx, ny) = gx.shape[-2:]
    ikx = xp.fft.fftfreq(nx, d=sampling[0])
    iky = xp.fft.fftfreq(ny, d=sampling[1])
    grid_ikx, grid_iky = xp.meshgrid(ikx, iky, indexing="ij")
    k = grid_ikx**2 + grid_iky**2
    k[k == 0] = 1e-12
    That = (xp.fft.fft2(gx) * grid_ikx + xp.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
    T = xp.real(xp.fft.ifft2(That))
    T -= xp.min(T)
    return T


def _fourier_space_bilinear_nodes_and_weight(
    old_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
    old_angular_sampling: Tuple[float, float],
    new_angular_sampling: Tuple[float, float],
    xp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = []
    weights = []

    old_sampling = (
        1 / old_angular_sampling[0] / old_shape[0],
        1 / old_angular_sampling[1] / old_shape[1],
    )
    new_sampling = (
        1 / new_angular_sampling[0] / new_shape[0],
        1 / new_angular_sampling[1] / new_shape[1],
    )

    for n, m, r, d in zip(old_shape, new_shape, old_sampling, new_sampling):
        k = xp.fft.fftshift(xp.fft.fftfreq(n, r).astype(xp.float32))
        k_new = xp.fft.fftshift(xp.fft.fftfreq(m, d).astype(xp.float32))
        distances = k_new[None] - k[:, None]
        distances[distances < 0.0] = np.inf
        w = distances.min(0) / (k[1] - k[0])
        w[w == np.inf] = 0.0
        nodes.append(distances.argmin(0))
        weights.append(w)

    v, u = nodes
    vw, uw = weights
    v, u, vw, uw = xp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])
    return v, u, vw, uw


def _gaussian_source_size(measurements, sigma: Union[float, Tuple[float, float]]):
    if len(_scan_axes(measurements)) < 2:
        raise RuntimeError(
            "Gaussian source size not implemented for diffraction patterns with less than two scan axes."
        )

    if np.isscalar(sigma):
        sigma = (sigma,) * 2

    xp = get_array_module(measurements.array)
    gaussian_filter = get_ndimage_module(measurements._array).gaussian_filter

    padded_sigma = ()
    depth = ()
    i = 0
    for axis, n in zip(measurements.ensemble_axes, measurements.ensemble_shape):
        if axis in _scan_axes(measurements):
            scan_sampling = _scan_sampling(measurements)[i]
            padded_sigma += (sigma[i] / scan_sampling,)
            depth += (min(int(np.ceil(4.0 * sigma[i] / scan_sampling)), n),)
            i += 1
        else:
            padded_sigma += (0.0,)
            depth += (0,)

    padded_sigma += (0.0,) * 2
    depth += (0,) * 2

    if measurements.is_lazy:
        array = measurements.array.map_overlap(
            gaussian_filter,
            sigma=padded_sigma,
            mode="wrap",
            depth=depth,
            meta=xp.array((), dtype=xp.float32),
        )
    else:
        array = gaussian_filter(measurements.array, sigma=padded_sigma, mode="wrap")

    kwargs = measurements._copy_kwargs(exclude=("array",))

    return measurements.__class__(array, **kwargs)


class DiffractionPatterns(BaseMeasurement):
    """
    One or more diffraction patterns.

    Parameters
    ----------
    array : np.ndarray
        2D or greater array containing data with `float` type. The second-to-last and last dimensions are the
        reciprocal space `y`- and `x`-axis of the diffraction pattern.
    sampling : float or two float
        The reciprocal-space sampling of the diffraction patterns [1 / Å].
    fftshift : bool, optional
        If True, the diffraction patterns are assumed to have the zero-frequency component to the center of the
        spectrum, otherwise the center(s) are assumed to be at (0,0).
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    """

    _base_dims = 2  # The dimension of diffraction patterns is 2.

    def __init__(
        self,
        array: Union[np.ndarray, da.core.Array],
        sampling: Union[float, Tuple[float, float]],
        fftshift: bool = False,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: dict = None,
    ):

        if np.isscalar(sampling):
            sampling = (float(sampling),) * 2
        else:
            sampling = float(sampling[0]), float(sampling[1])

        self._fftshift = fftshift
        self._sampling = sampling

        self._base_axes = (-2, -1)

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
            allow_base_axis_chunks=False,
        )

    @property
    def _area_per_pixel(self):
        return _scan_area_per_pixel(self)

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None):
        """
        Creates diffraction pattern(s) from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more 2D diffraction patterns. The second-to-last and last dimensions are the
            diffraction pattern `y`- and `x`-axis.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
            axes must be RealSpaceAxis.
        metadata : dict
            A dictionary defining the measurement metadata.

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
            Diffraction pattern(s) from the array and metadata.
        """
        sampling = (axes_metadata[-2].sampling, axes_metadata[-1].sampling)
        fftshift = axes_metadata[-1].fftshift
        axes_metadata = axes_metadata[:-2]
        return cls(
            array,
            sampling=sampling,
            ensemble_axes_metadata=axes_metadata,
            fftshift=fftshift,
            metadata=metadata,
        )

    @property
    def base_axes_metadata(self):
        limits = self.limits
        return [
            FourierSpaceAxis(
                sampling=self.sampling[0],
                offset=limits[0][0],
                label="kx",
                units="1 / Å",
                fftshift=self.fftshift,
            ),
            FourierSpaceAxis(
                sampling=self.sampling[1],
                offset=limits[1][0],
                label="ky",
                units="1 / Å",
                fftshift=self.fftshift,
            ),
        ]

    def index_diffraction_spots(
        self, cell: Union[Cell, float, Tuple[float, float, float]]
    ):
        """
        Indexes the Bragg reflections (diffraction spots) by their Miller indices.

        Parameters
        ----------
        cell : ase.cell.Cell or float or tuple of float
            The assumed unit cell with respect to the diffraction pattern should be indexed. Must be one of ASE `Cell`
            object, float (for a cubic unit cell) or three floats (for orthorhombic unit cells).

        Returns
        -------
        indexed_patterns : IndexedDiffractionPatterns
            The indexed diffraction pattern(s).
        """

        diffraction_patterns = self.block_direct()
        diffraction_patterns = diffraction_patterns.to_cpu()
        return IndexedDiffractionPatterns.index_diffraction_patterns(diffraction_patterns, cell)

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def sampling(self) -> Tuple[float, float]:
        return self._sampling

    @property
    def angular_sampling(self) -> Tuple[float, float]:
        return (
            self.sampling[0] * self.wavelength * 1e3,
            self.sampling[1] * self.wavelength * 1e3,
        )

    @property
    def max_angles(self):
        """Maximum scattering angle in `x` and `y` [mrad]."""

        return (
            self.shape[-2] // 2 * self.angular_sampling[0],
            self.shape[-1] // 2 * self.angular_sampling[1],
        )

    @property
    def limits(self) -> List[Tuple[float, float]]:
        """Lowest and highest spatial frequency in `x` and `y` [1 / Å]."""

        limits = []
        for i in (-2, -1):
            if self.shape[i] % 2:
                limits += [
                    (
                        -(self.shape[i] - 1) // 2 * self.sampling[i],
                        (self.shape[i] - 1) // 2 * self.sampling[i],
                    )
                ]
            else:
                limits += [
                    (
                        -self.shape[i] // 2 * self.sampling[i],
                        (self.shape[i] // 2 - 1) * self.sampling[i],
                    )
                ]
        return limits

    @property
    def angular_limits(self) -> List[Tuple[float, float]]:
        """Lowest and highest scattering angle in `x` and `y` [mrad]."""

        limits = self.limits
        limits[0] = (
            limits[0][0] * self.wavelength * 1e3,
            limits[0][1] * self.wavelength * 1e3,
        )
        limits[1] = (
            limits[1][0] * self.wavelength * 1e3,
            limits[1][1] * self.wavelength * 1e3,
        )
        return limits

    @property
    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reciprocal-space frequency coordinates [1 / Å]."""

        return (
            self.axes_metadata[-2].coordinates(self.base_shape[-2]),
            self.axes_metadata[-1].coordinates(self.base_shape[-1]),
        )

    @property
    def angular_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Scattering angle coordinates [mrad]."""

        xp = get_array_module(self.array)
        limits = self.angular_limits
        alpha_x = xp.linspace(
            limits[0][0], limits[0][1], self.shape[-2], dtype=xp.float32
        )
        alpha_y = xp.linspace(
            limits[1][0], limits[1][1], self.shape[-1], dtype=xp.float32
        )
        return alpha_x, alpha_y

    @staticmethod
    def _batch_interpolate_bilinear(array, new_sampling, sampling, new_gpts):

        xp = get_array_module(array)
        v, u, vw, uw = _fourier_space_bilinear_nodes_and_weight(
            array.shape[-2:], new_gpts, sampling, new_sampling, xp
        )

        old_shape = array.shape
        array = array.reshape((-1,) + array.shape[-2:])

        old_sums = array.sum((-2, -1), keepdims=True)

        if xp is cp:
            array = interpolate_bilinear_cuda(array, v, u, vw, uw)
        else:
            array = interpolate_bilinear(array, v, u, vw, uw)

        array = array / array.sum((-2, -1), keepdims=True) * old_sums

        return array.reshape(old_shape[:-2] + array.shape[-2:])

    # def interpolate_scan(self, sampling: Union[str, float, Tuple[float, float]] = None):
    #     """
    #     Interpolate between scan positions producing additional diffraction patterns.
    #
    #     Parameters
    #     ----------
    #     sampling : two float or 'uniform'
    #         Target scan sampling after interpolation in `x` and `y` [1 / Å].
    #
    #     Returns
    #     -------
    #     interpolated_diffraction_patterns : DiffractionPatterns
    #     """
    #
    #     if np.isscalar(sampling):
    #         sampling = (sampling,) * 2
    #
    #     scan_sampling = self.scan_sampling
    #
    #     if len(scan_sampling) != 2:
    #         raise NotImplementedError()
    #
    #     sampling, gpts = adjusted_gpts(sampling, self.scan_sampling, self.base_shape)
    #
    #     xp = get_array_module(self.array)
    #
    #     if self.is_lazy:
    #         array = da.moveaxis(self.array, self.scan_axes, (-2, -1))
    #         array = array.rechunk(('auto',) * (len(array.shape) - 2) + (-1, -1))
    #         array = array.map_blocks(self._batch_interpolate_bilinear,
    #                                  sampling=scan_sampling,
    #                                  new_sampling=sampling,
    #                                  new_gpts=gpts,
    #                                  chunks=array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
    #                                  dtype=np.float32)
    #         array = da.moveaxis(self.array, (-2, -1), self.scan_axes)
    #     else:
    #         array = xp.moveaxis(self.array, self.scan_axes, (-2, -1))
    #         array = self._batch_interpolate_bilinear(array, sampling=self.sampling, new_sampling=sampling,
    #                                                  new_gpts=gpts)
    #         array = xp.moveaxis(self.array, (-2, -1), self.scan_axes)
    #
    #     return array
    #     # print(array.chunks[-2:], gpts[0] / self.scan_shape[0] * sum(array.chunks[-2]), gpts)

    def interpolate(
        self, sampling: Union[str, float, Tuple[float, float]] = None, gpts=None
    ):
        """
        Interpolate diffraction pattern(s) producing equivalent pattern(s) with a different sampling.

        Parameters
        ----------
        sampling : 'uniform' or float or two floats
            Sampling of diffraction patterns after interpolation in `x` and `y` [1 / Å]. If a single value, the same
            sampling is used for both axes. If 'uniform', the diffraction patterns are down-sampled along the axis with
            the smallest pixel size such that the sampling is uniform.

        Returns
        -------
        interpolated_diffraction_patterns : DiffractionPatterns
            The interpolated diffraction pattern(s).
        """
        if gpts is None:
            if sampling == "uniform":
                sampling = (max(self.sampling),) * 2

            elif not isinstance(sampling, str) and np.isscalar(sampling):
                sampling = (sampling,) * 2

            sampling, gpts = adjusted_gpts(sampling, self.sampling, self.base_shape)
        else:
            sampling = tuple(
                d * old_n / new_n
                for d, old_n, new_n in zip(self.sampling, self.base_shape, gpts)
            )

        if self.is_lazy:
            array = self.array.map_blocks(
                self._batch_interpolate_bilinear,
                sampling=self.sampling,
                new_sampling=sampling,
                new_gpts=gpts,
                chunks=self.array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
                dtype=np.float32,
            )
        else:
            array = self._batch_interpolate_bilinear(
                self.array, sampling=self.sampling, new_sampling=sampling, new_gpts=gpts
            )

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["sampling"] = sampling
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def _check_integration_limits(self, inner: float, outer: float):

        if inner > outer:
            raise RuntimeError(
                f"Inner detection ({inner} mrad) angle cannot exceed the outer detection angle."
                f"({outer} mrad)"
            )

        if (outer > self.max_angles[0]) or (outer > self.max_angles[1]):
            if not np.isclose(min(self.max_angles), outer, atol=1e-5):
                raise RuntimeError(
                    f"Outer integration limit cannot exceed the maximum simulated angle ({outer} mrad > "
                    f"{min(self.max_angles)} mrad), please increase the number of grid points."
                )

    def gaussian_filter(self, sigma):
        # if len(_scan_axes(self)) < 2:
        #     raise RuntimeError(
        #         "Gaussian source size not implemented for diffraction patterns with less than two scan axes."
        #     )
        #
        # if np.isscalar(sigma):
        #     sigma = (sigma,) * 2

        xp = get_array_module(self.array)
        gaussian_filter = get_ndimage_module(self._array).gaussian_filter

        # padded_sigma = ()
        # depth = ()
        # i = 0
        # for axis, n in zip(self.ensemble_axes, self.ensemble_shape):
        #     if axis in _scan_axes(self):
        #         scan_sampling = _scan_sampling(self)[i]
        #         padded_sigma += (sigma[i] / scan_sampling,)
        #         depth += (min(int(np.ceil(4.0 * sigma[i] / scan_sampling)), n),)
        #         i += 1
        #     else:
        #         padded_sigma += (0.0,)
        #         depth += (0,)
        #
        # padded_sigma += (0.0,) * 2
        # depth += (0,) * 2

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        sigma = tuple(s / d for s, d in zip(sigma, self.sampling))
        padded_sigma = (0,) * len(self.ensemble_shape) + sigma
        depth = tuple(int(np.ceil(4.0 * ps)) for ps in padded_sigma)

        if self.is_lazy:
            array = self.array.map_overlap(
                gaussian_filter,
                sigma=padded_sigma,
                mode="wrap",
                depth=depth,
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = gaussian_filter(self.array, sigma=padded_sigma, mode="wrap")

        return self.__class__(
            array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
            fftshift=self.fftshift,
        )

    def gaussian_source_size(
        self, sigma: Union[float, Tuple[float, float]]
    ) -> "DiffractionPatterns":
        """
        Simulate the effect of a finite source size on diffraction pattern(s) using a Gaussian filter.

        The filter is not applied to diffraction pattern individually, but the intensity of diffraction patterns are mixed
        across scan axes. Applying this filter requires two linear scan axes.

        Applying this filter before integrating the diffraction patterns will produce the same image as integrating
        the diffraction patterns first then applying a Gaussian filter.

        Parameters
        ----------
        sigma : float or two float
            Standard deviation of Gaussian kernel in the `x` and `y`-direction. If given as a single number, the standard
            deviation is equal for both axes.

        Returns
        -------
        filtered_diffraction_patterns : DiffractionPatterns
            The filtered diffraction pattern(s).
        """

        return _gaussian_source_size(self, sigma)

        # if len(_scan_axes(self)) < 2:
        #     raise RuntimeError(
        #         "Gaussian source size not implemented for diffraction patterns with less than two scan axes."
        #     )
        #
        # if np.isscalar(sigma):
        #     sigma = (sigma,) * 2
        #
        # xp = get_array_module(self.array)
        # gaussian_filter = get_ndimage_module(self._array).gaussian_filter
        #
        # padded_sigma = ()
        # depth = ()
        # i = 0
        # for axis, n in zip(self.ensemble_axes, self.ensemble_shape):
        #     if axis in _scan_axes(self):
        #         scan_sampling = _scan_sampling(self)[i]
        #         padded_sigma += (sigma[i] / scan_sampling,)
        #         depth += (min(int(np.ceil(4.0 * sigma[i] / scan_sampling)), n),)
        #         i += 1
        #     else:
        #         padded_sigma += (0.0,)
        #         depth += (0,)
        #
        # padded_sigma += (0.0,) * 2
        # depth += (0,) * 2
        #
        # if self.is_lazy:
        #     array = self.array.map_overlap(
        #         gaussian_filter,
        #         sigma=padded_sigma,
        #         mode="wrap",
        #         depth=depth,
        #         meta=xp.array((), dtype=xp.float32),
        #     )
        # else:
        #     array = gaussian_filter(self.array, sigma=padded_sigma, mode="wrap")
        #
        # return self.__class__(
        #     array,
        #     sampling=self.sampling,
        #     ensemble_axes_metadata=self.ensemble_axes_metadata,
        #     metadata=self.metadata,
        #     fftshift=self.fftshift,
        # )

    def polar_binning(
        self,
        nbins_radial: int,
        nbins_azimuthal: int,
        inner: float = 0.0,
        outer: float = None,
        rotation: float = 0.0,
        offset: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Create polar measurements from the diffraction patterns by binning the measurements on a polar grid. This
        method may be used to simulate a segmented detector with a specified number of radial and azimuthal bins.

        Each bin is a segment of an annulus and the bins are spaced equally in the radial and azimuthal directions.
        The bins fit between a given inner and outer integration limit, they may be rotated around the origin, and their
        center may be shifted from the origin.

        Parameters
        ----------
        nbins_radial : int
            Number of radial bins.
        nbins_azimuthal : int
            Number of angular bins.
        inner : float
            Inner integration limit of the bins [mrad] (default is 0.0).
        outer : float
            Outer integration limit of the bins [mrad]. If not specified, this is set to be the maximum detected angle
            of the diffraction pattern.
        rotation : float
            Rotation of the bins around the origin [mrad] (default is 0.0).
        offset : two float
            Offset of the bins from the origin in `x` and `y` [mrad] (default is (0.0, 0.0).

        Returns
        -------
        polar_measurements : PolarMeasurements
            The polar measurements.
        """

        if nbins_radial <= 0 or nbins_azimuthal <= 0:
            raise RuntimeError("number of bins must be greater than zero")

        if outer is None:
            outer = min(self.max_angles)

        self._check_integration_limits(inner, outer)
        xp = get_array_module(self.array)

        def radial_binning(array, nbins_radial, nbins_azimuthal, sampling):
            xp = get_array_module(array)

            indices = _polar_detector_bins(
                gpts=array.shape[-2:],
                sampling=sampling,
                inner=inner,
                outer=outer,
                nbins_radial=nbins_radial,
                nbins_azimuthal=nbins_azimuthal,
                fftshift=self.fftshift,
                rotation=rotation,
                offset=offset,
                return_indices=True,
            )

            separators = xp.concatenate(
                (xp.array([0]), xp.cumsum(xp.array([len(i) for i in indices])))
            )

            new_shape = array.shape[:-2] + (nbins_radial, nbins_azimuthal)

            array = array.reshape(
                (
                    -1,
                    array.shape[-2] * array.shape[-1],
                )
            )[..., np.concatenate(indices)]

            result = xp.zeros(
                (
                    array.shape[0],
                    len(indices),
                ),
                dtype=xp.float32,
            )

            if xp is cp:
                sum_run_length_encoded_cuda(array, result, separators)

            else:
                _sum_run_length_encoded(array, result, separators)

            return result.reshape(new_shape)

        if self.is_lazy:
            array = self.array.map_blocks(
                radial_binning,
                nbins_radial=nbins_radial,
                nbins_azimuthal=nbins_azimuthal,
                sampling=self.angular_sampling,
                drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                chunks=self.array.chunks[:-2]
                + (
                    (nbins_radial,),
                    (nbins_azimuthal,),
                ),
                new_axis=(
                    len(self.shape) - 2,
                    len(self.shape) - 1,
                ),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = radial_binning(
                self.array,
                nbins_radial=nbins_radial,
                nbins_azimuthal=nbins_azimuthal,
                sampling=self.angular_sampling,
            )

        radial_sampling = (outer - inner) / nbins_radial
        azimuthal_sampling = 2 * np.pi / nbins_azimuthal

        return PolarMeasurements(
            array,
            radial_sampling=radial_sampling,
            azimuthal_sampling=azimuthal_sampling,
            radial_offset=inner,
            azimuthal_offset=rotation,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def radial_binning(
        self, step_size: float = 1.0, inner: float = 0.0, outer: float = None
    ) -> "PolarMeasurements":
        """
        Create polar measurement(s) from the diffraction pattern(s) by binning the measurements in annular regions. This
        method may be used to simulate a segmented detector with a specified number of radial bins.

        This is equivalent to detecting a wave function using the `FlexibleAnnularDetector`.

        Parameters
        ----------
        step_size : float, optional
            Radial extent of the bins [mrad] (default is 1.0).
        inner : float, optional
            Inner integration limit of the bins [mrad] (default is 0.0).
        outer : float, optional
            Outer integration limit of the bins [mrad]. If not specified, this is set to be the maximum detected angle
            of the diffraction pattern.

        Returns
        -------
        radially_binned_measurement : PolarMeasurements
            Radially binned polar measurement(s).
        """

        if outer is None:
            outer = min(self.max_angles)

        nbins_radial = int((outer - inner) / step_size)
        return self.polar_binning(nbins_radial, 1, inner, outer)

    def integrate_radial(self, inner: float, outer: float, offset=(0.0, 0.0)) -> Images:
        """
        Create images by integrating the diffraction patterns over an annulus defined by an inner and outer integration
        angle.

        Parameters
        ----------
        inner : float
            Inner integration limit [mrad].
        outer : float
            Outer integration limit [mrad].

        Returns
        -------
        integrated_images : Images
            The integrated images.
        """

        self._check_integration_limits(inner, outer)

        xp = get_array_module(self.array)

        def integrate_fourier_space(array, sampling):
            xp = get_array_module(array)

            bins = _annular_detector_mask(
                gpts=array.shape[-2:],
                sampling=sampling,
                inner=inner,
                outer=outer,
                fftshift=self.fftshift,
                xp=xp,
            )

            # import matplotlib.pyplot as plt
            # plt.imshow(xp.asnumpy(bins))
            # plt.show()


            #bins = xp.ones_like(array)

            return xp.sum(array * bins, axis=(-2, -1))

        if self.is_lazy:
            integrated_intensity = self.array.map_blocks(
                integrate_fourier_space,
                sampling=self.angular_sampling,
                drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            integrated_intensity = integrate_fourier_space(
                self.array, sampling=self.angular_sampling
            )

        return _reduced_scanned_images_or_line_profiles(integrated_intensity, self)

    def integrated_center_of_mass(self) -> Images:
        """
        Calculate integrated center-of-mass (iCOM) images from diffraction patterns. This method is only implemented
        for diffraction patterns with exactly two scan axes.

        Returns
        -------
        icom_images : Images
            The iCOM images.
        """
        com = self.center_of_mass()

        if isinstance(com, Images):
            return com.integrate_gradient()
        else:
            raise RuntimeError(
                f"Integrated center-of-mass not implemented for DiffractionPatterns with "
                f"{len(_scan_shape(self))} scan axes."
            )

    @staticmethod
    def _com(array, x, y):
        com_x = (array * x[:, None]).sum(axis=(-2, -1))
        com_y = (array * y[None]).sum(axis=(-2, -1))
        com = com_x + 1.0j * com_y
        return com

    def center_of_mass(
        self, units: str = "reciprocal"
    ) -> Union[Images, RealSpaceLineProfiles]:
        """
        Calculate center-of-mass images or line profiles from diffraction patterns. The results are of type `complex`
        where the real and imaginary part represents the `x` and `y` component.

        Returns
        -------
        com_images : Images
            Center-of-mass images.
        com_line_profiles : RealSpaceLineProfiles
            Center-of-mass line profiles (returned if there is only one scan axis).
        """

        if units in angular_units:
            x, y = self.angular_coordinates
        elif units in reciprocal_units:
            x, y = self.coordinates
        else:
            raise ValueError()

        xp = get_array_module(self.array)

        x, y = xp.asarray(x), xp.asarray(y)

        if self.is_lazy:
            array = self.array.map_blocks(
                self._com, x=x, y=y, drop_axis=self.base_axes, dtype=np.complex64
            )
        else:
            array = self._com(self.array, x=x, y=y)

        return _reduced_scanned_images_or_line_profiles(array, self)

    def epie(
        self,
        probe_guess,
        max_batch: int = 8,
        max_iter: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        fix_probe: bool = False,
        fix_com: bool = True,
        crop_to_scan: bool = True,
    ) -> Images:

        """
        Ptychographic reconstruction with the extended ptychographical engine (ePIE) algorithm.

        probe_guess : Probe
            The initial guess for the electron probe.
        max_batch : int, optional
            Maximum number of probe positions to update at every step (default is 8).
        max_iter : int, optional
            Maximum number of iterations to run the ePIE algorithm (default is 4).
        alpha : float, optional
            Step size of iterative updates for the object (default is 1.0).
        beta : float, optional
            Step size of iterative updates for the probe (default is 1.0).
        fix_probe : bool, optional
            If True, the probe will not be updated by the algorithm (default is False).
        fix_com : bool, optional
            If True, the center of mass of the probe will be centered (default).
        crop_to_scan : bool, optional
            If True, the output is cropped to the scan area (default).
        """

        from abtem.reconstruct.epie import epie

        reconstruction = epie(
            self,
            probe_guess,
            max_batch=max_batch,
            max_iter=max_iter,
            alpha=alpha,
            beta=beta,
            fix_com=fix_com,
            fix_probe=fix_probe,
        )

        if crop_to_scan:
            reconstruction = reconstruction.crop(_scan_extent(self))

        return reconstruction

    def bandlimit(self, inner: float, outer: float) -> "DiffractionPatterns":
        """
        Bandlimit diffraction pattern(s) by setting everything outside an annulus defined by two radial angles to
        zero.

        Parameters
        ----------
        inner : float
            Inner limit of zero region [mrad].
        outer : float
            Outer limit of zero region [mrad].

        Returns
        -------
        band-limited_diffraction_patterns : DiffractionPatterns
            The band-limited diffraction pattern(s).
        """

        def bandlimit(array, inner, outer):
            alpha_x, alpha_y = self.angular_coordinates
            alpha = alpha_x[:, None] ** 2 + alpha_y[None] ** 2
            block = (alpha >= inner**2) * (alpha < outer**2)
            return array * block

        xp = get_array_module(self.array)

        if self.is_lazy:
            array = self.array.map_blocks(
                bandlimit, inner=inner, outer=outer, meta=xp.array((), dtype=xp.float32)
            )
        else:
            array = bandlimit(self.array, inner, outer)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def crop(self, max_angle=None, gpts=None):
        if gpts is None:
            gpts = (
                int(2 * np.round(max_angle / self.angular_sampling[0])) + 1,
                int(2 * np.round(max_angle / self.angular_sampling[1])) + 1,
            )

        xp = get_array_module(self.array)
        array = xp.fft.fftshift(
            fft_crop(xp.fft.ifftshift(self.array, axes=(-2, -1)), new_shape=gpts),
            axes=(-2, -1),
        )
        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def block_direct(self, radius: float = None) -> "DiffractionPatterns":
        """
        Block the direct beam by setting the pixels of the zeroth-order Bragg reflection (non-scattered beam) to zero.

        Parameters
        ----------
        radius : float, optional
            The radius of the zeroth-order reflection to block [mrad]. If not given this will be inferred from the
            metadata, if available.

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
            The diffraction pattern(s) with the direct beam removed.
        """

        if radius is None:
            if "semiangle_cutoff" in self.metadata.keys():
                radius = self.metadata["semiangle_cutoff"]
            else:
                radius = max(self.angular_sampling) * 1.0001

        return self.bandlimit(radius, outer=np.inf)

    def _center_bin(self):
        return self.array.shape[0] // 2, self.array.shape[1] // 2

    def _select_frequency_bin(self, bins):
        bins = np.array(bins)
        center = np.array([self.base_shape[0] // 2, self.base_shape[1] // 2])
        indices = bins + center
        if len(bins.shape) == 2:
            array = self.array[..., indices[:, 0], indices[:, 1]]
        else:
            array = self.array[..., indices[0], indices[1]]

        return array

    def show(
        self,
        units: str = "reciprocal",
        cmap: str = "viridis",
        explode: bool = False,
        ax: Axes = None,
        figsize: Tuple[int, int] = None,
        title: Union[bool, str] = True,
        panel_titles: Union[bool, List[str]] = True,
        x_ticks: bool = True,
        y_ticks: bool = True,
        x_label: Union[bool, str] = True,
        y_label: Union[bool, str] = True,
        row_super_label: Union[bool, str] = False,
        col_super_label: Union[bool, str] = False,
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        common_color_scale=False,
        cbar: bool = False,
        cbar_labels: str = None,
        sizebar: bool = False,
        float_formatting: str = ".2f",
        panel_labels: dict = None,
        image_grid_kwargs: dict = None,
        imshow_kwargs: dict = None,
        anchored_text_kwargs: dict = None,
        complex_coloring_kwargs: dict = None,
    ) -> Axes:
        """
        Show the diffraction pattern(s) using matplotlib.

        Parameters
        ----------
        units : bool, optional
            The units of the diffraction pattern(s) can be either 'reciprocal' (resulting in [1 / Å]), or 'mrad'.
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. Ignored if image array is complex.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown.
        ax : matplotlib Axes, optional
            If given the plots are added to the Axes. This is not available for image grids.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Add a title to the figure. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the metadata dictionary, if this item exists.
        panel_titles : bool or list of str, optional
            Add titles to each panel. If True a title will be created from the axis metadata. If given as a list of
            strings an item must exist for each panel.
        x_ticks : bool or list, optional
            If False, the ticks on the `x`-axis will be removed.
        y_ticks : bool or list, optional
            If False, the ticks on the `y`-axis will be removed.
        x_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        y_label : bool or str, optional
            Add label to the `x`-axis of every plot. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this.
        row_super_label : bool or str, optional
            Add super label to the rows of an image grid. If True (default) the label will be created from the corresponding axis
            metadata. A string may be given to override this. The default is no super label.
        col_super_label : bool or str, optional
            Add super label to the columns of an image grid. If True (default) the label will be created from the corresponding
            axis metadata. A string may be given to override this. The default is no super label.
        power : float, optional
            Show image on a power scale (default is a power of 1.0, ie. linear scale).
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The position and size of the colorbar(s) may be controlled by passing
            keyword arguments to `mpl_toolkits.axes_grid1.axes_grid.ImageGrid` through 'image_grid_kwargs'.
        sizebar : bool, optional,
            Add a size bar to the image(s).
        float_formatting : str, optional
            A formatting string used for formatting the floats of the panel titles.
        panel_labels : list of str
            A list of labels for each panel of a grid of images.
        image_grid_kwargs : dict
            Additional keyword arguments passed to `mpl_toolkits.axes_grid1.axes_grid.ImageGrid`.
        imshow_kwargs : dict
            Additional keyword arguments passed to `matplotlib.axes.Axes.imshow`.
        anchored_text_kwargs : dict
            Additional keyword arguments passed to `matplotlib.offsetbox.AnchoredText`. This is used for creating panel
            labels.

        Returns
        -------
        matplotlib Axes
        """

        if not explode:
            measurements = self[(0,) * len(self.ensemble_shape)]
        else:
            if ax is not None:
                raise NotImplementedError(
                    "`ax` not implemented for with `explode = True`."
                )
            measurements = self

        if units.lower() in angular_units:
            x_label = "scattering angle x [mrad]"
            y_label = "scattering angle y [mrad]"
            extent = list(self.angular_limits[0] + self.angular_limits[1])
        elif units in bins:

            def bin_extent(n):
                if n % 2 == 0:
                    return -n // 2 - 0.5, n // 2 - 0.5
                else:
                    return -n // 2 + 0.5, n // 2 + 0.5

            x_label = "frequency bin n"
            y_label = "frequency bin m"
            extent = bin_extent(self.base_shape[0]) + bin_extent(self.base_shape[1])
        elif units.lower().strip() in reciprocal_units:
            extent = list(self.limits[0] + self.limits[1])
        else:
            raise ValueError()

        return show_measurement_2d(
            measurements=measurements,
            figsize=figsize,
            super_title=title,
            sub_title=panel_titles,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label=x_label,
            y_label=y_label,
            extent=extent,
            cmap=cmap,
            row_super_label=row_super_label,
            col_super_label=col_super_label,
            power=power,
            vmin=vmin,
            vmax=vmax,
            common_color_scale=common_color_scale,
            cbar=cbar,
            cbar_labels=cbar_labels,
            sizebar=sizebar,
            float_formatting=float_formatting,
            panel_labels=panel_labels,
            image_grid_kwargs=image_grid_kwargs,
            imshow_kwargs=imshow_kwargs,
            anchored_text_kwargs=anchored_text_kwargs,
            complex_coloring_kwargs=complex_coloring_kwargs,
            axes=ax,
        )


class PolarMeasurements(BaseMeasurement):
    """
    Class describing polar measurements with a specified number of radial and azimuthal bins.

    Each bin is a segment of an annulus and the bins are spaced equally in the radial and azimuthal directions.
    The bins may be rotated around the origin, and their center may be shifted from the origin.

    Parameters
    ----------
    array : np.ndarray
        Array containing the measurement.
    radial_sampling : float
        Sampling of the radial bins [mrad].
    azimuthal_sampling : int
        Sampling of the angular bins [mrad].
    radial_offset : float, optional
        Offset of the bins from the origin [mrad] (default is 0.0).
    azimuthal_offset : float, optional
        Rotation of the bins around the origin [mrad] (default is 0.0).

    Returns
    -------
    polar_measurements : PolarMeasurements
        The polar measurements.
    """

    _base_dims = 2  # The dimension of polar measurements is 2.

    def __init__(
        self,
        array: np.ndarray,
        radial_sampling: float,
        azimuthal_sampling: float,
        radial_offset: float = 0.0,
        azimuthal_offset: float = 0.0,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: dict = None,
    ):
        self._radial_sampling = radial_sampling
        self._azimuthal_sampling = azimuthal_sampling
        self._radial_offset = radial_offset
        self._azimuthal_offset = azimuthal_offset

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @classmethod
    def from_array_and_metadata(
        cls, array, axes_metadata, metadata
    ) -> "PolarMeasurements":
        """
        Creates polar measurements(s) from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more polar measurements. The second-to-last and last dimensions are the
            measurement `y`- and `x`-axis.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
            axes must be RealSpaceAxis.
        metadata : dict
            A dictionary defining the measurement metadata.

        Returns
        -------
        polar_measurements : PolarMeasurements
            Polar measurement(s) from the array and metadata.
        """
        radial_sampling = axes_metadata[-2].sampling
        radial_offset = axes_metadata[-2].offset
        azimuthal_sampling = axes_metadata[-1].sampling
        azimuthal_offset = axes_metadata[-1].offset
        return cls(
            array,
            radial_sampling=radial_sampling,
            radial_offset=radial_offset,
            azimuthal_sampling=azimuthal_sampling,
            azimuthal_offset=azimuthal_offset,
            ensemble_axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    def _area_per_pixel(self):
        return _scan_area_per_pixel(self)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [
            LinearAxis(
                label="Radial scattering angle",
                offset=self.radial_offset,
                sampling=self.radial_sampling,
                _concatenate=False,
                units="mrad",
            ),
            LinearAxis(
                label="Azimuthal scattering angle",
                offset=self.azimuthal_offset,
                sampling=self.azimuthal_sampling,
                _concatenate=False,
                units="rad",
            ),
        ]

    @property
    def radial_offset(self) -> float:
        return self._radial_offset

    @property
    def outer_angle(self) -> float:
        return self._radial_offset + self.radial_sampling * self.shape[-2]

    @property
    def radial_sampling(self) -> float:
        return self._radial_sampling

    @property
    def azimuthal_sampling(self) -> float:
        return self._azimuthal_sampling

    @property
    def azimuthal_offset(self) -> float:
        return self._azimuthal_offset

    def integrate_radial(
        self, inner: float, outer: float
    ) -> Union[Images, RealSpaceLineProfiles]:
        """
        Create images by integrating the polar measurements over an annulus defined by an inner and outer integration
        angle.

        Parameters
        ----------
        inner : float
            Inner integration limit [mrad].
        outer : float
            Outer integration limit [mrad].

        Returns
        -------
        integrated_images : Images
            The integrated images.
        realspace_line_profiles : RealSpaceLineProfiles
            Integrated line profiles (returned if there is only one scan axis).
        """
        return self.integrate(radial_limits=(inner, outer))

    # TODO: to be documented
    def integrate(
        self,
        radial_limits: Tuple[float, float] = None,
        azimuthal_limits: Tuple[float, float] = None,
        detector_regions: Sequence[int] = None,
    ) -> Union[Images, RealSpaceLineProfiles]:
        """
        Integrate polar regions to produce an image or line profiles.

        Parameters
        ----------
        radial_limits : tuple of floats
        azimuthal_limits : tuple of floats
        detector_regions : sequence of int

        Returns
        -------
        integrated_images : Images
        """

        if detector_regions is not None:
            if (radial_limits is not None) or (azimuthal_limits is not None):
                raise ValueError()

            array = self.array.reshape(self.shape[:-2] + (-1,))[
                ..., list(detector_regions)
            ].sum(axis=-1)
        else:
            if radial_limits is None:
                radial_slice = slice(None)
            else:
                inner_index = int(
                    (radial_limits[0] - self.radial_offset) / self.radial_sampling
                )
                outer_index = int(
                    (radial_limits[1] - self.radial_offset) / self.radial_sampling
                )
                radial_slice = slice(inner_index, outer_index)

                if outer_index > self.shape[-2]:
                    raise RuntimeError("Integration limit exceeded.")

            if azimuthal_limits is None:
                azimuthal_slice = slice(None)
            else:
                left_index = int(azimuthal_limits[0] / self.radial_sampling)
                right_index = int(azimuthal_limits[1] / self.radial_sampling)
                azimuthal_slice = slice(left_index, right_index)

            array = self.array[..., radial_slice, azimuthal_slice].sum(axis=(-2, -1))

        return _reduced_scanned_images_or_line_profiles(array, self)

    def gaussian_source_size(
        self, sigma: Union[float, Tuple[float, float]]
    ) -> "PolarMeasurements":
        """
        Simulate the effect of a finite source size on diffraction pattern(s) using a Gaussian filter.

        The filter is not applied to diffraction pattern individually, but the intensity of diffraction patterns are mixed
        across scan axes. Applying this filter requires two linear scan axes.

        Applying this filter before integrating the diffraction patterns will produce the same image as integrating
        the diffraction patterns first then applying a Gaussian filter.

        Parameters
        ----------
        sigma : float or two float
            Standard deviation of Gaussian kernel in the `x` and `y`-direction. If given as a single number, the standard
            deviation is equal for both axes.

        Returns
        -------
        filtered_diffraction_patterns : DiffractionPatterns
            The filtered diffraction pattern(s).
        """

        return _gaussian_source_size(self, sigma)

    def differentials(
        self,
        direction_1,
        direction_2,
        return_complex: bool = True,
    ):
        differential_1 = self.integrate(
            detector_regions=direction_1[1]
        ) - self.integrate(detector_regions=direction_1[0])

        differential_2 = self.integrate(
            detector_regions=direction_2[1]
        ) - self.integrate(detector_regions=direction_2[0])

        if not return_complex:
            return differential_1, differential_2

        xp = get_array_module(self.device)
        array = xp.zeros_like(differential_1.array, dtype=xp.complex64)

        array.real = differential_1.array
        array.imag = differential_2.array

        return differential_1.__class__(
            array, **differential_1._copy_kwargs(exclude=("array",))
        )

    # TODO: to be documented.
    def show(
        self,
        ax: Axes = None,
        title: str = None,
        min_azimuthal_division: float = np.pi / 20,
        grid: bool = True,
        figsize=None,
        radial_ticks=None,
        azimuthal_ticks=None,
        cbar=False,
        **kwargs,
    ):

        import matplotlib.patheffects as pe

        fig = plt.figure(figsize=figsize)

        if ax is None:
            ax = fig.add_subplot(projection="polar")

        if title is not None:
            ax.set_title(title)

        array = self.array[(0,) * (len(self.shape) - 2)]

        array = array[..., ::-1]

        repeat = int(self.azimuthal_sampling / min_azimuthal_division)
        r = np.pi / (4 * repeat) + self.azimuthal_offset + np.pi / 2
        azimuthal_grid = np.linspace(
            r, 2 * np.pi + r, self.shape[-1] * repeat, endpoint=False
        )

        d = (self.outer_angle - self.radial_offset) / 2 / self.shape[-2]
        radial_grid = np.linspace(
            self.radial_offset + d, self.outer_angle - d, self.shape[-2]
        )

        z = np.repeat(array, repeat, axis=-1)
        r, th = np.meshgrid(radial_grid, azimuthal_grid)

        im = ax.pcolormesh(th, r, z.T, shading="auto", **kwargs)
        ax.set_rlim([0, self.outer_angle * 1.1])

        if radial_ticks is None:
            radial_ticks = ax.get_yticks()

        if azimuthal_ticks is None:
            azimuthal_ticks = ax.get_xticks()

        ax.set_rgrids(
            radial_ticks, path_effects=[pe.withStroke(linewidth=4, foreground="white")]
        )
        ax.set_xticks(azimuthal_ticks)

        if cbar:
            fig.colorbar(im, label=_make_cbar_label(self))

        if grid:
            ax.grid(linewidth=2, color="white")

        return ax, im
