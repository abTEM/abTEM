"""Module for describing abTEM measurement objects."""

from __future__ import annotations

import copy
import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from numbers import Number
from typing import TYPE_CHECKING, Dict, Sequence, Type, TypeVar

import dask.array as da
import numpy as np
from ase import Atom
from ase.cell import Cell
from matplotlib.axes import Axes
from numba import jit, prange

from abtem.array import ArrayObject, stack
from abtem.core import config
from abtem.core.axes import (
    AxisMetadata,
    LinearAxis,
    NonLinearAxis,
    RealSpaceAxis,
    ReciprocalSpaceAxis,
    ScaleAxis,
    ScanAxis,
)
from abtem.core.backend import asnumpy, cp, get_array_module, get_ndimage_module
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength
from abtem.core.fft import fft_crop, fft_interpolate
from abtem.core.grid import (
    adjusted_gpts,
    polar_spatial_frequencies,
    spatial_frequencies,
)
from abtem.core.units import _get_conversion_factor, _validate_units
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    is_broadcastable,
    label_to_index,
)
from abtem.distributions import BaseDistribution
from abtem.noise import NoiseTransform, ScanNoiseTransform
from abtem.visualize.visualizations import Visualization
from abtem.visualize.widgets import ImageGUI, LinesGUI, ScatterGUI

# Enables CuPy-accelerated functions if it is available.
if cp is not None:
    from abtem.core._cuda import interpolate_bilinear as interpolate_bilinear_cuda
    from abtem.core._cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
else:
    sum_run_length_encoded_cuda = None
    interpolate_bilinear_cuda = None


if TYPE_CHECKING:
    from abtem.waves import BaseWaves

# Ensures that `Measurement` objects created by `Measurement` objects retain their type (e.g. `Images`).
T = TypeVar("T", bound="BaseMeasurements")


def _scanned_measurement_type(
    measurement: BaseMeasurements | BaseWaves,
) -> Type["BaseMeasurements"]:
    if len(_scan_shape(measurement)) == 1:
        return RealSpaceLineProfiles

    elif len(_scan_shape(measurement)) == 2:
        return Images

    else:
        return MeasurementsEnsemble


def _bin_extent(n):
    if n % 2 == 0:
        return -n // 2 - 0.5, n // 2 - 0.5
    else:
        return -n // 2 + 0.5, n // 2 + 0.5


# def _move_scan_axes_to_back(measurement):
#     ensemble_axes = tuple(range(len(measurement.ensemble_shape)))
#
#     source = _scan_axes(measurement)
#     destination = tuple(range(len(ensemble_axes) - len(source), len(ensemble_axes)))
#
#     if source != destination:
#         measurement = moveaxis(measurement, source, destination)
#     return measurement


def _reduced_scanned_images_or_line_profiles(
    new_array,
    old_measurement,
    metadata=None,
) -> RealSpaceLineProfiles | Images | MeasurementsEnsemble | np.ndarray:
    if metadata is None:
        metadata = {}

    metadata = {**old_measurement.metadata, **metadata}

    ensemble_axes = tuple(range(len(old_measurement.ensemble_shape)))

    source = _scan_axes(old_measurement)
    destination = tuple(range(len(ensemble_axes) - len(source), len(ensemble_axes)))
    scan_axes_metadata = [old_measurement.ensemble_axes_metadata[i] for i in source]

    ensemble_axes_metadata = [
        m
        for i, m in enumerate(old_measurement.ensemble_axes_metadata)
        if i not in source
    ]

    if source != destination:
        xp = get_array_module(new_array)
        if old_measurement.is_lazy:
            new_array = da.moveaxis(new_array, source, destination)
        else:
            new_array = xp.moveaxis(new_array, source, destination)

    if len(scan_axes_metadata) == 1:
        sampling = scan_axes_metadata[-1].sampling

        return RealSpaceLineProfiles(
            new_array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    elif len(scan_axes_metadata) == 2:
        sampling = (
            scan_axes_metadata[-2].sampling,
            scan_axes_metadata[-1].sampling,
        )

        images = Images(
            new_array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

        return images
    elif _scanned_measurement_type(old_measurement) is MeasurementsEnsemble:
        ensemble_axes_metadata = old_measurement.ensemble_axes_metadata

        measurement_ensemble = MeasurementsEnsemble(
            new_array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )
        return measurement_ensemble
    else:
        return new_array


def _scan_axes(measurement):
    num_scan_axes = 0
    scan_axes = ()
    for i, axis in enumerate(measurement.ensemble_axes_metadata):
        if num_scan_axes == 2:
            break

        if isinstance(axis, ScanAxis) and axis._main is True:
            scan_axes += (i,)
            num_scan_axes += 1

    scan_axes = scan_axes[-2:]

    return scan_axes


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
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    inner: float,
    outer: float,
    offset: tuple[float, float] = (0.0, 0.0),
    fftshift: bool = False,
    xp=np,
) -> np.ndarray | list[np.ndarray]:
    kx, ky = spatial_frequencies(
        gpts, (1 / sampling[0] / gpts[0], 1 / sampling[1] / gpts[1]), False, xp
    )

    k2 = kx[:, None] ** 2 + ky[None] ** 2

    bins = (k2 >= inner**2) & (k2 < outer**2)

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
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    inner: float,
    outer: float,
    nbins_radial: int,
    nbins_azimuthal: int,
    rotation: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    fftshift: bool = False,
    return_indices: bool = False,
) -> np.ndarray | list[np.ndarray]:
    alpha, phi = polar_spatial_frequencies(
        gpts, sampling=(1 / sampling[0] / gpts[0], 1 / sampling[1] / gpts[1])
    )
    phi = (phi - rotation) % (2 * np.pi)

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


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _sum_run_length_encoded(array, result, separators):
    for x in prange(result.shape[1]):  # pylint: disable=not-an-iterable
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
    output = xp.zeros((array.shape[0], positions.shape[0]), dtype=array.dtype)

    for i in range(array.shape[0]):
        map_coordinates(array[i], positions.T, output=output[i], order=order, **kwargs)

    output = output.reshape(old_shape[:-2] + positions_shape[:-1])
    return output


class BaseMeasurements(ArrayObject, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    Base class for all measurement types.

    Parameters
    ----------
    array : ndarray
        Array containing data of type `float` or ´complex´.
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    """

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        ensemble_axes_metadata: list[AxisMetadata],
        metadata: dict,
    ):
        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list:
        """List of AxisMetadata of the base axes."""
        pass

    @property
    def metadata(self) -> dict:
        """Metadata describing the measurement."""
        return self._metadata

    def _get_from_metadata(self, key):
        if key not in self.metadata.keys():
            raise RuntimeError(f"{key} not in measurement metadata.")
        return self.metadata[key]

    def _check_is_complex(self):
        if not np.iscomplexobj(self.array):
            raise RuntimeError("Function not implemented for non-complex measurements.")

    def real(self) -> T:
        """Returns the real part of a complex-valued measurement."""
        self._check_is_complex()
        self.metadata["label"] = "real"
        self.metadata["units"] = "arb. unit"
        return self._apply_element_wise_func(get_array_module(self.array).real)

    def imag(self) -> T:
        """Returns the imaginary part of a complex-valued measurement."""
        self._check_is_complex()
        self.metadata["label"] = "imaginary"
        self.metadata["units"] = "arb. unit"
        return self._apply_element_wise_func(get_array_module(self.array).imag)

    def phase(self) -> T:
        """Calculates the phase of a complex-valued measurement."""
        self._check_is_complex()
        self.metadata["label"] = "phase"
        self.metadata["units"] = "rad."
        return self._apply_element_wise_func(get_array_module(self.array).angle)

    def abs(self) -> T:
        """Calculates the absolute value of a complex-valued measurement."""
        # self._check_is_complex()
        self.metadata["label"] = "amplitude"
        self.metadata["units"] = "arb. unit"
        return self._apply_element_wise_func(get_array_module(self.array).abs)

    def intensity(self) -> T:
        """Calculates the squared norm of a complex-valued measurement."""
        self._check_is_complex()
        self.metadata["label"] = "intensity"
        self.metadata["units"] = "arb. unit"
        return self._apply_element_wise_func(abs2)

    def relative_difference(
        self, other: BaseMeasurements, min_relative_tol: float = 0.0
    ) -> T:
        """
        Calculates the relative difference with respect to another compatible measurement.

        Parameters
        ----------
        other : BaseMeasurements
            Measurement to which the difference is calculated.
        min_relative_tol : float
            Avoids division by zero errors by defining a minimum value of the divisor in the relative difference.

        Returns
        -------
        difference : BaseMeasurements
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
        difference.metadata["tex_units"] = "$\%$"
        return difference

    def normalize_ensemble(self, scale: str = "max", shift: str = "mean"):
        """
        Normalize the ensemble by shifting ad scaling each member.

        Parameters
        ----------
        scale : {'max', 'min', 'sum', 'mean', 'ptp'}
        shift : {'max', 'min', 'sum', 'mean', 'ptp'}

        Returns
        -------
        normalized_measurements : BaseMeasurements or subclass of _BaseMeasurement
        """
        if shift != "none":
            array = self.array - getattr(np, shift)(self.array, axis=-1, keepdims=True)
        else:
            array = self.array

        array = array / getattr(np, scale)(self.array, axis=-1, keepdims=True)
        kwargs = self._copy_kwargs(exclude=("array",))
        return self.__class__(array, **kwargs)

    @classmethod
    @abstractmethod
    def from_array_and_metadata(
        cls, array: np.ndarray, axes_metadata: list[AxisMetadata], metadata: dict
    ) -> "T":
        pass

    def reduce_ensemble(self) -> "T":
        """Calculates the mean of an ensemble measurement (e.g. of frozen phonon configurations)."""
        axis = tuple(
            i
            for i, axis in enumerate(self.axes_metadata)
            if hasattr(axis, "_ensemble_mean") and axis._ensemble_mean
        )

        if len(axis) == 0:
            return self

        return self.mean(axis=axis)

    def _apply_element_wise_func(self, func: callable) -> "T":
        d = self._copy_kwargs(exclude=("array",))
        d["array"] = func(self.array)
        return self.__class__(**d)

    @property
    @abstractmethod
    def _area_per_pixel(self):
        pass

    def poisson_noise(
        self,
        dose_per_area: float | Sequence[float] | None = None,
        total_dose: float | Sequence[float] | None = None,
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
        noisy_measurement : BaseMeasurements or subclass of _BaseMeasurement
            The noisy measurement.
        """

        wrong_dose_error = RuntimeError(
            "Provide one of 'dose_per_area' or 'total_dose'."
        )

        if (dose_per_area is not None) and (total_dose is not None):
            raise RuntimeError("Provide one of 'dose_per_area' or 'total_dose'.")

        dose_axes_metadata = None
        if dose_per_area is not None:
            if np.isscalar(dose_per_area):
                total_dose = self._area_per_pixel * dose_per_area
            else:
                total_dose = self._area_per_pixel * np.array(
                    dose_per_area, dtype=np.float32
                )
                dose_axes_metadata = NonLinearAxis(
                    label="Dose", values=tuple(dose_per_area), units="e/Å^2"
                )

        elif total_dose is not None:
            if dose_per_area is not None:
                raise wrong_dose_error

        else:
            raise wrong_dose_error

        # xp = get_array_module(self.array)

        total_dose = np.array(total_dose, dtype=np.float32)

        transform = NoiseTransform(total_dose, samples, seeds=seed)
        measurement = self.apply_transform(transform)

        if isinstance(transform.dose, BaseDistribution) and dose_axes_metadata:
            measurement = measurement.set_ensemble_axes_metadata(
                dose_axes_metadata, axis=0
            )

        return measurement

    def _scale_axis_from_metadata(self):
        return ScaleAxis(
            label=self.metadata.get("label", ""),
            units=self.metadata.get("units", None),
            tex_label=None,
        )

    def to_measurement_ensemble(self):
        return MeasurementsEnsemble(
            array=self.array,
            ensemble_axes_metadata=self.axes_metadata,
            metadata=self.metadata,
        )

    @abstractmethod
    def show(self, *args, **kwargs):
        """Documented in subclasses"""
        pass


class MeasurementsEnsemble(BaseMeasurements):
    _base_dims = 0

    def __init__(
        self,
        array: np.ndarray,
        ensemble_axes_metadata: list[AxisMetadata],
        metadata: dict = None,
    ):
        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def _area_per_pixel(self):
        raise RuntimeError("Cannot infer pixel area from metadata.")

    @property
    def base_axes_metadata(self):
        return []

    @classmethod
    def from_array_and_metadata(
        cls, array: np.ndarray, axes_metadata: list[AxisMetadata], metadata: dict
    ) -> "T":
        return cls(array, axes_metadata, metadata)

    def show(
        self,
        type: str = "lines",
        ax: Axes = None,
        power: float = 1.0,
        common_scale: bool = False,
        explode: bool | Sequence[int] = (),
        overlay: bool | Sequence[int] = (),
        figsize: tuple[int, int] = None,
        title: bool | str = True,
        units: str = None,
        interact: bool = False,
        display: bool = True,
        **kwargs,
    ) -> Visualization:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for exploded plots.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The size and padding of the colorbars may be adjusted using the
            `set_cbar_size` and `set_cbar_padding` methods.
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. If the measurement is complex the colormap
            must be one of 'hsv' or 'hsluv'.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        power : float
            Show image on a power scale.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown. May be given as a sequence of axis indices to create a grid of images from
            the specified axes. The default is determined by the axis metadata.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Set the column title of the images. If True is given instead of a string the title will be given by the
            value corresponding to the "name" key of the axes metadata dictionary, if this item exists.
        units : str
            The units used for the x and y axes. The given units must be compatible with the axes of the images.
        interact : bool
            If True, create an interactive visualization. This requires enabling the ipympl Matplotlib backend.
        display : bool, optional
            If True (default) the figure is displayed immediately.

        Returns
        -------
        measurement_visualization_2d : VisualizationImshow
        """
        pass
        # if not interact:
        #     self.compute()

        # scale_axis = self._scale_axis_from_metadata()

        # # base_axes_metadata = self._plot_base_axes_metadata(units)

        # array = self.array

        # # raise RuntimeError("Cannot infer pixel area from metadata.")

        # #if display_axes != (-2, -1):
        # #    array = np.moveaxis(self.array, source=display_axes, destination=(-2, -1))

        # display_axes = normalize_axes(display_axes, self.shape)

        # # base_axes_metadata = [self.axes_metadata[i] for i in display_axes]
        # # ensemble_axes_metadata = [
        # #     self.axes_metadata[i]
        # #     for i in range(len(self.shape))
        # #     if i not in display_axes
        # # ]

        # artist_type = LinesArtist

        # visualization = Visualization(
        #     self,
        #     ax=ax,
        #     artist_type=artist_type,
        #     power=power,
        #     common_scale=common_scale,
        #     explode=explode,
        #     overlay=overlay,
        #     figsize=figsize,
        #     interact=interact,
        #     title=title,
        #     **kwargs,
        # )

        # return visualization

    # def show(
    #     self,
    #     ax: Axes = None,
    #     common_scale: bool = True,
    #     explode: bool | Sequence[int] = None,
    #     overlay: bool | Sequence[int] = None,
    #     figsize: tuple[int, int] = None,
    #     title: str = None,
    #     units: str = None,
    #     legend: bool = False,
    #     interact: bool = False,
    #     display: bool = True,
    #     **kwargs,
    # ):
    #     # if not interact:
    #     #     self.compute()
    #
    #     visualization = VisualizationLines(
    #         array=self.array,
    #         coordinate_axes=self.ensemble_axes_metadata[-1:],
    #         scale_axis=self._scale_axis_from_metadata(),
    #         ensemble_axes=self.ensemble_axes_metadata[:-1],
    #         ax=ax,
    #         common_scale=common_scale,
    #         explode=explode,
    #         overlay=overlay,
    #         figsize=figsize,
    #         interact=interact,
    #         title=title,
    #         **kwargs,
    #     )
    #
    #     if not display and not interact:
    #         plt.close()
    #
    #     if interact and display:
    #         from IPython.display import display as ipython_display
    #
    #         ipython_display(visualization.layout_widgets())
    #
    #     return visualization


class _BaseMeasurement2D(BaseMeasurements):
    _base_dims = 2

    @abstractmethod
    def _get_1d_equivalent(self):
        pass

    @property
    @abstractmethod
    def sampling(self) -> tuple[float, float]:
        """
        Sampling of the measurements in `x` and `y` [Å] or [1/Å].
        """
        pass

    @property
    @abstractmethod
    def extent(self) -> tuple[float, float]:
        """
        Extent of measurements in `x` and `y` [Å] or [1/Å].
        """
        pass

    @property
    @abstractmethod
    def offset(self) -> tuple[float, float]:
        """
        The offset of the origin of the measurement coordinates [Å] or [1/Å].
        """
        pass

    def interpolate_line(
        self,
        start: tuple[float, float] | Atom = None,
        end: tuple[float, float] | Atom = None,
        sampling: float = None,
        gpts: int = None,
        width: float = 0.0,
        margin: float = 0.0,
        order: int = 3,
        endpoint: bool = False,
        fractional: bool = False,
    ):
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
        fractional : bool
            If True, use fractional coordinates with respect to the extent of the measurement.

        Returns
        -------
        line_profiles : RealSpaceLineProfiles
            The interpolated line(s).
        """
        from abtem.scan import LineScan

        # if self.is_complex:
        #    raise NotImplementedError

        if (sampling is None) and (gpts is None):
            sampling = min(self.sampling)

        xp = get_array_module(self.array)

        if start is None:
            start = (0.0, 0.0)

        if end is None and fractional:
            end = (0.0, 1.0)
        elif end is None:
            end = (0.0, self.extent[0])

        if fractional:
            extent = self.extent
        else:
            extent = None

        scan = LineScan(
            start=start,
            end=end,
            gpts=gpts,
            sampling=sampling,
            endpoint=endpoint,
            potential=extent,
            fractional=fractional,
        )

        if margin != 0.0:
            scan.add_margin(margin)

        positions = xp.asarray(
            (scan.get_positions(lazy=False) - self.offset) / self.sampling
        )

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
            base_axes = tuple(range(len(self.base_shape)))
            array = self.array.map_blocks(
                _interpolate_stack,
                positions=positions,
                mode="wrap",
                order=order,
                drop_axis=base_axes,
                new_axis=base_axes[0],
                chunks=self.array.chunks[:-2] + (positions.shape[0],),
                meta=xp.array((), dtype=np.float32),
            )
        else:
            array = _interpolate_stack(self.array, positions, mode="wrap", order=order)

        metadata = copy.copy(self.metadata)
        metadata.update(scan.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        if width:
            array = array.mean(-1)
            metadata["width"] = width

        ensemble_axes_metadata = self.ensemble_axes_metadata

        return self._get_1d_equivalent()(
            array=array,
            sampling=scan.sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    def gaussian_filter(
        self,
        sigma: float | tuple[float, float],
        boundary: str = "periodic",
        cval: float = 0.0,
    ):
        """
        Apply 2D gaussian filter to measurements.

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
                min(int(np.ceil(4.0 * s)), n) for s, n in zip(sigma, self.shape)
            )

            array = self.array.map_overlap(
                gaussian_filter,
                sigma=sigma,
                boundary=boundary,
                mode=mode,
                cval=cval,
                depth=depth,
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = gaussian_filter(self.array, sigma=sigma, mode=mode, cval=cval)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def interpolate_line_at_position(
        self,
        center: tuple[float, float] | Atom,
        angle: float,
        extent: float,
        gpts: int = None,
        sampling: float = None,
        width: float = 0.0,
        order: int = 3,
        endpoint: bool = True,
    ):
        """
        Interpolate image(s) along a line centered at a specified position.

        Parameters
        ----------
        center : two float
            Center position of the line [Å]. May be given as an Atom.
        angle : float
            Angle of the line [deg.].
        extent : float
            Extent of the line [Å].
        gpts : int
            Number of grid points along the line.
        sampling : float
            Sampling of grid points along the line [Å].
        width : float, optional
            The interpolation will be averaged across a perpendicular distance equal to this width.
        order : int, optional
            The spline interpolation order.
        endpoint : bool
            Sets whether the ending position is included or not.

        Returns
        -------
        line_profiles : RealSpaceLineProfiles or ReciprocalSpaceProfiles
            The interpolated line(s).
        """

        from abtem.scan import LineScan

        scan = LineScan.at_position(center=center, extent=extent, angle=angle)

        return self.interpolate_line(
            scan.start,
            scan.end,
            gpts=gpts,
            sampling=sampling,
            width=width,
            order=order,
            endpoint=endpoint,
        )

    def show(
        self,
        ax: Axes = None,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_color_scale: bool = False,
        explode: bool | Sequence[int] = (),
        overlay: bool | Sequence[int] = (),
        figsize: tuple[int, int] = None,
        title: bool | str = True,
        units: str = None,
        interact: bool = False,
        display: bool = True,
        **kwargs,
    ) -> Visualization:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for exploded plots.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The size and padding of the colorbars may be adjusted using the
            `set_cbar_size` and `set_cbar_padding` methods.
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. If the measurement is complex the colormap
            must be one of 'hsv' or 'hsluv'.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        power : float
            Show image on a power scale.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown. May be given as a sequence of axis indices to create a grid of images from
            the specified axes. The default is determined by the axis metadata.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Set the column title of the images. If True is given instead of a string the title will be given by the
            value corresponding to the "name" key of the axes metadata dictionary, if this item exists.
        units : str
            The units used for the x and y axes. The given units must be compatible with the axes of the images.
        interact : bool
            If True, create an interactive visualization. This requires enabling the ipympl Matplotlib backend.
        display : bool, optional
            If True (default) the figure is displayed immediately.

        Returns
        -------
        measurement_visualization_2d : VisualizationImshow
        """

        visualization = Visualization(
            measurement=self,
            ax=ax,
            common_scale=common_color_scale,
            figsize=figsize,
            title=title,
            aspect=True,
            share_x=True,
            share_y=True,
            explode=explode,
            overlay=overlay,
            interactive=not interact and display,
            value_limits=(vmin, vmax),
            power=power,
            cmap=cmap,
            cbar=cbar,
            units=units,
            **kwargs,
        )

        if interact:
            gui = visualization.interact(ImageGUI, display=display)

        return visualization


class Images(_BaseMeasurement2D):
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
        List of metadata associated with the ensemble axes. The length and item order must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.
    """

    def __init__(
        self,
        array: da.core.Array | np.array,
        sampling: float | tuple[float, float],
        ensemble_axes_metadata: list[AxisMetadata] = None,
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
        )

    @classmethod
    def from_array_and_metadata(
        cls,
        array: np.ndarray,
        axes_metadata: list[AxisMetadata],
        metadata: dict = None,
    ) -> "Images":
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
        x_axis, y_axis = axes_metadata[-2:]

        if isinstance(x_axis, LinearAxis) and isinstance(y_axis, LinearAxis):
            sampling = (x_axis.sampling, y_axis.sampling)
        else:
            raise RuntimeError()

        return cls(
            array,
            sampling=sampling,
            ensemble_axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    def _get_1d_equivalent(self):
        return RealSpaceLineProfiles

    @property
    def _area_per_pixel(self):
        return np.prod(self.sampling)

    @property
    def sampling(self) -> tuple[float, float]:
        return self._sampling

    @property
    def offset(self) -> tuple[float, float]:
        return 0.0, 0.0

    @property
    def extent(self) -> tuple[float, float]:
        return (
            self.sampling[0] * self.base_shape[0],
            self.sampling[1] * self.base_shape[1],
        )

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of pixels in `x` and `y` [Å]."""
        x = np.linspace(0.0, self.shape[-2] * self.sampling[0], self.shape[-2])
        y = np.linspace(0.0, self.shape[-1] * self.sampling[1], self.shape[-1])
        return x, y

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="Å", tex_label="$x$"
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="Å", tex_label="$y$"
            ),
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
        kwargs["metadata"].update({"label": "iCOM", "units": "arb. unit"})
        return self.__class__(**kwargs)

    def crop(
        self, extent: tuple[float, float], offset: tuple[float, float] = (0.0, 0.0)
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
        sampling: float | tuple[float, float] = None,
        gpts: int | tuple[int, int] = None,
        method: str = "fft",
        boundary: str = "periodic",
        order: int = 3,
        normalization: str = "values",
        cval: float = 0.0,
    ) -> Images:
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

        def _interpolate_spline(array, old_gpts, new_gpts, pad_mode, order, cval):
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
                    _interpolate_spline,
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
                array = _interpolate_spline(
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

    def tile(self, repetitions: tuple[int, int]) -> Images:
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

    def scan_noise(
        self,
        dwell_time: float,
        flyback_time: float,
        rms_power: float,
        max_frequency: float = 500.0,
        num_components: int = 200,
        seed: int = None,
    ):
        """
        Apply scan noise to images.

        Parameters
        ----------
        dwell_time : float
            Dwell time of the beam [s].
        flyback_time : float
            Flyback time of the beam [s].
        rms_power : float
            RMS power of the scan noise [V].
        max_frequency : float
            Maximum frequency of the scan noise [1/Å].
        """
        transform = ScanNoiseTransform(
            dwell_time=dwell_time,
            flyback_time=flyback_time,
            rms_power=rms_power,
            max_frequency=max_frequency,
            num_components=num_components,
            seeds=seed,
        )
        return self.apply_transform(transform)

    def diffractograms(self) -> DiffractionPatterns:
        """
        Calculate diffractograms (i.e. power spectra) from image(s).

        Returns
        -------
        diffractograms : DiffractionPatterns
            Diffractograms of image(s).
        """
        xp = get_array_module(self.array)

        def _diffractograms(array):
            array = xp.fft.fft2(array)
            return xp.fft.fftshift(xp.abs(array), axes=(-2, -1))

        if self.is_lazy:
            array = self.array.rechunk(
                chunks=self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],))
            )
            array = array.map_blocks(
                _diffractograms, meta=xp.array((), dtype=xp.float32)
            )
        else:
            array = _diffractograms(self.array)

        sampling = 1 / self.extent[0], 1 / self.extent[1]
        return DiffractionPatterns(
            array=array,
            sampling=sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def _plot_base_axes_metadata(self, units: str = None):
        return self.base_axes_metadata


class _BaseMeasurement1D(BaseMeasurements):
    _base_dims = 1

    def __init__(
        self,
        array: np.ndarray,
        sampling: float = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        self._sampling = sampling

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def _area_per_pixel(self):
        raise RuntimeError("Cannot infer pixel area from metadata.")

    @classmethod
    def from_array_and_metadata(
        cls, array: np.ndarray, axes_metadata: list[AxisMetadata], metadata: dict = None
    ) -> "T":
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
        x_axis = axes_metadata[-1]
        if isinstance(x_axis, LinearAxis):
            sampling = x_axis.sampling
        else:
            raise RuntimeError()

        axes_metadata = axes_metadata[:-1]
        return cls(
            array,
            sampling=sampling,
            ensemble_axes_metadata=axes_metadata,
            metadata=metadata,
        )

    @property
    def extent(self) -> float:
        """
        Extent of measurements [Å] or [1/Å].
        """
        return self.sampling * self.shape[-1]

    @property
    def sampling(self) -> float:
        """
        Extent of measurements [Å] or [1/Å].
        """
        return self._sampling

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list[RealSpaceAxis | ReciprocalSpaceAxis]:
        pass

    def _line_scan(self, sampling=None):
        start, end = self.metadata["start"], self.metadata["end"]
        from abtem.scan import LineScan

        return LineScan(start=start, end=end, sampling=sampling)

    def _add_to_visualization(self, *args, **kwargs):
        if not all(key in self.metadata for key in ("start", "end")):
            raise RuntimeError(
                "The metadata does not contain the keys 'start' and 'end'"
            )

        if "width" in self.metadata:
            kwargs["width"] = self.metadata["width"]

        self._line_scan().add_to_axes(*args, **kwargs)

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

        def _calculate_widths(array, sampling, height):
            xp = get_array_module(array)
            array = array - xp.max(array, axis=-1, keepdims=True) * height

            widths = xp.zeros(array.shape[:-1], dtype=np.float32)
            for i in np.ndindex(array.shape[:-1]):
                zero_crossings = xp.where(xp.diff(xp.sign(array[i]), axis=-1))[0]
                left, right = zero_crossings[0], zero_crossings[-1]
                widths[i] = (right - left) * sampling

            return widths

        if self.is_lazy:
            return self.array.map_blocks(
                _calculate_widths,
                drop_axis=(len(self.array.shape) - 1,),
                dtype=np.float32,
                sampling=self.sampling,
                height=height,
            )
        else:
            return _calculate_widths(self.array, self.sampling, height)

    def interpolate(
        self,
        sampling: float = None,
        gpts: int = None,
        order: int = 3,
        endpoint: bool = False,
    ) -> T:
        """
        Interpolate line profile(s) producing equivalent line profile(s) with a different sampling. Either 'sampling' or
        'gpts' must be provided (but not both).

        Parameters
        ----------
        sampling : float, optional
            Sampling of line profiles after interpolation [Å].
        gpts : int, optional
            Number of grid points of line profiles after interpolation. Do not use if 'sampling' is used.
        order : int, optional
            The order of the spline interpolation (default is 3). The order has to be in the range 0-5.
        endpoint : bool, optional
            If True, end is the last position. Otherwise, it is not included. Default is False.

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

        def _interpolate(array, gpts, endpoint, order):
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
                _interpolate,
                gpts=gpts,
                endpoint=endpoint,
                order=order,
                chunks=self.array.chunks[:-1] + (gpts,),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = _interpolate(self.array, gpts, endpoint, order)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["sampling"] = sampling
        return self.__class__(**kwargs)

    def show(
        self,
        ax: Axes = None,
        common_scale: bool = True,
        explode: bool | Sequence[int] = False,
        overlay: bool | Sequence[int] = None,
        figsize: tuple[int, int] = None,
        title: str = True,
        units: str = None,
        legend: bool = False,
        interact: bool = False,
        display: bool = True,
        **kwargs,
    ) -> Visualization:
        """
        Show the reciprocal-space line profile(s) using matplotlib.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If given the plots are added to the Axes. This is not available for image grids.
        common_scale : bool
            If True all plots are shown with a common y-axis. Default is False.
        explode : bool or sequence of bool, optional
            If True, a grid of plots is created for all the items of the last two ensemble axes. If False, only the
            one plot is created. May be given as a sequence of axis indices to create a grid of plots from the specified
            axes. The default is determined by the axis metadata.
        overlay : bool or sequence of int, optional
            If True, all line profiles in the ensemble are shown in a single plot. If False, only the first ensemble
            item is shown. May be given as a sequence of axis indices to specify which line profiles in the ensemble to
            show together. The default is determined by the axis metadata.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
        title : bool or str, optional
            Set the column title of the plots. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the axes metadata dictionary, if this item exists.
        legend : bool
            Add a legend to the plot. The labels will be derived from
        units : str, optional
            The units used for the x-axis. The given units must be compatible.
        interact : bool
            If True, create an interactive visualization. This requires enabling the ipympl Matplotlib backend.
        display : bool, optional
            If True (default) the figure is displayed immediately.

        Returns
        -------
        visualization : Visualization
        """

        if overlay is None and explode is False:
            overlay = True
        elif overlay is False or overlay is None:
            overlay = ()

        visualization = Visualization(
            measurement=self,
            ax=ax,
            figsize=figsize,
            title=title,
            aspect=False,
            share_x=True,
            share_y=common_scale,
            explode=explode,
            overlay=overlay,
            interactive=not interact and display,
            legend=legend,
            common_scale=common_scale,
            **kwargs,
        )

        if interact:
            gui = visualization.interact(LinesGUI, display=display)

        if common_scale is False and visualization._explode:
            visualization.axes.set_sizes(padding=0.8)

        return visualization


class RealSpaceLineProfiles(_BaseMeasurement1D):
    """
    A collection of real-space line profile(s).

    Parameters
    ----------
    array : np.ndarray
        1D or greater array containing data of type `float` or ´complex´.
    sampling : float
        Sampling of line profiles [Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        List of metadata associated with the ensemble axes. The length and item order must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        sampling: float = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        super().__init__(
            array=array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def base_axes_metadata(self) -> list[RealSpaceAxis]:
        return [
            RealSpaceAxis(label="r", sampling=self.sampling, units="Å", tex_label="$r$")
        ]

    def tile(self, repetitions: int) -> "RealSpaceLineProfiles":
        """
        Tile line profiles(s).

        Parameters
        ----------
        repetitions : int
            The number of repetitions of the line profiles.

        Returns
        -------
        tiled_line_profiles : RealSpaceLineProfiles
            The tiled line profiles(s).
        """

        kwargs = self._copy_kwargs(exclude=("array",))
        xp = get_array_module(self.array)
        reps = (1,) * (len(self.array.shape) - 1) + (repetitions,)

        if self.is_lazy:
            kwargs["array"] = da.tile(self.array, reps)
        else:
            kwargs["array"] = xp.tile(self.array, reps)

        return self.__class__(**kwargs)

    def _plot_extent(self, units=None):
        scale = _get_conversion_factor(units, "Å")
        return [0, self.extent * scale]

    # def _plot_extent(self, units=None):
    #     scale = {"Å": 1, "nm": 0.1}[_validate_real_space_units(units)]
    #     return [0, self.extent * scale]

    def _plot_x_label(self, units=None):
        return f"x [{_validate_units(units, 'Å')}]"

    def _plot_y_label(self, units=None):
        return f"y [{_validate_units(units, 'Å')}]"


class ReciprocalSpaceLineProfiles(_BaseMeasurement1D):
    """
    A collection of reciprocal-space line profile(s).

    Parameters
    ----------
    array : np.ndarray
        1D or greater array containing data of type `float` or ´complex´.
    sampling : float
        Sampling of line profiles [1 / Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        List of metadata associated with the ensemble axes. The length and item order must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        sampling: float = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        super().__init__(
            array=array,
            sampling=sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [
            ReciprocalSpaceAxis(
                label="k", sampling=self.sampling, units="1/Å", tex_label="$k$"
            )
        ]

    @property
    def angular_extent(self):
        """Extent of line profiles given as scattering angels [mrad]."""
        wavelength = energy2wavelength(self._get_from_metadata("energy"))
        return self.extent * wavelength * 1e3

    def _plot_x_label(self, units=None):
        return f"x [{_validate_units(units, '1/Å')}]"

    def _plot_y_label(self, units=None):
        return f"y [{_validate_units(units, '1/Å')}]"

    def _plot_extent(self, units=None):
        if units is None:
            units = "1/Å"

        if units == "mrad":
            return [0, self.angular_extent]
        elif units == "1/Å":
            return [0, self.extent]


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
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
    old_angular_sampling: tuple[float, float],
    new_angular_sampling: tuple[float, float],
    xp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _gaussian_source_size(measurements, sigma: float | tuple[float, float]):
    if len(_scan_axes(measurements)) < 2:
        raise RuntimeError(
            "Gaussian source size not implemented for diffraction patterns with less than two scan axes."
        )

    if np.isscalar(sigma):
        sigma = (sigma,) * 2

    xp = get_array_module(measurements.array)
    gaussian_filter = get_ndimage_module(measurements._array).gaussian_filter

    ensemble_axes = tuple(range(len(measurements.ensemble_shape)))
    padded_sigma = ()
    depth = ()
    i = 0
    for axis, n in zip(ensemble_axes, measurements.ensemble_shape):
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


def _infer_lines(b, H, W, out_H, out_W, kH, kW):
    target_size = 2**17
    line_size = b * (H * W // out_H + kH * kW * out_W)
    target_lines = target_size // line_size

    if target_lines < out_H:
        lines = 1
        while True:
            next_lines = lines * 2
            if next_lines > target_lines:
                break
            lines = next_lines
    else:
        lines = out_H

    return lines


def _interpolate_bilinear(x, v, u, vw, uw):
    B, H, W = x.shape
    out_H, out_W = v.shape

    # Interpolation is done by each output panel (i.e. multi lines)
    # in order to better utilize CPU cache memory.
    lines = _infer_lines(B, H, W, out_H, out_W, 2, 2)

    vcol = np.empty((2, lines, out_W), dtype=v.dtype)
    ucol = np.empty((2, lines, out_W), dtype=u.dtype)
    wcol = np.empty((2, 2, lines, out_W), dtype=x.dtype)

    y = np.empty((B, out_H * out_W), dtype=x.dtype)

    for i in range(0, out_H, lines):
        n = min(lines, out_H - i)
        vcol = vcol[:, :n]
        ucol = ucol[:, :n]
        wcol = wcol[:, :, :n]
        i_end = i + n

        # indices
        vcol[0] = v[i:i_end]
        ucol[0] = u[i:i_end]
        np.add(vcol[0], 1, out=vcol[1])
        np.add(ucol[0], 1, out=ucol[1])
        np.minimum(vcol[1], H - 1, out=vcol[1])
        np.minimum(ucol[1], W - 1, out=ucol[1])

        wcol[0, 1] = uw[i:i_end]
        np.subtract(1, wcol[0, 1], out=wcol[0, 0])
        np.multiply(wcol[0], vw[i:i_end], out=wcol[1])
        wcol[0] -= wcol[1]

        # packing to the panel whose shape is (B, C, 2, 2, l, out_W)
        panel = x[:, vcol[:, None], ucol[None, :]]

        # interpolation
        panel = panel.reshape((B, 4, n * out_W))
        weights = wcol.reshape((4, n * out_W))
        iout = i * out_W
        iout_end = i_end * out_W
        np.einsum("ijk,jk->ik", panel, weights, out=y[:, iout:iout_end])
        del panel, weights

    return y.reshape((B, out_H, out_W))


class DiffractionPatterns(_BaseMeasurement2D):
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
        List of metadata associated with the ensemble axes. The length and item order must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.
    """

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        sampling: float | tuple[float, float],
        fftshift: bool = False,
        ensemble_axes_metadata: list[AxisMetadata] = None,
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
        )

    @property
    def _area_per_pixel(self):
        return _scan_area_per_pixel(self)

    @classmethod
    def from_array_and_metadata(
        cls,
        array: np.ndarray,
        axes_metadata: list[AxisMetadata],
        metadata: dict = None,
    ):
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
        x_axis, y_axis = axes_metadata[-2:]

        if isinstance(x_axis, ReciprocalSpaceAxis) and isinstance(
            y_axis, ReciprocalSpaceAxis
        ):
            sampling = (x_axis.sampling, y_axis.sampling)
            fftshift = x_axis.fftshift
        else:
            raise RuntimeError()

        axes_metadata = axes_metadata[:-2]
        return cls(
            array,
            sampling=sampling,
            ensemble_axes_metadata=axes_metadata,
            fftshift=fftshift,
            metadata=metadata,
        )

    def _get_1d_equivalent(self):
        return ReciprocalSpaceLineProfiles

    @property
    def base_axes_metadata(self):
        limits = self.limits
        return [
            ReciprocalSpaceAxis(
                sampling=self.sampling[0],
                offset=limits[0][0],
                label="kx",
                units="1/Å",
                fftshift=self.fftshift,
                tex_label="$k_x$",
            ),
            ReciprocalSpaceAxis(
                sampling=self.sampling[1],
                offset=limits[1][0],
                label="ky",
                units="1/Å",
                fftshift=self.fftshift,
                tex_label="$k_y$",
            ),
        ]

    def tile_scan(self, repetitions: tuple[int, int]) -> DiffractionPatterns:
        """
        Tile the scan axes of the diffraction patterns. The diffraction patterns must have one or more scan axes.

        Parameters
        ----------
        repetitions : two int
            The number of repetitions of the scan positions along the `x`- and `y`-axis, respectively.

        Returns
        -------
        tiled_diffraction_patterns : DiffractionPatterns
            The tiled diffraction patterns.
        """

        scan_axes = _scan_axes(self)

        if len(scan_axes) != 2:
            raise NotImplementedError

        xp = get_array_module(self.array)

        tiling = ()
        j = 0
        for i in range(len(self.shape)):
            if i in scan_axes:
                tiling += (repetitions[j],)
                j += 1
            else:
                tiling += (1,)

        array = xp.tile(self.array, tiling)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def index_diffraction_spots(
        self,
        cell: Cell | float | tuple[float, float, float],
        sg_max: float = None,
        k_max: float = None,
        orientation_matrices: np.ndarray = None,
        radius: float = None,
        centering: str = "P",
        energy: float = None,
    ) -> IndexedDiffractionPatterns:
        """
        Indexes the Bragg reflections (diffraction spots) by their Miller indices.

        Parameters
        ----------
        cell : ase.cell.Cell or float or tuple of float
            The assumed unit cell with respect to the diffraction pattern should be indexed. Must be one of ASE `Cell`
            object, float (for a cubic unit cell) or three floats (for orthorhombic unit cells).
        orientation_matrices : np.ndarray, optional
            Orientation matrices used for indexing the diffraction spots. The shape of the orientation matrices must be
            broadcastable with the ensemble shape of the diffraction patterns.
        sg_max : float, optional
            Maximum excitation error [1/Å] of the indexed diffraction spots The default is estimated from the energy and `k_max`.
        k_max : float, optional
            Maximum scattering vector [1/Å] of the indexed diffraction spots. The default is the maximum frequency of the diffraction
            patterns.
        radius : float, optional
            Integration Radius of the diffraction spots [1/Å]. The default is the reciprocal-space sampling of the diffraction patterns.
        centering : {'P', 'F', 'I', 'A', 'B', 'C'}
            Assumed lattice centering used for determining the reflection conditions.
        energy : float, optional
            The energy of the electrons [keV]. The default is the energy stored in the metadata.

        Returns
        -------
        indexed_patterns : IndexedDiffractionPatterns
            The indexed diffraction pattern(s).
        """
        from abtem.bloch.indexing import (
            estimate_necessary_excitation_error,
            index_diffraction_spots,
            validate_cell,
        )
        from abtem.bloch.utils import filter_reciprocal_space_vectors, make_hkl_grid

        if orientation_matrices is not None and not is_broadcastable(
            self.ensemble_shape, orientation_matrices.shape[:-2]
        ):

            raise ValueError(
                "The ensemble shape and the shape of the orientation matrices must be broadcastable."
            )

        if energy is None:
            energy = self._get_from_metadata("energy")

        if k_max is None:
            k_max = max(self.max_frequency)

        if sg_max is None:
            sg_max = estimate_necessary_excitation_error(energy, k_max)

        cell = validate_cell(cell)

        hkl = make_hkl_grid(cell, k_max)

        mask = filter_reciprocal_space_vectors(
            hkl,
            cell,
            energy=energy,
            sg_max=sg_max,
            k_max=k_max,
            centering=centering,
            orientation_matrices=orientation_matrices,
        )

        hkl = hkl[mask]

        def _index_diffraction_spots(
            array, orientation_matrices, hkl, sampling, cell, energy, radius
        ):
            return index_diffraction_spots(
                array=array,
                hkl=hkl,
                sampling=sampling,
                cell=cell,
                energy=energy,
                radius=radius,
                orientation_matrices=orientation_matrices,
            )

        if self.is_lazy:
            chunks = tuple(
                c if n == sum(c) else 1
                for n, c in zip(orientation_matrices.shape, self.array.chunks[:-2])
            )
            lazy_orientation_matrices = da.from_array(
                orientation_matrices, chunks=chunks + (3, 3)
            )

            intensities = da.map_blocks(
                _index_diffraction_spots,
                self.array,
                lazy_orientation_matrices,
                hkl=hkl,
                sampling=self.sampling,
                cell=cell,
                energy=energy,
                radius=radius,
                drop_axis=len(self.array.shape) - 1,
                chunks=self.array.chunks[:-2] + (hkl.shape[0],),
                meta=np.array((), dtype=self.dtype),
            )
        else:
            intensities = index_diffraction_spots(
                array=self.array,
                hkl=hkl,
                sampling=self.sampling,
                cell=cell,
                energy=energy,
                radius=radius,
                orientation_matrices=orientation_matrices,
            )

        if orientation_matrices is None:
            orientation_matrices = np.eye(3)[(None,) * len(self.array.shape[:-2])]

        reciprocal_lattice_vectors = np.matmul(
            cell.reciprocal(),
            np.swapaxes(orientation_matrices, -2, -1),
        )

        return IndexedDiffractionPatterns(
            intensities,
            hkl,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    @property
    def fftshift(self) -> bool:
        """
        True if the zero-frequency is shifted to the center of the array.
        """
        return self._fftshift

    @property
    def sampling(self) -> tuple[float, float]:
        return self._sampling

    @property
    def angular_sampling(self) -> tuple[float, float]:
        """
        Angular sampling of diffraction patterns in `x` and `y` [mrad].
        """
        wavelength = energy2wavelength(self._get_from_metadata("energy"))
        return (
            self.sampling[0] * wavelength * 1e3,
            self.sampling[1] * wavelength * 1e3,
        )

    @property
    def max_angles(self) -> tuple[float, float]:
        """Maximum scattering angle in `x` and `y` [mrad]."""
        return (
            self.shape[-2] // 2 * self.angular_sampling[0],
            self.shape[-1] // 2 * self.angular_sampling[1],
        )

    @property
    def max_frequency(self):
        return abs(self.limits[0][1]), abs(self.limits[1][1])

    @property
    def limits(self) -> list[tuple[float, float]]:
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
    def angular_limits(self) -> list[tuple[float, float]]:
        """Lowest and highest scattering angle in `x` and `y` [mrad]."""

        limits = self.limits
        wavelength = energy2wavelength(self._get_from_metadata("energy"))
        limits[0] = (
            limits[0][0] * wavelength * 1e3,
            limits[0][1] * wavelength * 1e3,
        )
        limits[1] = (
            limits[1][0] * wavelength * 1e3,
            limits[1][1] * wavelength * 1e3,
        )
        return limits

    @property
    def offset(self) -> tuple[float, float]:
        limits = self.limits
        return limits[0][0], limits[1][0]

    @property
    def extent(self) -> tuple[float, float]:
        limits = self.limits
        return limits[0][0] - limits[0][1], limits[1][0] - limits[1][1]

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Reciprocal-space frequency coordinates [1 / Å]."""

        return (
            self.axes_metadata[-2].coordinates(self.base_shape[-2]),
            self.axes_metadata[-1].coordinates(self.base_shape[-1]),
        )

    @property
    def angular_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Scattering angle coordinates [mrad]."""

        xp = get_array_module(self.array)
        limits = self.angular_limits
        alpha_x = xp.linspace(
            limits[0][0], limits[0][1], self.shape[-2], dtype=xp.float32
        )
        alpha_y = xp.linspace(
            limits[1][0], limits[1][1], self.shape[-1], dtype=xp.float32
        )
        if self.fftshift:
            return alpha_x, alpha_y
        else:
            return np.fft.fftshift(alpha_x), np.fft.fftshift(alpha_y)

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
            array = _interpolate_bilinear(array, v, u, vw, uw)

        array = array / array.sum((-2, -1), keepdims=True) * old_sums

        return array.reshape(old_shape[:-2] + array.shape[-2:])

    def interpolate(
        self,
        sampling: str | float | tuple[float, float] = None,
        gpts: tuple[int, int] = None,
    ):
        """
        Interpolate diffraction pattern(s) producing equivalent pattern(s) with a different sampling.

        Parameters
        ----------
        sampling : 'uniform' or float or two floats
            Sampling of diffraction patterns after interpolation in `x` and `y` [1 / Å]. If a single value, the same
            sampling is used for both axes. If 'uniform', the diffraction patterns are down-sampled along the axis with
            the smallest pixel size such that the sampling is uniform.
        gpts : tuple of int
            Number of grid points of the diffraction patterns after interpolation in `x` and `y`.
            Do not use if 'sampling' is used.

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
            if np.isscalar(gpts):
                gpts = (gpts,) * 2

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

    def gaussian_source_size(
        self, sigma: float | tuple[float, float]
    ) -> DiffractionPatterns:
        """
        Simulate the effect of a finite source size on diffraction pattern(s) using a Gaussian filter.

        The filter is not applied to diffraction pattern individually, but the intensity of diffraction patterns are
        mixed across scan axes. Applying this filter requires two linear scan axes.

        Applying this filter before integrating the diffraction patterns will produce the same image as integrating
        the diffraction patterns first then applying a Gaussian filter.

        Parameters
        ----------
        sigma : float or two float
            Standard deviation of Gaussian kernel in the `x` and `y`-direction. If given as a single number, the
            standard deviation is equal for both axes.

        Returns
        -------
        filtered_diffraction_patterns : DiffractionPatterns
            The filtered diffraction pattern(s).
        """

        return _gaussian_source_size(self, sigma)

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
            The irradiation dose per unit of scan area [electrons per Å:sup:`2`]. This is only valid if the diffraction
            patterns has two scan axes.
        total_dose : float, optional
            The irradiation dose per diffraction pattern.
        samples : int, optional
            The number of samples to draw from a Poisson distribution. If this is greater than 1, an additional
            ensemble axis will be added to the measurement.
        seed : int, optional
            Seed the random number generator.

        Returns
        -------
        noisy_measurement : BaseMeasurements
            The noisy measurement.
        """

        if len(_scan_shape(self)) < 2 and dose_per_area is not None:
            raise ValueError(
                "diffraction patterns has less than two scan axes, provide 'total_dose' not 'dose_per_area' "
            )

        # TODO: normalization

        return super().poisson_noise(
            dose_per_area=dose_per_area,
            total_dose=total_dose,
            samples=samples,
            seed=seed,
        )

    def polar_binning(
        self,
        nbins_radial: int,
        nbins_azimuthal: int,
        inner: float = 0.0,
        outer: float = None,
        rotation: float = 0.0,
        offset: tuple[float, float] = (0.0, 0.0),
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

        def _radial_binning(array, nbins_radial, nbins_azimuthal, sampling):
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
                _radial_binning,
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
            array = _radial_binning(
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
    ) -> PolarMeasurements:
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

    def integrate_radial(
        self, inner: float, outer: float, offset: tuple[float, float] = (0.0, 0.0)
    ) -> Images:
        """
        Create images by integrating the diffraction patterns over an annulus defined by an inner and outer integration
        angle.

        Parameters
        ----------
        inner : float
            Inner integration limit [mrad].
        outer : float
            Outer integration limit [mrad].
        offset : tuple of float
            Offset of center of annular integration region [mrad].

        Returns
        -------
        integrated_images : Images
            The integrated images.
        """
        if isinstance(inner, Sequence) or isinstance(outer, Sequence):
            if isinstance(inner, Number):
                inners = (inner,) * len(outer)
                outers = outer
            else:
                outers = (outer,) * len(inner)
                inners = inner

            measurements = [
                self.integrate_radial(inner=inner, outer=outer)
                for inner, outer in zip(inners, outers)
            ]

            measurements = stack(
                measurements,
                axis_metadata=NonLinearAxis(
                    label="Limits", values=tuple(zip(inners, outers)), units="mrad"
                ),
            )
            return measurements

        self._check_integration_limits(inner, outer)

        xp = get_array_module(self.array)

        def _integrate_fourier_space(array, sampling):
            xp = get_array_module(array)

            bins = _annular_detector_mask(
                gpts=array.shape[-2:],
                sampling=sampling,
                inner=inner,
                outer=outer,
                fftshift=self.fftshift,
                offset=offset,
                xp=xp,
            )

            return xp.sum(array * bins, axis=(-2, -1))

        if self.is_lazy:
            integrated_intensity = self.array.map_blocks(
                _integrate_fourier_space,
                sampling=self.angular_sampling,
                drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            integrated_intensity = _integrate_fourier_space(
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

    def center_of_mass(self, units: str = "1/Å") -> Images | RealSpaceLineProfiles:
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

        if units == "mrad":
            x, y = self.angular_coordinates
        elif units == "1/Å":
            x, y = self.coordinates
        else:
            raise ValueError()

        xp = get_array_module(self.array)

        x, y = xp.asarray(x), xp.asarray(y)

        if self.is_lazy:
            base_axes = tuple(
                range(
                    len(self.ensemble_shape),
                    len(self.base_shape) + len(self.ensemble_shape),
                )
            )
            array = self.array.map_blocks(
                self._com, x=x, y=y, drop_axis=base_axes, dtype=np.complex64
            )
        else:
            array = self._com(self.array, x=x, y=y)

        return _reduced_scanned_images_or_line_profiles(array, self)

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

        def _bandlimit(array, inner, outer):
            alpha_x, alpha_y = self.angular_coordinates
            alpha = np.sqrt(alpha_x[:, None] ** 2 + alpha_y[None] ** 2)

            block = alpha > inner

            if outer != np.inf:
                block *= alpha < outer

            return array * block

        xp = get_array_module(self.array)

        if self.is_lazy:
            array = self.array.map_blocks(
                _bandlimit,
                inner=inner,
                outer=outer,
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            array = _bandlimit(self.array, inner, outer)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def crop(
        self,
        max_angle: float = None,
        max_frequency: float = None,
        gpts: tuple[int, int] = None,
    ) -> DiffractionPatterns:
        """
        Crop the diffraction patterns such that they only include spatial frequencies (scattering angles) up to a given
        limit.

        Parameters
        ----------
        max_angle : float, optional
            The maximum included scattering angle in the cropped diffraction patterns.
        max_frequency : float, optional
            The maximum included spatial frequency in the cropped diffraction patterns.
        gpts : tuple of int
            The number of gpts in the cropped diffraction patterns.

        Returns
        -------
        cropped_diffraction_patterns : DiffractionPatterns
        """

        none_args = (max_angle is None) + (max_frequency is None) + (gpts is None)

        if none_args != 2:
            raise ValueError(
                "provide exactly one of 'max_angle', 'max_frequency' or 'gpts'"
            )

        if gpts is None and max_angle is not None:
            gpts = (
                int(2 * np.round(max_angle / self.angular_sampling[0])) + 1,
                int(2 * np.round(max_angle / self.angular_sampling[1])) + 1,
            )
        elif gpts is None and max_frequency is not None:
            gpts = (
                int(2 * np.round(max_frequency / self.sampling[0])) + 1,
                int(2 * np.round(max_frequency / self.sampling[1])) + 1,
            )

        if gpts is None:
            raise ValueError()

        def _do_crop(array):
            xp = get_array_module(array)
            array = xp.fft.fftshift(
                fft_crop(xp.fft.ifftshift(array, axes=(-2, -1)), new_shape=gpts),
                axes=(-2, -1),
            )
            return array

        if self.is_lazy:
            xp = get_array_module(self.array)
            array = self.array.map_blocks(
                _do_crop,
                chunks=self.array.chunks[:-2] + gpts,
                meta=xp.array((), dtype=self.dtype),
            )
        else:
            array = _do_crop(self.array)

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        return self.__class__(**kwargs)

    def azimuthal_average(
        self,
        max_angle: float = None,
        radial_sampling: float = 1.0,
        weighting_function: str = "step",
        width: float = 1.0,
    ) -> ReciprocalSpaceLineProfiles:
        """
        Calculate the azimuthal averages of the diffraction patterns.

        Parameters
        ----------
        max_angle : float, optional
            The maximum included scattering angle in the azimuthal averages [mrad].
        order : float, optional
            The spline interpolation order. Default is 1.
        radial_sampling : float, optional
            The radial sampling of the azimuthal averages [mrad]. Default is equal to the smallest value of the x and y
            component of the angular sampling.
        weigthing_method : str

        """

        def _map_azimuthal_average(
            array,
            angular_coordinates,
            max_angle,
            radial_sampling,
            weighting_function,
            width,
        ):
            x, y = np.meshgrid(*angular_coordinates, indexing="ij")
            r = np.sqrt(x**2 + y**2)

            centers = np.arange(0, max_angle, radial_sampling)

            values = np.zeros(array.shape[:-2] + centers.shape)
            for i, center in enumerate(centers):
                if weighting_function == "step":
                    mask = np.abs(r - center) < width
                elif weighting_function == "gaussian":
                    mask = np.exp(-((r - center) ** 2) / (width**2 / 2))
                else:
                    raise ValueError()

                weight = np.sum(mask)

                if weight > 0:
                    values[..., i] = np.sum(array * mask, axis=(-2, -1)) / weight
                else:
                    values[..., i] = 0.0

            return values

        if max_angle is None:
            max_angle = -min(min(self.angular_limits))

        radial_sampling = radial_sampling * min(self.angular_sampling)
        width = width * min(self.angular_sampling)

        if self.is_lazy:
            xp = get_array_module(self.array)
            n = int(max_angle / radial_sampling)
            base_axes = tuple(
                range(
                    len(self.ensemble_shape),
                    len(self.ensemble_shape) + len(self.base_shape),
                )
            )
            array = self.array.map_blocks(
                _map_azimuthal_average,
                angular_coordinates=self.angular_coordinates,
                max_angle=max_angle,
                radial_sampling=radial_sampling,
                weighting_function=weighting_function,
                width=width,
                drop_axis=base_axes,
                new_axis=base_axes[0],
                chunks=self.array.chunks[:-2] + (n,),
                meta=xp.array((), dtype=np.float32),
            )
        else:
            array = _map_azimuthal_average(
                self.array,
                angular_coordinates=self.angular_coordinates,
                max_angle=max_angle,
                radial_sampling=radial_sampling,
                weighting_function=weighting_function,
                width=width,
            )

        wavelength = energy2wavelength(self._get_from_metadata("energy"))

        return ReciprocalSpaceLineProfiles(
            array,
            sampling=radial_sampling / (wavelength * 1e3),
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def fourier_shell_correlation(
        self,
        other: DiffractionPatterns,
        radial_sampling: float = 1.0,
        width: float = 1.0,
        weighting_function: str = "step",
    ):
        fsc = (self**0.5 * other**0.5).azimuthal_average(
            radial_sampling=radial_sampling,
            width=width,
            weighting_function=weighting_function,
        ) / (
            self.azimuthal_average(
                radial_sampling=radial_sampling,
                width=width,
                weighting_function=weighting_function,
            )
            * other.azimuthal_average(
                radial_sampling=radial_sampling,
                width=width,
                weighting_function=weighting_function,
            )
        )
        return fsc

    def block_direct(
        self, radius: float = None, margin: bool = None
    ) -> DiffractionPatterns:
        """
        Block the direct beam by setting the pixels of the zeroth-order Bragg reflection (non-scattered beam) to zero.

        Parameters
        ----------
        radius : float, optional
            The radius of the zeroth-order reflection to block [mrad]. If not given this will be inferred from the
            metadata, if available.
        margin : bool, optional
            If True adds a margin to the blocking radius to fully block soft apertures. Margin is true by default for
            diffraction patterns with 'semiangle_cutoff' in metadata.

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

        if "semiangle_cutoff" in self.metadata.keys() and margin is None:
            margin = True

        if margin:
            radius += max(self.angular_sampling)

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

    # def _plot_base_axes_metadata(self, units: str = None):
    #     if units is None:
    #         return self.base_axes_metadata
    #
    #     if units == "mrad":
    #         return [
    #             ReciprocalSpaceAxis(
    #                 sampling=self.angular_sampling[0],
    #                 offset=self.angular_limits[0][0],
    #                 label="kx",
    #                 units="mrad",
    #                 fftshift=self.fftshift,
    #                 _tex_label="$k_x$",
    #             ),
    #             ReciprocalSpaceAxis(
    #                 sampling=self.angular_sampling[1],
    #                 offset=self.angular_limits[1][0],
    #                 label="ky",
    #                 units="mrad",
    #                 fftshift=self.fftshift,
    #                 _tex_label="$k_y$",
    #             ),
    #         ]


class PolarMeasurements(BaseMeasurements):
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
        Sampling of the azimuthal bins [rad].
    radial_offset : float, optional
        Offset of the bins from the origin [mrad] (default is 0.0).
    azimuthal_offset : float, optional
        Rotation of the bins around the origin [rad] (default is 0.0).
    ensemble_axes_metadata : list of AxisMetadata, optional
        List of metadata associated with the ensemble axes. The length and item order must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.

    Returns
    -------
    polar_measurements : PolarMeasurements
        The polar measurements.
    """

    _base_dims = 2

    def __init__(
        self,
        array: np.ndarray,
        radial_sampling: float,
        azimuthal_sampling: float,
        radial_offset: float = 0.0,
        azimuthal_offset: float = 0.0,
        ensemble_axes_metadata: list[AxisMetadata] = None,
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
        cls,
        array: np.ndarray,
        axes_metadata: list[AxisMetadata],
        metadata: dict = None,
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
        radial_axis, azimuthal_axis = axes_metadata[-2:]

        if isinstance(radial_axis, LinearAxis) and isinstance(
            azimuthal_axis, LinearAxis
        ):
            radial_sampling = radial_axis.sampling
            radial_offset = radial_axis.offset
            azimuthal_sampling = azimuthal_axis.sampling
            azimuthal_offset = azimuthal_axis.offset
        else:
            raise RuntimeError()

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
    def base_axes_metadata(self) -> list[AxisMetadata]:
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
        """Offset of the bins from the origin [mrad]."""
        return self._radial_offset

    @property
    def outer_angle(self) -> float:
        """The outer angle of the outermost radial bin [mrad]."""
        return self._radial_offset + self.radial_sampling * self.shape[-2]

    @property
    def radial_sampling(self) -> float:
        """Sampling of the radial bins [mrad]."""
        return self._radial_sampling

    @property
    def azimuthal_sampling(self) -> float:
        """Sampling of the azimuthal bins [rad]."""
        return self._azimuthal_sampling

    @property
    def azimuthal_offset(self) -> float:
        """Rotation of the bins around the origin [rad]."""
        return self._azimuthal_offset

    def integrate_radial(
        self, inner: float, outer: float
    ) -> Images | RealSpaceLineProfiles:
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
        real_space_line_profiles : RealSpaceLineProfiles
            Integrated line profiles (returned if there is only one scan axis).
        """
        return self.integrate(radial_limits=(inner, outer))

    def integrate(
        self,
        radial_limits: tuple[float, float] = None,
        azimuthal_limits: tuple[float, float] = None,
        detector_regions: int | Sequence[int] = None,
    ) -> Images | RealSpaceLineProfiles:
        """
        Integrate polar regions to produce an image or line profiles.

        Parameters
        ----------
        radial_limits : tuple of float
            Inner and outer radial angles of the integration limits [mrad].
        azimuthal_limits : tuple of float
            Lower and upper azimuthal angles of the integration limits [rad].
        detector_regions : int or sequence of int
            The explicit detector regions to integrate over.

        Returns
        -------
        integrated_images : Images or RealSpaceLineProfiles
        """

        if detector_regions is not None:
            if (radial_limits is not None) or (azimuthal_limits is not None):
                raise ValueError()

            if np.isscalar(detector_regions):
                detector_regions = [detector_regions]

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
        self, sigma: float | tuple[float, float]
    ) -> PolarMeasurements:
        """
        Simulate the effect of a finite source size on diffraction pattern(s) using a Gaussian filter.

        The filter is not applied to diffraction pattern individually, but the intensity of diffraction patterns are
        mixed across scan axes. Applying this filter requires two linear scan axes.

        Applying this filter before integrating the diffraction patterns will produce the same image as integrating
        the diffraction patterns first then applying a Gaussian filter.

        Parameters
        ----------
        sigma : float or two float
            Standard deviation of Gaussian kernel in the `x` and `y`-direction. If given as a single number, the
            standard deviation is equal for both axes.

        Returns
        -------
        filtered_diffraction_patterns : DiffractionPatterns
            The filtered diffraction pattern(s).
        """

        return _gaussian_source_size(self, sigma)

    def to_diffraction_patterns(
        self, gpts: int | tuple[int, int], margin: float | tuple[float, float] = 0.1
    ):
        """
        Convert the polar measurements to diffraction patterns by discretizing the polar bins on a regular grid.

        Parameters
        ----------
        gpts : int or two int
            Number of grid points describing the diffraction patterns.
        margin : float or two float, optional
            The margin as a fraction of the outer angle of the polar measurements to add to the maximum angle of the
            diffraction patterns.

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
        """

        if np.isscalar(gpts):
            gpts = (gpts,) * 2

        if np.isscalar(margin):
            margin = (margin,) * 2

        angular_sampling = (
            (1 + margin[0]) * self.outer_angle / gpts[0] * 2,
            (1 + margin[1]) * self.outer_angle / gpts[1] * 2,
        )

        nbins_radial, nbins_azimuthal = self.base_shape

        regions = _polar_detector_bins(
            gpts=gpts,
            sampling=angular_sampling,
            inner=self.radial_offset,
            outer=self.outer_angle,
            nbins_radial=nbins_radial,
            nbins_azimuthal=nbins_azimuthal,
            fftshift=True,
            rotation=self.azimuthal_offset,
            offset=(0.0, 0.0),
            return_indices=False,
        )

        new_array = np.zeros(self.ensemble_shape + regions.shape, dtype=np.float32)
        for i, indices in enumerate(label_to_index(regions)):
            x, y = np.unravel_index(indices, regions.shape)
            radial, azimuthal = np.unravel_index(i, (nbins_radial, nbins_azimuthal))
            new_array[..., x, y] = self.array[..., radial, azimuthal][..., None]

        new_array[..., regions < 0] = np.nan

        wavelength = energy2wavelength(self._get_from_metadata("energy"))
        sampling = (
            angular_sampling[0] / (wavelength * 1e3),
            angular_sampling[1] / (wavelength * 1e3),
        )

        return DiffractionPatterns(
            new_array,
            sampling=sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def differentials(
        self,
        direction_1: tuple[int | tuple[int, ...], int | tuple[int, ...]],
        direction_2: tuple[int | tuple[int, ...], int | tuple[int, ...]],
        return_complex: bool = True,
    ):
        """
        Calculate the differential signal by subtracting the intensity of specified detector regions.

        Parameters
        ----------
        direction_1 : tuple of int or tuple of tuple of int
            The detector regions used for calculating the differential signal for the first direction. The first item
            is the detector region(s) contributing to the positive term and the second item is the detector region(s)
            contributing to the negative terms.
        direction_2 : tuple of int or tuple of tuple of int
            The detector regions used for calculating the differential signal for the second direction. The first item
            is the detector region(s) contributing to the positive term and the second item is the detector region(s)
            contributing to the negative terms.
        return_complex : bool, optional
            If True, return a complex image where the real and imaginary part represents `direction_1` and
            `direction_2`. If False, return images with an ensemble dimension for the directions.

        Returns
        -------
        differential_image : Images
            The (complex) differential image(s).
        """
        differential_1 = self.integrate(
            detector_regions=direction_1[1]
        ) - self.integrate(detector_regions=direction_1[0])

        differential_2 = self.integrate(
            detector_regions=direction_2[1]
        ) - self.integrate(detector_regions=direction_2[0])

        if not return_complex:
            stacked = stack(
                (differential_1, differential_2), ("direction_1", "direction_2")
            )
            return stacked

        xp = get_array_module(self.device)
        array = xp.zeros_like(differential_1.array, dtype=xp.complex64)

        array.real = differential_1.array
        array.imag = differential_2.array

        return differential_1.__class__(
            array, **differential_1._copy_kwargs(exclude=("array",))
        )

    def to_image_ensemble(self):
        """
        Convert the polar measurements to an ensemble of images, where the radial and azimuthal angles becomes ensemble
        axes.

        Returns
        -------
        image_ensemble : Images
        """

        image_axes = _scan_axes(self)

        xp = get_array_module(self.array)

        array = xp.moveaxis(self.array, image_axes, (-2, -1))[..., 0, :, :]

        ensemble_axes_metadata = [
            axis.copy()
            for i, axis in enumerate(self.axes_metadata[:-1])
            if i not in image_axes
        ]
        ensemble_axes_metadata[-1]._default_type = "range"

        sampling = _scan_sampling(self)

        return Images(
            array,
            sampling=(sampling[0], sampling[1]),
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def show(
        self,
        ax: Axes = None,
        gpts: int | tuple[int, int] = (512, 512),
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_color_scale: bool = False,
        explode: bool | Sequence[bool] = (),
        overlay: bool | Sequence[int] = (),
        figsize: tuple[int, int] = None,
        title: bool | str = True,
        units: str = None,
        interact: bool = False,
        display: bool = True,
    ) -> Visualization:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        gpts : int or tuple of int, optional
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for exploded plots.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The size and padding of the colorbars may be adjusted using the
            `set_cbar_size` and `set_cbar_padding` methods.
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. If the measurement is complex the colormap
            must be one of 'hsv' or 'hsluv'.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        power : float
            Show image on a power scale.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown. May be given as a sequence of axis indices to create a grid of images from
            the specified axes. The default is determined by the axis metadata.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Set the column title of the images. If True is given instead of a string the title will be given by the
            value corresponding to the "name" key of the axes metadata dictionary, if this item exists.
        units : str
            The units used for the x and y axes. The given units must be compatible with the axes of the images.
        interact : bool
            If True, create an interactive visualization. This requires enabling the ipympl Matplotlib backend.
        display : bool, optional
            If True (default) the figure is displayed immediately.

        Returns
        -------
        measurement_visualization_2d : MeasurementVisualizationImshow
        """

        diffraction_patterns = self.to_diffraction_patterns(gpts=gpts)

        if not interact:
            diffraction_patterns.compute()

        return diffraction_patterns.show(
            ax=ax,
            cbar=cbar,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            power=power,
            common_color_scale=common_color_scale,
            explode=explode,
            overlay=overlay,
            figsize=figsize,
            title=title,
            units=units,
            interact=interact,
            display=display,
        )


@jit(nopython=True, nogil=True, fastmath=True)
def calculate_max_reciprocal_space_vector(hkl, reciprocal_lattice_vectors):

    k_max = 0.0
    for i in range(len(hkl)):
        lengths = (
            (
                hkl[i, 0] * reciprocal_lattice_vectors[..., 0, :]
                + hkl[i, 1] * reciprocal_lattice_vectors[..., 1, :]
                + hkl[i, 2] * reciprocal_lattice_vectors[..., 2, :]
            )
            ** 2
        ).sum(-1)

        if hasattr(lengths, "max"):
            lengths = lengths.max()

        k_max = max(k_max, lengths)

    return np.sqrt(k_max)


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def reciprocal_lattice_vector_mask(mask, hkl, reciprocal_lattice_vectors, k_max):

    for i in prange(len(hkl)):  # pylint: disable=not-an-iterable
        lengths = (
            (
                hkl[i, 0] * reciprocal_lattice_vectors[..., 0, :]
                + hkl[i, 1] * reciprocal_lattice_vectors[..., 1, :]
                + hkl[i, 2] * reciprocal_lattice_vectors[..., 2, :]
            )
            ** 2
        ).sum(-1)

        include = lengths < k_max**2

        if hasattr(lengths, "any"):
            include = include.any()

        mask[i] = include

    return mask


class IndexedDiffractionPatterns(BaseMeasurements):
    """
    Diffraction patterns indexed by their Miller indices.

    Parameters
    ----------
    array : np.ndarray
        1D or greater array of type `float` or `complex`. The last axis represents the diffraction spots and should have the same
        length as the number of miller indices, any preceding axis represents an ensemble axis.
    miller_indices : np.ndarray
        The miller indices of the diffraction spots as an N x 3 array where N is the number of miller indices. The
        order of the miller indices must correspond to the array of intensities. The second axis represents each
        hkl miller index.
    reciprocal_lattice_vectors : np.ndarray
        The reciprocal lattice vectors of the crystal as a 3 x 3 array. The first axis represents
        miller indices and the order of the items must correspond to the array of intensities. The second axis
        represents the reciprocal space positions in x, y and z [1/Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        List of metadata associated with the ensemble axes. The length and item order must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.
    """

    _base_dims = 1

    def __init__(
        self,
        array: np.ndarray,
        miller_indices: np.ndarray,
        reciprocal_lattice_vectors: np.ndarray,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):

        if len(reciprocal_lattice_vectors.shape) <= len(array.shape):
            reciprocal_lattice_vectors = reciprocal_lattice_vectors[
                (None,) * (len(reciprocal_lattice_vectors.shape) - len(array.shape) + 1)
            ]

        # if not is_broadcastable(array.shape[:-1], reciprocal_lattice_vectors.shape[:-2]):
        #    raise ValueError()

        # if not len(miller_indices) == reciprocal_lattice_vectors.shape[-3]:
        #     raise ValueError(
        #         "The number of miller indices and reciprocal space positions must be equal."
        #     )

        if not len(miller_indices) == array.shape[-1]:
            raise ValueError(
                "The number of miller indices must be equal to the number of diffraction spots."
            )

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

        self._miller_indices = miller_indices
        self._intensities = array
        self._reciprocal_lattice_vectors = reciprocal_lattice_vectors

    def from_array_and_metadata(
        self, array: np.ndarray, axes_metadata: list[AxisMetadata], metadata: dict
    ) -> "T":
        raise NotImplementedError

    def _area_per_pixel(self):
        raise NotImplementedError

    @property
    def base_axes_metadata(self) -> list:
        return [AxisMetadata(label="hkl")]

    @property
    def intensities(self) -> np.ndarray:
        """
        Intensities of the diffraction spots.
        """
        return self._array

    @property
    def reciprocal_lattice_vectors(self) -> np.ndarray:
        """
        Reciprocal lattice vectors of the diffraction spots.
        """
        return self._reciprocal_lattice_vectors

    @property
    def positions(self) -> np.ndarray:
        """
        Reciprocal space positions of the diffraction spots.
        """
        positions = self.miller_indices @ self.reciprocal_lattice_vectors
        return positions

    @property
    def all_positions(self) -> np.ndarray:
        """
        Reciprocal space positions of the diffraction spots.
        """
        repeats = ()
        for n, m in zip(self.shape[:-1], self.reciprocal_lattice_vectors.shape[:-2]):
            if n == m:
                repeats += (1,)
            elif m == 1:
                repeats += (n,)
            else:
                raise RuntimeError("Incompatible shapes.")

        positions = np.tile(self.positions, repeats + (1, 1))
        return positions

    @property
    def miller_indices(self) -> np.ndarray:
        """
        Miller indices of the diffraction spots.
        """
        return self._miller_indices

    @property
    def angular_positions(self):
        """
        Scattering angles of the diffraction spots.
        """
        wavelength = energy2wavelength(self._get_from_metadata("energy"))
        return self.positions * wavelength * 1e3

    @property
    def ensemble_shape(self) -> tuple:
        return self.intensities.shape[:-1]

    @classmethod
    def _pack_kwargs(cls, kwargs):
        kwargs["miller_indices"] = [tuple(hkl) for hkl in kwargs["miller_indices"]]
        kwargs["positions"] = [
            (float(position[0]), float(position[1]), float(position[2]))
            for position in kwargs["positions"]
        ]
        return super()._pack_kwargs(kwargs)

    @classmethod
    def _unpack_kwargs(cls, attrs):
        kwargs = super()._unpack_kwargs(attrs)
        kwargs["miller_indices"] = np.array(kwargs["miller_indices"], dtype=int)
        return kwargs

    def __getitem__(self, items):
        items = self._validate_items(items)
        kwargs = self.get_items(items)

        new_items = ()
        for i, n in zip(items, self.reciprocal_lattice_vectors.shape):
            if n == 1:
                if isinstance(i, int):
                    new_items += (0,)
                else:
                    new_items += (slice(None),)
            else:
                new_items += (i,)

        # items = tuple(
        #    0 if (isinstance(i, int) and (n == 1)) else slice(None)
        #    for i, n in zip(items, self.reciprocal_lattice_vectors.shape)
        # )
        kwargs["reciprocal_lattice_vectors"] = self.reciprocal_lattice_vectors[
            new_items
        ]

        return self.__class__(**kwargs)

    def remove_low_intensity(self, threshold: float = 1e-3):
        """
        Remove diffraction spots with intensity below a threshold for all ensemble dimensions.

        Parameters
        ----------
        threshold : float
            Intensity threshold for removing diffraction spots.

        Returns
        -------
        thresholded_spots : IndexedDiffractionPatterns
            The indexed diffraction spots with an intensity above the given threshold.
        """
        if self.is_lazy:
            raise RuntimeError("Cannot threshold lazy IndexedDiffractionPatterns.")

        ensemble_axes = tuple(range(self.ensemble_dims))

        xp = get_array_module(self.intensities)
        mask = xp.max(self.intensities, axis=ensemble_axes) > threshold

        mask = asnumpy(mask)

        miller_indices = self.miller_indices[mask]
        intensities = self.intensities[..., mask]

        return self.__class__(
            intensities,
            miller_indices,
            reciprocal_lattice_vectors=self.reciprocal_lattice_vectors,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self._metadata,
        )

    def sort(self, criterion: str = "distance"):
        """
        Sort the diffraction spots according to a given criterion.

        Parameters
        ----------
        criterion : {'distance', 'intensity'}
            The boundary parameter determines how the images are extended beyond their boundaries when the filter
            overlaps with a border.

                ``distance`` :
                    Sort according to the distance in reciprocal space from the zero frequency.

                ``intensity`` :
                    Sort according to the intensity of the diffraction spots.

        Returns
        -------
        sorted_spots : IndexedDiffractionPatterns
        """
        if self.lazy:
            raise RuntimeError("Cannot sort lazy IndexedDiffractionPatterns.")

        if criterion == "distance":
            criterion = -np.linalg.norm(self.positions, axis=1)
        elif criterion == "intensity":
            ensemble_axes = tuple(range(len(self.ensemble_shape)))
            criterion = -np.max(self.intensities, axis=ensemble_axes)
        else:
            raise ValueError()

        order = np.argsort(criterion)
        array = self.array[..., order]
        miller_indices = self.miller_indices[order]
        reciprocal_lattice_vectors = self.reciprocal_lattice_vectors[..., order, :, :]

        return self.__class__(
            array,
            miller_indices,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self._metadata,
        )

    def crop(self, max_angle: float = None, k_max: float = None):
        """
        Crop the indexed diffraction patterns such that they only include spots with spatial frequencies
        (scattering angles) up to a given limit.

        Parameters
        ----------
        max_angle : float, optional
            The maximum included scattering angle in the cropped diffraction patterns.
        k_max : float, optional
            The maximum included reciprocal lattice vector in the cropped diffraction spots.

        Returns
        -------
        cropped : IndexedDiffractionPatterns
        """

        if max_angle is not None and k_max is None:
            wavelength = energy2wavelength(self._get_from_metadata("energy"))
            k_max = max_angle / wavelength / 1e3

        elif not k_max or max_angle:
            raise ValueError("Either 'max_angle' or 'k_max' must be given.")

        mask = np.zeros(len(self.miller_indices), dtype=bool)

        mask = reciprocal_lattice_vector_mask(
            mask,
            self.miller_indices.astype(self.reciprocal_lattice_vectors.dtype),
            self.reciprocal_lattice_vectors,
            k_max,
        )

        miller_indices = self.miller_indices[mask]
        array = self.array[..., mask]

        return self.__class__(
            array,
            miller_indices,
            reciprocal_lattice_vectors=self.reciprocal_lattice_vectors,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self._metadata,
        )

    def normalize_to_spot(self, spot: tuple[int, int, int] = None):
        """
        Normalize the intensity of the diffraction spots.

        Parameters
        ----------
        spot : tuple of three int
            The intensities will be normalized with respect to the intensity of this spot. Defaults to the most
            intense spot.

        Returns
        -------
        normalized_indexed_diffraction_patterns : IndexedDiffractionPatterns
        """
        intensities_dict = self.intensities_dict

        if spot is None:
            c = np.max(self.intensities)
        else:
            c = intensities_dict[spot]

        intensities = self.intensities / c
        return self.__class__(
            intensities,
            self.miller_indices.copy(),
            self.positions.copy(),
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self._metadata,
        )

    def to_data_array(self):
        """
        Convert the indexed diffraction patterns to xarray DataArray.

        Returns
        -------
        data_array_of_indexed_spots : xarray.DataArray
        """
        import xarray

        coords = []
        for axes_metadata, n in zip(self.ensemble_axes_metadata, self.ensemble_shape):
            coords.append(list(axes_metadata.coordinates(n)))

        coords.append(["{} {} {}".format(*hkl) for hkl in self.miller_indices])

        dims = [
            axes_metadata.label for axes_metadata in self.ensemble_axes_metadata
        ] + ["hkl"]

        data = xarray.DataArray(
            self.array,
            coords=coords,
            dims=dims,
            attrs={
                "long_name": "intensity",
                "units": self.metadata.get("units", "arb. unit"),
            },
        )

        return data

    def _miller_indices_to_string(self):
        return ["{} {} {}".format(*hkl) for hkl in self.miller_indices]

    def to_dataframe(self):
        """
        Convert the indexed diffraction patterns to pandas DataFrame.

        Returns
        -------
        data_frame : pd.DataFrame
        """
        import pandas as pd

        if self.ensemble_shape:
            if len(self.ensemble_shape) > 1:
                raise RuntimeError(
                    "cannot convert indexed diffraction patterns with more than one ensemble axis to"
                    "dataframe"
                )

            intensities = {
                hkl: self.intensities[..., i]
                for i, hkl in enumerate(self._miller_indices_to_string())
            }

            axes_metadata = self.ensemble_axes_metadata[0]

            if hasattr(axes_metadata, "values"):
                index = axes_metadata.values
            else:
                index = list(range(len(self.intensities)))

            df = pd.DataFrame(intensities, index=index)

            with config.set({"visualize.use_tex": False}):
                df.index.name = self.axes_metadata[0].format_label()
                df.columns.name = self.axes_metadata[1].format_label()

            return df
        else:
            intensities = {
                hkl: intensity
                for hkl, intensity in zip(
                    self._miller_indices_to_string(), self.intensities
                )
            }
            return pd.DataFrame(intensities, index=[0])

    def block_direct(self):
        """
        Remove the zero-order spot.

        Returns
        -------
        blocked : IndexedDiffractionPatterns
            The indexed diffraction spots without the zero-order spot.
        """

        to_delete = np.where(np.all(self.miller_indices == 0, axis=1))[0]

        miller_indices = np.delete(self.miller_indices, to_delete, axis=0)
        intensities = np.delete(self.intensities, to_delete, axis=-1)

        return self.__class__(
            intensities,
            miller_indices,
            reciprocal_lattice_vectors=self.reciprocal_lattice_vectors,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self._metadata,
        )

    def max_reciprocal_space_vector_length(self):
        return calculate_max_reciprocal_space_vector(
            self.miller_indices, self.reciprocal_lattice_vectors
        )

    def show(
        self,
        ax: Axes = None,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_color_scale: bool = False,
        scale: float = 0.5,
        explode: bool | Sequence[bool] = (),
        overlay: bool | Sequence[bool] = (),
        figsize: tuple[int, int] = None,
        title: bool | str = True,
        units: str = None,
        interact: bool = False,
        display: bool = True,
        **kwargs,
    ):
        """
        Show the diffraction spots as an EllipseCollection using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for exploded plots.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The size and padding of the colorbars may be adjusted using the
            `set_cbar_size` and `set_cbar_padding` methods.
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. If the measurement is complex the colormap
            must be one of 'hsv' or 'hsluv'.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        power : float
            Show diffraction spots intensities on a power scale.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        scale : float, optional
            Scale the radii of the circles representing the diffraction spots.
        explode : bool or sequence of bool, optional
            If True, a grid of plots is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown. May be given as a sequence of axis indices to create a grid of plots from
            the specified axes. The default is determined by the axis metadata.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Set the column title of the plots. If True is given instead of a string the title will be given by the
            value corresponding to the "name" key of the axes metadata dictionary, if this item exists.
        units : str
            The units used for the x and y axes. The given units must be compatible with the axes of the plots.
        interact : bool
            If True, create an interactive visualization. This requires enabling the ipympl Matplotlib backend.
        display : bool, optional
            If True (default) the figure is displayed immediately.

        Returns
        -------
        visualization : Visualization
        """

        k_max = self.max_reciprocal_space_vector_length() * _get_conversion_factor(
            units, "1/Å", self.metadata.get("energy", None)
        )

        xlim = [-k_max, k_max]
        ylim = [-k_max, k_max]

        visualization = Visualization(
            measurement=self,
            ax=ax,
            common_scale=common_color_scale,
            figsize=figsize,
            title=title,
            aspect=True,
            share_x=True,
            share_y=True,
            explode=explode,
            overlay=overlay,
            interactive=not interact and display,
            value_limits=(vmin, vmax),
            power=power,
            cmap=cmap,
            cbar=cbar,
            scale=scale,
            units=units,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        if interact:
            gui = visualization.interact(ScatterGUI, display=display)

        return visualization

    @property
    def intensities_dict(self) -> Dict[tuple[int, int, int], np.ndarray]:
        """
        A dictionary mapping miller indices to intensities.
        """
        intensities = {
            tuple(hkl): intensity
            for hkl, intensity in zip(
                self.miller_indices,
                np.moveaxis(self.intensities, -1, 0),
            )
        }

        values = np.zeros(self.shape[:-1], dtype=np.float32)
        intensities = defaultdict(lambda: values, intensities)
        return intensities

    @property
    def positions_dict(self) -> Dict[tuple[int, int, int], np.ndarray]:
        """
        A dictionary mapping miller indices to reciprocal space positions [1/Å].
        """

        positions = {
            tuple(hkl): position
            for hkl, position in zip(
                self.miller_indices,
                np.moveaxis(self.reciprocal_lattice_vectors, -2, 0),
            )
        }
        return positions

    @classmethod
    def _stack(
        cls,
        diffraction_spots: IndexedDiffractionPatterns,
        axis_metadata: list[AxisMetadata],
        axis: int,
    ):
        intensities = [spots.intensities_dict for spots in diffraction_spots]
        # positions = [spots.positions_dict for spots in diffraction_spots]

        # def merge_dicts_no_overwrite(dict1, dict2):
        #     return {**dict1, **{k: v for k, v in dict2.items() if k not in dict1}}

        # merged = {}
        # for positions1 in positions:
        #     merged = merge_dicts_no_overwrite(merged, positions1)

        # positions = [
        #     merge_dicts_no_overwrite(positions1, merged) for positions1 in positions
        # ]

        miller_indices = list(
            set(itertools.chain(*[intensities1.keys() for intensities1 in intensities]))
        )

        new_intensities = {}
        # new_positions = {}
        for hkl in miller_indices:
            new_intensities[hkl] = []
            # new_positions[hkl] = []

            for intensities1 in intensities:
                new_intensities[hkl].append(intensities1[hkl])
                # new_positions[hkl].append(positions1[hkl])

            new_intensities[hkl] = np.stack(new_intensities[hkl], axis=axis)
            # new_positions[hkl] = np.stack(new_positions[hkl], axis=axis)

        miller_indices = np.stack(list(new_intensities.keys()), axis=0)

        # positions = np.stack(list(new_positions.values()), axis=-2)
        intensities = np.stack(list(new_intensities.values()), axis=-1)

        positions = np.stack(
            [spots.reciprocal_lattice_vectors for spots in diffraction_spots], axis=0
        )

        ensemble_axes_metadata = [
            axis_metadata.copy()
            for axis_metadata in diffraction_spots[0].ensemble_axes_metadata
        ]
        ensemble_axes_metadata.insert(axis, axis_metadata)
        metadata = diffraction_spots[0].metadata

        return IndexedDiffractionPatterns(
            intensities, miller_indices, positions, ensemble_axes_metadata, metadata
        )
