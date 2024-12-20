"""Module for describing abTEM spectrum objects."""

from __future__ import annotations

import copy
import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from numbers import Number
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import dask.array as da
import numpy as np
from ase import Atom
from ase.cell import Cell
from matplotlib.axes import Axes
from numba import jit  # type: ignore

from abtem.array import ArrayObject, stack
from abtem.core import config
from abtem.core.axes import (
    AxisMetadata,
    LinearAxis,
    NonLinearAxis,
    RealSpaceAxis,
    ReciprocalSpaceAxis,
    EnergyAxis,
    MomentumAxis,
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
from abtem.core.units import get_conversion_factor
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    is_broadcastable,
    label_to_index,
    get_dtype
)
from abtem.distributions import BaseDistribution
from abtem.noise import NoiseTransform, ScanNoiseTransform
from abtem.visualize.artists import LinesArtist
from abtem.visualize.visualizations import Visualization
from abtem.visualize.widgets import ImageGUI, LinesGUI, ScatterGUI

interpolate_bilinear_cuda: Optional[Callable] = None
sum_run_length_encoded: Optional[Callable] = None
sum_run_length_encoded_cuda: Optional[Callable] = None
if cp is not None:
    from abtem.core._cuda import interpolate_bilinear as interpolate_bilinear_cuda
    from abtem.core._cuda import sum_run_length_encoded as sum_run_length_encoded_cuda

xr: Optional[ModuleType] = None
try:
    import xarray as xr
except ImportError:
    xr = None

pd: Optional[ModuleType] = None
try:
    import pandas as pd
except ImportError:
    pd = None

if TYPE_CHECKING:
    from abtem.waves import BaseWaves


BaseSpectraSubclass = TypeVar("BaseSpectraSubclass", bound="BaseSpectra")

class BaseSpectra(ArrayObject, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    Base class for all spectrum types.

    Parameters
    ----------
    array : ndarray
        Array containing data of type `float` or `complex`.
    axes_metadata : list of AxisMetadata, optional
        Metadata associated with an ensemble axis.
    metadata : dict, optional
        A dictionary defining simulation metadata.
    """

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ):
        super().__init__(
            array=array,
            ensemble_axes_metadata=axes_metadata,
            metadata=metadata,
        )

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list:
        """List of AxisMetadata of the base axes."""

    @property
    def metadata(self) -> dict:
        """Metadata describing the spectrum."""
        return self._metadata

    def _get_from_metadata(self, key: Hashable):
        if key not in self.metadata.keys():
            raise RuntimeError(f"{key} not in spectrum metadata.")
        return self.metadata[key]

    def relative_difference(
        self, other: BaseSpectra, min_relative_tol: float = 0.0
    ) -> BaseSpectra:
        """
        Calculates the relative difference with respect to another compatible spectrum.

        Parameters
        ----------
        other : BaseSpectra
            Measurement to which the difference is calculated.
        min_relative_tol : float
            Avoids division by zero errors by defining a minimum value of the divisor in the relative difference.

        Returns
        -------
        difference : BaseSpectra
            The relative difference as a spectrum of the same type.
        """
        difference = self - other

        xp = get_array_module(self.array)

        valid = xp.abs(self.array) >= min_relative_tol * self.array.max()
        difference._array[valid] /= self.array[valid]
        difference._array[valid == 0] = np.nan
        difference._array *= 100.0

        difference.metadata["label"] = "Relative difference"
        difference.metadata["units"] = "%"
        difference.metadata["tex_units"] = r"$\%$"
        return difference

    def normalize_ensemble(self, scale: str = "max", shift: str = "mean"):
        """
        Normalize the ensemble by shifting and scaling each member.

        Parameters
        ----------
        scale : {'max', 'min', 'sum', 'mean', 'ptp'}
        shift : {'max', 'min', 'sum', 'mean', 'ptp'}

        Returns
        -------
        normalized_measurements : BaseSpectra or subclass of _BaseSpectra
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
        cls,
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: Optional[dict] = None,
    ) -> BaseSpectra:
        pass

    def reduce_ensemble(self) -> ArrayObject:
        """Calculates the mean of an ensemble measurement (e.g. of frozen phonon
        configurations)."""
        axis = tuple(
            i
            for i, axis in enumerate(self.axes_metadata)
            if hasattr(axis, "_ensemble_mean") and axis._ensemble_mean
        )

        if len(axis) == 0:
            return self

        return self.mean(axis=axis)

    def _apply_element_wise_func(self, func: callable) -> "BaseSpectraSubclass":
        d = self._copy_kwargs(exclude=("array",))
        d["array"] = func(self.array)
        return self.__class__(**d)

    def _scale_axis_from_metadata(self):
        return ScaleAxis(
            label=self.metadata.get("label", ""),
            units=self.metadata.get("units", ""),
            tex_label=None,
        )

    def to_spectrum_ensemble(self):
        return SpectraEnsemble(
            array=self.array,
            ensemble_axes_metadata=self.axes_metadata,
            metadata=self.metadata,
        )

    @abstractmethod
    def show(self, *args, **kwargs):
        """Documented in subclasses"""

class SpectraEnsemble(BaseSpectra):
    _base_dims = 0

    def __init__(
        self,
        array: np.ndarray,
        ensemble_axes_metadata: list[AxisMetadata],
        metadata: dict | None = None,
    ):
        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def base_axes_metadata(self):
        return []

    @classmethod
    def from_array_and_metadata(
        cls, array: np.ndarray, axes_metadata: list[AxisMetadata], metadata: dict
    ) -> "BaseSpectraSubclass":
        return cls(array, axes_metadata, metadata)

    def show(
        self,
        type: str = "lines",
        ax: Optional[Axes] = None,
        power: float = 1.0,
        common_scale: bool = False,
        explode: bool | Sequence[int] = (),
        overlay: bool | Sequence[int] = (),
        figsize: Optional[tuple[int, int]] = None,
        title: bool | str = True,
        units: Optional[str] = None,
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
        power : float
            Show image on a power scale.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown. May be given as a sequence of axis indices to create a grid of images from
            the specified axes. The default is determined by the axis metadata.
        overlay : bool or sequence of int, optional
            If True, all line profiles in the ensemble are shown in a single plot. If False, only the first ensemble
            item is shown. May be given as a sequence of axis indices to specify which line profiles in the ensemble to
            show together. The default is determined by the axis metadata.
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
        if not interact:
            self.compute()

        scale_axis = self._scale_axis_from_metadata()

        array = self.array

        artist_type = LinesArtist

        visualization = Visualization(
            measurement=self,
            ax=ax,
            artist_type=artist_type,
            # power=power,
            aspect=False,
            share_x=True,
            share_y=common_scale,
            common_scale=common_scale,
            explode=explode,
            overlay=overlay,
            figsize=figsize,
            # interact=interact,
            title=title,
            **kwargs,
        )

        if common_scale is False and visualization._explode:
            visualization.axes.set_sizes(padding=0.8)

        return visualization

class _BaseSpectra2D(BaseSpectra):
    _base_dims = 2

    @property
    def base_shape(self) -> tuple[int, int]:
        return super().base_shape[-2], super().base_shape[-1]

    #@abstractmethod
    #def _get_1d_equivalent(self):
    #    pass

    @property
    @abstractmethod
    def values(self) -> tuple[list[float], values[float]]:
        """
        Axis values of the measurements in `E` and `q` [eV] or [1/Å].
        """

    @property
    @abstractmethod
    def extent(self) -> tuple[float, float]:
        """
        Extent of measurements in `E` and `q` [eV] or [1/Å].
        """

    def show(
        self,
        ax: Optional[Axes] = None,
        cbar: bool = False,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        power: float = 1.0,
        common_color_scale: bool = False,
        explode: bool | Sequence[int] = (),
        overlay: bool | Sequence[int] = (),
        figsize: Optional[tuple[int, int]] = None,
        title: bool | str = True,
        units: Optional[str] = None,
        interact: bool = False,
        display: bool = True,
        **kwargs,
    ) -> Visualization:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for exploded
            plots.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The size and padding of the colorbars may
            be adjusted using the `set_cbar_size` and `set_cbar_padding` methods.
        cmap : str, optional
            Matplotlib colormap name used to map scalar data to colors. If the
            measurement is complex the colormap must be one of 'hsv' or 'hsluv'.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array
            values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array
            values.
        power : float
            Show image on a power scale.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a
            single colorbar is created (if it is requested). Default is False.
        explode : bool, optional
            If True, a grid of images is created for all the items of the last two
            ensemble axes. If False, the first ensemble item is shown. May be given as a
            sequence of axis indices to create a grid of images from the specified axes.
            The default is determined by the axis metadata.
        overlay : bool or sequence of int, optional
            If True, all line profiles in the ensemble are shown in a single plot.
            If False, only the first ensemble item is shown. May be given as a sequence
            of axis indices to specify which line profiles in the ensemble to show
            together. The default is determined by the axis metadata.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to
            `matplotlib.pyplot.figure`.
        title : bool or str, optional
            Set the column title of the images. If True is given instead of a string the
            title will be given by the
            value corresponding to the "name" key of the axes metadata dictionary, if
            this item exists.
        units : str
            The units used for the x and y axes. The given units must be compatible with
            the axes of the images.
        interact : bool
            If True, create an interactive visualization. This requires enabling the
            ipympl Matplotlib backend.
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


class MomentumResolvedSpectrum(_BaseSpectra2D):
    """
    A collection of 2D measurements such as momentum-resolved energy-loss spectra.

    Parameters
    ----------
    array : np.ndarray
        2D or greater array containing data of type `float`. The
        second-to-last and last
        dimensions are the image `E`- and `q`-axis, respectively.
    sampling : two float
        Lateral sampling of images in `x` and `y` [Å].
    ensemble_axes_metadata : list of AxisMetadata, optional
        List of metadata associated with the ensemble axes. The length and item order
        must match the ensemble axes.
    metadata : dict, optional
        A dictionary defining measurement metadata.
    """

    def __init__(
        self,
        array: da.core.Array | np.array,
        values: list[float],
        axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[Dict] = None,
    ):
        self._values = values

        super().__init__(
            array=array,
            axes_metadata=axes_metadata,
            metadata=metadata,
        )

    @classmethod
    def from_array_and_metadata(
        cls,
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: Optional[dict] = None,
    ) -> "MomentumResolvedSpectrum":
        """
        Creates a momentum-resolved spectrum from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more 2D wave functions. The second-to-last and
            last dimensions are the `y`- and `x`-axis.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the
            shape of the array. The last two axes must be RealSpaceAxis.
        metadata : dict
            A dictionary defining the measurement metadata.

        Returns
        -------
        spectra : MomentumResolvedSpectrum
            MomentumResolvedSpectrum from the array and metadata.
        """
        q_axis, E_axis = axes_metadata[-2:]

        if isinstance(E_axis, EnergyAxis) and isinstance(q_axis, MomentumAxis):
            values = (E_axis.values, q_axis.values)
        else:
            raise RuntimeError()

        return cls(
            array,
            values=values,
            axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    @property
    def values(self) -> tuple[list[float], list[float]]:
        return self._values

    @property
    def offset(self) -> tuple[float, float]:
        #return 0.0, 0.0
        pass

    @property
    def extent(self) -> tuple[float, float]:
        #return (
        #    self.sampling[0] * self.base_shape[0],
        #    self.sampling[1] * self.base_shape[1],
        #)
        pass

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
    #    """Coordinates of pixels in `x` and `y` [Å]."""
    #    x = np.linspace(0.0, self.shape[-2] * self.sampling[0], self.shape[-2])
    #    y = np.linspace(0.0, self.shape[-1] * self.sampling[1], self.shape[-1])
    #    return x, y
        pass

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [
            MomentumAxis(
                label="q", values=self.values[0], units="mrad", tex_label="$mrad$"
            ),
            EnergyAxis(
                label="E", values=self.values[1], units="eV", tex_label="$eV$"
            ),
        ]

# class _BaseSpectra1D(BaseSpectra):
#     _base_dims = 1
#
#     def __init__(
#         self,
#         array: np.ndarray,
#         values: Optional[list[float]] = None,
#         axes_metadata: Optional[list[AxisMetadata]] = None,
#         metadata: Optional[dict] = None,
#     ):
#         self._values = values
#
#         super().__init__(
#             array=array,
#             axes_metadata=axes_metadata,
#             metadata=metadata,
#         )
#
#     @classmethod
#     def from_array_and_metadata(
#         cls,
#         array: np.ndarray,
#         axes_metadata: list[AxisMetadata],
#         metadata: Optional[dict] = None,
#     ) -> "BaseSpectraSubclass":
#         """
#         Creates line profile(s) from a given array and metadata.
#
#         Parameters
#         ----------
#         array : array
#             Complex array defining one or more 1D line profiles.
#         axes_metadata : list of AxesMetadata
#             Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
#             axes must be RealSpaceAxis.
#         metadata : dict, optional
#             A dictionary defining the measurement metadata.
#
#         Returns
#         -------
#         line_profiles : RealSpaceLineProfiles
#             Line profiles from the array and metadata.
#         """
#         E_axis = axes_metadata[-2]
#         if isinstance(E_axis, EnergyAxis):
#             values = E_axis.values
#         else:
#             raise RuntimeError()
#
#         axes_metadata = axes_metadata[:-1]
#         return cls(
#             array,
#             #sampling=sampling,
#             axes_metadata=axes_metadata,
#             metadata=metadata,
#         )
#
#     @property
#     def extent(self) -> float:
#         """
#         Extent of spectrum [eV].
#         """
#         #return self.sampling * self.shape[-1]
#         pass
#
#     @property
#     def values(self) -> list[float]:
#         """
#         Energy values of spectrum [eV].
#         """
#         return self._values
#
#     @property
#     @abstractmethod
#     def base_axes_metadata(self) -> list[EnergyAxis]:
#         pass
#
#     def _add_to_visualization(self, *args, **kwargs):
#         if not all(key in self.metadata for key in ("start", "end")):
#             raise RuntimeError(
#                 "The metadata does not contain the keys 'start' and 'end'"
#             )
#
#         if "width" in self.metadata:
#             kwargs["width"] = self.metadata["width"]
#
#         self._line_scan().add_to_axes(*args, **kwargs)
#
#     @staticmethod
#     def _calculate_widths(array, sampling, height):
#         xp = get_array_module(array)
#         array = array - xp.max(array, axis=-1, keepdims=True) * height
#
#         widths = xp.zeros(array.shape[:-1], dtype=np.float32)
#         for i in np.ndindex(array.shape[:-1]):
#             zero_crossings = xp.where(xp.diff(xp.sign(array[i]), axis=-1))[0]
#             left, right = zero_crossings[0], zero_crossings[-1]
#             widths[i] = (right - left) * sampling
#
#         return widths
#
#     def width(self, height: float = 0.5):
#         """
#         Calculate the width of line(s) at a given height, e.g. full width at half
#         maximum (the default).
#
#         Parameters
#         ----------
#         height : float
#             Fractional height at which the width is calculated.
#
#         Returns
#         -------
#         width : float
#             The calculated width.
#         """
#
#         if self.is_lazy:
#             return self.array.map_blocks(
#                 self._calculate_widths,
#                 drop_axis=(len(self.array.shape) - 1,),
#                 dtype=np.float32,
#                 sampling=self.sampling,
#                 height=height,
#             )
#         else:
#             return self._calculate_widths(self.array, self.sampling, height)
#
#     @staticmethod
#     def _interpolate(array, gpts, endpoint, order):
#         xp = get_array_module(array)
#         map_coordinates = get_ndimage_module(array).map_coordinates
#
#         old_shape = array.shape
#         array = array.reshape((-1, array.shape[-1]))
#
#         array = xp.pad(array, ((0,) * 2, (3,) * 2), mode="wrap")
#         new_points = xp.linspace(3.0, array.shape[-1] - 3.0, gpts, endpoint=endpoint)[
#             None
#         ]
#
#         new_array = xp.zeros(array.shape[:-1] + (gpts,), dtype=xp.float32)
#         for i in range(len(array)):
#             map_coordinates(array[i], new_points, new_array[i], order=order)
#
#         return new_array.reshape(old_shape[:-1] + (gpts,))
#
#     def interpolate(
#         self,
#         sampling: Optional[float] = None,
#         gpts: Optional[int] = None,
#         order: int = 3,
#         endpoint: bool = False,
#     ) -> BaseSpectraSubclass:
#         """
#         Interpolate line profile(s) producing equivalent line profile(s) with a different sampling. Either 'sampling' or
#         'gpts' must be provided (but not both).
#
#         Parameters
#         ----------
#         sampling : float, optional
#             Sampling of line profiles after interpolation [Å].
#         gpts : int, optional
#             Number of grid points of line profiles after interpolation. Do not use if 'sampling' is used.
#         order : int, optional
#             The order of the spline interpolation (default is 3). The order has to be in the range 0-5.
#         endpoint : bool, optional
#             If True, end is the last position. Otherwise, it is not included. Default is False.
#
#         Returns
#         -------
#         interpolated_profiles : RealSpaceLineProfiles
#             The interpolated line profile(s).
#         """
#
#         xp = get_array_module(self.array)
#
#         if (gpts is not None) and (sampling is not None):
#             raise RuntimeError()
#
#         if sampling is None and gpts is None:
#             sampling = self.sampling
#
#         if gpts is None:
#             gpts = int(np.ceil(self.extent / sampling))
#
#         if sampling is None:
#             sampling = self.extent / gpts
#
#         if self.is_lazy:
#             array = self.array.rechunk(self.array.chunks[:-1] + ((self.shape[-1],),))
#             array = array.map_blocks(
#                 self._interpolate,
#                 gpts=gpts,
#                 endpoint=endpoint,
#                 order=order,
#                 chunks=self.array.chunks[:-1] + (gpts,),
#                 meta=xp.array((), dtype=xp.float32),
#             )
#         else:
#             array = self._interpolate(self.array, gpts, endpoint, order)
#
#         kwargs = self._copy_kwargs(exclude=("array",))
#         kwargs["array"] = array
#         kwargs["sampling"] = sampling
#         return self.__class__(**kwargs)
#
#     def show(
#         self,
#         ax: Optional[Axes] = None,
#         common_scale: bool = True,
#         explode: bool | Sequence[int] = False,
#         overlay: Optional[bool | Sequence[int]] = None,
#         figsize: Optional[tuple[int, int]] = None,
#         title: str = True,
#         units: Optional[str] = None,
#         legend: bool = False,
#         interact: bool = False,
#         display: bool = True,
#         **kwargs,
#     ) -> Visualization:
#         """
#         Show the reciprocal-space line profile(s) using matplotlib.
#
#         Parameters
#         ----------
#         ax : matplotlib Axes, optional
#             If given the plots are added to the Axes. This is not available for image grids.
#         common_scale : bool
#             If True all plots are shown with a common y-axis. Default is False.
#         explode : bool or sequence of bool, optional
#             If True, a grid of plots is created for all the items of the last two ensemble axes. If False, only the
#             one plot is created. May be given as a sequence of axis indices to create a grid of plots from the specified
#             axes. The default is determined by the axis metadata.
#         overlay : bool or sequence of int, optional
#             If True, all line profiles in the ensemble are shown in a single plot. If False, only the first ensemble
#             item is shown. May be given as a sequence of axis indices to specify which line profiles in the ensemble to
#             show together. The default is determined by the axis metadata.
#         figsize : two int, optional
#             The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
#         title : bool or str, optional
#             Set the column title of the plots. If True is given instead of a string the title will be given by the value
#             corresponding to the "name" key of the axes metadata dictionary, if this item exists.
#         legend : bool
#             Add a legend to the plot. The labels will be derived from
#         units : str, optional
#             The units used for the x-axis. The given units must be compatible.
#         interact : bool
#             If True, create an interactive visualization. This requires enabling the ipympl Matplotlib backend.
#         display : bool, optional
#             If True (default) the figure is displayed immediately.
#
#         Returns
#         -------
#         visualization : Visualization
#         """
#
#         if overlay is None and explode is False:
#             overlay = True
#         elif overlay is False or overlay is None:
#             overlay = ()
#
#         visualization = Visualization(
#             measurement=self,
#             ax=ax,
#             figsize=figsize,
#             title=title,
#             aspect=False,
#             share_x=True,
#             share_y=common_scale,
#             explode=explode,
#             overlay=overlay,
#             interactive=not interact and display,
#             legend=legend,
#             common_scale=common_scale,
#             **kwargs,
#         )
#
#         if interact:
#             gui = visualization.interact(LinesGUI, display=display)
#
#         if common_scale is False and visualization._explode:
#             visualization.axes.set_sizes(padding=0.8)
#
#         return visualization