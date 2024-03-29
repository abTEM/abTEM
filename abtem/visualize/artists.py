from __future__ import annotations

import itertools
from abc import abstractmethod, ABCMeta
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, artist
from matplotlib.axes import Axes
from matplotlib.collections import (
    CircleCollection,
    Collection,
    mpath,
    transforms,
)
from matplotlib.transforms import Affine2D
from matplotlib.colors import Colormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core import config
from abtem.core.axes import LinearAxis
from abtem.core.colors import hsluv_cmap
from abtem.core.units import _get_conversion_factor
from abtem.core.delegate import DelegateTo

if TYPE_CHECKING:
    from matplotlib.text import Annotation
    from abtem.measurements import (
        IndexedDiffractionPatterns,
        DiffractionPatterns,
        MeasurementsEnsemble,
    )


def _get_norm(vmin=None, vmax=None, power=1.0, logscale=False):
    if (power == 1.0) and (logscale is False):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif (power != 1.0) and (logscale is False):
        norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)
    elif (power == 1.0) and (logscale is True):
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("")

    return norm


def _get_value_limits(array, value_limits: tuple[float, float] = None, margin=None):
    if np.iscomplexobj(array):
        array = np.abs(array)

    if value_limits is None:
        value_limits = [None, None]

    value_limits = list(value_limits).copy()

    if value_limits[0] is None:
        value_limits[0] = float(np.nanmin(array))

    if value_limits[1] is None:
        value_limits[1] = float(np.nanmax(array))

    if margin:
        margin = (value_limits[1] - value_limits[0]) * margin
        value_limits[0] -= margin
        value_limits[1] += margin

    return value_limits


class ScaleBar:
    def __init__(
        self,
        ax: Axes,
        label: str = "",
        size: float = None,
        loc: str = "lower right",
        borderpad: float = 0.5,
        size_vertical: float = None,
        sep: float = 6,
        pad: float = 0.3,
        label_top: bool = True,
        frameon: bool = False,
        **kwargs,
    ):
        if size is None:
            xlim = ax.get_xlim()
            size = (xlim[1] - xlim[0]) / 3

        if size_vertical is None:
            ylim = ax.get_ylim()
            size_vertical = (ylim[1] - ylim[0]) / 20

        self._anchored_size_bar = AnchoredSizeBar(
            ax.transData,
            label=label,
            label_top=label_top,
            size=size,
            borderpad=borderpad,
            loc=loc,
            size_vertical=size_vertical,
            sep=sep,
            pad=pad,
            frameon=frameon,
            **kwargs,
        )
        ax.add_artist(self._anchored_size_bar)


class AreaIndicator:
    def __init__(
        self,
        ax,
        xy,
        width,
        height,
        alpha: float = 0.33,
        facecolor: str = "r",
        edgecolor: str = "r",
        **kwargs,
    ):
        rect = Rectangle(
            xy,
            width,
            height,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        ax.add_patch(rect)


class Artist(metaclass=ABCMeta):
    def __init__(self, ax, measurement):
        self._ax = ax
        self._measuremet = measurement.compute()

    @abstractmethod
    def get_xlim(self):
        pass

    @abstractmethod
    def get_ylim(self):
        pass

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self, power=1.0):
        pass

    def set_logscale(self):
        pass

    @abstractmethod
    def get_power(self):
        pass

    @abstractmethod
    def remove(self):
        pass

    def set_xlabel(self, label):
        self._ax.set_xlabel(label)

    def set_ylabel(self, label):
        self._ax.set_ylabel(label)

    def set_ylim(self, ylim):
        self._ax.set_ylim(ylim)

    def set_xlim(self, xlim):
        self._ax.set_xlim(xlim)


class Artist1D(Artist):
    pass


class LinesArtist(Artist1D):
    num_cbars = 0

    def __init__(
        self,
        ax: Axes,
        measurement,
        caxes: list[Axes] = None,
        label=None,
        units: str = None,
        legend: bool = False,
        **kwargs,
    ):
        super().__init__(ax=ax, measurement=measurement)

        y = self._reshape_data(measurement.array)
        x = measurement.base_axes_metadata[-1].coordinates(measurement.shape[-1])

        if label is None and measurement.ensemble_shape:
            label = [
                [l.format_title("") for l in axis]
                for axis in measurement.ensemble_axes_metadata
            ]
            label = list(itertools.product(*label))
            label = [", ".join(l) for l in label]

        if np.iscomplexobj(y):
            if label is not None:
                label = [l + "(real)" for l in label] + [l + "(imag)" for l in label]
            else:
                label = ["real", "imaginary"]

        self._lines = ax.plot(x, y, label=label, **kwargs)

        xlabel = measurement.base_axes_metadata[-1].format_label(units)
        ylabel = measurement._scale_axis_from_metadata().format_label()

        self.set_xlabel(xlabel)

        if not np.iscomplexobj(ylabel):
            self.set_ylabel(ylabel)

        if label and legend:
            self.set_legend()

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)

    def remove(self):
        for line in self._lines:
            line.remove()

        self._ax.set_prop_cycle(None)

    @staticmethod
    def _reshape_data(y):
        if len(y.shape) > 1:
            y = np.moveaxis(y, -1, 0).reshape((y.shape[-1], -1))

        if np.iscomplexobj(y):
            if len(y.shape) == 2:
                y = np.concatenate((y.real, y.imag), axis=-1)
            else:
                y = np.stack((y.real, y.imag), axis=-1)

        return y

    def get_xlim(self):
        xlim = [np.inf, -np.inf]
        for line in self._lines:
            data = line.get_data()[0]
            new_xlim = [np.min(data), np.max(data)]
            xlim = [min(new_xlim[0], xlim[0]), max(new_xlim[1], xlim[1])]
        ptp = xlim[1] - xlim[0]
        return [xlim[0] - 0.05 * ptp, xlim[1] + 0.05 * ptp]

    def get_ylim(self):
        ylim = [np.inf, -np.inf]
        for line in self._lines:
            data = line.get_data()[1]
            new_ylim = [np.min(data), np.max(data)]
            ylim = [min(new_ylim[0], ylim[0]), max(new_ylim[1], ylim[1])]
        ptp = ylim[1] - ylim[0]
        ptp = max(ptp, ylim[1] * 0.01)
        return [ylim[0] - 0.05 * ptp, ylim[1] + 0.05 * ptp]

    def get_value_limits(self):
        return self.get_ylim()

    def set_data(self, measurement):
        y = self._reshape_data(measurement.array)
        x = measurement.base_axes_metadata[-1].coordinates(measurement.shape[-1])

        if len(y.shape) == 1:
            y = y[..., None]

        for i, line in enumerate(self._lines):
            line.set_data(x, y[..., i])

    def get_logscale(self):
        return self._ax.set_yscale("log")

    def set_logscale(self):
        self._ax.set_yscale("log")

    def get_power(self):
        return 1.0

    def set_power(self, power: float = 1.0) -> None:
        raise NotImplementedError
        #
        # def forward(x):
        #     if power == 1:
        #         return x
        #
        #     return x ** (1 / power)
        #
        # def inverse(x):
        #     if power == 1:
        #         return x
        #
        #     return x**power
        #
        # if power == 1.0:
        #     self._ax.set_yscale("linear")
        # else:
        #     self._ax.set_yscale("function", functions=(forward, inverse))
        #     self._ax.set_yscale("function", functions=(forward, inverse))

    def set_value_limits(self, value_limits: list[float] = None):
        data = np.stack([line.get_data()[1] for line in self._lines], axis=0)
        value_limits = _get_value_limits(data, value_limits, margin=0.05)
        self._ax.set_ylim(value_limits)

    def set_legend(self, **kwargs):
        self._ax.legend(**kwargs)


class Artist2D(Artist):
    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self, power: float = 1.0):
        pass

    @abstractmethod
    def set_cmap(self, cmap):
        pass

    @abstractmethod
    def set_cbars(self, cmap):
        pass

    def set_logscale(self):
        pass

    @staticmethod
    def _set_vmin_vmax(norm, vmin, vmax, array=None):
        if vmin is None and array is not None:
            vmin = array.min()

        if vmax is None and array is not None:
            vmax = array.max()

        with norm.callbacks.blocked():
            norm.vmin = vmin
        norm.vmax = vmax

    @staticmethod
    def _update_norm(old_norm, power, artist):
        if (power != 1.0) and isinstance(old_norm, colors.PowerNorm):
            old_norm.gamma = power
            artist.norm = old_norm
            old_norm._changed()
        else:
            norm = _get_norm(vmin=old_norm.vmin, vmax=old_norm.vmax, power=power)
            artist.norm = norm

    @staticmethod
    def _make_cbar(mappable, cax, **kwargs):
        return plt.colorbar(mappable, cax=cax, **kwargs)

    @abstractmethod
    def get_ylim(self):
        pass

    @abstractmethod
    def get_xlim(self):
        pass

    @property
    @abstractmethod
    def num_cbars(self):
        pass

    @abstractmethod
    def get_value_limits(self):
        pass

    def set_scale_bars(self, **kwargs):
        self._scale_bar = ScaleBar(ax=self._ax, **kwargs)

    def add_area_indicator(self, area_indicator, panel="first", **kwargs):
        for i, ax in enumerate(np.array(self.axes).ravel()):
            if panel == "first" and i == 0:
                area_indicator._add_to_visualization(ax, **kwargs)
            elif panel == "all":
                area_indicator._add_to_visualization(ax, **kwargs)


def default_cbar_scalar_formatter():
    format = ScalarFormatter(useMathText=True)
    format.set_powerlimits((-3, 3))
    return format


def validate_cmap(cmap, measurement, complex_conversion="none"):
    if cmap is None:
        if measurement.is_complex and complex_conversion in ("none", "phase"):
            cmap = config.get("visualize.phase_cmap", "hsluv")
        else:
            cmap = config.get("visualize.cmap", "viridis")

    if cmap == "hsluv":
        cmap = hsluv_cmap
    elif isinstance(cmap, str) and cmap[:5] == "solid":
        cmap = colors.ListedColormap([cmap.split(" ")[-1]])

    return cmap


def get_extent(measurement, units=None):
    energy = measurement.metadata.get("energy", None)

    conversion_x = _get_conversion_factor(
        units, measurement.base_axes_metadata[0].units, energy=energy
    )
    conversion_y = _get_conversion_factor(
        units, measurement.base_axes_metadata[0].units, energy=energy
    )

    left = (
        measurement.base_axes_metadata[0].offset
        - measurement.base_axes_metadata[0].sampling / 2
    ) * conversion_x
    right = (
        left
        + measurement.base_axes_metadata[0].sampling
        * measurement.base_shape[0]
        * conversion_x
    )
    bottom = (
        measurement.base_axes_metadata[1].offset
        - measurement.base_axes_metadata[1].sampling / 2
    ) * conversion_y
    top = (
        bottom
        + measurement.base_axes_metadata[1].sampling
        * measurement.base_shape[1]
        * conversion_y
    )

    return (left, right, bottom, top)


class ImageArtist(Artist2D):
    num_cbars = 1

    def __init__(
        self,
        ax: Axes,
        measurement: Images | DiffractionPatterns | MeasurementsEnsemble,
        caxes: list[Axes] = None,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        origin: Literal["upper", "lower"] | None = "lower",
        units: str = None,
        **kwargs,
    ):
        
        super().__init__(ax=ax, measurement=measurement)

        if measurement.is_complex:
            raise ValueError("Complex measurements are not supported.")

        extent = get_extent(measurement, units=units)

        cmap = validate_cmap(cmap, measurement)
        
        self._axes_image = ax.imshow(
            measurement.array.T,
            origin=origin,
            cmap=cmap,
            extent=extent,
            interpolation=kwargs.pop("interpolation", "none"),
            **kwargs,
        )
        norm = _get_norm(vmin, vmax, power, logscale)
        self._axes_image.set_norm(norm)
        self._cbar = None

        xlabel = measurement.base_axes_metadata[0].format_label(units)
        ylabel = measurement.base_axes_metadata[1].format_label(units)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)

        if caxes:
            cbar_label = measurement._scale_axis_from_metadata().format_label()
            self.set_cbars(caxes=caxes, label=cbar_label)

    @property
    def axes_image(self):
        return self._axes_image

    @property
    def norm(self):
        return self.axes_image.norm

    def remove(self):
        self.axes_image.remove()

    def get_power(self):
        if hasattr(self.norm, "gamma"):
            return self.norm.gamma
        else:
            return 1.0

    def get_value_limits(self):
        array = self.axes_image.get_array()
        return [array.min(), array.max()]

    def get_xlim(self):
        return self.axes_image.get_extent()[:2]

    def get_ylim(self):
        return self.axes_image.get_extent()[2:]

    def set_cbars(self, caxes, **kwargs):
        format = kwargs.pop("format", default_cbar_scalar_formatter())
        cbar = self._make_cbar(self.axes_image, caxes[0], format=format, **kwargs)
        cbar.ax.yaxis.set_offset_position("left")

    def set_cmap(self, cmap):
        self.axes_image.set_cmap(cmap)

    def set_data(self, data):
        self.axes_image.set_data(data._array.T)

    def set_extent(self, extent):
        self.axes_image.set_extent(extent)

    def set_power(self, power: float = 1.0):
        self._update_norm(self.norm, power, self.axes_image)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._set_vmin_vmax(self.norm, *value_limits)


class ScaledCircleCollection(Collection):

    def __init__(self, array, scale=1., **kwargs):
        super().__init__(array=array, **kwargs)
        self.set_transform(transforms.IdentityTransform())
        self._transforms = np.empty((0, 3, 3))
        self._paths = [mpath.Path.unit_circle()]
        self._scale = scale
        self._radii = self._calculate_radii()
        self._set_transforms()
        self.callbacks.connect("changed", lambda *args: self._update_radii())
        
    def _set_transforms(self):
        ax = self.axes
        self._transforms = np.zeros((len(self._radii), 3, 3))
        self._transforms[:, 0, 0] = self._radii
        self._transforms[:, 1, 1] = self._radii
        self._transforms[:, 2, 2] = 1.0

        if ax is not None:
            A = ax.transData.get_affine().get_matrix().copy()
            A[:2, 2:] = 0
            self.set_transform(transforms.Affine2D(A))

    @artist.allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)

    def get_radii(self):
        return self._radii
    
    def set_norm(self, norm):
        super().set_norm(norm)
        self.norm.callbacks.connect("changed", lambda *args: self._update_radii())

    def set_array(self, array):
        super().set_array(array)
        self.changed()
    
    def _calculate_radii(self):
        norm = self.norm
        data = self.get_array()
        radii = np.sqrt(norm(data) * self._scale)
        return radii

    def _update_radii(self):
        self._radii = self._calculate_radii()

    def get_scale(self):
        return self._scale
    
    def set_scale(self, scale):
        self._scale = scale
        self._update_radii()
        self.changed()


class CircleAnnotations:
    _placement_to_alignment = {"top":"bottom", "center":"center", "bottom":"top"}
    
    def __init__(self, circle_collection, annotations, threshold:float=0, fontsize=8, placement="top", **kwargs):
        self._circle_collection = circle_collection
        self._threshold = threshold
        self._placement = placement

        ax = circle_collection.axes
        positions = self._get_positions()
        visibilities = self._get_visibilities()

        self._annotations = []
        for annotation, position, visible in zip(annotations, positions, visibilities):
            self._annotations.append(
                ax.annotate(
                    annotation,
                    xy=position,
                    ha="center",
                    va=self._placement_to_alignment.get(placement),
                    visible=visible,
                    fontsize=fontsize,
                    **kwargs,
                )
            )
        circle_collection.callbacks.connect("changed", lambda *args: self._update_visibilities())
        circle_collection.callbacks.connect("changed", lambda *args: self._update_positions())

    def __getattr__(self, name):
        try:
            super(self.__class__).__getattr__(name)
        except AttributeError:
            pass

        def method(*args, **kwargs):
            return tuple(getattr(annotation, name)(*args, **kwargs) for annotation in self._annotations)
        
        return method 
    
    def get_threshold(self):
        return self._threshold
        
    def set_threshold(self, threshold):
        self._threshold = threshold
        self._update_visibilities()

    def set_placement(self, placement):
        self._placement = placement
        self.set_verticalalignment(self._placement_to_alignment[placement])
        self._update_positions()
    
    def _get_visibilities(self):
        return self._circle_collection.get_array() > self._threshold
    
    def _get_positions(self):
        positions = self._circle_collection.get_offsets().copy()
        radii = self._circle_collection.get_radii()

        if self._placement == "top":
            positions[:, 1] += radii
        elif self._placement == "bottom":
            positions[:, 1] -= radii
        elif self._placement != "center":
            raise ValueError()

        return positions
    
    def _update_visibilities(self):
        visibilities = self._get_visibilities()
        for annotation, visible in zip(self._annotations, visibilities):
            annotation.set_visible(visible)
    
    def _update_positions(self):
        positions = self._get_positions()
        for annotation, position in zip(
            self._annotations, positions
        ):
            annotation.set_position(position)


class ScatterArtist(Artist2D):
    num_cbars = 1

    def __init__(
        self,
        ax: Axes,
        measurement: IndexedDiffractionPatterns,
        caxes: list[Axes] = None,
        cmap: str | Colormap | None = None,
        value_limits: tuple(float, float) = None,
        power: float = 1.0,
        logscale: bool = False,
        units: str = None,
        scale: float = 0.5,
        annotation_threshold: float = 0.1,
        **kwargs,
    ):

        super().__init__(ax=ax, measurement=measurement)

        vmin, vmax = _get_value_limits(measurement.array, value_limits=value_limits)
        norm = _get_norm(vmin, vmax, power, logscale)

        energy = measurement.metadata.get("energy", None)

        self._unit_conversion = _get_conversion_factor(
            units, old_units="1/Å", energy=energy
        )
        
        cmap = validate_cmap(cmap, measurement)

        self._circles = ScaledCircleCollection(
            array=measurement.array,
            cmap=cmap,
            offsets=measurement.positions[:, :2] * self._unit_conversion,
            transOffset=ax.transData,
            norm=norm,
            scale=scale,
            **kwargs,
        )

        ax.add_collection(self._circles)

        units = "1/Å" if units is None else units
        
        x_axis = LinearAxis(label="k_x", units=units, _tex_label="$k_x$")
        y_axis = LinearAxis(label="k_y", units=units, _tex_label="$k_y$")
        
        self.set_xlabel(x_axis.format_label(units))
        self.set_ylabel(y_axis.format_label(units))
        
        annotations = []
        for hkl in measurement.miller_indices:
            if config.get("visualize.use_tex"):
                annotation = " \ ".join(
                    [f"\\bar{{{abs(i)}}}" if i < 0 else f"{i}" for i in hkl]
                )
                annotations.append(f"${annotation}$")
            else:
                annotations.append("{} {} {}".format(*hkl))

        self._annotations = CircleAnnotations(self._circles, annotations, annotation_threshold)

        if caxes:
            cbar_label = measurement._scale_axis_from_metadata().format_label()
            self.set_cbars(caxes=caxes, label=cbar_label)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        
        if hasattr(self._circles, name):
            return getattr(self._circles, name)
        
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
    
    @property
    def circle_collection(self) -> FixedCircleCollection:
        return self._circles

    @property
    def annotations(self):
        return self._annotations 
    
    @property
    def annotations(self) -> list[Annotation]:
        return self._annotations

    def get_ylim(self):
        return [self.get_offsets()[:, 1].min() * 1.1, self.get_offsets()[:, 1].max() * 1.1]

    def get_xlim(self):
        return [self.get_offsets()[:, 0].min() * 1.1, self.get_offsets()[:, 0].max() * 1.1]

    def get_value_limits(self):
        array = self.circle_collection.get_array()
        return [array.min(), array.max()]

    def remove(self):
        self.circle_collection.remove()

    def get_power(self):
        if hasattr(self.norm, "gamma"):
            return self.norm.gamma
        else:
            return 1.0

    def set_data(self, measurement: np.ndarray):
        self._circles.set_array(measurement.array)
        self._circles.set_offsets(measurement.positions[:, :2] * self._unit_conversion)

    def set_annotations(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self._annotations, f"set_{k}")(**{k: v})
    
    def get_annotation_threshold(self):
        return self._annotations.get_annotation_threshold()
    
    def set_annotation_threshold(self, threshold):
        self._annotations.set_threshold(threshold=threshold)
    
    def get_scale(self):
        return self.circle_collection.get_scale()
    
    def set_scale(self, scale: float):
        self.circle_collection.set_scale(scale)

    def set_cmap(self, cmap: str):
        self.circle_collection.set_cmap(cmap)

    def set_cbars(self, caxes=None, **kwargs):
        self._make_cbar(self.circle_collection, caxes[0], **kwargs)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        data = self._circles.get_array().data
        self._set_vmin_vmax(self.norm, *value_limits, data)

    def set_power(self, power: float = 1.0):
        self._update_norm(self.norm, power, self._circles)


class DomainColoringArtist(Artist2D):
    num_cbars = 2

    def __init__(
        self,
        ax: Axes,
        measurement,
        caxes: list[Axes] = None,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        units: str = None,
        **kwargs,
    ):
        super().__init__(ax=ax, measurement=measurement)

        norm = _get_norm(vmin, vmax, power, logscale)

        abs_array = np.abs(measurement.array)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        extent = get_extent(measurement, units=None)

        cmap = validate_cmap(cmap, measurement)

        self._phase_axes_image = ax.imshow(
            np.angle(measurement.array).T,
            origin="lower",
            interpolation=kwargs.pop("interpolation", "none"),
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
            extent=extent,
            **kwargs,
        )
        self._amplitude_axes_image = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation=kwargs.pop("interpolation", "none"),
            cmap="gray",
            zorder=-1,
            extent=extent,
            **kwargs,
        )

        self._amplitude_axes_image.set_norm(norm)
        self._amplitude_cbar = None
        
        self.set_xlabel(measurement.base_axes_metadata[0].format_label(units))
        self.set_ylabel(measurement.base_axes_metadata[1].format_label(units))

        if caxes is not None and len(caxes):
            cbar_label = measurement._scale_axis_from_metadata().format_label()
            self.set_cbars(caxes, label=cbar_label)

    def remove(self):
        self.amplitude_axes_image.remove()
        self.phase_axes_image.remove()

    def get_power(self):
        if hasattr(self.amplitude_norm, "gamma"):
            return self.amplitude_norm.gamma
        else:
            return 1.0

    def get_value_limits(self):
        array = self.amplitude_axes_image.get_array()
        return [array.min(), array.max()]

    def get_xlim(self):
        return self.amplitude_axes_image.get_extent()[:2]

    def get_ylim(self):
        return self.amplitude_axes_image.get_extent()[2:]

    @property
    def amplitude_norm(self):
        return self.amplitude_axes_image.norm

    @property
    def amplitude_axes_image(self):
        return self._amplitude_axes_image

    @property
    def phase_axes_image(self):
        return self._phase_axes_image

    def _update_alpha(self):
        data = self.amplitude_axes_image.get_array().data
        alpha = self.amplitude_axes_image.norm(np.abs(data))
        alpha = np.clip(alpha, a_min=0, a_max=1)
        self.phase_axes_image.set_alpha(alpha)

    def set_value_limits(self, value_limits: tuple[float, float] = (None, None)):
        self._set_vmin_vmax(self.amplitude_norm, *value_limits)
        self._update_alpha()
        if self._amplitude_cbar:
            self._amplitude_cbar.ax.yaxis.set_offset_position("left")

    def set_cmap(self, cmap):
        self.phase_axes_image.set_cmap(cmap)
        self._phase_cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self._phase_cbar.set_ticklabels(
            [
                "$-\pi$",
                "$-\dfrac{\pi}{2}$",
                "$0$",
                "$\dfrac{\pi}{2}$",
                "$\pi$",
            ]
        )

    def set_power(self, power: float = 1.0):
        self._update_norm(self.amplitude_norm, power, self.amplitude_axes_image)
        self._update_alpha()

    def set_extent(self, extent):
        self.phase_axes_image.set_extent(extent)
        self.amplitude_axes_image.set_extent(extent)

    def set_data(self, data):
        abs_array = np.abs(data._array)
        alpha = self.amplitude_norm(abs_array)
        alpha = np.clip(alpha, a_min=0, a_max=1)

        self.phase_axes_image.set_alpha(alpha.T)
        self.phase_axes_image.set_data(np.angle(data._array).T)
        self.amplitude_axes_image.set_data(abs_array.T)

    def set_cbars(self, caxes, label=None, **kwargs):
        if caxes is None:
            caxes = [None, None]

        self._phase_cbar = self._make_cbar(
            self.phase_axes_image, cax=caxes[0], **kwargs
        )

        format = default_cbar_scalar_formatter()

        self._amplitude_cbar = self._make_cbar(
            self.amplitude_axes_image, cax=caxes[1], format=format, **kwargs
        )

        self._phase_cbar.set_label("arg", rotation=0, ha="center", va="top")
        self._phase_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
        self._phase_cbar.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        self._phase_cbar.set_ticklabels(
            [
                "$-\pi$",
                "$-\dfrac{\pi}{2}$",
                "$0$",
                "$\dfrac{\pi}{2}$",
                "$\pi$",
            ]
        )

        self._amplitude_cbar.set_label("abs", rotation=0, ha="center", va="top")
        self._amplitude_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
        self._amplitude_cbar.ax.yaxis.set_offset_position("left")


class OverlayImshowArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
    ):
        raise NotImplementedError

        # cmaps = [ListedColormap(c) for c in cmap]
        #
        # alphas = [np.clip(norm(alpha), a_min=0.0, a_max=1.0) for alpha in array]
        # # print(alphas[0], array.shape, array[0].shape)
        # ims = [
        #     ax.imshow(
        #         np.ones_like(alpha.T),
        #         origin="lower",
        #         interpolation="none",
        #         cmap=cmap,
        #         alpha=alpha.T,
        #     )
        #     for alpha, cmap in zip(alphas, cmaps)
        # ]
        #
        # ax.set_facecolor("k")
        #
        #
        # from matplotlib import colors
        # from matplotlib.cm import ScalarMappable
        # from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        #
        # fig, ax = plt.subplots()
        # ax.set_facecolor("k")
        #
        # cmap = ListedColormap(["lime"])
        # norm = colors.Normalize()
        # norm.autoscale_None(stacked.array[0])
        # alpha = norm(stacked.array[0])
        #
        # im = ax.imshow(
        #     np.ones_like(stacked.array[0]).T, alpha=alpha.T, cmap=cmap, origin="lower"
        # )
        #
        # cmap = LinearSegmentedColormap.from_list("red", ["k", "lime"])
        # plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        #
        # c = "r"
        # cmap = ListedColormap([c])
        # norm = colors.Normalize()
        # norm.autoscale_None(stacked.array[1])
        # alpha = norm(stacked.array[1])
        #
        # im = ax.imshow(
        #     np.ones_like(stacked.array[1]).T, alpha=alpha.T, cmap=cmap, origin="lower"
        # )
        #
        # cmap = LinearSegmentedColormap.from_list("c", ["k", c])
        # plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        #
