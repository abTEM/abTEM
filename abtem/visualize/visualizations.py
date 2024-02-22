"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import Sequence, TYPE_CHECKING

import ipywidgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, EllipseCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.axes import ScaleAxis
from abtem.core.backend import get_array_module
from abtem.core.colors import hsluv_cmap
from abtem.core.utils import label_to_index, flatten_list_of_lists
from abtem.visualize.axes_grid import (
    _determine_axes_types,
    _validate_axes,
)
from abtem.visualize.widgets import (
    make_sliders_from_ensemble_axes,
    make_scale_button,
    make_autoscale_button,
    make_power_scale_slider,
    make_cmap_dropdown,
    make_complex_visualization_dropdown,
)

if TYPE_CHECKING:
    from abtem.core.axes import AxisMetadata


def _format_options(options):
    formatted_options = []
    for option in options:
        if isinstance(option, float):
            formatted_options.append(f"{option:.3f}")
        elif isinstance(option, tuple):
            formatted_options.append(
                ", ".join(tuple(f"{value:.3f}" for value in option))
            )
        else:
            formatted_options.append(option)

    return formatted_options


def discrete_cmap(num_colors, base_cmap):
    if isinstance(base_cmap, str):
        base_cmap = plt.get_cmap(base_cmap)
    colors = base_cmap(range(0, num_colors))
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors, num_colors)


def _get_joined_titles(measurement, formatting, **kwargs):
    titles = []
    for axes_metadata in measurement.ensemble_axes_metadata:
        titles.append(axes_metadata.format_title(formatting, **kwargs))
    return "\n".join(titles)


def _get_overlay_labels(axes_metadata, axes_types):
    labels = [
        [i.format_title(".3f") for i in axis]
        for axis, axis_type in zip(axes_metadata, axes_types)
        if axis_type == "overlay"
    ]
    labels = list(itertools.product(*labels))
    return [" - ".join(label) for label in labels]


def _get_axes_titles(axes_metadata, axes_types, axes_shape):
    titles = []

    for axis, axis_type, n in zip(axes_metadata, axes_types, axes_shape):
        if axis_type == "explode":
            axis_titles = []
            if hasattr(axis, "__len__"):
                assert len(axis) == n

                for i, axis_element in enumerate(axis):
                    title = axis_element.format_title(".3g", include_label=i == 0)
                    axis_titles.append(title)

            else:
                axis_titles = [""] * n
            titles.append(axis_titles)

    titles_product = list(itertools.product(*titles))
    return titles_product


def _get_coordinate_limits(axes_metadata: list[AxisMetadata], shape, margin=None):
    # shape = self._array.shape[-self._num_coordinate_dims :]
    limits = []
    for axis, n in zip(axes_metadata, shape):
        coordinates = axis.coordinates(n)

        min_limit = coordinates[0] - axis.sampling / 2
        max_limit = coordinates[-1] + axis.sampling / 2

        if margin:
            margin = (max_limit - min_limit) * margin
            min_limit -= margin
            max_limit += margin

        limits.append([min_limit, max_limit])
    return limits


def _get_value_limits(array, value_limits, margin=None):
    if np.iscomplexobj(array):
        array = np.abs(array)

    value_limits = value_limits.copy()
    if value_limits[0] is None:
        value_limits[0] = float(np.nanmin(array))

    if value_limits[1] is None:
        value_limits[1] = float(np.nanmax(array))

    if margin:
        margin = (value_limits[1] - value_limits[0]) * margin
        value_limits[0] -= margin
        value_limits[1] += margin

    return value_limits


class BaseVisualization:
    # _num_coordinate_dims: int = None

    def __init__(
        self,
        array,
        coordinate_axes,
        ax=None,
        scale_axis=None,
        ensemble_axes_metadata=None,
        common_scale: bool = False,
        figsize=None,
        aspect: bool = False,
        explode: bool = False,
        overlay: bool = False,
        coordinate_labels: list[str] = None,
        scale_label: str = None,
        ncbars: int = 0,
        interact: bool = False,
        share_x: bool = False,
        share_y: bool = False,
        title: bool | str = True,
        limits_margin: float = 0.0,
    ):
        xp = get_array_module(array)

        if hasattr(xp, "asnumpy"):
            array = xp.asnumpy(array)

        self._common_scale = common_scale
        self._array = array
        self._coordinate_axes = coordinate_axes

        # if coordinate_labels is None:
        #     coordinate_labels = [None] * len(coordinate_axes)
        #
        # if label is not None:
        #     self._coordinate_labels[0] = label
        #
        # if self._coordinate_labels[0] is None:
        #     label = self._coordinate_axes[0].format_label()
        # else:
        #     label = self._coordinate_labels[0]
        #
        #
        # self._coordinate_labels = coordinate_labels

        if scale_axis is None:
            scale_axis = ScaleAxis()

        self._scale_axis = scale_axis
        self._scale_label = scale_label

        self._autoscale = config.get("visualize.autoscale", False)
        self._num_coordinate_dims = len(coordinate_axes)

        # assert len(coordinate_axes) == self._num_coordinate_dims

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._ensemble_shape = array.shape[: -self._num_coordinate_dims]
        self._limits_margin = limits_margin

        self._axes_types = _determine_axes_types(
            ensemble_axes_metadata, explode=explode, overlay=overlay
        )

        self._axes = _validate_axes(
            self._axes_types,
            self._ensemble_shape,
            ax=ax,
            ioff=interact,
            explode=explode,
            aspect=aspect,
            ncbars=ncbars,
            common_color_scale=common_scale,
            figsize=figsize,
            sharex=share_x,
            sharey=share_y,
        )
        self._indices = self._validate_ensemble_indices()

        self._complex_conversion = "none"
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._artists = None

        self.get_figure().canvas.header_visible = False

        # if self.axes.ncols > 1 and title:
        #     self.set_column_titles(title)
        #
        # if self.axes.nrows > 1 and title:
        #     self.set_row_titles()

    def _get_array_for_scaling(self):
        return self._array

    def get_figure(self):
        return self.axes[0, 0].get_figure()

    @property
    def autoscale(self):
        return self._autoscale

    @autoscale.setter
    def autoscale(self, autoscale: bool):
        self._autoscale = autoscale
        self.set_values_lim()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    @property
    def ensemble_shape(self):
        return self._ensemble_shape

    @property
    def array(self):
        return self._array

    @property
    def artists(self):
        return self._artists

    @property
    def axes_types(self) -> list[str]:
        return self._axes_types

    @property
    def axes(self):
        return self._axes

    def _validate_ensemble_indices(self, indices: int | tuple[int, ...] = ()):
        num_indexing_axes = sum(
            1 for axes_type in self._axes_types if axes_type in ("index", "range")
        )

        if len(indices) > num_indexing_axes:
            raise ValueError

        if isinstance(indices, int):
            indices = (indices,)

        validated_indices = []
        j = 0
        for i, axes_type in enumerate(self.axes_types):
            if axes_type in ("explode", "overlay"):
                validated_indices.append(slice(None))
            elif j < len(indices):
                validated_indices.append(indices[j])
                j += 1
            elif axes_type == "index":
                validated_indices.append(0)
            elif axes_type == "range":
                validated_indices.append(slice(None))
            else:
                raise RuntimeError(
                    "axes type must be one of 'index', 'range', 'explode' or 'overlay'"
                )

        return tuple(validated_indices)

    def set_ensemble_indices(self, indices: int | tuple[int, ...] = ()):
        """
        Set the indices into the ensemble dimensions to select the visualized ensemble members. Interactive
        visualization are updated.

        Parameters
        ----------
        indices : int or tuple of int

        """

        self._indices = self._validate_ensemble_indices(indices)
        self.update_artists()
        # self.update_panel_labels()

        #
        # titles = [
        #     [
        #         axis_element.format_title(".3g", include_label=i == 0)
        #         for i, axis_element in enumerate(axis)
        #     ]
        #     for axis, axis_type in zip(self.ensemble_axes_metadata, self.axes_types)
        #     if axis_type == "explode"
        # ]
        #
        # titles = list(itertools.product(*titles))
        # return titles

    def _get_array_for_current_indices(self):
        array = self._array[self._indices]

        summed_axes = tuple(
            i for i, axes_type in enumerate(self._axes_types) if axes_type == "range"
        )

        array = array.sum(axis=summed_axes)

        if np.iscomplexobj(array) and self._complex_conversion not in (
            "domain_coloring",
            "none",
        ):
            if self._complex_conversion == "abs":
                array = np.abs(array)
            elif self._complex_conversion == "intensity":
                array = np.abs(array) ** 2
            elif self._complex_conversion == "phase":
                array = np.angle(array)
            elif self._complex_conversion == "real":
                array = array.real
            elif self._complex_conversion == "imag":
                array = array.imag
            else:
                raise NotImplementedError

        return array

    def _get_array_for_axis(self, axis_index):
        array = self._get_array_for_current_indices()

        num_explode_axes = sum(axis_type == "explode" for axis_type in self.axes_types)
        axes_types = [
            axis_type
            for axis_type in self.axes_types
            if axis_type in ("explode", "overlay")
        ]

        if num_explode_axes == 0:
            return array
        elif num_explode_axes == 1:
            i = sum(axis_index)
            index = tuple(
                i if axis_type == "explode" else slice(None) for axis_type in axes_types
            )
        elif num_explode_axes == 2:
            iter_axis_index = iter(axis_index)
            index = tuple(
                next(iter_axis_index) if axis_type == "explode" else slice(None)
                for axis_type in self.axes_types
            )
        else:
            raise NotImplementedError

        array = array[index]

        return array

    # def _validate_value_limits(self, value_limits: list[float, float] = None):
    #     if value_limits is None:
    #         fixed_value_lim = [None, None]
    #     else:
    #         fixed_value_lim = value_limits.copy()
    #
    #     if self._common_scale:
    #         array = self._get_array_for_current_indices()
    #         common_ylim = _get_value_limits(array, fixed_value_lim, self._limits_margin)
    #     else:
    #         common_ylim = None
    #
    #     value_limits = np.zeros(self.axes.shape + (2,))
    #     for i in np.ndindex(self.axes.shape):
    #         array = self._get_array_for_axis(i)
    #         if common_ylim is None:
    #             value_limits[i] = _get_value_limits(
    #                 array, fixed_value_lim, self._limits_margin
    #             )
    #         else:
    #             value_limits[i] = common_ylim
    #
    #     return value_limits

    def _get_coordinate_labels(self):
        return [axis.format_label(units) for axis, units in zip(self._coordinate_axes)]

    def set_xlabel(self, label: str = None):
        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_xlabel(label)

    def set_ylabel(self, label: str = None):
        # if self._num_coordinate_dims == 1:
        #     if label is not None:
        #         self._scale_label = label
        #
        #     if self._scale_label is None:
        #         label = self._scale_axis.format_label()
        #
        # elif self._num_coordinate_dims == 2:
        #     if label is not None:
        #         self._coordinate_labels[1] = label
        #
        #     if self._coordinate_labels[1] is None:
        #         label = self._coordinate_axes[1].format_label()
        #
        # else:
        #     raise NotImplementedError

        for i, j in np.ndindex(self.axes.shape):
            if i == 0:
                self.axes[i, j].set_ylabel(label)

    def set_ylim(self, ylim: tuple[float, float] = None):
        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_ylim(ylim)

    def set_xlim(self, xlim: tuple[float, float] = None):
        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_xlim(xlim)

    @abstractmethod
    def set_values_lim(self, values_lim: tuple[float, float] = None):
        pass

    def set_column_titles(
        self,
        titles: str | list[str] | bool = None,
        pad: float = 10.0,
        format: str = ".3g",
        units: str = None,
        fontsize: float = 12,
        **kwargs,
    ):
        if titles is None or titles is True:
            titles = [
                title[0]
                for title in _get_axes_titles(
                    self.ensemble_axes_metadata, self.axes_types, self.axes.shape
                )[:: self.axes.shape[1]]
            ]

        elif isinstance(titles, str):
            titles = [titles] * self.axes.shape[0]

        for column_title in self._column_titles:
            column_title.remove()

        column_titles = []
        for i, ax in enumerate(self.axes[:, -1]):
            annotation = ax.annotate(
                titles[i],
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                fontsize=fontsize,
                **kwargs,
            )
            column_titles.append(annotation)

        self._column_titles = column_titles

    def set_row_titles(
        self,
        titles: str | list[str] = None,
        shift: float = 0.0,
        format: str = ".3g",
        units: str = None,
        fontsize: float = 12,
        **kwargs,
    ):
        if titles is None:
            titles = [
                title[1]
                for title in _get_axes_titles(
                    self.ensemble_axes_metadata, self.axes_types, self.axes.shape
                )[: self.axes.shape[1]]
            ]
        elif isinstance(titles, str):
            titles = [titles] * self.axes.shape[1]

        for row_title in self._row_titles:
            row_title.remove()

        row_titles = []
        for i, ax in enumerate(self.axes[0, :]):
            annotation = ax.annotate(
                titles[i],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - shift, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90,
                fontsize=fontsize,
                **kwargs,
            )
            row_titles.append(annotation)

        self._row_titles = row_titles

    def set_panel_labels(
        self,
        labels: str = None,
        frameon: bool = True,
        loc: str = "upper left",
        pad: float = 0.1,
        borderpad: float = 0.1,
        prop: dict = None,
        formatting: str = ".3g",
        units: str = None,
        **kwargs,
    ):
        if labels is None:
            titles = self._get_panel_titles()
            labels = ["\n".join(title) for title in titles]

        if not isinstance(labels, (tuple, list)):
            raise ValueError()

        if len(labels) != np.array(self.axes).size:
            raise ValueError()

        if prop is None:
            prop = {}

        for old_label in self._panel_labels:
            old_label.remove()

        panel_labels = []
        for ax, label in zip(np.array(self.axes).ravel(), labels):
            anchored_text = AnchoredText(
                label,
                pad=pad,
                borderpad=borderpad,
                frameon=frameon,
                loc=loc,
                prop=prop,
                **kwargs,
            )
            anchored_text.formatting = formatting

            anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(anchored_text)

            panel_labels.append(anchored_text)

        self._panel_labels = panel_labels

        # if labels_type == "metadata":
        #     self._metadata_labels = self._panel_labels
        # else:
        #     self._metadata_labels = []

    def axis(self, mode="all", ticks: bool = True):
        if mode == "all":
            return

        if mode == "none":
            indices = ()
        else:
            indices = tuple(self.axes._axis_location_to_indices(mode))

        for index in np.ndindex(self.axes.shape):
            if index in indices:
                continue

            ax = self.axes[index]
            ax._axislines["bottom"].toggle(ticklabels=False, label=False, ticks=ticks)
            ax._axislines["left"].toggle(ticklabels=False, label=False, ticks=ticks)

            # if not spines:
            #     ax.spines["top"].set_visible(False)
            #     ax.spines["right"].set_visible(False)
            #     ax.spines["bottom"].set_visible(False)
            #     ax.spines["left"].set_visible(False)

    def remove_artists(self):
        for ax in np.array(self.axes).ravel():
            for child in ax.get_children():
                if isinstance(child, AxesImage):
                    child.remove()

    @abstractmethod
    def set_artists(self, value_limits: list[float], power: float = 1):
        pass

    @abstractmethod
    def update_artists(self):
        pass

    def make_widgets(self):
        widgets = {}
        widgets["canvas"] = self.axes.fig.canvas

        (
            sliders,
            reset_button,
            continuous_update_button,
        ) = make_sliders_from_ensemble_axes(
            self,
            self.axes_types,
        )
        scale_button = make_scale_button(self)
        autoscale_button = make_autoscale_button(self)

        widgets["sliders"] = sliders
        widgets["reset_button"] = reset_button
        widgets["continuous_update_button"] = continuous_update_button
        widgets["scale_button"] = scale_button
        widgets["autoscale_button"] = autoscale_button
        return widgets

    def layout_widgets(self):
        widgets = self.make_widgets()

        scale_box = ipywidgets.VBox(
            [
                ipywidgets.HBox(
                    [
                        widgets["reset_button"],
                        widgets["continuous_update_button"],
                    ]
                ),
                ipywidgets.HBox(
                    [
                        widgets["scale_button"],
                        widgets["autoscale_button"],
                    ]
                ),
            ]
        )
        scale_box.layout = ipywidgets.Layout(width="300px")

        return ipywidgets.HBox(
            [
                ipywidgets.VBox(
                    [
                        ipywidgets.VBox(widgets["sliders"]),
                        scale_box,
                    ]
                ),
                widgets["canvas"],
            ]
        )


class VisualizationLines(BaseVisualization):
    _num_coordinate_dims = 1

    def __init__(
        self,
        array,
        coordinate_axes,
        scale_axis,
        ax,
        ensemble_axes=None,
        common_scale: bool = True,
        explode: Sequence[str] | bool = False,
        overlay: Sequence[str] | bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
        title: bool | str = True,
        legend: bool = True,
    ):
        super().__init__(
            array=array,
            coordinate_axes=coordinate_axes,
            scale_axis=scale_axis,
            ax=ax,
            ensemble_axes_metadata=ensemble_axes,
            common_scale=common_scale,
            aspect=False,
            explode=explode,
            overlay=overlay,
            interact=interact,
            figsize=figsize,
            share_x=True,
            share_y=common_scale,
            title=title,
            limits_margin=0.05,
        )

        # ylim = self._validate_value_limits(ylim)
        # for i in np.ndindex(self.axes.shape):
        #     self.axes[i].set_ylim(ylim[i])

        self.set_artists()
        self.set_ylim()
        self.set_xlabel()
        self.set_ylabel()

        if any(axis_type == "overlay" for axis_type in self.axes_types) and legend:
            self.set_legends()

        if not common_scale:
            self.axes.set_sizes(padding=0.5)

        for ax in np.array(self.axes).ravel():
            # ax.yaxis.set_label_coords(0.5, -0.02)
            # cbar2.formatter.set_powerlimits((0, 0))
            # ax.get_yaxis().formatter.set_useMathText(True)

            ax.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
            ax.get_yaxis().get_offset_text().set_horizontalalignment("center")
            # ax.yaxis.set_offset_position("right")
            # ax.yaxis.set_offset_position("left")

            # self.axes[i].set_ylabel(format_label(self._y_label, self._y_units))

    def set_artists(self, **kwargs):
        x = self._coordinate_axes[0].coordinates(self._array.shape[-1])

        labels = _get_overlay_labels(self.ensemble_axes_metadata, self.axes_types)
        artists = np.zeros(self.axes.shape, dtype=object)

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            array = self._get_array_for_axis(axis_index=i)

            if len(array.shape) > 1:
                array = np.moveaxis(array, -1, 0).reshape((array.shape[-1], -1))

            lines = ax.plot(x, array, label=labels)

            artists.itemset(i, lines)

        self._artists = artists

    def update_artists(self):
        for i in np.ndindex(self.axes.shape):
            lines = self._artists[i]
            array = self._get_array_for_axis(axis_index=i)

            if len(array.shape) > 1:
                array = np.moveaxis(array, -1, 0).reshape((array.shape[-1], -1))
            else:
                array = array[None]

            for i, line in enumerate(lines):
                x = line.get_data()[0]
                line.set_data(x, array)

    def set_values_lim(self, values_lim: tuple[float, float] = None):
        self.set_ylim(values_lim)

    def set_legends(self, loc: str = "first", **kwargs):
        indices = [index for index in np.ndindex(*self.axes.shape)]

        if loc == "first":
            loc = indices[:1]
        elif loc == "last":
            loc = indices[-1:]
        elif loc == "all":
            loc = indices

        for i in np.ndindex(self.axes.shape):
            if i in loc:
                self.axes[i].legend(**kwargs)


class Artist2D:
    @abstractmethod
    def __init__(self, norm):
        self._norm = norm
        self._cbars = []

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self):
        pass

    @abstractmethod
    def set_cmap(self, cmap):
        pass

    def set_logscale(self):
        pass


class ImageArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        extent=None,
        power: float = 1.0,
        logscale: bool = False,
    ):
        norm = _get_norm(vmin, vmax, power, logscale)

        super().__init__(norm)

        self._artist = ax.imshow(
            data.T,
            origin="lower",
            interpolation="none",
            cmap=cmap,
            extent=extent,
        )
        self._artist.set_norm(norm)

    def set_extent(self, extent):
        self._artist.set_extent(extent)

    def set_data(self, data):
        self._artist.set_data(data.T)

    def set_cmap(self, cmap):
        self._artist.set_cmap(cmap)

    def set_cbars(self, caxes, label=None, **kwargs):
        self._cbar = plt.colorbar(self._artist, cax=caxes[0])

        self._cbar.set_label(label, **kwargs)
        self._cbar.formatter.set_powerlimits((-2, 2))
        self._cbar.formatter.set_useMathText(True)
        self._cbar.ax.yaxis.set_offset_position("left")

    def value_limits(self, value_limits: tuple[float, float] = None):
        self._norm.vmin = value_limits[0]
        self._norm.vmax = value_limits[1]

    def set_power(self, power: float = 1.0):
        if (power != 1.0) and isinstance(self._norm, colors.PowerNorm):
            self._norm.gamma = power
            self._norm._changed()
        else:
            self._norm = _get_norm(
                vmin=self._norm.vmin, vmax=self._norm.vmax, power=power
            )
            self._artist.norm = self._norm


class ScatterArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        scale_spots=60
    ):
        intensities = data[:, 0]
        scales = intensities * scale_spots
        positions = data[:, 1:3]

        vmin, vmax = _get_value_limits(intensities, value_limits=[vmin, vmax])
        norm = _get_norm(vmin, vmax, power, logscale)

        super().__init__(norm)

        self._artist = EllipseCollection(
            widths=scales,
            heights=scales,
            angles=0.0,
            units="xy",
            array=intensities,
            cmap=cmap,
            offsets=positions,
            transOffset=ax.transData,
        )

        self._artist.set_norm(norm)

        ax.add_collection(self._artist)

    # def set_extent(self, extent):
    #     self._image.set_extent(extent)

    def set_data(self, data):
        self._artist.set_array(data.T)

    def set_cmap(self, cmap):
        self._artist.set_cmap(cmap)

    def set_cbars(self, caxes, label=None, **kwargs):
        self._cbar = plt.colorbar(self._artist, cax=caxes[0])

        self._cbar.set_label(label, **kwargs)
        self._cbar.formatter.set_powerlimits((-2, 2))
        self._cbar.formatter.set_useMathText(True)
        self._cbar.ax.yaxis.set_offset_position("left")

    def value_limits(self, value_limits: tuple[float, float] = None):
        self._norm.vmin = value_limits[0]
        self._norm.vmax = value_limits[1]

    def set_power(self, power: float = 1.0):
        if (power != 1.0) and isinstance(self._norm, colors.PowerNorm):
            self._norm.gamma = power
            self._norm._changed()
        else:
            self._norm = _get_norm(
                vmin=self._norm.vmin, vmax=self._norm.vmax, power=power
            )
            self._image.norm = self._norm


class DomainColoringArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        extent:list[float]=None,
        power: float = 1.0,
        logscale: bool = False,
    ):
        norm = _get_norm(vmin, vmax, power, logscale)

        abs_array = np.abs(data)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        self._image_phase = ax.imshow(
            np.angle(data).T,
            origin="lower",
            interpolation="none",
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
            extent=extent
        )
        self._image_amplitude = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation="none",
            cmap="gray",
            zorder=-1,
            extent=extent
        )

        self._image_amplitude.set_norm(norm)
        super().__init__(norm=norm)

    def _update_alpha(self):
        data = self._image_amplitude.get_array().data
        norm = self._norm
        alpha = norm(np.abs(data))
        alpha = np.clip(alpha, a_min=0, a_max=1)
        self._image_phase.set_alpha(alpha)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._norm.vmin = value_limits[0]
        self._norm.vmax = value_limits[1]
        self._update_alpha()

    def set_cmap(self, cmap):
        self._image_phase.set_cmap(cmap)

    def set_power(self, power: float = 1.0):
        if (power != 1.0) and isinstance(self._norm, colors.PowerNorm):
            self._norm.gamma = power
            self._norm._changed()
        else:
            self._norm = _get_norm(
                vmin=self._norm.vmin, vmax=self._norm.vmax, power=power
            )
            self._image_amplitude.norm = self._norm
        self._update_alpha()

    def set_extent(self, extent):
        self._image_phase.set_extent(extent)
        self._image_amplitude.set_extent(extent)

    def set_data(self, data):
        norm = self._norm
        abs_array = np.abs(data)
        alpha = norm(abs_array)
        alpha = np.clip(alpha, a_min=0, a_max=1)

        self._image_phase.set_alpha(alpha.T)
        self._image_phase.set_data(np.angle(data).T)
        self._image_amplitude.set_data(abs_array.T)

    def set_cbars(self, caxes, label=None, **kwargs):
        self._cbar_phase = plt.colorbar(self._image_phase, cax=caxes[0])
        self._cbar_amplitude = plt.colorbar(self._image_amplitude, cax=caxes[1])

        self._cbar_phase.set_label("arg", rotation=0, ha="center", va="top")
        self._cbar_phase.ax.yaxis.set_label_coords(0.5, -0.02)
        self._cbar_phase.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        self._cbar_phase.set_ticklabels(
            [
                r"$-\pi$",
                r"$-\dfrac{\pi}{2}$",
                "$0$",
                r"$\dfrac{\pi}{2}$",
                r"$\pi$",
            ]
        )
        self._cbar_amplitude.set_label("abs", rotation=0, ha="center", va="top")
        self._cbar_amplitude.ax.yaxis.set_label_coords(0.5, -0.02)
        self._cbar_amplitude.formatter.set_powerlimits((0, 0))
        self._cbar_amplitude.formatter.set_useMathText(True)
        self._cbar_amplitude.ax.yaxis.set_offset_position("left")


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


class BaseVisualization2D(BaseVisualization):
    def __init__(
        self,
        array,
        coordinate_axes,
        scale_axis,
        ax=None,
        ensemble_axes: list[AxisMetadata] = None,
        common_scale: bool = False,
        explode: Sequence[int] | bool = False,
        overlay: Sequence[int] | bool = False,
        cmap: str = None,
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        cbar: bool = False,
        figsize: tuple[float, float] = None,
        title: str | bool = True,
        interact: bool = False,
    ):
        self._cmap = cmap
        self._size_bar = []

        if cbar:
            if np.iscomplexobj(array):
                ncbars = 2
            else:
                ncbars = 1
        else:
            ncbars = 0

        super().__init__(
            array=array,
            coordinate_axes=coordinate_axes,
            scale_axis=scale_axis,
            ax=ax,
            ensemble_axes_metadata=ensemble_axes,
            common_scale=common_scale,
            explode=explode,
            overlay=overlay,
            ncbars=ncbars,
            interact=interact,
            figsize=figsize,
            title=title,
            aspect=True,
            share_x=True,
            share_y=True,
        )
        self.set_artists(value_limits=[vmin, vmax], power=power, common_scale=common_scale)
        self.set_xlabel(coordinate_axes[0].format_label())
        self.set_ylabel(coordinate_axes[1].format_label())

        if cbar:
            self.set_cbars()

    @abstractmethod
    def set_artists(self, value_limits: list[float], power: float = 1, common_scale: bool = False):
        pass

    def set_cbars(self, label=None, **kwargs):
        if label is None:
            label = self._scale_axis.format_label()

        for i in np.ndindex(self.axes.shape):
            artist = self._artists[i]

            if hasattr(self.axes, "set_cbar_layout"):
                caxes = self.axes._caxes[i]
                artist.set_cbars(caxes=caxes, label=label)
            else:
                raise NotImplementedError

    def set_complex_conversion(self, complex_conversion: str):
        self._complex_conversion = complex_conversion

        if np.iscomplexobj(self.array) and (self._complex_conversion == "none"):
            self.axes.set_cbar_layout(ncbars=2)
        else:
            self.axes.set_cbar_layout(ncbars=1)

        self.set_artists(value_limits=None)
        self.set_cbars()

    def _validate_cmap(self, cmap=None):
        if cmap is not None:
            pass

        elif np.iscomplexobj(self.array) and (
            self._complex_conversion in ("none", "phase")
        ):
            cmap = config.get("visualize.phase_cmap", "hsluv")

        else:
            cmap = config.get("visualize.cmap", "viridis")

        if cmap == "hsluv":
            cmap = hsluv_cmap

        return cmap

    def set_cmaps(self, cmap):
        cmap = self._validate_cmap(cmap)

        for i in np.ndindex(self.axes.shape):
            self._artists[i].set_cmap(cmap)

    def set_value_limits(self, values_limits: tuple[float, float] = None):
        values_limits = self._validate_value_limits(values_limits)

        for i in np.ndindex(self.axes.shape):
            self._artists[i].set_value_limits(values_limits[i])

    def set_extent(self, extent: list[float] = None):
        for artist in self._artists.ravel():
            artist.set_extent(extent)

    def set_power(self, power: float = 1.0):
        for i in np.ndindex(self.axes.shape):
            self._artists[i].set_power(power)

    def update_artists(self):
        for i in np.ndindex(self.axes.shape):
            array = self._get_array_for_axis(axis_index=i)
            self._artists[i].set_data(array)

    # def set_scalebars(
    #     self,
    #     panel_loc: tuple[int, ...] | str = "lower right",
    #     label: str = "",
    #     size: float = None,
    #     loc: str = "lower right",
    #     borderpad: float = 0.5,
    #     formatting: str = ".3f",
    #     size_vertical: float = None,
    #     sep: float = 6,
    #     pad: float = 0.3,
    #     label_top: bool = True,
    #     frameon: bool = False,
    #     **kwargs,
    # ):
    #     if isinstance(panel_loc, str):
    #         panel_loc = self.axes._axis_location_to_indices(panel_loc)
    #
    #     if size is None:
    #         limits = self._get_coordinate_limits()[0]
    #         size = (limits[1] - limits[0]) / 3
    #
    #     if size_vertical is None:
    #         limits = self._get_coordinate_limits()[1]
    #         size_vertical = (limits[1] - limits[0]) / 20
    #
    #     if label is None:
    #         label = f"{self._coordinate_axes[0].format_label(formatting=formatting)}"
    #
    #     for size_bar in self._size_bars:
    #         size_bar.remove()
    #
    #     self._size_bars = []
    #     for ax in panel_loc:
    #         ax = self.axes[ax]
    #         anchored_size_bar = AnchoredSizeBar(
    #             ax.transData,
    #             label=label,
    #             label_top=label_top,
    #             size=size,
    #             borderpad=borderpad,
    #             loc=loc,
    #             size_vertical=size_vertical,
    #             sep=sep,
    #             pad=pad,
    #             frameon=frameon,
    #             **kwargs,
    #         )
    #         ax.add_artist(anchored_size_bar)
    #         self._size_bars.append(anchored_size_bar)

    # def add_area_indicator(self, area_indicator, panel="first", **kwargs):
    #     xlim = self.axes[0, 0].get_xlim()
    #     ylim = self.axes[0, 0].get_ylim()
    #
    #     for i, ax in enumerate(np.array(self.axes).ravel()):
    #         if panel == "first" and i == 0:
    #             area_indicator._add_to_visualization(ax, **kwargs)
    #         elif panel == "all":
    #             area_indicator._add_to_visualization(ax, **kwargs)
    #
    #         ax.set_xlim(xlim)
    #         ax.set_ylim(ylim)

    def make_widgets(self):
        widgets = super().make_widgets()

        power_scale_slider = make_power_scale_slider(self)
        cmap_dropdown = make_cmap_dropdown(self)
        complex_visualization_dropdown = make_complex_visualization_dropdown(self)

        widgets["power_scale_slider"] = power_scale_slider
        widgets["cmap_dropdown"] = cmap_dropdown
        widgets["complex_visualization_dropdown"] = complex_visualization_dropdown

        return widgets

    def layout_widgets(self):
        widgets = self.make_widgets()

        widget_box = ipywidgets.VBox(
            [
                ipywidgets.VBox(widgets["sliders"]),
                ipywidgets.HBox(
                    [
                        widgets["reset_button"],
                        widgets["continuous_update_button"],
                    ]
                ),
                ipywidgets.HBox(
                    [
                        widgets["scale_button"],
                        widgets["autoscale_button"],
                    ]
                ),
                widgets["power_scale_slider"],
                widgets["cmap_dropdown"],
                widgets["complex_visualization_dropdown"],
            ]
        )
        layout = ipywidgets.HBox([widget_box, widgets["canvas"]])
        return layout


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


class VisualizationImshow(BaseVisualization2D):
    def __init__(
        self,
        array,
        coordinate_axes,
        scale_axis,
        ax=None,
        ensemble_axes: list[AxisMetadata] = None,
        common_color_scale: bool = False,
        explode: Sequence[int] | bool = False,
        overlay: Sequence[int] | bool = False,
        cmap: str = None,
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        cbar: bool = False,
        figsize: tuple[float, float] = None,
        title: str | bool = True,
        interact: bool = False,
    ):
        if common_color_scale is True:
            vmin, vmax = _get_value_limits(array, value_limits=[vmin, vmax])

        super().__init__(
            array,
            coordinate_axes,
            ax=ax,
            scale_axis=scale_axis,
            ensemble_axes=ensemble_axes,
            common_scale=common_color_scale,
            explode=explode,
            overlay=overlay,
            cmap=cmap,
            power=power,
            vmin=vmin,
            vmax=vmax,
            cbar=cbar,
            interact=interact,
            figsize=figsize,
            title=title,
        )

        shape = self.array.shape[-2:]
        xlim, ylim = _get_coordinate_limits(coordinate_axes, shape=shape)

        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def set_artists(
        self,
        value_limits: list[float] = None,
        common_scale:bool=None,
        power: float = 1.0,
        logscale: bool = False,
        cmap: str = None,
    ):
        self.remove_artists()

        shape = self.array.shape[-2:]

        if common_scale:
            vmin, vmax = _get_value_limits(self.array, value_limits=value_limits)
        else:
            vmin = None
            vmax = None

        extent = flatten_list_of_lists(
            _get_coordinate_limits(self._coordinate_axes, shape=shape)
        )
        artists = np.zeros(self.axes.shape, dtype=object)

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            #vmin, vmax = value_limits[i]
            array = self._get_array_for_axis(axis_index=i)
            cmap = self._validate_cmap(cmap)

            if np.iscomplexobj(self.array) and (self._complex_conversion == "none"):
                artist = DomainColoringArtist(
                    ax,
                    array,
                    cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    power=power,
                    logscale=logscale,
                )
            else:
                artist = ImageArtist(
                    ax,
                    array,
                    cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    power=power,
                    logscale=logscale,
                )

            artists.itemset(i, artist)

        self._artists = artists


class VisualizationScatter(BaseVisualization2D):
    def __init__(
        self,
        array,
        coordinate_axes,
        ax=None,
        scale_axis=None,
        ensemble_axes: list[AxisMetadata] = None,
        common_scale: bool = False,
        explode: Sequence[int] | bool = False,
        overlay: Sequence[int] | bool = False,
        cmap: str = None,
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        cbar: bool = False,
        figsize: tuple[float, float] = None,
        title: str | bool = True,
        interact: bool = False,
    ):

        super().__init__(
            array=array,
            coordinate_axes=coordinate_axes,
            scale_axis=scale_axis,
            ensemble_axes=ensemble_axes,
            common_scale=common_scale,
            explode=explode,
            overlay=overlay,
            cmap=cmap,
            power=power,
            vmin=vmin,
            vmax=vmax,
            cbar=cbar,
            interact=interact,
            figsize=figsize,
            title=title,
        )

        xlim = (min(coordinate_axes[0].values), max(coordinate_axes[0].values))
        ylim = (min(coordinate_axes[1].values), max(coordinate_axes[0].values))

        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def set_artists(
        self,
        value_limits: list[float] = None,
        power: float = 1.0,
        logscale: bool = False,
        cmap: str = None,
        common_scale: bool = None
    ):
        self.remove_artists()
        vmin, vmax = None, None
        artists = np.zeros(self.axes.shape, dtype=object)

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            # vmin, vmax = value_limits[i]
            array = self._get_array_for_axis(axis_index=i)
            cmap = self._validate_cmap(cmap)

            if np.iscomplexobj(self.array) and (self._complex_conversion == "none"):
                artist = DomainColoringArtist(
                    ax,
                    array,
                    cmap,
                    vmin=vmin,
                    vmax=vmax,
                    power=power,
                    logscale=logscale,
                )
            else:
                artist = ScatterArtist(
                    ax=ax,
                    data=array,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    power=power,
                    logscale=logscale,
                )

            artists.itemset(i, artist)

        self._artists = artists


def make_toggle_hkl_button(visualization):
    toggle_hkl_button = ipywidgets.ToggleButton(description="Toggle hkl", value=False)

    def update_toggle_hkl_button(change):
        if change["new"]:
            visualization.set_miller_index_annotations()
        else:
            visualization.remove_miller_index_annotations()

    toggle_hkl_button.observe(update_toggle_hkl_button, "value")

    return toggle_hkl_button


_cube = np.array(
    [
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 1]],
        [[0, 0, 1], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 0, 1]],
        [[0, 1, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[1, 1, 0], [1, 1, 1]],
    ]
)


def _merge_columns(atoms: Atoms, plane, tol: float = 1e-7) -> Atoms:
    uniques, labels = np.unique(atoms.numbers, return_inverse=True)

    new_atoms = Atoms(cell=atoms.cell)
    for unique, indices in zip(uniques, label_to_index(labels)):
        positions = atoms.positions[indices]
        positions = _merge_positions(positions, plane, tol)
        numbers = np.full((len(positions),), unique)
        new_atoms += Atoms(positions=positions, numbers=numbers)

    return new_atoms


def _merge_positions(positions, plane, tol: float = 1e-7) -> np.ndarray:
    axes = plane_to_axes(plane)
    rounded_positions = tol * np.round(positions[:, axes[:2]] / tol)
    unique, labels = np.unique(rounded_positions, axis=0, return_inverse=True)

    new_positions = np.zeros((len(unique), 3))
    for i, label in enumerate(label_to_index(labels)):
        top_atom = np.argmax(-positions[label][:, axes[2]])
        new_positions[i] = positions[label][top_atom]
        # new_positions[i, axes[2]] = np.max(positions[label][top_atom, axes[2]])

    return new_positions


def show_atoms(
    atoms: Atoms,
    plane: tuple[float, float] | str = "xy",
    ax: Axes = None,
    scale: float = 0.75,
    title: str = None,
    numbering: bool = False,
    show_periodic: bool = False,
    figsize: tuple[float, float] = None,
    legend: bool = False,
    merge: float = 1e-2,
    tight_limits: bool = False,
    show_cell: bool = None,
    **kwargs,
):
    """
    Display 2D projection of atoms as a matplotlib plot.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms to be shown.
    plane : str, two float
        The projection plane given as a concatenation of 'x' 'y' and 'z', e.g. 'xy', or as two floats representing the
        azimuth and elevation angles of the viewing direction [degrees], e.g. (45, 45).
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axes.
    scale : float
        Factor scaling their covalent radii for the atom display sizes (default is 0.75).
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Display the index of the Atoms as a number. Default is False.
    show_periodic : bool
        If True, show the periodic images of the atoms at the cell boundary.
    figsize : two int, optional
        The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
    legend : bool
        If True, add a legend indicating the color of the atomic species.
    merge: float
        To speed up plotting large numbers of atoms, those closer than the given value [Å] are merged.
    tight_limits : bool
        If True the limits of the plot are adjusted
    kwargs : Keyword arguments for matplotlib.collections.PatchCollection.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    if show_periodic:
        atoms = atoms.copy()
        atoms = pad_atoms(atoms, margins=1e-3)

    if merge > 0.0:
        atoms = _merge_columns(atoms, plane, merge)

    if tight_limits and show_cell is None:
        show_cell = False
    elif show_cell is None:
        show_cell = True

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cell = atoms.cell
    axes = plane_to_axes(plane)

    cell_lines = np.array(
        [[np.dot(line[0], cell), np.dot(line[1], cell)] for line in _cube]
    )
    cell_lines_x, cell_lines_y = cell_lines[..., axes[0]], cell_lines[..., axes[1]]

    if show_cell:
        for cell_line_x, cell_line_y in zip(cell_lines_x, cell_lines_y):
            ax.plot(cell_line_x, cell_line_y, "k-")

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(-atoms.positions[:, axes[2]])
        positions = positions[order]

        colors = jmol_colors[atoms.numbers[order]]
        sizes = covalent_radii[atoms.numbers[order]] * scale

        circles = []
        for position, size in zip(positions, sizes):
            circles.append(Circle(position, size))

        coll = PatchCollection(circles, facecolors=colors, edgecolors="black", **kwargs)
        ax.add_collection(coll)

        ax.axis("equal")
        ax.set_xlabel(plane[0] + " [Å]")
        ax.set_ylabel(plane[1] + " [Å]")

        ax.set_title(title)

        if numbering:
            if merge:
                raise ValueError("atom numbering requires 'merge' to be False")

            for i, (position, size) in enumerate(zip(positions, sizes)):
                ax.annotate(
                    "{}".format(order[i]), xy=position, ha="center", va="center"
                )

    if legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="k",
                label=chemical_symbols[unique],
                markerfacecolor=jmol_colors[unique],
                markersize=12,
            )
            for unique in np.unique(atoms.numbers)
        ]

        ax.legend(handles=legend_elements, loc="upper right")

    if tight_limits:
        ax.set_adjustable("box")
        ax.set_xlim([np.min(cell_lines_x), np.max(cell_lines_x)])
        ax.set_ylim([np.min(cell_lines_y), np.max(cell_lines_y)])

    return fig, ax


"""
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

fig, ax = plt.subplots()
ax.set_facecolor("k")

cmap = ListedColormap(["lime"])
norm = colors.Normalize()
norm.autoscale_None(stacked.array[0])
alpha = norm(stacked.array[0])

im = ax.imshow(
    np.ones_like(stacked.array[0]).T, alpha=alpha.T, cmap=cmap, origin="lower"
)

cmap = LinearSegmentedColormap.from_list("red", ["k", "lime"])
plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)

c = "r"
cmap = ListedColormap([c])
norm = colors.Normalize()
norm.autoscale_None(stacked.array[1])
alpha = norm(stacked.array[1])

im = ax.imshow(
    np.ones_like(stacked.array[1]).T, alpha=alpha.T, cmap=cmap, origin="lower"
)

cmap = LinearSegmentedColormap.from_list("c", ["k", c])
plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)

"""
