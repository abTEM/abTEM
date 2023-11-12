"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from __future__ import annotations

import itertools
import string
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import TYPE_CHECKING, Sequence, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, EllipseCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import Size, Divider
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory
from traitlets.traitlets import link

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.axes import ReciprocalSpaceAxis, format_label, LinearAxis
from abtem.core.colors import hsluv_cmap
from abtem.core.units import _get_conversion_factor
from abtem.core.utils import label_to_index

try:
    import ipywidgets
except ImportError:
    ipywidgets = None

ipywidgets_not_installed = RuntimeError(
    "This functionality of abTEM requires ipywidgets, see "
    "https://ipywidgets.readthedocs.io/en/stable/user_install.html."
)


if TYPE_CHECKING:
    from abtem.measurements import (
        BaseMeasurements,
        _BaseMeasurement1D,
        _BaseMeasurement2D,
        IndexedDiffractionPatterns,
    )


def _make_default_sizes():
    sizes = {
        "cbar_padding_left": Size.Fixed(0.15),
        "cbar_spacing": Size.Fixed(0.4),
        "cbar_padding_right": Size.Fixed(0.9),
        "padding": Size.Fixed(0.1),
    }
    return sizes


def _cbar_layout(n, sizes):
    if n == 0:
        return []

    layout = [sizes["cbar_padding_left"]]
    for i in range(n):
        layout.extend([sizes["cbar"]])

        if i < n - 1:
            layout.extend([sizes["cbar_spacing"]])

    layout.extend([sizes["cbar_padding_right"]])
    return layout


def _make_grid_layout(
    axes, ncbars: int, sizes: dict, cbar_mode: str = "each", direction: str = "col"
):
    sizes_layout = []

    if cbar_mode not in ("single", "each"):
        raise ValueError()

    for i, ax in enumerate(axes):
        if direction == "col":
            sizes_layout.append(Size.AxesX(ax, aspect="axes", ref_ax=axes[0]))
        elif direction == "row":
            sizes_layout.append(Size.AxesY(ax, aspect="axes", ref_ax=axes[0]))
        else:
            raise ValueError()

        if not "cbar" in sizes:
            sizes["cbar"] = Size.from_any("5%", sizes_layout[0])

        if cbar_mode == "each":
            sizes_layout.extend(_cbar_layout(ncbars, sizes))

        if i < len(axes) - 1:
            sizes_layout.append(sizes["padding"])

    if cbar_mode == "single":
        sizes_layout.extend(_cbar_layout(ncbars, sizes))

    return sizes_layout


class AxesGrid:
    def __init__(
        self,
        fig,
        ncols: int,
        nrows: int,
        ncbars: int = 0,
        cbar_mode: str = "single",
        aspect: bool = True,
        sharex: bool = True,
        sharey: bool = True,
        rect: tuple = (0.1, 0.1, 0.85, 0.85),
        col_sizes: dict = None,
        row_sizes: dict = None,
    ):
        from mpl_toolkits.axes_grid1.mpl_axes import Axes

        self._ncols = ncols
        self._nrows = nrows
        self._ncbars = ncbars
        self._aspect = aspect
        self._sharex = sharex
        self._sharey = sharey

        if col_sizes is None:
            col_sizes = _make_default_sizes()

        if row_sizes is None:
            row_sizes = _make_default_sizes()

        self._col_sizes = col_sizes
        self._row_sizes = row_sizes

        axes = []
        for nx in range(ncols):
            for ny in range(nrows):
                if len(axes) > 0:
                    if sharex:
                        sharex = axes[0]
                    else:
                        sharex = None

                    if sharey:
                        sharey = axes[0]
                    else:
                        sharey = None

                    ax = Axes(fig, rect, sharex=sharex, sharey=sharey)
                else:
                    ax = Axes(fig, rect, sharex=None, sharey=None)
                axes.append(ax)

        for ax in axes:
            fig.add_axes(ax)

        cols = np.array(axes, dtype=object).reshape((ncols, nrows))[:, 0]
        rows = np.array(axes, dtype=object).reshape((ncols, nrows))[0]

        col_layout = _make_grid_layout(
            cols,
            ncbars=ncbars,
            sizes=self._col_sizes,
            cbar_mode=cbar_mode,
            direction="col",
        )

        row_layout = _make_grid_layout(
            rows, ncbars=0, sizes=self._row_sizes, direction="row"
        )

        self._divider = Divider(
            fig, rect, horizontal=col_layout, vertical=row_layout, aspect=aspect
        )

        axes_index = 0
        caxes_index = 0

        if cbar_mode == "single":
            caxes = {axes[0]: []}
        else:
            caxes = {ax: [] for ax in axes}

        for nx, col_size in enumerate(col_layout):
            for ny, row_size in enumerate(row_layout):
                if isinstance(col_size, Size.AxesX) and (
                    isinstance(row_size, Size.AxesY)
                ):
                    ax = axes[axes_index]
                    ax.set_axes_locator(self._divider.new_locator(nx=nx, ny=ny))
                    axes_index += 1

                if (
                    (cbar_mode == "each")
                    and (col_size is self._col_sizes["cbar"])
                    and (isinstance(row_size, Size.AxesY))
                ):
                    ax = axes[
                        np.ravel_multi_index(
                            (caxes_index // (ncbars * nrows), caxes_index % nrows),
                            (ncols, nrows),
                        )
                    ]

                    caxes_index += 1

                    cb_ax = _cbaraxes_class_factory(Axes)(
                        fig, self._divider.get_position(), orientation="vertical"
                    )

                    fig.add_axes(cb_ax)
                    cb_ax.set_axes_locator(self._divider.new_locator(nx=nx, ny=ny))
                    caxes[ax].append(cb_ax)

                if (
                    (cbar_mode == "single")
                    and (len(caxes[axes[0]]) < ncbars)
                    and (col_size is self._col_sizes["cbar"])
                    and (isinstance(row_size, Size.AxesY))
                ):
                    for i in range(ncbars):
                        cb_ax = _cbaraxes_class_factory(Axes)(
                            fig, self._divider.get_position(), orientation="vertical"
                        )
                        fig.add_axes(cb_ax)
                        cb_ax.set_axes_locator(
                            self._divider.new_locator(nx=nx + i * 2, ny=0, ny1=-1)
                        )
                        caxes[axes[0]].append(cb_ax)

        axes = np.array(axes, dtype=object).reshape((ncols, nrows))

        if sharex:
            for inner_axes in axes[:, 1:]:
                for ax in inner_axes:
                    ax._axislines["bottom"].toggle(ticklabels=False, label=False)

        if sharey:
            for inner_axes in axes[1:]:
                for ax in inner_axes:
                    ax._axislines["left"].toggle(ticklabels=False, label=False)

        self._axes = axes
        self._caxes = caxes

    @property
    def divider(self):
        return self._divider

    @property
    def ncols(self) -> int:
        return self._axes.shape[0]

    @property
    def nrows(self) -> int:
        return self._axes.shape[1]

    def __getitem__(self, item):
        return self._axes[item]

    def __len__(self):
        return len(self._axes)

    def item(self):
        return self._axes.item()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._axes.shape

    def set_cbar_padding(self, padding: tuple[float, float] = (0.1, 0.1)):
        if np.isscalar(padding):
            padding = (padding,) * 2

        self._col_sizes["cbar_padding_left"].fixed_size = padding[0]
        self._row_sizes["cbar_padding_left"].fixed_size = padding[0]
        self._col_sizes["cbar_padding_right"].fixed_size = padding[1]
        self._row_sizes["cbar_padding_right"].fixed_size = padding[1]

    def set_cbar_size(self, fraction: float):
        self._col_sizes["cbar"]._fraction = fraction
        self._row_sizes["cbar"]._fraction = fraction

    def set_cbar_spacing(self, spacing: float):
        self._col_sizes["cbar_spacing"].fixed_size = spacing
        self._row_sizes["cbar_spacing"].fixed_size = spacing

    def set_axes_padding(self, padding: float | tuple[float, float] = (0.0, 0.0)):
        if np.isscalar(padding):
            padding = (padding,) * 2

        self._col_sizes["padding"].fixed_size = padding[0]
        self._row_sizes["padding"].fixed_size = padding[1]

    @property
    def fig(self):
        return self._axes[0, 0].get_figure()


def _axes_grid_cols_and_rows(ensemble_shape, axes_types):
    shape = tuple(
        n
        for n, axes_type in zip(ensemble_shape, axes_types)
        if not axes_type in ("index", "range", "overlay")
    )

    if len(shape) > 0:
        ncols = shape[0]
    else:
        ncols = 1

    if len(shape) > 1:
        nrows = shape[1]
    else:
        nrows = 1

    return ncols, nrows


def _determine_axes_types(
    ensemble_axes_metadata,
    explode: bool | tuple[bool, ...] | None,
    overlay: bool | tuple[bool, ...] | None,
):
    num_ensemble_axes = len(ensemble_axes_metadata)

    axes_types = []
    for axis_metadata in ensemble_axes_metadata:
        if axis_metadata._default_type is not None:
            axes_types.append(axis_metadata._default_type)
        else:
            axes_types.append("index")

    if explode is True:
        explode = tuple(range(max(num_ensemble_axes - 2, 0), num_ensemble_axes))
    elif explode is False:
        explode = ()

    if overlay is True:
        overlay = tuple(range(max(num_ensemble_axes - 2, 0), num_ensemble_axes))
    elif overlay is False:
        overlay = ()

    axes_types = list(axes_types)
    for i, axis_type in enumerate(axes_types):
        if explode is not None:
            if i in explode:
                axes_types[i] = "explode"
            else:
                axes_types[i] = "index"

        if overlay is not None:
            if i in overlay:
                axes_types[i] = "overlay"
            elif i not in explode:
                axes_types[i] = "index"

    return axes_types


def _validate_axes(
    axes_types,
    ensemble_shape,
    ax: Axes = None,
    explode: bool = False,
    overlay: bool = False,
    ncbars: int = 0,
    common_color_scale: bool = False,
    figsize: tuple[float, float] = None,
    ioff: bool = False,
    aspect: bool = True,
    sharex: bool = True,
    sharey: bool = True,
):
    if common_color_scale:
        cbar_mode = "single"
    else:
        cbar_mode = "each"

    if ax is None:
        if ioff:
            with plt.ioff():
                fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)

    if ax is None:  # and ("explode" in axes_types):
        ncols, nrows = _axes_grid_cols_and_rows(ensemble_shape, axes_types)

        axes = AxesGrid(
            fig=fig,
            ncols=ncols,
            nrows=nrows,
            ncbars=ncbars,
            cbar_mode=cbar_mode,
            aspect=aspect,
            sharex=sharex,
            sharey=sharey,
        )
    # elif ax is None:
    #    ax = fig.add_subplot()
    #    axes = np.array([[ax]])
    else:
        if explode:
            raise NotImplementedError("`ax` not implemented with `explode = True`.")

        axes = np.array([[ax]])

    return axes


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


def make_sliders_from_ensemble_axes(
    visualizations: BaseVisualization,
    axes_types: list[str],
    continuous_update: bool = False,
    callbacks: tuple[callable, ...] = (),
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    ensemble_axes_metadata = visualizations[0].ensemble_axes_metadata
    ensemble_shape = visualizations[0].ensemble_shape

    for visualization in visualizations[1:]:
        if not isinstance(visualization, MeasurementVisualization):
            raise ValueError()

        if not (
            (
                visualization.measurements.ensemble_axes_metadata
                == ensemble_axes_metadata
            )
            and (visualization.measurements.ensemble_shape == ensemble_shape)
        ):
            raise ValueError()

    sliders = []
    for axes_metadata, n, axes_type in zip(
        ensemble_axes_metadata,
        ensemble_shape,
        axes_types,
    ):
        options = _format_options(axes_metadata.coordinates(n))

        with config.set({"visualize.use_tex": False}):
            label = axes_metadata.format_label()

        if axes_type == "range":
            sliders.append(
                ipywidgets.SelectionRangeSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=(0, len(options) - 1),
                )
            )
        elif axes_type == "index":
            sliders.append(
                ipywidgets.SelectionSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                )
            )

    for visualization in visualizations:
        _set_update_indices_callback(sliders, visualization, callbacks)

    return sliders


def _set_update_indices_callback(sliders, visualization, callbacks=()):
    def update_indices(change):
        indices = ()
        for slider in sliders:
            idx = slider.index
            if isinstance(idx, tuple):
                idx = slice(*idx)
            indices += (idx,)

        with sliders[0].hold_trait_notifications():
            visualization.set_ensemble_indices(indices)
            if visualization.autoscale:
                visualization.set_values_lim()

    for slider in sliders:
        slider.observe(update_indices, "value")
        for callback in callbacks:
            slider.observe(callback, "value")


def _make_continuous_button(sliders):
    continuous_update = config.get("visualize.continuous_update", False)

    continuous_update_checkbox = ipywidgets.ToggleButton(
        description="Continuous update", value=continuous_update
    )
    for slider in sliders:
        link((continuous_update_checkbox, "value"), (slider, "continuous_update"))
    return continuous_update_checkbox


def _get_max_range(array, axes_types):
    if np.iscomplexobj(array):
        array = np.abs(array)

    max_values = array.max(
        tuple(
            i for i, axes_type in enumerate(axes_types) if axes_type not in ("range",)
        )
    )

    positive_indices = np.where(max_values > 0)[0]

    if len(positive_indices) <= 1:
        max_value = np.max(max_values)
    else:
        max_value = np.sum(max_values[positive_indices])

    return max_value


def _make_vmin_vmax_slider(visualization):
    axes_types = (
        tuple(visualization._axes_types)
        + ("base",) * visualization.measurements.num_base_axes
    )

    max_value = _get_max_range(visualization.measurements.array, axes_types)
    min_value = -_get_max_range(-visualization.measurements.array, axes_types)

    step = (max_value - min_value) / 1e6

    vmin_vmax_slider = ipywidgets.FloatRangeSlider(
        value=visualization._get_vmin_vmax(),
        min=min_value,
        max=max_value,
        step=step,
        disabled=visualization._autoscale,
        description="Normalization",
        # readout=False,
        continuous_update=True,
    )

    def vmin_vmax_slider_changed(change):
        vmin, vmax = change["new"]
        vmax = max(vmax, vmin + step)

        with vmin_vmax_slider.hold_trait_notifications():
            visualization._update_vmin_vmax(vmin, vmax)

    vmin_vmax_slider.observe(vmin_vmax_slider_changed, "value")
    return vmin_vmax_slider


def _make_scale_button(visualization):
    scale_button = ipywidgets.Button(description="Scale")

    def scale_button_clicked(*args):
        visualization.set_values_lim()

    scale_button.on_click(scale_button_clicked)
    return scale_button


def _make_autoscale_button(visualization):
    def autoscale_button_changed(change):
        if change["new"]:
            visualization._autoscale = True
        else:
            visualization._autoscale = False

    autoscale_button = ipywidgets.ToggleButton(
        value=visualization._autoscale,
        description="Autoscale",
        tooltip="Autoscale",
    )
    autoscale_button.observe(autoscale_button_changed, "value")

    return autoscale_button


def _make_power_scale_slider(visualization):
    def powerscale_slider_changed(change):
        visualization._update_power(change["new"])

    power_scale_slider = ipywidgets.FloatSlider(
        value=visualization._get_power(),
        min=0.01,
        max=2,
        step=0.01,
        description="Power",
        tooltip="Power",
    )
    power_scale_slider.observe(powerscale_slider_changed, "value")

    return power_scale_slider


def _get_joined_titles(measurement, formatting, **kwargs):
    titles = []
    for axes_metadata in measurement.ensemble_axes_metadata:
        titles.append(axes_metadata.format_title(formatting, **kwargs))
    return "\n".join(titles)


class BaseVisualization:
    _num_coordinate_dims: int = None

    def __init__(
        self,
        array,
        coordinate_axes,
        values_axis=None,
        ensemble_axes_metadata=None,
        common_scale: bool = False,
        aspect: bool = False,
        explode: bool = False,
        overlay: bool = False,
        coordinate_labels: list[str] = None,
        coordinate_units: list[str] = None,
        values_label: str = None,
        values_units: str = None,
        ncbars: int = 0,
    ):
        self._array = array
        self._coordinate_axes = coordinate_axes
        self._common_scale = common_scale
        self._values_axis = None

        if coordinate_labels is None:
            coordinate_labels = [None] * len(coordinate_axes)

        if coordinate_units is None:
            coordinate_units = [None] * len(coordinate_axes)

        self._coordinate_labels = coordinate_labels
        self._coordinate_units = coordinate_units
        self._values_label = values_label
        self._values_units = values_units

        self._autoscale = config.get("visualize.autoscale", False)

        assert len(coordinate_axes) == self._num_coordinate_dims

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._ensemble_shape = array.shape[: -self._num_coordinate_dims]

        self._axes_types = _determine_axes_types(
            ensemble_axes_metadata, explode=explode, overlay=overlay
        )

        self._axes = _validate_axes(
            self._axes_types,
            self._ensemble_shape,
            ioff=True,
            explode=explode,
            aspect=aspect,
            ncbars=ncbars,
        )

        self._indices = self._validate_ensemble_indices()

        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._artists = None

    @property
    def autoscale(self):
        return self._autoscale

    @autoscale.setter
    def autoscale(self, autoscale: bool):
        self._autoscale = autoscale
        self.set_values_lim()

    @property
    def ensemble_axes_metadata(self):
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
    def axes_types(self):
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

    def _get_panel_titles(self):
        titles = [
            [i.format_title(".3f") for i in axis]
            for axis, axis_type in zip(self._ensemble_axes_metadata, self._axes_types)
            if axis_type == "explode"
        ]
        titles = list(itertools.product(*titles))
        return titles

    def _get_overlay_labels(self):
        labels = [
            [i.format_title(".3f") for i in axis]
            for axis, axis_type in zip(self._ensemble_axes_metadata, self._axes_types)
            if axis_type == "overlay"
        ]
        labels = list(itertools.product(*labels))
        return [" - ".join(label) for label in labels]

    def _get_array_for_current_indices(self):
        array = self._array[self._indices]

        summed_axes = tuple(
            i for i, axes_type in enumerate(self._axes_types) if axes_type == "range"
        )

        array = array.sum(axis=summed_axes)
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

    def _validate_value_limits(self, value_lim: list[float, float] = None):
        if value_lim is None:
            fixed_value_lim = [None, None]
        else:
            fixed_value_lim = value_lim.copy()

        def _get_value_lim(array, value_lim):
            if np.iscomplexobj(array):
                array = np.abs(array)

            value_lim = value_lim.copy()
            if value_lim[0] is None:
                value_lim[0] = float(np.nanmin(array))

            if value_lim[1] is None:
                value_lim[1] = float(np.nanmax(array))

            return value_lim
            # min_value = array.min()
            # max_value = array.max()
            # margin = (max_value - min_value) * 0.05
            # return [min_value - margin, max_value + margin]

        if self._common_scale:
            array = self._get_array_for_current_indices()
            common_ylim = _get_value_lim(array, fixed_value_lim)
        else:
            common_ylim = None

        value_lim = np.zeros(self.axes.shape + (2,))
        for i in np.ndindex(self.axes.shape):
            array = self._get_array_for_axis(i)
            if common_ylim is None:
                value_lim[i] = _get_value_lim(array, fixed_value_lim)
            else:
                value_lim[i] = common_ylim

        return value_lim

    def _get_coordinate_limits(self):
        shape = self._array.shape[-self._num_coordinate_dims :]
        limits = []
        for axis, n in zip(self._coordinate_axes, shape):
            coordinates = axis.coordinates(n)
            limits.append([coordinates[0], coordinates[-1]])
        return limits

    def _get_coordinate_labels(self):
        return [
            axis.format_label(units)
            for axis, units in zip(self._coordinate_axes, self._coordinate_units)
        ]

    def _get_coordinate_units(self):
        return [axis.units for axis in self._coordinate_axes]

    def _get_values_label(self):
        return ""

    def _get_values_units(self):
        return ""

    def set_xunits(self, units: str = None):
        self._coordinate_units[0] = units

        self.set_xlabel()
        self.set_xlim()

    def set_yunits(self, units: str = None):
        if self._num_coordinate_dims == 1:
            self._values_units = units
        elif self._num_coordinate_dims == 2:
            self._coordinate_units[1] = units
        else:
            raise NotImplementedError

        self.set_ylabel()
        self.set_ylim()

    def set_xlabel(self, label: str = None):
        if label is not None:
            self._coordinate_labels[0] = label

        if self._coordinate_labels[0] is None:
            label = self._coordinate_axes[0].format_label(self._coordinate_units[0])
        else:
            label = self._coordinate_labels[0]

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            ax.set_xlabel(label)

    def set_ylabel(self, label: str = None):
        if self._num_coordinate_dims == 1:
            if label is not None:
                self._values_label = label

            if self._values_label is None:
                label = ""

        elif self._num_coordinate_dims == 2:
            if label is not None:
                self._coordinate_labels[1] = label

            if self._coordinate_labels[1] is None:
                label = self._coordinate_axes[1].format_label(self._coordinate_units[1])

        else:
            raise NotImplementedError

        for i, j in np.ndindex(self.axes.shape):
            if i == 0:
                ax = self.axes[i, j]
                ax.set_ylabel(label)

    def set_ylim(self, ylim: tuple[float, float] = None):
        if self._num_coordinate_dims == 1:
            ylim = self._validate_value_limits(ylim)
        elif self._num_coordinate_dims == 2:
            ylim = self._get_coordinate_limits()[1]
        else:
            raise NotImplementedError

        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_ylim(ylim[i])

    def set_xlim(self, xlim: tuple[float, float] = None):
        if self._num_coordinate_dims == 1:
            xlim = self._validate_value_limits(xlim)
        elif self._num_coordinate_dims == 2:
            xlim = self._get_coordinate_limits()[0]
        else:
            raise NotImplementedError

        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_xlim(xlim[i])

    @abstractmethod
    def set_values_lim(self, values_lim: tuple[float, float] = None):
        pass

    def set_column_titles(
        self,
        titles: str | list[str] = None,
        pad: float = 10.0,
        format: str = ".3g",
        units: str = None,
        fontsize: float = 12,
        **kwargs,
    ):
        if titles is None:
            titles = [
                title[0] for title in self._get_panel_titles()[:: self.axes.shape[0]]
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
                title[1] for title in self._get_panel_titles()[: self.axes.shape[1]]
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

    def axis_off(self, spines: bool = True):
        for ax in np.array(self.axes).ravel():
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if not spines:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

    @abstractmethod
    def set_artists(self):
        pass

    @abstractmethod
    def update_artists(self):
        pass

    def make_widgets(self):
        if ipywidgets is None:
            raise ipywidgets_not_installed

        widgets = {}
        widgets["canvas"] = self.axes.fig.canvas
        widgets["sliders"] = make_sliders_from_ensemble_axes(
            self,
            self.axes_types,
        )
        widgets["scale_button"] = _make_scale_button(self)
        widgets["scale_button"].layout = ipywidgets.Layout(width="20%")

        widgets["autoscale_button"] = _make_autoscale_button(self)
        widgets["autoscale_button"].layout = ipywidgets.Layout(width="30%")

        widgets["continuous_update_button"] = _make_continuous_button(
            widgets["sliders"]
        )
        widgets["continuous_update_button"].layout = ipywidgets.Layout(width="50%")
        return widgets

    def layout_widgets(self):
        widgets = self.make_widgets()

        scale_box = ipywidgets.VBox(
            [
                ipywidgets.HBox(
                    [
                        widgets["scale_button"],
                        widgets["autoscale_button"],
                        widgets["continuous_update_button"],
                    ]
                )
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
        coordinate_axis,
        values_axis,
        ensemble_axes=None,
        common_scale: bool = True,
        explode: Sequence[str] | bool = False,
        overlay: Sequence[str] | bool = False,
        # figsize: tuple[float, float] = None,
        # interact: bool = False,
        **kwargs,
    ):
        super().__init__(
            array,
            coordinate_axis,
            values_axis,
            ensemble_axes,
            common_scale=common_scale,
            aspect=False,
            explode=explode,
            overlay=overlay,
        )

        self.set_artists()
        self.set_xlabel()
        self.set_ylabel()
        self.set_legends()

    def set_artists(self, **kwargs):
        x = self._coordinate_axes[0].coordinates(self._array.shape[-1])
        labels = self._get_overlay_labels()
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


class VisualizationImshow(BaseVisualization):
    _num_coordinate_dims = 2

    def __init__(
        self,
        array,
        coordinate_axis,
        values_axis,
        ensemble_axes=None,
        common_scale: bool = False,
        explode: Sequence[str] | bool = False,
        cmap: str = None,
        power: float = 1,
        vmin: float = None,
        vmax: float = None,
        cbar:bool = False,
        # figsize: tuple[float, float] = None,
        # interact: bool = False,
        **kwargs,
    ):
        if cmap is None and np.iscomplexobj(array):
            cmap = config.get("visualize.phase_cmap", "hsluv")
        elif cmap is None:
            cmap = config.get("visualize.cmap", "viridis")

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
            array,
            coordinate_axis,
            values_axis,
            ensemble_axes,
            common_scale=common_scale,
            aspect=True,
            explode=explode,
            overlay=False,
            ncbars=ncbars,
        )

        self.set_normalization(power=power, vmin=vmin, vmax=vmax)
        self.set_artists()
        self.set_xlabel()
        self.set_ylabel()

        if cbar:
            self.set_cbars()


    def set_artists(
        self,
    ):
        artists = np.zeros(self.axes.shape, dtype=object)

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            norm = self._normalization[i]
            array = self._get_array_for_axis(axis_index=i)

            if np.iscomplexobj(array):
                abs_array = np.abs(array)
                alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

                cmap = self._cmap

                if cmap == "hsluv":
                    cmap = hsluv_cmap

                im1 = ax.imshow(
                    np.angle(array).T,
                    origin="lower",
                    interpolation="none",
                    alpha=alpha.T,
                    vmin=-np.pi,
                    vmax=np.pi,
                    cmap=cmap,
                )

                im2 = ax.imshow(
                    abs_array.T,
                    origin="lower",
                    interpolation="none",
                    cmap="gray",
                    zorder=-1,
                )

                im2.set_norm(norm)
                ims = [im1, im2]
            else:
                ims = [
                    ax.imshow(
                        array.T,
                        origin="lower",
                        interpolation="none",
                        cmap=self._cmap,
                    )
                ]

            artists.itemset(i, ims)

        self._artists = artists

    def update_artists(self):
        for i in np.ndindex(self.axes.shape):
            im = self._artists[i]
            array = self._get_array_for_axis(axis_index=i)

            if len(im) == 1:
                im[0].set_data(array.T)
            else:
                norm = self._normalization[i]
                abs_array = np.abs(array)
                alpha = norm(abs_array)
                alpha = np.clip(alpha, a_min=0, a_max=1)

                im[0].set_alpha(alpha)
                im[0].set_data(np.angle(array).T)
                im[1].set_data(abs_array)

    def set_values_lim(self, values_lim: tuple[float, float] = None):
        values_lim = self._validate_value_limits(values_lim)

        for i in np.ndindex(self.axes.shape):
            im = self._artists[i]
            norm = self._normalization[i]
            norm.vmin = values_lim[i][0]
            norm.vmax = values_lim[i][1]

            if len(im) == 2:
                array = self._get_array_for_axis(i)
                alpha = norm(np.abs(array))
                alpha = np.clip(alpha, a_min=0, a_max=1)
                im[0].set_alpha(alpha)

    def set_normalization(
        self,
        power: float = None,
        vmin: float = None,
        vmax: float = None,
    ):
        if self._common_scale:
            vmin, vmax = self._validate_value_limits([vmin, vmax])

        normalization = np.zeros(self.axes.shape, dtype=object)
        for i in np.ndindex(self.axes.shape):
            array = self._get_array_for_axis(i)
            # im = self._artists[i]

            if power == 1.0:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)

            if np.iscomplexobj(array):
                array = np.abs(array)

            norm.autoscale_None(array)

            normalization[i] = norm

        self._normalization = normalization

    def _update_power(self, power: float = 1.0):
        for i in np.ndindex(self.axes.shape):
            artists = self._artists[i]
            norm = self._normalization[i]

            if (power != 1.0) and isinstance(norm, colors.Normalize):
                self._normalization[i] = colors.PowerNorm(
                    gamma=power, vmin=norm.vmin, vmax=norm.vmax
                )
                artists.norm = self._normalization[i]

            if (power == 1.0) and isinstance(norm, colors.PowerNorm):
                self._normalization[i] = colors.Normalize(
                    vmin=norm.vmin, vmax=norm.vmax
                )
                artists.norm = self._normalization[i]

            if (power != 1.0) and isinstance(norm, colors.PowerNorm):
                self._normalization[i].gamma = power

    def set_cbar_labels(self, label: str = None, **kwargs):
        if self._domain_coloring:
            for cbar1, cbar2 in self._cbars.values():
                cbar1.set_label("arg", rotation=0, ha="center", va="top")
                cbar1.ax.yaxis.set_label_coords(0.5, -0.02)
                cbar1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
                cbar1.set_ticklabels(
                    [
                        r"$-\pi$",
                        r"$-\dfrac{\pi}{2}$",
                        "$0$",
                        r"$\dfrac{\pi}{2}$",
                        r"$\pi$",
                    ]
                )
                cbar2.set_label("abs", rotation=0, ha="center", va="top")
                cbar2.ax.yaxis.set_label_coords(0.5, -0.02)
                cbar2.formatter.set_powerlimits((0, 0))
                cbar2.formatter.set_useMathText(True)
                cbar2.ax.yaxis.set_offset_position("left")

        else:
            super().set_cbar_labels(label, **kwargs)

    def set_cbars(self, **kwargs):
        cbars = defaultdict(list)

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            ims = self._artists[i]

            if isinstance(self.axes, AxesGrid):
                if ax in self.axes._caxes.keys():
                    cax = self.axes._caxes[ax]
                else:
                    continue

                for j, im in enumerate(ims):
                    cbars[ax].append(plt.colorbar(im, cax=cax[j], **kwargs))

            else:
                if isinstance(im, np.ndarray):
                    for j, image in enumerate(im):
                        cbars[ax].append(plt.colorbar(image, ax=ax, **kwargs))
                else:
                    cbars[ax].append(plt.colorbar(im, ax=ax, **kwargs))

        self._cbars = cbars

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

    def set_scalebars(
        self,
        panel_loc: tuple[int, ...] | str = "lower right",
        label: str = "",
        size: float = None,
        loc: str = "lower right",
        borderpad: float = 0.5,
        formatting: str = ".3f",
        size_vertical: float = None,
        sep: float = 6,
        pad: float = 0.3,
        label_top: bool = True,
        frameon: bool = False,
        **kwargs,
    ):
        panel_locs = {
            "all": tuple(np.ndindex(self.axes.shape)),
            "upper left": ((0, -1)),
            "upper right": ((-1, -1),),
            "lower left": ((0, 0),),
            "lower right": ((-1, 0),),
        }

        if isinstance(panel_loc, str):
            panel_loc = panel_locs.get(panel_loc, ((0, 0),))

        conversion = _get_conversion_factor(
            self._coordinate_units[0], self._coordinate_axes[0].units
        )

        if size is None:
            limits = self._get_coordinate_limits()[0]
            size = (limits[1] - limits[0]) / 3

        if size_vertical is None:
            limits = self._get_coordinate_limits()[1]
            size = (limits[1] - limits[0]) / 3

        size = conversion * size
        size_vertical = conversion * size_vertical

        if label is None:
            label = f"{size:>{formatting}} {self._get_coordinate_units()}"

        for size_bar in self._size_bars:
            size_bar.remove()

        self._size_bars = []
        for ax in panel_loc:
            ax = self.axes[ax]
            anchored_size_bar = AnchoredSizeBar(
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
            ax.add_artist(anchored_size_bar)
            self._size_bars.append(anchored_size_bar)


class MeasurementVisualization(metaclass=ABCMeta):
    def __init__(
        self,
        measurements: BaseMeasurements,
        axes: AxesGrid | np.ndarray,
        axes_types: Sequence[str] = (),
        autoscale: bool = False,
    ):
        self._measurements = measurements.to_cpu()
        self._axes = axes
        self._axes_types = axes_types
        self._indices = self._validate_ensemble_indices()
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._metadata_labels = np.array([])
        self._xunits = None
        self._yunits = None
        self._autoscale = autoscale

        for ax in np.array(self.axes).ravel():
            ax.ticklabel_format(
                style="sci", scilimits=(-3, 3), axis="both", useMathText=True
            )

        self.fig.canvas.header_visible = False

    @property
    def autoscale(self):
        return self._autoscale

    @autoscale.setter
    def autoscale(self, value):
        self._autoscale = value

    @property
    def fig(self):
        return self._axes[0, 0].get_figure()

    @property
    @abstractmethod
    def artists(self):
        pass

    def adjust_figure_aspect(self):
        bbox = self.fig.get_tightbbox()
        aspect = (bbox.ymax - bbox.ymin) / (bbox.xmax - bbox.xmin)
        size = self.fig.get_size_inches()
        self.fig.set_size_inches((size[0], size[0] * aspect))

    def adjust_axes_position(self, rect):
        self.axes.divider.set_position(rect)

    def _generate_measurements(self, keepdims: bool = True):
        indexed_measurements = self._get_indexed_measurements()

        shape = tuple(
            n if axes_type != "overlay" else 1
            for n, axes_type in zip(
                indexed_measurements.ensemble_shape, self._axes_types
            )
        )

        for indices in np.ndindex(*shape):
            axes_index = ()
            for i, axes_type in zip(indices, self._axes_types):
                if axes_type == "explode":
                    axes_index += (i,)
            axes_index = (axes_index + (0,) * (2 - len(axes_index)))[:2]

            indices = tuple(
                i if axes_type != "overlay" else slice(None)
                for i, axes_type in zip(indices, self._axes_types)
            )
            yield axes_index, indexed_measurements.get_items(indices, keepdims=keepdims)

    def set_axes_padding(self, padding: float | tuple[float, float] = (0.0, 0.0)):
        """
        Set the padding between the axes in an :class:`.AxesGrid`.

        Parameters
        ----------
        padding : float or tuple of float
            The padding along columns and rows.
        """
        self._axes.set_axes_padding(padding)

    def _get_axes_from_axes_types(self, axes_type):
        return tuple(
            i
            for i, checked_axes_type in enumerate(self.axes_types)
            if checked_axes_type == axes_type
        )

    def _get_indexed_measurements(self, keepdims: bool = True):
        indexed = self.measurements.get_items(self._indices, keepdims=keepdims)

        if keepdims:
            summed_axes = tuple(
                i
                for i, axes_type in enumerate(self._axes_types)
                if axes_type == "range"
            )
        else:
            i = 0
            summed_axes = ()
            for axes_type in self._axes_types:
                if axes_type == "range":
                    summed_axes += (i,)
                    i += 1

        indexed = indexed.sum(axis=summed_axes, keepdims=keepdims)

        return indexed

    def set_column_titles(
        self,
        titles: str | list[str] = None,
        pad: float = 10.0,
        format: str = ".3g",
        units: str = None,
        fontsize=12,
        **kwargs,
    ):
        indexed_measurements = self._get_indexed_measurements(keepdims=False)

        if titles is None or titles is True:
            if not len(indexed_measurements.ensemble_shape):
                return

            # TODO: same for row titles
            j = 0
            for j, axes_type in enumerate(self.axes_types):
                if not axes_type == "overlay":
                    break

            axes_metadata = indexed_measurements.ensemble_axes_metadata[j]

            if hasattr(axes_metadata, "to_nonlinear_axis"):
                axes_metadata = axes_metadata.to_nonlinear_axis(
                    indexed_measurements.ensemble_shape[j]
                )

            titles = []
            for i, axis_metadata in enumerate(axes_metadata):
                titles.append(
                    axis_metadata.format_title(
                        format, units=units, include_label=i == 0
                    )
                )
                if i == indexed_measurements.ensemble_shape[j]:
                    break

        elif isinstance(titles, str):
            if indexed_measurements.ensemble_shape:
                n = indexed_measurements.ensemble_shape[0]
            else:
                n = 1

            titles = [titles] * n

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

    def set_xlim(self, *args, **kwargs):
        for ax in np.array(self.axes).ravel():
            ax.set_xlim(args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        for ax in np.array(self.axes).ravel():
            ax.set_ylim(args, **kwargs)

    @abstractmethod
    def _get_default_xlabel(self, units=None):
        pass

    @abstractmethod
    def _get_default_ylabel(self, units=None):
        pass

    def set_xlabels(self, label: str = None):
        if label is None:
            label = self._get_default_xlabel(units=self._xunits)

        for i, j in np.ndindex(self.axes.shape):  # noqa
            if j == 0:
                ax = self.axes[i, j]
                ax.set_xlabel(label)

    def set_ylabels(self, label: str = None):
        if label is None:
            label = self._get_default_ylabel(units=self._yunits)

        for i, j in np.ndindex(self.axes.shape):  # noqa
            if i == 0:
                ax = self.axes[i, j]
                ax.set_ylabel(label)

    @abstractmethod
    def set_xlim(self):
        pass

    @abstractmethod
    def set_ylim(self):
        pass

    @abstractmethod
    def _get_default_xunits(self):
        pass

    @abstractmethod
    def _get_default_yunits(self):
        pass

    def set_xunits(self, units: str = None):
        """
        Set the units for the x-axis.

        Parameters
        ----------
        units : str
            The name of the units. Must be compatible with existing units.
        """
        if units is None:
            self._xunits = self._get_default_xunits()
        else:
            self._xunits = units

        self.set_xlabels()
        self.set_xlim()

    def set_yunits(self, units: str = None):
        """
        Set the units for the y-axis.

        Parameters
        ----------
        units : str
            The name of the units. Must be compatible with existing units.
        """
        if units is None:
            self._yunits = self._get_default_yunits()
        else:
            self._yunits = units

        self.set_ylabels()
        self.set_ylim()

    def set_row_titles(
        self,
        titles: str | list[str] = None,
        shift: float = 0.0,
        format: str = ".3g",
        units: str = None,
        **kwargs,
    ):
        """
        Set the titles for the rows of the grid of axes.

        Parameters
        ----------
        titles : str or list of str, optional
            If given as list, each item is given as a title for a row, the list must have the same length as the number
            of rows. If given as string the same title is given to all rows. If not given the titles are derived from
            the axes metadata.
        shift : float, optional
            Horizontal shift of the title positions.
        format : str, optional
            String formatting of titles derived from axes metadata.
        units : str, optional
            The units used for titles derived from axes metadata.
        """
        indexed_measurements = self._get_indexed_measurements()

        if not "fontsize" in kwargs:
            kwargs.update({"fontsize": 12})

        if titles is None:
            if not len(indexed_measurements.ensemble_shape) > 1:
                return

            axes_metadata = indexed_measurements.ensemble_axes_metadata[1]

            if hasattr(axes_metadata, "to_nonlinear_axis"):
                axes_metadata = axes_metadata.to_nonlinear_axis(
                    indexed_measurements.ensemble_shape[1]
                )

            titles = []
            for i, axis_metadata in enumerate(axes_metadata):
                titles.append(
                    axis_metadata.format_title(
                        format, units=units, include_label=i == 0
                    )
                )

                if i == indexed_measurements.ensemble_shape[1]:
                    break
        elif isinstance(titles, str):
            titles = [titles] * max(len(indexed_measurements.ensemble_shape), 1)

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
                **kwargs,
            )
            row_titles.append(annotation)

        self._row_titles = row_titles

    @property
    def ncols(self):
        return self._axes.shape[0]

    @property
    def nrows(self):
        return self._axes.shape[1]

    @property
    def axes_types(self):
        return self._axes_types

    @property
    def indices(self):
        return self._indices

    @property
    def measurements(self):
        return self._measurements

    @property
    def axes(self):
        return self._axes

    def _validate_ensemble_indices(self, indices: int | tuple[int, ...] = ()):
        if isinstance(indices, int):
            indices = (indices,)

        num_ensemble_dims = len(self.measurements.ensemble_shape)
        explode_axes = self._get_axes_from_axes_types("explode")
        overlay_axes = self._get_axes_from_axes_types("overlay")
        num_indexing_axes = num_ensemble_dims - len(explode_axes) - len(overlay_axes)

        if len(indices) > num_indexing_axes:
            raise ValueError

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
        self.update_panel_labels()

    @abstractmethod
    def update_artists(self):
        pass

    def get_global_vmin_vmax(
        self, vmin: float = None, vmax: float = None
    ) -> tuple[float, float]:
        measurements = self._get_indexed_measurements()

        if measurements.is_complex:
            measurements = measurements.abs()

        if vmin is None:
            vmin = float(np.nanmin(measurements.array))

        if vmax is None:
            vmax = float(np.nanmax(measurements.array))

        return vmin, vmax

    def set_panel_labels(
        self,
        labels: str = "metadata",
        frameon: bool = True,
        loc: str = "upper left",
        pad: float = 0.1,
        borderpad: float = 0.1,
        prop: dict = None,
        formatting: str = ".3g",
        units: str = None,
        **kwargs,
    ):
        labels_type = labels

        if labels == "alphabetic":
            labels = string.ascii_lowercase
            labels = [f"({label})" for label in labels]
            if config.get("visualize.use_tex", False):
                labels = [f"${label}$" for label in labels]
        elif labels == "metadata":
            labels = []

            for i, measurement in self._generate_measurements(keepdims=True):
                titles = []
                for axes_metadata in measurement.ensemble_axes_metadata:
                    titles.append(
                        axes_metadata.format_title(formatting, units=units, **kwargs)
                    )
                labels.append("\n".join(titles))

            # for i, measurement in self.generate_measurements(keepdims=True):
            #     labels.append(_get_joined_titles(measurement, formatting))
        elif (
            not isinstance(labels, (tuple, list))
            and len(labels) != np.array(self.axes).size
        ):
            raise ValueError()

        if prop is None:
            prop = {}

        for old_label in self._panel_labels:
            old_label.remove()

        anchored_text = []
        for ax, l in zip(np.array(self.axes).ravel(), labels):
            at = AnchoredText(
                l,
                pad=pad,
                borderpad=borderpad,
                frameon=frameon,
                loc=loc,
                prop=prop,
                **kwargs,
            )
            at.formatting = formatting

            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            anchored_text.append(at)

        self._panel_labels = anchored_text

        if labels_type == "metadata":
            self._metadata_labels = self._panel_labels
        else:
            self._metadata_labels = []

    def update_panel_labels(self):
        for anchored_text, (i, measurement) in zip(
            self._metadata_labels, self._generate_measurements(keepdims=True)
        ):
            label = _get_joined_titles(measurement, anchored_text.formatting)
            anchored_text.txt.set_text(label)

    def animate(
        self,
        interval=20,
        blit=True,
        repeat: bool = False,
        adjust_scale: bool = True,
        **kwargs,
    ):
        def update(i):
            self.set_ensemble_indices((i,))
            if adjust_scale:
                self._update_vmin_vmax()

            return self.artists.ravel()

        index_axes = self._get_axes_from_axes_types("index")

        if len(index_axes) == 0:
            raise RuntimeError()

        frames = self.measurements.shape[index_axes[0]]

        animation = FuncAnimation(
            self.fig,
            update,
            frames=frames,
            interval=interval,
            blit=blit,
            repeat=repeat,
            **kwargs,
        )

        return animation


class BaseMeasurementVisualization2D(MeasurementVisualization):
    def __init__(
        self,
        measurements: _BaseMeasurement2D | IndexedDiffractionPatterns,
        ax: Axes = None,
        common_scale: bool = False,
        cbar: bool = False,
        explode: bool = None,
        figsize: tuple[float, float] = None,
        interact: bool = False,
    ):
        # measurements = measurements.compute().to_cpu()

        axes_types = _determine_axes_types(
            measurements=measurements, explode=explode, overlay=None
        )

        if "overlay" in axes_types:
            raise NotImplementedError

        axes = _validate_axes(
            measurements=measurements,
            ax=ax,
            explode=explode,
            overlay=None,
            cbar=cbar,
            common_color_scale=common_scale,
            figsize=figsize,
            ioff=interact,
        )

        super().__init__(measurements=measurements, axes=axes, axes_types=axes_types)

        self._xunits = None
        self._yunits = None
        self._scale_units = None
        self._xlabel = None
        self._ylabel = None
        self._column_titles = []
        self._row_titles = []
        self._artists = None
        self._autoscale = config.get("visualize.autoscale", False)
        self._common_scale = common_scale
        self._size_bars = []

        if self.ncols > 1:
            self.set_column_titles()

        if self.nrows > 1:
            self.set_row_titles()

    def _get_vmin_vmax(self):
        vmin = np.inf
        vmax = -np.inf
        for norm in self._normalization.ravel():
            vmin = min(vmin, norm.vmin)
            vmax = max(vmax, norm.vmax)
        return vmin, vmax

    def _get_power(self):
        power = None
        for norm in self._normalization.ravel():
            if isinstance(norm, colors.PowerNorm):
                if power is None:
                    power = norm.gamma
                else:
                    power = min(power, norm.gamma)
            else:
                if power is None:
                    power = 1.0
                else:
                    power = min(power, 1.0)

        return power

    def set_normalization(
        self,
        power: float = None,
        vmin: float = None,
        vmax: float = None,
    ):
        if self._common_scale:
            vmin, vmax = self.get_global_vmin_vmax(vmin=vmin, vmax=vmax)

        self._normalization = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):
            if power == 1.0:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)

            if measurement.is_complex:
                measurement = measurement.abs()

            norm.autoscale_None(measurement.array[np.isnan(measurement.array) == 0])

            self._normalization[i] = norm

    def _update_vmin_vmax(self, vmin: float = None, vmax: float = None):
        for norm, measurement in zip(
            self._normalization.ravel(), self._generate_measurements(keepdims=False)
        ):
            norm.vmin = vmin
            norm.vmax = vmax

    def _update_power(self, power: float = 1.0):
        for i, measurement in self._generate_measurements(keepdims=False):
            artists = self._artists[i]
            norm = self._normalization[i]

            if (power != 1.0) and isinstance(norm, colors.Normalize):
                self._normalization[i] = colors.PowerNorm(
                    gamma=power, vmin=norm.vmin, vmax=norm.vmax
                )
                artists.norm = self._normalization[i]

            if (power == 1.0) and isinstance(norm, colors.PowerNorm):
                self._normalization[i] = colors.Normalize(
                    vmin=norm.vmin, vmax=norm.vmax
                )
                artists.norm = self._normalization[i]

            if (power != 1.0) and isinstance(norm, colors.PowerNorm):
                self._normalization[i].gamma = power

    def add_area_indicator(self, area_indicator, panel="first", **kwargs):
        xlim = self.axes[0, 0].get_xlim()
        ylim = self.axes[0, 0].get_ylim()

        for i, ax in enumerate(np.array(self.axes).ravel()):
            if panel == "first" and i == 0:
                area_indicator._add_to_visualization(ax, **kwargs)
            elif panel == "all":
                area_indicator._add_to_visualization(ax, **kwargs)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    @property
    def artists(self):
        return self._artists

    @abstractmethod
    def set_artists(self):
        pass

    def set_scale_units(self, units: str = None):
        if units is None:
            units = self.measurements.metadata.get("units", "")

        self._scale_units = units

    def set_cbar_labels(self, label: str = None, **kwargs):
        if label is None:
            label = self.measurements.metadata.get("label", "")

        # TODO: make units work more generally

        if self._scale_units is None or len(self._scale_units) == 0:
            label = f"{label}"
        else:
            label = f"{label} [{self._scale_units}]"

        for cbars in self._cbars.values():
            for cbar in cbars:
                cbar.set_label(label, **kwargs)
                cbar.formatter.set_powerlimits((-3, 3))
                cbar.formatter.set_useMathText(True)
                cbar.ax.yaxis.set_offset_position("left")

    def set_cbar_padding(self, padding: tuple[float, float] = (0.1, 0.1)):
        self._axes.set_cbar_padding(padding)

    def set_cbar_size(self, fraction: float):
        self._axes.set_cbar_size(fraction)

    def set_cbar_spacing(self, spacing: float):
        self._axes.set_cbar_spacing(spacing)

    def set_cbars(self, **kwargs):
        cbars = defaultdict(list)

        for i, _ in self._generate_measurements():
            ax = self.axes[i]
            images = self._artists[i]

            if isinstance(self.axes, AxesGrid):
                if ax in self.axes._caxes.keys():
                    cax = self.axes._caxes[ax]
                else:
                    continue

                if isinstance(images, np.ndarray):
                    for j, image in enumerate(images):
                        cbars[ax].append(plt.colorbar(image, cax=cax[j], **kwargs))
                else:
                    cbars[ax].append(plt.colorbar(images, cax=cax[0], **kwargs))

            else:
                if isinstance(images, np.ndarray):
                    for j, image in enumerate(images):
                        cbars[ax].append(plt.colorbar(image, ax=ax, **kwargs))
                else:
                    cbars[ax].append(plt.colorbar(images, ax=ax, **kwargs))

        self._cbars = cbars

    def set_scalebars(
        self,
        panel_loc: tuple[int, ...] = ((-1, 0),),
        label: str = "",
        size: float = None,
        loc: str = "lower right",
        borderpad: float = 0.5,
        formatting: str = ".3f",
        size_vertical: float = None,
        sep: float = 6,
        pad: float = 0.3,
        label_top: bool = True,
        frameon: bool = False,
        **kwargs,
    ):
        if panel_loc == "all":
            panel_loc = np.ndindex(self.axes.shape)  # noqa
            panel_loc = tuple(panel_loc)
        elif panel_loc == "upper left":
            panel_loc = ((0, -1),)
        elif panel_loc == "upper right":
            panel_loc = ((-1, -1),)
        elif panel_loc == "lower left":
            panel_loc = ((0, 0),)
        elif panel_loc == "lower right":
            panel_loc = ((-1, 0),)
        else:
            panel_loc = ((0, 0),)

        conversion = _get_conversion_factor(
            self._xunits, self.measurements.axes_metadata[-2].units
        )

        if size is None:
            size = (
                self.measurements.base_axes_metadata[-2].sampling
                * self.measurements.base_shape[-2]
                / 3
            )

        if size_vertical is None:
            size_vertical = (
                self.measurements.base_axes_metadata[-1].sampling
                * self.measurements.base_shape[-1]
                / 20
            )

        size = conversion * size
        size_vertical = conversion * size_vertical

        if label is None:
            label = f"{size:>{formatting}} {self._xunits}"

        for size_bar in self._size_bars:
            size_bar.remove()

        self._size_bars = []
        for ax in panel_loc:
            ax = self.axes[ax]
            anchored_size_bar = AnchoredSizeBar(
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
            ax.add_artist(anchored_size_bar)
            self._size_bars.append(anchored_size_bar)

    def axis_off(self, spines: bool = True):
        for ax in np.array(self.axes).ravel():
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if not spines:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

    def adjust_tight_bbox(self):
        # x_extent = self.measurements._plot_extent_x(self._xunits)
        # y_extent = self.measurements._plot_extent_y(self._yunits)
        # aspect = (y_extent[1] - y_extent[0]) / (x_extent[1] - x_extent[0])

        aspect = 1

        size_x = self.fig.get_size_inches()[0]
        size_y = size_x * aspect

        self.fig.set_size_inches((size_x, size_y))
        self.fig.subplots_adjust(left=0, bottom=0, right=1.0, top=1)


class MeasurementVisualization2D(BaseMeasurementVisualization2D):
    """
    Show the image(s) using matplotlib.

    Parameters
    ----------
    measurements : _BaseMeasurement2D
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axis. This is not available for image grids.
    cmap : str, optional
        Matplotlib colormap name used to map scalar data to colors. Ignored if image array is complex.
    power : float
        Show image on a power scale.
    vmin : float, optional
        Minimum of the intensity color scale. Default is the minimum of the array values.
    vmax : float, optional
        Maximum of the intensity color scale. Default is the maximum of the array values.
    common_scale : bool, optional
        If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
        it is requested). Default is False.
    cbar : bool, optional
        Add colorbar(s) to the image(s). The position and size of the colorbar(s) may be controlled by passing
        keyword arguments to `mpl_toolkits.axes_grid1.axes_grid.ImageGrid` through `image_grid_kwargs`.

    """

    def __init__(
        self,
        measurements: _BaseMeasurement2D,
        ax: Axes = None,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_scale: bool = False,
        explode: bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
    ):
        super().__init__(
            measurements,
            ax=ax,
            cbar=cbar,
            common_scale=common_scale,
            explode=explode,
            figsize=figsize,
            interact=interact,
        )

        if cmap is None and measurements.is_complex:
            cmap = config.get("visualize.phase_cmap", "hsluv")
        elif cmap is None:
            cmap = config.get("visualize.cmap", "viridis")

        self._normalization = None
        self._cmap = cmap

        self.set_normalization(power=power, vmin=vmin, vmax=vmax)
        self.set_artists()

        if cbar:
            self.set_cbars()
            self.set_scale_units()
            self.set_cbar_labels()

        self.set_extent()
        self.set_xunits()
        self.set_yunits()
        self.set_xlabels()
        self.set_ylabels()
        self.set_column_titles()

    @property
    def _domain_coloring(self):
        return self.measurements.is_complex

    def set_cbar_labels(self, label: str = None, **kwargs):
        if self._domain_coloring:
            for cbar1, cbar2 in self._cbars.values():
                cbar1.set_label("arg", rotation=0, ha="center", va="top")
                cbar1.ax.yaxis.set_label_coords(0.5, -0.02)
                cbar1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
                cbar1.set_ticklabels(
                    [
                        r"$-\pi$",
                        r"$-\dfrac{\pi}{2}$",
                        "$0$",
                        r"$\dfrac{\pi}{2}$",
                        r"$\pi$",
                    ]
                )
                cbar2.set_label("abs", rotation=0, ha="center", va="top")
                cbar2.ax.yaxis.set_label_coords(0.5, -0.02)
                cbar2.formatter.set_powerlimits((0, 0))
                cbar2.formatter.set_useMathText(True)
                cbar2.ax.yaxis.set_offset_position("left")

        else:
            super().set_cbar_labels(label, **kwargs)

    def _get_default_xlabel(self, units: str = None):
        return self.measurements.axes_metadata[-2].format_label(units=units)

    def _get_default_ylabel(self, units: str = None):
        return self.measurements.axes_metadata[-1].format_label(units=units)

    def _get_default_xunits(self):
        return self.measurements.axes_metadata[-2].units

    def _get_default_yunits(self):
        return self.measurements.axes_metadata[-1].units

    def set_xlim(self):
        self.set_extent()

    def set_ylim(self):
        self.set_extent()

    def set_extent(self, extent=None):
        if extent is None:
            x_extent = self.measurements._plot_extent_x(self._xunits)
            y_extent = self.measurements._plot_extent_y(self._yunits)
            extent = x_extent + y_extent

        for image in self._artists.ravel():
            image.set_extent(extent)

    def _add_domain_coloring_imshow(self, ax, array, norm):
        abs_array = np.abs(array)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        if self._cmap is None:
            cmap = config.get("phase_cmap", "hsluv")
        else:
            cmap = self._cmap

        if cmap == "hsluv":
            cmap = hsluv_cmap

        im1 = ax.imshow(
            np.angle(array).T,
            origin="lower",
            interpolation="none",
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
        )

        im2 = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation="none",
            cmap="gray",
            zorder=-1,
        )

        return im1, im2

    def _add_real_imshow(self, ax, array):
        im = ax.imshow(
            array.T,
            origin="lower",
            interpolation="none",
            cmap=self._cmap,
        )
        return im

    def set_artists(
        self,
    ):
        if self.measurements.is_complex:
            artists_per_axes = 2
        else:
            artists_per_axes = 1

        images = np.zeros(self.axes.shape + (artists_per_axes,), dtype=object)

        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]
            norm = self._normalization[i]

            if self._domain_coloring:
                images[i] = self._add_domain_coloring_imshow(
                    ax, measurement.array, norm
                )
                images[i][1].set_norm(norm)
            else:
                images[i] = self._add_real_imshow(ax, measurement.array)
                images[i][0].set_norm(norm)

        if images.shape[-1] == 1:
            images = np.squeeze(images, -1)

        self._artists = images

    def _update_domain_coloring_alpha(self, values, image, normalization):
        alpha = normalization(values)
        alpha = np.clip(alpha, a_min=0, a_max=1)
        image.set_alpha(alpha)

    def _update_vmin_vmax(self, vmin: float = None, vmax: float = None):
        super()._update_vmin_vmax(vmin=vmin, vmax=vmax)
        if self._domain_coloring:
            for i, measurement in self._generate_measurements(keepdims=False):
                images = self._artists[i]
                abs_array = np.abs(measurement.array).T
                self._update_domain_coloring_alpha(
                    abs_array, images[0], self._normalization[i]
                )

    def update_artists(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            images = self._artists[i]

            array = measurement.array.T

            if self._domain_coloring:
                abs_array = np.abs(array)
                self._update_domain_coloring_alpha(
                    abs_array, images[0], self._normalization[i]
                )
                images[0].set_data(np.angle(array))
                images[1].set_data(abs_array)
            else:
                images.set_data(array)

    @property
    def widgets(self):
        if widgets is None:
            raise ipywidgets_not_installed

        canvas = self.fig.canvas

        def index_update_callback(change):
            if self._autoscale:
                vmin, vmax = self.get_global_vmin_vmax()
                self._update_vmin_vmax(vmin, vmax)

        sliders = make_sliders_from_ensemble_axes(
            self,
            self.axes_types,  # callbacks=(index_update_callback,)
        )
        power_scale_button = _make_power_scale_slider(self)
        scale_button = _make_scale_button(self)
        autoscale_button = _make_autoscale_button(self)
        continuous_update_button = _make_continuous_button(sliders)

        scale_button.layout = widgets.Layout(width="20%")
        autoscale_button.layout = widgets.Layout(width="30%")
        continuous_update_button.layout = widgets.Layout(width="50%")

        scale_box = widgets.VBox(
            [widgets.HBox([scale_button, autoscale_button, continuous_update_button])]
        )
        scale_box.layout = widgets.Layout(width="300px")

        gui = widgets.VBox(
            [
                widgets.VBox(sliders),
                scale_box,
                # vmin_vmax_slider,
                power_scale_button,
            ]
        )

        return widgets.HBox([gui, canvas])


class MeasurementVisualization1D(MeasurementVisualization):
    def __init__(
        self,
        measurements: _BaseMeasurement1D,
        ax: Axes = None,
        common_scale: bool = True,
        explode: Sequence[str] | bool = False,
        overlay: Sequence[str] | bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
        **kwargs,
    ):
        axes_types = _determine_axes_types(
            measurements, explode=explode, overlay=overlay
        )

        axes = _validate_axes(
            measurements=measurements,
            ax=ax,
            explode=explode,
            overlay=overlay,
            cbar=False,
            common_color_scale=False,
            figsize=figsize,
            aspect=False,
            ioff=interact,
            sharey=common_scale,
        )

        super().__init__(measurements=measurements, axes=axes, axes_types=axes_types)

        self._xunits = None
        self._yunits = None
        self._xlabel = None
        self._ylabel = None
        self._column_titles = []
        self._lines = np.array([[]])
        self._common_scale = common_scale
        self.set_artists(**kwargs)
        self.set_xunits()
        self.set_yunits()
        self._autoscale = config.get("visualize.autoscale", False)

        if self.ncols > 1:
            self.set_column_titles()

    @property
    def artists(self):
        return self._artists

    def _get_default_xlabel(self, units: str = None):
        return self.measurements.axes_metadata[-1].format_label(units)

    def _get_default_ylabel(self, units: str = None):
        axes = LinearAxis(label=self.measurements.metadata.get("label", ""))
        return format_label(axes, units)

    def _get_default_xunits(self):
        return self.measurements.axes_metadata[-1].units

    def _get_default_yunits(self):
        return self.measurements.metadata.get("units", "")

    def set_xlim(self, xlim=None):
        extent = self.measurements._plot_extent(self._xunits)
        margin = (extent[1] - extent[0]) * 0.05
        if xlim is None:
            xlim = [-extent[0] - margin, extent[1] + margin]

        for i, measurement in self._generate_measurements():
            self.axes[i].set_xlim(xlim)
            artists = self.artists[i]
            for artist in artists:
                x = self._get_xdata()
                artist.set_xdata(x)

    def set_ylim(self, ylim=None):
        def _get_extent(measurements):
            min_value = measurements.min()
            max_value = measurements.max()
            margin = (max_value - min_value) * 0.05
            return [min_value - margin, max_value + margin]

        if self._common_scale and ylim is None:
            common_ylim = _get_extent(self.measurements)

        else:
            common_ylim = ylim

        for i, measurement in self._generate_measurements():
            if common_ylim is None:
                ylim = _get_extent(measurement)
            else:
                ylim = common_ylim

            self.axes[i].set_ylim(ylim)

    def set_legends(self, loc: str = "first", **kwargs):
        indices = [index for index in np.ndindex(*self.axes.shape)]

        if loc == "first":
            loc = indices[:1]
        elif loc == "last":
            loc = indices[-1:]
        elif loc == "all":
            loc = indices

        for i, _ in self._generate_measurements():
            if i in loc:
                self.axes[i].legend(**kwargs)

    def _update_vmin_vmax(self, vmin: float = None, vmax: float = None):
        self.set_ylim([vmin, vmax])
        # for _, measurement in self._generate_measurements(keepdims=False):
        #    self.set_ylim([measurement.min(), measurement.max()])

    def set_artists(self, **kwargs):
        artists = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]
            x = self._get_xdata()
            new_lines = []
            for _, line_profile in measurement.generate_ensemble(keepdims=True):
                if not "label" in kwargs:
                    labels = []
                    for axis in line_profile.ensemble_axes_metadata:
                        labels += [axis.format_title(".3f")]

                    kwargs["label"] = "-".join(labels)

                new_lines.append(
                    ax.plot(
                        x,
                        line_profile.array[(0,) * (len(line_profile.shape) - 1)],
                        **kwargs,
                    )[0]
                )
            artists.itemset(i, new_lines)

        self._artists = artists

    def _get_xdata(self):
        extent = self.measurements._plot_extent(self._xunits)
        return np.linspace(
            extent[0],
            extent[1],
            self.measurements.shape[-1],
            endpoint=False,
        )

    def update_artists(self):
        for i, measurements in self._generate_measurements(keepdims=False):
            lines = self._artists[i]
            for line, measurement in zip(lines, measurements):
                y = measurement.array
                x = self._get_xdata()
                line.set_data(x, y)

    @property
    def widgets(self):
        if widgets is None:
            raise ipywidgets_not_installed

        canvas = self.fig.canvas

        # def index_update_callback(change):
        #     if self._autoscale:
        #         vmin, vmax = self.get_global_vmin_vmax()
        #         self._update_vmin_vmax(vmin, vmax)

        sliders = make_sliders_from_ensemble_axes(
            self,
            self.axes_types,  # callbacks=(index_update_callback,)
        )
        # power_scale_button = _make_power_scale_slider(self)
        scale_button = _make_scale_button(self)
        autoscale_button = _make_autoscale_button(self)
        continuous_update_button = _make_continuous_button(sliders)

        scale_button.layout = widgets.Layout(width="20%")
        autoscale_button.layout = widgets.Layout(width="30%")
        continuous_update_button.layout = widgets.Layout(width="50%")

        scale_box = widgets.VBox(
            [widgets.HBox([scale_button, autoscale_button, continuous_update_button])]
        )
        scale_box.layout = widgets.Layout(width="300px")

        gui = widgets.VBox(
            [
                widgets.VBox(sliders),
                scale_box,
                # vmin_vmax_slider,
                # power_scale_button,
            ]
        )

        return widgets.HBox([gui, canvas])


class DiffractionSpotsVisualization(BaseMeasurementVisualization2D):
    """
    Display a diffraction pattern as indexed Bragg reflections.

    Parameters
    ----------
    measurements : IndexedDiffractionPattern
        Diffraction pattern to be displayed.
    scale : float
        Size of the circles representing the diffraction spots.
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axis.
    Returns
    -------
    figure, axis_handle : matplotlib.figure.Figure, matplotlib.axis.Axis
    """

    def __init__(
        self,
        measurements: IndexedDiffractionPatterns,
        ax: Axes,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        scale: float = 0.1,
        common_scale: bool = False,
        explode: bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
    ):
        measurements = measurements.sort(criterion="distance")

        super().__init__(
            measurements,
            ax=ax,
            cbar=cbar,
            common_scale=common_scale,
            explode=explode,
            figsize=figsize,
            interact=interact,
        )

        # positions = measurements.positions[:, :2]

        self._scale = scale

        # (
        #     np.sqrt(np.min(squareform(distance_matrix(positions, positions)))) * scale
        # )

        if cmap is None:
            cmap = config.get("visualize.cmap", "viridis")

        self._normalization = None
        self._scale_normalization = None
        self._annotation_threshold = 0.0
        self._cmap = cmap
        self._size_bars = []
        self._miller_index_annotations = None

        self.set_normalization(power=power, vmin=vmin, vmax=vmax)
        self.set_artists()

        if cbar:
            self.set_cbars()
            self.set_scale_units()
            self.set_cbar_labels()

        self.set_xunits()
        self.set_yunits()
        self.set_xlabels()
        self.set_ylabels()
        self.set_xlim()
        self.set_ylim()

    def _get_scales(self, indexed_diffraction_spots, norm):
        conversion = _get_conversion_factor(self._xunits, self._get_default_xunits())

        return (
            norm(indexed_diffraction_spots.intensities) ** 0.5
            * self._scale
            * 0.5
            * conversion
        )

    def _get_positions(self, indexed_diffraction_spots):
        positions = indexed_diffraction_spots.positions[:, :2].copy()
        positions[:, 0] *= _get_conversion_factor(
            self._xunits, self._get_default_xunits()
        )
        positions[:, 1] *= _get_conversion_factor(
            self._yunits, self._get_default_yunits()
        )
        return positions

    def _update_scales(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            artists = self._artists[i]
            norm = self._normalization[i]
            scales = self._get_scales(measurement, norm)
            artists._widths = scales
            artists._heights = scales
            artists.set()

    def _update_vmin_vmax(self, vmin: float = None, vmax: float = None):
        super()._update_vmin_vmax(vmin, vmax)
        self._update_scales()

    def _update_power(self, power: float = 1.0):
        super()._update_power(power)
        self._update_scales()

    def set_artists(self):
        if self._artists is not None:
            for artist in self.artists.ravel():
                artist.remove()

        self._artists = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]

            norm = self._normalization[i]

            scales = self._get_scales(measurement, norm)
            positions = self._get_positions(measurement)

            if self._cmap not in plt.colormaps():
                cmap = ListedColormap([self._cmap])
            else:
                cmap = self._cmap

            ellipse_collection = EllipseCollection(
                widths=scales,
                heights=scales,
                angles=0.0,
                units="xy",
                array=measurement.intensities,
                cmap=cmap,
                offsets=positions,
                transOffset=ax.transData,
            )

            ellipse_collection.set_norm(norm)

            ax.add_collection(ellipse_collection)

            self._artists[i] = ellipse_collection

            ax.axis("equal")

    @property
    def _reciprocal_space_axes(self):
        return [
            ReciprocalSpaceAxis(
                label="kx", sampling=1.0, units="1/Å", _tex_label="$k_x$"
            ),
            ReciprocalSpaceAxis(
                label="ky", sampling=1.0, units="1/Å", _tex_label="$k_y$"
            ),
        ]

    def _get_default_xlabel(self, units: str = None):
        return self._reciprocal_space_axes[-2].format_label(units)

    def _get_default_ylabel(self, units: str = None):
        return self._reciprocal_space_axes[-1].format_label(units)

    def _get_default_xunits(self):
        return self._reciprocal_space_axes[-1].units

    def _get_default_yunits(self):
        return self._reciprocal_space_axes[-1].units

    def set_xlim(self, xlim: tuple[float, float] = None):
        """
        Set the x-axis view limits.
        """

        if xlim is not None:
            common_xlim = True
        else:
            common_xlim = False

        for i, measurement in self._generate_measurements():
            if common_xlim is False:
                xlim = np.abs(measurement.positions[:, 0]).max() * 1.2
                xlim = (
                    _get_conversion_factor(self._xunits, self._get_default_xunits())
                    * xlim
                )
                xlim = [-xlim, xlim]

            if xlim is not None:
                self.axes[i].set_xlim(xlim)

    def set_ylim(self, ylim: tuple[float, float] = None):
        """
        Set the y-axis view limits.
        """
        if ylim is not None:
            common_ylim = True
        else:
            common_ylim = False

        for i, measurement in self._generate_measurements():
            if common_ylim is False:
                ylim = np.abs(measurement.positions[:, 1]).max() * 1.2
                ylim = (
                    _get_conversion_factor(self._xunits, self._get_default_xunits())
                    * ylim
                )
                ylim = [-ylim, ylim]

            if ylim is not None:
                self.axes[i].set_ylim(ylim)

    def set_xunits(self, units: str = None):
        super().set_xunits(units)
        self.set_artists()

    def set_yunits(self, units: str = None):
        super().set_yunits(units)
        self.set_artists()

        # for i, measurement in self.iterate_measurements():
        #     artist = self.artists[i]
        #     positions = measurement.positions[:, :2].copy()
        #     positions[:, 0] *= _get_conversion_factor(self._x_units, self._get_default_x_units())
        #     positions[:, 1] *= _get_conversion_factor(self._y_units, self._get_default_y_units())
        #     artist.set(offsets=positions)

    def update_artists(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            artists = self._artists[i]
            norm = self._normalization[i]

            scales = self._get_scales(measurement, norm)

            artists._widths = np.clip(scales, a_min=1e-3, a_max=1e3)
            artists._heights = np.clip(scales, a_min=1e-3, a_max=1e3)

            artists.set(array=measurement.intensities)
            self._set_hkl_visibility()

    def remove_miller_index_annotations(self):
        for annotation in self._miller_index_annotations:
            annotation.remove()
        self._miller_index_annotations = []

    def set_hkl_threshold(self, threshold):
        self._annotation_threshold = threshold
        self._set_hkl_visibility()

    def _set_hkl_visibility(self):
        if self._miller_index_annotations is None:
            self.set_miller_index_annotations()

        for i, measurement in self._generate_measurements(keepdims=False):
            visibility = measurement.intensities > self._annotation_threshold
            for annotation, visible in zip(self._miller_index_annotations, visibility):
                annotation.set_visible(visible)

    def set_miller_index_annotations(
        self,
        threshold: float = 1.0,
        size: int = 8,
        alignment: str = "top",
        **kwargs,
    ):
        self._annotation_threshold = threshold
        self._miller_index_annotations = []
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]
            norm = self._normalization[i]
            visibility = measurement.intensities > threshold
            positions = self._get_positions(measurement)
            scales = self._get_scales(measurement, norm)

            for hkl, position, visible, scale in zip(
                measurement.miller_indices, positions, visibility, scales
            ):
                if alignment == "top":
                    xy = position[:2] + [0, scale / 2]
                    va = "bottom"
                elif alignment == "center":
                    xy = position[:2]
                    va = "center"
                elif alignment == "bottom":
                    xy = position[:2] - [0, scale / 2]
                    va = "top"
                else:
                    raise ValueError()

                if config.get("visualize.use_tex"):
                    text = " \ ".join(
                        [f"\\bar{{{abs(i)}}}" if i < 0 else f"{i}" for i in hkl]
                    )
                    text = f"${text}$"
                else:
                    text = "{} {} {}".format(*hkl)

                annotation = ax.annotate(
                    text,
                    xy=xy,
                    ha="center",
                    va=va,
                    size=size,
                    visible=visible,
                    **kwargs,
                )
                self._miller_index_annotations.append(annotation)

    def pick_events(self):
        self._pick_annotations = {}
        for ax, artist in zip(np.array(self.axes).ravel(), self.artists.ravel()):
            artist.set_picker(True)
            annotation = ax.annotate(
                "",
                xy=(0, 0),
                xycoords="data",
                xytext=(20.0, 20.0),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
                visible=False,
            )
            self._pick_annotations[artist] = annotation

        def onpick(event):
            hkl = self.measurements.miller_indices[event.ind][0].tolist()
            position = self.measurements.positions[event.ind][0]
            intensity = event.artist.get_array()[event.ind].item()
            annotation = self._pick_annotations[event.artist]

            annotation.set_text(
                "\n".join(
                    (
                        f"hkl: {' '.join(map(str, hkl))}",
                        f"coordinate: {'{:.2f}, {:.2f}, {:.2f}'.format(*position.tolist())}",
                        f"intensity: {intensity:.4g}",
                    )
                )
            )
            annotation.xy = position[:2]
            annotation.set_visible(True)

        self.fig.canvas.mpl_connect("pick_event", onpick)

    @property
    def widgets(self):
        if widgets is None:
            raise ipywidgets_not_installed

        canvas = self.fig.canvas

        sliders = make_sliders_from_ensemble_axes(self, self.axes_types)

        def index_update_callback(change):
            if self._autoscale:
                vmin, vmax = self.get_global_vmin_vmax()
                self._update_vmin_vmax(vmin, vmax)

        _set_update_indices_callback(sliders, self, callbacks=(index_update_callback,))

        def hkl_slider_changed(change):
            self.set_hkl_threshold(change["new"])

        hkl_slider = widgets.FloatLogSlider(
            description="Index threshold", min=-10, max=0, value=1, step=1e-6
        )
        hkl_slider.observe(hkl_slider_changed, "value")

        power_scale_slider = _make_power_scale_slider(self)
        scale_button = _make_scale_button(self)
        autoscale_button = _make_autoscale_button(self)
        continuous_update_button = _make_continuous_button(sliders)

        scale_button.layout = widgets.Layout(width="20%")
        autoscale_button.layout = widgets.Layout(width="30%")
        continuous_update_button.layout = widgets.Layout(width="50%")

        scale_box = widgets.VBox(
            [widgets.HBox([scale_button, autoscale_button, continuous_update_button])]
        )
        scale_box.layout = widgets.Layout(width="300px")

        gui = widgets.VBox(
            [
                widgets.VBox(sliders),
                scale_box,
                # vmin_vmax_slider,
                power_scale_slider,
                hkl_slider,
            ]
        )

        return widgets.HBox([gui, canvas])


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
