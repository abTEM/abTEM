"""Module for plotting atoms, images, line scans, and diffraction patterns."""
import string
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import TYPE_CHECKING, List
from typing import Union, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection, EllipseCollection, CircleCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import Size, Divider
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from traitlets import link

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.axes import ReciprocalSpaceAxis, format_label, LinearAxis
from abtem.core.colors import hsluv_cmap
from abtem.core.units import _get_conversion_factor
from abtem.core.utils import label_to_index
from matplotlib.colors import ListedColormap

if TYPE_CHECKING:
    from abtem.measurements import (
        BaseMeasurement2D,
    )


def _make_default_sizes():
    sizes = {
        "cbar_padding_left": Size.Fixed(0.15),
        "cbar_spacing": Size.Fixed(0.4),
        "cbar_padding_right": Size.Fixed(0.7),
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
        caxes = defaultdict(list)
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
                    cb_ax = _cbaraxes_class_factory(Axes)(
                        fig, self._divider.get_position(), orientation="vertical"
                    )
                    fig.add_axes(cb_ax)
                    cb_ax.set_axes_locator(
                        self._divider.new_locator(nx=nx, ny=0, ny1=-1)
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
    def ncols(self):
        return self._axes.shape[0]

    @property
    def nrows(self):
        return self._axes.shape[1]

    def __getitem__(self, item):
        return self._axes[item]

    def __len__(self):
        return len(self._axes)

    @property
    def shape(self):
        return self._axes.shape

    def set_cbar_padding(self, padding: tuple = (0.1, 0.1)):
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

    def set_axes_padding(self, padding=(0.0, 0.0)):
        self._col_sizes["padding"].fixed_size = padding[0]
        self._row_sizes["padding"].fixed_size = padding[1]


def _format_options(options):
    return [
        f"{option:.3f}" if isinstance(option, float) else option for option in options
    ]


def _make_indexing_sliders(
    visualization, axes_types, continuous_update: bool = False, callbacks=()
):
    sliders = []
    for axes_metadata, n, axes_type in zip(
        visualization.measurements.ensemble_axes_metadata,
        visualization.measurements.ensemble_shape,
        axes_types,
    ):
        options = _format_options(axes_metadata.coordinates(n))

        with config.set({"visualize.use_tex": False}):
            label = axes_metadata.format_label()

        if axes_type == "range":
            sliders.append(
                widgets.SelectionRangeSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=(0, len(options) - 1),
                )
            )
        elif axes_type == "index":
            sliders.append(
                widgets.SelectionSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                )
            )

    return sliders


def _set_update_indices_callback(sliders, visualization, callbacks):
    def update_indices(change):
        indices = ()
        for slider in sliders:
            idx = slider.index
            if isinstance(idx, tuple):
                idx = range(*idx)
            indices += (idx,)

        with sliders[0].hold_trait_notifications():
            visualization.set_indices(indices)

    for slider in sliders:
        slider.observe(update_indices, "value")
        for callback in callbacks:
            slider.observe(callback, "value")


def _make_continuous_button(sliders):
    continuous_update_checkbox = widgets.ToggleButton(
        description="Continuous update", value=False
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

    vmin_vmax_slider = widgets.FloatRangeSlider(
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
    scale_button = widgets.Button(description="Scale")

    def scale_button_clicked(*args):
        vmin, vmax = visualization._get_global_vmin_vmax()
        visualization._update_vmin_vmax(vmin, vmax)

    scale_button.on_click(scale_button_clicked)

    return scale_button


def _make_autoscale_button(visualization):
    def autoscale_button_changed(change):
        if change["new"]:
            visualization._autoscale = True
        else:
            visualization._autoscale = False

    autoscale_button = widgets.ToggleButton(
        value=visualization._autoscale,
        description="Autoscale",
        tooltip="Autoscale",
    )
    autoscale_button.observe(autoscale_button_changed, "value")

    return autoscale_button


def _make_power_scale_slider(visualization):
    def powerscale_slider_changed(change):
        visualization._update_power(change["new"])

    power_scale_slider = widgets.FloatSlider(
        value=visualization._get_power(),
        min=0.01,
        max=2,
        step=0.01,
        description="Power",
        tooltip="Power",
    )
    power_scale_slider.observe(powerscale_slider_changed, "value")

    return power_scale_slider


def _get_joined_titles(measurement, formatting):
    titles = []
    for axes_metadata in measurement.ensemble_axes_metadata:
        titles.append(axes_metadata.format_title(formatting))
    return "\n".join(titles)


class MeasurementVisualization(metaclass=ABCMeta):
    def __init__(self, axes, measurements, axes_types=(), x_units=None, y_units=None):
        self._measurements = measurements
        self._axes = axes
        self._axes_types = axes_types
        self._indices = self._validate_indices()
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._metadata_labels = np.array([])
        self._x_units = x_units
        self._y_units = y_units

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

    def iterate_measurements(self, keep_dims: bool = True):
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
            yield axes_index, indexed_measurements.get_items(
                indices, keep_dims=keep_dims
            )

    def set_axes_padding(self, padding: Tuple[float, float] = (0.0, 0.0)):
        self._axes.set_axes_padding(padding)

    def _get_indexed_measurements(self, keepdims: bool = True):

        indexed = self.measurements.get_items(self._indices, keep_dims=keepdims)

        indexed = indexed.sum(
            axes=tuple(
                i
                for i, axes_type in enumerate(self._axes_types)
                if axes_type == "range"
            )
        )
        return indexed

    def set_column_titles(
        self,
        titles: Union[str, List[str]] = None,
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

            axes_metadata = indexed_measurements.ensemble_axes_metadata[0]
            titles = []
            for i, axis_metadata in enumerate(axes_metadata):
                titles.append(
                    axis_metadata.format_title(
                        format, units=units, include_label=i == 0
                    )
                )
                if i == indexed_measurements.ensemble_shape[0]:
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

    @abstractmethod
    def _get_default_x_label(self, units=None):
        pass

    @abstractmethod
    def _get_default_y_label(self, units=None):
        pass

    def set_x_labels(self, label: str = None):
        if label is None:
            label = self._get_default_x_label(units=self._x_units)

        for ax in np.array(self.axes).ravel():
            ax.set_xlabel(label)

    def set_y_labels(self, label: str = None):
        if label is None:
            label = self._get_default_y_label(units=self._y_units)

        for i, ax in enumerate(np.array(self.axes).ravel()):
            # print(self.axes.shape[0], i, i % self.axes.shape[0])
            # if i % self.axes.shape[0] == 1:

            ax.set_ylabel(label)

    @abstractmethod
    def set_xlim(self):
        pass

    @abstractmethod
    def set_ylim(self):
        pass

    @abstractmethod
    def _get_default_x_units(self):
        pass

    @abstractmethod
    def _get_default_y_units(self):
        pass

    def set_x_units(self, units=None):
        if units is None:
            self._x_units = self._get_default_x_units()
        else:
            self._x_units = units

        self.set_x_labels()
        self.set_xlim()

    def set_y_units(self, units=None):
        if units is None:
            self._y_units = self._get_default_y_units()
        else:
            self._y_units = units

        self.set_y_labels()
        self.set_ylim()

    def set_row_titles(
        self,
        titles: Union[str, List[str]] = None,
        pad: float = 0.0,
        format: str = ".3g",
        units: str = None,
        fontsize=12,
        **kwargs,
    ):

        indexed_measurements = self._get_indexed_measurements()

        if titles is None:
            if not len(indexed_measurements.ensemble_shape) > 1:
                return

            axes_metadata = indexed_measurements.ensemble_axes_metadata[1]
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
                xytext=(-ax.yaxis.labelpad - pad, 0),
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

    @property
    def ncols(self):
        return self._axes.shape[0]

    @property
    def nrows(self):
        return self._axes.shape[1]

    @property
    def axes_types(self):
        return self._axes_types

    def _get_axes_from_axes_types(self, axes_type):
        return tuple(
            i
            for i, checked_axes_type in enumerate(self.axes_types)
            if checked_axes_type == axes_type
        )

    @property
    def overlay_axes(self):
        return self._get_axes_from_axes_types("overlay")

    @property
    def index_axes(self):
        return self._get_axes_from_axes_types("index")

    @property
    def range_axes(self):
        return self._get_axes_from_axes_types("range")

    @property
    def explode_axes(self):
        return self._get_axes_from_axes_types("explode")

    @property
    def indices(self):
        return self._indices

    @property
    def measurements(self):
        return self._measurements

    @property
    def axes(self):
        return self._axes

    def _validate_indices(self, indices: tuple = ()):
        num_ensemble_dims = len(self.measurements.ensemble_shape)
        num_indexing_axes = (
            num_ensemble_dims - len(self.explode_axes) - len(self.overlay_axes)
        )

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
                raise RuntimeError()

        return tuple(validated_indices)

    def set_indices(self, indices=()):

        self._indices = self._validate_indices(indices)
        self.update_artists()
        self.update_panel_labels()

    @abstractmethod
    def update_artists(self):
        pass

    def set_panel_labels(
        self,
        labels: str = "alphabetic",
        frameon: bool = True,
        loc: str = "upper left",
        pad: float = 0.1,
        borderpad: float = 0.1,
        prop: dict = None,
        formatting: str = ".3g",
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
            for i, measurement in self.iterate_measurements(keep_dims=True):
                labels.append(_get_joined_titles(measurement, formatting))
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
            self._metadata_labels, self.iterate_measurements(keep_dims=True)
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
            self.set_indices((i,))
            if adjust_scale:
                self._update_vmin_vmax(*self._validate_vmin_vmax())

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
        self, measurements, axes, axes_types: tuple = None, common_scale: bool = True
    ):

        measurements = measurements.compute().to_cpu()

        super().__init__(axes, measurements, axes_types=axes_types)

        self._x_units = None
        self._y_units = None
        self._scale_units = None
        self._x_label = None
        self._y_label = None
        self._column_titles = []
        self._row_titles = []
        self._artists = None
        self._autoscale = None
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
            vmin, vmax = self._get_global_vmin_vmax(vmin=vmin, vmax=vmax)

        self._normalization = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self.iterate_measurements(keep_dims=False):

            if power == 1.0:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)

            if measurement.is_complex:
                measurement = measurement.abs()

            norm.autoscale_None(measurement.array)

            self._normalization[i] = norm

    def _get_global_vmin_vmax(self, vmin=None, vmax=None):
        measurements = self._get_indexed_measurements()

        if measurements.is_complex:
            measurements = measurements.abs()

        if vmin is None:
            vmin = float(measurements.array.min())

        if vmax is None:
            vmax = float(measurements.array.max())

        return vmin, vmax

    def _validate_vmin_vmax(self, vmin: float = None, vmax: float = None):
        if self._common_scale:
            global_vmin, global_vmax = self._get_global_vmin_vmax()

        return vmin, vmax

    def _update_vmin_vmax(self, vmin=None, vmax=None):
        for norm, measurement in zip(
            self._normalization.ravel(), self.iterate_measurements(keep_dims=False)
        ):
            norm.vmin = vmin
            norm.vmax = vmax

    def add_area_indicator(self, area_indicator, panel="first", **kwargs):

        xlim = self.axes[0, 0].get_xlim()
        ylim = self.axes[0, 0].get_ylim()

        for i, ax in enumerate(np.array(self.axes).ravel()):
            if panel == "first" and i == 0:
                area_indicator.add_to_axes(ax, **kwargs)
            elif panel == "all":
                area_indicator.add_to_axes(ax, **kwargs)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def _update_power(self, power: float = 1.0):
        for i, measurement in self.iterate_measurements(keep_dims=False):
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

    @property
    def artists(self):
        return self._artists

    @property
    @abstractmethod
    def _artists_per_axes(self):
        pass

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

    def set_cbar_padding(self, padding: Tuple[float, float] = (0.1, 0.1)):
        self._axes.set_cbar_padding(padding)

    def set_cbar_size(self, fraction: float):
        self._axes.set_cbar_size(fraction)

    def set_cbar_spacing(self, spacing: float):
        self._axes.set_cbar_spacing(spacing)

    def set_cbars(self, **kwargs):
        cbars = defaultdict(list)

        for i, _ in self.iterate_measurements():
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
                        cbars[ax].append(plt.colorbar(image, **kwargs))
                else:
                    cbars[ax].append(plt.colorbar(images, **kwargs))

        self._cbars = cbars

    def set_sizebars(
        self,
        axes: Tuple[int, ...] = ((-1, 0),),
        label="",
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

        conversion = _get_conversion_factor(
            self._x_units, self.measurements.axes_metadata[-2].units
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
            label = f"{size:>{formatting}} {self._x_units}"

        for size_bar in self._size_bars:
            size_bar.remove()

        self._size_bars = []
        for ax in axes:
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


class MeasurementVisualization2D(BaseMeasurementVisualization2D):
    """
    Show the image(s) using matplotlib.

    Parameters
    ----------
    measurements
    axes : matplotlib.axes.Axes, optional
        If given the plots are added to the axis. This is not available for image grids.
    figsize : two int, optional
        The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
    title : bool or str, optional
        Add a title to the figure. If True is given instead of a string the title will be given by the value
        corresponding to the "name" key of the metadata dictionary, if this item exists.
    cmap : str, optional
        Matplotlib colormap name used to map scalar data to colors. Ignored if image array is complex.
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

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    def __init__(
        self,
        measurements: "BaseMeasurement2D",
        axes,
        axes_types: tuple = None,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_scale: bool = False,
        units: str = None,
        convert_complex: str = "domain_coloring",
        autoscale: bool = False,
        title=None,
    ):

        super().__init__(
            measurements, axes, axes_types=axes_types, common_scale=common_scale
        )

        self._normalization = None
        self._cmap = cmap
        self._convert_complex = convert_complex
        self._autoscale = autoscale

        self.set_normalization(power=power, vmin=vmin, vmax=vmax)
        self.set_artists()

        if cbar:
            self.set_cbars()
            self.set_scale_units()
            self.set_cbar_labels()

        self.set_extent()
        self.set_x_units(units)
        self.set_y_units(units)
        self.set_x_labels()
        self.set_y_labels()

        if title is not None:
            self.set_column_titles(title)

    @property
    def _artists_per_axes(self):
        if self._convert_complex == "domain_coloring" and self.measurements.is_complex:
            return 2
        else:
            return 1

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

    def _get_default_x_label(self, units: str = None):
        return self.measurements.axes_metadata[-2].format_label(units=units)

    def _get_default_y_label(self, units: str = None):
        return self.measurements.axes_metadata[-1].format_label(units=units)

    def _get_default_x_units(self):
        return self.measurements.axes_metadata[-2].units

    def _get_default_y_units(self):
        return self.measurements.axes_metadata[-1].units

    def set_xlim(self):
        self.set_extent()

    def set_ylim(self):
        self.set_extent()

    def set_extent(self, extent=None):

        if extent is None:
            x_extent = self.measurements._plot_extent_x(self._x_units)
            y_extent = self.measurements._plot_extent_y(self._y_units)
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
        images = np.zeros(self.axes.shape + (self._artists_per_axes,), dtype=object)

        for i, measurement in self.iterate_measurements(keep_dims=False):
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

    def update_artists(self):

        for i, measurement in self.iterate_measurements(keep_dims=False):
            images = self._artists[i]

            if self._domain_coloring:
                array = measurement.array
                abs_array = np.abs(array)
                alpha = self._normalization[i](abs_array)
                alpha = np.clip(alpha, a_min=0, a_max=1)

                images[0].set_alpha(alpha)
                images[0].set_data(np.angle(array))
                images[1].set_data(abs_array)
            else:

                images.set_data(measurement.array.T)

    @property
    def widgets(self):
        canvas = self.fig.canvas
        # vmin_vmax_slider = make_vmin_vmax_slider(self)

        def index_update_callback(change):
            if self._autoscale:
                vmin, vmax = self._get_global_vmin_vmax()
                self._update_vmin_vmax(vmin, vmax)
                # vmin_vmax_slider.value = self._get_global_vmin_vmax()

        sliders = _make_indexing_sliders(
            self, self.axes_types, callbacks=(index_update_callback,)
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
        measurements: "BaseMeasurement2D",
        axes,
        axes_types: tuple = None,
        units=None,
        common_scale: bool = True,
        legend=False,
    ):

        measurements = measurements.compute().to_cpu()

        super().__init__(axes, measurements, axes_types=axes_types)

        self._x_units = None
        self._y_units = None
        self._x_label = None
        self._y_label = None
        self._column_titles = []
        self._lines = np.array([[]])
        self._common_scale = common_scale
        self.set_artists()
        self.set_x_units(units=units)
        self.set_y_units()

        if "overlay" in axes_types and legend:
            self.set_legends()

        if self.ncols > 1:
            self.set_column_titles()

        # self.set_y_units()
        # self.set_x_labels()
        # self.set_y_labels()
        #
        # if any(axes_type == "explode" for axes_type in axes_types):
        #     self.set_column_titles()
        #     self.set_row_titles()
        #
        # if any(axes_type == "overlay" for axes_type in axes_types):
        #     self.set_legends()
        #
        # for i, _ in self.iterate_measurements():
        #     #     #self.axes[i].yaxis.set_label_coords(0.5, -0.02)
        #     #     #cbar2.formatter.set_powerlimits((0, 0))
        #     #     #self.axes[i].get_yaxis().formatter.set_useMathText(True)
        #     #     #self.axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
        #     self.axes[i].get_yaxis().get_offset_text().set_horizontalalignment("right")
        # # #
        #     #self.axes[i].yaxis.set_offset_position("right")
        #     #self.axes[i].yaxis.set_offset_position("left")
        #     #self.axes[i].set_ylabel(format_label(self._y_label, self._y_units))

    @property
    def artists(self):
        return self._artists

    def _get_default_x_label(self, units: str = None):
        return self.measurements.axes_metadata[-1].format_label(units)

    def _get_default_y_label(self, units: str = None):
        axes = LinearAxis(label=self.measurements.metadata.get("label", ""))
        return format_label(axes, units)

    def _get_default_x_units(self):
        return self.measurements.axes_metadata[-1].units

    def _get_default_y_units(self):
        return self.measurements.metadata.get("units", "")

    def set_xlim(self):
        extent = self.measurements._plot_extent(self._x_units)

        margin = (extent[1] - extent[0]) * 0.05
        for i, measurement in self.iterate_measurements():
            self.axes[i].set_xlim([-extent[0] - margin, extent[1] + margin])
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

        for i, measurement in self.iterate_measurements():

            if not self._common_scale and common_ylim is None:
                y_lim = _get_extent(measurement)
            else:
                y_lim = common_ylim
            self.axes[i].set_ylim(y_lim)

    def set_legends(self, **kwargs):
        for i, _ in self.iterate_measurements():
            self.axes[i].legend(**kwargs)

    def set_artists(self):

        artists = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self.iterate_measurements(keep_dims=False):
            ax = self.axes[i]
            x = self._get_xdata()
            new_lines = []
            for _, line_profile in measurement.iterate_ensemble(keep_dims=True):

                labels = []
                for axis in line_profile.ensemble_axes_metadata:
                    labels += [axis.format_title(".3f")]

                label = "-".join(labels)

                new_lines.append(
                    ax.plot(
                        x,
                        line_profile.array[(0,) * (len(line_profile.shape) - 1)],
                        label=label,
                    )[0]
                )
            artists.itemset(i, new_lines)

        self._artists = artists

    def _get_xdata(self):
        extent = self.measurements._plot_extent(self._x_units)
        # conversion = _get_conversion_factor(self._x_units, self._get_default_x_units())
        return np.linspace(
            extent[0],
            extent[1],
            self.measurements.shape[-1],
            endpoint=False,
        )

    def update_artists(self):
        pass


class DiffractionSpotsVisualization(BaseMeasurementVisualization2D):
    """
    Display a diffraction pattern as indexed Bragg reflections.

    Parameters
    ----------
    indexed_diffraction_pattern : IndexedDiffractionPattern
        Diffraction pattern to be displayed.
    spot_scale : float
        Size of the circles representing the diffraction spots.
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axis.
    figsize : two int, optional
        The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
    title : bool or str, optional
        Add a title to the figure. If True is given instead of a string the title will be given by the value
        corresponding to the "name" key of the metadata dictionary, if this item exists
    overlay_indices : bool

    annotate_kwargs : dict
        Additional keyword arguments passed to `matplotlib.axes.Axes.annotate` to change the formatting of the labels.
    inequivalency_threshold : float
        Relative intensity difference to determine whether two symmetry-equivalent diffraction spots should be indepdendently
        labeled (e.g. due to a unit cell with a basis of more than one element).
    Returns
    -------
    figure, axis_handle : matplotlib.figure.Figure, matplotlib.axis.Axis
    """

    def __init__(
        self,
        measurements,
        axes,
        axes_types: tuple = None,
        cbar: bool = False,
        cmap: str = None,
        phase_cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        scale: float = 1.0,
        common_color_scale: bool = False,
        convert_complex: str = "domain_coloring",
        autoscale: bool = False,
    ):

        measurements = measurements.order()

        positions = measurements.positions[:, :2]

        self._scale = (
            np.sqrt(squareform(distance_matrix(positions, positions)).min()) * scale
        )

        self._normalization = None
        self._scale_normalization = None
        self._autoscale = autoscale
        self._annotation_threshold = 0.0

        super().__init__(
            measurements,
            axes,
            axes_types=axes_types,
        )

        if cmap is None:
            cmap = config.get("visualize.cmap", "viridis")

        if phase_cmap is None:
            phase_cmap = config.get("phase_cmap", "hsluv")

        self._cmap = cmap
        self._phase_cmap = phase_cmap
        self._size_bars = []
        self._common_color_scale = common_color_scale
        self._convert_complex = convert_complex
        self._autoscale = autoscale
        self._miller_index_annotations = None

        self.set_normalization(power=power, vmin=vmin, vmax=vmax)
        self.set_artists()

        if cbar:
            self.set_cbars()
            self.set_scale_units()
            self.set_cbar_labels()

        # # self.set_extent()
        self.set_x_units()
        self.set_y_units()
        self.set_x_labels()
        self.set_y_labels()

    @property
    def _artists_per_axes(self):
        return 1

    def _get_scales(self, indexed_diffraction_spots, norm):
        conversion = _get_conversion_factor(self._x_units, self._get_default_x_units())
        return (
            norm(indexed_diffraction_spots.intensities) ** 0.5
            * self._scale
            * 0.5
            * conversion
        )

    def _get_positions(self, indexed_diffraction_spots):
        positions = indexed_diffraction_spots.positions[:, :2].copy()
        positions[:, 0] *= _get_conversion_factor(
            self._x_units, self._get_default_x_units()
        )
        positions[:, 1] *= _get_conversion_factor(
            self._y_units, self._get_default_y_units()
        )
        return positions

    def _update_scales(self):
        for i, measurement in self.iterate_measurements(keep_dims=False):
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
        for i, measurement in self.iterate_measurements(keep_dims=False):
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

    def _get_default_x_label(self, units: str = None):
        return self._reciprocal_space_axes[-2].format_label(units)

    def _get_default_y_label(self, units: str = None):
        return self._reciprocal_space_axes[-1].format_label(units)

    def _get_default_x_units(self):
        return self._reciprocal_space_axes[-1].units

    _get_default_y_units = _get_default_x_units

    def set_xlim(self):
        for i, measurement in self.iterate_measurements():
            x_lim = np.abs(measurement.positions[:, 0]).max() * 1.1
            x_lim = (
                _get_conversion_factor(self._x_units, self._get_default_x_units())
                * x_lim
            )
            self.axes[i].set_xlim([-x_lim, x_lim])

    def set_ylim(self):
        for i, measurement in self.iterate_measurements():
            x_lim = np.abs(measurement.positions[:, 0]).max() * 1.1
            x_lim = (
                _get_conversion_factor(self._x_units, self._get_default_x_units())
                * x_lim
            )
            self.axes[i].set_xlim([-x_lim, x_lim])

    def set_x_units(self, units=None):
        super().set_x_units(units)
        self.set_artists()

    def set_y_units(self, units=None):
        super().set_y_units(units)
        self.set_artists()

        # for i, measurement in self.iterate_measurements():
        #     artist = self.artists[i]
        #     positions = measurement.positions[:, :2].copy()
        #     positions[:, 0] *= _get_conversion_factor(self._x_units, self._get_default_x_units())
        #     positions[:, 1] *= _get_conversion_factor(self._y_units, self._get_default_y_units())
        #     artist.set(offsets=positions)

    def update_artists(self):
        for i, measurement in self.iterate_measurements(keep_dims=False):
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

        for i, measurement in self.iterate_measurements(keep_dims=False):
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
        for i, measurement in self.iterate_measurements(keep_dims=False):
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
        canvas = self.fig.canvas

        sliders = _make_indexing_sliders(
            self, self.axes_types  # , callbacks=(index_update_callback,)
        )

        def index_update_callback(change):
            # for slider in sliders:

            if self._autoscale:
                vmin, vmax = self._get_global_vmin_vmax()
                self._update_vmin_vmax(vmin, vmax)

            # if self._autoscale:
            #    vmin_vmax_slider.value = self._get_global_vmin_vmax()

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
    toggle_hkl_button = widgets.ToggleButton(description="Toggle hkl", value=False)

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
    plane: Union[Tuple[float, float], str] = "xy",
    ax=None,
    scale: float = 0.75,
    title: str = None,
    numbering: bool = False,
    show_periodic: bool = False,
    figsize: Tuple[float, float] = None,
    legend: bool = False,
    merge: float = 1e-2,
    tight_limits=False,
    show_cell=None,
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
