"""Module for plotting atoms, images, line scans, and diffraction patterns."""
import string
from abc import abstractmethod
from typing import TYPE_CHECKING, List
from typing import Union, Tuple

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid, SubplotDivider
from mpl_toolkits.axes_grid1 import Size, Divider
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.colors import hsluv_cmap
from abtem.core.units import _get_conversion_factor, _validate_units, _format_units
from abtem.core.utils import label_to_index

if TYPE_CHECKING:
    from abtem.measurements import (
        BaseMeasurement,
        BaseMeasurement2D,
        IndexedDiffractionSpots,
    )


def _iterate_axes(axes: Union[ImageGrid, Axes]):
    try:
        for ax in axes:
            yield ax
    except TypeError:
        yield axes


def format_options(options):
    return [
        f"{option:.3f}" if isinstance(option, float) else option for option in options
    ]


def make_indexing_sliders(
    visualization,
    continuous_update: bool = False,
    range_axes=(),
):

    sliders = []
    for i in range(len(visualization.measurements.ensemble_shape)):
        axes_metadata = visualization.measurements.ensemble_axes_metadata[i]
        options = format_options(
            axes_metadata.coordinates(visualization.measurements.ensemble_shape[i])
        )

        with config.set({"visualize.use_tex": False}):
            label = axes_metadata.format_label()

        if i in range_axes:
            sliders.append(
                widgets.SelectionRangeSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=(0, len(options) - 1),
                )
            )
        else:
            sliders.append(
                widgets.SelectionSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                )
            )
    return sliders


def rescale_vmin_vmax_slider(slider, visualization):
    vmin = np.inf
    vmax = -np.inf
    for artist in visualization.artists.ravel():
        vmin = min(vmin, artist.norm.vmin)
        vmax = max(vmax, artist.norm.vmax)
    vmin = min(vmin, 0.0)

    slider.value = [vmin, vmax]
    slider.min = vmin
    slider.max = vmax
    slider.step = (vmax - vmin) / 100.


def make_vmin_vmax_slider(visualization):
    def vmin_vmax_slider_changed(change):
        vmin, vmax = change["owner"].value

        for artist in visualization.artists.ravel():
            artist.vmin = vmin
            artist.vmax = vmax

    vmin_vmax_slider = widgets.FloatRangeSlider(
        value=[0, 1],
        min=0,
        max=1,
        step=0.1,
        disabled=visualization._autoscale,
        description="Normalization",
        continuous_update=True,
        readout_format=".2e",
    )

    rescale_vmin_vmax_slider(vmin_vmax_slider, visualization)

    vmin_vmax_slider.observe(vmin_vmax_slider_changed, "value")
    return vmin_vmax_slider


def make_scale_button(visualization, vmin_vmax_slider=None):
    scale_button = widgets.Button(description="Scale once")

    def scale_button_clicked(*args):
        visualization._update_vmin()
        visualization._update_vmax()

        visualization.update_artists()

        if vmin_vmax_slider is not None:
            rescale_vmin_vmax_slider(vmin_vmax_slider, visualization)

    scale_button.on_click(scale_button_clicked)

    return scale_button


def make_autoscale_button(visualization, vmin_vmax_slider=None):
    def autoscale_button_changed(change):
        visualization.set_autoscale(change["new"])

        if vmin_vmax_slider is not None:
            rescale_vmin_vmax_slider(vmin_vmax_slider, visualization)

    autoscale_button = widgets.ToggleButton(
        value=visualization._autoscale,
        description="Autoscale",
        tooltip="Autoscale",
    )
    autoscale_button.observe(autoscale_button_changed, "value")

    return autoscale_button


class MeasurementVisualization:
    def __init__(self, axes, measurements, axes_types=()):
        self._measurements = measurements
        self._axes = axes
        self._axes_types = axes_types
        self._indices = self._validate_indices()

    @property
    def fig(self):
        return self._axes[0, 0].get_figure()

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

    def _iterate_index(self):
        shape = self._get_indexed_measurements().ensemble_shape

        if len(shape) == 0:
            yield (0, 0)

        else:
            if len(shape) == 1:
                if self.ncols == 1:
                    shape = (1,) + shape
                elif self.nrows == 1:
                    shape = shape + (1,)
                else:
                    raise RuntimeError()
            elif len(shape) != 2:
                raise RuntimeError()

            for i in np.ndindex(*shape):
                yield i

    def set_axes_padding(self, padding: Tuple[float, float] = (0.0, 0.0)):
        self._axes.set_axes_padding(padding)

    def _get_indexed_measurements(self):
        return self.measurements.get_items(self._indices, keep_dims=True)

    def set_column_titles(
        self,
        titles: Union[str, List[str]] = None,
        pad: float = 10.0,
        format: str = ".3g",
        units: str = None,
        fontsize=12,
        **kwargs,
    ):
        indexed_measurements = self._get_indexed_measurements()

        if titles is None:
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

        elif isinstance(titles, str):
            titles = [titles] * max(len(indexed_measurements.ensemble_shape), 1)

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
        titles: Union[str, List[str]] = None,
        pad: float = 0.0,
        format: str = ".2g",
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

    def _validate_indices(self, indices=()):
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
            else:
                validated_indices.append(0)

        # print(validated_indices)

        # validated_indices = validated_indices + [0] * (
        #     num_ensemble_dims - len(validated_indices)
        # )
        return tuple(validated_indices)

    def set_indices(self, indices=()):
        self._indices = self._validate_indices(indices)
        self.update_artists()

    @abstractmethod
    def update_artists(self):
        pass

    def add_panel_labels(self, labels: str = None, **kwargs):
        if "loc" not in kwargs:
            kwargs["loc"] = 2

        if isinstance(self.axes, Axes):
            axes = [self.axes]
        else:
            axes = self.axes

        if labels is None:
            labels = string.ascii_lowercase
            labels = [f"({label})" for label in labels]
            if config.get("visualize.use_tex", False):
                labels = [f"${label}$" for label in labels]

        for ax, l in zip(_iterate_axes(axes), labels):
            at = AnchoredText(l, pad=0.0, borderpad=0.5, frameon=False, **kwargs)
            ax.add_artist(at)
            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=5)])


def format_label(label, units: str = None, italic: bool = False) -> str:
    if config.get("visualize.use_tex", False):
        if not italic:
            label = f"\mathrm{{{label}}}"

        return f"${label} \ [{_format_units(units)}]$"
    else:
        return f"{label} [{units}]"


def split_axes(axes, pad=0.2):
    axes.set_visible(False)
    locator = axes.get_axes_locator()

    spacing = Size.Fixed(pad)

    h = [
        Size.AxesX(axes),
        Size.Fixed(pad),
        Size.AxesX(axes),
    ]

    v = [Size.AxesY(axes)]

    fig = axes.get_figure()
    rect = (0.1, 0.1, 0.8, 0.8)
    divider = Divider(fig, rect, h, v)
    divider.set_locator(locator)
    locator1 = divider.new_locator(nx=0, nx1=1, ny=0, ny1=1)
    locator2 = divider.new_locator(nx=2, nx1=3, ny=0, ny1=1)

    ax1 = fig.add_axes(rect)
    ax2 = fig.add_axes(rect)
    ax1.set_axes_locator(locator1)
    ax2.set_axes_locator(locator2)
    return ax1, ax2, spacing


def set_cbar_axes(axes, n, sizes):
    fig = axes.get_figure()

    divider = AxesDivider(axes)
    locator = divider.new_locator(nx=0, ny=0)
    axes.set_axes_locator(locator)

    divider._horizontal += cbar_layout(n, sizes)
    rect = (0.1, 0.1, 0.8, 0.8)

    caxes = []
    for i in range(n):
        locator = divider.new_locator(nx=(i + 1) * 2, nx1=(i + 1) * 2 + 1, ny=0, ny1=1)
        cax = fig.add_axes(rect)
        cax.set_axes_locator(locator)
        caxes.append(cax)

    axes.cax = caxes


def make_default_sizes():
    sizes = {
        "cbar_padding_left": Size.Fixed(0.1),
        # "cbar": Size.Fixed(0.001),
        "cbar_spacing": Size.Fixed(0.5),
        "cbar_padding_right": Size.Fixed(0.7),
        "padding": Size.Fixed(0.1),
    }
    return sizes


def cbar_layout(n, sizes):
    if n == 0:
        return []

    layout = [sizes["cbar_padding_left"]]
    for i in range(n):
        # layout.extend([size])
        layout.extend([sizes["cbar"]])

        if i < n - 1:
            layout.extend([sizes["cbar_spacing"]])

    layout.extend([sizes["cbar_padding_right"]])
    return layout


def make_col_layout(axes, n_cbars, sizes, cbar_mode="each", direction="x"):
    sizes_layout = []

    for i, ax in enumerate(axes):
        if direction == "x":
            sizes_layout.append(Size.AxesX(ax, aspect="axes", ref_ax=axes[0]))
        else:
            sizes_layout.append(Size.AxesY(ax, aspect="axes", ref_ax=axes[0]))

        if not "cbar" in sizes:
            sizes["cbar"] = Size.from_any("5%", sizes_layout[0])

        if cbar_mode == "each":
            sizes_layout.extend(cbar_layout(n_cbars, sizes))

        if i < len(axes) - 1:
            sizes_layout.append(sizes["padding"])

    if cbar_mode == "single":
        # size = Size.from_any("5%", sizes_layout[0])
        sizes_layout.extend(cbar_layout(n_cbars, sizes))

    return sizes_layout


class AxesGrid:
    def __init__(
        self,
        fig,
        ncols,
        nrows,
        cbars: int = 0,
        cbar_mode: str = "single",
        aspect: bool = True,
        sharex: bool = True,
        sharey: bool = True,
    ):
        from mpl_toolkits.axes_grid1.mpl_axes import Axes

        rect = (0.1, 0.1, 0.8, 0.8)
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

        self._col_sizes = make_default_sizes()
        self._row_sizes = make_default_sizes()

        col_layout = make_col_layout(
            cols, cbars, self._col_sizes, cbar_mode=cbar_mode, direction="x"
        )
        row_layout = make_col_layout(rows, 0, self._row_sizes, direction="y")

        divider = SubplotDivider(
            fig, 111, horizontal=col_layout, vertical=row_layout, aspect=aspect
        )

        i = 0
        caxes = []
        for nx, col_size in enumerate(col_layout):
            add_cb_axes = True
            for ny, row_size in enumerate(row_layout):

                if isinstance(col_size, Size.AxesX) and (
                    isinstance(row_size, Size.AxesY)
                ):
                    ax = axes[i]
                    ax.set_axes_locator(divider.new_locator(nx=nx, ny=ny))
                    i += 1

                if (
                    (col_size is self._col_sizes["cbar"])
                    # col_size is col_layout[-2]
                    and (isinstance(row_size, Size.AxesY))
                    and add_cb_axes
                ):

                    cb_ax = _cbaraxes_class_factory(Axes)(
                        fig, divider.get_position(), orientation="vertical"
                    )

                    fig.add_axes(cb_ax)

                    caxes.append(cb_ax)
                    if cbar_mode == "each":
                        cb_ax.set_axes_locator(divider.new_locator(nx=nx, ny=ny))
                    else:
                        cb_ax.set_axes_locator(divider.new_locator(nx=-3, ny=0, ny1=-1))
                        add_cb_axes = False

        if cbar_mode == "single":
            caxes = caxes * len(axes)

        new_caxes = [[] for _ in range(nrows * ncols)]
        for i in range(nrows * ncols * cbars):
            col = i // (cbars * nrows)
            row = i % nrows
            j = np.ravel_multi_index((col, row), (ncols, nrows))
            new_caxes[j].append(caxes[i])

        for ax, cax in zip(axes, new_caxes):
            ax.cax = cax

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

    @property
    def ncols(self):
        return self._axes.shape[0]

    @property
    def nrows(self):
        return self._axes.shape[1]

    @classmethod
    def from_measurements(
        cls,
        fig,
        measurements,
        axes_types,
        cbars=0,
        cbar_mode="single",
        aspect=True,
        sharex: bool = True,
        sharey: bool = True,
    ):

        shape = measurements.ensemble_shape
        assert len(shape) == len(axes_types)
        shape = tuple(
            n
            for n, axes_type in zip(shape, axes_types)
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

        return cls(
            fig,
            ncols,
            nrows,
            cbars,
            cbar_mode,
            aspect=aspect,
            sharex=sharex,
            sharey=sharey,
        )

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


class BaseMeasurementVisualization2D(MeasurementVisualization):
    def __init__(
        self,
        measurements,
        axes,
        axes_types: tuple = None,
    ):

        super().__init__(axes, measurements, axes_types=axes_types)

        self._x_units = None
        self._y_units = None
        self._scale_units = None
        self._x_label = None
        self._y_label = None
        self._column_titles = []
        self._row_titles = []
        self._artists = None

        if self.ncols > 1:
            self.set_column_titles()

        if self.nrows > 1:
            self.set_row_titles()

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
            units = self.measurements.metadata.get("units", "undefined")

        self._scale_units = units

    def set_cbar_labels(self, label: str = None, **kwargs):
        if label is None:
            label = self.measurements.metadata.get("label", "undefined")

        label = format_label(label, self._scale_units)

        for cbar in self._cbars.ravel():
            cbar.set_label(label, **kwargs)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)
            cbar.ax.yaxis.set_offset_position("left")

    def set_cbar_padding(self, padding: Tuple[float, float] = (0.1, 0.1)):
        self._axes.set_cbar_padding(padding)

    def set_cbar_size(self, fraction: float):
        self._axes.set_cbar_size(fraction)

    def set_cbar_spacing(self, spacing: float):
        self._axes.set_cbar_spacing(spacing)

    def set_cbars(self, **kwargs):
        cbars = np.zeros(self.axes.shape + (self._artists_per_axes,), dtype=object)
        for i, _ in self.iterate_measurements():
            ax = self.axes[i]
            images = self._artists[i]

            if isinstance(images, np.ndarray):
                for j, image in enumerate(images):
                    cbars[i + (j,)] = plt.colorbar(image, cax=ax.cax[j], **kwargs)
            else:
                cbars[i] = plt.colorbar(images, cax=ax.cax[0], **kwargs)

        if cbars.shape[-1] == 1:
            cbars = np.squeeze(cbars, -1)

        self._cbars = cbars

    def axis_off(self, spines=True):
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
    def __init__(
        self,
        measurements: "BaseMeasurement2D",
        axes,
        axes_types: tuple = None,
        cbar: bool = False,
        cmap: str = None,
        phase_cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_color_scale: bool = False,
        units: str = None,
        convert_complex: str = "domain_coloring",
        autoscale: bool = True,
    ):

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

        self.set_artists()
        self.set_normalization(vmin=vmin, vmax=vmax, power=power)

        # 1
        # if cbar:
        #     self.set_cbars()
        #     self.set_scale_units()
        #     self.set_cbar_labels()
        #
        # self.set_extent()
        # self.set_x_units(units)
        # self.set_y_units(units)
        # self.set_x_labels()
        # self.set_y_labels()

    @property
    def _artists_per_axes(self):
        if self._convert_complex == "domain_coloring" and self.measurements.is_complex:
            return 2
        else:
            return 1

    @property
    def _domain_coloring(self):
        return self.measurements.is_complex

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
                **kwargs,
            )
            ax.add_artist(anchored_size_bar)
            self._size_bars.append(anchored_size_bar)

    def _set_domain_coloring_cbar_labels(self, **kwargs):
        for cbar1, cbar2 in self._cbars.ravel():
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

    def set_normalization(
        self,
        power: float = None,
        vmin: float = None,
        vmax: float = None,
    ):

        if self._common_color_scale:
            measurements = self._get_indexed_measurements().abs()
            vmin = float(measurements.array.min())
            vmax = float(measurements.array.max())

        for i, measurement in self.iterate_measurements(keep_dims=False):
            artists = self._artists[i]

            norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)
            norm.autoscale_None(measurement.array)

            # if self._domain_coloring:
            #    images[1].norm = norm1
            # else:
            artists.set_norm(norm)

    def set_x_labels(self, label=None):
        if label is None:
            self._x_label = self.measurements.axes_metadata[-2].label
        else:
            self._x_label = label

        for ax in np.array(self.axes).ravel():
            ax.set_xlabel(format_label(self._x_label, self._x_units))

    def set_y_labels(self, label=None):
        if label is None:
            self._y_label = self.measurements.axes_metadata[-1].label
        else:
            self._y_label = label

        for ax in np.array(self.axes).ravel():
            ax.set_ylabel(format_label(self._y_label, self._y_units))

    def set_x_units(self, units=None):
        if units is None:
            self._x_units = self.measurements.base_axes_metadata[1].units
        else:
            self._x_units = units

        self.set_x_labels()
        self.set_extent()

    def set_y_units(self, units=None):
        if units is None:
            self._y_units = self.measurements.base_axes_metadata[1].units
        else:
            self._y_units = units

        self.set_y_labels()
        self.set_extent()

    def set_extent(self, extent=None):

        if extent is None:
            extent = self.measurements._plot_extent_x(
                self._x_units
            ) + self.measurements._plot_extent_y(self._y_units)

        for image in self._artists.ravel():
            image.set_extent(extent)

    def _add_domain_coloring_imshow(self, ax, array):
        abs_array = np.abs(array)
        alpha = (abs_array - abs_array.min()) / abs_array.ptp()
        cmap = hsluv_cmap if self._phase_cmap == "hsluv" else self._phase_cmap

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

            if self._domain_coloring:
                images[i] = self._add_domain_coloring_imshow(ax, measurement.array)
            else:
                images[i] = self._add_real_imshow(ax, measurement.array)

        if images.shape[-1] == 1:
            images = np.squeeze(images, -1)

        self._artists = images

    def update_artists(self):
        for i, measurement in self.iterate_measurements(keep_dims=False):
            images = self._artists[i]

            if self._domain_coloring:
                array = measurement.array
                abs_array = np.abs(array)
                alpha = (abs_array - abs_array.min()) / abs_array.ptp()
                images[0].set_alpha(alpha)
                images[0].set_data(np.angle(array))
                images[1].set_data(abs_array)
            else:
                images.set_data(measurement.array.T)

        # if self._autoscale:
        #     self.set_normalization()
        #


class MeasurementVisualization1D(MeasurementVisualization):
    def __init__(
        self,
        measurements: "BaseMeasurement2D",
        axes,
        axes_types: tuple = None,
        units=None,
    ):

        super().__init__(axes, measurements, axes_types=axes_types)

        self._x_units = None
        self._y_units = None
        self._x_label = None
        self._y_label = None
        self._lines = np.array([[]])
        self.set_plots()
        self.set_x_units(units=units)
        self.set_y_units()
        self.set_x_labels()
        self.set_y_labels()

        if any(axes_type == "explode" for axes_type in axes_types):
            self.set_column_titles()
            self.set_row_titles()

        if any(axes_type == "overlay" for axes_type in axes_types):
            self.set_legends()

        for i, _ in self.iterate_measurements():
            #     #self.axes[i].yaxis.set_label_coords(0.5, -0.02)
            #     #cbar2.formatter.set_powerlimits((0, 0))
            #     #self.axes[i].get_yaxis().formatter.set_useMathText(True)
            #     #self.axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
            self.axes[i].get_yaxis().get_offset_text().set_horizontalalignment("right")
        # #
        #     #self.axes[i].yaxis.set_offset_position("right")
        #     #self.axes[i].yaxis.set_offset_position("left")
        #     #self.axes[i].set_ylabel(format_label(self._y_label, self._y_units))

    def set_legends(self):
        for i, _ in self.iterate_measurements():
            self.axes[i].legend()

    def set_plots(self):
        # indexed_measurements = self._get_indexed_measurements()

        # for line in self._lines:
        #     line.remove()

        lines = np.zeros(self.axes.shape, dtype=object)
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
            lines.itemset(i, new_lines)

        self._lines = lines

    def _get_xdata(self):
        extent = self.measurements._plot_extent(self._x_units)
        return np.linspace(
            extent[0], extent[1], self.measurements.shape[-1], endpoint=False
        )

    def set_x_units(self, units=None):
        if units is None:
            units = self.measurements.axes_metadata[-1].units
        self._x_units = units
        self.set_x_labels()

        # for i, _ in self.iterate_measurements():

        for lines in self._lines.ravel():
            for line in lines:
                line.set_xdata(self._get_xdata())

        for ax in np.array(self.axes).ravel():
            ax.relim()
            ax.autoscale()

    def set_y_units(self, units=None):
        self._y_units = _validate_units(
            units, self.measurements.metadata.get("units", None)
        )
        self.set_y_labels()

    def set_y_labels(self, label=None):
        if label is None:
            self._y_label = self.measurements.metadata.get("label", None)
        else:
            self._y_label = label

        if self._y_label is None:
            return

        for ax in np.array(self.axes).ravel():
            ax.set_ylabel(format_label(self._y_label, self._y_units))

    def set_x_labels(self, label=None):
        if label is None:
            self._x_label = self.measurements.base_axes_metadata[-1].label
        else:
            self._x_label = label

        for i, _ in self.iterate_measurements():
            self.axes[i].set_xlabel(format_label(self._x_label, self._x_units))

    def update_artists(self):
        pass


def show_measurements_2d(
    measurements: "BaseMeasurement",
    axes: Axes = None,
    figsize: Tuple[int, int] = None,
    title: str = None,
    power: float = 1.0,
    vmin: float = None,
    vmax: float = None,
    common_color_scale: bool = False,
    cbar: bool = False,
    cmap: str = None,
):
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
    measurements = measurements.compute().to_cpu()

    # if cbar and measurements.is_complex and measurements.ensemble_shape:
    #     raise NotImplementedError(
    #         "colorbar not implemented for exploded plot with domain coloring"
    #     )

    complex_representation = "domain_coloring"

    if measurements.is_complex and complex_representation != "domain_coloring":
        measurements = getattr(measurements, complex_representation)()

    if not measurements.ensemble_shape:
        if axes is None:
            fig, axes = plt.subplots(figsize=figsize)
        else:
            fig = axes.get_figure()
    else:
        if axes is None:
            fig = plt.figure(1, figsize, clear=True)
            image_grid_kwargs = {}
            image_grid_kwargs["share_all"] = True
            if common_color_scale:
                if cbar:
                    image_grid_kwargs["cbar_mode"] = "single"

                image_grid_kwargs["axes_pad"] = [0.1, 0.1]
            else:
                if cbar:
                    image_grid_kwargs["cbar_mode"] = "each"
                    image_grid_kwargs["cbar_pad"] = 0.05
                    image_grid_kwargs["axes_pad"] = [0.8, 0.3]
                else:
                    image_grid_kwargs["axes_pad"] = [0.1, 0.1]

            axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), **image_grid_kwargs)
        else:
            fig = axes.get_figure()

    measurements = measurements[(0,) * max(len(measurements.ensemble_shape) - 2, 0)]

    visualization = MeasurementVisualization2D(
        axes, measurements, vmin=vmin, vmax=vmax, common_color_scale=common_color_scale
    )

    if title:
        visualization.set_column_titles()
        visualization.set_row_titles()

    if cbar:
        visualization.set_cbars()

    return visualization


def _show_indexed_diffraction_pattern(
    indexed_diffraction_pattern,
    scale: float = 1.0,
    ax: Axes = None,
    figsize: Tuple[float, float] = (6, 6),
    title: str = None,
    overlay_hkl: bool = True,
    power: float = 1.0,
    cmap: str = "viridis",
    colors: str = "cmap",
    background_color: str = "white",
):
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
    indexed_diffraction_pattern = indexed_diffraction_pattern.block_direct()

    positions = indexed_diffraction_pattern.positions[:, :2]

    intensities = indexed_diffraction_pattern.intensities**power

    order = np.argsort(-np.linalg.norm(positions, axis=1))

    positions = positions[order]
    intensities = intensities[order]

    scales = intensities / intensities.max()

    min_distance = squareform(distance_matrix(positions, positions)).min()

    scale_factor = min_distance / scales.max() * scale

    scales = scales**power * scale_factor

    if colors == "cmap":
        norm = matplotlib.colors.Normalize(vmin=0, vmax=intensities.max())
        cmap = matplotlib.cm.get_cmap(cmap)
        colors = cmap(norm(intensities))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if title:
        ax.set_title(title)

    ax.add_collection(
        EllipseCollection(
            widths=scales,
            heights=scales,
            angles=0.0,
            units="xy",
            facecolors=colors,
            offsets=positions,
            transOffset=ax.transData,
        )
    )

    x_lim = np.abs(positions[:, 0]).max() * 1.1
    y_lim = np.abs(positions[:, 1]).max() * 1.1

    ax.axis("equal")
    ax.set_xlim(-x_lim * 1.1, x_lim * 1.1)
    ax.set_ylim(-y_lim * 1.1, y_lim * 1.1)
    ax.set_xlabel("kx [1/Å]")
    ax.set_ylabel("ky [1/Å]")

    # fig.patch.set_facecolor(background_color)
    # ax.axis("off")

    if overlay_hkl:
        add_miller_index_annotations(ax, indexed_diffraction_pattern)

    return fig, ax


class DiffractionSpotsVisualization(BaseMeasurementVisualization2D):
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
        units: str = None,
        convert_complex: str = "domain_coloring",
        autoscale: bool = True,
    ):

        self._scale = scale

        positions = measurements.positions[:, :2]
        self._scale_factor = np.sqrt(
            squareform(distance_matrix(positions, positions)).min()
        )

        self._normalization = None
        self._scale_normalization = None
        self._autoscale = autoscale

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
        self._miller_index_annotations = []

        self._set_scale_factor_normalization()
        self._set_normalization(vmin=vmin, vmax=vmax, power=power)
        self.set_artists()


        #
        # if cbar:
        #     self.set_cbars()
        #     self.set_scale_units()
        #     self.set_cbar_labels()

        # self.set_extent()
        # self.set_x_units(units)
        # self.set_y_units(units)
        # self.set_x_labels()
        # self.set_y_labels()

    # def update_vmin(self, vmin=None):
    #     self._update_scale_vmin(vmin=vmin)
    #     self.update_artists()
    #     self._update_vmin(vmin=vmin)
    #
    def _update_scale_vmin(self, vmin=None):
        for i, measurement in self.iterate_measurements():
            norm = self._scale_normalization[i]
            vmin = measurement.min() if vmin is None else vmin
            norm.vmin = vmin ** 0.5

    def _update_scale_vmax(self, vmax=None):
        for i, measurement in self.iterate_measurements():
            norm = self._scale_normalization[i]
            vmax = measurement.max() if vmax is None else vmax
            norm.vmax = vmax ** 0.5

    def _update_vmin(self, vmin=None):
        for i, measurement in self.iterate_measurements():
            norm = self._normalization[i]
            vmin = measurement.min() if vmin is None else vmin
            norm.vmin = vmin

    def _update_vmax(self, vmax=None):
        for i, measurement in self.iterate_measurements():
            norm = self._normalization[i]
            vmax = measurement.max() if vmax is None else vmax
            norm.vmax = vmax

    def update_vmin_vmax(self, vmin=None, vmax=None):
        self._update_vmin(vmin)
        self._update_vmax(vmax)
        self._update_scale_vmin(vmin)
        self._update_scale_vmax(vmax)
        self.update_artists()

    # def _update_vmax(self, vmax=None):
    #     for i, measurement in self.iterate_measurements():
    #         norm = self._normalization[i]
    #         scale_norm = self._scale_normalization[i]
    #
    #         if vmax is None:
    #             validated_vmax = measurement.max()
    #         else:
    #             validated_vmax = vmax
    #
    #         norm.vmax = validated_vmax
    #         scale_norm.vmax = validated_vmax ** 0.5

    def _set_scale_factor_normalization(
        self,
        power: float = None,
        vmin: float = None,
        vmax: float = None,
    ):

        if self._common_color_scale:
            measurements = self._get_indexed_measurements().abs()
            vmin = float(measurements.array.min()) ** 0.5
            vmax = float(measurements.array.max()) ** 0.5

        if power is None:
            power = 1.0

        self._scale_normalization = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self.iterate_measurements(keep_dims=False):

            vmin = vmin**0.5 if vmin is not None else vmin
            vmax = vmax**0.5 if vmax is not None else vmax

            norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)
            norm.autoscale_None(measurement.array**0.5)

            self._scale_normalization[i] = norm

    def _set_normalization(
        self,
        power: float = None,
        vmin: float = None,
        vmax: float = None,
    ):

        if self._common_color_scale:
            measurements = self._get_indexed_measurements().abs()
            vmin = float(measurements.array.min())
            vmax = float(measurements.array.max())

        self._normalization = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self.iterate_measurements(keep_dims=False):

            norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)
            norm.autoscale_None(measurement.array)

            # if self._domain_coloring:
            #    images[1].norm = norm1
            # else:
            self._normalization[i] = norm


    def _get_plot_data(
        self, indexed_diffraction_spots: "IndexedDiffractionSpots", norm
    ):

        positions = indexed_diffraction_spots.positions[:, :2]

        intensities = indexed_diffraction_spots.intensities

        order = np.argsort(-np.linalg.norm(positions, axis=1))

        positions = positions[order]
        intensities = intensities[order]

        sqrt_intensities = intensities**0.5

        scales = norm(sqrt_intensities) * self._scale_factor
        return intensities, positions, scales

    def set_artists(self):

        self._artists = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self.iterate_measurements(keep_dims=False):
            ax = self.axes[i]

            scale_norm = self._scale_normalization[i]
            norm = self._normalization[i]

            intensities, positions, scales = self._get_plot_data(
                measurement, scale_norm
            )

            ellipse_collection = EllipseCollection(
                widths=scales,
                heights=scales,
                angles=0.0,
                units="xy",
                # facecolors=colors,
                array=intensities,
                cmap=self._cmap,
                offsets=positions,
                transOffset=ax.transData,
            )

            ellipse_collection.norm = norm

            ellipse_collection = ax.add_collection(ellipse_collection)

            self._artists[i] = ellipse_collection

            x_lim = np.abs(positions[:, 0]).max() * 1.1
            y_lim = np.abs(positions[:, 1]).max() * 1.1

            ax.axis("equal")
            ax.set_xlim(-x_lim * 1.1, x_lim * 1.1)
            ax.set_ylim(-y_lim * 1.1, y_lim * 1.1)
            ax.set_xlabel("kx [1/Å]")
            ax.set_ylabel("ky [1/Å]")

    def update_artists(self):
        # if self._autoscale:
        #     self._set_scale_factor_normalization()

        for i, measurement in self.iterate_measurements(keep_dims=False):
            artists = self._artists[i]
            norm = self._scale_normalization[i]

            intensities, positions, scales = self._get_plot_data(measurement, norm)

            artists._widths = scales * 0.5
            artists._heights = scales * 0.5

            artists.set(array=intensities, offsets=positions)

        # if self._autoscale:
        #     self._set_normalization()

    def remove_miller_index_annotations(self):
        for annotation in self._miller_index_annotations:
            annotation.remove()

    def add_miller_index_annotations(self):

        self._miller_index_annotations = []
        for i, measurement in self.iterate_measurements():
            ax = self.axes[i]
            for hkl, position in zip(
                measurement.miller_indices,
                measurement.positions,
            ):
                annotation = ax.annotate(
                    "{} {} {}".format(*hkl),
                    xy=position[:2],
                    ha="center",
                    va="center",
                    size=8,
                )
                annotation.set_path_effects([withStroke(foreground="w", linewidth=3)])

                self._miller_index_annotations.append(annotation)

    def set_autoscale(self, autoscale: bool):
        self._autoscale = autoscale
        if autoscale is True:
            self.set_normalization()

    def set_normalization(self, vmin=None, vmax=None, power=None):
        self._set_scale_factor_normalization(vmin=vmin, vmax=vmax, power=power)
        self._set_normalization(vmin=vmin, vmax=vmax)
        self.update_artists()

    def interact(self, continuous_update: bool = False):
        canvas = self.fig.canvas

        def update(change):
            idx = change["owner"].index
            self.set_indices((idx,))

        sliders = make_indexing_sliders(self, continuous_update=continuous_update)

        for slider in sliders:
            slider.observe(update, "value")

        sliders = widgets.VBox(sliders)

        app = widgets.HBox([sliders, canvas])

        return app




def get_annotations(ax):
    return [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.text.Annotation)
    ]


def remove_annotations(ax):
    annotations = get_annotations(ax)
    for annotation in annotations:
        annotation.remove()


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
