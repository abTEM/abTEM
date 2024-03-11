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
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.axes import ScaleAxis, LinearAxis, NonLinearAxis
from abtem.core.backend import get_array_module
from abtem.core.colors import hsluv_cmap
from abtem.core.utils import label_to_index, flatten_list_of_lists
from abtem.visualize.artists import (
    ScatterArtist,
    ImageArtist,
    DomainColoringArtist,
    _get_value_limits,
    LinesArtist,
    ScaleBar,
)
from abtem.visualize.axes_grid import (
    _find_axes_types,
    _validate_axes,
)
from abtem.visualize.data import VisualizationData

# from abtem.visualize.widgets import (
#     make_sliders_from_ensemble_axes,
#     make_scale_button,
#     make_autoscale_button,
#     make_power_scale_slider,
#     make_cmap_dropdown,
#     make_complex_visualization_dropdown,
# )

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


class BaseVisualization:
    def __init__(
        self,
        data: VisualizationData,
        ax: Axes = None,
        figsize: tuple[int, int] = None,
        aspect: bool = False,
        common_scale: bool = False,
        ncbars: int = 0,
        interact: bool = False,
        share_x: bool = False,
        share_y: bool = False,
        column_titles: list[str] = None,
    ):
        self._data = data
        self._autoscale = config.get("visualize.autoscale", False)

        self._axes = _validate_axes(
            shape=data.axes_shape,
            ax=ax,
            ioff=interact,
            aspect=aspect,
            ncbars=ncbars,
            common_color_scale=common_scale,
            figsize=figsize,
            sharex=share_x,
            sharey=share_y,
        )
        self._indices = ()

        self._complex_conversion = "none"
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._artists = None

        self.get_figure().canvas.header_visible = False

        if column_titles:
            self.set_column_titles(column_titles)

        # if hasattr(self.axes, "nrows") and (self.axes.nrows > 1):
        #     self.set_row_titles()

    def get_figure(self):
        return self.axes[0, 0].get_figure()

    @property
    def autoscale(self):
        return self._autoscale

    @autoscale.setter
    def autoscale(self, autoscale: bool):
        self._autoscale = autoscale
        self.set_value_limits()

    @property
    def data(self):
        return self._data

    @property
    def artists(self):
        return self._artists

    @property
    def axes(self):
        return self._axes

    def set_xlabel(self, label: str = None):
        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_xlabel(label)

    def set_ylabel(self, label: str = None):
        for i, j in np.ndindex(self.axes.shape):
            if i == 0:
                self.axes[i, j].set_ylabel(label)

    def set_ylim(self, ylim: tuple[float, float] | list[float] = None):
        for i in np.ndindex(self.axes.shape):
            self.axes[i].set_ylim(ylim)

    def set_xlim(self, xlim: tuple[float, float] | list[float] = None):
        for axes in np.array(self.axes).ravel():
            axes.set_xlim(xlim)

    def adjust_coordinate_limits_to_artists(self):
        xlim = [np.inf, -np.inf]
        ylim = [np.inf, -np.inf]
        for artist in self.artists.ravel():
            new_xlim = artist.get_xlim()
            xlim = [min(new_xlim[0], xlim[0]), max(new_xlim[1], xlim[1])]
            new_ylim = artist.get_ylim()
            ylim = [min(new_ylim[0], ylim[0]), max(new_ylim[1], ylim[1])]

        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def set_value_limits(self, values_limits: tuple[float, float] | list[float] = None):
        if values_limits is None:
            values_limits = [None, None]

        for artist in np.array(self.artists).ravel():
            artist.set_value_limits(values_limits)

    def set_power(self, power: float = 1.0):
        for artist in self.artists.ravel():
            artist.set_power(power)

    def set_common_value_limits(self):
        value_limits = [np.inf, -np.inf]
        for artist in self.artists.ravel():
            new_value_limits = artist.get_value_limits()
            value_limits = [
                min(new_value_limits[0], value_limits[0]),
                max(new_value_limits[1], value_limits[1]),
            ]

        self.set_value_limits(value_limits)

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
            titles = _get_axes_titles(
                self.ensemble_axes_metadata, self.axes_types, self.axes.shape
            )
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

    def axis(self, mode: str = "all", ticks: bool = True, spines: bool = True):
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

            if not spines:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

    def remove_artists(self):
        for ax in np.array(self.axes).ravel():
            for child in ax.get_children():
                if isinstance(child, AxesImage):
                    child.remove()

    def update_data_indices(self, indices):
        self._indices = indices
        for i in np.ndindex(self.axes.shape):
            data = self._data.get_data_for_indices(indices, i)
            self._artists[i].set_data(data)

    @abstractmethod
    def set_artists(self, value_limits: list[float], power: float = 1):
        pass

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


class Visualization1D(BaseVisualization):
    def __init__(
        self,
        data,
        ax: Axes = None,
        xlabel: str = None,
        ylabel: str = None,
        common_scale: bool = True,
        figsize: tuple[float, float] = None,
        interact: bool = False,
        title: bool | str = True,
        legend: bool = True,
        **kwargs,
    ):
        super().__init__(
            data=data,
            ax=ax,
            aspect=False,
            interact=interact,
            figsize=figsize,
            share_x=True,
            share_y=common_scale,
            title=title,
        )

        self.set_artists(**kwargs)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)

        if data.overlay and legend:
            self.set_legends()

        if not common_scale:
            self.axes.set_sizes(padding=0.5)

        for ax in np.array(self.axes).ravel():
            ax.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
            ax.get_yaxis().get_offset_text().set_horizontalalignment("center")
            ax.yaxis.set_offset_position("left")

    def set_artists(self, **kwargs):
        artists = np.zeros(self.axes.shape, dtype=object)

        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            data = self.data.get_data_for_indices(self._indices, i)

            artist = LinesArtist(ax, data, **kwargs)
            artists.itemset(i, artist)

        self._artists = artists

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


def validate_cmap(cmap, data):
    if cmap is None:
        if data.is_complex and data.complex_conversion in ("none", "phase"):
            cmap = config.get("visualize.phase_cmap", "hsluv")
        else:
            cmap = config.get("visualize.cmap", "viridis")

    if cmap == "hsluv":
        cmap = hsluv_cmap

    return cmap


class Visualization2D(BaseVisualization):
    def __init__(
        self,
        data: VisualizationData,
        ax: Axes = None,
        xlabel: str = None,
        ylabel: str = None,
        cbar_label: str = None,
        common_scale: bool = False,
        cmap: str = None,
        power: float = 1.0,
        logscale: bool = False,
        vmin: float = None,
        vmax: float = None,
        cbar: bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
        row_titles: list[str] = None,
        column_titles: list[str] = None,
        **kwargs,
    ):
        self._artists = None
        self._scale_bars = None

        artist = self._get_artist_type(data)

        if cbar:
            ncbars = artist.num_cbars
        else:
            ncbars = 0

        super().__init__(
            data=data,
            ax=ax,
            common_scale=common_scale,
            ncbars=ncbars,
            interact=interact,
            figsize=figsize,
            aspect=True,
            share_x=True,
            share_y=True,
            # row_titles=row_titles,
            # column_titles=column_titles,
        )

        cmap = validate_cmap(cmap, data)

        self.set_artists(
            artist_type=artist,
            vmin=vmin,
            vmax=vmax,
            power=power,
            logscale=logscale,
            cmap=cmap,
            **kwargs,
        )

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)

        self.adjust_coordinate_limits_to_artists()

        if common_scale:
            self.set_common_value_limits()

        if cbar:
            self.set_cbars(label=cbar_label)

    def _get_artist_type(self, data, complex_conversion=None):
        if data.is_complex and complex_conversion in ("domain_coloring", "none", None):
            return DomainColoringArtist
        else:
            return ImageArtist

    def set_artists(
        self,
        artist_type,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        cmap: str = None,
        **kwargs,
    ):
        self.remove_artists()

        artists = np.zeros(self.axes.shape, dtype=object)
        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]
            data = self._data.get_data_for_indices(self._indices, i)

            artist = artist_type(
                ax=ax,
                data=data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                power=power,
                logscale=logscale,
                **kwargs,
            )

            artists.itemset(i, artist)

        self._artists = artists

    def set_cbars(self, **kwargs):
        for i in np.ndindex(self.axes.shape):
            artist = self._artists[i]

            if hasattr(self.axes, "_caxes"):
                caxes = self.axes._caxes[i]
            else:
                caxes = None

            artist.set_cbars(caxes=caxes, **kwargs)

    def _change_artist_type(self):
        artist = self.data.get_artist()

        if self.data.complex_out:
            self.axes.set_cbar_layout(ncbars=2)
        else:
            self.axes.set_cbar_layout(ncbars=1)

        self.remove_artists()

        for i in np.ndindex(self.axes.shape):
            print(self._indices)
            data = self._data.get_data_for_indices(self._indices, i)
            self.artists[i] = self.artists[i]._change_artist_type(
                artist, self.axes[i], data
            )

    def set_complex_conversion(self, complex_conversion: str):
        self.data.complex_conversion = complex_conversion
        self._change_artist_type()

    def set_cmap(self, cmap):
        cmap = validate_cmap(cmap, self.data)

        for artist in self.artists.ravel():
            artist.set_cmap(cmap)

    def set_extent(self, extent: list[float] = None):
        for artist in self._artists.ravel():
            artist.set_extent(extent)

    def set_scale_bars(self, panel_locs: str = "lower right", **kwargs):
        if isinstance(panel_locs, str):
            panel_locs = self.axes._axis_location_to_indices(panel_locs)

        self._scale_bars = []
        for panel_loc in panel_locs:
            axes = self.axes[panel_loc]
            self._scale_bars.append(ScaleBar(ax=axes, **kwargs))

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
