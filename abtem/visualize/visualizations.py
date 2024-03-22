"""Module for plotting atoms, images, line scans, and diffraction patterns."""

from __future__ import annotations


import itertools
from typing import TYPE_CHECKING


import matplotlib
import matplotlib as mpl
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np


from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.utils import label_to_index
from abtem.visualize.artists import (
    ImageArtist,
    DomainColoringArtist,
    LinesArtist,
    ScatterArtist,
    validate_cmap,
    _get_value_limits
)
from abtem.visualize.axes_grid import (
    AxesGrid,
)
from abtem.visualize.widgets import slider_from_axes_metadata

if TYPE_CHECKING:
    from abtem.measurements import BaseMeasurements


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


def _make_cax(ax, use_gridspec=True, **kwargs):
    if ax is None:
        raise ValueError(
            "Unable to determine Axes to steal space for Colorbar. "
            "Either provide the *cax* argument to use as the Axes for "
            "the Colorbar, provide the *ax* argument to steal space "
            "from it, or add *mappable* to an Axes."
        )
    fig = (  # Figure of first axes; logic copied from make_axes.
        [*ax.flat] if isinstance(ax, np.ndarray) else [*ax] if np.iterable(ax) else [ax]
    )[0].figure
    current_ax = fig.gca()
    if (
        fig.get_layout_engine() is not None
        and not fig.get_layout_engine().colorbar_gridspec
    ):
        use_gridspec = False
    if (
        use_gridspec
        and isinstance(ax, mpl.axes._base._AxesBase)
        and ax.get_subplotspec()
    ):
        cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
    else:
        cax, kwargs = cbar.make_axes(ax, **kwargs)
    # make_axes calls add_{axes,subplot} which changes gca; undo that.
    fig.sca(current_ax)
    cax.grid(visible=False, which="both", axis="both")
    return cax


def _validate_axes_types(overlay, explode, ensemble_dims):
    if explode is True:
        explode = tuple(range(ensemble_dims))
    elif explode is False:
        explode = ()

    if overlay is True:
        overlay = tuple(range(ensemble_dims))
    elif overlay is False:
        overlay = ()

    if len(overlay + explode) > ensemble_dims:
        raise ValueError

    if len(set(explode) & set(overlay)) > 0:
        raise ValueError("An axis cannot be both exploded and overlaid.")

    return overlay, explode


def convert_complex(measurement: BaseMeasurements, method: str) -> BaseMeasurements:
    if not measurement.is_complex:
        return measurement

    if method in ("domain_coloring", "none", None):
        return measurement

    if method in ("phase", "angle"):
        measurement = measurement.phase()
    elif method in ("amplitude", "abs"):
        measurement = measurement.abs()
    elif method in ("intensity", "abs2"):
        measurement = measurement.intensity()
    elif method in ("real",):
        measurement = measurement.real()
    elif method in ("imaginary", "imag"):
        measurement = measurement.imag()
    else:
        raise ValueError(f"complex conversion '{method}" f"' not implemented")

    return measurement


def _get_artist_type(measurement, complex_conversion):
    if len(measurement.base_shape) == 2:
        if measurement.is_complex and (
            complex_conversion in ("domain_coloring", "none", None)
        ):
            return DomainColoringArtist
        else:
            return ImageArtist
    elif hasattr(measurement, "positions"):
        return ScatterArtist
    elif len(measurement.base_shape) == 1:
        return LinesArtist


class Visualization:
    def __init__(
        self,
        measurement,
        ax: Axes = None,
        figsize: tuple[int, int] = None,
        aspect: bool = False,
        common_scale: bool = False,
        value_limits: tuple[float, float] = (None, None),
        overlay: bool | tuple[int, ...] = False,
        explode: bool | tuple[int, ...] = False,
        share_x: bool = False,
        share_y: bool = False,
        cbar: bool = False,
        interactive: bool = True,
        title: str = None,
        **kwargs,
    ):
        self._measurement = measurement

        overlay, explode = _validate_axes_types(
            overlay, explode, len(measurement.ensemble_shape)
        )
        
        self._overlay = overlay
        self._explode = explode

        if ax is None and not interactive:
            with plt.ioff():
                fig = plt.figure(figsize=figsize)

        elif ax is None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = ax.get_figure()

        artist_type = _get_artist_type(measurement, complex_conversion="none")

        if cbar:
            ncbars = artist_type.num_cbars
        else:
            ncbars = 0

        if ax is None:
            axes_shape = tuple(
                n
                for i, n in enumerate(measurement.ensemble_shape)
                if not i in self.indexing_axes and i not in overlay
            )
            shape = axes_shape + (0,) * (2 - len(axes_shape))
            ncols, nrows = (max(shape[0], 1), max(shape[1], 1))

            if common_scale:
                cbar_mode = "single"
            else:
                cbar_mode = "each"

            axes = AxesGrid(
                fig=fig,
                ncols=ncols,
                nrows=nrows,
                ncbars=ncbars,
                cbar_mode=cbar_mode,
                aspect=aspect,
                sharex=share_x,
                sharey=share_y,
            )

        else:
            axes = np.array([[ax]])

        self._axes = axes

        self._indices = ()

        self._complex_conversion = "none"
        self._autoscale = config.get("visualize.autoscale", False)
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._artists = None

        self.get_figure().canvas.header_visible = False

        if isinstance(title, str):
            self.set_column_titles(title)

        elif title and len(explode) > 0:
            axes_metadata = measurement.axes_metadata[explode[0]].to_ordinal_axis(measurement.shape[explode[0]])
            
            column_titles = [
                l.format_title(".3g", include_label=i==0) for i, l in enumerate(axes_metadata)
            ]

            self.set_column_titles(column_titles)

        if title and len(explode) > 1:
            row_titles = [
                l.format_title(".3g", include_label=i==0) for i, l in enumerate(measurement.axes_metadata[explode[1]])
            ]

            self.set_row_titles(row_titles)

        self._make_new_artists(**kwargs)
        
        self.adjust_coordinate_limits_to_artists()

        if common_scale:
            self.set_common_value_limits(value_limits)
        else:
            self.set_value_limits(value_limits)

        if artist_type is DomainColoringArtist:
            self.axes.set_sizes(cbar_spacing=0.5)

    def interact(self, gui_type, display):
        if not "ipympl" in matplotlib.get_backend():
            raise RuntimeError(
                f"interactive visualizations requires the 'ipympl' matplotlib backend"
            )

        sliders = [
            slider_from_axes_metadata(
                self.measurement.axes_metadata[i], self.measurement.shape[i]
            )
            for i in self.indexing_axes
        ]

        gui = gui_type(sliders, self.axes.fig.canvas)
        gui.attach_visualization(self)

        if display:
            from IPython.display import display as ipython_display

            ipython_display(gui)

        return gui

    @property
    def autoscale(self):
        return self._autoscale

    @property
    def indexing_axes(self):
        ensemble_axes = set(range(len(self.measurement.ensemble_shape)))
        return tuple(ensemble_axes - set(self._overlay) - set(self._explode))

    @autoscale.setter
    def autoscale(self, autoscale: bool):
        self._autoscale = autoscale
        self.set_value_limits()

    def _reduce_measurement(
        self, indices: tuple[int | tuple[int, int], ...], axis_indices
    ) -> BaseMeasurements:
        assert len(indices) <= len(self.indexing_axes)
        assert len(axis_indices) == 2

        validated_indices = ()
        summed_axes = ()
        removed_axes = 0
        j = 0
        k = 0
        for i in range(len(self._measurement.ensemble_shape)):
            if i in self.indexing_axes:
                if j >= len(indices):
                    validated_indices += (0,)
                elif isinstance(indices[j], int):
                    validated_indices += (indices[j],)
                    removed_axes += 1
                elif isinstance(indices[j], tuple):
                    validated_indices += (slice(*indices[j]),)
                    summed_axes += (i - removed_axes,)
                j += 1
            elif i in self._explode:
                validated_indices += (axis_indices[k],)
                k += 1
                removed_axes += 1
            elif i not in self._overlay:
                validated_indices += (0,)

        measurement = self._measurement[validated_indices]
        if len(summed_axes) > 0:
            measurement = measurement.sum(axis=summed_axes)

        measurement = convert_complex(measurement, self._complex_conversion)

        return measurement

    @property
    def measurement(self) -> BaseMeasurements:
        return self._measurement

    @property
    def artists(self):
        return self._artists

    @property
    def axes(self):
        return self._axes

    def get_figure(self):
        return self.axes[0, 0].get_figure()
    
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

    def set_xlabel(self, label: str = None):
        self.set_artists("xlabel", label=label)

    def set_ylabel(self, label: str = None):
        self.set_artists("ylabel", label=label)

    def set_xlim(self, xlim: tuple[float, float] | list[float] = None):
        self.set_artists("xlim", xlim=xlim)

    def set_ylim(self, ylim: tuple[float, float] | list[float] = None):
        self.set_artists("ylim", ylim=ylim)

    def set_value_limits(self, value_limits: tuple[float, float] | list[float] = (None, None)):
        self.set_artists("value_limits", value_limits=value_limits)

    def set_power(self, power: float = 1.0):
        self.set_artists("power", power=power)

    def set_common_value_limits(self, value_limits=(None, None)):
        value_limits = _get_value_limits(self._measurement.array, value_limits=value_limits)
        self.set_value_limits(value_limits)

    def set_column_titles(
        self,
        titles: str | list[str],
        pad: float = 10.0,
        fontsize: float = 12,
        **kwargs,
    ):
        if isinstance(titles, str):
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
        titles: str | list[str],
        shift: float = 0.0,
        fontsize: float = 12,
        **kwargs,
    ):
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

    # def set_panel_labels(
    #     self,
    #     labels: str = None,
    #     frameon: bool = True,
    #     loc: str = "upper left",
    #     pad: float = 0.1,
    #     borderpad: float = 0.1,
    #     prop: dict = None,
    #     formatting: str = ".3g",
    #     units: str = None,
    #     **kwargs,
    # ):
    #     if labels is None:
    #         titles = _get_axes_titles(
    #             self.ensemble_axes_metadata, self.axes_types, self.axes.shape
    #         )
    #         labels = ["\n".join(title) for title in titles]
    #
    #     if not isinstance(labels, (tuple, list)):
    #         raise ValueError()
    #
    #     if len(labels) != np.array(self.axes).size:
    #         raise ValueError()
    #
    #     if prop is None:
    #         prop = {}
    #
    #     for old_label in self._panel_labels:
    #         old_label.remove()
    #
    #     panel_labels = []
    #     for ax, label in zip(np.array(self.axes).ravel(), labels):
    #         anchored_text = AnchoredText(
    #             label,
    #             pad=pad,
    #             borderpad=borderpad,
    #             frameon=frameon,
    #             loc=loc,
    #             prop=prop,
    #             **kwargs,
    #         )
    #         anchored_text.formatting = formatting
    #
    #         anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    #         ax.add_artist(anchored_text)
    #
    #         panel_labels.append(anchored_text)
    #
    #     self._panel_labels = panel_labels

    def axis(self, mode: str = "all", ticks: bool = False, spines: bool = True):
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
        if self.artists is None:
            return

        for artist in self.artists.ravel():
            artist.remove()
    
    def update_data_indices(self, indices):
        self._indices = indices
        for i in np.ndindex(self.axes.shape):
            data = self._reduce_measurement(indices, i)
            self._artists[i].set_data(data)

    def _make_new_artists(
        self,
        **kwargs,
    ):
        self.remove_artists()
        artist_type = _get_artist_type(
            self.measurement, complex_conversion=self._complex_conversion
        )
        

        artists = np.zeros(self.axes.shape, dtype=object)
        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]

            if hasattr(self.axes, "_caxes"):
                caxes = self.axes._caxes[i]
                
                if self.axes._cbar_mode == "single" and not i == (0, 0):
                    caxes = None
            else:
                ss
                caxes = [_make_cax(ax) for _ in range(artist_type.num_cbars)]

            measurement = self._reduce_measurement(self._indices, i)
            
            artist = artist_type(
                ax=ax,
                caxes=caxes,
                measurement=measurement,
                **kwargs,
            )

            artists.itemset(i, artist)

        self._artists = artists

    def set_artists(self, name, locs: str | tuple[int, ...] = "all", **kwargs):
        artist_type = _get_artist_type(self._measurement, self._complex_conversion)

        if not hasattr(artist_type, f"set_{name}"):
            raise RuntimeError(
                f"artist of type '{artist_type.__name__}' does not have a method 'set_{name}'"
            )

        if not hasattr(self.axes, "_axis_location_to_indices"):
            locs = tuple(i for i in np.ndindex(self.axes.shape))

        if isinstance(locs, str):
            locs = self.axes._axis_location_to_indices(locs)

        for i in locs:
            getattr(self.artists[i], f"set_{name}")(**kwargs)

    def set_legend(self, **kwargs):
        self.set_artists("legend", locs="all", **kwargs)

    def set_cbars(self, **kwargs):
        self.set_artists("cbars", locs="all", **kwargs)

    def set_complex_conversion(self, complex_conversion: str):
        self._complex_conversion = complex_conversion
        artist_type = _get_artist_type(
            self._measurement, complex_conversion=self._complex_conversion
        )
        self.axes.set_cbar_layout(ncbars=artist_type.num_cbars)
        self.set_artists()

    def set_cmap(self, cmap):
        cmap = validate_cmap(cmap, self.measurement, self._complex_conversion)
        self.set_artists("cmap", cmap=cmap)

    def set_scale_bars(self, locs: str = "lower right", **kwargs):
        self.set_artists("scale_bars", locs=locs, **kwargs)



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
