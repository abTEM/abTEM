"""Module for plotting atoms, images, line scans, and diffraction patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation as R

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.utils import itemset, label_to_index
from abtem.visualize.artists import (
    DomainColoringArtist,
    ImageArtist,
    LinesArtist,
    ScatterArtist,
    _get_value_limits,
    validate_cmap,
)
from abtem.visualize.axes_grid import AxesCollection, AxesGrid
from abtem.visualize.widgets import slider_from_axes_metadata

if TYPE_CHECKING:
    from abtem.measurements import BaseMeasurements


def discrete_cmap(num_colors, base_cmap):
    if isinstance(base_cmap, str):
        base_cmap = plt.get_cmap(base_cmap)
    colors = base_cmap(range(0, num_colors))
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors, num_colors)


def _validate_axes_types(overlay, explode, ensemble_dims):
    if explode is True:
        explode = tuple(range(ensemble_dims))
    elif explode is False:
        explode = ()

    if overlay is True:
        overlay = tuple(range(ensemble_dims))
    elif overlay is False:
        overlay = ()

    if isinstance(explode, int):
        explode = (explode,)

    if isinstance(overlay, int):
        overlay = (overlay,)

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


def _validate_artist_type(measurement, complex_conversion, artist_type=None):
    if artist_type is not None:
        return artist_type

    if len(measurement.base_shape) == 2:
        if measurement.is_complex and (
            complex_conversion in ("domain_coloring", "none", None)
        ):
            return DomainColoringArtist
        else:
            return ImageArtist
    elif hasattr(measurement, "miller_indices"):
        return ScatterArtist
    elif len(measurement.base_shape) == 1:
        return LinesArtist
    elif artist_type is None:
        raise ValueError("artist type not recognized")
    else:
        return artist_type


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
        and isinstance(ax, matplotlib.axes._base._AxesBase)
        and ax.get_subplotspec()
    ):
        cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
    else:
        cax, kwargs = cbar.make_axes(ax, **kwargs)
    # make_axes calls add_{axes,subplot} which changes gca; undo that.
    fig.sca(current_ax)
    cax.grid(visible=False, which="both", axis="both")
    return cax


class Visualization:
    def __init__(
        self,
        measurement,
        ax: Axes = None,
        artist_type=None,
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
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        convert_complex: str = "none",
        **kwargs,
    ):
        self._measurement = measurement.to_cpu().compute()

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

        self._artist_type = _validate_artist_type(
            measurement, complex_conversion="none", artist_type=artist_type
        )

        if cbar:
            ncbars = self._artist_type.num_cbars
        else:
            ncbars = 0

        if common_scale:
            cbar_mode = "single"
        else:
            cbar_mode = "each"

        if ax is None:
            axes_shape = tuple(
                n
                for i, n in enumerate(measurement.ensemble_shape)
                if i not in self.indexing_axes and i not in overlay
            )
            shape = axes_shape + (0,) * (2 - len(axes_shape))
            ncols, nrows = (max(shape[0], 1), max(shape[1], 1))

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
            axes = np.array([[ax]], dtype=object)
            caxes = np.zeros_like(axes, dtype=object)

            for i in np.ndindex(axes.shape):
                caxes[i] = [_make_cax(ax, **kwargs) for i in range(ncbars)]

            axes = AxesCollection(axes, caxes, cbar_mode=cbar_mode)

        self._axes = axes

        self._indices = ()

        self._complex_conversion = convert_complex

        self._autoscale = config.get("visualize.autoscale", False)
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._artists = None

        self.get_figure().canvas.header_visible = False

        if isinstance(title, str):
            self.set_column_titles(title)

        elif title and len(explode) > 0:
            axes_metadata = measurement.axes_metadata[explode[0]].to_ordinal_axis(
                measurement.shape[explode[0]]
            )

            column_titles = [
                l.format_title(".3g", include_label=i == 0)
                for i, l in enumerate(axes_metadata)
            ]

            self.set_column_titles(column_titles)

        if title and len(explode) > 1:
            axes_metadata = measurement.axes_metadata[explode[1]].to_ordinal_axis(
                measurement.shape[explode[1]]
            )

            row_titles = [
                l.format_title(".3g", include_label=i == 0)
                for i, l in enumerate(axes_metadata)
            ]

            self.set_row_titles(row_titles)

        self._make_new_artists(artist_type=self._artist_type, **kwargs)

        self.adjust_coordinate_limits_to_artists(xlim=xlim, ylim=ylim)

        if common_scale:
            self.set_common_value_limits(value_limits)
        else:
            self.set_value_limits(value_limits)

        if self._artist_type is DomainColoringArtist and isinstance(
            self._axes, AxesGrid
        ):
            self.axes.set_sizes(cbar_spacing=0.5)

    def interact(self, gui_type, display):
        if "ipympl" not in matplotlib.get_backend():
            raise RuntimeError(
                "interactive visualizations requires the 'ipympl' matplotlib backend"
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

    def adjust_coordinate_limits_to_artists(self, xlim=None, ylim=None):
        if xlim is None:
            xlim = [np.inf, -np.inf]
        if ylim is None:
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

    def set_value_limits(
        self, value_limits: tuple[float, float] | list[float] = (None, None)
    ):
        self.set_artists("value_limits", value_limits=value_limits)

    def set_power(self, power: float = 1.0):
        self.set_artists("power", power=power)

    def set_common_value_limits(self, value_limits=(None, None)):
        value_limits = _get_value_limits(
            self._measurement.array, value_limits=value_limits
        )
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

            # annotation = ax.annotate(
            #     titles[i],
            #     xy=(1, 0.5),
            #     xytext=(pad, 0),
            #     xycoords="axes fraction",
            #     textcoords="offset points",
            #     ha="right",
            #     va="center",
            #     rotation=270,
            #     fontsize=fontsize,
            #     **kwargs,
            # )
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
            indices = tuple(self.axes.axis_location_to_indices(mode))

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

    def axis_off(self):
        self.axis("none", spines=False)

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
        artist_type=None,
        **kwargs,
    ):
        self.remove_artists()
        artist_type = _validate_artist_type(
            self.measurement,
            complex_conversion=self._complex_conversion,
            artist_type=self._artist_type,
        )

        artists = np.zeros(self.axes.shape, dtype=object)
        for i in np.ndindex(self.axes.shape):
            ax = self.axes[i]

            # if hasattr(self.axes, "_caxes"):
            caxes = self.axes._caxes[i]

            if self.axes._cbar_mode == "single" and not i == (0, 0):
                caxes = None
            # else:
            #     caxes = [_make_cax(ax) for _ in range(artist_type.num_cbars)]

            measurement = self._reduce_measurement(self._indices, i)

            artist = artist_type(
                ax=ax,
                caxes=caxes,
                measurement=measurement,
                **kwargs,
            )

            itemset(artists, i, artist)

        self._artists = artists

    def set_artists(self, name, locs: str | tuple[int, ...] = "all", **kwargs):
        artist_type = _validate_artist_type(
            self._measurement, self._complex_conversion, artist_type=self._artist_type
        )

        if not hasattr(artist_type, f"set_{name}"):
            raise RuntimeError(
                f"artist of type '{artist_type.__name__}' does not have a method 'set_{name}'"
            )

        if not hasattr(self.axes, "_axis_location_to_indices"):
            locs = tuple(i for i in np.ndindex(self.axes.shape))

        if isinstance(locs, str):
            locs = self.axes.axis_location_to_indices(locs)

        for i in locs:
            getattr(self.artists[i], f"set_{name}")(**kwargs)

    def set_legend(self, **kwargs):
        self.set_artists("legend", locs="all", **kwargs)

    def set_cbars(self, **kwargs):
        self.set_artists("cbars", locs="all", **kwargs)

    def set_complex_conversion(self, complex_conversion: str):
        raise NotImplementedError()
        # self._complex_conversion = complex_conversion
        # artist_type = _validate_artist_type(
        #     self._measurement, complex_conversion=self._complex_conversion
        # )
        # self.axes.set_cbar_layout(ncbars=artist_type.num_cbars)
        # self.set_artists()

    def set_cmap(self, cmap):
        cmap = validate_cmap(cmap, self.measurement, self._complex_conversion)
        self.set_artists("cmap", cmap=cmap)

    def set_scale_bars(self, locs: str = "lower right", **kwargs):
        self.set_artists("scale_bars", locs=locs, **kwargs)


#  ====================================================  #
#    Tools and functions for 2D and 3D plots of atoms    #
#  ====================================================  #


class CellCalculations:
    """
    Class for assisting in plotting unit cells and atoms. Handles
    calculations of cell vertexes, lines, and faces necessary for plotting,
    as well as default atomic radii, positions, and coloring.
    Intended for internal use by other abTEM functions.
    """

    def __init__(self, ase_atoms: Atoms):
        """
        extract the atomic and lattice information necessary for plotting
        from an `ase.atoms.Atoms` object
        """
        # create the vertices, edges, and faces for a unit cube
        _vx = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        _vy = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        _vz = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        _unit_verts = np.stack([_vx, _vy, _vz]).T
        _el = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3])
        _er = np.array([1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7])
        self._unit_edges = np.stack([_el, _er]).T
        _fa = np.array([0, 0, 1, 2, 3, 4])
        _fb = np.array([1, 1, 5, 6, 7, 5])
        _fc = np.array([2, 4, 6, 7, 4, 6])
        _fd = np.array([3, 5, 2, 3, 0, 7])
        self._unit_faces = np.stack([_fa, _fb, _fc, _fd, _fa]).T

        # use the Atoms.cell information to scale the points
        # NOTE: faces are not currently used in abtem, but are easy to
        # calculate at this step and could be handy in off-axis plots for
        # highlighing the entry and exit faces. They can be added to a plot
        # with `patches.PathPatch(self.faces[0]).`
        self.cell = ase_atoms.cell
        self.verts = np.dot(_unit_verts, self.cell)
        self.edges = self.verts[self._unit_edges]
        self.faces = self.verts[self._unit_faces]
        self.center = np.dot([0.5, 0.5, 0.5], self.cell)
        # collect the atomic positions, coloring, radii, etc.
        self.elem = ase_atoms.arrays["numbers"]
        self.pos = ase_atoms.arrays["positions"]
        self.colors = jmol_colors[self.elem]
        self.radii = covalent_radii[self.elem]
        # define the 2D plotting axes.
        self.x_plot_axis_2d = np.array([1, 0, 0])
        self.y_plot_axis_2d = np.array([0, 1, 0])

    def rotate(self, alpha: float, beta: float, gamma: float):
        """
        Rotates atomic positions and lattices by an alpha/beta/gamma tilt.

        specifically, this is an active (ie, the part moves, not
        the obverver) extrinsic(the rotation axes are fixed to the reference
        not the object) rotation clockwise via the righthand rule around the
        x-axis, then the y-axis, then the z-axis.

        Thus, alpha and beta align with the alpha/beta tilt angles used in a
        TEM, and gamma represents a clockwise rotation around the viewing axis.

        NOTE:, this is rotating the VIEWING DIRECTION ONLY, and will have no
        effect on simulation properties.
        """
        r = R.from_euler("xyz", [alpha, beta, gamma], degrees=True).as_matrix()
        # NOTE: this rotates the cell around (0,0,0), NOT the cell centroid.
        self.verts = np.dot(r, self.verts.T).T
        self.edges = self.verts[self._unit_edges]
        self.faces = self.verts[self._unit_faces]
        self.center = np.dot(r, self.center.T).T
        self.x_plot_axis_2d = np.dot(r, self.x_plot_axis_2d.T).T
        self.y_plot_axis_2d = np.dot(r, self.y_plot_axis_2d.T).T
        if self.pos.size > 0:
            self.pos = np.dot(r, self.pos.T).T

    def merge_atomic_columns(self, tol: float = 1e-7):
        """
        for large 2D plots, it is often prudent to plot only the atomic
        columns as opposed to every individual overlapping atom. This function
        clusters identical atoms whose xy- positions are within a given
        tolerance and preserves only the atoms with the highest z-position
        from each cluster.

        Note, this function should be called AFTER all viewing rotations are
        applied, as it is not data-preserving
        """
        new_p, new_e = [], []
        for elem in np.unique(self.elem):
            pos = self.pos[self.elem == elem]
            columns = []
            while len(pos) > 0:
                # pick a candidate to look at
                x = pos[0]
                # find all their neighbors in the xy plane
                mask = np.sum(((pos - x)[:, :2]) ** 2, axis=1) ** 0.5 < tol
                # the highest neighbor defines the colum, the rest are trashed
                columns.append(pos[mask][np.argmax(pos[mask, 2])])
                pos = pos[~mask]
            new_p.append(columns)
            new_e.append(np.repeat(elem, len(columns)))
        self.pos = np.vstack(new_p)
        self.elem = np.hstack(new_e)
        self.colors = jmol_colors[self.elem]
        self.radii = covalent_radii[self.elem]

    def get_plot_labels(self):
        labels = []
        for vec in [self.x_plot_axis_2d, self.y_plot_axis_2d]:
            vec = vec / np.linalg.norm(vec)
            if np.abs(vec).max() > 0.999999:
                # this is a primary axis, label it as such
                name = "XYZ"[np.argmax(vec)]
            else:
                name = "({})".format(str(np.around(vec, 3))[1:-2])
            labels.append(name + " [Å]")
        return labels


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
    # For consistency, use the same atom padding as elsewhere in abtem
    if show_periodic:
        atoms = atoms.copy()
        atoms = pad_atoms(atoms, margins=1e-3)

    if isinstance(plane, str):
        # use plane_to_axes to get the proper axis reordering, and use scipy
        # to calc the alpha/beta/gamma that gives that axis alignment.
        plane = R.align_vectors(
            np.eye(3)[:2, :], np.eye(3)[plane_to_axes(plane)[:2], :]
        )[0].as_euler("xyz", degrees=True)
    # cast plane to 1D numpy array if possible and do a sanity check
    plane = np.asanyarray(plane).flatten()
    if plane.size == 2:
        plane = np.array([plane[0], plane[1], 0])
    assert plane.size == 3, (
        "plane must be either a string interpretable by"
        + "abtem.atoms.plane_to_axes, an alpha/beta angle pair, or an"
        + "alpha/beta/gamma triplet that can be cast to a numpy array"
    )

    # use CellCalculations to precalculate information for plotting
    cc = CellCalculations(atoms)
    cc.rotate(*plane)

    if merge > 0.0:
        cc.merge_atomic_columns(merge)

    if tight_limits and show_cell is None:
        show_cell = False
    elif show_cell is None:
        show_cell = True

    # Either use an existing plot or generate a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if show_cell:
        # NOTE TO REVIEWERS: this dash-or-not-dash method is imperfect.
        # the better way to do this would be with shapely, but that's adding
        # a big dependency for one plotting function. Otherwise i can add some
        # ad-hoc "does the point intercept a face above" logic later.
        above = cc.edges[:, :, 2].mean(axis=1) > cc.center[2]
        print(above)
        # plot the lower edges with dashed lines
        [ax.plot(*x.T, "k--") for x in cc.edges[~above, :, :2]]
        # plot the upper edges with solid lines
        [ax.plot(*x.T, "k-") for x in cc.edges[above, :, :2]]

    if cc.pos.size > 0:
        # convert the atomic data into circular patches and plot them
        circles = [Circle(*x) for x in zip(cc.pos, cc.radii * scale)]
        coll = PatchCollection(
            circles, facecolors=cc.colors, edgecolors="black", **kwargs
        )
        ax.add_collection(coll)
        # add numerical labels to atoms if desired.
        if numbering:
            if merge:
                raise ValueError("atom numbering requires 'merge' to be False")

            for i, (position, size) in enumerate(zip(cc.pos, cc.radii)):
                ax.annotate("{}".format(i), xy=position, ha="center", va="center")
    # clean up the plotting and add labels
    ax.set_aspect("equal")
    axis_labels = cc.get_plot_labels()
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)

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
        ax.set_xlim(np.min(cc.edges[:, :, 0]), np.min(cc.edges[:, :, 0]))
        ax.set_ylim(np.min(cc.edges[:, :, 1]), np.min(cc.edges[:, :, 1]))

    return fig, ax


def show_atoms3d(
    atoms: Atoms,
    ax: Axes = None,
    scale: float | str = "infer",
    title: str = None,
    numbering: bool = False,
    show_periodic: bool = False,
    figsize: tuple[float, float] = None,
    legend: bool = False,
    show_cell: bool = None,
    #    backemd: str = 'matplotlib'
    **kwargs,
):
    """
    Create 3D plot of atoms

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms to be shown.
    ax : matplotlib.axes.Axes, optional
        If given, the plots are added to the axes.
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
    # For consistency, use the same atom padding as elsewhere in abtem
    if show_periodic:
        atoms = atoms.copy()
        atoms = pad_atoms(atoms, margins=1e-3)

    # use CellCalculations to precalculate information for plotting
    cc = CellCalculations(atoms)

    if show_cell is None:
        show_cell = True

    # Either use an existing plot or generate a new one
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    else:
        assert isinstance(ax, Axes3D)
        fig = ax.get_figure()

    if scale == "infer":
        # TODO: this is a bit ad-hoc, since scatter spot size is based on
        # pixel count. Also, this will absolutely over-predict size on highly
        # anisotropic cells
        # get the square root of the pixel count in your plotting axis
        pixel_count_sqrt = np.prod(ax.get_tightbbox().size) ** 0.5
        # use the cube root of the atom count and average atomic radii to
        # predict the density which which things will be plotted
        plot_density = (cc.elem.size ** (1 / 3)) * cc.radii.mean() / pixel_count_sqrt
        # 20 percent seems to look nice, so set the scale to achieve that
        # density
        scale = 0.2 / plot_density

    if show_cell:
        [ax.plot3D(*x.T, "k-") for x in cc.edges]

    if cc.pos.size > 0:
        # NOTE TO REVIEWERS: Its possible to do true 3d spheres here, but it
        # can very quickly crash matplotlib, as every sphere needs a mesh
        # with ~100 or so vertices and faces. This CAN be rendered in pyvista
        # smoothly, but that has a lot of overhead baggage.
        # If there is interest, I can include that, but for now, sticking to
        # simple scaled scatter plots
        ax.scatter3D(*cc.pos.T, s=(cc.radii * scale) ** 2, c=cc.colors)

    # clean up the plotting and add labels
    ax.set_aspect("equal")
    axis_labels = cc.get_plot_labels()
    ax.set_xlabel("X [Å]")
    ax.set_ylabel("Y [Å]")
    ax.set_zlabel("Z [Å]")
    ax.set_title(title)

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

    return fig, ax
