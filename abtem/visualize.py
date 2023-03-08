"""Module for plotting atoms, images, line scans, and diffraction patterns."""
import string
from typing import TYPE_CHECKING, List
from typing import Union, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib import cm, colors
from matplotlib.axes import Axes

import ipywidgets as widgets
from scipy.spatial.distance import squareform
from scipy.spatial import distance_matrix

from abtem.core.colors import hsluv_cmap
from matplotlib.collections import PatchCollection, CircleCollection, EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core.backend import copy_to_device
from abtem.core import config
from abtem.core.utils import label_to_index
from abtem.core.units import _get_conversion_factor
from abtem.atoms import pad_atoms, plane_to_axes

if TYPE_CHECKING:
    from abtem.measurements import (
        BaseMeasurement,
        RealSpaceLineProfiles,
        ReciprocalSpaceLineProfiles,
        BaseMeasurement2D,
    )


def _iterate_axes(axes: Union[ImageGrid, Axes]):
    try:
        for ax in axes:
            yield ax
    except TypeError:
        yield axes


class MeasurementVisualizationAxes:

    def __init__(self, axes, measurements):
        self._axes = axes
        self._measurements = measurements

    @property
    def measurements(self):
        return self._measurements

    @property
    def axes(self):
        return self._axes

    @property
    def fig(self):
        return self._axes[0].get_figure()

    def iterate_axes(self):
        try:
            for ax in self.axes:
                yield ax
        except TypeError:
            yield self.axes

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


class Measurement2DVisualizationAxes(MeasurementVisualizationAxes):

    def __init__(self, axes, measurements):
        super().__init__(axes, measurements)

    def set_column_titles(
            self, pad=10.0, format: str = ".3g", units=None, fontsize=12, **kwargs
    ):
        include_label = True
        for ax, axis_metadata in zip(
                np.array(self.axes.axes_column)[:, 0], self.measurements.ensemble_axes_metadata[0]
        ):

            for child in ax.get_children():
                if hasattr(child, "is_column_title"):
                    child.remove()

            annotation = ax.annotate(
                axis_metadata.format_title(
                    format, units=units, include_label=include_label
                ),
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                fontsize=fontsize,
                **kwargs,
            )
            annotation.is_column_title = True

            include_label = False

    def iterate_image_grid(self, flip_vertical=True):
        axes = np.array(self.axes.axes_column, dtype=object)

        if flip_vertical:
            axes = np.fliplr(axes)

        for i in np.ndindex(axes.shape):
            yield axes[i]

    def iterate_measurements(self):
        for _, measurement in self.measurements.iterate_ensemble(keep_dims=True):
            yield measurement

    def iterate_images(self):
        for ax in self.iterate_image_grid():
            for im in ax.get_images():
                yield im

    def set_x_extent(self, extent):
        for im in self.iterate_images():
            old_extent = im.get_extent()
            y_extent = old_extent[2:]
            new_extent = extent + y_extent
            im.set_extent(new_extent)

    def set_y_extent(self, extent):
        for im in self.iterate_images():
            old_extent = im.get_extent()
            x_extent = old_extent[:2]
            new_extent = x_extent + extent
            im.set_extent(new_extent)

    def set_x_axes(
            self,
            units: str = None,
            label: str = None,
    ):
        if label is None:
            label = self.measurements.axes_metadata[-2].format_label(units)

        for ax, measurement in zip(self.iterate_image_grid(), self.iterate_measurements()):
            ax.set_xlabel(label)
            print(measurement._plot_extent())


def _make_cbar_label(measurement):
    cbar_label = (
        measurement.metadata["label"] if "label" in measurement.metadata else ""
    )
    cbar_label += (
        f" [{measurement.metadata['units']}]" if "units" in measurement.metadata else ""
    )
    return cbar_label


def add_sizebar(
        ax,
        measurements,
        size=None,
        loc="lower right",
        borderpad=0.5,
        formatting: str = ".3f",
        units=None,
        **kwargs,
):
    if units is None:
        units = measurements.base_axes_metadata[-2].units

    if size is None:
        size = (
                measurements.base_axes_metadata[-2].sampling
                * measurements.base_shape[-2]
                / 3
        )

        num = size * _get_conversion_factor(units, units)
    else:
        num = size

    label = f"{num:>{formatting}} {units}"

    anchored_size_bar = AnchoredSizeBar(
        ax.transData, label=label, size=size, borderpad=borderpad, loc=loc, **kwargs
    )
    ax.add_artist(anchored_size_bar)

    return anchored_size_bar


def add_imshow(
        ax: Axes,
        measurements: "BaseMeasurement",
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
):
    if not measurements.is_complex and cmap is None:
        cmap = config.get("cmap", "viridis")
    elif cmap is None:
        cmap = config.get("phase_cmap", "hsluv")

    array = measurements.array

    if measurements.is_complex:

        abs_array = np.abs(array)
        alpha = (abs_array - abs_array.min()) / abs_array.ptp()

        im1 = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation="none",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        if cmap == "hsluv":
            cmap = hsluv_cmap

        im2 = ax.imshow(
            np.angle(array).T,
            origin="lower",
            interpolation="none",
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
        )
    else:
        im1 = ax.imshow(
            array.T,
            origin="lower",
            interpolation="none",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    set_extent(ax, measurements)
    return im1


def set_image_data(
        axes, measurements, complex_representation="intensity", recreate=False
):
    measurements = measurements.compute().to_cpu()

    if measurements.is_complex and complex_representation != "domain_coloring":
        measurements = getattr(measurements, complex_representation)()

    for ax in _iterate_axes(axes):
        if recreate:
            fig = axes.get_figure()
            fig.clear()
            for im in ax.get_images():
                im.remove()
            ax = fig.add_subplot()
            add_imshow(ax, measurements)
            set_colorbars(ax, measurements)
        else:
            if measurements.is_complex:
                im1, im2 = ax.get_images()[:2]
                abs_array = np.abs(measurements.array)
                im1.set_data(abs_array.T)
                alpha = im1.norm(abs_array)
                alpha = np.clip(alpha, a_min=0, a_max=1)
                im2.set_alpha(alpha.T)
                im2.set_data(np.angle(measurements.array).T)
            else:
                im = ax.get_images()[0]
                im.set_data(measurements.array.T)


def set_colorbars(axes, measurements, label=None, fontsize=12, **kwargs):
    for ax, (_, measurement) in zip(
            _iterate_axes(axes), measurements.iterate_ensemble(keep_dims=True)
    ):

        ims = ax.get_images()

        if label is None:
            label = _make_cbar_label(measurement)

        if measurements.is_complex:
            cbar1 = plt.colorbar(ims[0], ax=ax)
            cbar2 = plt.colorbar(ims[1], ax=ax)

            cbar1.set_label("abs", rotation=0, ha="center", va="top")
            cbar1.ax.yaxis.set_label_coords(0.5, -0.03)

            cbar2.set_label("arg", rotation=0, ha="center", va="top")
            cbar2.ax.yaxis.set_label_coords(0.5, -0.03)
            cbar2.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
            cbar2.set_ticklabels(
                [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
            )
        else:
            if hasattr(ax, "cax"):
                cbar = ax.cax.colorbar(ims[0], **kwargs)
                cbar.set_label(label, fontsize=fontsize)
            else:
                cbar = plt.colorbar(ims[0], ax=ax, label=label, **kwargs)
                cbar.set_label(label, fontsize=fontsize)


def set_normalization(
        axes: Axes,
        measurements: "BaseMeasurement2D",
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
):
    for ax, (_, measurement) in zip(
            _iterate_axes(axes), measurements.iterate_ensemble(keep_dims=True)
    ):

        if measurement.is_complex:
            measurement = measurement.abs()

        if vmin is None:
            measurement_vmin = measurement.array.min()
        else:
            measurement_vmin = vmin

        if vmax is None:
            measurement_vmax = measurement.array.max()
        else:
            measurement_vmax = vmax

        if power != 1:
            norm = colors.PowerNorm(
                gamma=power, vmin=measurement_vmin, vmax=measurement_vmax
            )
        else:
            norm = colors.Normalize(vmin=measurement_vmin, vmax=measurement_vmax)

        im = ax.get_images()[0]
        im.norm = norm


def set_super_title(axes, measurements, super_title):
    fig = axes.get_figure()

    if isinstance(super_title, str):
        fig.suptitle(super_title)

    elif "name" in measurements.metadata:
        fig.suptitle(measurements.metadata["name"])


def set_sub_titles(axes, measurements):
    fig = axes.get_figure()

    if len(measurements.ensemble_axes_metadata) > 0:
        fig.supxlabel(f"{measurements.ensemble_axes_metadata[-1].format_label()}")

    if len(measurements.ensemble_axes_metadata) > 1:
        fig.supylabel(f"{measurements.ensemble_axes_metadata[-2].format_label()}")


def set_row_titles(
        axes,
        measurements,
        pad: float = 0.0,
        format: str = ".2g",
        units: str = None,
        fontsize: int = 12,
        **kwargs,
):
    include_label = True
    for ax, axis_metadata in zip(
            np.array(axes.axes_column)[0, ::-1], measurements.axes_metadata[1]
    ):

        for child in ax.get_children():
            if hasattr(child, "is_row_title"):
                child.remove()

        at = ax.annotate(
            axis_metadata.format_title(
                format, units=units, include_label=include_label
            ),
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

        at.is_row_title = True
        include_label = False


def set_titles(axes, measurements, format: str = ".3g", title=None, units=None):
    if not measurements.ensemble_shape:
        return

    set_column_titles(axes, measurements, format=format, units=units)

    if len(measurements.ensemble_shape) != 2:
        return

    set_row_titles(axes, measurements, format=format, units=units)


def set_extent(axes, measurements, units=None, extent=None):
    if extent is None:
        extent = measurements._plot_extent(units)

    for ax, (_, measurement) in zip(
            _iterate_axes(axes), measurements.iterate_ensemble(keep_dims=True)
    ):
        for im in ax.get_images():
            im.set_extent(extent)





def set_ylabels(
        axes: Axes,
        measurements: "BaseMeasurement",
        units: str = None,
        label: str = None,
):
    if label is None:
        label = measurements.axes_metadata[-1].format_label(units)

    for ax, (_, measurement) in zip(
            _iterate_axes(axes), measurements.iterate_ensemble(keep_dims=True)
    ):
        ax.set_ylabel(label)


def _iterate_image_grid(axes, flip_vertical=True):
    axes = np.array(axes.axes_column, dtype=object)

    if flip_vertical:
        axes = np.fliplr(axes)

    for i in np.ndindex(axes.shape):
        yield axes[i]


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
        units: str = None,
        axes_pad=None,
        axis_off=False,
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

    if cbar and measurements.is_complex and measurements.ensemble_shape:
        raise NotImplementedError(
            "colorbar not implemented for exploded plot with domain coloring"
        )

    complex_representation = "domain_coloring"

    if measurements.is_complex and complex_representation != "domain_coloring":
        measurements = getattr(measurements, complex_representation)()

    if not measurements.ensemble_shape:
        if axes is None:
            fig, axes = plt.subplots(figsize=figsize)
        else:
            fig = axes.get_figure()

        add_imshow(axes, measurements, cmap=cmap)

        if cbar:
            set_colorbars(axes, measurements)

        # set_extent(axes, measurements, units=units)
        # set_titles(axes, measurements, title=title)
        #set_xlabels(axes, measurements, units=units)
        #set_ylabels(axes, measurements, units=units)
        set_normalization(axes, measurements, vmin=vmin, vmax=vmax, power=power)
        return fig, axes

    measurements = measurements[(0,) * max(len(measurements.ensemble_shape) - 2, 0)]

    if axes is None:
        fig = plt.figure(1, figsize, clear=True)

        if len(measurements.ensemble_shape) == 1:
            ncols = measurements.ensemble_shape[0]
            nrows = 1

        elif len(measurements.ensemble_shape) == 2:
            ncols = measurements.ensemble_shape[-2]
            nrows = measurements.ensemble_shape[-1]

        else:
            raise RuntimeError()

        image_grid_kwargs = {}
        image_grid_kwargs["share_all"] = True
        if common_color_scale:
            if cbar:
                image_grid_kwargs["cbar_mode"] = "single"

            image_grid_kwargs["axes_pad"] = 0.1
        else:
            if cbar:
                image_grid_kwargs["cbar_mode"] = "each"
                image_grid_kwargs["cbar_pad"] = 0.05
                image_grid_kwargs["axes_pad"] = 0.8
            else:
                image_grid_kwargs["axes_pad"] = 0.1

        if axes_pad is not None:
            image_grid_kwargs["axes_pad"] = axes_pad

        axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), **image_grid_kwargs)
    else:
        fig = axes.get_figure()

    if common_color_scale:
        vmin = measurements.array.min()
        vmax = measurements.array.max()

    for ax, (_, measurement) in zip(
            _iterate_image_grid(axes), measurements.iterate_ensemble(keep_dims=False)
    ):
        add_imshow(ax=ax, measurements=measurement, cmap=cmap, vmin=vmin, vmax=vmax)

    if cbar:
        set_colorbars(axes, measurements)

    if axis_off:
        for ax in _iterate_axes(axes):
            ax.set_xticks([])
            ax.set_yticks([])
    #
    # else:
    #     set_xlabels(axes, measurements, units=units)
    #     set_ylabels(axes, measurements, units=units)

    # if title:
    #    set_titles(axes, measurements, title=title)

    set_normalization(axes, measurements, vmin=vmin, vmax=vmax, power=power)

    return Measurement2DVisualizationAxes(axes, measurements)


def _add_plot(x: np.ndarray, y: np.ndarray, ax: Axes, label: str = None, **kwargs):
    y = copy_to_device(y, np)
    x, y = np.squeeze(x), np.squeeze(y)

    if np.iscomplexobj(x):
        if label is None:
            label = ""

        line1 = ax.plot(x, y.real, label=f"Real {label}", **kwargs)
        line2 = ax.plot(x, y.imag, label=f"Imag. {label}", **kwargs)
        line = (line1, line2)
    else:
        line = ax.plot(x, y, label=label, **kwargs)

    return line


def show_measurements_1d(
        measurements: Union["RealSpaceLineProfiles", "ReciprocalSpaceLineProfiles"],
        extent: Tuple[float, float] = None,
        ax: Axes = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        figsize: Tuple[int, int] = None,
        units=None,
        **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if title is None and "name" in measurements.metadata:
        title = measurements.metadata["name"]

    if title is not None:
        ax.set_title(title)

    if extent is None:
        extent = [0, measurements.extent]

    if x_label is None:
        x_label = f"{measurements.axes_metadata[-1].label} [{measurements.axes_metadata[-1].units}]"

    x = np.linspace(extent[0], extent[1], measurements.shape[-1], endpoint=False)

    for index, line_profile in measurements.iterate_ensemble(keep_dims=True):

        labels = []
        for axis in line_profile.ensemble_axes_metadata:
            labels += [axis.format_title(".2e")]

        label = "-".join(labels)

        _add_plot(x, line_profile.array, ax, label, **kwargs)

    ax.set_xlabel(x_label)

    if y_label is None and "label" in measurements.metadata:
        y_label = measurements.metadata["label"]
        if "units" in measurements.metadata:
            y_label += f' [{measurements.metadata["units"]}]'
        ax.set_ylabel(y_label)

    ax.set_ylabel(y_label)

    if len(measurements.ensemble_shape) > 0:
        ax.legend()

    return fig, ax


def _show_indexed_diffraction_pattern(
        indexed_diffraction_pattern,
        scale: float = 1.0,
        ax: Axes = None,
        figsize: Tuple[float, float] = (6, 6),
        title: str = None,
        overlay_hkl: bool = True,
        inequivalency_threshold: float = 1.0,
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

    intensities = indexed_diffraction_pattern.intensities ** power

    order = np.argsort(-np.linalg.norm(positions, axis=1))

    positions = positions[order]
    intensities = intensities[order]

    scales = intensities / intensities.max()

    min_distance = squareform(distance_matrix(positions, positions)).min()

    scale_factor = min_distance / scales.max() * scale

    scales = scales ** power * scale_factor

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
    # fig.patch.set_facecolor(background_color)
    # ax.axis("off")

    if overlay_hkl:
        add_miller_index_annotations(ax, indexed_diffraction_pattern)

    return fig, ax


def add_miller_index_annotations(ax, indexed_diffraction_patterns):
    for hkl, position in zip(
            indexed_diffraction_patterns.miller_indices,
            indexed_diffraction_patterns.positions,
    ):
        annotation = ax.annotate(
            "{} {} {}".format(*hkl),
            xy=position[:2],
            ha="center",
            va="center",
            size=8,
        )
        annotation.set_path_effects([withStroke(foreground="w", linewidth=3)])


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
        ax: Axes = None,
        scale: float = 0.75,
        title: str = None,
        numbering: bool = False,
        show_periodic: bool = False,
        figsize: Tuple[float, float] = None,
        legend: bool = False,
        merge: float = 1e-2,
        tight_limits=False,
        show_cell=True,
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
