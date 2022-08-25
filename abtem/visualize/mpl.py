"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from collections.abc import Iterable
from typing import Union, Tuple, TYPE_CHECKING, List

import cplot
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core.axes import OrdinalAxis
from abtem.core.backend import copy_to_device

from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable


if TYPE_CHECKING:
    from abtem.waves.scan import AbstractScan
    from abtem.measure.measure import (
        AbstractMeasurement,
        LineProfiles,
        FourierSpaceLineProfiles,
    )

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


def _plane2axes(plane):
    axes = ()
    last_axis = [0, 1, 2]
    for axis in list(plane):
        if axis == "x":
            axes += (0,)
            last_axis.remove(0)
        if axis == "y":
            axes += (1,)
            last_axis.remove(1)
        if axis == "z":
            axes += (2,)
            last_axis.remove(2)
    return axes + (last_axis[0],)


def merge_columns(atoms: Atoms, tol: float = 1e-7) -> Atoms:
    pass

def show_atoms(
    atoms: Atoms,
    repeat: Tuple[int, int] = (1, 1),
    scans: List["AbstractScan"] = None,
    plane: Union[Tuple[float, float], str] = "xy",
    ax: Axes = None,
    scale_atoms: float = 0.5,
    title: str = None,
    numbering: bool = False,
    figsize: Tuple[int, int] = None,
    legend: bool = False,
):
    """
    Display 2d projection of atoms as a Matplotlib plot.

    Parameters
    ----------
    atoms : ASE Atoms
        The atoms to be shown.
    repeat : two int, optional
        Tiling of the atoms. Default is (1,1), i.e. no tiling.
    scans : ndarray, optional
        List of scans to apply. Default is None.
    plane : str, two float
        The projection plane given as a combination of 'x' 'y' and 'z', e.g. 'xy', or the as two floats representing the
        azimuth and elevation angles in degrees of the viewing direction, e.g. (45, 45).
    ax : matplotlib Axes, optional
        If given the plots are added to the axes.
    scale_atoms : float
        Scaling factor for the atom display sizes. Default is 0.5.
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Show the index of the Atoms on the plot. Default is False.
    figsize : two int, optional
        The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
    legend : bool
        If True, add a legend
    """

    atoms = atoms.copy()
    atoms *= repeat + (1,)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cell = atoms.cell
    axes = _plane2axes(plane)

    for line in _cube:
        cell_lines = np.array([np.dot(line[0], cell), np.dot(line[1], cell)])
        ax.plot(cell_lines[:, axes[0]], cell_lines[:, axes[1]], "k-")

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(atoms.positions[:, axes[2]])
        positions = positions[order]

        colors = jmol_colors[atoms.numbers[order]]
        sizes = covalent_radii[atoms.numbers[order]] * scale_atoms

        circles = []
        for position, size in zip(positions, sizes):
            circles.append(Circle(position, size))

        coll = PatchCollection(circles, facecolors=colors, edgecolors="black")
        ax.add_collection(coll)

        ax.axis("equal")
        ax.set_xlabel(plane[0] + " [Å]")
        ax.set_ylabel(plane[1] + " [Å]")

        ax.set_title(title)

        if numbering:
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

        ax.legend(handles=legend_elements)

    if scans is not None:
        if not isinstance(scans, Iterable):
            scans = [scans]

        for scan in scans:
            scan.add_to_mpl_plot(ax)

    return ax


def add_imshow(
    ax,
    array,
    extent,
    title,
    x_label,
    y_label,
    x_ticks,
    y_ticks,
    power,
    vmin,
    vmax,
    domain_coloring_kwargs=None,
    **kwargs,
):
    if power != 1:
        array = array ** power
        vmin = array.min() if vmin is None else vmin ** power
        vmax = array.max() if vmax is None else vmax ** power
        # vmin = vmin ** power
        # vmax = vmax ** power

    array = array.T

    if np.iscomplexobj(array):
        if domain_coloring_kwargs is None:
            domain_coloring_kwargs = {}

        array = cplot.get_srgb1(array, **domain_coloring_kwargs)

    im = ax.imshow(array, extent=extent, origin="lower", vmin=vmin, vmax=vmax, **kwargs)

    if title:
        ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if not x_ticks:
        ax.set_xticks([])

    if not y_ticks:
        ax.set_yticks([])

    return im


def add_panel_label(ax, title, **kwargs):
    # if size is None:
    # size = dict(size=plt.rcParams['legend.fontsize'])

    if "loc" not in kwargs:
        kwargs["loc"] = 2

    at = AnchoredText(
        title, pad=0.0, borderpad=0.5, frameon=False, **kwargs  # loc=loc, prop=size,
    )
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def add_sizebar(ax, label, **kwargs):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    if "loc" not in kwargs:
        kwargs["loc"] = 3

    if "borderpad" not in kwargs:
        kwargs["borderpad"] = 0.5

    asb = AnchoredSizeBar(ax.transData, label=label, **kwargs)
    ax.add_artist(asb)


def show_measurement_2d_exploded(
    measurements: "AbstractMeasurement",
    figsize: Tuple[int, int],
    super_title: Union[str, bool],
    sub_title: bool,
    x_label: bool,
    y_label: bool,
    x_ticks: bool,
    y_ticks: bool,
    row_super_label: bool,
    col_super_label: bool,
    power: float,
    vmin: float,
    vmax: float,
    common_color_scale: bool,
    cbar: bool,
    cbar_labels: str,
    float_formatting: str,
    cmap="viridis",
    extent: List[float] = None,
    panel_labels: list = None,
    sizebar=False,
    image_grid_kwargs: dict = None,
    imshow_kwargs: dict = None,
    anchored_text_kwargs: dict = None,
    anchored_size_bar_kwargs: dict = None,
    domain_coloring_kwargs: dict = None,
    axes: Axes = None,
):
    measurements = measurements.to_cpu().compute()

    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
    image_grid_kwargs = {} if image_grid_kwargs is None else image_grid_kwargs
    anchored_text_kwargs = {} if anchored_text_kwargs is None else anchored_text_kwargs
    anchored_size_bar_kwargs = (
        {} if anchored_size_bar_kwargs is None else anchored_size_bar_kwargs
    )

    if domain_coloring_kwargs is None:
        domain_coloring_kwargs = {
            "abs_scaling": lambda x: x / (x + 1),
            "saturation_adjustment": 1.38,
        }

    if cbar and not np.iscomplexobj(measurements.array):
        if common_color_scale:
            image_grid_kwargs["cbar_mode"] = "single"

        else:
            image_grid_kwargs["cbar_mode"] = "each"
            image_grid_kwargs["cbar_pad"] = 0.05

    measurements = measurements[(0,) * max(len(measurements.ensemble_shape) - 2, 0)]

    if common_color_scale and np.iscomplexobj(measurements.array):
        vmin = np.abs(measurements.array).min() if vmin is None else vmin
        vmax = np.abs(measurements.array).max() if vmax is None else vmax
    elif common_color_scale:
        vmin = measurements.array.min() if vmin is None else vmin
        vmax = measurements.array.max() if vmax is None else vmax

    nrows = (
        measurements.ensemble_shape[-2] if len(measurements.ensemble_shape) > 1 else 1
    )
    ncols = (
        measurements.ensemble_shape[-1] if len(measurements.ensemble_shape) > 0 else 1
    )

    if axes is None:
        fig = plt.figure(1, figsize)

        if "axes_pad" not in image_grid_kwargs:
            if sub_title:
                image_grid_kwargs["axes_pad"] = 0.3

            else:
                image_grid_kwargs["axes_pad"] = 0.1

        axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), **image_grid_kwargs,)
    elif isinstance(axes, Axes):
        axes = [axes]
        fig = None

    if x_label is True:
        x_label = measurements.base_axes_metadata[-2].format_label()

    if y_label is True:
        y_label = measurements.base_axes_metadata[-1].format_label()

    for index, measurement in measurements.iterate_ensemble(keep_dims=True):

        title = None
        if len(measurement.ensemble_shape) == 0:
            if isinstance(super_title, str):
                title = super_title
            super_title = False
            i = 0
        elif len(measurement.ensemble_shape) == 1:
            title = (
                measurement.axes_metadata[0].format_title(float_formatting)
                if sub_title
                else None
            )
            i = np.ravel_multi_index((0,) + index, (nrows, ncols))
        elif len(measurement.ensemble_shape) == 2:
            if sub_title:
                titles = tuple(
                    axis.format_title(float_formatting)
                    for axis in measurement.ensemble_axes_metadata
                )
                title = ", ".join(titles)

            i = np.ravel_multi_index(index, (nrows, ncols))
        else:
            raise RuntimeError()

        ax = axes[i]

        if extent is None:
            extent = [0, measurement.extent[0], 0, measurement.extent[1]]

        array = measurement.array[(0,) * len(measurement.ensemble_shape)]

        im = add_imshow(
            ax=ax,
            array=array,
            title=title,
            extent=extent,
            x_label=x_label,
            y_label=y_label,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            power=power,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            domain_coloring_kwargs=domain_coloring_kwargs,
            **imshow_kwargs,
        )

        if panel_labels is not None:
            add_panel_label(ax, panel_labels[i], **anchored_text_kwargs)

        if cbar:
            if cbar_labels is None:
                cbar_label = (
                    measurement.metadata["label"]
                    if "label" in measurement.metadata
                    else ""
                )
                cbar_label += (
                    f" [{measurement.metadata['units']}]"
                    if "units" in measurement.metadata
                    else ""
                )
            else:
                cbar_label = cbar_labels

            if np.iscomplexobj(array):
                from cplot._main import _add_colorbar_arg, _add_colorbar_abs

                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.3)
                cax2 = divider.append_axes("right", size="5%", pad=0.5)

                _add_colorbar_abs(cax2, domain_coloring_kwargs["abs_scaling"], 10)
                _add_colorbar_arg(cax1, domain_coloring_kwargs["saturation_adjustment"])

                # vmin = np.abs(measurement.array).min() if vmin is None else vmin
                # vmax = np.abs(measurement.array).max() if vmax is None else vmax
                # add_domain_coloring_cbar(ax,
                #                         domain_coloring_kwargs['abs_scaling'],
                #                         domain_coloring_kwargs['saturation_adjustment'])
            else:
                try:
                    ax.cax.colorbar(im, label=cbar_label)
                except AttributeError:
                    plt.colorbar(im, ax=ax, label=cbar_label)

        if sizebar:
            size = (
                measurement.base_axes_metadata[-2].sampling
                * measurement.base_shape[-2]
                / 3
            )
            label = (
                f"{size:>{float_formatting}} {measurement.base_axes_metadata[-2].units}"
            )
            add_sizebar(ax=ax, label=label, size=size, **anchored_size_bar_kwargs)

    if super_title is True and "name" in measurements.metadata:
        fig.suptitle(measurements.metadata["name"])

    elif isinstance(super_title, str):

        fig.suptitle(super_title)

    if len(measurements.ensemble_axes_metadata) > 0 and row_super_label:
        fig.supylabel(f"{measurements.ensemble_axes_metadata[-1].format_label()}")

    if len(measurements.ensemble_axes_metadata) > 1 and col_super_label:
        fig.supylabel(f"{measurements.ensemble_axes_metadata[-2].format_label()}")

    # if cbar:
    #     if np.iscomplexobj(array):
    #         vmin = np.abs(array).min() if vmin is None else vmin
    #         vmax = np.abs(array).max() if vmax is None else vmax
    #         add_domain_coloring_cbar(ax, vmin, vmax)
    #     else:
    #         plt.colorbar(im, ax=ax)

    return fig, ax


def add_plot(x: np.ndarray, y: np.ndarray, ax: Axes, label: str = None, **kwargs):
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
    measurements: Union["LineProfiles", "FourierSpaceLineProfiles"],
    extent=None,
    ax=None,
    x_label=None,
    y_label=None,
    title=None,
    figsize=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

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

            if isinstance(axis, OrdinalAxis):
                labels += [f"{axis.values[0]}"]

        label = "-".join(labels)

        add_plot(x, line_profile.array, ax, label)

    ax.set_xlabel(x_label)

    if y_label is None and "label" in measurements.metadata:
        y_label = measurements.metadata["label"]
        if "units" in measurements.metadata:
            y_label += f' [{measurements.metadata["units"]}]'
        ax.set_ylabel(y_label)

    ax.set_ylabel(y_label)

    if len(measurements.ensemble_shape) > 0:
        ax.legend()

    return ax
