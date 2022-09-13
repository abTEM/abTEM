"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from typing import Union, Tuple, TYPE_CHECKING, List

import cplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core.backend import copy_to_device
from abtem.measurements.indexing import (
    miller_to_miller_bravais,
    find_equivalent_spots,
    validate_cell_edges,
)
from abtem.visualize.complex_plot import get_colors, add_colorbar_arg, add_colorbar_abs

if TYPE_CHECKING:
    from abtem.measurements.core import (
        Measurement,
        RealSpaceLineProfiles,
        FourierSpaceLineProfiles,
    )


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
    complex_coloring_kwargs=None,
    **kwargs,
):
    if power != 1:
        array = array ** power
        vmin = array.min() if vmin is None else vmin ** power
        vmax = array.max() if vmax is None else vmax ** power

    array = array.T

    if np.iscomplexobj(array):
        if complex_coloring_kwargs is None:
            complex_coloring_kwargs = {}

        array = get_colors(array, vmin=vmin, vmax=vmax, **complex_coloring_kwargs)

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
    measurements: "Measurement",
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
    sizebar: bool = False,
    image_grid_kwargs: dict = None,
    imshow_kwargs: dict = None,
    anchored_text_kwargs: dict = None,
    anchored_size_bar_kwargs: dict = None,
    complex_coloring_kwargs: dict = None,
    axes: Axes = None,
):
    measurements = measurements.to_cpu().compute()

    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
    image_grid_kwargs = {} if image_grid_kwargs is None else image_grid_kwargs
    anchored_text_kwargs = {} if anchored_text_kwargs is None else anchored_text_kwargs
    anchored_size_bar_kwargs = (
        {} if anchored_size_bar_kwargs is None else anchored_size_bar_kwargs
    )

    if complex_coloring_kwargs is None:
        complex_coloring_kwargs = {
            "saturation": 3,
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
        fig = plt.figure(1, figsize, clear=True)

        if "axes_pad" not in image_grid_kwargs:
            if sub_title:
                image_grid_kwargs["axes_pad"] = 0.3

            else:
                image_grid_kwargs["axes_pad"] = 0.1

        axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), **image_grid_kwargs,)
    elif isinstance(axes, Axes):
        fig = axes.get_figure()
        axes = [axes]

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
            complex_coloring_kwargs=complex_coloring_kwargs,
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
                #print(ax)
                #divider = make_axes_locatable(axes)
                bbox_ax = ax.get_position()
                #print(ax.)

                # fig.add_axes() adds the colorbar axes
                # they're bounded by [x0, y0, x_width, y_width]
                cax1 = fig.add_axes([1.01, bbox_ax.y0, 0.02, bbox_ax.y1 - bbox_ax.y0])
                #cbar_im1a = plt.colorbar(im1a, cax=cbar_im1a_ax)

                #cax1 = divider.append_axes("right", size="5%", pad=0.2)
                #cax2 = divider.append_axes("right", size="5%", pad=0.4)

                add_colorbar_arg(cax1, complex_coloring_kwargs["saturation"])

                #vmin = np.abs(measurement.array).min() if vmin is None else vmin
                #vmax = np.abs(measurement.array).max() if vmax is None else vmax

                #add_colorbar_abs(cax2, vmin, vmax)

                # _add_colorbar_abs(cax2, domain_coloring_kwargs["abs_scaling"], 10)
                # _add_colorbar_arg(cax1, domain_coloring_kwargs["saturation_adjustment"])

                #
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
    measurements: Union["RealSpaceLineProfiles", "FourierSpaceLineProfiles"],
    float_formatting: str,
    extent=None,
    ax: Axes = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    figsize: Tuple[int, int] = None,
    **kwargs,
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
            labels += [axis.format_title(float_formatting)]

        label = "-".join(labels)

        add_plot(x, line_profile.array, ax, label, **kwargs)

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


def plot_diffraction_pattern(
    diffraction_pattern,
    cell,
    spot_scale: float = 1,
    ax: Axes = None,
    figsize=(6, 6),
    spot_threshold: float = 0.02,
    title=None,
    annotate_kwargs=None,
    intensity_split: float = 1.0,
):
    from abtem.measurements.indexing import map_all_bin_indices_to_miller_indices
    import matplotlib

    if annotate_kwargs is None:
        annotate_kwargs = {}

    bins, hkl = map_all_bin_indices_to_miller_indices(
        diffraction_pattern.array, diffraction_pattern.sampling, cell,
    )
    intensities = diffraction_pattern.select_frequency_bin(bins)
    max_intensity = intensities.max()

    # include = intensities > max_intensity * spot_threshold

    # bins, hkl, intensities = bins[include], hkl[include], intensities[include]

    coordinates = bins * diffraction_pattern.sampling
    coordinates = coordinates / coordinates.max()

    max_step = (
        max([np.max(np.diff(np.sort(coordinates[:, i]))) for i in (0, 1)]) * spot_scale
    )
    scales = intensities / intensities.max() * max_step

    norm = matplotlib.colors.Normalize(vmin=0, vmax=intensities.max())
    cmap = matplotlib.cm.get_cmap("viridis")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if title:
        ax.set_title(title)

    ax.add_collection(
        EllipseCollection(
            widths=scales,
            heights=scales,
            angles=0.0,
            units="xy",
            facecolors=cmap(norm(intensities)),
            offsets=coordinates,
            transOffset=ax.transData,
        )
    )

    _, hexagonal = validate_cell_edges(cell)

    include = find_equivalent_spots(
        hkl, intensities, hexagonal=hexagonal, intensity_split=intensity_split
    )

    for i, coordinate in enumerate(coordinates):
        if include[i]:  # or label_mode == "all":
            if hexagonal:
                spot = miller_to_miller_bravais(hkl[i][None])[0]
            else:
                spot = hkl[i]

            t = ax.annotate(
                "".join(map(str, list(spot))),
                coordinate,
                ha="center",
                va="center",
                size=12,
                **annotate_kwargs,
            )
            t.set_path_effects([withStroke(foreground="w", linewidth=3)])

    ax.axis("equal")
    ax.set_xlim([-1.0 - max_step / 2.0, 1.0 + max_step / 2.0])
    ax.set_ylim([-1.0 - max_step / 2.0, 1.0 + max_step / 2.0])
    ax.axis("off")

    return ax
