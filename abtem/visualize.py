"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from typing import TYPE_CHECKING, List
from typing import Union, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from cplot._colors import get_srgb1
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core.backend import copy_to_device
from abtem.core.indexing import map_all_bin_indices_to_miller_indices
from abtem.core.indexing import (
    miller_to_miller_bravais,
    find_equivalent_spots,
    validate_cell_edges,
)
from abtem.core.utils import label_to_index
from abtem.atoms import pad_atoms, plane_to_axes

if TYPE_CHECKING:
    from abtem.measurements import (
        DiffractionPatterns,
        BaseMeasurement,
        RealSpaceLineProfiles,
        FourierSpaceLineProfiles,
    )
    from abtem.core.indexing import IndexedDiffractionPattern


def _get_complex_colors(
    z: np.ndarray, vmin: float = None, vmax: float = None, saturation: float = 2.0
):
    abs_z = np.abs(z)
    angle_z = np.angle(z)

    if vmin is None:
        vmin = abs_z.min()

    if vmax is None:
        vmax = abs_z.max()

    vmin_rel = (vmin - abs_z.min()) / abs_z.ptp()
    vmax_rel = (vmax - abs_z.max()) / abs_z.ptp()

    abs_scaled = (abs_z - abs_z.min()) / abs_z.ptp() * (
        1 - vmax_rel - vmin_rel
    ) + vmin_rel

    abs_scaled = np.clip(abs_scaled, 0, np.inf)
    z_scaled = abs_scaled * np.exp(1.0j * angle_z)

    return get_srgb1(z_scaled, lambda x: x, saturation)


def _add_colorbar_abs(cax, vmin, vmax):
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray), cax=cax,)
    cb0.set_label("abs", rotation=0, ha="center", va="top")
    cb0.ax.yaxis.set_label_coords(0.5, -0.03)


def _add_colorbar_arg(cax, saturation_adjustment: float):
    z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
    rgb_vals = get_srgb1(
        z,
        abs_scaling=lambda z: np.full_like(z, 0.5),
        saturation_adjustment=saturation_adjustment,
    )
    rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
    newcmp = colors.ListedColormap(rgba_vals)
    #
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)

    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp), cax=cax)

    cb1.set_label("arg", rotation=0, ha="center", va="top")
    cb1.ax.yaxis.set_label_coords(0.5, -0.03)
    cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
    cb1.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )


def _add_imshow(
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

        array = _get_complex_colors(
            array, vmin=vmin, vmax=vmax, **complex_coloring_kwargs
        )

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


def _add_panel_label(ax, title, **kwargs):
    if "loc" not in kwargs:
        kwargs["loc"] = 2

    at = AnchoredText(
        title, pad=0.0, borderpad=0.5, frameon=False, **kwargs  # loc=loc, prop=size,
    )
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def _add_sizebar(ax, label, **kwargs):
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


def _make_cbar_label(measurement):
    cbar_label = (
        measurement.metadata["label"] if "label" in measurement.metadata else ""
    )
    cbar_label += (
        f" [{measurement.metadata['units']}]" if "units" in measurement.metadata else ""
    )
    return cbar_label


def show_measurement_2d(
    measurements: "BaseMeasurement",
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
    cmap: str = "viridis",
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

        im = _add_imshow(
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
            _add_panel_label(ax, panel_labels[i], **anchored_text_kwargs)

        if cbar:
            if cbar_labels is None:
                cbar_label = _make_cbar_label(measurement)
            else:
                cbar_label = cbar_labels

            if np.iscomplexobj(array):
                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.2)
                cax2 = divider.append_axes("right", size="5%", pad=0.4)

                _add_colorbar_arg(cax1, complex_coloring_kwargs["saturation"])

                vmin = np.abs(measurement.array).min() if vmin is None else vmin
                vmax = np.abs(measurement.array).max() if vmax is None else vmax

                _add_colorbar_abs(cax2, vmin, vmax)
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
            _add_sizebar(ax=ax, label=label, size=size, **anchored_size_bar_kwargs)

    if super_title is True and "name" in measurements.metadata:
        fig.suptitle(measurements.metadata["name"])

    elif isinstance(super_title, str):

        fig.suptitle(super_title)

    if len(measurements.ensemble_axes_metadata) > 0 and row_super_label:
        fig.supylabel(f"{measurements.ensemble_axes_metadata[-1].format_label()}")

    if len(measurements.ensemble_axes_metadata) > 1 and col_super_label:
        fig.supylabel(f"{measurements.ensemble_axes_metadata[-2].format_label()}")

    return fig, ax


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
    measurements: Union["RealSpaceLineProfiles", "FourierSpaceLineProfiles"],
    float_formatting: str,
    extent: Tuple[float, float] = None,
    ax: Axes = None,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    figsize: Tuple[int, int] = None,
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
            labels += [axis.format_title(float_formatting)]

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


def plot_diffraction_pattern(
    indexed_diffraction_pattern: "IndexedDiffractionPattern",
    # cell: Union[float, Tuple[float, float], Tuple[float, float, float], Cell],
    spot_scale: float = 1.0,
    ax: Axes = None,
    figsize: Tuple[float, float] = (6, 6),
    spot_threshold: float = 0.02,
    title: str = None,
    annotate_kwargs: dict = None,
    divide_threshold: float = 1.0,
):

    if annotate_kwargs is None:
        annotate_kwargs = {}

    coordinates = indexed_diffraction_pattern._vectors
    normalize_coordinates = coordinates.max()

    coordinates = coordinates / normalize_coordinates

    max_step = (
        max([np.max(np.diff(np.sort(coordinates[:, i]))) for i in (0, 1)]) * spot_scale
    )
    intensities = indexed_diffraction_pattern.intensities

    scales = intensities / intensities.max() * max_step

    norm = matplotlib.colors.Normalize(vmin=0, vmax=intensities.max())
    cmap = matplotlib.cm.get_cmap("viridis")

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
            facecolors=cmap(norm(intensities)),
            offsets=coordinates,
            transOffset=ax.transData,
        )
    )

    indexed_diffraction_pattern = indexed_diffraction_pattern.remove_equivalent(divide_threshold=divide_threshold)
    coordinates = indexed_diffraction_pattern._vectors
    coordinates = coordinates / normalize_coordinates

    miller_indices = indexed_diffraction_pattern.miller_indices

    for hkl, coordinate in zip(miller_indices, coordinates):
        # if include[i]:  # or label_mode == "all":
        # if hexagonal:
        #     spot = miller_to_miller_bravais(hkl[i][None])[0]
        # else:
        #     spot = hkl[i]
        # print(hkl, coordinate)

        t = ax.annotate(
            "".join(map(str, list(hkl))),
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

    return fig, ax


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
        top_atom = np.argmax(positions[label][:, axes[2]])
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
):
    """
    Display 2d projection of atoms as a Matplotlib plot.

    Parameters
    ----------
    atoms : ASE Atoms
        The atoms to be shown.
    plane : str, two float
        The projection plane given as a combination of 'x' 'y' and 'z', e.g. 'xy', or the as two floats representing the
        azimuth and elevation angles in degrees of the viewing direction, e.g. (45, 45).
    ax : matplotlib Axes, optional
        If given the plots are added to the axes.
    scale : float
        Scaling factor for the atom display sizes. Default is 0.5.
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Display the index of the Atoms as a number. Default is False.
    show_periodic : bool
        If True, show the periodic images of the atoms at the cell boundary.
    figsize : two int, optional
        The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
    legend : bool
        If True, add a legend indicating the color of the atomic species.
    merge: float
        Plotting large numbers of atoms can be slow. To speed up plotting atoms closer than the given value
        (in Ångstrom) are merged.
    """

    if show_periodic:
        atoms = atoms.copy()
        atoms = pad_atoms(atoms, margins=1e-3)

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

    for cell_line_x, cell_line_y in zip(cell_lines_x, cell_lines_y):
        ax.plot(cell_line_x, cell_line_y, "k-")

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(atoms.positions[:, axes[2]])
        positions = positions[order]

        colors = jmol_colors[atoms.numbers[order]]
        sizes = covalent_radii[atoms.numbers[order]] * scale

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

        ax.legend(handles=legend_elements, loc="upper right")
    # ax.set_xlim([0, np.max(cell_lines_x)])
    # ax.set_ylim([0, np.max(cell_lines_y)])

    return fig, ax
