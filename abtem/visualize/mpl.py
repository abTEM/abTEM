"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from collections.abc import Iterable
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from abtem.visualize.utils import domain_coloring
from abtem.visualize.utils import format_label

#: Array to facilitate the display of cell boundaries.
_cube = np.array([[[0, 0, 0], [0, 0, 1]],
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
                  [[1, 1, 0], [1, 1, 1]]])


def _plane2axes(plane):
    """Internal function for extracting axes from a plane."""
    axes = ()
    last_axis = [0, 1, 2]
    for axis in list(plane):
        if axis == 'x':
            axes += (0,)
            last_axis.remove(0)
        if axis == 'y':
            axes += (1,)
            last_axis.remove(1)
        if axis == 'z':
            axes += (2,)
            last_axis.remove(2)
    return axes + (last_axis[0],)


def label_to_index_generator(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


def merge_close_points(points, distance):
    if len(points) < 2:
        return points, np.arange(len(points))

    clusters = fcluster(linkage(pdist(points), method='complete'), distance, criterion='distance')
    new_points = np.zeros_like(points)
    indices = np.zeros(len(points), dtype=np.int)
    k = 0
    for i, cluster in enumerate(label_to_index_generator(clusters, 1)):
        new_points[i] = np.mean(points[cluster], axis=0)
        indices[i] = np.min(indices)
        k += 1
    return new_points[:k], indices[:k]


def show_atoms(atoms, repeat: Tuple[int, int] = (1, 1), scans=None, plane: Union[Tuple[float, float], str] = 'xy',
               ax=None, scale_atoms: float = .5, title: str = None, numbering: bool = False, figsize=None,
               legend=False):
    """
    Show atoms function

    Function to display atoms, especially in Jupyter notebooks.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms to be shown.
    repeat : two ints, optional
        Tiling of the image. Default is (1,1), ie. no tiling.
    scans : ndarray, optional
        List of scans to apply. Default is None.
    plane : str, two float
        The projection plane given as a combination of 'x' 'y' and 'z', e.g. 'xy', or the as two floats representing the
        azimuth and elevation angles in degrees of the viewing direction, e.g. (45, 45).
    ax : axes object
        pyplot axes object.
    scale_atoms : float
        Scaling factor for the atom display sizes. Default is 0.5.
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Option to set plot numbering. Default is False.
    """

    atoms = atoms.copy()
    atoms *= repeat + (1,)

    if isinstance(plane, str):
        ax = _show_atoms_2d(atoms, scans, plane, ax, scale_atoms, title, numbering, figsize, legend=legend)
    else:
        if scans is not None:
            raise NotImplementedError()

        if numbering:
            raise NotImplementedError()
        ax = _show_atoms_3d(atoms, plane[0], plane[1], scale_atoms=scale_atoms, ax=ax, figsize=figsize)

    return ax


def _show_atoms_2d(atoms, scans=None, plane: Union[Tuple[float, float], str] = 'xy', ax=None, scale_atoms: float = .5,

                   title: str = None, numbering: bool = False, figsize=None, legend=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cell = atoms.cell
    axes = _plane2axes(plane)

    for line in _cube:
        cell_lines = np.array([np.dot(line[0], cell), np.dot(line[1], cell)])
        ax.plot(cell_lines[:, axes[0]], cell_lines[:, axes[1]], 'k-')

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(atoms.positions[:, axes[2]])
        positions = positions[order]

        #distance = .1
        #positions, indices = merge_close_points(positions, distance)


        colors = jmol_colors[atoms.numbers[order]]
        sizes = covalent_radii[atoms.numbers[order]] * scale_atoms

        circles = []
        for position, size in zip(positions, sizes):
            circles.append(Circle(position, size))

        coll = PatchCollection(circles, facecolors=colors, edgecolors='black')
        ax.add_collection(coll)

        ax.axis('equal')
        ax.set_xlabel(plane[0] + ' [Å]')
        ax.set_ylabel(plane[1] + ' [Å]')

        ax.set_title(title)

        if numbering:
            for i, (position, size) in enumerate(zip(positions, sizes)):
                ax.annotate('{}'.format(order[i]), xy=position, ha="center", va="center")

    if legend:
        legend_elements = [Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label=chemical_symbols[unique],
                                  markerfacecolor=jmol_colors[unique], markersize=12)
                           for unique in np.unique(atoms.numbers)]

        ax.legend(handles=legend_elements)

    if scans is not None:
        if not isinstance(scans, Iterable):
            scans = [scans]

        for scan in scans:
            scan.add_to_mpl_plot(ax)

    return ax


def _show_atoms_3d(atoms, azimuth=45., elevation=30., ax=None, scale_atoms=500., margin=1., figsize=None):
    cell = atoms.cell
    colors = jmol_colors[atoms.numbers]
    sizes = covalent_radii[atoms.numbers] ** 2 * scale_atoms
    positions = atoms.positions

    for line in _cube:
        cell_lines = np.array([np.dot(line[0], cell), np.dot(line[1], cell)])
        start = cell_lines[0]
        end = cell_lines[1]
        cell_line_points = start + (end - start)[None] * np.linspace(0, 1, 100)[:, None]
        positions = np.vstack((positions, cell_line_points))
        sizes = np.concatenate((sizes, [1] * len(cell_line_points)))
        colors = np.vstack((colors, [(0, 0, 0)] * len(cell_line_points)))

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d', proj_type='ortho')

    ax.scatter(positions[:, 0],
               positions[:, 1],
               positions[:, 2],
               c=colors,
               marker='o',
               s=sizes,
               alpha=1,
               linewidth=1,
               edgecolor='k')

    xmin = min(min(atoms.positions[:, 0]), min(atoms.cell[:, 0])) - margin
    xmax = max(max(atoms.positions[:, 0]), max(atoms.cell[:, 0])) + margin
    ymin = min(min(atoms.positions[:, 1]), min(atoms.cell[:, 1])) - margin
    ymax = max(max(atoms.positions[:, 1]), max(atoms.cell[:, 1])) + margin
    zmin = min(min(atoms.positions[:, 2]), min(atoms.cell[:, 2])) - margin
    zmax = max(max(atoms.positions[:, 2]), max(atoms.cell[:, 2])) + margin

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')

    ax.grid(False)

    ax.azim = azimuth
    ax.elev = elevation

    ax.set_box_aspect([xmax - xmin, ymax - ymin, zmax - zmin])
    return ax


def show_measurement_2d(measurement,
                        ax=None,
                        figsize=None,
                        cbar=False,
                        cbar_label=None,
                        cmap='gray',
                        discrete_cmap=False,
                        vmin=None,
                        vmax=None,
                        power=1.,
                        log_scale=False,
                        title=None,
                        equal_ticks=False,
                        is_rgb=False,
                        x_label=None,
                        y_label=None,
                        **kwargs):
    """
    Show image function

    Function to display an image.

    Parameters
    ----------
    array : ndarray
        Image array.
    calibrations : tuple of calibration objects.
        Spatial calibrations.
    ax : axes object
        pyplot axes object.
    title : str, optional
        Image title. Default is None.
    colorbar : bool, optional
        Option to show a colorbar. Default is False.
    cmap : str, optional
        Colormap name. Default is 'gray'.
    figsize : float, pair of float, optional
        Size of the figure in inches, either as a square for one number or a rectangle for two. Default is None.
    scans : ndarray, optional
        Array of scans. Default is None.
    discrete : bool, optional
        Option to discretize intensity values to integers. Default is False.
    cbar_label : str, optional
        Text label for the color bar. Default is None.
    vmin : float, optional
        Minimum of the intensity scale. Default is None.
    vmax : float, optional
        Maximum of the intensity scale. Default is None.
    kwargs :
        Remaining keyword arguments are passed to pyplot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if is_rgb:
        calibrations = measurement.calibrations[-3:-1]
    else:
        calibrations = measurement.calibrations[-2:]

    if not is_rgb:
        array = measurement.array[(0,) * (measurement.dimensions - 2) + (slice(None),) * 2]
    else:
        array = measurement.array[:, :, :]

    if np.iscomplexobj(array):
        array = domain_coloring(array)

    if power != 1:
        array = array ** power

    if log_scale:
        array = np.log(array)

    extent = []
    for calibration, num_elem in zip(calibrations, array.shape):
        extent.append(calibration.offset)
        extent.append(calibration.offset + num_elem * calibration.sampling - calibration.sampling)

    if vmin is None:
        vmin = np.min(array)
        if discrete_cmap:
            vmin -= .5

    if vmax is None:
        vmax = np.max(array)
        if discrete_cmap:
            vmax += .5

    if discrete_cmap:
        cmap = plt.get_cmap(cmap, np.max(array) - np.min(array) + 1)

    im = ax.imshow(np.swapaxes(array, 0, 1), extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   interpolation='nearest',
                   **kwargs)

    if cbar:
        if cbar_label is None:
            cbar_label = format_label(measurement)

        cax = plt.colorbar(im, ax=ax, label=cbar_label)
        if discrete_cmap:
            cax.set_ticks(ticks=np.arange(np.min(array), np.max(array) + 1))

    if x_label is None:
        x_label = format_label(calibrations[-2])

    if y_label is None:
        y_label = format_label(calibrations[-1])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)
    elif len(measurement.array.shape) > 2:
        if any([n > 1 for n in measurement.array.shape[:-2]]):
            ax.set_title(f'Slice {(0,) * (len(measurement.array.shape) - 2)} of {measurement.array.shape} measurement')

    if equal_ticks:
        d = max(np.diff(ax.get_xticks())[0], np.diff(ax.get_yticks())[0])
        xticks = np.arange(*ax.get_xlim(), d)
        yticks = np.arange(*ax.get_ylim(), d)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    return ax, im


def show_measurement_1d(measurement, ax=None, figsize=None, legend=False, title=None, label=None,
                        x_label=None, y_label=None, **kwargs):
    """
    Show line function

    Function to display a line scan.

    Parameters
    ----------
    array : ndarray
        Array of measurement values along a line.
    calibration : calibration object
        Spatial calibration for the line.
    ax : axes object, optional
        pyplot axes object.
    title : str, optional
        Title for the plot. Default is None.
    legend : bool, optional
        Option to display a plot legend. Default is False.
    kwargs :
       Remaining keyword arguments are passed to pyplot.
    """

    calibration = measurement.calibrations[0]
    array = measurement.array
    if calibration is None:
        x = np.arange(len(array))
    else:
        x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not label:
        label = measurement.name

    lines = ax.plot(x, array, label=label, **kwargs)

    if x_label is None:
        x_label = format_label(calibration)

    if y_label is None:
        y_label = format_label(measurement)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if legend:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    return ax, lines[0]
