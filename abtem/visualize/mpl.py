"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
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


def show_atoms(atoms, repeat=(1, 1), scans=None, plane='xy', ax=None, scale_atoms=.5, title=None, numbering=False):
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
    plane : str
        The projection plane.
    ax : axes object
        pyplot axes object.
    scale_atoms : float
        Scaling factor for the atom display sizes. Default is 0.5.
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Option to set plot numbering. Default is False.
    """

    if ax is None:
        fig, ax = plt.subplots()

    axes = _plane2axes(plane)

    atoms = atoms.copy()
    cell = atoms.cell
    atoms *= repeat + (1,)

    for line in _cube:
        cell_lines = np.array([np.dot(line[0], cell), np.dot(line[1], cell)])
        ax.plot(cell_lines[:, axes[0]], cell_lines[:, axes[1]], 'k-')

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(atoms.positions[:, axes[2]])
        positions = positions[order]

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

    if scans is not None:
        if not isinstance(scans, Iterable):
            scans = [scans]

        for scan in scans:
            scan.add_to_mpl_plot(ax)


def show_measurement_2d(measurement,
                        ax=None,
                        figsize=None,
                        colorbar=False,
                        cmap='gray',
                        discrete_cmap=False,
                        vmin=None,
                        vmax=None,
                        power=1.,
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

    calibrations = measurement.calibrations[-2:]
    array = measurement.array[(0,) * (measurement.dimensions - 2) + (slice(None),) * 2]

    if power != 1:
        array = array ** power

    extent = []
    for calibration, num_elem in zip(calibrations, array.shape):
        extent.append(calibration.offset)
        extent.append(calibration.offset + num_elem * calibration.sampling)

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

    im = ax.imshow(array.T, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest',
                   **kwargs)

    if colorbar:
        cax = plt.colorbar(im, ax=ax, label=format_label(measurement))
        if discrete_cmap:
            cax.set_ticks(ticks=np.arange(np.min(array), np.max(array) + 1))

    ax.set_xlabel(format_label(calibrations[-2]))
    ax.set_ylabel(format_label(calibrations[-1]))

    return ax, im


def show_measurement_1d(measurement, ax=None, figsize=None, legend=False, title=None, **kwargs):
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
    x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    lines = ax.plot(x, array, label=measurement.name, **kwargs)
    ax.set_xlabel(format_label(calibration))
    ax.set_ylabel(format_label(measurement))

    if legend:
        ax.legend()

    return ax, lines[0]
