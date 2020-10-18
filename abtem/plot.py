"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from collections.abc import Iterable

from numbers import Number
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from abtem.cpu_kernels import abs2
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

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

        colors = cpk_colors[atoms.numbers[order]]
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


def show_image(array, calibrations, ax=None, title=None, colorbar=False, cmap='gray', figsize=None, scans=None,
               log_scale=False, discrete=False, cbar_label=None, vmin=None, vmax=None, power=1,**kwargs):
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
    log_scale : bool, optional
        Option to set a logarithmic intensity scale. Default is False.
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

    if np.iscomplexobj(array):
        array = abs2(array)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        if isinstance(log_scale, Number) & (not isinstance(log_scale, bool)):
            array = np.log(1 + log_scale * array)
        else:
            array = np.log(array)

    if power != 1.:
        array = array ** power

    extent = []
    for calibration, num_elem in zip(calibrations, array.shape):
        extent.append(calibration.offset)
        extent.append(calibration.offset + num_elem * calibration.sampling)

    if vmin is None:
        vmin = np.min(array)
        if discrete:
            vmin -= .5

    if vmax is None:
        vmax = np.max(array)
        if discrete:
            vmax += .5

    if discrete:
        cmap = plt.get_cmap(cmap, np.max(array) - np.min(array) + 1)

    im = ax.imshow(array.T, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, **kwargs)

    if colorbar:
        cax = plt.colorbar(im, ax=ax, label=cbar_label)
        if discrete:
            cax.set_ticks(ticks=np.arange(np.min(array), np.max(array) + 1))

    ax.set_xlabel('{} [{}]'.format(calibrations[0].name, calibrations[0].units))
    ax.set_ylabel('{} [{}]'.format(calibrations[1].name, calibrations[1].units))

    if title is not None:
        ax.set_title(title)

    if scans is not None:
        if not isinstance(scans, Iterable):
            scans = [scans]

        for scan in scans:
            scan.add_to_mpl_plot(ax)

    return ax, im


def show_line(array, calibration, ax=None, title=None, legend=False, **kwargs):
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

    x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))

    if ax is None:
        ax = plt.subplot()

    ax.plot(x, array, **kwargs)
    ax.set_xlabel('{} [{}]'.format(calibration.name, calibration.units))

    if title is not None:
        ax.set_title(title)

    if legend:
        ax.legend()

    return ax


class PlotableMixin:

    @abstractmethod
    def add_to_bokeh_plot(self, p, *args, **kwargs):
        pass

    def show_bokeh(self, p=None, push_notebook=False, **kwargs):
        from bokeh import plotting
        from bokeh.io import show

        if push_notebook:
            from bokeh.io import push_notebook

        if p is None:
            p = plotting.Figure(plot_width=300, plot_height=300)

        self.add_to_bokeh_plot(p, push_notebook=push_notebook, **kwargs)

        show(p, notebook_handle=push_notebook)
        return p

    @abstractmethod
    def add_to_mpl_plot(self, *args, **kwargs):
        pass

    def show(self, ax=None, *args, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplot()
        self.add_to_mpl_plot(ax=ax, **kwargs)
        plt.show()
        return ax
