"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from collections.abc import Iterable
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

#: Array to facilitate the display of cell boundaries.
from abtem.structures.structures import pad_atoms

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


# def show_atoms(atoms,
#                repeat: Tuple[int, int] = (1, 1),
#                scans=None,
#                plane: Union[Tuple[float, float], str] = 'xy',
#                ax=None,
#                scale: float = .75,
#                title: str = None,
#                numbering: bool = False,
#                figsize=None,
#                legend: bool = False):
#
#     atoms = atoms.copy()
#     atoms *= repeat + (1,)
#
#     if isinstance(plane, str):
#         ax = _show_atoms_2d(atoms, scans, plane, ax, scale, title, numbering, figsize, legend=legend)
#     else:
#         if scans is not None:
#             raise NotImplementedError()
#
#         if numbering:
#             raise NotImplementedError()
#         ax = _show_atoms_3d(atoms, plane[0], plane[1], scale_atoms=scale, ax=ax, figsize=figsize)
#
#     return ax


def show_atoms(atoms,
               plane: Union[Tuple[float, float], str] = 'xy',
               ax: Axes = None,
               scale: float = .75,
               title: str = None,
               numbering: bool = False,
               show_periodic: bool = False,
               figsize: Tuple[float, float] = None,
               legend: bool = False):
    """
    Display atoms using matplotlib especially in Jupyter notebooks.

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

    if show_periodic:
        atoms = atoms.copy()
        atoms = pad_atoms(atoms, margins=1e-3)

    # wrapped = atoms[wrap]
    # wrapped.set_scaled_positions(wrapped.get_scaled_positions() + shift)
    # atoms = atoms + wrapped

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cell = atoms.cell
    axes = _plane2axes(plane)

    cell_lines = np.array([[np.dot(line[0], cell), np.dot(line[1], cell)] for line in _cube])
    cell_lines_x, cell_lines_y = cell_lines[..., axes[0]], cell_lines[..., axes[1]]

    for cell_line_x, cell_line_y in zip(cell_lines_x, cell_lines_y):
        ax.plot(cell_line_x, cell_line_y, 'k-')

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(atoms.positions[:, axes[2]])
        positions = positions[order]

        colors = jmol_colors[atoms.numbers[order]]
        sizes = covalent_radii[atoms.numbers[order]] * scale

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

        ax.legend(handles=legend_elements, loc='upper right')

    # ax.set_xlim([0, np.max(cell_lines_x)])
    # ax.set_ylim([0, np.max(cell_lines_y)])

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
