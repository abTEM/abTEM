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
from abtem.structures.transform import pad_atoms, plane_to_axes
from ase import Atoms

from abtem.structures.utils import label_to_index_generator

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


def merge_columns(atoms: Atoms, plane, tol: float = 1e-7) -> Atoms:
    uniques, labels = np.unique(atoms.numbers, return_inverse=True)

    new_atoms = Atoms(cell=atoms.cell)
    for unique, indices in zip(uniques, label_to_index_generator(labels)):
        positions = atoms.positions[indices]
        positions = merge_positions(positions, plane, tol)
        numbers = np.full((len(positions),), unique)
        new_atoms += Atoms(positions=positions, numbers=numbers)

    return new_atoms


def merge_positions(positions, plane, tol: float = 1e-7) -> np.ndarray:
    axes = plane_to_axes(plane)
    rounded_positions = tol * np.round(positions[:, axes[:2]] / tol)
    unique, labels = np.unique(rounded_positions, axis=0, return_inverse=True)

    new_positions = np.zeros((len(unique), 3))
    for i, label in enumerate(label_to_index_generator(labels)):
        top_atom = np.argmax(positions[label][:, axes[2]])
        new_positions[i] = positions[label][top_atom]
        #new_positions[i, axes[2]] = np.max(positions[label][top_atom, axes[2]])

    return new_positions


def show_atoms(
    atoms,
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

    atoms = merge_columns(atoms, plane, merge)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

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

    return ax
