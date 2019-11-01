import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from matplotlib.patches import Rectangle


def plane2axes(plane):
    axes = ()
    for axis in list(plane):
        if axis == 'x': axes += (0,)
        if axis == 'y': axes += (1,)
        if axis == 'z': axes += (2,)
    return axes


def plot_atoms(atoms, scan_area=None, plane='xy', ax=None, scale=100):
    if ax is None:
        fig, ax = plt.subplots()

    axes = plane2axes(plane)
    origin = np.array([0., 0.])
    cell = np.diag(atoms.cell)

    edges = np.zeros((2, 5))
    edges[0, :] += origin[axes[0]]
    edges[1, :] += origin[axes[1]]
    edges[0, 2:4] += np.array([cell[0], cell[1], cell[2]])[axes[0]]
    edges[1, 1:3] += np.array([cell[0], cell[1], cell[2]])[axes[1]]

    ax.plot(edges[0, :], edges[1, :], 'k-')

    if len(atoms) > 0:
        positions = atoms.positions[:, axes]
        colors = cpk_colors[atoms.numbers]
        sizes = covalent_radii[atoms.numbers]

        ax.scatter(*positions.T, c=colors, s=scale * sizes, linewidths=1, edgecolor='k')
        ax.axis('equal')

    if scan_area is not None:
        ax.add_patch(Rectangle(xy=scan_area[0],
                               width=scan_area[1][0] - scan_area[0][0],
                               height=scan_area[1][1] - scan_area[0][1], alpha=.25, color='c'))
