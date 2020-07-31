from collections import Iterable
from colorsys import hls_to_rgb

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from abtem.cpu_kernels import abs2
from scipy.ndimage import gaussian_filter

cube = np.array([[[0, 0, 0], [0, 0, 1]],
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


def plane2axes(plane):
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


def show_atoms(atoms, repeat=(1, 1), scans=None, plane='xy', ax=None, scale_atoms=.5, numbering=False):
    if ax is None:
        fig, ax = plt.subplots()

    axes = plane2axes(plane)

    atoms = atoms.copy()
    cell = atoms.cell
    atoms *= repeat + (1,)

    for line in cube:
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

        if numbering:
            for i, (position, size) in enumerate(zip(positions, sizes)):
                ax.annotate('{}'.format(order[i]), xy=position, ha="center", va="center")

    if scans is not None:
        if not isinstance(scans, Iterable):
            scans = [scans]

        for scan in scans:
            scan.add_to_mpl_plot(ax)


def show_image(array, calibrations, ax=None, title=None, colorbar=False, cmap='gray', figsize=None, scans=None,
               log_scale=False, discrete=False, cbar_label=None, vmin=None, vmax=None, **kwargs):

    if np.iscomplexobj(array):
        array = abs2(array)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        array = np.log(array)

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


def show_line(array, calibration, ax=None, title=None, **kwargs):
    x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))

    if ax is None:
        ax = plt.subplot()

    ax.plot(x, array, **kwargs)
    ax.set_xlabel('{} [{}]'.format(calibration.name, calibration.units))

    if title is not None:
        ax.set_title(title)

    return ax


def domain_coloring(z, fade_to_white=False, saturation=1, k=.5):
    h = (np.angle(z) + np.pi) / (2 * np.pi) + 0.5
    if fade_to_white:
        l = k ** np.abs(z)
    else:
        l = 1 - k ** np.abs(z)
    c = np.vectorize(hls_to_rgb)(h, l, saturation)
    c = np.array(c).T

    c = (c - c.min()) / c.ptp()
    return c
