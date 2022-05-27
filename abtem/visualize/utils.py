from colorsys import hls_to_rgb

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpn, interp1d
from abtem.visualize.colors import hsluv
from matplotlib.colors import ListedColormap


def format_label(calibration):
    if calibration is None:
        return ''

    label = ''
    if calibration.name:
        label += f'{calibration.name}'

    if calibration.units:
        label += f' [{calibration.units}]'

    return label


def domain_coloring(z: np.ndarray, vmin: float = None, vmax: float = None):
    """
    Domain coloring function.

    Function to color a complex values.

    Parameters
    ----------
    z : ndarray, complex
        Complex number to be colored.
    vmin, vmax : scalar, optional
        Define the range of absolute values that the colormap covers. By default, the colormap covers the complete
        value range of the absolute values.
    """

    phase = (np.angle(z) + np.pi) / (2 * np.pi)

    cmap = ListedColormap(hsluv)
    colors = cmap(phase)[..., :3]

    abs_z = np.abs(z)

    if vmin is None:
        vmin = abs_z.min()

    if vmax is None:
        vmax = abs_z.max()

    abs_z = (abs_z - vmin) / (vmax - vmin)
    abs_z = np.clip(abs_z, a_min=0, a_max=1.)
    colors = colors * abs_z[..., None]
    return colors


def add_domain_coloring_cbar(ax, vmin, vmax, aspect=2):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="60%", pad=0.3, aspect=aspect)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')

    cbar_array = np.linspace(vmin, vmax, 100)
    cbar_array = cbar_array[:, None] * np.exp(1.j * np.linspace(-np.pi, np.pi, 100))[None]

    cax.set_yticks(np.linspace(0, 99, 5))
    # cax.set_yticklabels([f'{vmin:.1e}', f'{vmax:.1e}'])
    cax.set_yticklabels([f'{v:.2e}' for v in np.linspace(vmin, vmax, 5)])
    cax.set_xticks(np.linspace(0, 99, 3))
    # cax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cax.set_xticklabels(["-π", "0", "π"])
    cax.set_xlabel('arg')
    cax.set_ylabel('abs')
    cax.imshow(domain_coloring(cbar_array, vmin=vmin, vmax=vmax), aspect=aspect, origin='lower')


def _line_intersect_rectangle(point0, point1, lower_corner, upper_corner):
    if point0[0] == point1[0]:
        return (point0[0], lower_corner[1]), (point0[0], upper_corner[1])

    m = (point1[1] - point0[1]) / (point1[0] - point0[0])

    def y(x):
        return m * (x - point0[0]) + point0[1]

    def x(y):
        return (y - point0[1]) / m + point0[0]

    if y(0) < lower_corner[1]:
        intersect0 = (x(lower_corner[1]), y(x(lower_corner[1])))
    else:
        intersect0 = (0, y(lower_corner[0]))

    if y(upper_corner[0]) > upper_corner[1]:
        intersect1 = (x(upper_corner[1]), y(x(upper_corner[1])))
    else:
        intersect1 = (upper_corner[0], y(upper_corner[0]))

    return intersect0, intersect1
