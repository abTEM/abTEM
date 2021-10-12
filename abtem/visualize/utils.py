import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from abtem.visualize.colors import hsluv


def format_label(calibration):
    if calibration is None:
        return ''

    label = ''
    if calibration.name:
        label += f'{calibration.name}'

    if calibration.units:
        label += f' [{calibration.units}]'

    return label


def add_domain_coloring_cbar(ax, abs_range, aspect=4):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="30%", pad=0.3, aspect=aspect)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')

    cbar_array = np.linspace(0, 1, 100)
    cbar_array = cbar_array[None] * np.exp(1.j * np.linspace(-np.pi, np.pi, 100))[:, None]

    cax.set_xticks([0, 99])
    cax.set_xticklabels([f'{n:.1e}' for n in abs_range])
    cax.set_yticks(np.linspace(0, 99, 5))
    cax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cax.set_xlabel('abs')
    cax.set_ylabel('arg')

    cax.imshow(domain_coloring(cbar_array), aspect=aspect, origin='lower')


def domain_coloring(z):
    """
    Domain coloring function.

    Function to color a complex domain.

    Parameters
    ----------
    z : ndarray, complex
        Complex number to be colored.
    saturation : float, optional
        RGB color saturation. Default is 1.0.
    k : float, optional
        Scaling factor for the coloring. Default is 0.5.
    """

    phase = (np.angle(z) + np.pi) / (2 * np.pi)

    cmap = ListedColormap(hsluv)
    colors = cmap(phase)[..., :3]

    abs_z = np.abs(z)
    abs_z = (abs_z - abs_z.min()) / abs_z.ptp()
    colors = colors * abs_z[..., None]

    return colors


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
