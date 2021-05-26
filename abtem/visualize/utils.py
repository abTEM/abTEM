from colorsys import hls_to_rgb

import numpy as np
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


def domain_coloring(z, pure_phase=False):
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
    if not pure_phase:
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
