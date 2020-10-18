
from colorsys import hls_to_rgb

import numpy as np


def format_label(calibration):
    label = ''
    if calibration.name:
        label += f'{calibration.name}'

    if calibration.units:
        label += f' [{calibration.units}]'

    return label


def domain_coloring(z, fade_to_white=False, saturation=1.0, k=.5):
    """
    Domain coloring function.

    Function to color a complex domain.

    Parameters
    ----------
    z : ndarray, complex
        Complex number to be colored.
    fade_to_white : bool, optional
        Option to fade the coloring to white instead of black. Default is False.
    saturation : float, optional
        RGB color saturation. Default is 1.0.
    k : float, optional
        Scaling factor for the coloring. Default is 0.5.
    """

    h = (np.angle(z) + np.pi) / (2 * np.pi) + 0.5
    if fade_to_white:
        r = k ** np.abs(z)
    else:
        r = 1 - k ** np.abs(z)
    c = np.vectorize(hls_to_rgb)(h, r, saturation)
    c = np.array(c).T

    c = (c - c.min()) / c.ptp()
    return c


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


