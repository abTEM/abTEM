import cplot
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from abtem.visualize.colors import hsluv
from matplotlib.colors import hsv_to_rgb


def add_domain_coloring_cbar(ax, abs_scaling, saturation_adjustment, aspect=2):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="60%", pad=0.3, aspect=aspect)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')

    cbar_array = 1 - abs_scaling(np.linspace(0, 1, 100))[::-1]
    cbar_array = cbar_array[:, None] * np.exp(1.j * np.linspace(-np.pi, np.pi, 100))[None]
    colors = cplot.get_srgb1(cbar_array, abs_scaling=abs_scaling, saturation_adjustment=saturation_adjustment)

    cax.set_yticks(np.linspace(0, 99, 5))
    # cax.set_yticklabels([f'{vmin:.1e}', f'{vmax:.1e}'])
    #cax.set_yticklabels([f'{v:.2e}' for v in np.linspace(vmin, vmax, 5)])
    cax.set_xticks(np.linspace(0, 99, 3))
    # cax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cax.set_xticklabels(["-π", "0", "π"])
    cax.set_xlabel('arg')
    cax.set_ylabel('abs')
    cax.imshow(colors, aspect=aspect, origin='lower')


def domain_coloring(z: np.ndarray,
                    shift_hue: float = np.pi/ 4,
                    lightness_model=None):
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

    hue = (np.angle(z) + np.pi + shift_hue) / (2 * np.pi)
    # hue = hue / np.pi * 180.
    # cmap = ListedColormap(hsluv)
    # colors = cmap(phase)[..., :3]

    abs_z = np.abs(z) * 200

    if lightness_model is None:
        #lightness_model = lambda x: x ** 1 / (x ** 1 + 1)
        lightness_model = lambda x: x
        #lightness_model = lambda x: 2/np.pi*np.arctan(x)

    lightness = lightness_model(abs_z)
    values = lightness + .7 * np.minimum(lightness, 1 - lightness)
    saturation = np.zeros_like(lightness)

    mask = values > 0
    saturation[mask] = 2 * (1 - lightness[mask] / values[mask])
    hsv = np.stack([
        hue, saturation, values
    ], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb


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
