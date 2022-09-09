import numpy as np
from cplot._colors import get_srgb1
from matplotlib import cm, colors
import matplotlib.pyplot as plt


def get_colors(
    z: np.ndarray, vmin: float = None, vmax: float = None, saturation: float = 2.0
):
    abs_z = np.abs(z)
    angle_z = np.angle(z)

    if vmin is None:
        vmin = abs_z.min()

    if vmax is None:
        vmax = abs_z.max()

    vmin_rel = (vmin - abs_z.min()) / abs_z.ptp()
    vmax_rel = (vmax - abs_z.max()) / abs_z.ptp()

    abs_scaled = (abs_z - abs_z.min()) / abs_z.ptp() * (
        1 - vmax_rel - vmin_rel
    ) + vmin_rel

    z_scaled = abs_scaled * np.exp(1.0j * angle_z)

    return get_srgb1(z_scaled, lambda x: x, saturation)


def add_colorbar_abs(cax, vmin, vmax):
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb0 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.gray), cax=cax,)
    cb0.set_label("abs", rotation=0, ha="center", va="top")
    cb0.ax.yaxis.set_label_coords(0.5, -0.03)


def add_colorbar_arg(cax, saturation_adjustment: float):
    z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
    rgb_vals = get_srgb1(
        z,
        abs_scaling=lambda z: np.full_like(z, 0.5),
        saturation_adjustment=saturation_adjustment,
    )
    rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
    newcmp = colors.ListedColormap(rgba_vals)
    #
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)

    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp), cax=cax)

    cb1.set_label("arg", rotation=0, ha="center", va="top")
    cb1.ax.yaxis.set_label_coords(0.5, -0.03)
    cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
    cb1.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )
