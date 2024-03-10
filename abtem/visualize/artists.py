from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Colormap
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core import config


def _get_norm(vmin=None, vmax=None, power=1.0, logscale=False):
    if (power == 1.0) and (logscale is False):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif (power != 1.0) and (logscale is False):
        norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)
    elif (power == 1.0) and (logscale is True):
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("")

    return norm


def _get_value_limits(array, value_limits, margin=None):
    if np.iscomplexobj(array):
        array = np.abs(array)

    value_limits = value_limits.copy()

    if value_limits[0] is None:
        value_limits[0] = float(np.nanmin(array))

    if value_limits[1] is None:
        value_limits[1] = float(np.nanmax(array))

    if margin:
        margin = (value_limits[1] - value_limits[0]) * margin
        value_limits[0] -= margin
        value_limits[1] += margin

    return value_limits


class ScaleBar:
    def __init__(
        self,
        ax: Axes,
        label: str = "",
        size: float = None,
        loc: str = "lower right",
        borderpad: float = 0.5,
        size_vertical: float = None,
        sep: float = 6,
        pad: float = 0.3,
        label_top: bool = True,
        frameon: bool = False,
        **kwargs,
    ):
        if size is None:
            xlim = ax.get_xlim()
            size = (xlim[1] - xlim[0]) / 3

        if size_vertical is None:
            ylim = ax.get_ylim()
            size_vertical = (ylim[1] - ylim[0]) / 20

        self._anchored_size_bar = AnchoredSizeBar(
            ax.transData,
            label=label,
            label_top=label_top,
            size=size,
            borderpad=borderpad,
            loc=loc,
            size_vertical=size_vertical,
            sep=sep,
            pad=pad,
            frameon=frameon,
            **kwargs,
        )
        ax.add_artist(self._anchored_size_bar)


class Artist:
    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self):
        pass

    def set_logscale(self):
        pass


class Artist1D(Artist):
    pass


class LinesArtist(Artist1D):
    def __init__(
        self,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        value_limits: list[float] = None,
        labels: str | list[str] = None,
    ):
        y = self._reshape_data(y)
        self._lines = ax.plot(x, y, label=labels)
        ax.set_ylim(value_limits)
        self._ax = ax

    @staticmethod
    def _reshape_data(data):
        if len(data.shape) > 1:
            data = np.moveaxis(data, -1, 0).reshape((data.shape[-1], -1))
        return data

    def set_data(self, data: np.ndarray):
        data = self._reshape_data(data)

        for i, line in enumerate(self._lines):
            x = line.get_data()[0]
            line.set_data(x, data)

    def set_value_limits(self, value_limits: list[float] = None):
        data = np.stack([line.get_data()[1] for line in self._lines], axis=0)
        value_limits = _get_value_limits(data, value_limits, margin=0.05)
        self._ax.set_ylim(value_limits)


class Artist2D:
    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self):
        pass

    @abstractmethod
    def set_cmap(self, cmap):
        pass

    @abstractmethod
    def set_cbars(self, cmap):
        pass

    def set_logscale(self):
        pass

    @staticmethod
    def _set_vmin_vmax(norm, vmin, vmax):
        norm.vmin = vmin
        norm.vmax = vmax

    @staticmethod
    def _update_norm(old_norm, power, artist):
        if (power != 1.0) and isinstance(old_norm, colors.PowerNorm):
            old_norm.gamma = power
            artist.norm = old_norm
        else:
            norm = _get_norm(vmin=old_norm.vmin, vmax=old_norm.vmax, power=power)
            artist.norm = norm

    @staticmethod
    def _make_cbar(mappable, cax, **kwargs):
        return plt.colorbar(mappable, cax=cax, **kwargs)

    @abstractmethod
    def get_ylim(self):
        pass

    @abstractmethod
    def get_xlim(self):
        pass

    @abstractmethod
    def get_value_limits(self):
        pass


def default_cbar_scalar_formatter():
    format = ScalarFormatter(useMathText=True)
    format.set_powerlimits((-3, 3))
    return format


class ImageArtist(Artist2D):
    def __init__(
        self,
        ax: Axes,
        data: np.ndarray,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        extent: tuple[float, float, float, float] = None,
        power: float = 1.0,
        logscale: bool = False,
        origin: Literal["upper", "lower"] | None = None,
        **kwargs,
    ):
        interpolation = kwargs.pop("interpolation", "none")

        self._axes_imshow = ax.imshow(
            data.T,
            origin=origin,
            cmap=cmap,
            extent=extent,
            interpolation=interpolation,
            **kwargs,
        )

        norm = _get_norm(vmin, vmax, power, logscale)
        self._axes_imshow.set_norm(norm)
        self._cbar = None

    @property
    def norm(self):
        return self._axes_imshow.norm

    @property
    def axes_imshow(self):
        return self._axes_imshow

    def set_extent(self, extent):
        self.axes_imshow.set_extent(extent)

    def set_data(self, data):
        self.axes_imshow.set_data(data.T)

    def set_cmap(self, cmap):
        self.axes_imshow.set_cmap(cmap)

    def set_cbars(self, caxes=None, **kwargs):
        format = kwargs.pop("format", default_cbar_scalar_formatter())
        self._make_cbar(self._axes_imshow, caxes[0], format=format, **kwargs)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._set_vmin_vmax(self.norm, *value_limits)

    def set_power(self, power: float = 1.0):
        self._update_norm(self.norm, power, self._axes_imshow)


class ScatterArtist(Artist2D):
    def __init__(
        self,
        ax: Axes,
        data,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        scale: float = 0.5,
        annotation_threshold: float = 0.1,
        **kwargs,
    ):
        intensities = data["intensities"]
        positions = data["positions"][:, :2]
        hkl = data["hkl"]

        vmin, vmax = _get_value_limits(intensities, value_limits=[vmin, vmax])
        norm = _get_norm(vmin, vmax, power, logscale)

        radii = self._calculate_radii(norm, scale, intensities)

        self._ellipse_collection = EllipseCollection(
            widths=radii,
            heights=radii,
            angles=0.0,
            units="xy",
            array=intensities,
            cmap=cmap,
            offsets=positions,
            transOffset=ax.transData,
            **kwargs,
        )
        ax.add_collection(self._ellipse_collection)

        self._ellipse_collection.set_norm(norm)

        self._scale = scale

        self._make_annotations(
            ax=ax,
            hkl=hkl,
            positions=positions,
            intensities=intensities,
            radii=radii,
            threshold=annotation_threshold,
            alignment="top",
        )

    def get_ylim(self):
        return [self.offsets[:, 1].min(), self.offsets[:, 1].max()]

    def get_xlim(self):
        return [self.offsets[:, 0].min(), self.offsets[:, 0].max()]

    def get_value_limits(self):
        array = self.ellipse_collection.get_array()
        return [array.min(), array.max()]

    @property
    def ellipse_collection(self) -> EllipseCollection:
        return self._ellipse_collection

    @property
    def norm(self):
        return self.ellipse_collection.norm

    @property
    def offsets(self):
        return self.ellipse_collection.get_offsets()

    @staticmethod
    def _calculate_radii(norm, scale, data):
        radii = (norm(data) * scale) ** 0.5
        radii = np.clip(radii, 0.0001, None)
        return radii

    def set_data(self, data: np.ndarray):
        self._ellipse_collection.set_array(data["intensities"])
        self._ellipse_collection.set_offsets(data["positions"])
        # self.norm._changed()
        self._set_radii()

    def _set_radii(self):
        data = self._ellipse_collection.get_array().data
        radii = self._calculate_radii(self.norm, self._scale, data)

        self._ellipse_collection._widths = radii
        self._ellipse_collection._heights = radii
        self._ellipse_collection.set()

    def set_cmap(self, cmap: str):
        self.ellipse_collection.set_cmap(cmap)

    def set_cbars(self, caxes=None, **kwargs):
        self._make_cbar(self.ellipse_collection, caxes[0], **kwargs)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._set_vmin_vmax(self.norm, *value_limits)
        self._set_radii()

    def set_power(self, power: float = 1.0):
        self._update_norm(self.norm, power, self._ellipse_collection)
        self._set_radii()

    @staticmethod
    def _get_annotation_positions(positions, radii, alignment):
        positions = positions.copy()
        if alignment == "top":
            positions[:, 1] += radii / 2
        elif alignment == "bottom":
            positions[:, 1] -= radii / 2
        elif alignment != "center":
            raise ValueError()
        return positions

    def _make_annotations(
        self, ax, hkl, positions, intensities: np.array, radii, threshold, alignment
    ):
        positions = self._get_annotation_positions(positions, radii, alignment)

        if alignment == "top":
            va = "bottom"
        elif alignment == "center":
            va = "center"
        elif alignment == "bottom":
            va = "top"
        else:
            raise ValueError()

        for hkl, position, intensity, radius in zip(hkl, positions, intensities, radii):
            if config.get("visualize.use_tex"):
                text = " \ ".join(
                    [f"\\bar{{{abs(i)}}}" if i < 0 else f"{i}" for i in hkl]
                )
                text = f"${text}$"
            else:
                text = "{} {} {}".format(*hkl)

            annotation = ax.annotate(
                text,
                xy=position,
                ha="center",
                va=va,
                # size=size,
                visible=intensity > threshold,
                # **kwargs,
            )


class DomainColoringArtist(Artist2D):
    def __init__(
        self,
        ax: Axes,
        data: np.ndarray,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        **kwargs,
    ):
        norm = _get_norm(vmin, vmax, power, logscale)

        abs_array = np.abs(data)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        self._phase_axes_image = ax.imshow(
            np.angle(data).T,
            origin="lower",
            interpolation="none",
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
            **kwargs,
        )
        self._amplitude_axes_image = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation="none",
            cmap="gray",
            zorder=-1,
            **kwargs,
        )

        self._amplitude_axes_image.set_norm(norm)

    @property
    def amplitude_norm(self):
        return self.amplitude_axes_image.norm

    @property
    def amplitude_axes_image(self):
        return self._amplitude_axes_image

    @property
    def phase_axes_image(self):
        return self._phase_axes_image

    def _update_alpha(self):
        data = self.amplitude_axes_image.get_array().data
        alpha = self.amplitude_axes_image.norm(np.abs(data))
        alpha = np.clip(alpha, a_min=0, a_max=1)
        self.phase_axes_image.set_alpha(alpha)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._set_vmin_vmax(self.amplitude_norm, *value_limits)
        self._update_alpha()

    def set_cmap(self, cmap):
        self.phase_axes_image.set_cmap(cmap)

    def set_power(self, power: float = 1.0):
        self._update_norm(self.amplitude_norm, power, self.amplitude_axes_image)
        self._update_alpha()

    def set_extent(self, extent):
        self.phase_axes_image.set_extent(extent)
        self.amplitude_axes_image.set_extent(extent)

    def set_data(self, data):
        abs_array = np.abs(data)
        alpha = self.amplitude_norm(abs_array)
        alpha = np.clip(alpha, a_min=0, a_max=1)

        self.phase_axes_image.set_alpha(alpha.T)
        self.phase_axes_image.set_data(np.angle(data).T)
        self.amplitude_axes_image.set_data(abs_array.T)

    def set_cbars(self, caxes, label=None, **kwargs):
        phase_cbar = self._make_cbar(self.phase_axes_image, cax=caxes[0], **kwargs)

        format = default_cbar_scalar_formatter()
        amplitude_cbar = self._make_cbar(
            self.amplitude_axes_image, cax=caxes[1], format=format, **kwargs
        )

        phase_cbar.set_label("arg", rotation=0, ha="center", va="top")
        phase_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
        phase_cbar.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        phase_cbar.set_ticklabels(
            [
                "$-\pi$",
                "$-\dfrac{\pi}{2}$",
                "$0$",
                "$\dfrac{\pi}{2}$",
                "$\pi$",
            ]
        )

        amplitude_cbar.set_label("abs", rotation=0, ha="center", va="top")
        amplitude_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
        amplitude_cbar.ax.yaxis.set_offset_position("left")


class OverlayImshowArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
    ):
        raise NotImplementedError

        # cmaps = [ListedColormap(c) for c in cmap]
        #
        # alphas = [np.clip(norm(alpha), a_min=0.0, a_max=1.0) for alpha in array]
        # # print(alphas[0], array.shape, array[0].shape)
        # ims = [
        #     ax.imshow(
        #         np.ones_like(alpha.T),
        #         origin="lower",
        #         interpolation="none",
        #         cmap=cmap,
        #         alpha=alpha.T,
        #     )
        #     for alpha, cmap in zip(alphas, cmaps)
        # ]
        #
        # ax.set_facecolor("k")
        #
        #
        # from matplotlib import colors
        # from matplotlib.cm import ScalarMappable
        # from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        #
        # fig, ax = plt.subplots()
        # ax.set_facecolor("k")
        #
        # cmap = ListedColormap(["lime"])
        # norm = colors.Normalize()
        # norm.autoscale_None(stacked.array[0])
        # alpha = norm(stacked.array[0])
        #
        # im = ax.imshow(
        #     np.ones_like(stacked.array[0]).T, alpha=alpha.T, cmap=cmap, origin="lower"
        # )
        #
        # cmap = LinearSegmentedColormap.from_list("red", ["k", "lime"])
        # plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        #
        # c = "r"
        # cmap = ListedColormap([c])
        # norm = colors.Normalize()
        # norm.autoscale_None(stacked.array[1])
        # alpha = norm(stacked.array[1])
        #
        # im = ax.imshow(
        #     np.ones_like(stacked.array[1]).T, alpha=alpha.T, cmap=cmap, origin="lower"
        # )
        #
        # cmap = LinearSegmentedColormap.from_list("c", ["k", c])
        # plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        #
