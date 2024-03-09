from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import EllipseCollection
from scipy import spatial

from abtem.core import config

if TYPE_CHECKING:
    pass


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


class Artist1D:
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


class LinesArtist(Artist1D):
    def __init__(self, ax, x, y, value_limits, labels):
        y = self._reshape_data(y)

        self._lines = ax.plot(x, y, label=labels)

        if not isinstance(self._lines, list):
            self._lines = [self._lines]

        # ax.set_ylim(value_limits)

        self._ax = ax

    @staticmethod
    def _reshape_data(data):
        if len(data.shape) > 1:
            data = np.moveaxis(data, -1, 0).reshape((data.shape[-1], -1))
        return data

    def set_data(self, data):
        data = self._reshape_data(data)
        for i, line in enumerate(self._lines):
            x = line.get_data()[0]
            line.set_data(x, data)

    def set_value_limits(self, value_limits):
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


class ImageArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        extent=None,
        power: float = 1.0,
        logscale: bool = False,
        **kwargs,
    ):
        norm = _get_norm(vmin, vmax, power, logscale)

        interpolation = kwargs.pop("interpolation", "none")
        origin = kwargs.pop("origin", "lower")

        self._image = ax.imshow(
            data.T,
            origin=origin,
            cmap=cmap,
            extent=extent,
            interpolation=interpolation,
            **kwargs,
        )

        self._image.set_norm(norm)

    def set_extent(self, extent):
        self._image.set_extent(extent)

    def set_data(self, data):
        self._image.set_data(data.T)

    def set_cmap(self, cmap):
        self._image.set_cmap(cmap)

    def set_cbars(self, caxes=None, label=None, **kwargs):
        if caxes is not None:
            self._cbar = plt.colorbar(self._image, cax=caxes[0])
        else:
            self._cbar = plt.colorbar(self._image)

        self._cbar.set_label(label, **kwargs)
        self._cbar.formatter.set_powerlimits((-2, 2))
        self._cbar.formatter.set_useMathText(True)
        self._cbar.ax.yaxis.set_offset_position("left")

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._norm.vmin = value_limits[0]
        self._norm.vmax = value_limits[1]

    def set_power(self, power: float = 1.0):
        if (power != 1.0) and isinstance(self._norm, colors.PowerNorm):
            self._norm.gamma = power
            self._norm._changed()
        else:
            self._norm = _get_norm(
                vmin=self._norm.vmin, vmax=self._norm.vmax, power=power
            )
            self._image.norm = self._norm


class ScatterArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        scale: float = 0.5,
        annotation_threshold: float = 0.1,
    ):
        intensities = data["intensities"]
        positions = data["positions"][:, :2]
        hkl = data["hkl"]

        vmin, vmax = _get_value_limits(intensities, value_limits=[vmin, vmax])
        norm = _get_norm(vmin, vmax, power, logscale)

        self._norm = norm
        self._scale = scale
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
        )

        self._ellipse_collection.set_norm(norm)

        ax.add_collection(self._ellipse_collection)

        self._make_annotations(
            ax=ax,
            hkl=hkl,
            positions=positions,
            intensities=intensities,
            radii=radii,
            threshold=annotation_threshold,
            alignment="top",
        )

    def _get_annotation_positions(self, positions, radii, alignment):
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
        visibilities = intensities > threshold

        if alignment == "top":
            va = "bottom"
        elif alignment == "center":
            va = "center"
        elif alignment == "bottom":
            va = "top"
        else:
            raise ValueError()

        for hkl, position, visible, radius in zip(hkl, positions, visibilities, radii):
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
                visible=visible,
                # **kwargs,
            )

    @staticmethod
    def _calculate_radii(norm, scale, data):
        radii = (norm(data) * scale) ** 0.5
        radii = np.clip(radii, 0.0001, None)
        return radii

    def set_data(self, data):
        intensities = data["intensities"]
        self._ellipse_collection.set_array(intensities)
        self._norm._changed()
        self._set_radii()

    def _set_radii(self):
        data = self._ellipse_collection.get_array().data
        radii = self._calculate_radii(self._norm, self._scale, data)

        self._ellipse_collection._widths = radii
        self._ellipse_collection._heights = radii
        self._ellipse_collection.set()

    def set_cmap(self, cmap):
        self._ellipse_collection.set_cmap(cmap)

    def set_cbars(self, caxes, label=None, **kwargs):
        self._cbar = plt.colorbar(self._ellipse_collection, cax=caxes[0])

        self._cbar.set_label(label, **kwargs)
        self._cbar.formatter.set_powerlimits((-2, 2))
        self._cbar.formatter.set_useMathText(True)
        self._cbar.ax.yaxis.set_offset_position("left")

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._norm.vmin = value_limits[0]
        self._norm.vmax = value_limits[1]
        self._set_radii()

    def set_power(self, power: float = 1.0):
        if (power != 1.0) and isinstance(self._norm, colors.PowerNorm):
            self._norm.gamma = power
            self._norm._changed()
        else:
            self._norm = _get_norm(
                vmin=self._norm.vmin, vmax=self._norm.vmax, power=power
            )
            self._ellipse_collection.norm = self._norm

        self._set_radii()


class DomainColoringArtist(Artist2D):
    def __init__(
        self,
        ax,
        data,
        cmap,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        extent=None,
    ):
        norm = _get_norm(vmin, vmax, power, logscale)

        abs_array = np.abs(data)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        self._image_phase = ax.imshow(
            np.angle(data).T,
            origin="lower",
            interpolation="none",
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
            extent=extent,
        )
        self._image_amplitude = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation="none",
            cmap="gray",
            zorder=-1,
            extent=extent,
        )

        self._image_amplitude.set_norm(norm)

    def _update_alpha(self):
        data = self._image_amplitude.get_array().data
        norm = self._norm
        alpha = norm(np.abs(data))
        alpha = np.clip(alpha, a_min=0, a_max=1)
        self._image_phase.set_alpha(alpha)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._norm.vmin = value_limits[0]
        self._norm.vmax = value_limits[1]
        self._update_alpha()

    def set_cmap(self, cmap):
        self._image_phase.set_cmap(cmap)

    def set_power(self, power: float = 1.0):
        if (power != 1.0) and isinstance(self._norm, colors.PowerNorm):
            self._norm.gamma = power
            self._norm._changed()
        else:
            self._norm = _get_norm(
                vmin=self._norm.vmin, vmax=self._norm.vmax, power=power
            )
            self._image_amplitude.norm = self._norm
        self._update_alpha()

    def set_extent(self, extent):
        self._image_phase.set_extent(extent)
        self._image_amplitude.set_extent(extent)

    def set_data(self, data):
        norm = self._norm
        abs_array = np.abs(data)
        alpha = norm(abs_array)
        alpha = np.clip(alpha, a_min=0, a_max=1)

        self._image_phase.set_alpha(alpha.T)
        self._image_phase.set_data(np.angle(data).T)
        self._image_amplitude.set_data(abs_array.T)

    def set_cbars(self, caxes, label=None, **kwargs):
        self._cbar_phase = plt.colorbar(self._image_phase, cax=caxes[0])
        self._cbar_amplitude = plt.colorbar(self._image_amplitude, cax=caxes[1])

        self._cbar_phase.set_label(
            "phase [rad.]"
        )  # , rotation=0, ha="center", va="top")
        # self._cbar_phase.ax.yaxis.set_label_coords(0.5, -0.02)
        self._cbar_phase.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        self._cbar_phase.set_ticklabels(
            [
                r"$-\pi$",
                r"$-\dfrac{\pi}{2}$",
                "$0$",
                r"$\dfrac{\pi}{2}$",
                r"$\pi$",
            ]
        )
        self._cbar_amplitude.set_label(
            "amplitude [arb. unit]"
        )  # , rotation=0, ha="center", va="top")
        # self._cbar_amplitude.ax.yaxis.set_label_coords(0.5, -0.02)
        self._cbar_amplitude.formatter.set_powerlimits((0, 0))
        self._cbar_amplitude.formatter.set_useMathText(True)
        self._cbar_amplitude.ax.yaxis.set_offset_position("left")


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
