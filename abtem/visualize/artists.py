from __future__ import annotations

from abc import abstractmethod, ABCMeta
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Colormap
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from abtem.core import config
from abtem.core.axes import complex_labels
from abtem.core.colors import hsluv_cmap

if TYPE_CHECKING:
    from abtem.visualize.data import VisualizationData, ImageData, LinesData, PointsData
    from matplotlib.text import Annotation


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

        print(kwargs)

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


class Artist(metaclass=ABCMeta):
    def __init__(self, ax):
        self._ax = ax

    @abstractmethod
    def get_xlim(self):
        pass

    @abstractmethod
    def get_ylim(self):
        pass

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self, power=1.0):
        pass

    def set_logscale(self):
        pass

    @abstractmethod
    def get_power(self):
        pass

    @abstractmethod
    def _change_artist_type(self, artist, ax, data, caxes=None, **kwargs):
        pass


class Artist1D(Artist):
    pass


class LinesArtist(Artist1D):
    def __init__(
        self,
        ax: Axes,
        data,
        caxes: list[Axes] = None,
        label: list[str] = None,
        **kwargs,
    ):
        y = self._reshape_data(data._y)
        x = data._x
        self._lines = ax.plot(x, y, label=label, **kwargs)
        # ax.set_ylim(value_limits)
        super().__init__(ax)

    @staticmethod
    def _reshape_data(data):
        if len(data.shape) > 1:
            data = np.moveaxis(data, -1, 0).reshape((data.shape[-1], -1))
        return data

    def get_xlim(self):
        xlim = [np.inf, -np.inf]
        for line in self._lines:
            data = line.get_data()[0]
            new_xlim = [np.min(data), np.max(data)]
            xlim = [min(new_xlim[0], xlim[0]), max(new_xlim[1], xlim[1])]
        ptp = xlim[1] - xlim[0]
        return [xlim[0] - 0.05 * ptp, xlim[1] + 0.05 * ptp]

    def get_ylim(self):
        ylim = [np.inf, -np.inf]
        for line in self._lines:
            data = line.get_data()[1]
            new_ylim = [np.min(data), np.max(data)]
            ylim = [min(new_ylim[0], ylim[0]), max(new_ylim[1], ylim[1])]
        ptp = ylim[1] - ylim[0]
        ptp = max(ptp, ylim[1] * 0.01)
        return [ylim[0] - 0.05 * ptp, ylim[1] + 0.05 * ptp]

    def get_value_limits(self):
        return self.get_ylim()

    def set_data(self, data: LinesData):
        y = self._reshape_data(data._y)
        x = data._x

        for i, line in enumerate(self._lines):
            line.set_data(x, y)

    def _change_artist_type(self, artist, ax, data, **kwargs):
        raise NotImplementedError

    def get_power(self):
        return 1

    def set_power(self, power=1.0) -> None:
        raise NotImplementedError

    def set_value_limits(self, value_limits: list[float] = None):
        data = np.stack([line.get_data()[1] for line in self._lines], axis=0)
        value_limits = _get_value_limits(data, value_limits, margin=0.05)
        self._ax.set_ylim(value_limits)

    def set_legend(self, **kwargs):
        self._ax.legend(**kwargs)


class Artist2D(Artist):
    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def set_value_limits(self, value_limits):
        pass

    @abstractmethod
    def set_power(self, power: float = 1.0):
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
            old_norm._changed()
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

    @property
    @abstractmethod
    def num_cbars(self):
        pass

    @abstractmethod
    def get_value_limits(self):
        pass

    def set_scale_bars(self, **kwargs):
        self._scale_bar = ScaleBar(ax=self._ax, **kwargs)


def default_cbar_scalar_formatter():
    format = ScalarFormatter(useMathText=True)
    format.set_powerlimits((-3, 3))
    return format


def validate_cmap(cmap, data):
    if cmap is None:
        if data.is_complex and data.complex_conversion in ("none", "phase"):
            cmap = config.get("visualize.phase_cmap", "hsluv")
        else:
            cmap = config.get("visualize.cmap", "viridis")

    if cmap == "hsluv":
        cmap = hsluv_cmap

    return cmap


class ImageArtist(Artist2D):
    num_cbars = 1

    def __init__(
        self,
        ax: Axes,
        data: ImageData,
        caxes: list[Axes] = None,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        extent: tuple[float, float, float, float] = None,
        power: float = 1.0,
        logscale: bool = False,
        origin: Literal["upper", "lower"] | None = None,
        cbar_label: str = None,
        **kwargs,
    ):
        interpolation = kwargs.pop("interpolation", "none")

        self._axes_image = ax.imshow(
            data._array.T,
            origin=origin,
            cmap=cmap,
            extent=extent,
            interpolation=interpolation,
            **kwargs,
        )
        norm = _get_norm(vmin, vmax, power, logscale)
        self._axes_image.set_norm(norm)
        self._cbar = None

        if caxes:
            self.set_cbars(caxes=caxes, label=cbar_label)

        super().__init__(ax)

    @property
    def axes_image(self):
        return self._axes_image

    @property
    def norm(self):
        return self.axes_image.norm

    def get_power(self):
        if hasattr(self.norm, "gamma"):
            return self.norm.gamma
        else:
            return 1.0

    def _change_artist_type(self, artist, ax, data, caxes: list[Axes] = None, **kwargs):
        kwargs = {
            "extent": self.axes_image.get_extent(),
            "cmap": self.axes_image.get_cmap(),
            "cbar_label": cbar_label,
        }
        return artist(ax, data, caxes=caxes, **kwargs)

    def get_value_limits(self):
        array = self.axes_image.get_array()
        return [array.min(), array.max()]

    def get_xlim(self):
        return self.axes_image.get_extent()[:2]

    def get_ylim(self):
        return self.axes_image.get_extent()[2:]

    def set_cbars(self, caxes, **kwargs):
        format = kwargs.pop("format", default_cbar_scalar_formatter())
        self._make_cbar(self.axes_image, caxes[0], format=format, **kwargs)

    def set_cmap(self, cmap):
        self.axes_image.set_cmap(cmap)

    def set_data(self, data):
        self.axes_image.set_data(data._array.T)

    def set_extent(self, extent):
        self.axes_image.set_extent(extent)

    def set_power(self, power: float = 1.0):
        self._update_norm(self.norm, power, self.axes_image)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._set_vmin_vmax(self.norm, *value_limits)


class ScatterArtist(Artist2D):
    num_cbars = 1

    def __init__(
        self,
        ax: Axes,
        data: PointsData,
        caxes: list[Axes] = None,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        scale: float = 0.5,
        annotation_threshold: float = 0.1,
        annotations: list[str] = None,
        **kwargs,
    ):
        # intensities = data["intensities"]
        # positions = data["positions"][:, :2]
        # hkl = data["hkl"]

        vmin, vmax = _get_value_limits(data.array, value_limits=[vmin, vmax])
        norm = _get_norm(vmin, vmax, power, logscale)

        radii = self._calculate_radii(norm, scale, data.array)

        self._ellipse_collection = EllipseCollection(
            widths=radii,
            heights=radii,
            angles=0.0,
            units="xy",
            array=data.array,
            cmap=cmap,
            offsets=data.points,
            transOffset=ax.transData,
            **kwargs,
        )

        ax.add_collection(self._ellipse_collection)

        self._ellipse_collection.set_norm(norm)

        self._scale = scale
        self._annotation_threshold = annotation_threshold
        self._annotation_placement = "top"
        self._annotations = None

        if annotations:
            assert len(annotations) == len(data.points)

            self.set_annotations(
                annotations,
                threshold=annotation_threshold,
                placement="top",
            )

    def get_ylim(self):
        return [self.offsets[:, 1].min() * 1.1, self.offsets[:, 1].max() * 1.1]

    def get_xlim(self):
        return [self.offsets[:, 0].min() * 1.1, self.offsets[:, 0].max() * 1.1]

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

    @property
    def radii(self):
        return self.ellipse_collection._widths

    @staticmethod
    def _calculate_radii(norm, scale, data):
        radii = (norm(data) * scale) ** 0.5
        radii = np.clip(radii, 0.0001, None)
        return radii

    def get_power(self):
        if hasattr(self.norm, "gamma"):
            return self.norm.gamma
        else:
            return 1.0

    def _change_artist_type(self, artist, ax, data, **kwargs):
        raise NotImplementedError
        # kwargs = {
        #     "extent": self.axes_image.get_extent(),
        #     "cmap": self.axes_image.get_cmap(),
        # }
        # return artist(ax, data, **kwargs)
        #

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

    def _get_annotation_positions(self, placement):
        positions = self.offsets.copy()
        radii = self.radii

        if placement == "top":
            positions[:, 1] += radii
        elif placement == "bottom":
            positions[:, 1] -= radii
        elif placement != "center":
            raise ValueError()

        return positions

    def _get_annotation_vilibilities(self, threshold) -> np.ndarray:
        return np.array(self.ellipse_collection.get_array() > threshold)

    def update_annotations(self):
        visibilities = self._get_annotation_vilibilities(self._annotation_threshold)
        positions = self._get_annotation_positions(self._annotation_placement)

        for annotation, visible, position in zip(
            self.annotations, visibilities, positions
        ):
            annotation.set_visible(visible)
            annotation.set_position(positions)

    @property
    def annotations(self) -> list[Annotation]:
        return self._annotations

    def set_annotations(self, annotations, placement, threshold, **kwargs):
        if placement == "top":
            va = "bottom"
        elif placement == "center":
            va = "center"
        elif placement == "bottom":
            va = "top"
        else:
            raise ValueError()

        self._annotation_threshold = threshold
        self._annotation_placement = placement

        ax = self.ellipse_collection.axes
        positions = self._get_annotation_positions(placement)
        visibilities = self._get_annotation_vilibilities(self._annotation_threshold)

        self._annotations = []
        for annotation, position, visible in zip(annotations, positions, visibilities):
            self._annotations.append(
                ax.annotate(
                    annotation,
                    xy=position,
                    ha="center",
                    va=va,
                    visible=visible,
                    **kwargs,
                )
            )


class DomainColoringArtist(Artist2D):
    num_cbars = 2

    def __init__(
        self,
        ax: Axes,
        data: ImageData,
        caxes: list[Axes] = None,
        cmap: str | Colormap | None = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        logscale: bool = False,
        cbar_label: str = None,
        **kwargs,
    ):
        norm = _get_norm(vmin, vmax, power, logscale)

        abs_array = np.abs(data._array)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        cmap = validate_cmap(cmap, data)

        interpolation = kwargs.pop("interpolation", "none")
        self._phase_axes_image = ax.imshow(
            np.angle(data._array).T,
            origin="lower",
            interpolation=interpolation,
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
            **kwargs,
        )
        self._amplitude_axes_image = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation=interpolation,
            cmap="gray",
            zorder=-1,
            **kwargs,
        )

        self._amplitude_axes_image.set_norm(norm)

        if caxes is not None:
            self.set_cbars(caxes, label=cbar_label)

        super().__init__(ax)

    def get_power(self):
        if hasattr(self.amplitude_norm, "gamma"):
            return self.amplitude_norm.gamma
        else:
            return 1.0

    def get_value_limits(self):
        array = self.amplitude_axes_image.get_array()
        return [array.min(), array.max()]

    def get_xlim(self):
        return self.amplitude_axes_image.get_extent()[:2]

    def get_ylim(self):
        return self.amplitude_axes_image.get_extent()[2:]

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

    def _change_artist_type(self, artist, ax, data, **kwargs):
        default_kwargs = {
            "extent": self.amplitude_axes_image.get_extent(),
            "cmap": self.amplitude_axes_image.get_cmap(),
            # "interpolation": self.norm.vmin,
            # "vmax": self.norm.vmax,
            # "vmax": self.norm.vmax,
        }

        default_kwargs.update(kwargs)

        return artist(ax, data, **default_kwargs)

    def set_value_limits(self, value_limits: tuple[float, float] = None):
        self._set_vmin_vmax(self.amplitude_norm, *value_limits)
        self._update_alpha()

    def set_cmap(self, cmap):
        self.phase_axes_image.set_cmap(cmap)
        self._phase_cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self._phase_cbar.set_ticklabels(
            [
                "$-\pi$",
                "$-\dfrac{\pi}{2}$",
                "$0$",
                "$\dfrac{\pi}{2}$",
                "$\pi$",
            ]
        )

    def set_power(self, power: float = 1.0):
        self._update_norm(self.amplitude_norm, power, self.amplitude_axes_image)
        self._update_alpha()

    def set_extent(self, extent):
        self.phase_axes_image.set_extent(extent)
        self.amplitude_axes_image.set_extent(extent)

    def set_data(self, data):
        abs_array = np.abs(data._array)
        alpha = self.amplitude_norm(abs_array)
        alpha = np.clip(alpha, a_min=0, a_max=1)

        self.phase_axes_image.set_alpha(alpha.T)
        self.phase_axes_image.set_data(np.angle(data._array).T)
        self.amplitude_axes_image.set_data(abs_array.T)

    def set_cbars(self, caxes, label=None, **kwargs):
        if caxes is None:
            caxes = [None, None]

        self._phase_cbar = self._make_cbar(
            self.phase_axes_image, cax=caxes[0], **kwargs
        )

        format = default_cbar_scalar_formatter()

        amplitude_cbar = self._make_cbar(
            self.amplitude_axes_image, cax=caxes[1], format=format, **kwargs
        )

        self._phase_cbar.set_label("arg", rotation=0, ha="center", va="top")
        self._phase_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
        self._phase_cbar.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        self._phase_cbar.set_ticklabels(
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
