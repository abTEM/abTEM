"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import Size, Divider
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory

from abtem.core.colors import hsluv_cmap
from abtem.core.utils import interleave, flatten_list_of_lists


if TYPE_CHECKING:
    from abtem.measurements import (
        BaseMeasurements,
    )


def _cbar_orientation(cbar_loc):
    if cbar_loc == "right":
        orientation = "vertical"
    elif cbar_loc == "below":
        orientation = "horizontal"
    else:
        raise ValueError()
    return orientation


def _make_default_sizes(cbar_loc="right"):
    sizes = {
        "cbar_spacing": Size.Fixed(0.4),
        "padding": Size.Fixed(0.1),
        "cbar_shift": Size.Fixed(0.1),
        "cax": Size.Fixed(0.1),
        # "padding": Size.Fixed(0.1),
    }

    if _cbar_orientation(cbar_loc) == "vertical":
        sizes["cbar_padding_left"] = Size.Fixed(0.1)
        sizes["cbar_padding_right"] = Size.Fixed(0.9)

    elif _cbar_orientation(cbar_loc) == "horizontal":
        sizes["cbar_padding_left"] = Size.Fixed(0.4)
        sizes["cbar_padding_right"] = Size.Fixed(0.4)

    return sizes


def _cbar_layout(n: int, sizes: dict) -> list:
    if n == 0:
        return []

    layout = [sizes["cbar_padding_left"]]
    for i in range(n):
        layout.extend([sizes["cbar"]])

        if i < n - 1:
            layout.extend([sizes["cbar_spacing"]])

    layout.extend([sizes["cbar_padding_right"]])
    return layout


def _insert_multiple(l, l2, i):
    return l[:i] + l2 + l[i:]


def _make_grid_layout(
    axes,
    ncbars: int,
    sizes: dict,
    cbar_mode: str = "each",
    cbar_loc: str = "right",
    direction: str = "col",
):
    sizes_layout = []
    for i, ax in enumerate(axes):
        if direction == "col":
            sizes_layout.append([Size.AxesX(ax, aspect="axes", ref_ax=axes[0])])
        elif direction == "row":
            sizes_layout.append(
                [Size.AxesY(ax, aspect="axes", ref_ax=None)]
            )  # ref_ax=axes[0])])
        else:
            raise ValueError()

        if (i < len(axes) - 1) and ((ncbars == 0) or (cbar_mode == "single")):
            sizes_layout[-1].append(sizes["padding"])

    if not "cbar" in sizes:
        sizes["cbar"] = Size.from_any("5%", sizes_layout[0][0])

    if cbar_mode == "each":
        if _cbar_orientation(cbar_loc) == "vertical":
            cbar_layouts = [_cbar_layout(ncbars, sizes) for _ in range(len(axes))]
            sizes_layout = interleave(sizes_layout, cbar_layouts)
        elif _cbar_orientation(cbar_loc) == "horizontal":
            cbar_layouts = [_cbar_layout(ncbars, sizes)[::-1] for _ in range(len(axes))]
            sizes_layout = interleave(cbar_layouts, sizes_layout)
    elif cbar_mode == "single":
        if _cbar_orientation(cbar_loc) == "vertical":
            sizes_layout.extend([_cbar_layout(ncbars, sizes)])
        elif _cbar_orientation(cbar_loc) == "horizontal":
            sizes_layout = [_cbar_layout(ncbars, sizes)[::-1]] + sizes_layout
    else:
        raise ValueError()

    sizes_layout = flatten_list_of_lists(sizes_layout)
    return sizes_layout


class AxesGrid:
    def __init__(
        self,
        fig,
        ncols: int,
        nrows: int,
        ncbars: int = 0,
        cbar_mode: str = "single",
        cbar_loc: str = "right",
        aspect: bool = True,
        anchor: str = "NW",
        sharex: bool = True,
        sharey: bool = True,
        rect: tuple = (0.1, 0.1, 0.9, 0.9),
        origin="lower",
    ):
        self._fig = fig
        self._ncols = ncols
        self._nrows = nrows
        self._ncbars = ncbars
        self._aspect = aspect
        self._sharex = sharex
        self._sharey = sharey
        self._rect = rect
        self._anchor = anchor
        self._cbar_loc = cbar_loc
        self._cbar_mode = cbar_mode

        if not origin == "lower":
            # TODO
            raise NotImplementedError()

        self._sizes = {
            "cbar_spacing": Size.Fixed(0.9),
            "padding": Size.Fixed(0.1),
            "cbar_shift": Size.Fixed(0.0),
            "cbar_width": Size.Fixed(0.15),
            "left": Size.Fixed(0.0),
            "right": Size.Fixed(0.2),
            "top": Size.Fixed(0.2),
            "bottom": Size.Fixed(0.0),
        }

        self._axes = self._make_axes()
        self._caxes = self._make_caxes()
        self._divider = self._make_divider()

        self._set_axes_locators()
        self._set_caxes_locators()

        if sharex:
            for inner_axes in self._axes[:, 1:]:
                for ax in inner_axes:
                    ax._axislines["bottom"].toggle(ticklabels=False, label=False)

        if sharey:
            for inner_axes in self._axes[1:]:
                for ax in inner_axes:
                    ax._axislines["left"].toggle(ticklabels=False, label=False)

    def _axis_location_to_indices(self, axis_location):
        axis_locations = {
            "all": tuple(np.ndindex(self.shape)),
            "upper left": ((0, self.axes.shape[1] - 1),),
            "upper right": (self.axes.shape[0] - 1, self.axes.shape[1] - 1),
            "lower left": ((0, 0),),
            "lower right": ((self.axes.shape[0] - 1, 0),),
        }
        return axis_locations[axis_location]

    def _make_axes(self):
        from mpl_toolkits.axes_grid1.mpl_axes import Axes

        ax = Axes(self.fig, (0, 0, 1, 1), sharex=None, sharey=None)

        sharex = ax if self._sharex else None
        sharey = ax if self._sharey else None

        axes = [ax] + [
            Axes(self.fig, (0, 0, 1, 1), sharex=sharex, sharey=sharey)
            for _ in range(self._nrows * self._ncols - 1)
        ]

        for i, ax in enumerate(axes):
            ax._tag = i

        for ax in axes:
            self._fig.add_axes(ax)

        axes = np.array(axes, dtype=object).reshape(
            (self._ncols, self._nrows), order="C"
        )

        return axes

    def _remove_caxes(self):
        if self._cbar_mode == "single":
            self._caxes = self._caxes[0, 0]

        for cax in self._caxes.ravel():
            cax.remove()

    def _make_caxes(self):
        orientation = _cbar_orientation(self._cbar_loc)

        if self._cbar_mode == "each":
            caxes = [
                _cbaraxes_class_factory(Axes)(
                    self._fig, self._rect, orientation=orientation
                )
                for _ in range(self._nrows * self._ncols * self._ncbars)
            ]
        else:
            caxes = (
                [
                    _cbaraxes_class_factory(Axes)(
                        self._fig, self._rect, orientation=orientation
                    )
                    for _ in range(self._ncbars)
                ]
                * self._nrows
                * self._ncols
            )

        for cax in caxes:
            self._fig.add_axes(cax)

        caxes = np.array(caxes, dtype=object).reshape(
            (self._ncols, self._nrows, self._ncbars)
        )

        return caxes

    def _make_size(self, axes, line_types, axes_size):
        i = 0
        line_sizes = []
        for row_type in line_types:
            if row_type == "ax":
                # line_sizes.append(Size.Scaled(1.)) # TODO
                line_sizes.append(axes_size(axes[i], aspect=0.2, ref_ax=axes[0]))
                i += 1
            else:
                line_sizes.append(self._sizes[row_type])

        return line_sizes

    def _make_divider(self):
        row_types, col_types = self._get_row_types(), self._get_col_types()

        row_sizes = self._make_size(self._axes[0], row_types, Size.AxesY)
        col_sizes = self._make_size(self._axes[:, 0], col_types, Size.AxesX)

        divider = Divider(
            self._fig,
            self._rect,
            horizontal=col_sizes,
            vertical=row_sizes,
            aspect=self._aspect,
            anchor="C",
        )
        return divider

    def _set_axes_locators(self):
        row_types, col_types = self._get_row_types(), self._get_col_types()
        i = 0
        for nx, col_type in enumerate(col_types):
            for ny, row_type in enumerate(row_types):
                if (row_type == "ax") and (col_type == "ax"):
                    locator = self._divider.new_locator(nx=nx, ny=ny)
                    self._axes.ravel()[i].set_axes_locator(locator)
                    i += 1

    def _set_caxes_locators(self):
        row_types, col_types = self._get_row_types(), self._get_col_types()
        i = 0
        if (self._cbar_mode == "single") and (self._cbar_loc == "right"):
            for nx, col_type in enumerate(col_types):
                if col_type == "cbar_width":
                    locator = self._divider.new_locator(nx=nx, ny=1, ny1=-2)
                    self._caxes.ravel()[i].set_axes_locator(locator)
                    i += 1

        elif (self._cbar_mode == "single") and (self._cbar_loc == "below"):
            for ny, row_type in enumerate(row_types):
                if row_type == "cbar_width":
                    locator = self._divider.new_locator(ny=ny, nx=1, nx1=-2)
                    self._caxes.ravel()[i].set_axes_locator(locator)
                    i += 1
        else:
            for ny, row_type in enumerate(row_types):
                for nx, col_type in enumerate(col_types):
                    if ((row_type == "ax") and (col_type == "cbar_width")) or (
                        (row_type == "cbar_width") and (col_type == "ax")
                    ):
                        locator = self._divider.new_locator(nx=nx, ny=ny)
                        self._caxes.ravel()[i].set_axes_locator(locator)
                        i += 1

    def _get_col_types(self):
        if self._cbar_loc == "right":
            return self._get_line_types(
                n=self._ncols, ncbars=self._ncbars, orientation="horizontal"
            )
        else:
            return self._get_line_types(
                n=self._ncols, ncbars=0, orientation="horizontal"
            )

    def _get_row_types(self):
        if self._cbar_loc == "right":
            return self._get_line_types(n=self._nrows, ncbars=0, orientation="vertical")
        else:
            return self._get_line_types(
                n=self._nrows, ncbars=self._ncbars, orientation="vertical"
            )

    def _get_line_types(self, n, ncbars, orientation):
        cbar_types = [
            ["cbar_shift"] * (ncbars > 0) + ["cbar_width", "cbar_spacing"] * ncbars
        ]
        axes_types = [["ax", "padding"]] * (n - 1) + [["ax", "padding"]]

        if ncbars == 0:
            axes_types[-1] = axes_types[-1][:-1]

        if self._cbar_mode == "each":
            line_types = interleave(axes_types, cbar_types * n)
            line_types = flatten_list_of_lists(line_types)
        else:
            line_types = flatten_list_of_lists(axes_types + cbar_types)

        if self._cbar_loc == "below":
            line_types = line_types[::-1]

        if orientation == "horizontal":
            line_types = ["left"] + line_types + ["right"]
        else:
            line_types = ["bottom"] + line_types + ["top"]

        return line_types

    def set_sizes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self._sizes[key], "fixed_size", value)

        self.fig.canvas.draw_idle()

    def set_cbar_layout(self, **kwargs):
        self._ncbars = kwargs.pop("ncbars", self._ncbars)
        self._cbar_loc = kwargs.pop("cbar_loc", self._cbar_loc)
        self._cbar_mode = kwargs.pop("cbar_mode", self._cbar_mode)

        self._remove_caxes()
        self._caxes = self._make_caxes()
        self._divider = self._make_divider()

        self._set_axes_locators()
        self._set_caxes_locators()

    def adjust_figure_to_bbox(self):
        self.set_sizes(left=0.0, bottom=0.0)

        size = self.fig.get_size_inches()
        bbox_inches = self.fig.get_tightbbox()
        pad_inches = plt.rcParams["savefig.pad_inches"]
        bbox_inches.padded(pad_inches)
        aspect = bbox_inches.width / bbox_inches.height
        self.fig.set_size_inches((size[0], size[0] / aspect))

        bbox_inches = self.fig.get_tightbbox().padded(pad_inches)
        self.set_sizes(left=-bbox_inches.xmin)

        bbox_inches = self.fig.get_tightbbox().padded(pad_inches)
        self.set_sizes(bottom=-bbox_inches.ymin)
        self.fig.canvas.draw_idle()

    @property
    def axes(self):
        return self._axes

    @property
    def fig(self):
        return self._fig

    @property
    def ncols(self) -> int:
        return self._axes.shape[0]

    @property
    def nrows(self) -> int:
        return self._axes.shape[1]

    def __getitem__(self, item):
        return self._axes[item]

    def __len__(self) -> int:
        return len(self._axes)

    @property
    def shape(self) -> tuple[int, int]:
        return self._axes.shape


def _axes_grid_cols_and_rows(ensemble_shape, axes_types):
    shape = tuple(
        n
        for n, axes_type in zip(ensemble_shape, axes_types)
        if not axes_type in ("index", "range", "overlay")
    )

    if len(shape) > 0:
        ncols = shape[0]
    else:
        ncols = 1

    if len(shape) > 1:
        nrows = shape[1]
    else:
        nrows = 1

    return ncols, nrows


def _determine_axes_types(
    ensemble_axes_metadata,
    explode: bool | tuple[bool, ...] | None,
    overlay: bool | tuple[bool, ...] | None,
):
    num_ensemble_axes = len(ensemble_axes_metadata)

    axes_types = []
    for axis_metadata in ensemble_axes_metadata:
        if axis_metadata._default_type is not None:
            axes_types.append(axis_metadata._default_type)
        else:
            axes_types.append("index")

    if explode is True:
        explode = tuple(range(max(num_ensemble_axes - 2, 0), num_ensemble_axes))
    elif explode is False:
        explode = ()
        axes_types = [axes_type for axes_type in axes_types if axes_type != "explode"]

    if overlay is True:
        overlay = tuple(range(max(num_ensemble_axes - 2, 0), num_ensemble_axes))
    elif overlay is False:
        overlay = ()
        axes_types = [
            axes_type if axes_type != "overlay" else "index" for axes_type in axes_types
        ]

    axes_types = list(axes_types)
    for i, axis_type in enumerate(axes_types):
        if explode is not None:
            if i in explode:
                axes_types[i] = "explode"
            elif i not in overlay:
                axes_types[i] = "index"

        if overlay is not None:
            if i in overlay:
                axes_types[i] = "overlay"
            elif (explode is not None) and (i not in explode):
                axes_types[i] = "index"

    return axes_types


def _validate_axes(
    axes_types,
    ensemble_shape,
    ax: Axes = None,
    explode: bool = False,
    overlay: bool = False,
    ncbars: int = 0,
    common_color_scale: bool = False,
    figsize: tuple[float, float] = None,
    ioff: bool = False,
    aspect: bool = True,
    sharex: bool = True,
    sharey: bool = True,
) -> AxesGrid:
    if common_color_scale:
        cbar_mode = "single"
    else:
        cbar_mode = "each"

    if ax is None:
        if ioff:
            with plt.ioff():
                fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    if ax is None:
        ncols, nrows = _axes_grid_cols_and_rows(ensemble_shape, axes_types)

        axes = AxesGrid(
            fig=fig,
            ncols=ncols,
            nrows=nrows,
            ncbars=ncbars,
            cbar_mode=cbar_mode,
            aspect=aspect,
            sharex=sharex,
            sharey=sharey,
        )

    # elif ax is None:
    #     ax = fig.add_subplot()
    #     axes = np.array([[ax]])
    else:
        if explode:
            raise NotImplementedError("`ax` not implemented with `explode = True`.")

        axes = np.array([[ax]])

    return axes
