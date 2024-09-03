"""Module for plotting atoms, images, line scans, and diffraction patterns."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory
import matplotlib
import matplotlib.colorbar as cbar
from abtem.core.utils import flatten_list_of_lists, interleave


def _cbar_orientation(cbar_loc):
    if cbar_loc == "right":
        orientation = "vertical"
    elif cbar_loc == "below":
        orientation = "horizontal"
    else:
        raise ValueError()
    return orientation


class AxesCollection:
    def __init__(self, axes, caxes, cbar_mode="single"):
        self._axes = axes
        self._caxes = caxes
        self._cbar_mode = cbar_mode

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
        origin: str = "lower",
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
            "cbar_spacing": Size.Fixed(1.1),
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

    def axis_location_to_indices(self, axis_location):
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
