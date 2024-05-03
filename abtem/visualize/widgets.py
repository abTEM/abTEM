"""Module for plotting atoms, images, line scans, and diffraction patterns."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Sequence, Any, TYPE_CHECKING

import numpy as np
from traitlets.traitlets import link
from matplotlib import colors
from abtem.core import config
from abtem.core.axes import AxisMetadata

if TYPE_CHECKING:
    from abtem.visualize.visualizations import Visualization


try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

ipywidgets_not_installed = RuntimeError(
    "This functionality of abTEM requires ipywidgets, see "
    "https://ipywidgets.readthedocs.io/en/stable/user_install.html."
)


def _format_options(options):
    formatted_options = []
    for option in options:
        if isinstance(option, float):
            formatted_options.append(f"{option:.3f}")
        elif isinstance(option, tuple):
            formatted_options.append(
                ", ".join(tuple(f"{value:.3f}" for value in option))
            )
        else:
            formatted_options.append(option)

    return formatted_options


def slider_from_axes_metadata(
    axis_metadata: AxisMetadata,
    length,
    slider_type=None,
    continuous_update=None,
    default_value=None,
):
    if slider_type is None:
        slider_type = axis_metadata._default_type

    if continuous_update is None:
        continuous_update = config.get("visualize.continuous_update", False)

    values = np.array(axis_metadata.coordinates(length))
    options = _format_options(values)

    with config.set({"visualize.use_tex": False}):
        label = axis_metadata.format_label()

    if slider_type == "range":
        if default_value is None:
            default_value = (values[0], values[-1])

        index = (
            int(np.argmin(np.abs(values - default_value[0]))),
            int(np.argmin(np.abs(values - default_value[1]))),
        )

        slider = widgets.SelectionRangeSlider(
            description=label,
            options=options,
            continuous_update=continuous_update,
            index=index,
        )
    elif slider_type == "index":
        if default_value is None:
            default_value = 0

        # print(values, default_value)
        # print(int(np.argmin(np.abs(values - default_value))))

        try:
            index = int(np.argmin(np.abs(values - default_value)))
        except:
            index = 0

        slider = widgets.SelectionSlider(
            description=label,
            options=options,
            continuous_update=continuous_update,
            index=index,
        )
    else:
        raise ValueError("")

    return slider


def _get_max_range(array, axes_types):
    if np.iscomplexobj(array):
        array = np.abs(array)

    max_values = array.max(
        tuple(
            i for i, axes_type in enumerate(axes_types) if axes_type not in ("range",)
        )
    )

    positive_indices = np.where(max_values > 0)[0]

    if len(positive_indices) <= 1:
        max_value = np.max(max_values)
    else:
        max_value = np.sum(max_values[positive_indices])

    return max_value


def _make_vmin_vmax_slider(visualization):
    axes_types = (
        tuple(visualization._axes_types)
        + ("base",) * visualization.measurements.num_base_axes
    )

    max_value = _get_max_range(visualization.measurements.array, axes_types)
    min_value = -_get_max_range(-visualization.measurements.array, axes_types)

    step = (max_value - min_value) / 1e6

    vmin_vmax_slider = widgets.FloatRangeSlider(
        value=visualization._get_vmin_vmax(),
        min=min_value,
        max=max_value,
        step=step,
        disabled=visualization._autoscale,
        description="Normalization",
        continuous_update=True,
    )

    def vmin_vmax_slider_changed(change):
        vmin, vmax = change["new"]
        vmax = max(vmax, vmin + step)

        with vmin_vmax_slider.hold_trait_notifications():
            visualization._update_vmin_vmax(vmin, vmax)

    vmin_vmax_slider.observe(vmin_vmax_slider_changed, "value")
    return vmin_vmax_slider


class BaseGUI(widgets.HBox):
    def __init__(self, sliders, canvas, *args):
        self._sliders = sliders

        default_indices = [slider.index for slider in sliders]

        self._reset_button = widgets.Button(
            description="Reset sliders",
            disabled=False,
        )

        def reset(*args):
            for slider, index in zip(sliders, default_indices):
                slider.index = index

        self._reset_button.on_click(reset)

        self._continuous_update_toggle = widgets.ToggleButton(
            description="Continuous update",
            value=config.get("visualize.continuous_update", False),
        )
        for slider in sliders:
            link(
                (self._continuous_update_toggle, "value"),
                (slider, "continuous_update"),
            )

        self._scale_button = widgets.Button(
            description="Scale", layout=widgets.Layout(width="80px")
        )

        self._common_scale_button = widgets.Button(
            description="Common scale", layout=widgets.Layout(width="126px")
        )

        self._autoscale_button = widgets.ToggleButton(
            value=False,
            description="Autoscale",
            tooltip="Autoscale",
            layout=widgets.Layout(width="86px"),
        )

        self._powerscale_slider = widgets.FloatSlider(
            value=1.0,
            min=0.01,
            max=2.0,
            step=0.01,
            description="Power scale",
            tooltip="Power scale",
        )

        widgets_column = [
            *sliders,
            widgets.HBox([self._reset_button, self._continuous_update_toggle]),
            widgets.HBox(
                [self._scale_button, self._common_scale_button, self._autoscale_button]
            ),
            self._powerscale_slider,
        ]

        widgets_column = widgets_column + list(args)

        widgets_vbox = widgets.VBox(widgets_column)

        super().__init__(children=[widgets_vbox, canvas])

    def attach_visualization(self, visualization):
        def update_indices(change):
            indices = tuple(slider.index for slider in self.sliders)

            with ExitStack() as exit_stack:
                for slider in self.sliders:
                    exit_stack.enter_context(slider.hold_trait_notifications())

                visualization.update_data_indices(indices)
                if visualization.autoscale:
                    visualization.set_value_limits()

        for slider in self.sliders:
            slider.observe(update_indices, "value")

        self.scale_button.on_click(lambda *args: visualization.set_value_limits())

        self.common_scale_button.on_click(
            lambda *args: visualization.set_common_value_limits()
        )

        def autoscale_toggle_changed(change):
            setattr(visualization, "autoscale", change["new"])

        self.autoscale_button.observe(autoscale_toggle_changed, "value")
        self.autoscale_button.value = visualization.autoscale

        self.powerscale_slider.observe(
            lambda change: visualization.set_power(change["new"]), "value"
        )
        self.powerscale_slider.value = visualization.artists[0, 0].get_power()

    @property
    def sliders(self):
        return self._sliders

    @property
    def scale_button(self):
        return self._scale_button

    @property
    def common_scale_button(self):
        return self._common_scale_button

    @property
    def autoscale_button(self):
        return self._autoscale_button

    @property
    def powerscale_slider(self):
        return self._powerscale_slider


class LinesGUI(BaseGUI):
    def __init__(self, sliders, canvas):
        self._complex_dropdown = widgets.Dropdown(
            options=[
                ("Real and imaginary", "none"),
                ("Amplitude", "abs"),
                ("Intensity", "intensity"),
                ("Phase", "phase"),
            ],
            value="none",
            description="Complex visualization:",
        )

        super().__init__(sliders, canvas, self._complex_dropdown)

    @property
    def complex_dropdown(self):
        return self._complex_dropdown

    def attach_visualization(self, visualization):
        super().attach_visualization(visualization)

        self.complex_dropdown.observe(
            lambda change: visualization.set_complex_conversion(change["new"]), "value"
        )

        if not visualization.measurement.is_complex:
            self.complex_dropdown.disabled = True


class ImageGUI(BaseGUI):
    _default_cmap_options = [
        "default",
        "viridis",
        "magma",
        "gray",
        "jet",
        "hsluv",
        "hsv",
        "twilight",
    ]

    def __init__(self, sliders, canvas, cmap_options=None):
        self._complex_dropdown = widgets.Dropdown(
            options=[
                ("Domain coloring", "none"),
                ("Amplitude", "abs"),
                ("Intensity", "intensity"),
                ("Phase", "phase"),
                ("Real", "real"),
                ("Imaginary", "imag"),
            ],
            value="none",
            description="Complex visualization:",
        )

        if cmap_options is None:
            cmap_options = self._default_cmap_options

        self._cmap_dropdown = widgets.Dropdown(
            options=cmap_options,
            value=cmap_options[0],
            description="Colormap:",
        )

        super().__init__(sliders, canvas, self._complex_dropdown, self._cmap_dropdown)

    def attach_visualization(self, visualization):
        super().attach_visualization(visualization)

        def dropdown_changed(change):
            cmap = change["new"]
            if cmap == "default":
                cmap = None
            visualization.set_cmap(cmap)

        self.cmap_dropdown.observe(dropdown_changed, "value")

        self.complex_dropdown.observe(
            lambda change: visualization.set_complex_conversion(change["new"]), "value"
        )

        if not visualization.measurement.is_complex:
            self.complex_dropdown.disabled = True

    @property
    def cmap_dropdown(self):
        return self._cmap_dropdown

    @property
    def complex_dropdown(self):
        return self._complex_dropdown


class ScatterGUI(BaseGUI):
    _default_cmap_options = [
        "default",
        "viridis",
        "plasma",
        "cividis",
        "solid black",
        "jet",
    ]

    def __init__(self, sliders, canvas, cmap_options=None):

        self._scale_slider = widgets.FloatSlider(
            description="Point size",
        )

        self._annotations_slider = widgets.FloatLogSlider(
            description="Annotations",
            min=-7,
            max=0,
            value=1,
            step=1e-2,
            continuous_update=True,
        )

        if cmap_options is None:
            cmap_options = self._default_cmap_options

        self._cmap_dropdown = widgets.Dropdown(
            options=cmap_options,
            value=cmap_options[0],
            description="Colormap",
        )

        super().__init__(
            sliders,
            canvas,
            self._scale_slider,
            self._annotations_slider,
            self._cmap_dropdown,
        )

    def attach_visualization(self, visualization):
        super().attach_visualization(visualization)

        scale = visualization.artists[0, 0].get_scale()
        self.scale_slider.min = 0.0
        self.scale_slider.max = scale * 5.0
        self.scale_slider.step = scale * 0.05
        self.scale_slider.value = scale

        self.scale_slider.observe(
            lambda change: visualization.set_artists("scale", scale=change["new"]),
            "value",
        )

        if visualization.artists[0, 0].annotations is not None:
            annotation_threshold = visualization.artists[0, 0].annotations.threshold
            self.annotations_slider.value = annotation_threshold
            self.annotations_slider.observe(
                lambda change: visualization.set_artists(
                    "annotation_kwargs", threshold=change["new"]
                ),
                "value",
            )
        else:
            self.annotations_slider.disabled = True

        def dropdown_changed(change):
            cmap = None if change["new"] == "default" else change["new"]
            visualization.set_cmap(cmap)

        self.cmap_dropdown.observe(dropdown_changed, "value")

    @property
    def scale_slider(self):
        return self._scale_slider

    @property
    def annotations_slider(self):
        return self._annotations_slider

    @property
    def cmap_dropdown(self):
        return self._cmap_dropdown


def make_toggle_hkl_button(visualization):
    toggle_hkl_button = widgets.ToggleButton(description="Toggle hkl", value=False)

    def update_toggle_hkl_button(change):
        if change["new"]:
            visualization.set_miller_index_annotations()
        else:
            visualization.remove_miller_index_annotations()

    toggle_hkl_button.observe(update_toggle_hkl_button, "value")

    return toggle_hkl_button
