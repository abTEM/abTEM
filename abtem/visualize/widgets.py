"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from __future__ import annotations

from typing import Sequence, Any

import numpy as np
from traitlets.traitlets import link
from matplotlib import colors
from abtem.core import config

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


def _make_update_selection_sliders_button(sliders, indices):
    reset_button = widgets.Button(
        description="Reset sliders",
        disabled=False,
    )

    def reset(*args):
        for slider, index in zip(sliders, indices):
            slider.index = index

    reset_button.on_click(reset)

    return reset_button


def _make_update_indices_function(visualization, sliders):
    def update_indices(change):
        indices = ()
        for slider in sliders:
            idx = slider.index
            if isinstance(idx, tuple):
                idx = slice(*idx)
            indices += (idx,)

        if not sliders:
            return

        with sliders[0].hold_trait_notifications():
            visualization.set_ensemble_indices(indices)
            if visualization._autoscale:
                visualization.set_values_lim()

    return update_indices


def make_continuous_update_button(sliders: list, continuous_update=None):
    if continuous_update is None:
        continuous_update = config.get("visualize.continuous_update", False)

    continuous_update_checkbox = widgets.ToggleButton(
        description="Continuous update", value=continuous_update
    )
    for slider in sliders:
        link((continuous_update_checkbox, "value"), (slider, "continuous_update"))
    return continuous_update_checkbox


def make_sliders_from_ensemble_axes(
    visualizations,
    axes_types: Sequence[str, ...] = None,
    continuous_update: bool = None,
    callbacks: tuple[callable, ...] = (),
    default_values: Any = None,
):
    if continuous_update is None:
        continuous_update = config.get("visualize.continuous_update", False)

    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    ensemble_axes_metadata = visualizations[0].ensemble_axes_metadata
    ensemble_shape = visualizations[0].ensemble_shape
    if axes_types is None:
        axes_types = [metadata._default_type for metadata in ensemble_axes_metadata]
    elif not (len(axes_types) == len(ensemble_shape)):
        raise ValueError()

    for visualization in visualizations[1:]:
        if not (
            (
                visualization.measurements.ensemble_axes_metadata
                == ensemble_axes_metadata
            )
            and (visualization.measurements.ensemble_shape == ensemble_shape)
        ):
            raise ValueError()

    if not isinstance(default_values, Sequence):
        default_values = (default_values,) * len(ensemble_shape)

    elif not (len(default_values) == len(ensemble_shape)):
        raise ValueError()

    sliders = []
    default_indices = []
    for axes_metadata, n, axes_type, default_value in zip(
        ensemble_axes_metadata, ensemble_shape, axes_types, default_values
    ):
        values = np.array(axes_metadata.coordinates(n))
        options = _format_options(values)

        if default_value is None:
            index = 0
        else:
            index = int(np.argmin(np.abs(values - default_value)))

        default_indices.append(index)

        with config.set({"visualize.use_tex": False}):
            label = axes_metadata.format_label()

        if axes_type == "range":
            sliders.append(
                widgets.SelectionRangeSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=(0, len(options) - 1),
                )
            )
        elif axes_type == "index" or axes_type is None:
            sliders.append(
                widgets.SelectionSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=index,
                )
            )

    for visualization in visualizations:
        update_indices = _make_update_indices_function(visualization, sliders)

        update_indices({})
        for slider in sliders:
            slider.observe(update_indices, "value")
            for callback in callbacks:
                slider.observe(callback, "value")

    reset_button = _make_update_selection_sliders_button(
        sliders, indices=default_indices
    )

    continuous_update_button = make_continuous_update_button(sliders, continuous_update)

    return sliders, reset_button, continuous_update_button


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


def make_scale_button(
    visualizations,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def scale_button_clicked(*args):
        for visualization in visualizations:
            visualization.set_values_lim()

    scale_button = widgets.Button(description="Scale")
    scale_button.on_click(scale_button_clicked)
    return scale_button


def make_autoscale_button(
    visualizations,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def autoscale_button_changed(change):
        for visualization in visualizations:
            if change["new"]:
                visualization.autoscale = True
            else:
                visualization.autoscale = False

    autoscale_button = widgets.ToggleButton(
        value=visualizations[0].autoscale,
        description="Autoscale",
        tooltip="Autoscale",
    )
    autoscale_button.observe(autoscale_button_changed, "value")
    return autoscale_button


def make_power_scale_slider(
    visualizations,
    **kwargs,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def powerscale_slider_changed(change):
        for visualization in visualizations:
            visualization.set_power(change["new"])

    default_kwargs = {
        "min": 0.01,
        "max": 2.0,
        "step": 0.01,
        "description": "Power",
        "tooltip": "Power scale",
    }

    kwargs = {**default_kwargs, **kwargs}

    power = 1.0
    # TODO
    # for norm in visualizations[0]._normalization.ravel():
    #     if isinstance(norm, colors.PowerNorm):
    #         if power is None:
    #             power = norm.gamma
    #         else:
    #             power = min(power, norm.gamma)
    #     else:
    #         if power is None:
    #             power = 1.0
    #         else:
    #             power = min(power, 1.0)

    power_scale_slider = widgets.FloatSlider(value=power, **kwargs)
    power_scale_slider.observe(powerscale_slider_changed, "value")
    return power_scale_slider


def make_complex_visualization_dropdown(
    visualizations,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def dropdown_changed(change):
        for visualization in visualizations:
            visualization.set_complex_conversion(change["new"])

    dropdown = widgets.Dropdown(
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

    dropdown.observe(dropdown_changed, "value")
    return dropdown


def make_cmap_dropdown(
    visualizations,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def dropdown_changed(change):
        cmap = change["new"]
        if cmap == "default":
            cmap = None
        for visualization in visualizations:
            visualization.set_cmaps(cmap)

    options = ["default", "viridis", "magma", "gray", "jet", "hsluv", "hsv", "twilight"]

    dropdown = widgets.Dropdown(
        options=options,
        value="default",
        description="Colormap:",
    )

    dropdown.observe(dropdown_changed, "value")

    return dropdown
