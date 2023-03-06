from contextlib import ExitStack
from typing import List, Tuple, TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.colors as colors
import numpy as np


from abtem.visualize import set_image_data, _iterate_axes
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from abtem.measurements import BaseMeasurement


def format_options(options):
    return [
        f"{option:.3f}" if isinstance(option, float) else option for option in options
    ]


def _set_alpha(ims):
    if len(ims) > 1:
        alpha = ims[0].norm(ims[0].get_array())
        alpha = np.clip(alpha, a_min=0, a_max=1)
        ims[1].set_alpha(alpha)


def _make_update_scale_handler(fig, vmin_vmax_slider):
    def update_scale_handler(change):
        axes = fig.get_axes()[0]

        for ax in _iterate_axes(axes):
            ims = ax.get_images()

            array = ims[0].get_array()
            vmin = array.min()
            vmax = array.max()
            ims[0].set_clim(vmin, vmax)

            _set_alpha(ims)

            with vmin_vmax_slider.hold_trait_notifications():
                min_value = min(0.0, vmin)
                max_value = max(vmax, 0.0)

                vmin_vmax_slider.min = min_value
                vmin_vmax_slider.max = max_value
                vmin_vmax_slider.step = (max_value - min_value) / 1000.0
                vmin_vmax_slider.value = [vmin, vmax]

    return update_scale_handler


def _index_from_sliders(measurements, sliders):
    indices = []
    sum_axes = []
    i = 0
    for slider in sliders:
        if slider is None:
            indices.append(slice(None))
            i += 1
        elif isinstance(slider.index, int):
            indices.append(slider.index)
        elif isinstance(slider.index, tuple):
            indices.append(slice(*slider.index))
            sum_axes.append(i)
            i += 1
        else:
            raise RuntimeError

    return measurements[tuple(indices)].sum(axes=sum_axes)


def _make_update_image_handler(
    fig,
    measurements,
    sliders,
    complex_representation_dropdown,
    autoscale_button,
    vmin_vmax_slider,
):

    update_scale_handler = _make_update_scale_handler(fig, vmin_vmax_slider)

    def update_image_data_handler(change):
        axes = fig.get_axes()[0]

        indexed_measurements = _index_from_sliders(measurements, sliders)

        recreate = False
        if change["owner"] is complex_representation_dropdown:
            if change["old"] == "domain_coloring" or change["new"] == "domain_coloring":
                recreate = True

        if complex_representation_dropdown is not None:
            complex_representation = complex_representation_dropdown.value
        else:
            complex_representation = None

        with ExitStack() as stack:
            for slider in sliders:
                slider.hold_trait_notifications()

            set_image_data(
                axes,
                indexed_measurements,
                complex_representation=complex_representation,
                recreate=recreate,
            )

            if autoscale_button.value:
                update_scale_handler({})

    return update_image_data_handler


def _make_autoscale_handler(fig, vmin_vmax_slider):
    update_scale_handler = _make_update_scale_handler(fig, vmin_vmax_slider)

    def autoscale_button_changed(change):
        with vmin_vmax_slider.hold_trait_notifications():
            if change["new"]:
                update_scale_handler({})
            vmin_vmax_slider.disabled = change["new"]

    return autoscale_button_changed


def _make_vmin_vmax_handler(fig, vmin_vmax_slider):
    def vmin_vmax_slider_handler(change):
        axes = fig.get_axes()[0]

        for ax in _iterate_axes(axes):
            ims = ax.get_images()
            ims[0].set_clim(*vmin_vmax_slider.value)

            _set_alpha(ims)

    return vmin_vmax_slider_handler


def _make_power_scale_handler(fig, power_slider, vmin_vmax_slider):
    def power_slider_handler(change):
        axes = fig.get_axes()[0]

        for ax in _iterate_axes(axes):
            ims = ax.get_images()
            vmin = vmin_vmax_slider.value[0]
            vmax = vmin_vmax_slider.value[1]
            ims[0].norm = colors.PowerNorm(
                vmin=vmin, vmax=vmax, gamma=power_slider.value
            )

            _set_alpha(ims)

    return power_slider_handler


def _make_sliders(
    measurements: List["BaseMeasurement"],
    continuous_update: bool,
    indexing_axes: List[Tuple[int, ...]],
    integration_axes: List[Tuple[int, ...]],
):

    sliders = []
    for measurement, measurement_indexing_axes, measurement_integration_axes in zip(
        measurements, indexing_axes, integration_axes
    ):
        sliders.append([])
        for i in range(len(measurement.ensemble_shape)):

            shape = measurement.ensemble_shape[i]
            axes_metadata = measurement.ensemble_axes_metadata[i]
            label = axes_metadata.format_label()
            options = format_options(axes_metadata.coordinates(shape))

            if i in measurement_indexing_axes:
                sliders[-1].append(
                    widgets.SelectionSlider(
                        description=label,
                        options=options,
                        continuous_update=continuous_update,
                    )
                )
            elif i in measurement_integration_axes:
                sliders[-1].append(
                    widgets.SelectionRangeSlider(
                        description=label,
                        options=options,
                        continuous_update=continuous_update,
                        index=(0, len(options) - 1),
                    )
                )
            else:
                sliders[-1].append(None)
    return sliders


def _make_scale_widgets():
    scale_button = widgets.Button(description="Scale once")

    autoscale_button = widgets.ToggleButton(
        value=True,
        description="Autoscale",
        tooltip="Autoscale",
    )
    return widgets.HBox([scale_button, autoscale_button])


def interact_ensemble_indices_2d(
    fig,
    measurements,
    sliders,
    continuous_update: bool = True,
    scale_widget=None,
    complex_representation="domain_coloring",
):

    vbox_children = []

    if measurements.is_complex:
        complex_representation_dropdown = widgets.Dropdown(
            description="Complex representation",
            options=[
                "intensity",
                "abs",
                "phase",
                "domain_coloring",
            ],
            value=complex_representation,
        )
    else:
        complex_representation_dropdown = None

    if scale_widget is None:
        scale_widget = _make_scale_widgets()
        vbox_children = vbox_children + [scale_widget]

    vmin_vmax_slider = widgets.FloatRangeSlider(
        value=[0, 1],
        min=0,
        max=1,
        step=0.0,
        disabled=True,
        description="Normalization",
        continuous_update=continuous_update,
        readout_format=".2e",
    )

    power_scale_slider = widgets.FloatSlider(
        min=1e-3, max=2, value=1.0, description="Power"
    )

    update_image_handler = _make_update_image_handler(
        fig,
        measurements,
        sliders,
        complex_representation_dropdown,
        scale_widget.children[1],
        vmin_vmax_slider,
    )

    update_scale_handler = _make_update_scale_handler(
        fig,
        vmin_vmax_slider,
    )

    update_autoscale_handler = _make_autoscale_handler(fig, vmin_vmax_slider)

    update_vmin_vmax_handler = _make_vmin_vmax_handler(fig, vmin_vmax_slider)
    update_power_scale_handler = _make_power_scale_handler(
        fig, power_scale_slider, vmin_vmax_slider
    )

    scale_widget.children[0].on_click(update_scale_handler)
    scale_widget.children[1].observe(update_autoscale_handler, "value")
    vmin_vmax_slider.observe(update_vmin_vmax_handler, names="value")
    power_scale_slider.observe(update_power_scale_handler, "value")

    update_scale_handler({})

    for slider in sliders:
        if slider is not None:
            slider.observe(update_image_handler, names="value")

    if complex_representation_dropdown is not None:
        complex_representation_dropdown.observe(update_image_handler, names="value")
        vbox_children = vbox_children + [complex_representation_dropdown]  # noqa

    vbox_children = vbox_children + [
        vmin_vmax_slider,
        power_scale_slider,
    ]

    return widgets.VBox(vbox_children)


def _make_update_limits_handler(fig):
    def update_limits_handler(change):
        axes = fig.get_axes()[0]

        for ax in _iterate_axes(axes):
            ax = fig.get_axes()[0]

            ymin = min([min(line.get_ydata()) for line in ax.get_lines()])
            ymax = max([max(line.get_ydata()) for line in ax.get_lines()])
            yptp = ymax - ymin

            xmin = min([min(line.get_xdata()) for line in ax.get_lines()])
            xmax = max([max(line.get_xdata()) for line in ax.get_lines()])
            xptp = xmax - xmin

            ax.set_ylim([ymin - yptp * 0.05, ymax + yptp * 0.05])
            ax.set_xlim([xmin - xptp * 0.05, xmax + xptp * 0.05])

    return update_limits_handler


def _make_update_lines_handler(
    fig,
    measurements,
    sliders,
):
    update_limits_handler = _make_update_limits_handler(fig)

    def update_lines_data_handler(change):
        axes = fig.get_axes()[0]

        indexed_measurements = _index_from_sliders(measurements, sliders)
        indexed_measurements = indexed_measurements.compute().to_cpu()

        if indexed_measurements.ensemble_shape:
            for line, measurement in zip(axes.get_lines(), indexed_measurements):
                line.set_ydata(measurement.array)
        else:
            axes.get_lines()[0].set_ydata(indexed_measurements.array)

        # if scale_widget.children[1].value:
        #     update_limits_handler({})

    return update_lines_data_handler


def interact_ensemble_indices_1d(
    fig,
    measurements,
    sliders,
):

    scale_widget = _make_scale_widgets()

    update_lines_handler = _make_update_lines_handler(fig, measurements, sliders)

    update_limits_handler = _make_update_limits_handler(fig)

    scale_widget.children[0].on_click(update_limits_handler)
    scale_widget.children[1].observe(update_limits_handler, names="value")

    for slider in sliders:
        slider.observe(update_lines_handler, names="value")

    vbox_children = widgets.VBox([scale_widget])

    return vbox_children


def depth(l):
    if isinstance(l, (list, tuple)):
        return 1 + max(depth(item) for item in l)
    else:
        return 0


def interact(
    measurements,
    continuous_update: bool = True,
    indexing_axes: tuple = None,
    integration_axes: tuple = None,
    **kwargs,
):
    if widgets is None:
        raise RuntimeError()

    if not isinstance(measurements, list):
        measurements = [measurements]

    if integration_axes is None:
        integration_axes = [()] * len(measurements)
    elif depth(integration_axes) == 1:
        integration_axes = [integration_axes]

    if indexing_axes is None:
        indexing_axes = [
            tuple(i for i in range(len(measurement.ensemble_shape)) if not i in axes)
            for axes, measurement in zip(integration_axes, measurements)
        ]
    elif depth(indexing_axes) == 1:
        indexing_axes = [indexing_axes]

    sliders = _make_sliders(
        measurements,
        continuous_update=continuous_update,
        indexing_axes=indexing_axes,
        integration_axes=integration_axes,
    )
    gui = [] + sliders[0]
    figs = []

    complex_representation = "domain_coloring"

    for i, measurement in enumerate(measurements):

        # assert len(measurement.ensemble_shape) == len(at)

        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.ion()

        figs.append(fig)

        if measurement._base_dims == 1:
            indexed_measurements = _index_from_sliders(measurement, sliders[0])

            fig, ax = indexed_measurements.show(ax=ax, **kwargs)

            if len(measurements) > 1:
                gui.append(widgets.Label(f"Figure {fig.number}"))

            gui.append(
                interact_ensemble_indices_1d(
                    fig,
                    measurement,
                    sliders=sliders[0],
                )
            )

        elif measurement._base_dims == 2:
            fig, ax = measurement.show(ax=ax, **kwargs)

            if len(measurements) > 1:
                gui.append(widgets.Label(f"Figure {fig.number}"))

            gui.append(
                interact_ensemble_indices_2d(
                    fig,
                    measurement,
                    continuous_update=continuous_update,
                    sliders=sliders[0],
                    scale_widget=None,
                    complex_representation=complex_representation,
                )
            )

        fig.tight_layout()

    app_layout = widgets.AppLayout(
        center=widgets.HBox(
            [widgets.HBox([fig.canvas], width="300px") for fig in figs]
        ),
        left_sidebar=widgets.VBox(gui),
        pane_heights=[0, 6, 0],
        justify_items="left",
        pane_widths=["310px", 1, 0],
    )

    return app_layout
