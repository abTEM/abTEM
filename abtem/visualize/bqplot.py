import ipywidgets as widgets
import numpy as np
from bqplot import Figure, LinearScale, Axis, ColorScale, Toolbar, Lines
from bqplot_image_gl import ImageGL
import bqplot.pyplot as plt
from abtem.visualize.utils import format_label
from matplotlib.colors import TABLEAU_COLORS
from abtem.visualize.widgets import throttle


def update_bqplot_image(image, measurement):
    calibrations = measurement.calibrations

    image.x = [calibrations[-2].offset,
               calibrations[-2].offset + calibrations[-2].sampling * measurement.array.shape[-2]]

    image.y = [calibrations[-1].offset,
               calibrations[-1].offset + calibrations[-1].sampling * measurement.array.shape[-1]]

    array = measurement.array[(0,) * (measurement.dimensions - 2) + (slice(None),) * 2]

    image.image = array.T
    image.scales['image'].min = float(array.min())
    image.scales['image'].max = float(array.max())

    L = max(image.x[1], image.y[1])
    image.scales['x'].min = image.x[0]
    image.scales['x'].max = L
    image.scales['y'].min = image.y[0]
    image.scales['y'].max = L


def create_image_figure():
    scale_x = LinearScale(allow_padding=False)
    scale_y = LinearScale(allow_padding=False)

    scales = {'x': scale_x, 'y': scale_y}

    axis_x = Axis(scale=scales['x'])
    axis_y = Axis(scale=scales['y'], orientation='vertical')

    figure = Figure(scales=scales,
                    axes=[axis_x, axis_y],
                    min_aspect_ratio=1,
                    max_aspect_ratio=1,
                    fig_margin={'top': 0, 'bottom': 50, 'left': 50, 'right': 0})

    figure.layout.height = '400px'
    figure.layout.width = '400px'

    scales_image = {'x': scale_x,
                    'y': scale_y,
                    'image': ColorScale(scheme='viridis')}
    # 'image': ColorScale(colors=['black', 'white'])}

    image = ImageGL(image=np.zeros((0, 0)), scales=scales_image)

    figure.marks = (image,)

    return figure, image


def show_measurement_2d(measurement_or_func):
    try:
        measurement = measurement_or_func()
        return_callback = True

    except TypeError:
        measurement = measurement_or_func
        return_callback = False

    figure, image = create_image_figure()

    toolbar = Toolbar(figure=figure)

    figure.axes[0].label = format_label(measurement.calibrations[-2])
    figure.axes[1].label = format_label(measurement.calibrations[-1])

    figure.marks = (image,)

    if return_callback:
        update_bqplot_image(image, measurement)

        def callback(*args, **kwargs):
            update_bqplot_image(image, measurement_or_func())

        return widgets.VBox([figure, toolbar]), callback
    else:
        return widgets.VBox([figure, toolbar])


def show_measurement_1d(measurements_or_func, figure=None, throttling=False, **kwargs):
    if figure is None:
        figure = plt.figure(fig_margin={'top': 0, 'bottom': 50, 'left': 50, 'right': 0})

        figure.layout.height = '250px'
        figure.layout.width = '300px'

    try:
        measurements = measurements_or_func()
        return_callback = True

    except TypeError:
        measurements = measurements_or_func
        return_callback = False

    lines = []
    for measurement, color in zip(measurements, TABLEAU_COLORS.values()):
        calibration = measurement.calibrations[0]
        array = measurement.array
        x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))

        line = plt.plot(x, array, colors=[color], **kwargs)
        figure.axes[0].label = format_label(measurement.calibrations[0])
        lines.append(line)

    # figure.axes[1].label = format_label(measurement)

    if return_callback:

        @throttle(throttling)
        def callback(*args, **kwargs):
            for line, measurement in zip(lines, measurements_or_func()):
                x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))
                line.x = x
                line.y = measurement.array

        return figure, callback
    else:
        return figure


def create_reset_button(figure):
    def reset_zoom(*args):
        figure.axes[0].scale.min = None
        figure.axes[1].scale.min = None
        figure.axes[0].scale.max = None
        figure.axes[1].scale.max = None

    reset_button = widgets.Button(description='Reset')
    reset_button.on_click(reset_zoom)
    return reset_button
