import ipywidgets as widgets
import numpy as np
from IPython.display import display
from bokeh import plotting, models
from bokeh.models import Slider
from bokeh.io import show, push_notebook
from abc import ABCMeta, abstractmethod
from bokeh.palettes import Category10


class InteractiveFigure(metaclass=ABCMeta):

    def __init__(self, bokeh_figure, push_notebook=False):
        self._figure = bokeh_figure
        self._push_notebook = push_notebook

    def show(self):
        if self._push_notebook:
            show(self._figure, notebook_handle=True)
        else:
            show(self._figure)

    @abstractmethod
    def update(self, measurement):
        pass


class InteractivePlot(InteractiveFigure):

    def __init__(self, plot_width=400, plot_height=400, push_notebook=False):
        figure = plotting.Figure(plot_width=plot_width,
                                 plot_height=plot_height,
                                 title=None,
                                 toolbar_location='below',
                                 tools='pan,wheel_zoom,box_zoom,reset')

        self._source = models.ColumnDataSource(data=dict(xs=[], ys=[], colors=[], legend_label=[]))

        self._model = figure.multi_line(xs='xs',
                                        ys='ys',
                                        line_width=2,
                                        line_color='colors',
                                        legend_field='legend_label',
                                        #legend_label='sa',
                                        source=self._source)

        #figure.add_glyph(self._source, glyph=self._model)

        super().__init__(bokeh_figure=figure, push_notebook=push_notebook)

    def update(self, measurements):
        xs = []
        ys = []
        labels = []
        for measurement in measurements:
            calibration = measurement.calibrations[0]
            array = measurement.array
            x = np.linspace(calibration.offset, calibration.offset + len(array) * calibration.sampling, len(array))
            xs += [x]
            ys += [array]
            labels += [measurement.name]

        colors = Category10[10][:len(xs)]

        self._source.data = {'xs': xs, 'ys': ys, 'colors': colors, 'legend_label': labels}

        self._figure.legend.location = 'top_left'
        self._figure.legend.click_policy = 'hide'

        if self._push_notebook:
            push_notebook()


class InteractiveImage(InteractiveFigure):

    def __init__(self, plot_width=400, plot_height=400, push_notebook=False):
        figure = plotting.Figure(plot_width=plot_width,
                                 plot_height=plot_height,
                                 title=None,
                                 toolbar_location='below',
                                 tools='pan,wheel_zoom,box_zoom,reset',
                                 match_aspect=True)

        self._source = models.ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))

        self._model = models.Image(image='image', x='x', y='y', dw='dw', dh='dh', palette='Greys256')

        figure.add_glyph(self._source, glyph=self._model)

        super().__init__(bokeh_figure=figure, push_notebook=push_notebook)

    def update(self, measurement):
        measurement = measurement.squeeze()
        calibrations = measurement.calibrations

        array = measurement.array
        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

        self._source.data = {'image': [array],
                             'x': [calibrations[1].offset],
                             'y': [calibrations[0].offset],
                             'dw': [calibrations[1].sampling * measurement.array.shape[1]],
                             'dh': [calibrations[0].sampling * measurement.array.shape[0]]}

        if self._push_notebook:
            push_notebook()


def interact_object(obj, **kwargs):
    def create_callback(attr):
        def callback(_, __, new):
            setattr(obj, attr, new)

        return callback

    sliders = []
    for key, value in kwargs.items():
        slider = Slider(value=getattr(obj, key),
                        start=value[0],
                        end=value[1],
                        step=value[2],
                        title=key)

        slider.on_change('value', create_callback(key))
        sliders.append(slider)

    return sliders

# def interact_object(obj, continuous_update=True, **kwargs):
#    create_callback = lambda key: lambda change: setattr(obj, key, change['new'])
#    for key, value in kwargs.items():
#        slider = widgets.FloatSlider(value=getattr(obj, key),
#                                     min=value[0],
#                                     max=value[1],
#                                     step=value[2],
#                                     description=key,
#                                     continuous_update=continuous_update)
#
#        slider.observe(create_callback(key), 'value')
#        display(slider)
