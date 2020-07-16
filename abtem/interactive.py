import ipywidgets as widgets
import numpy as np
from IPython.display import display
from bokeh import plotting, models
from bokeh.io import show, push_notebook


def link_widget(widget, obj, property_name):
    def callback(change):
        setattr(obj, property_name, change['new'])

    widget.observe(callback, 'value')
    return callback


class BokehImage:

    def __init__(self, callback, push_notebook=False):
        self._figure = plotting.Figure(plot_width=400, plot_height=400, title=None,
                                       toolbar_location='below', tools='pan,wheel_zoom,box_zoom,reset',
                                       match_aspect=True)
        self._source = models.ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self._model = models.Image(image='image', x='x', y='y', dw='dw', dh='dh', palette='Greys256')
        self._glyph = self._figure.add_glyph(self._source, glyph=self._model)
        if push_notebook:
            self._notebook_handle = show(self._figure, notebook_handle=True)
        else:
            self._notebook_handle = None
        self._callback = callback

    def update(self, *args, **kwargs):
        measurement = self._callback()
        array = measurement.array

        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        self._source.data = {'image': [array],
                             'x': [measurement.calibrations[1].offset],
                             'y': [measurement.calibrations[0].offset],
                             'dw': [measurement.calibrations[1].sampling * measurement.array.shape[1]],
                             'dh': [measurement.calibrations[0].sampling * measurement.array.shape[0]]}

        if self._notebook_handle:
            push_notebook(handle=self._notebook_handle)


def interact_object(obj, continuous_update=True, **kwargs):
    create_callback = lambda key: lambda change: setattr(obj, key, change['new'])
    for key, value in kwargs.items():
        slider = widgets.FloatSlider(value=getattr(obj, key),
                                     min=value[0],
                                     max=value[1],
                                     step=value[2],
                                     description=key,
                                     continuous_update=continuous_update)

        slider.observe(create_callback(key), 'value')
        display(slider)
