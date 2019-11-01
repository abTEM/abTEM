import io

import PIL.Image
import bqplot
import ipywidgets
import numpy as np
from bqplot import LinearScale, Axis, Figure, PanZoom

from abtem.bases import SelfObservable, notifying_property
from abtem.utils import linspace, fftfreq
import numbers


def link_widget(widget, o, property_name):
    def callback(change):
        setattr(o, property_name, change['new'])

    widget.observe(callback, 'value')
    return callback


def link_widget_component(widget, o, property_name, component):
    def callback(change):
        value = getattr(o, property_name).copy()
        value[component] = change['new']
        setattr(o, property_name, value)

    widget.observe(callback, 'value')
    return callback


def property_slider(o, property_name, description=None, components=None, **kwargs):
    if description is None:
        description = property_name
    value = getattr(o, property_name)

    if isinstance(components, numbers.Number):
        components = (components,)

    if components is not None:
        value = value[components[0]]

    slider = ipywidgets.FloatSlider(description=description, value=value, **kwargs)

    if components is None:
        link_widget(slider, o, property_name)
    else:
        for component in components:
            link_widget_component(slider, o, property_name, component)
    return slider


class UpdatingMark(SelfObservable):

    def __init__(self, observables, auto_update=True, **kwargs):
        self._observables = observables
        self._auto_update = auto_update
        for observable in observables:
            observable.register_observer(self)

        super().__init__(**kwargs)

    def update(self):
        raise NotImplementedError()

    def notify(self, observable, message):
        if self._auto_update:
            self.update()


class UpdatingLine(UpdatingMark):

    def __init__(self, scales, observables, func, auto_update=True, **kwargs):
        super().__init__(observables, auto_update=auto_update)
        self._func = func
        self._mark = bqplot.Lines(scales=scales, **kwargs)
        self.update()

    def update(self):
        lineprofile = self._func()
        if lineprofile.space == 'direct':
            x = linspace(lineprofile)
        elif lineprofile.space == 'fourier':
            x = np.fft.fftshift(fftfreq(lineprofile))
        else:
            raise RuntimeError()
        y = lineprofile.array
        self._mark.x = x
        self._mark.y = y


def array_to_png(array):
    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    image = PIL.Image.fromarray(np.flipud(array.T))
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='png')
    return ipywidgets.Image(value=bytes_io.getvalue())


class UpdatingImage(UpdatingMark):

    def __init__(self, scales, observables, func, auto_update=True, color_scale='linear', **kwargs):

        self._func = func
        self._color_scale = color_scale
        self._mark = bqplot.Image(image=array_to_png(func().array), scales=scales, **kwargs)
        self.update()

        super().__init__(observables, auto_update=auto_update)

    def update(self):
        array = self._func().array

        if self.color_scale == 'log':
            sign = np.sign(array)
            array = sign * np.log(1 + .005 * np.abs(array))

        elif self.color_scale != 'linear':
            raise RuntimeError()

        self._mark.image = array_to_png(array)

    color_scale = notifying_property('_color_scale')


class InteractiveFigure:

    def __init__(self, height='400px', width='400px', margin=None, **kwargs):
        self._scales = {'x': LinearScale(), 'y': LinearScale()}
        self._axes = {'x': Axis(scale=self._scales['x']),
                      'y': Axis(scale=self._scales['y'], orientation='vertical')}

        panzoom = PanZoom(scales={'x': [self._scales['x']], 'y': [self._scales['y']]})

        if margin is None:
            margin = {'top': 0, 'bottom': 40, 'left': 60, 'right': 0}

        self._figure = Figure(marks=[], axes=list(self._axes.values()), interaction=panzoom, fig_margin=margin,
                              **kwargs)

        self._figure.layout.height = height
        self._figure.layout.width = width

    @property
    def scales(self):
        return self._scales

    @property
    def axes(self):
        return self._axes

    @property
    def figure(self):
        return self._figure

    @property
    def marks(self):
        return self._figure.marks

    def add_line(self, observables, func, **kwargs):
        ul = UpdatingLine(self._scales, observables, func, **kwargs)
        self._figure.marks = self._figure.marks + [ul._mark]
        return ul

    def add_image(self, observables, func, **kwargs):
        image = UpdatingImage(self._scales, observables, func, **kwargs)
        self._figure.marks = self._figure.marks + [image._mark]
        return image
