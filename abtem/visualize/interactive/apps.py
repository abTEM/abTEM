import ipywidgets as widgets
import numpy as np
from traitlets import HasTraits, observe, Dict, default, List, link, Float
from traitlets import validate, Any, Unicode
from traittypes import Array

from abtem.visualize.interactive.artists import ImageArtist, LinesArtist
from abtem.visualize.interactive.canvas import Canvas


class ArrayView(HasTraits):
    array = Array()

    def __init__(self, array, artist, navigation_axes=None, canvas=None, **kwargs):
        if canvas is None:
            self._canvas = Canvas()
            self._canvas.artists = {'array_view': artist}
        else:
            self._canvas = canvas

        self._data_dims = len(array.shape)
        self._navigation_axes = navigation_axes

        self._sliders = []
        for i in range(self.navigation_dims):
            description = f'Axis {navigation_axes[i]}'
            slider = widgets.IntSlider(min=0, description=description)

            slider.observe(self._update_view, 'value')
            self._sliders.append(slider)

        super().__init__(array=array, **kwargs)

    @property
    def navigation_dims(self):
        return len(self.navigation_axes)

    @property
    def data_dims(self):
        return self._data_dims

    @property
    def navigation_axes(self):
        return self._navigation_axes

    @property
    def display_axes(self):
        return [i for i in range(self._data_dims) if not i in self._navigation_axes]

    @validate('array')
    def _validate_array(self, proposal):
        if not len(proposal['value'].shape) == self.data_dims:
            raise ValueError()
        return proposal['value']

    def _make_slice(self):
        indices = [slider.value for slider in self._sliders]

        s = [slice(None)] * len(self.array.shape)
        for axis, index in zip(self.navigation_axes, indices):
            s[axis] = index

        return tuple(s)

    @observe('array')
    def _observe_array(self, change):
        self._update_sliders()
        self._update_view()

    def _update_view(self, _=None):
        raise NotImplementedError()

    def _update_sliders(self, _=None):
        for i, slider in enumerate(self._sliders):
            max_index = self.array.shape[self.navigation_axes[i]] - 1
            slider.value = min(slider.value, max_index)
            slider.max = max_index

    def update(self):
        self._update_view()
        self._update_sliders()

    @property
    def sliders(self):
        return widgets.HBox([slider for slider in self._sliders])


class ArrayView2d(ArrayView):
    extent = List(allow_none=True)
    x_label = Unicode()
    y_label = Unicode()

    def __init__(self, array, navigation_axes=None, **kwargs):
        self._image_artist = ImageArtist()

        if navigation_axes is None:
            navigation_axes = list(range(len(array.shape) - 2))

        super().__init__(array=array,
                         navigation_axes=navigation_axes,
                         artist=self._image_artist,
                         **kwargs)

        link((self, 'x_label'), (self._canvas.x_axis, 'label'))
        link((self, 'y_label'), (self._canvas.y_axis, 'label'))

    @default('extent')
    def _default_extent(self):
        return None

    @observe('extent')
    def _observe_extent(self, change=None):
        self._image_artist.extent = self.extent
        self._canvas.adjust_to_artists()

    def _update_view(self, _=None):
        self._image_artist.image = self.array[self._make_slice()]


class ArrayView1d(ArrayView):
    extent = List(allow_none=True)
    x_label = Unicode()
    y_label = Unicode()

    def __init__(self, array, navigation_axes=None, **kwargs):
        self._lines_artist = LinesArtist()

        if navigation_axes is None:
            navigation_axes = list(range(len(array.shape) - 1))

        super().__init__(array=array,
                         navigation_axes=navigation_axes,
                         artist=self._lines_artist,
                         **kwargs)

        link((self, 'x_label'), (self._canvas.x_axis, 'label'))
        link((self, 'y_label'), (self._canvas.y_axis, 'label'))

    @default('extent')
    def _default_extent(self):
        return None

    def _update_view(self, _=None):
        self._lines_artist.y = self.array[self._make_slice()]

        if self.extent is None:
            extent = [0, len(self._lines_artist.y)]
        else:
            extent = self.extent

        self._lines_artist.x = np.linspace(extent[0], extent[1], len(self._lines_artist.y))


# class MeasurementView(HasTraits):
#
#     def __init__(self, array_view, **kwargs):
#         self._array_view = array_view
#
#         super().__init__(**kwargs)
#
#     @property
#     def canvas(self):
#         return self._array_view._canvas
#
#     @property
#     def figure(self):
#         return self.canvas.figure
#
#     @property
#     def artist(self):
#         return self.canvas.artists['array_view']


class MeasurementView1d(HasTraits):
    measurements = Dict()

    def __init__(self, measurements=None, navigation_axes=None, **kwargs):

        if measurements is None:
            array = np.zeros((0,))
        else:
            array = measurements.array

        canvas = Canvas()

        self._array_view = ArrayView1d(array=array, navigation_axes=navigation_axes)

        super().__init__(array_view=array_view, measurement=measurement, **kwargs)

    @property
    def canvas(self):
        return self._array_view._canvas

    @property
    def figure(self):
        return self.canvas.figure

    @property
    def artist(self):
        return self.canvas.artists['array_view']

    @observe('measurements')
    def _observe_measurements(self, change):
        extent = self.measurement.calibration_limits
        units = self.measurement.calibration_units
        names = self.measurement.calibration_names

        with self._array_view.hold_trait_notifications():
            self._array_view.array = self.measurement.array

            self._array_view.extent = extent[self._array_view.display_axes[0]]

            self._array_view.x_label = (f'{names[self._array_view.display_axes[0]]}'
                                        + f' [{units[self._array_view.display_axes[0]]}]')
            # self._array_view.y_label = (f'{names[self._array_view.display_axes[1]]}'
            #                            + f' [{units[self._array_view.display_axes[1]]}]')


class MeasurementView2d(HasTraits):
    measurement = Any()
    power_scale = Float()

    def __init__(self, measurement=None, navigation_axes=None, **kwargs):

        if measurement is None:
            array = np.zeros((0, 0))
        else:
            array = measurement.array

        array_view = ArrayView2d(array=array, navigation_axes=navigation_axes)

        super().__init__(array_view=array_view, measurement=measurement, **kwargs)

    @observe('measurement')
    def _observe_measurement(self, change):
        extent = self.measurement.calibration_limits
        units = self.measurement.calibration_units
        names = self.measurement.calibration_names

        with self._array_view.hold_trait_notifications():
            self._array_view.array = self.measurement.array
            self._array_view.extent = [extent[i] for i in self._array_view.display_axes]

            self._array_view.x_label = (f'{names[self._array_view.display_axes[0]]}'
                                        + f' [{units[self._array_view.display_axes[0]]}]')
            self._array_view.y_label = (f'{names[self._array_view.display_axes[1]]}'
                                        + f' [{units[self._array_view.display_axes[1]]}]')
