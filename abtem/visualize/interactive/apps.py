import ipywidgets as widgets
import numpy as np
from bqplot import Figure, LinearScale, Axis
from traitlets import HasTraits, observe, Dict, Instance, default, List, link, Float
from traitlets import validate, Any, Unicode
from traittypes import Array

from abtem.visualize.interactive.artists import ImageArtist, LinesArtist


class Canvas(HasTraits):
    artists = Dict()
    tools = Dict()
    figure = Instance(Figure)

    def __init__(self,
                 x_scale=None,
                 y_scale=None,
                 x_axis=None,
                 y_axis=None,
                 height='450px',
                 width='450px',
                 min_aspect_ratio=1,
                 max_aspect_ratio=1,
                 fig_margin=None,
                 **kwargs):

        x_scale = x_scale or LinearScale(allow_padding=False)
        y_scale = y_scale or LinearScale(allow_padding=False)

        scales = {'x': x_scale, 'y': y_scale}

        x_axis = x_axis or Axis(scale=scales['x'])
        y_axis = y_axis or Axis(scale=scales['y'], orientation='vertical')

        fig_margin = fig_margin or {'top': 0, 'bottom': 50, 'left': 50, 'right': 0}

        figure = Figure(scales=scales,
                        axes=[x_axis, y_axis],
                        min_aspect_ratio=min_aspect_ratio,
                        max_aspect_ratio=max_aspect_ratio,
                        fig_margin=fig_margin)

        figure.layout.height = height
        figure.layout.width = width
        super().__init__(figure=figure, **kwargs)

    @property
    def x_axis(self):
        return self.figure.axes[0]

    @property
    def y_axis(self):
        return self.figure.axes[1]

    @property
    def x_scale(self):
        return self.x_axis.scale

    @property
    def y_scale(self):
        return self.y_axis.scale

    @observe('artists')
    def _observe_artists(self, change):
        self._update_marks()

    def _update_marks(self):
        self.figure.marks = []
        for key, artist in self.artists.items():
            artist._add_to_canvas(self)

    def _enforce_scale_lock(self):
        if None in (self.x_scale.min, self.x_scale.min, self.y_scale.min, self.y_scale.max):
            return

        extent = max(self.x_scale.max - self.x_scale.min, self.y_scale.max - self.y_scale.min) / 2
        x_center = (self.x_scale.min + self.x_scale.max) / 2
        y_center = (self.y_scale.min + self.y_scale.max) / 2

        with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
            self.x_scale.min = x_center - extent
            self.x_scale.max = x_center + extent
            self.y_scale.min = y_center - extent
            self.y_scale.max = y_center + extent

    @property
    def toolbar(self):
        tool_names = ['None'] + list(self.tools.keys())

        tool_selector = widgets.ToggleButtons(options=tool_names)
        tool_selector.style.button_width = '80px'

        def change_tool(change):
            if change['old'] != 'None':
                self.tools[change['old']].deactivate(self)

            if change['new'] == 'None':
                self.figure.interaction = None
            else:
                self.tools[change['new']].activate(self)

        tool_selector.observe(change_tool, 'value')

        reset_button = widgets.Button(description='Reset', layout=widgets.Layout(width='80px'))
        reset_button.on_click(lambda _: self.adjust_to_artists())

        return widgets.HBox([tool_selector, reset_button])

    def adjust_to_artists(self):
        xmin = np.min([artist.limits[0][0] for artist in self.artists.values()])
        xmax = np.max([artist.limits[0][1] for artist in self.artists.values()])
        ymin = np.min([artist.limits[1][0] for artist in self.artists.values()])
        ymax = np.max([artist.limits[1][1] for artist in self.artists.values()])

        # with self.x_scale.hold_sync(), self.y_scale.hold_sync():
        with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
            self.x_scale.min = float(xmin)
            self.x_scale.max = float(xmax)
            self.y_scale.min = float(ymin)
            self.y_scale.max = float(ymax)
            self._enforce_scale_lock()


class ArrayView(HasTraits):
    array = Array()

    def __init__(self, array, artist, navigation_axes=None, **kwargs):
        self._canvas = Canvas()
        self._canvas.artists = {'array_view': artist}

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


class MeasurementView(HasTraits):

    def __init__(self, array_view, **kwargs):
        self._array_view = array_view

        super().__init__(**kwargs)

    @property
    def canvas(self):
        return self._array_view._canvas

    @property
    def figure(self):
        return self.canvas.figure

    @property
    def artist(self):
        return self.canvas.artists['array_view']


class MeasurementView1d(MeasurementView):
    measurement = Any()

    def __init__(self, measurement=None, navigation_axes=None, **kwargs):

        if measurement is None:
            array = np.zeros((0,))
        else:
            array = measurement.array

        array_view = ArrayView1d(array=array, navigation_axes=navigation_axes)

        super().__init__(array_view=array_view, measurement=measurement, **kwargs)

    @observe('measurement')
    def _observe_measurement(self, change):
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


class MeasurementView2d(MeasurementView):
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
