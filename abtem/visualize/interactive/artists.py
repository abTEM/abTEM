import ipywidgets as widgets
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from bqplot import LinearScale, ColorScale, Lines, Scatter, Figure, Axis, ScatterGL
from bqplot_image_gl import ImageGL
from traitlets import HasTraits, observe, default, List, link, Float, Unicode, Instance, Bool, Int, Any

from abtem.measure import Measurement
from abtem.visualize.interactive.utils import Array


class ColorBar(widgets.HBox):
    min = Float(0.)
    max = Float(1.)
    label = Unicode()

    def __init__(self, color_scale, label='', margin=(50, 0), direction='horizontal', width=80, **kwargs):
        self._x_scale = LinearScale(allow_padding=False)
        self._y_scale = LinearScale(allow_padding=False, orientation='vertical')
        scales = {'x': self._x_scale,
                  'y': self._y_scale,
                  'image': color_scale}

        self._mark = ImageGL(image=np.zeros((1,)), scales=scales)

        self._direction = direction

        if direction == 'horizontal':
            # self._x_axis.num_ticks = 5
            axis = Axis(scale=scales['x'], label=label)

            fig_margin = {'top': 0, 'bottom': 50, 'left': margin[0], 'right': margin[1]}
            layout = widgets.Layout(height=f'{width}px', width='auto')
        else:
            axis = Axis(scale=scales['y'], label=label, orientation='vertical', side='right')

            fig_margin = {'top': margin[1], 'bottom': margin[0], 'left': 0, 'right': 50}
            layout = widgets.Layout(width=f'{width}px', height='auto')

        self._figure = Figure(scales=scales,
                              layout=layout,
                              axes=[axis],
                              fig_margin=fig_margin)

        self._figure.marks = [self._mark]

        super().__init__(children=[self._figure])

    def _get_array(self):
        array = np.linspace(self.min, self.max, 255)
        if self._direction == 'horizontal':
            array = array[None]
        else:
            array = array[:, None]
        return array

    @property
    def layout(self):
        return self._figure.layout

    @observe('label')
    def _observe_label(self, change):
        self._x_axis.label = change['new']

    @observe('max')
    def _observe_min(self, change):
        self._mark.image = self._get_array()

        if self._direction == 'horizontal':
            self._x_scale.min = self.min
            self._x_scale.max = self.max
            self._mark.x = [self.min, self.max]
        else:
            self._y_scale.min = self.min
            self._y_scale.max = self.max
            self._mark.y = [self.min, self.max]


class Artist(HasTraits):
    x_label = Unicode(None, allow_none=True)
    y_label = Unicode(None, allow_none=True)

    visible = Bool(True)

    def _add_to_canvas(self, canvas):
        raise NotImplementedError()

    @property
    def limits(self):
        raise NotImplementedError()


class ImageArtist(Artist):
    image = Array(check_equal=False)
    extent = List(allow_none=True)
    power = Float(1.)
    color_scheme = Unicode('Greys')
    autoadjust_colorscale = Bool(True)
    center_pixels = Bool(True)

    def __init__(self, rgb=False, **kwargs):
        self._color_scale = ColorScale(colors=['black', 'white'], min=0, max=1)

        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'),
                  'image': self._color_scale}

        if rgb:
            self._mark = ImageGL(image=np.zeros((1, 1, 3)), scales=scales)
        else:
            self._mark = ImageGL(image=np.zeros((1, 1)), scales=scales)

        self._rgb = rgb

        link((self._mark, 'visible'), (self, 'visible'))
        super().__init__(**kwargs)

    def get_color_bar(self, label='', margin=(50, 0), width=80, height=None, direction='horizontal'):

        # if direction == 'horizontal':

        color_bar = ColorBar(color_scale=self._color_scale, margin=margin,
                             label=label, width=width, height=height, direction=direction)
        link((self._color_scale, 'min'), (color_bar, 'min'))
        link((self._color_scale, 'max'), (color_bar, 'max'))
        return color_bar

    @property
    def color_scale(self):
        return self._color_scale

    @property
    def power_scale_slider(self):
        slider = widgets.FloatSlider(description='Power scale', min=1e-3, max=1, value=1, step=1e-3, )
        slider.observe(self._observe_image, 'value')
        link((slider, 'value'), (self, 'power'))
        return slider

    @property
    def color_scheme_picker(self):
        schemes = ['Greys', 'viridis', 'inferno', 'plasma', 'magma', 'Spectral', 'RdBu']
        dropdown = widgets.Dropdown(description='Scheme', options=schemes)
        link((self, 'color_scheme'), (dropdown, 'value'))
        return dropdown

    @observe('color_scheme')
    def _observe_color_scheme(self, change):
        if change['new'] == 'Greys':
            self._color_scale.colors = ['Black', 'White']
        else:
            self._color_scale.colors = []
            self._color_scale.scheme = change['new']

    @default('extent')
    def _default_extent(self):
        return None

    @default('image')
    def _default_image(self):
        return np.zeros((0, 0, 3))

    def _add_to_canvas(self, canvas):
        scales = {'x': canvas.figure.axes[0].scale,
                  'y': canvas.figure.axes[1].scale,
                  'image': self._color_scale}

        self._mark.scales = scales
        canvas.figure.marks = [self._mark] + canvas.figure.marks

    def update_image(self, *args):
        image = self.image
        if self.power != 1:
            image = image ** self.power

        image = np.swapaxes(image, 0, 1)

        if self.extent is None:
            with self._mark.hold_sync():
                self._mark.x = [-.5, image.shape[0] - .5]
                self._mark.y = [-.5, image.shape[1] - .5]
                self._mark.image = image
        else:
            self._mark.image = image

        if self._rgb:
            return

        self._set_extent()

        if (not self.image.size == 0) & self.autoadjust_colorscale:
            with self._mark.hold_sync():

                col_min = float(image.min())
                col_max = float(image.max())

                if np.isclose(col_min, col_max):
                    col_max = col_min + 1e-3


                self._mark.scales['image'].min = col_min
                self._mark.scales['image'].max = col_max

            # with self._color_bar._mark.hold_sync():
            #    self._color_bar.min = float(image.min())
            #    self._color_bar.max = float(image.max())

    def _set_extent(self):
        if self.extent is None:
            return

        sampling = ((self.extent[0][1] - self.extent[0][0]) / self.image.shape[0],
                    (self.extent[1][1] - self.extent[1][0]) / self.image.shape[1])

        if self.center_pixels:
            self._mark.x = [value - .5 * sampling[0] for value in self.extent[0]]
            self._mark.y = [value - .5 * sampling[1] for value in self.extent[1]]
        else:
            self._mark.x = [value for value in self.extent[0]]
            self._mark.y = [value for value in self.extent[1]]

    @observe('image')
    def _observe_image(self, *args):
        self.update_image(*args)

    @observe('extent')
    def _observe_extent(self, change):
        if self.image.size == 0:
            return

        self._set_extent()

    @property
    def display_sampling(self):
        if self.image.size == 0:
            return

        extent_x = self._mark.x[1] - self._mark.x[0]
        extent_y = self._mark.y[1] - self._mark.y[0]
        pixel_extent_x = self.image.shape[0]
        pixel_extent_y = self.image.shape[1]
        return (extent_x / pixel_extent_x, extent_y / pixel_extent_y)

    def position_to_index(self, position):
        sampling = self.display_sampling
        px = int(np.round((position[0] - self._mark.x[0]) / sampling[0] - .5))
        py = int(np.round((position[1] - self._mark.y[0]) / sampling[1] - .5))
        return [px, py]

    def indices_to_position(self, indices):
        sampling = self.display_sampling
        px = (indices[0] + .5) * sampling[0] + self._mark.x[0]
        py = (indices[1] + .5) * sampling[1] + self._mark.y[0]
        return [px, py]

    @property
    def limits(self):
        if (self.extent is None) or (self.display_sampling is None):
            return [(-.5, self.image.shape[0] - .5), (-.5, self.image.shape[1] - .5)]

        if self.center_pixels:
            return [tuple([l - .5 * s for l in L]) for L, s in zip(self.extent, self.display_sampling)]
        else:
            return [tuple([l for l in L]) for L, s in zip(self.extent, self.display_sampling)]


class ItemSelector(HasTraits):
    sequence = Any()
    current_index = Int(0)
    current_item = Any()

    def __init__(self, **kwargs):
        self._slider = widgets.IntSlider(min=0, max=0, step=1)
        link((self._slider, 'value'), (self, 'current_index'))
        super().__init__(**kwargs)

    @observe('current_index')
    def _observe_current_index(self, change):
        self.current_item = self.sequence[self.current_index]

    @observe('sequence')
    def _observe_sequence(self, change):
        self._slider.max = len(self.sequence) - 1

        clipped_current_index = min(self.current_index, self._slider.max)
        force_trigger = clipped_current_index == self.current_index

        self.current_index = clipped_current_index

        if force_trigger:
            self._observe_current_index(None)


class ArrayViewArtist(Artist):
    array = Array()
    index = Int()

    def __init__(self, **kwargs):
        self._image_artist = ImageArtist()
        super().__init__(**kwargs)

    @property
    def image_artist(self):
        return self._image_artist

    @observe('array', 'index')
    def _observe_array(self, change):
        self._image_artist.image = self.array[self.index]

    def _add_to_canvas(self, canvas):
        self._image_artist._add_to_canvas(canvas)

    @property
    def navigation_sliders(self):
        slider = widgets.IntSlider(min=0, max=len(self.array) - 1, step=1)
        link((slider, 'value'), (self, 'index'))
        return slider

    @property
    def limits(self):
        return self._image_artist.limits


class PointSeriesArtist(Artist):
    points = List()
    index = Int()

    def __init__(self, **kwargs):
        self._scatter_artist = ScatterArtist()
        super().__init__(**kwargs)

    @observe('points', 'index')
    def _observe_points(self, change):
        self._scatter_artist.x = self.points[self.index][:, 0]
        self._scatter_artist.y = self.points[self.index][:, 1]

    def _add_to_canvas(self, canvas):
        self._scatter_artist._add_to_canvas(canvas)

    @property
    def limits(self):
        return self._scatter_artist.limits


class MeasurementArtist2d(Artist):
    measurement = Instance(Measurement, allow_none=True)

    def __init__(self, measurement=None):
        self._image_artist = ImageArtist()
        super().__init__(measurement=measurement)

    @property
    def image_artist(self):
        return self._image_artist

    @observe('measurement')
    def _observe_measurement(self, change):
        if change['new'] is None:
            return

        extent = self.measurement.calibration_limits

        with self._image_artist.hold_trait_notifications():
            self._image_artist.image = change['new'].array  # + np.random.rand()*1e-3
            self._image_artist.extent = extent

        units = self.measurement.calibration_units
        names = self.measurement.calibration_names

        self.x_label = f'{names[0]} [{units[0]}]'
        self.y_label = f'{names[1]} [{units[1]}]'

    def _add_to_canvas(self, canvas):
        self._image_artist._add_to_canvas(canvas)

    @property
    def limits(self):
        return self._image_artist.limits


class Artist1d(Artist):
    x = Array()
    y = Array()

    def __init__(self, mark, color_scale=None, **kwargs):
        self._mark = mark
        self._color_scale = color_scale
        super().__init__(**kwargs)
        link((self._mark, 'visible'), (self, 'visible'))

    @observe('x')
    def _observe_x(self, change):
        self._mark.x = self.x

    @observe('y')
    def _observe_y(self, change):
        self._mark.y = self.y

    def _add_to_canvas(self, canvas):
        scales = {'x': canvas.figure.axes[0].scale,
                  'y': canvas.figure.axes[1].scale}

        try:
            scales['color'] = self._mark.scales['color']
        except KeyError:
            pass

        try:
            scales['size'] = self._mark.scales['size']
        except KeyError:
            pass

        self._mark.scales = scales
        canvas.figure.marks = [self._mark] + canvas.figure.marks

    @property
    def limits(self):
        return [(self.x.min(), self.x.max()), (self.y.min(), self.y.max())]


class ScatterArtist(Artist1d):
    color = Any()

    def __init__(self, colors='red', gl=False, **kwargs):
        if isinstance(colors, str):
            colors = [colors]

        color_scale = ColorScale(scheme='plasma')

        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'),
                  'color': color_scale,
                  'size': LinearScale(min=0, max=1),
                  }

        if gl:
            mark = ScatterGL(x=np.zeros((1,)), y=np.zeros((1,)), scales=scales, colors=['red'])
        else:
            mark = Scatter(x=np.zeros((1,)), y=np.zeros((1,)), scales=scales, colors=['red'])

        # link((self, 'color'), (mark, 'color'))

        super().__init__(mark=mark, **kwargs)

        link((self._mark, 'visible'), (self, 'visible'))

    @observe('color')
    def _observe_color(self):
        self._mark.color = self.color

    # @default('color')
    # def _default_colors(self):
    #     return ['red']


class LinesArtist(Artist1d):

    def __init__(self, colors='red', **kwargs):
        if isinstance(colors, str):
            colors = [colors]

        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'), }

        mark = Lines(x=np.zeros((2,)), y=np.zeros((2,)), scales=scales, colors=colors)

        super().__init__(mark=mark, **kwargs)


class CircleArtist(Artist):
    center = Array()
    radius = Float(1.)

    def __init__(self, **kwargs):
        self._mark = Lines(scales={'x': LinearScale(), 'y': LinearScale()}, colors=['red'])
        super().__init__(**kwargs)
        self.update_mark()

    @default('center')
    def _default_center(self):
        return np.zeros((2,))

    def update_mark(self):
        x = self.center[0] + np.cos(np.linspace(0, 2 * np.pi, 100)) * self.radius
        y = self.center[1] + np.sin(np.linspace(0, 2 * np.pi, 100)) * self.radius
        with self._mark.hold_sync():
            self._mark.x = x
            self._mark.y = y

    @observe('center', 'radius')
    def _observe_center_and_radius(self, change):
        self.update_mark()

    def _add_to_canvas(self, canvas):
        self._mark.scales = {'x': canvas.figure.axes[0].scale, 'y': canvas.figure.axes[1].scale}
        canvas.figure.marks = [self._mark] + canvas.figure.marks

    @property
    def limits(self):
        return [(self.center[0] - self.radius, self.center[0] + self.radius),
                (self.center[1] - self.radius, self.center[1] + self.radius)]


class MeasurementArtist1d(Artist):
    measurement = Instance(Measurement, allow_none=True)

    def __init__(self, measurement=None, **kwargs):
        self._lines_artist = LinesArtist(**kwargs)
        super().__init__(measurement=measurement, **kwargs)

    @property
    def lines_artist(self):
        return self._lines_artist

    @observe('measurement')
    def _observe_measurement(self, change):
        if change['new'] is None:
            return

        with self._lines_artist.hold_trait_notifications():
            self._lines_artist._mark.y = change['new'].array
            self._lines_artist.x = change['new'].calibrations[0].coordinates(len(change['new'].array))

            units = self.measurement.calibration_units
            names = self.measurement.calibration_names

            self.x_label = f'{names[0]} [{units[0]}]'

    def _add_to_canvas(self, canvas):
        self._lines_artist._add_to_canvas(canvas)

    @property
    def limits(self):
        return self._lines_artist.limits


from abtem.visualize.mpl import _plane2axes, _cube


class AtomsArtist(Artist):
    atoms = Instance(Atoms, allow_none=True)
    visible = Bool(True)
    scale = Float(100.)
    plane = Unicode('xy')

    def __init__(self, **kwargs):
        self._scatter_artist = ScatterArtist()
        self._lines_artist = LinesArtist(colors='black')
        self._scatter_artist._mark.stroke = 'black'
        super().__init__(**kwargs)

        link((self._scatter_artist._mark, 'visible'), (self, 'visible'))
        link((self._lines_artist._mark, 'visible'), (self, 'visible'))

        self.x_label = f'{self.plane[0]} [Å]'
        self.y_label = f'{self.plane[1]} [Å]'

    def _add_to_canvas(self, canvas):
        self._scatter_artist._add_to_canvas(canvas)
        self._lines_artist._add_to_canvas(canvas)

    @property
    def limits(self):
        return self._lines_artist.limits

    @observe('atoms', 'scale', 'plane')
    def _observe_atoms(self, change):
        if self.atoms is None:
            return

        axes = _plane2axes(self.plane)

        cell_lines_x = []
        cell_lines_y = []

        for line in _cube:
            cell_line = np.array([np.dot(line[0], self.atoms.cell), np.dot(line[1], self.atoms.cell)])
            cell_lines_x.append(cell_line[:, axes[0]])
            cell_lines_y.append(cell_line[:, axes[1]])

        with self._lines_artist._mark.hold_sync(), self._scatter_artist._mark.hold_sync():
            self._lines_artist.x = cell_lines_x
            self._lines_artist.y = cell_lines_y
            self._scatter_artist.x = self.atoms.get_positions()[:, axes[0]]
            self._scatter_artist.y = self.atoms.get_positions()[:, axes[1]]

        colors = ['#%02x%02x%02x' % tuple((jmol_colors[i] * 255).astype(np.int)) for i in self.atoms.numbers]
        sizes = [covalent_radii[i] * self.scale for i in self.atoms.numbers]
        self._scatter_artist._mark.colors = colors
        self._scatter_artist._mark.size = sizes
