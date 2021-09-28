from abc import abstractmethod, ABC

import numpy as np
from bqplot.interacts import BrushSelector, PanZoom
from bqplot_image_gl.interacts import MouseInteraction
from traitlets import HasTraits, Int, default, Bool, List, Float, Instance, observe
from traittypes import Array

from abtem.visualize.interactive.artists import ScatterArtist, CircleArtist, LinesArtist
from abtem.visualize.interactive.utils import link


class Tool(ABC):

    @abstractmethod
    def activate(self, canvas):
        pass


class BoxZoomTool(Tool):

    def activate(self, canvas):
        brush_selector = BrushSelector()
        brush_selector.x_scale = canvas.x_scale
        brush_selector.y_scale = canvas.y_scale

        def box_zoom(change):
            selected_x = brush_selector.selected_x
            selected_y = brush_selector.selected_y

            if (selected_x is None) or (selected_y is None):
                return

            canvas.x_scale.min = min(selected_x[0], selected_x[1])
            canvas.x_scale.max = max(selected_x[0], selected_x[1])
            canvas.y_scale.min = min(selected_y[0], selected_y[1])
            canvas.y_scale.max = max(selected_y[0], selected_y[1])

            canvas._enforce_scale_lock()

            brush_selector.selected_x = None
            brush_selector.selected_y = None
            canvas.figure.interaction = None
            canvas.figure.interaction = brush_selector

        brush_selector.observe(box_zoom, 'brushing')
        canvas.figure.interaction = brush_selector

    def deactivate(self, canvas):
        canvas.figure.interaction = None


class PanZoomTool(Tool):

    def activate(self, canvas):
        panzoom = PanZoom()
        panzoom.scales = {'x': [canvas.x_scale], 'y': [canvas.y_scale]}
        canvas.figure.interaction = panzoom

    def deactivate(self, canvas):
        canvas.figure.interaction = None


class SelectPixelTool(HasTraits):
    __metaclass__ = Tool

    indices = List()

    # index_y = Int()

    def __init__(self, image_artist, **kwargs):
        self._image_artist = image_artist
        self._point_artist = ScatterArtist()

        super().__init__(**kwargs)

    @default('indices')
    def _default_indices(self):
        return [0, 0]

    def activate(self, canvas):
        self._point_artist._add_to_canvas(canvas)

        interaction = MouseInteraction(x_scale=canvas.figure.axes[0].scale,
                                       y_scale=canvas.figure.axes[1].scale)

        def on_mouse_msg(_, change, __):
            if change['event'] in ('dragmove', 'click'):
                position = np.array([change['domain']['x'], change['domain']['y']])
                indices = self._image_artist.position_to_index(position)
                indices[0] = max(0, min(indices[0], self._image_artist.image.shape[0] - 1))
                indices[1] = max(0, min(indices[1], self._image_artist.image.shape[1] - 1))

                self.indices = indices
                # self.index_y = indices[1]

                rounded_position = self._image_artist.indices_to_position((indices[0], indices[1]))
                self._point_artist.x = np.array([rounded_position[0]])
                self._point_artist.y = np.array([rounded_position[1]])

        canvas.figure.interaction = interaction
        interaction.events = ['click', 'dragmove']
        canvas.figure.interaction.on_msg(on_mouse_msg)

    def deactivate(self, canvas):
        artists = canvas.artists
        for name, artist in canvas.artists.items():
            if artist == self._point_artist:
                artists.pop(name)

        canvas.artists = artists
        canvas._observe_artists(None)
        canvas.figure.interaction = None


class BoxSelectPixelTool(HasTraits):
    __metaclass__ = Tool

    selected_x = List()
    selected_y = List()

    def __init__(self, image_artist, continuous_update=True, **kwargs):
        self._image_artist = image_artist
        self.continuous_update = continuous_update
        super().__init__(**kwargs)

    def activate(self, canvas):
        brush_selector = BrushSelector(color='red')
        brush_selector.x_scale = canvas.x_scale
        brush_selector.y_scale = canvas.y_scale

        def box_zoom(change):

            if not self.continuous_update:
                if (not change['name'] == 'brushing') and (change['new']):
                    return

            selected_x = brush_selector.selected_x
            selected_y = brush_selector.selected_y

            if (selected_x is None) or (selected_y is None):
                return

            corner1 = self._image_artist.position_to_index([selected_x[0], selected_y[0]])
            corner2 = self._image_artist.position_to_index([selected_x[1], selected_y[1]])

            for corner in (corner1, corner2):
                corner[0] = max(0, min(corner[0], self._image_artist.image.shape[0] - 1))
                corner[1] = max(0, min(corner[1], self._image_artist.image.shape[1] - 1))

            with self.hold_trait_notifications():
                self.selected_x = [min(corner1[0], corner2[0]), max(corner1[0], corner2[0])]
                self.selected_y = [min(corner1[1], corner2[1]), max(corner1[1], corner2[1])]

        brush_selector.observe(box_zoom)
        canvas.figure.interaction = brush_selector

    def deactivate(self, canvas):
        canvas.figure.interaction = None


class SelectPositionTool1d(HasTraits):
    position = Array()
    marker = Bool()

    def __init__(self, allow_drag=True, direction='x', **kwargs):
        self._point_artist = LinesArtist()
        self._allow_drag = allow_drag
        self._interaction = MouseInteraction()
        self._direction = direction
        super().__init__(**kwargs)

    @default('position')
    def _default_position(self):
        return 0.

    @observe('position')
    def _update_line(self, *args):
        if self.marker:
            if self._direction == 'y':
                self._point_artist.y = [self.position, self.position]
                self._point_artist.x = [-100, 100]
            else:
                self._point_artist.x = [self.position, self.position]
                self._point_artist.y = [-100, 100]

    def activate(self, canvas):
        if self.marker:
            canvas._tool_artists = [self._point_artist]

        self._interaction.x_scale = canvas.figure.axes[0].scale
        self._interaction.y_scale = canvas.figure.axes[1].scale

        def on_mouse_msg(_, change, __):
            if change['event'] in ('dragmove', 'click'):
                self.position = change['domain'][self._direction]

        canvas.figure.interaction = self._interaction

        events = ['click']
        if self._allow_drag:
            events += ['dragmove']

        self._interaction.events = events
        self._interaction.on_msg(on_mouse_msg)

    def deactivate(self, canvas):
        canvas._tool_artists = []
        canvas.figure.interaction = None


class DragPointTool(HasTraits):
    artist = Instance(ScatterArtist)

    def activate(self, canvas):
        self.artist._mark.enable_move = True

    def deactivate(self, canvas):
        self.artist._mark.enable_move = False


class SelectRadiusTool(HasTraits):
    radius = Float(1.)
    always_visibile = Bool()

    def __init__(self, **kwargs):
        self._circle_artist = CircleArtist()
        self._select_position_tool = SelectPositionTool(marker=False)

        super().__init__(**kwargs)
        link((self, 'radius'), (self._circle_artist, 'radius'))

        def callback(*args):
            radius = min(np.linalg.norm(self._circle_artist.center - self._select_position_tool.position), 1)
            self.radius = radius

        self._callback = callback

    def activate(self, canvas):
        self._select_position_tool.activate(canvas)
        self._select_position_tool.observe(self._callback)
        canvas._tool_artists = [self._circle_artist]

    def deactivate(self, canvas):
        self._select_position_tool.unobserve(self._callback)
        # canvas._tool_artists = []


class SelectAnnularRadiiTool(HasTraits):
    center = Array()
    inner_radius = Float(1.)
    outer_radius = Float(2.)
    min_width = Float(0.)
    max_radius = Float(2.1)
    lock_radii = Bool()

    def __init__(self, **kwargs):
        self._outer_circle_artist = CircleArtist()
        self._inner_circle_artist = CircleArtist()
        self._mouse_interaction = MouseInteraction()
        self._width = None
        self._adjusting = 'inner'

        self._observe_radii()

        super().__init__(**kwargs)

        link((self, 'inner_radius'), (self._inner_circle_artist, 'radius'))
        link((self, 'outer_radius'), (self._outer_circle_artist, 'radius'))
        link((self, 'center'), (self._inner_circle_artist, 'center'))
        link((self, 'center'), (self._outer_circle_artist, 'center'))

        def callback(_, change, __):
            position = np.array([change['domain']['x'], change['domain']['y']])
            distance = np.linalg.norm(self.center - position)
            if change['event'] == 'dragstart':
                inner_distance = np.abs(distance - self._inner_circle_artist.radius)
                outer_distance = np.abs(distance - self._outer_circle_artist.radius)
                if inner_distance < outer_distance:
                    self._adjusting = 'inner'
                else:
                    self._adjusting = 'outer'

            if change['event'] in ('dragstart', 'dragmove'):
                if self._adjusting == 'inner':
                    with self.hold_trait_notifications():
                        new_inner_radius = min(distance, self.outer_radius - self.min_width)
                        if self.lock_radii:
                            self.inner_radius = min(new_inner_radius, self.max_radius - self._width)
                            self.outer_radius = self.inner_radius + self._width
                        else:
                            self.inner_radius = new_inner_radius

                else:
                    with self.hold_trait_notifications():
                        new_outer_radius = max(min(distance, self.max_radius), self.inner_radius + self.min_width)
                        if self.lock_radii:
                            self.outer_radius = max(new_outer_radius, self._width)
                            self.inner_radius = self.outer_radius - self._width

                        else:
                            self.outer_radius = new_outer_radius

        self._mouse_interaction.events = ['dragstart', 'dragmove']
        self._callback = callback

    @default('center')
    def _default_center(self):
        return np.zeros((2,))

    @observe('inner_radius', 'outer_radius')
    def _observe_radii(self, *args):
        self._width = self.outer_radius - self.inner_radius

    def activate(self, canvas):
        self._mouse_interaction.x_scale = canvas.figure.axes[0].scale
        self._mouse_interaction.y_scale = canvas.figure.axes[1].scale
        self._mouse_interaction.on_msg(self._callback)
        canvas.figure.interaction = self._mouse_interaction
        canvas._tool_artists = [self._inner_circle_artist, self._outer_circle_artist]

    def deactivate(self, canvas):
        self._mouse_interaction.unobserve(self._callback)
        canvas._tool_artists = []
