from abc import abstractmethod, ABC

import numpy as np
from bqplot.interacts import BrushSelector, PanZoom
from bqplot_image_gl.interacts import MouseInteraction
from traitlets import HasTraits, Int, default, Bool
from traittypes import Array

from abtem.visualize.interactive.artists import ScatterArtist


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

    index_x = Int()
    index_y = Int()

    def __init__(self, image_artist, **kwargs):
        self._image_artist = image_artist
        self._point_artist = ScatterArtist()

        super().__init__(**kwargs)

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
                rounded_position = self._image_artist.indices_to_position((indices[0], indices[1]))
                self.index_x = indices[0]
                self.index_y = indices[1]
                self._point_artist.x = np.array([rounded_position[0]])
                self._point_artist.y = np.array([rounded_position[1]])

        canvas.figure.interaction = interaction
        interaction.events = ['click']
        canvas.figure.interaction.on_msg(on_mouse_msg)

    def deactivate(self, canvas):
        artists = canvas.artists
        for name, artist in canvas.artists.items():
            if artist == self._point_artist:
                artists.pop(name)

        canvas.artists = artists
        canvas._observe_artists(None)
        canvas.figure.interaction = None


class SelectPositionTool(HasTraits):
    position = Array()
    marker = Bool()

    def __init__(self, **kwargs):
        self._point_artist = ScatterArtist()

        super().__init__(**kwargs)

    @default('position')
    def _default_position(self):
        return np.array((0., 0.))

    def activate(self, canvas):
        if self.marker:
            self._point_artist._add_to_canvas(canvas)

        interaction = MouseInteraction(x_scale=canvas.figure.axes[0].scale,
                                       y_scale=canvas.figure.axes[1].scale)

        def on_mouse_msg(_, change, __):
            if change['event'] in ('dragmove', 'click'):
                position = np.array([change['domain']['x'], change['domain']['y']])

                self.position = position

                self._point_artist.x = np.array([position[0]])
                self._point_artist.y = np.array([position[1]])

        canvas.figure.interaction = interaction
        interaction.events = ['click']
        canvas.figure.interaction.on_msg(on_mouse_msg)

    def deactivate(self, canvas):
        artists = canvas.artists
        for name, artist in canvas.artists.items():
            if artist == self._point_artist:
                artists.pop(name)

        canvas.artists = artists
        canvas._observe_artists(None)
        canvas.figure.interaction = None
