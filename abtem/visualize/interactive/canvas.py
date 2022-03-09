import ipywidgets as widgets
import numpy as np
from bqplot import Figure, LinearScale, Axis
from traitlets import HasTraits, observe, Dict, Instance, link, Bool, List, default, Float
from traitlets import Unicode


class Canvas(HasTraits):
    artists = Dict()
    tools = Dict()
    tool = Unicode('None')
    figure = Instance(Figure)
    title = Unicode()
    x_label = Unicode()
    y_label = Unicode()
    x_limits = List()
    y_limits = List()
    lock_scale = Bool(True, allow_none=True)
    _tool_artists = List()

    def __init__(self,
                 x_scale=None,
                 y_scale=None,
                 x_axis=None,
                 y_axis=None,
                 height=450.,
                 width=450.,
                 fig_margin=None,
                 **kwargs):

        x_scale = x_scale or LinearScale(allow_padding=False)
        y_scale = y_scale or LinearScale(allow_padding=False)

        scales = {'x': x_scale, 'y': y_scale}

        x_axis = x_axis or Axis(scale=scales['x'])
        y_axis = y_axis or Axis(scale=scales['y'], orientation='vertical')

        fig_margin = fig_margin or {'top': 0, 'bottom': 50, 'left': 50, 'right': 0}

        min_aspect_ratio = width / height
        max_aspect_ratio = width / height

        figure = Figure(scales=scales,
                        axes=[x_axis, y_axis],
                        min_aspect_ratio=min_aspect_ratio,
                        max_aspect_ratio=max_aspect_ratio,
                        fig_margin=fig_margin)

        figure.layout.height = f'{height}px'
        figure.layout.width = f'{width}px'

        super().__init__(figure=figure, **kwargs)
        link((self, 'x_label'), (x_axis, 'label'))
        link((self, 'y_label'), (y_axis, 'label'))
        #link((self, 'title'), (figure, 'title'))

    @property
    def widget(self):
        whitespace = widgets.HBox([])
        whitespace.layout.width = f'{self.figure.fig_margin["left"]}px'

        title = widgets.HTML(value=f"<p style='font-size:16px;text-align:center'> {self.title} </p>")
        title.layout.width = f'{float(self.figure.layout.width[:-2]) - self.figure.fig_margin["left"]}px'
        return widgets.VBox([widgets.HBox([whitespace, title]), self.figure])

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

    @observe('artists', '_tool_artists')
    def _observe_artists(self, change):
        self._update_marks()

    def _update_marks(self):
        self.figure.marks = []

        for key, artist in self.artists.items():
            artist._add_to_canvas(self)

        for artist in self._tool_artists:
            artist._add_to_canvas(self)

        # adjust_x = False
        # adjust_y = False
        # if (self.x_scale.min is not None) or (self.x_scale.max is not None):
        #     adjust_x = True
        #
        # if (self.y_scale.min is not None) or (self.y_scale.max is not None):
        #     adjust_y = True

        #self.adjust_limits_to_artists(adjust_x=adjust_x, adjust_y=adjust_y)

    def _enforce_scale_lock(self, adjust_x=True, adjust_y=True):
        if not self.lock_scale:
            return

        if None in (self.x_scale.min, self.x_scale.min, self.y_scale.min, self.y_scale.max):
            return

        extent = max(self.x_scale.max - self.x_scale.min, self.y_scale.max - self.y_scale.min) / 2
        x_center = (self.x_scale.min + self.x_scale.max) / 2
        y_center = (self.y_scale.min + self.y_scale.max) / 2

        with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
            if adjust_x:
                self.x_scale.min = x_center - extent
                self.x_scale.max = x_center + extent

            if adjust_y:
                self.y_scale.min = y_center - extent
                self.y_scale.max = y_center + extent

    @observe('tool')
    def _observe_tool(self, change):
        if change['old_tests'] != 'None':
            self.tools[change['old_tests']].deactivate(self)

        if change['new'] == 'None':
            self.figure.interaction = None
        else:
            self.tools[change['new']].activate(self)

    @property
    def toolbar(self):
        tool_names = ['None'] + list(self.tools.keys())

        tool_selector = widgets.ToggleButtons(options=tool_names, value=self.tool)
        tool_selector.style.button_width = '60px'

        link((tool_selector, 'value'), (self, 'tool'))

        whitespace = widgets.HBox([])
        whitespace.layout.width = f'{self.figure.fig_margin["left"]}px'

        reset_button = widgets.Button(description='Reset', layout=widgets.Layout(width='80px'))
        reset_button.on_click(lambda _: self.adjust_limits_to_artists())

        return widgets.HBox([whitespace, widgets.HBox([tool_selector]), reset_button])

    @observe('x_limits')
    def _observe_x_limits(self, change):
        if change['new'] is None:
            self.x_scale.min = None
            self.x_scale.max = None
        else:
            self.x_scale.min = change['new'][0]
            self.x_scale.max = change['new'][1]

    @observe('y_limits')
    def _observe_y_limits(self, change):
        if change['new'] is None:
            self.y_scale.min = None
            self.y_scale.max = None
        else:
            self.y_scale.min = change['new'][0]
            self.y_scale.max = change['new'][1]

    @property
    def visibility_checkboxes(self):
        checkboxes = []
        for key, artist in self.artists.items():
            checkbox = widgets.Checkbox(value=True, description=key, indent=False, layout=widgets.Layout(width='90%'))
            link((checkbox, 'value'), (artist, 'visible'))
            checkboxes.append(checkbox)
        return widgets.VBox(checkboxes)

    def adjust_labels_to_artists(self):
        x_labels = [artist.x_label for artist in self.artists.values() if artist.x_label is not None]
        if len(x_labels) > 0:
            self.x_label = x_labels[0]

        y_labels = [artist.y_label for artist in self.artists.values() if artist.y_label is not None]
        if len(y_labels) > 0:
            self.y_label = y_labels[0]

    def adjust_limits_to_artists(self, adjust_x=True, adjust_y=True, *args):
        xmin = np.min([artist.limits[0][0] for artist in self.artists.values()])
        xmax = np.max([artist.limits[0][1] for artist in self.artists.values()])
        ymin = np.min([artist.limits[1][0] for artist in self.artists.values()])
        ymax = np.max([artist.limits[1][1] for artist in self.artists.values()])

        with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
            if adjust_x:
                self.x_scale.min = float(xmin)
                self.x_scale.max = float(xmax)

            if adjust_y:
                self.y_scale.min = float(ymin)
                self.y_scale.max = float(ymax)

            self._enforce_scale_lock()
