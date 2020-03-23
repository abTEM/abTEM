import io

import PIL.Image
from bqplot import *
from bqplot import interacts
from ipyfilechooser import FileChooser
from psm.graph import GeometricGraph
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.transform import downscale_local_mean

from abtem.learn.utils import walk_dir


class Playback(widgets.VBox):

    def __init__(self, num_frames=1, **kwargs):
        self._slider = widgets.IntSlider(min=0, max=num_frames - 1, **kwargs)

        def next_frame(*args):
            self._slider.value = (self._frame_slider.value + 1) % (self._frame_slider.max + 1)

        self._next_button = widgets.Button(description='Next Frame')
        self._next_button.on_click(next_frame)

        def previous_frame(*args):
            self._slider.value = (self._frame_slider.value - 1) % (self._frame_slider.max + 1)

        self._previous_button = widgets.Button(description='Previous Frame')
        self._previous_button.on_click(previous_frame)

        super().__init__(children=[self._slider, widgets.HBox([self._next_button, self._previous_button])])

    @property
    def num_values(self):
        return self._frame_slider.max + 1

    @num_values.setter
    def num_values(self, value):
        self._frame_slider.max = value - 1

    @property
    def value(self):
        return self._frame_slider.value

    @property
    def on_change(self):
        return self._on_value_change

    @on_change.setter
    def on_change(self, func):
        self._on_value_change = func
        self._frame_slider.observe(self._on_value_change, names='value')

    @property
    def disabled(self):
        return self._next_button.disabled

    @disabled.setter
    def disabled(self, value):
        self._next_button.disabled = value
        self._previous_button.disabled = value
        self._frame_slider.disabled = value


class ImageReader(widgets.VBox):

    def __init__(self, path=os.getcwd(), filename='', **kwargs):
        self._file_chooser = FileChooser(path, change_desc='Set read path', **kwargs)
        self._read_image_button = widgets.Button(description='Read image(s)')
        self._next_image_button = widgets.Button(description='Next image(s)')
        self._previous_image_button = widgets.Button(description='Previous image(s)')
        self._read_image_button.on_click(self.read_images)
        self._file_chooser._apply_selection()

        # self._info = widgets.HTML(value='')

        def on_next(button):

            files = sorted(walk_dir(os.path.join(*os.path.split(self._file_chooser.selected)[:-1]), '.tif'))
            if self._file_chooser.selected_filename == '':
                idx = 0
            else:
                if button is self._next_image_button:
                    idx = (files.index(self._file_chooser.selected_filename) + 1) % len(files)
                else:
                    idx = (files.index(self._file_chooser.selected_filename) - 1) % len(files)

            self._file_chooser.reset(self._file_chooser.selected_path, files[idx])
            self._file_chooser._apply_selection()

            # print(self._file_chooser.selected_filename)
            # print(files, self._file_chooser.selected_filename)

        self._next_image_button.on_click(on_next)
        self._previous_image_button.on_click(on_next)

        self._on_read_images = []

        super().__init__(children=[self._file_chooser, widgets.HBox([self._read_image_button, self._next_image_button,
                                                                     self._previous_image_button])])

    @property
    def on_read_images(self):
        return self._on_read_images

    @on_read_images.setter
    def on_read_images(self, value):
        self._on_read_images.append(value)

    def read_images(self, *args):

        selected = self._file_chooser.selected

        file_ending = selected.split('.')[-1]

        data = {}
        if file_ending == 'npz':
            npzfile = np.load(selected)
            images = npzfile['image']
            data['points'] = npzfile['points']
            data['labels'] = npzfile['labels']
        else:
            images = imread(selected)

        if len(images.shape) == 2:
            images = images[None]

        data['images'] = downscale_local_mean(images, (1, 2, 2))

        # self._info.value = '&ensp; Read a {}x{} image'.format(*images.shape)

        for callback in self.on_read_images:
            callback(data)

        return data


class DataWriter(widgets.VBox):

    def __init__(self, path=os.getcwd(), filename='', **kwargs):
        self._write_file_button.on_click(self.read_images)
        self._file_chooser._apply_selection()

        super().__init__(children=[self._file_chooser, self._load_file_button])

    def write_images(self, *args):
        images = imread(self._file_chooser.selected)

        for callback in self.on_read_images:
            callback(images)

        return images


def array_as_image(array):
    if array.dtype == np.uint8:
        return array_as_image(array.astype(np.uint8))

    else:
        if len(array.shape) != 2:
            raise RuntimeError()
        array = ((array - array.min()) / array.ptp() * 255).astype(np.uint8)

    image = PIL.Image.fromarray(array)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='png')
    return bytes_io.getvalue()


class ArrayImage(Image):

    def __init__(self, array=None, **kwargs):
        self.set_array(array)
        super().__init__(**kwargs)

    def set_array(self, array):
        self.x = [0, array.shape[0]]
        self.y = [array.shape[1], 0]
        self.image = widgets.Image(value=array_as_image(array))


class GaussianFilterSlider(widgets.IntSlider):

    def __init__(self, min=0, max=10, description='Gaussian filter', **kwargs):
        super().__init__(description=description, min=min, max=max, **kwargs)

    def apply(self, image):
        return gaussian_filter(image, self.value)


class ClipSlider(widgets.FloatRangeSlider):

    def __init__(self, value=None, min=0, max=1, step=0.01, description='Clip', **kwargs):
        if value is None:
            value = [0, 1]
        super().__init__(value=value, min=min, max=max, step=step, description=description, **kwargs)

    def apply(self, image):
        image = (image - image.min()) / (image.max() - image.min())
        image = np.clip(image, a_min=self.value[0], a_max=self.value[1])
        return image


class EditableScatter(Scatter):

    def __init__(self, points=None, labels=None, **kwargs):
        if points is None:
            points = np.zeros((0, 2), dtype=np.float)

        self._on_edit = None

        super().__init__(x=points[:, 0], y=points[:, 1], color=labels, **kwargs)

    @property
    def labels(self):
        return self.color

    @property
    def points(self):
        return np.array([self.x, self.y]).T

    def on_edit(self, callback):
        self._on_edit = callback

    def add_points(self, new_points):
        new_points = np.array(new_points)
        with self.hold_sync():
            self.x = np.append(self.x, new_points[:, 0])
            self.y = np.append(self.y, new_points[:, 1])
            self.color = np.append(self.color, [0.])

        if self._on_edit:
            self._on_edit(self)

    def set_points(self, points, labels=None):
        if labels is None:
            labels = np.zeros(len(points), dtype=np.int)

        if len(points) != len(labels):
            raise RuntimeError()

        with self.hold_sync():
            self.x = points[:, 0]
            self.y = points[:, 1]
            self.color = labels
            self.selected = None

        if self._on_edit:
            self._on_edit(self)

    def set_labels(self, labels):
        if len(labels) != len(self.x):
            raise RuntimeError()
        self.color = labels

    def delete_points(self, indices):
        with self.hold_sync():
            self.x = np.delete(self.x, indices)
            self.y = np.delete(self.y, indices)
            self.color = np.delete(self.color, indices)
            self.selected = None

        if self._on_edit:
            self._on_edit(self)

    def delete_selected(self):
        if not ((self.selected is None) or (len(self.selected) == 0)):
            self.delete_points(self.selected)
            self.selected = None

        if self._on_edit:
            self._on_edit(self)

    def set_selected_label(self, new_label):
        if not ((self.selected is None) or (len(self.selected) == 0)):
            labels = self.labels.copy()
            labels[self.selected] = new_label
            with self.hold_sync():
                self.color = labels


colors = ['DodgerBlue', 'Yellow', 'HotPink', 'SeaGreen', 'OrangeRed']


class Annotator:

    def annotate_current_image(self, viewer):
        raise NotImplementedError()


class Viewer(widgets.HBox):

    def __init__(self, image_files, annotator, image_modifiers):
        self._image_files = image_files
        self._annotator = annotator

        self._scales = {'x': LinearScale(),
                        'y': LinearScale(),
                        'color': ColorScale(colors=colors, min=0, max=4)}
        self._axes = {'x': Axis(scale=self._scales['x']),
                      'y': Axis(scale=self._scales['y'], orientation='vertical')}

        self._figure = Figure(marks=[], axes=[self.axes['x'], self.axes['y']], min_aspect_ratio=1, max_aspect_ratio=1,
                              fig_margin={'top': 10, 'bottom': 40, 'left': 40, 'right': 10}, padding_x=.05,
                              padding_y=.05)

        def apply_image_modifications(image):
            image = gaussian_filter_slider.apply(image)
            image = clip_slider.apply(image)
            return image

        def update_image(*args):
            image = apply_image_modifications(self._data['images'][playback.current_frame_index])
            self._image_mark.set_array(image)


                # if model is not None:
                #     points, labels = model(self._data['images'][change['new']])
                #     self._scatter.set_points(points, labels)
                # else:
                #     try:
                #         self._scatter.set_points(self._data['points'], self._data['labels'])
                #     except KeyError:
                #         pass
                #
                # if self._lasso_selector is not None:
                #     self._lasso_selector.reset()
                #     self._lasso_selector = interacts.LassoSelector(marks=[self._scatter])
                #
                # on_tool_change({'new': tools_toggle_buttons.value})

        self._frame_playback = Playback()
        self._file_playback = Playback()

        super().__init__(children=[self._figure, self._frame_playback, self._file_playback])

    def set_current_image(self, ):
        image_file = imread(self._image_files[self._file_playback.current_frame_index])

        # if self._image_mark is None:
        #     self._image_mark = ArrayImage(array=apply_image_modifications(data['images'][0]), scales=self.scales)
        #
        #
        # with self._image_mark.hold_sync(), self._scatter.hold_sync():
        #     update_image()

    @property
    def axes(self):
        return self._axes

    @property
    def scales(self):
        return self._scales

    @property
    def figure(self):
        return self._figure


class GUI(widgets.HBox):

    def __init__(self, path=os.getcwd(), model=None):
        self._scales = {'x': LinearScale(),
                        'y': LinearScale(),
                        'color': ColorScale(colors=colors, min=0, max=4)}
        self._axes = {'x': Axis(scale=self._scales['x']),
                      'y': Axis(scale=self._scales['y'], orientation='vertical')}

        self._figure = Figure(marks=[], axes=[self.axes['x'], self.axes['y']], min_aspect_ratio=1, max_aspect_ratio=1,
                              fig_margin={'top': 10, 'bottom': 40, 'left': 40, 'right': 10}, padding_x=.05,
                              padding_y=.05)

        self._image_mark = None
        self._data = None
        self._scatter = EditableScatter(scales=self.scales, selected_style={'stroke': 'orange'})
        self._lines = Lines(scales={'x': self._scales['x'], 'y': self._scales['y']}, colors=['red'])
        self._lasso_selector = None

        def update_edges(scatter, *args):
            graph = GeometricGraph(scatter.points)
            graph.build_stable_delaunay_graph(1.)
            edges = graph.points[graph.edges]
            with self._lines.hold_sync():
                self._lines.x = edges[:, :, 0]
                self._lines.y = edges[:, :, 1]

        self._scatter.on_edit(update_edges)
        self._scatter.on_drag_end(update_edges)

        image_loader = ImageReader(path=path)
        playback = Playback()

        gaussian_filter_slider = GaussianFilterSlider()
        clip_slider = ClipSlider()

        def apply_image_modifications(image):
            image = gaussian_filter_slider.apply(image)
            image = clip_slider.apply(image)
            return image

        def update_image(*args):
            image = apply_image_modifications(self._data['images'][playback.current_frame_index])
            self._image_mark.set_array(image)

        def on_frame_change(change):
            with self._image_mark.hold_sync(), self._scatter.hold_sync():
                update_image()

                if model is not None:
                    points, labels = model(self._data['images'][change['new']])
                    self._scatter.set_points(points, labels)
                else:
                    try:
                        self._scatter.set_points(self._data['points'], self._data['labels'])
                    except KeyError:
                        pass

                if self._lasso_selector is not None:
                    self._lasso_selector.reset()
                    self._lasso_selector = interacts.LassoSelector(marks=[self._scatter])

                on_tool_change({'new': tools_toggle_buttons.value})

        playback.on_frame_change = on_frame_change
        gaussian_filter_slider.observe(update_image)
        clip_slider.observe(update_image)

        show_image_toggle_button = widgets.ToggleButton(value=True, description='Show image')
        show_scatter_toggle_button = widgets.ToggleButton(value=True, description='Show points')
        show_lines_toggle_button = widgets.ToggleButton(value=True, description='Show edges')

        def on_show_toggle_button_change(*args):
            marks = []
            if show_image_toggle_button.value & (self._image_mark is not None):
                marks += [self._image_mark]
            if show_lines_toggle_button.value & (self._lines is not None):
                marks += [self._lines]
            if show_scatter_toggle_button.value & (self._scatter is not None):
                marks += [self._scatter]
            self.figure.marks = marks

        show_image_toggle_button.observe(on_show_toggle_button_change, names='value')
        show_scatter_toggle_button.observe(on_show_toggle_button_change, names='value')
        show_lines_toggle_button.observe(on_show_toggle_button_change, names='value')

        def on_read_images(data):
            self._data = data
            self._image_mark = ArrayImage(array=apply_image_modifications(data['images'][0]), scales=self.scales)
            on_show_toggle_button_change()
            playback.num_frames = len(data['images'])
            on_frame_change({'new': 0})

        image_loader.on_read_images = on_read_images

        def add_point(_, target):
            self.scatter.add_points([[target['data']['click_x'], target['data']['click_y']]])

        def delete_point(_, target):
            self.scatter.delete_points([target['data']['index']])

        def on_tool_change(change):
            self.figure.interaction = None
            self.scatter.interactions = {'click': None}
            self.scatter.enable_move = False
            self.image_mark.on_element_click(add_point, remove=True)
            self.scatter.on_element_click(delete_point, remove=True)
            if self._lasso_selector is not None:
                self._lasso_selector.reset()
            self._lasso_selector = None

            if change['new'] == 'PanZoom':
                self.figure.interaction = PanZoom(scales={'x': [self.scales['x']], 'y': [self.scales['y']]})
            elif change['new'] == 'Lasso':
                self._lasso_selector = interacts.LassoSelector(marks=[self._scatter])
                self.figure.interaction = self._lasso_selector
            elif change['new'] == 'Select':
                self.scatter.interactions = {'click': 'select'}
            elif change['new'] == 'Drag':
                self.scatter.enable_move = True
            elif change['new'] == 'Add':
                self.image_mark.on_element_click(add_point)
                self.image_mark.enable_move = True
            elif change['new'] == 'Delete':
                self.scatter.on_element_click(delete_point)

        tools_toggle_buttons = widgets.ToggleButtons(
            options=['None', 'PanZoom', 'Select', 'Lasso', 'Drag', 'Add', 'Delete'],
            style={'button_width': '100px'}, layout={'width': '400px'})

        tools_toggle_buttons.observe(on_tool_change, names='value')

        new_labels_int_text = widgets.IntText(value=0, description='New label:', layout=widgets.Layout(width='200px'))
        set_labels_button = widgets.Button(description='Set', disabled=False, layout=widgets.Layout(width='120px'))

        def set_selected_label(*args):
            self.scatter.set_selected_label(int(new_labels_int_text.value))

        set_labels_button.on_click(set_selected_label)

        delete_selected_button = widgets.Button(description='Delete selected', layout=widgets.Layout(width='120px'))

        def delete_selected(*args):
            self.scatter.delete_selected()
            self._lasso_selector.reset()

        delete_selected_button.on_click(delete_selected)

        deselect_button = widgets.Button(description='Deselect', layout=widgets.Layout(width='120px'))

        def deselect(*args):
            self.scatter.selected = None
            self._lasso_selector.reset()

        deselect_button.on_click(deselect)

        reset_predictions_button = widgets.Button(description='Reset predictions', layout=widgets.Layout(width='120px'))

        def reset_prediction(*args):
            on_frame_change({'new': playback.current_frame_index})

        reset_predictions_button.on_click(reset_prediction)

        write_button = widgets.Button(description='Write labelled image', layout=widgets.Layout(width='300px'))

        def write(*args):
            if self._data['images'] is None:
                return
            base_path = '.'.join(image_loader._file_chooser.selected.split('.')[:-1])
            write_path = base_path + '.npz'.format(playback.current_frame_index)

            print(write_path)

            # output_dict = {'image': self._data['images'][playback.current_frame_index], 'points': self.scatter.points,
            #               'labels': self.scatter.labels}

            output_dict = {'points': self.scatter.points, 'labels': self.scatter.labels}

            np.savez(write_path, **output_dict)

        write_button.on_click(write)

        super().__init__(children=[self.figure,
                                   widgets.VBox([image_loader,
                                                 playback,
                                                 widgets.HBox([show_image_toggle_button,
                                                               show_scatter_toggle_button,
                                                               show_lines_toggle_button]),
                                                 gaussian_filter_slider,
                                                 clip_slider,
                                                 tools_toggle_buttons,
                                                 widgets.HBox([new_labels_int_text, set_labels_button]),
                                                 widgets.HBox([deselect_button, delete_selected_button,
                                                               reset_predictions_button]),
                                                 widgets.VBox([write_button])
                                                 ])])

    @property
    def axes(self):
        return self._axes

    @property
    def scales(self):
        return self._scales

    @property
    def figure(self):
        return self._figure

    @property
    def image_mark(self):
        return self._image_mark

    @property
    def scatter(self):
        return self._scatter
