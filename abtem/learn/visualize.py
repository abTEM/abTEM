from abtem.learn.calibrate import GrapheneCalibrator
from abtem.learn.dataset import Example
from bokeh import models
from bokeh import plotting
from bokeh.palettes import Category10
from bokeh.transform import linear_cmap
from bqplot import *
from psm.graph import stable_delaunay_faces
from psm.representation import faces_to_edges
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.transform import downscale_local_mean


class Playback:

    def __init__(self, num_items=0, **kwargs):
        self._previous_button = models.Button(label='Previous')
        self._next_button = models.Button(label='Next')
        self._slider = models.Slider(start=0, end=1, value=0, step=1)

        self.num_items = num_items

        def next_item(*args):
            self._slider.value = (self._slider.value + 1) % (self._slider.end + 1)

        self._next_button.on_click(next_item)

        def previous_item(*args):
            self._slider.value = (self._slider.value - 1) % (self._slider.end + 1)

        self._previous_button.on_click(previous_item)

    @property
    def num_items(self):
        return self._slider.end + 1

    @num_items.setter
    def num_items(self, value):
        if value < 2:
            value = 2
            self._slider.disabled = True
        else:
            self._slider.disabled = False

        self._slider.end = value - 1

    @property
    def value(self):
        return self._slider.value

    @value.setter
    def value(self, value):
        self._slider.value = value

    @property
    def widgets(self):
        return models.Column(self._slider, models.Row(self._previous_button, self._next_button, width=600), width=600)


class ViewerExtension:

    def __init__(self, widgets):
        self._widgets = widgets
        self._viewer = None

    def set_viewer(self, viewer):
        self._viewer = viewer

    @property
    def widgets(self):
        return self._widgets


class BinningSlider(ViewerExtension):

    def __init__(self, max=4, value=1):
        self._slider = models.Slider(start=1, end=max, value=value, step=1, title='Binning')
        super().__init__(self._slider)

    def modify_image(self, image):
        return downscale_local_mean(image, (self._slider.value,) * 2)

    def set_viewer(self, viewer):
        self._slider.on_change('value', lambda attr, old, new: viewer.set_image())


class GaussianFilterSlider(ViewerExtension):

    def __init__(self, min=0, max=10, value=0):
        self._slider = models.Slider(start=min, end=max, value=value, step=1, title='Gaussian filter')
        super().__init__(self._slider)

    def modify_image(self, image):
        return gaussian_filter(image, self._slider.value)

    def set_viewer(self, viewer):
        self._slider.on_change('value', lambda attr, old, new: viewer.update_image())


class ClipSlider(ViewerExtension):

    def __init__(self, value=None, step=0.001, title='Clip', **kwargs):
        if value is None:
            value = (0, 1)
        self._slider = models.RangeSlider(start=0, end=1, step=step, value=value, title=title)
        super().__init__(self._slider)

    def modify_image(self, image):
        ptp = image.ptp()
        clip_min = image.min() + self._slider.value[0] * ptp
        clip_max = image.min() + self._slider.value[1] * ptp
        image = np.clip(image, a_min=clip_min, a_max=clip_max)
        return image

    def set_viewer(self, viewer):
        self._slider.on_change('value', lambda attr, old, new: viewer.update_image())


class Annotator(ViewerExtension):

    def __init__(self, widgets):
        super().__init__(widgets)

    def annotate(self):
        raise NotImplementedError()

    def set_viewer(self, viewer):
        self._viewer = viewer


class ModelAnnotator(Annotator):

    def __init__(self, model, path):
        self._model = model

        self._checkbox_button_group = models.CheckboxButtonGroup(labels=['Points',
                                                                         'Graph'], active=[0, 1])

        def on_change(attr, old, new):
            self._viewer._point_glyph.visible = 0 in self._checkbox_button_group.active
            self._graph_glyph.visible = 1 in self._checkbox_button_group.active

        self._checkbox_button_group.on_change('active', on_change)

        self._path = path
        self._write_annotation_button = models.Button(label='Write annotation')

        def on_click(event):
            x = self._viewer._point_source.data['x']
            y = self._viewer._point_source.data['y']
            labels = self._viewer._point_source.data['labels']

            image_name = os.path.split(self._viewer._image_files[self._viewer._image_playback.value])[-1]
            image_name = '.'.join(image_name.split('.')[:-1])

            if len(self._viewer._images.shape) == 3:
                num_images = len(self._viewer._images)
                image_index = self._viewer._series_playback.value
                image_name = image_name + '_{}'.format(str(image_index).zfill(len(str(num_images))))

            write_path = os.path.join(self._path, image_name + '.npz')
            points = np.array([y, x]).T
            example = Example(image=self._viewer._image, points=points, labels=labels, sampling=-1)
            example.write(write_path)

        self._write_annotation_button.on_click(on_click)

        self._sampling = .05
        self._calibrator = GrapheneCalibrator(self._model, .01, .05)

        def on_click(event):
            self.calibrate()

        self._calibrate_button = models.Button(label='Calibrate')
        self._calibrate_button.on_click(on_click)

        super().__init__(
            models.Column(self._calibrate_button, self._checkbox_button_group, self._write_annotation_button,
                          width=600))

    def calibrate(self):

        self._sampling = .05 #self._calibrator(self._viewer._image)


    def update_graph(self):
        x = self._viewer._point_source.data['x']
        y = self._viewer._point_source.data['y']

        points = np.stack((x, y), axis=-1)
        faces = stable_delaunay_faces(points, 2.)
        edges = np.array(faces_to_edges(faces))

        if len(edges) > 0:
            edges = points[edges]

            xs = [[edge[0, 0], edge[1, 0]] for edge in edges]
            ys = [[edge[0, 1], edge[1, 1]] for edge in edges]
            self._graph_source.data = {'xs': xs, 'ys': ys}

    def set_viewer(self, viewer):
        self._graph_source = models.ColumnDataSource(dict(xs=[], ys=[]))
        self._graph_model = models.MultiLine(xs='xs', ys='ys', line_width=2)
        self._graph_glyph = viewer._figure.add_glyph(self._graph_source, glyph=self._graph_model)

        def on_change(attr, old, new):
            self.update_graph()

        viewer._point_source.on_change('data', on_change)
        self._viewer = viewer

    def annotate(self):
        image = self._viewer._image
        output = self._model(image, self._sampling)

        points = output['points']
        labels = output['labels']

        self._viewer._point_source.data = {'x': list(points[:, 0]), 'y': list(points[:, 1]), 'labels': list(labels)}


class LabelEditor(ViewerExtension):

    def __init__(self):
        self._slider = models.Slider(start=0, end=4, value=0, step=1)
        self._button = models.Button(label='Label selected')

        def label_selected(event):
            indices = self._viewer._point_source.selected.indices
            new_data = dict(self._viewer._point_source.data)
            for i in indices:
                new_data['labels'][i] = int(self._slider.value)
            self._viewer._point_source.data = new_data

        self._button.on_click(label_selected)

        def on_change(attr, old, new):
            self._viewer._point_draw_tool.empty_value = int(self._slider.value)

        self._slider.on_change('value', on_change)

        super().__init__(models.Row(self._slider, self._button, width=600))


class DeleteButton(ViewerExtension):

    def __init__(self):
        self._button = models.Button(label='Delete selected')

        def label_selected(event):
            indices = self._viewer._point_source.selected.indices
            new_data = dict(self._viewer._point_source.data)

            for i in sorted(indices, reverse=True):
                del new_data['x'][i]
                del new_data['y'][i]
                del new_data['labels'][i]

            self._viewer._point_source.selected.indices = []

            self._viewer._point_source.data = new_data

        self._button.on_click(label_selected)

        super().__init__(self._button)


class VisibilityCheckboxes(ViewerExtension):

    def __init__(self):
        self._checkbox_button_group = models.CheckboxButtonGroup(labels=['Image', 'Points', 'Edges'], active=[0, 1, 2])

        def on_change(attr, old, new):
            self._viewer._image_glyph.visible = 0 in self._checkbox_button_group.active
            self._viewer._circle_glyph.visible = 1 in self._checkbox_button_group.active
            self._viewer._multi_line_glyph.visible = 2 in self._checkbox_button_group.active

        self._checkbox_button_group.on_change('active', on_change)

        super().__init__(self._checkbox_button_group)


class WriteAnnotationButton(ViewerExtension):

    def __init__(self, path):
        self._path = path
        self._button = models.Button(label='Write annotation')

        def on_click(event):
            x = self._viewer._circle_source.data['x']
            y = self._viewer._circle_source.data['y']
            labels = self._viewer._circle_source.data['labels']

            image_name = os.path.split(self._viewer._image_files[self._viewer._image_playback.value])[-1]
            write_path = os.path.join(self._path, '.'.join(image_name.split('.')[:-1]) + '.npz')
            np.savez(write_path, x=x, y=y, labels=labels)

        self._button.on_click(on_click)

        super().__init__(self._button)


class Viewer:

    def __init__(self, image_files, extensions=None, annotator=None, ):
        self._image_files = image_files
        self._extensions = extensions
        self._annotator = annotator

        self._figure = plotting.Figure(plot_width=800, plot_height=800, title=None, toolbar_location='below',
                                       tools='lasso_select,box_select,pan,wheel_zoom,box_zoom,reset')

        self._image_source = models.ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self._image_model = models.Image(image='image', x='x', y='y', dw='dw', dh='dh', palette='Greys256')
        self._image_glyph = self._figure.add_glyph(self._image_source, glyph=self._image_model)

        mapper = linear_cmap(field_name='labels', palette=Category10[9], low=0, high=8)

        self._point_source = models.ColumnDataSource(data=dict(x=[], y=[], labels=[]))
        self._point_model = models.Square(x='x', y='y', fill_color=mapper, size=10)

        self._image_playback = Playback(len(image_files))
        self._image_playback._slider.on_change('value_throttled', lambda attr, old, new: self.import_image())
        self._image_playback._next_button.on_click(lambda event: self.import_image())
        self._image_playback._previous_button.on_click(lambda event: self.import_image())

        self._series_playback = Playback()
        self._series_playback._slider.on_change('value_throttled', lambda attr, old, new: self.update())
        self._series_playback._next_button.on_click(lambda event: self.update())
        self._series_playback._previous_button.on_click(lambda event: self.update())

        for extension in self._extensions:
            extension.set_viewer(self)

        if annotator is not None:
            annotator.set_viewer(self)

        self._point_glyph = self._figure.add_glyph(self._point_source, glyph=self._point_model)
        self._point_draw_tool = models.PointDrawTool(renderers=[self._point_glyph], empty_value=0)
        self._figure.add_tools(self._point_draw_tool)

        self.import_image()

    def update_image(self):

        if len(self._images.shape) == 3:
            self._image = self._images[self._series_playback.value]
        elif len(self._images.shape) == 2:
            self._image = self._images
        else:
            raise RuntimeError('')

        #self._image = downscale_local_mean(self._image, (2, 2))

        image = self._image.copy()

        for extension in self._extensions:
            if hasattr(extension, 'modify_image'):
                image = extension.modify_image(image)

        self._image_source.data = {'image': [image],
                                   'x': [0], 'y': [0], 'dw': [image.shape[1]],
                                   'dh': [image.shape[0]]}

    def import_image(self):
        image_file = self._image_files[self._image_playback.value]

        filename, file_extension = os.path.splitext(image_file)

        if (file_extension == '.tif') or (file_extension == '.tiff'):
            self._images = imread(image_file)

        elif file_extension in '.npy':
            self._images = np.load(image_file)

        else:
            raise RuntimeError()

        self._series_playback.num_items = len(self._images)
        self._series_playback.value = 0

        self.update_image()

        if self._annotator is not None:
            # self._annotator.calibrate()
            self._annotator.annotate()

    def update(self):

        self.update_image()

        if self._annotator is not None:
            self._annotator.annotate()

    @property
    def widgets(self):
        extension_widgets = [module.widgets for module in self._extensions]
        column = models.Column(self._image_playback.widgets, self._series_playback.widgets, *extension_widgets,
                               self._annotator.widgets, width=600)
        return models.Row(self._figure, column)
