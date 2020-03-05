import io
import os

import PIL.Image
import ipywidgets as widgets
import numpy as np
from bqplot import Image
from ipyfilechooser import FileChooser
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from bqplot import Image, Scatter
from bqplot.interacts import LassoSelector


class Playback:

    def __init__(self, num_frames=1, **kwargs):
        self._frame_slider = widgets.IntSlider(min=0, max=num_frames - 1, **kwargs)

        def next_frame(*args):
            self._frame_slider.value = (self._frame_slider.value + 1) % (self._frame_slider.max + 1)

        self._next_button = widgets.Button(description='Next')
        self._next_button.on_click(next_frame)

        def previous_frame(*args):
            self._frame_slider.value = (self._frame_slider.value - 1) % (self._frame_slider.max + 1)

        self._previous_button = widgets.Button(description='Previous')
        self._previous_button.on_click(previous_frame)

        super().__init__(**kwargs)

    @property
    def num_frames(self):
        return self._frame_slider.max + 1

    @num_frames.setter
    def num_frames(self, value):
        self._frame_slider.max = value - 1

    @property
    def current_frame_index(self):
        return self._frame_slider.value

    @property
    def on_frame_change(self):
        return self._on_frame_change

    @on_frame_change.setter
    def on_frame_change(self, func):
        self._on_frame_change = func
        self._frame_slider.observe(self._on_frame_change, names='value')

    def widgets(self):
        return widgets.VBox([self._frame_slider, widgets.HBox([self._next_button, self._previous_button])])


class ImageLoader:

    def __init__(self, path=os.getcwd(), filename='', **kwargs):
        self._file_chooser = FileChooser(path, filename)
        self._load_file_button = widgets.Button(description='Read image(s)')
        self._load_file_button.on_click(self.read_images)
        self._file_chooser._apply_selection()
        self._info = widgets.HTML(value='')
        self._images = None
        self._on_read_images = []

        super().__init__(**kwargs)

    @property
    def images(self):
        return self._images

    @property
    def on_read_images(self):
        return self._on_read_images

    @on_read_images.setter
    def on_read_images(self, value):
        self._on_read_images.append(value)

    def read_images(self, *args):
        self._images = imread(self._file_chooser.selected)
        if len(self._images.shape) == 2:
            self._info.value = '&ensp; Read a {}x{} image'.format(*self._images.shape)
            self._images = self._images[0]
        elif len(self._images.shape) == 3:
            self._info.value = '&ensp; Read {} {}x{} images'.format(*self._images.shape)

        for callback in self._on_read_images:
            callback(self.images)

    def widgets(self):
        return widgets.VBox([self._file_chooser, widgets.HBox([self._load_file_button, self._info])])


class ImageBrowser(ImageLoader, Playback):

    def __init__(self, path=os.getcwd(), filename='', num_frames=1):
        super().__init__(path=path, filename=filename, num_frames=num_frames)

        def num_frames_callback(images):
            self.num_frames = len(images)

        self._on_read_images.append(num_frames_callback)

    @property
    def current_frame(self):
        return self.images[self.current_frame_index]

    @property
    def widgets(self):
        return widgets.VBox([ImageLoader.widgets(self), Playback.widgets(self)])

        # return widgets.VBox([widgets.VBox([self._file_chooser, widgets.HBox([self._load_file_button, self._info])]),
        #                     self._frame_slider])


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


class ArrayImage:

    def __init__(self, array=None, modifiers=None, scales=None):
        if modifiers is None:
            modifiers = []

        if modifiers:
            for modifier in modifiers:
                for widget in modifier.observing_widgets():
                    widget.observe(self.update_image)

        self._modifiers = modifiers

        self._image = None
        self._scales = scales

        if array:
            self.set_array(array)

    @property
    def image(self):
        return self._image

    def apply_modifiers(self, array):
        for modifier in self._modifiers:
            array = modifier.apply(array)
        return array

    def update_image(self, *args):
        array = np.flipud(self.apply_modifiers(self._array).T)
        self._image.image = widgets.Image(value=array_as_image(array))

    def set_array(self, array):
        self._array = array
        array = np.flipud(self.apply_modifiers(self._array).T)
        if self._image is None:
            self._image = Image(image=widgets.Image(value=array_as_image(array)), scales=self._scales)
            with self.image.hold_sync():
                self._image.x = [0, array.shape[0]]
                self._image.y = [0, array.shape[1]]
        else:
            with self.image.hold_sync():
                self._image.image = widgets.Image(value=array_as_image(array))
                self._image.x = [0, array.shape[0]]
                self._image.y = [0, array.shape[1]]


class ArrayImageModifier:

    def apply(self, image):
        raise NotImplementedError()

    def observing_widgets(self):
        raise NotImplementedError()

    @property
    def widgets(self):
        raise NotImplementedError()


class GaussianFilterImageModifier:

    def __init__(self, min=0, max=10):
        self._sigma_slider = widgets.IntSlider(description='Gaussian filter:', value=0, min=min, max=max,
                                               continuous_update=True)

    def apply(self, image):
        print(self._sigma_slider.value)
        return gaussian_filter(image, self._sigma_slider.value)

    def observing_widgets(self):
        return [self._sigma_slider]

    @property
    def widgets(self):
        return self._sigma_slider


class ClipImageModifier:

    def __init__(self):
        self._limits_slider = widgets.FloatRangeSlider(value=[0, 1], min=0, max=1.0, step=0.01, description='Clip:',
                                                       continuous_update=True)

    def apply(self, image):
        image = (image - image.min()) / (image.max() - image.min())
        image = np.clip(image, a_min=self._limits_slider.value[0], a_max=self._limits_slider.value[1])
        return image

    def observing_widgets(self):
        return [self._limits_slider]

    @property
    def widgets(self):
        return self._limits_slider


class LabelledScatter(Scatter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._lasso_selector = LassoSelector(marks=[self])

    def set_points(self, points, labels=None):
        if labels is None:
            labels = np.zeros(len(points), dtype=np.int)

        if len(points) != len(labels):
            raise RuntimeError()

        with self.hold_sync():
            self.x = points[:, 0]
            self.y = points[:, 1]
            self.color = labels

    def delete_points(self, indices):
        with self.hold_sync():
            self.x = np.delete(self.x, indices)
            self.y = np.delete(self.y, indices)
            self.color = np.delete(self.color, indices)

    def delete_selected(self):
        if (self.selected is None) or (len(self.selected) == 0):
            return
        self.delete_points(self.selected)