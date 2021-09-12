from pathlib import Path

import ipywidgets as widgets
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.transform import downscale_local_mean
from traitlets import HasTraits, List, Unicode, Any, Int, observe, Union
from traittypes import Array

from abtem.visualize.interactive.utils import link


class GaussianFilterSlider(widgets.FloatSlider):
    image_in = Array()
    image_out = Array()

    def __init__(self, description='Gaussian filter', min=0., max=5, **kwargs):
        super().__init__(description=description, min=min, max=max, **kwargs)

    def update_image(self, *args):
        self.image_out = self(self.image_in)

    @observe('image_in')
    def _observe_image_in(self, *args):
        self.update_image()

    def __call__(self, image):
        return gaussian_filter(image, sigma=float(self.value))


class IntSliderWithButtons(widgets.HBox):
    value = Int(0)
    min = Int(0)
    max = Int(0)

    def __init__(self, **kwargs):
        self._slider = widgets.IntSlider(**kwargs)

        self._slider.layout.width = '300px'

        link((self._slider, 'value'), (self, 'value'))
        link((self._slider, 'min'), (self, 'min'))
        link((self._slider, 'max'), (self, 'max'))

        previous_button = widgets.Button(description='Previous')
        previous_button.layout.width = '100px'
        next_button = widgets.Button(description='Next')
        next_button.layout.width = '100px'

        def next_item(*args):
            self._slider.value += 1

        def previous_item(*args):
            self._slider.value -= 1

        next_button.on_click(next_item)
        previous_button.on_click(previous_item)

        super().__init__([self._slider, previous_button, next_button])


class ImageCollectionBrowser(HasTraits):
    paths = Union((List(), Unicode()))
    file_index = Int(0)
    path = Unicode()
    images = Array()
    image = Any()
    frame_index = Int(0)
    num_frames = Int(0)
    binning = Int(1)

    def __init__(self, **kwargs):
        self._path_text = widgets.HTML(description='File name')
        self._file_index_slider = IntSliderWithButtons(description='File index', min=0)
        self._frame_index_slider = IntSliderWithButtons(description='Frame index', min=0)
        link((self._file_index_slider, 'value'), (self, 'file_index'))
        link((self._frame_index_slider, 'value'), (self, 'frame_index'))
        super().__init__(**kwargs)

    @observe('frame_index')
    def _observe_frame_index(self, *args):
        image = self.images[self.frame_index]

        if self.binning != 1:
            image = downscale_local_mean(image, factors=(self.binning,) * 2)

        self.image = image

    @observe('path')
    def _observe_path(self, *args):
        self._path_text.value = str(Path(*Path(self.path).parts[-4:]))

        images = imread(self.path)

        # if images.shape[-1] in (3, 4):
        #    images = np.swapaxes(images, 0, 2)

        if len(images.shape) == 2:
            images = images[None]

        assert len(images.shape) == 3

        self.images = images

        self.num_frames = len(self.images)

        self._frame_index_slider.max = len(self.images) - 1

        if self.frame_index == 0:
            self._observe_frame_index()
        else:
            self.frame_index = 0

    @observe('file_index')
    def _observe_file_index(self, *args):
        self.path = str(self.paths[self.file_index])

    @observe('paths')
    def _observe_filename(self, change):

        self._file_index_slider.max = len(self.paths) - 1

        if self.file_index == 0:
            self._observe_file_index()
        else:
            self.file_index = 0

    @property
    def widgets(self):
        hbox = widgets.VBox([self._path_text, self._file_index_slider, self._frame_index_slider])
        return hbox
