import io

import PIL.Image
import ipywidgets as widgets
import numpy as np


def array_as_image(array):
    image = PIL.Image.fromarray(array)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='png')
    return bytes_io.getvalue()


class ArrayImage(widgets.Image):

    def __init__(self, array=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if array is not None:
            self.set_array(array)

    def set_array(self, array):
        if array.dtype == np.uint8:
            self.value = array_as_image(array.astype(np.uint8))

        else:
            if len(array.shape) != 2:
                raise RuntimeError()

            array = (array - array.min()) / array.ptp() * 255

            self.value = array_as_image(array.astype(np.uint8))


class Playback(widgets.VBox):

    def __init__(self, func, frames, **kwargs):
        def on_frame_change(change):
            func(change['new'])

        int_slider = widgets.IntSlider(min=0, max=frames - 1, **kwargs)
        int_slider.observe(on_frame_change, names='value')

        def next_frame(_):
            int_slider.value = (int_slider.value + 1) % (int_slider.max + 1)

        next_button = widgets.Button(description='Next')
        next_button.on_click(next_frame)

        def previous_frame(_):
            int_slider.value = (int_slider.value - 1) % (int_slider.max + 1)

        previous_button = widgets.Button(description='Previous')
        previous_button.on_click(previous_frame)

        super().__init__(children=(widgets.HBox((next_button, previous_button)), int_slider))

        on_frame_change({'new': 0})
