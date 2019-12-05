import io

import PIL
import ipywidgets as widgets
import numpy as np


def array_as_image(array):
    image = PIL.Image.fromarray(array)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='png')
    return bytes_io.getvalue()


class ArrayImage(widgets.Image):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_array(self, array):
        self.value = array_as_image((array * 255).astype(np.uint8))


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