import ipywidgets as widgets
import numpy as np
from bqplot import Figure, LinearScale, Axis
from traitlets import HasTraits, observe, Dict, Instance, link, Bool, List, default, Float
from traitlets import Unicode


class ImageShow:

    def __init__(self):
        x_scale = LinearScale(allow_padding=False)
        y_scale = LinearScale(allow_padding=False)



        figure = Figure(scales=scales,
                        axes=[x_axis, y_axis],
                        min_aspect_ratio=min_aspect_ratio,
                        max_aspect_ratio=max_aspect_ratio,
                        fig_margin=fig_margin)