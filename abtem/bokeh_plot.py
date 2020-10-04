from bokeh.models import Slider
from bokeh import plotting
from bokeh.models import ColumnDataSource


def new_image(figure, measurement):



    pass

def update_image():
    pass

def show_image():
    p = plotting.Figure(plot_width=400, plot_height=400, match_aspect=True)
    source = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
    glyph = p.image(image='image', x='x', y='y', dw='dw', dh='dh', source=source, palette=palette, **kwargs)



# def quick_sliders(obj, **kwargs):
#     def create_callback(attr):
#         def callback(_, __, new):
#             setattr(obj, attr, new)
#
#         return callback
#
#     sliders = []
#     for key, value in kwargs.items():
#         slider = Slider(value=getattr(obj, key),
#                         start=value[0],
#                         end=value[1],
#                         step=value[2],
#                         title=key)
#
#         slider.on_change('value', create_callback(key))
#         sliders.append(slider)
#
#     return sliders
