import numpy as np
from bokeh import io
from bokeh.models import ColumnDataSource, Label
from bokeh.palettes import Category10
from bokeh.plotting import Figure

from abtem.waves import _Probelike
from abtem.measure import calculate_fwhm


def _set_image_data_source(measurement, source, push_notebook):
    calibrations = measurement.calibrations[-2:]
    array = measurement.array[(0,) * (measurement.dimensions - 2) + (slice(None),) * 2]
    source.data = {'image': [array.T],
                   'x': [calibrations[-1].offset],
                   'y': [calibrations[-2].offset],
                   'dh': [calibrations[-1].sampling * measurement.array.shape[-1]],
                   'dw': [calibrations[-2].sampling * measurement.array.shape[-2]], }
    if push_notebook:
        io.push_notebook()


def _set_line_source(measurement, source, push_notebook):
    calibration = measurement.calibrations[0]
    x = np.linspace(calibration.offset, calibration.offset + len(measurement) * calibration.sampling,
                    len(measurement))
    source.data = {'x': x, 'y': measurement.array}
    if push_notebook:
        io.push_notebook()


def probe_image(probelike, p=None, push_notebook=False, palette='Viridis256', **kwargs):
    if not isinstance(probelike, _Probelike):
        raise RuntimeError()

    if p is None:
        p = Figure(plot_width=400, plot_height=400, match_aspect=True)

    source = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
    p.image(image='image', x='x', y='y', dw='dw', dh='dh', source=source, palette=palette, **kwargs)

    def callback(*args, **kwargs):
        _set_image_data_source(probelike.intensity(), source, push_notebook=push_notebook)

    _set_image_data_source(probelike.intensity(), source, push_notebook=False)
    probelike.changed.register(callback)
    return p


def probe_profile(probelike, show_fwhm=True, p=None, push_notebook=False, **kwargs):
    if not isinstance(probelike, _Probelike):
        raise RuntimeError()

    if p is None:
        p = Figure(plot_width=400, plot_height=400)

    source = ColumnDataSource(data=dict(x=[], y=[]))
    p.line(x='x', y='y', source=source, **kwargs)

    if show_fwhm:
        fwhm = calculate_fwhm(probelike.profile())

        label = Label(x=.05 * p.plot_width,
                      y=.85 * p.plot_height,
                      x_units='screen', y_units='screen',
                      text=f'FWHM: {fwhm:.3f} Å',
                      text_font_size='10pt',
                      render_mode='css',
                      text_align='left',
                      text_baseline='top')
        p.add_layout(label)

    def callback(*args, **kwargs):
        profile = probelike.profile()
        _set_line_source(profile, source, push_notebook=push_notebook)

        if show_fwhm:
            label.text = f'FWHM: {calculate_fwhm(profile):.3f} Å'

    _set_line_source(probelike.profile(), source, push_notebook=False)
    probelike.changed.register(callback)
    return p


def ctf_profile(ctf, max_semiangle=None, p=None, push_notebook=False, **kwargs):
    if p is None:
        p = Figure(plot_width=400, plot_height=400)

    profiles = ctf.profiles(max_semiangle)

    lines = {}
    for (key, profile), color in zip(profiles.items(), Category10[10]):
        visible = not np.all(profile.array == 1.)
        source = ColumnDataSource(data=dict(x=[], y=[]))
        line = p.line(x='x', y='y', source=source, line_color=color, legend_label=profile.name, visible=visible,
                      **kwargs)
        lines[key] = line

    def callback(*args, **kwargs):
        for key, profile in ctf.profiles(max_semiangle).items():
            _set_line_source(profile, lines[key].data_source, push_notebook=push_notebook)

            if np.all(profile.array == 1.):
                lines[key].visible = False
            else:
                lines[key].visible = True

        if push_notebook:
            io.push_notebook()

    for key, profile in ctf.profiles(max_semiangle).items():
        _set_line_source(profile, lines[key].data_source, push_notebook=False)

    ctf.changed.register(callback)

    p.legend.location = 'bottom_right'
    p.legend.click_policy = 'hide'
    p.legend.label_text_font_size = '10pt'
    return p


def hrtem_image(wave, ctf, p=None, push_notebook=False, palette='Greys256', **kwargs):
    if p is None:
        p = Figure(plot_width=400, plot_height=400, match_aspect=True)

    source = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
    p.image(image='image', x='x', y='y', dw='dw', dh='dh', source=source, palette=palette, **kwargs)

    def callback(*args, **kwargs):
        _set_image_data_source(wave.apply_ctf(ctf).intensity(), source, push_notebook=push_notebook)

    _set_image_data_source(wave.apply_ctf(ctf).intensity(), source, push_notebook=False)
    ctf.changed.register(callback)
    return p
