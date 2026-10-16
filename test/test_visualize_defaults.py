"""Which ensemble item ``show()`` displays when it is not told (issue #515)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from utils import devices, si_cubic_atoms

import abtem
from abtem.core.axes import ParameterAxis, ThicknessAxis
from abtem.measurements import Images
from abtem.visualize.widgets import default_ensemble_index


def _drawn(measurement):
    measurement.show()
    image = np.array(plt.gcf().axes[0].images[0].get_array())
    plt.close("all")
    return image


@devices
def test_show_defaults_to_the_exit_surface_of_a_thickness_series(device):
    # an integer exit_planes puts the unscattered entrance plane at index 0; showing
    # it by default made a scanned COM or ADF image look empty
    potential = abtem.Potential(
        si_cubic_atoms(), gpts=32, slice_thickness=1.0, exit_planes=2, device=device
    )
    waves = abtem.PlaneWave(energy=100e3, device=device).multislice(potential)
    intensity = waves.intensity().compute()

    assert isinstance(intensity.axes_metadata[0], ThicknessAxis)
    assert intensity.axes_metadata[0].values[0] == 0.0

    drawn = _drawn(intensity)

    np.testing.assert_array_equal(drawn, _drawn(intensity[-1]))
    assert not np.allclose(drawn, _drawn(intensity[0]))


def test_show_defaults_to_the_first_item_of_other_ensemble_axes():
    array = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4)
    images = Images(
        array,
        sampling=0.1,
        ensemble_axes_metadata=[ParameterAxis(values=(1.0, 2.0, 3.0))],
    )

    np.testing.assert_array_equal(_drawn(images), _drawn(images[0]))


def test_default_ensemble_index():
    assert default_ensemble_index(ThicknessAxis(values=(0.0, 1.0, 2.0)), 3) == 2
    assert default_ensemble_index(ParameterAxis(values=(0.0, 1.0, 2.0)), 3) == 0


def test_thickness_slider_starts_at_the_exit_surface():
    pytest.importorskip("ipywidgets")
    from abtem.visualize.widgets import slider_from_axes_metadata

    thickness = slider_from_axes_metadata(ThicknessAxis(values=(0.0, 1.0, 2.0)), 3)
    parameter = slider_from_axes_metadata(ParameterAxis(values=(0.0, 1.0, 2.0)), 3)

    assert thickness.index == 2
    assert parameter.index == 0
