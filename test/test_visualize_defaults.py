"""Which ensemble item ``show()`` displays when it is not told (issue #515)."""

import importlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from utils import devices, si_cubic_atoms

import abtem
from abtem.core.axes import (
    LinearAxis,
    NonLinearAxis,
    OrdinalAxis,
    ParameterAxis,
    ThicknessAxis,
    TiltAxis,
)
from abtem.measurements import Images
from abtem.visualize.visualizations import default_ensemble_index


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


def test_slider_starts_at_the_coordinate_nearest_a_given_default_value():
    pytest.importorskip("ipywidgets")
    from abtem.visualize.widgets import slider_from_axes_metadata

    # four coordinates 0.0, 0.5, 1.0, 1.5 and three 10, 20, 30: a plain Python
    # number is the usual default value
    linear = LinearAxis(label="x", sampling=0.5, units="Å")
    energies = NonLinearAxis(label="E", values=(10.0, 20.0, 30.0), units="eV")
    assert slider_from_axes_metadata(linear, 4, "index", default_value=1.0).index == 2
    assert slider_from_axes_metadata(linear, 4, "index", default_value=1).index == 2
    assert slider_from_axes_metadata(energies, 3, "index", default_value=29).index == 2
    assert slider_from_axes_metadata(energies, 3, "range").index == (0, 2)
    assert slider_from_axes_metadata(
        linear, 4, "range", default_value=(0.4, 1.1)
    ).index == (1, 2)

    # coordinates that are not numbers have no nearest one: the first is shown
    names = OrdinalAxis(label="name", values=("a", "b", "c"))
    tilts = TiltAxis(values=((0.0, 0.0), (5.0, 5.0)))
    assert slider_from_axes_metadata(names, 3, "index", default_value="c").index == 0
    assert slider_from_axes_metadata(tilts, 2, "index", default_value=5.0).index == 0
    # a pair is not a value of an index slider, also where it has the axis length
    pair = NonLinearAxis(label="E", values=(10.0, 20.0), units="eV")
    assert (
        slider_from_axes_metadata(pair, 2, "index", default_value=(19, 21)).index == 0
    )


def test_slider_lets_an_unrelated_error_through(monkeypatch):
    pytest.importorskip("ipywidgets")
    from abtem.visualize import widgets

    def nearest_index(values, value):
        raise RuntimeError("stand-in for an unrelated error")

    monkeypatch.setattr(widgets, "_nearest_index", nearest_index, raising=False)
    linear = LinearAxis(label="x", sampling=0.5, units="Å")
    with pytest.raises(RuntimeError, match="stand-in"):
        widgets.slider_from_axes_metadata(linear, 4, "index", default_value=1.0)


def test_a_missing_ipywidgets_is_reported_by_the_import_of_the_widgets(monkeypatch):
    monkeypatch.delitem(sys.modules, "abtem.visualize.widgets", raising=False)
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    with pytest.raises(ImportError, match="ipywidgets"):
        importlib.import_module("abtem.visualize.widgets")
