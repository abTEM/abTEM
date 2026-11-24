"""Fixes found comparing the artists with each other: the log scale of every artist
and the units of the y axis of an image."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

from abtem.core.axes import OrdinalAxis, RealSpaceAxis
from abtem.measurements import Images, RealSpaceLineProfiles
from abtem.visualize.artists import get_extent


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _line_profile_visualization():
    return RealSpaceLineProfiles(np.linspace(1, 100, 32), sampling=0.1).show()


def test_log_scale_of_a_line_profile_can_be_switched_on_and_off():
    visualization = _line_profile_visualization()
    axes = np.asarray(visualization.axes).ravel()[0]

    visualization.set_logscale(True)
    assert axes.get_yscale() == "log"

    visualization.set_logscale(False)
    assert axes.get_yscale() == "linear"


def test_log_scale_of_a_line_profile_is_reported():
    artist = _line_profile_visualization().artists.ravel()[0]

    assert artist.get_logscale() is False
    artist.set_logscale(True)
    assert artist.get_logscale() is True


def test_log_scale_of_a_complex_image_can_be_switched_on_and_off():
    # a domain-colouring artist inherited a set_logscale that took no argument, and
    # then one that did nothing
    rng = np.random.default_rng(0)
    # amplitudes over several decades, so that the log scale changes the opacity
    array = (10 ** rng.uniform(-3, 0, (8, 8))) * np.exp(2j * np.pi * rng.random((8, 8)))
    images = Images(array, sampling=0.1)
    visualization = images.show()
    artist = visualization.artists.ravel()[0]
    linear_alpha = np.array(artist.phase_axes_image.get_alpha())

    reference = images.show(logscale=True).artists.ravel()[0]

    visualization.set_logscale(True)
    assert isinstance(artist.amplitude_norm, LogNorm)
    np.testing.assert_allclose(
        artist.phase_axes_image.get_alpha(), reference.phase_axes_image.get_alpha()
    )
    assert not np.allclose(artist.phase_axes_image.get_alpha(), linear_alpha)

    visualization.set_logscale(False)
    assert not isinstance(artist.amplitude_norm, LogNorm)
    np.testing.assert_allclose(artist.phase_axes_image.get_alpha(), linear_alpha)


def _norms(visualization, complex_values):
    artists = visualization.artists.ravel()
    return [
        artist.amplitude_norm if complex_values else artist.norm for artist in artists
    ]


@pytest.mark.parametrize("zeros", ["pixel", "member"])
@pytest.mark.parametrize("explode", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_common_color_scale_of_a_log_scale_image_with_a_zero_value(
    complex_values, explode, zeros
):
    # a limit of 0 is not valid for a LogNorm: the common limits, taken over the
    # whole ensemble, give a non-positive vmin when a pixel or a whole member is
    # zero. The members differ in scale, so that a per-member vmin would differ.
    rng = np.random.default_rng(0)
    array = rng.random((2, 8, 8)) + 0.5
    array[1] *= 10
    if complex_values:
        array = array * np.exp(2j * np.pi * rng.random((2, 8, 8)))
    if zeros == "pixel":
        array[0, 0, 0] = 0
    else:
        array[0] = 0
    images = Images(
        array, sampling=0.1, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
    )

    visualization = images.show(logscale=True, common_color_scale=True, explode=explode)
    plt.gcf().canvas.draw()

    magnitude = np.abs(array)
    smallest_positive = magnitude[magnitude > 0].min()
    norms = _norms(visualization, complex_values)
    assert len(norms) == (2 if explode else 1)
    assert all(isinstance(norm, LogNorm) for norm in norms)
    assert [norm.vmin for norm in norms] == [smallest_positive] * len(norms)
    assert [norm.vmax for norm in norms] == [magnitude.max()] * len(norms)


@pytest.mark.parametrize("value", [-1.0, 0.0])
@pytest.mark.parametrize("complex_values", [False, True])
def test_common_color_scale_of_a_log_scale_image_without_a_positive_value(
    complex_values, value
):
    # nothing to replace the limits with: they stay as they are, and the image draws
    array = np.full((2, 4, 4), value, dtype=complex if complex_values else float)
    images = Images(
        array, sampling=0.1, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
    )

    images.show(logscale=True, common_color_scale=True, explode=True)
    plt.gcf().canvas.draw()


@pytest.mark.parametrize("complex_values", [False, True])
def test_value_limits_of_a_log_scale_image_can_start_at_zero(complex_values):
    rng = np.random.default_rng(0)
    array = rng.random((8, 8)) + 0.5
    if complex_values:
        array = array * np.exp(2j * np.pi * rng.random((8, 8)))
    visualization = Images(array, sampling=0.1).show(logscale=True)

    visualization.set_value_limits((0.0, 10.0))
    plt.gcf().canvas.draw()

    (norm,) = _norms(visualization, complex_values)
    assert 0 < norm.vmin <= np.abs(array).min()
    assert norm.vmax == 10.0


def test_extent_of_an_image_converts_each_axis_with_its_own_units():
    class Mixed:
        """A measurement whose x axis is in Å and y axis in nm."""

        metadata = {}
        base_shape = (4, 6)
        base_axes_metadata = [
            RealSpaceAxis(sampling=0.1, units="Å"),
            RealSpaceAxis(sampling=0.2, units="nm"),
        ]

    left, right, bottom, top = get_extent(Mixed, units="Å")

    assert right - left == pytest.approx(0.1 * 4)  # already in Å
    assert top - bottom == pytest.approx(0.2 * 6 * 10)  # converted from nm
