"""Fixes found comparing the artists with each other: the log scale of every artist
and the units of the y axis of an image."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from abtem.core.axes import RealSpaceAxis
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


def test_log_scale_of_a_complex_image_does_not_raise():
    # a domain-colouring artist inherited a set_logscale that took no argument
    rng = np.random.default_rng(0)
    images = Images(rng.random((8, 8)) + 1j * rng.random((8, 8)), sampling=0.1)
    visualization = images.show()

    visualization.set_logscale(False)
    visualization.set_logscale(True)


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
