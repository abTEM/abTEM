"""Fixes found comparing the artists with each other: the log scale of every artist
and the units of the y axis of an image."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

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


def test_show_atoms_refuses_a_plane_that_is_not_a_string_of_axes():
    # a pair of angles was documented but never implemented, and failed with a
    # KeyError of the first angle
    import ase.build

    from abtem import show_atoms

    with pytest.raises(ValueError, match="string of axes"):
        show_atoms(ase.build.bulk("Si", cubic=True), plane=(45.0, 45.0))
