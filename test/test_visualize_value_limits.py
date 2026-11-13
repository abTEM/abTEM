"""``set_value_limits`` of a shown 2D measurement keeps the colorbar and the radii
of scattered spots on the norm's limits.

``Visualization`` plots a CPU copy of the measurement, so these tests do not depend
on the device.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms

import abtem


def _values():
    return np.random.default_rng(0).random((32, 24)) + 0.5


def _indexed_diffraction_patterns():
    atoms = Atoms(
        "CO",
        positions=[(1.0, 1.0, 0.5), (2.5, 1.7, 2.5)],
        cell=(4.0, 3.0, 4.0),
        pbc=True,
    )
    probe = abtem.Probe(
        energy=100e3, semiangle_cutoff=20, extent=(4.0, 3.0), gpts=(40, 30)
    )
    patterns = probe.build(lazy=False).diffraction_patterns(max_angle=None)
    return patterns.index_diffraction_spots(cell=atoms.cell)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("complex_data", [False, True], ids=["real", "complex"])
def test_changing_only_vmin_updates_the_colorbar(complex_data):
    values = _values()
    data = values * np.exp(1j * values) if complex_data else values
    visualization = abtem.Images(data, sampling=0.1).show(cbar=True)
    fig = visualization.get_figure()
    cax = visualization.axes._caxes[0, 0][-1]

    for limits in ([0.0, 3.0], [0.2, 3.0]):
        visualization.set_value_limits(limits)
        fig.canvas.draw()

    assert cax.get_ylim() == pytest.approx((0.2, 3.0), rel=1e-6)


def test_changing_only_vmin_updates_the_spot_radii_and_colorbar():
    patterns = _indexed_diffraction_patterns()

    # the reference changes both limits at once, which signals the norm change
    reference = patterns.show(cbar=True)
    reference.set_value_limits([0.01, 3.0])
    reference.get_figure().canvas.draw()
    expected_radii = np.array(reference.artists[0, 0].circle_collection.get_radii())

    visualization = patterns.show(cbar=True)
    for limits in ([0.0, 3.0], [0.01, 3.0]):
        visualization.set_value_limits(limits)
        visualization.get_figure().canvas.draw()

    artist = visualization.artists[0, 0]
    cax = visualization.axes._caxes[0, 0][-1]
    assert cax.get_ylim() == pytest.approx((0.01, 3.0), rel=1e-6)
    np.testing.assert_allclose(
        artist.circle_collection.get_radii(), expected_radii, rtol=1e-6
    )


def test_first_draw_and_autoscale_keep_their_limits():
    values = _values()
    images = abtem.Images(values, sampling=0.1)

    visualization = images.show(cbar=True, vmin=0.7, vmax=1.2)
    visualization.get_figure().canvas.draw()
    cax = visualization.axes._caxes[0, 0][-1]
    assert cax.get_ylim() == pytest.approx((0.7, 1.2), rel=1e-6)

    visualization = images.show(cbar=True)
    visualization.get_figure().canvas.draw()
    cax = visualization.axes._caxes[0, 0][-1]
    assert cax.get_ylim() == pytest.approx((values.min(), values.max()), rel=1e-6)

    visualization.set_value_limits([0.0, 3.0])
    visualization.set_value_limits([None, None])
    visualization.get_figure().canvas.draw()
    assert cax.get_ylim() == pytest.approx((values.min(), values.max()), rel=1e-6)
