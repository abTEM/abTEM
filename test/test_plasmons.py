"""Tests for the Monte Carlo plasmon scattering module."""

import math

import numpy as np
import pytest

from abtem.inelastic.plasmons import (
    PlasmonAxis,
    excitations_weights,
)


def _plasmon_axis(radial_angles, azimuthal_angles, excitations):
    depths = tuple(float(i + 1) for i in range(len(radial_angles)))
    return PlasmonAxis(
        values=((depths, tuple(radial_angles), tuple(azimuthal_angles), excitations),)
    )


@pytest.mark.parametrize(
    "radial_angles, azimuthal_angles, expected",
    [
        # Equal and opposite deflections cancel.
        ((5.0, 5.0), (0.3, 0.3 + math.pi), (0.0, 0.0)),
        # Perpendicular deflections add in quadrature.
        ((3.0, 4.0), (0.0, math.pi / 2), (3.0, 4.0)),
        # Parallel deflections add linearly along the shared direction.
        ((2.0, 1.5), (math.pi / 2, math.pi / 2), (0.0, 3.5)),
    ],
)
def test_tilt_is_vector_sum_of_deflections(radial_angles, azimuthal_angles, expected):
    axis = _plasmon_axis(radial_angles, azimuthal_angles, excitations=2)
    np.testing.assert_allclose(axis.tilt, (expected,), atol=1e-12)


def test_tilt_counts_only_excitations_above_current_depth():
    axis = _plasmon_axis((3.0, 4.0), (0.0, math.pi / 2), excitations=0)
    assert axis.tilt == ((0.0, 0.0),)

    # Between the first (depth 1) and second (depth 2) scattering event.
    axis.update(1.5)
    np.testing.assert_allclose(axis.tilt, ((3.0, 0.0),), atol=1e-12)

    axis.update(10.0)
    np.testing.assert_allclose(axis.tilt, ((3.0, 4.0),), atol=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_excitations_weights_are_poisson(n):
    thickness, mean_free_path = 250.0, 100.0
    x = thickness / mean_free_path
    expected = x**n * math.exp(-x) / math.factorial(n)
    assert excitations_weights(n, thickness, mean_free_path) == pytest.approx(expected)
