"""Tests for the Monte Carlo plasmon scattering module."""

import math

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem.core.backend import asnumpy
from abtem.inelastic.plasmons import (
    MonteCarloPlasmons,
    PlasmonAxis,
    PlasmonScatteringEvents,
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


def test_draw_events_weights_are_poisson():
    potential = abtem.Potential(bulk("Si", cubic=True), gpts=16, slice_thickness=2)
    waves = abtem.PlaneWave(energy=100e3, gpts=16, extent=potential.extent)
    plasmons = MonteCarloPlasmons(
        mean_free_path=10.0,
        excitation_energy=15.0,
        critical_angle=10.0,
        num_excitations=2,
        num_samples=3,
        seed=0,
    )
    events = plasmons.draw_events(waves, potential)

    x = potential.thickness / plasmons.mean_free_path
    expected = [math.exp(-x)] + [x * math.exp(-x)] * 3 + [x**2 * math.exp(-x) / 2] * 3
    np.testing.assert_allclose(events.weights, expected)
    assert [len(depths) for depths in events.depths] == [0, 1, 1, 1, 2, 2, 2]
    assert events.ensemble_shape == (7,)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmon_deflection_in_multislice(device, lazy):
    potential = abtem.Potential(
        bulk("Si", cubic=True) * (1, 1, 2),
        gpts=24,
        slice_thickness=2,
        device=device,
    ).build(lazy=False)
    waves = abtem.PlaneWave(
        energy=100e3, gpts=24, extent=potential.extent, device=device
    ).build(lazy=lazy)
    thickness = potential.thickness
    theta, phi = 5.0, 0.3
    # The deflection applies from the first slice boundary below the event.
    k = potential.num_slices // 2
    depth = sum(potential.slice_thickness[:k]) - 1e-3

    events = PlasmonScatteringEvents(
        depths=((), (thickness + 1.0,), (depth,), (depth, depth)),
        radial_angles=((), (theta,), (theta,), (theta, theta)),
        azimuthal_angles=((), (phi,), (phi,), (phi, phi + math.pi)),
        weights=(1.0,) * 4,
        ensemble_mean=False,
    )
    exit_waves = events.apply(waves).multislice(potential).compute()
    assert isinstance(exit_waves.ensemble_axes_metadata[0], PlasmonAxis)
    zero_loss, beyond_exit, single, opposite_pair = asnumpy(exit_waves.array)

    upper = waves.multislice(potential[:k]).compute()
    upper.metadata.update(
        base_tilt_x=theta * math.cos(phi), base_tilt_y=theta * math.sin(phi)
    )
    expected_single = asnumpy(upper.multislice(potential[k:]).compute().array)

    # An event below the exit surface never happens.
    np.testing.assert_array_equal(beyond_exit, zero_loss)
    # Equal and opposite deflections cancel.
    np.testing.assert_allclose(opposite_pair, zero_loss, atol=1e-5)
    assert np.abs(single - zero_loss).max() > 1e-2
    np.testing.assert_allclose(single, expected_single, atol=1e-5)


@pytest.mark.parametrize("thickness, mean_free_path", [(10.0, 1000.0), (500.0, 800.0)])
def test_excitations_weights_are_poisson_probabilities(thickness, mean_free_path):
    # np.math was removed in NumPy 2, which made excitations_weights raise
    # AttributeError for every input.
    from scipy.stats import poisson

    n = np.arange(0, 20)
    weights = np.array([excitations_weights(i, thickness, mean_free_path) for i in n])

    assert np.allclose(weights, poisson.pmf(n, thickness / mean_free_path))
    assert excitations_weights(0, thickness, mean_free_path) == pytest.approx(
        np.exp(-thickness / mean_free_path)
    )
