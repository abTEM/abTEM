"""Tests for core-loss transition potentials."""

import ase
import numpy as np
import pytest

import abtem
from abtem.core import config
from abtem.inelastic.core_loss import SubshellTransitions


@pytest.fixture(scope="module")
def transitions():
    return SubshellTransitions(Z=5, n=1, l=0, xc="PBE", order=1, epsilon=10)


def _build(transitions, gpts=(64, 64), extent=(10.0, 10.0)):
    potentials = transitions.get_transition_potentials(energy=60e3)
    potentials.grid.gpts = gpts
    potentials.grid.extent = extent
    return potentials.build()


@pytest.mark.parametrize(
    "precision, expected",
    [("float32", np.complex64), ("float64", np.complex128)],
)
def test_build_dtype_follows_precision(transitions, precision, expected):
    with config.set({"precision": precision}):
        assert _build(transitions).array.dtype == expected


def test_scatter_preserves_wave_precision(transitions):
    """Transition potentials must not upcast the waves they scatter.

    Dividing by the sampling product promoted the built array to complex128
    regardless of the configured precision, which silently ran the whole
    inelastic channel -- the ifft2 in scatter and every wave function derived
    from it -- in double precision.
    """
    with config.set({"precision": "float32", "device": "cpu"}):
        built = _build(transitions, gpts=(64, 64), extent=(10.0, 10.0))
        probe = abtem.Probe(
            semiangle_cutoff=32, energy=60e3, gpts=(64, 64), extent=(10.0, 10.0)
        )
        waves = probe.build(lazy=False)
        sites = ase.Atoms("B", positions=[(5.0, 5.0, 0.0)], cell=(10, 10, 2))

        scattered = built.scatter(waves, sites)

        assert waves.array.dtype == np.complex64
        assert scattered.array.dtype == np.complex64


def test_build_matches_double_precision_reference(transitions):
    """Casting down must not change the result beyond float32 round-off."""
    with config.set({"precision": "float64"}):
        reference = _build(transitions).array
    with config.set({"precision": "float32"}):
        single = _build(transitions).array

    scale = np.abs(reference).max()
    assert np.abs(single.astype(np.complex128) - reference).max() / scale < 1e-6
