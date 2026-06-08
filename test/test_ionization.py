"""Tests for inelastic / core-loss simulation entry points.

These guard against regressions in the public API that the core-loss tutorial
depends on (see https://abtem.readthedocs.io/en/latest/user_guide/tutorials/core_loss.html).
The transition_potential_scan method was silently dropped in early 2025 and
restored later; the smoke tests below ensure it stays wired up.
"""
import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.waves import Probe


@pytest.fixture(scope="module")
def si_potential():
    atoms = ase.build.bulk("Si", cubic=True)
    return abtem.Potential(atoms, gpts=32, slice_thickness=2.7)


@pytest.fixture(scope="module")
def si_transition_potential(si_potential):
    # Synthetic transition potential: skips numerov / GPAW so the wiring is
    # exercised without depending on the heavy DFT setup the tutorial uses.
    n_transitions = 2
    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((n_transitions, *si_potential.gpts))
        + 1j * rng.standard_normal((n_transitions, *si_potential.gpts))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=14,
        array=array,
        energy=100e3,
        extent=si_potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


@pytest.fixture(scope="module")
def probe(si_potential):
    p = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    p.grid.match(si_potential)
    return p


def test_transition_potential_scan_is_defined_on_probe():
    """Guard against the Feb-2025 regression where the method was commented out."""
    assert hasattr(Probe, "transition_potential_scan")
    assert callable(Probe.transition_potential_scan)


def test_transition_potential_scan_builds_lazy_graph(
    probe, si_potential, si_transition_potential
):
    """Wiring smoke test: lazy call returns a measurements object without raising."""
    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=si_potential,
        transition_potentials=si_transition_potential,
        scan=(0, 0),
        detectors=detector,
        lazy=True,
    )
    assert result is not None
    assert hasattr(result, "compute")


def test_transition_potential_scan_forwards_inelastic_kwargs(
    probe, si_potential, si_transition_potential
):
    """double_channel / threshold must flow through to
    transition_potential_multislice_and_detect via **multislice_func_kwargs."""
    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=si_potential,
        transition_potentials=si_transition_potential,
        scan=(0, 0),
        detectors=detector,
        double_channel=False,
        threshold=0.95,
        lazy=True,
    )
    assert result is not None


def test_transition_potential_scan_accepts_grid_scan(
    probe, si_potential, si_transition_potential
):
    """The tutorial uses GridScan + sites=... for the EELS-map calls."""
    detector = abtem.FlexibleAnnularDetector()
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), fractional=True,
        potential=si_potential, endpoint=False, sampling=2.0,
    )
    result = probe.transition_potential_scan(
        potential=si_potential,
        transition_potentials=si_transition_potential,
        scan=scan,
        detectors=detector,
        sites=ase.build.bulk("Si", cubic=True),
        lazy=True,
    )
    assert result is not None
