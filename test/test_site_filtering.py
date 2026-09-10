"""Site filtering happens once per slice, not once per scatter chunk."""

import sys

import ase
import numpy as np
import pytest

import abtem
from abtem.core import config
from abtem.inelastic.core_loss import SubshellTransitions, TransitionPotentialArray

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

pytestmark = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")


@pytest.fixture(scope="module")
def setup():
    a = 2.504
    b = a * np.sqrt(3)
    base = ase.Atoms(
        "BNBN",
        positions=[(0, 0, 1.0), (0, b / 3, 1.0), (a / 2, b / 2, 1.0),
                   (a / 2, b / 2 + b / 3, 1.0)],
        cell=(a, b, 4.0), pbc=True,
    )
    atoms = base * (6, 4, 1)
    potential = abtem.Potential(atoms, gpts=(96, 96), slice_thickness=2.0)
    transitions = SubshellTransitions(Z=5, n=1, l=0, xc="PBE", order=1, epsilon=10)
    potentials = transitions.get_transition_potentials(energy=60e3)
    potentials.grid.match(potential)
    probe = abtem.Probe(semiangle_cutoff=32, energy=60e3)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(0, 0), end=(0.5, 0.5), gpts=(2, 2), fractional=True,
        potential=potential, endpoint=False,
    )
    return potential, potentials.build(), probe, scan, atoms[atoms.numbers == 5]


def _run(setup, scatter_max_batch, counter=None):
    potential, tp, probe, scan, sites = setup
    original = TransitionPotentialArray.filter_sites

    def counting(self, waves, sites, threshold=None, *args, **kwargs):
        # Only a call with a live threshold does the work that synchronises
        # the device; the others short-circuit.
        if counter is not None and threshold:
            counter["calls"] += 1
        return original(self, waves, sites, threshold=threshold, *args, **kwargs)

    TransitionPotentialArray.filter_sites = counting
    try:
        with config.set({"device": "cpu"}):
            measurement = probe.transition_potential_scan(
                scan=scan, potential=potential,
                detectors=abtem.FlexibleAnnularDetector(),
                transition_potentials=tp, double_channel=False, sites=sites,
                max_batch=2, threshold=0.9, lazy=True,
                scatter_max_batch=scatter_max_batch,
            ).integrate_radial(inner=0, outer=40)
            return np.asarray(
                measurement.compute(progress_bar=False).to_cpu().array
            )
    finally:
        TransitionPotentialArray.filter_sites = original


def test_filtering_does_not_scale_with_the_number_of_chunks(setup):
    """Chunking the sites more finely must not multiply the filter calls.

    filter_sites copies its mask back to the host, synchronising the device,
    so calling it per chunk stalls the pipeline once per chunk.
    """
    coarse, fine = {"calls": 0}, {"calls": 0}
    _run(setup, scatter_max_batch=64, counter=coarse)
    _run(setup, scatter_max_batch=1, counter=fine)

    assert fine["calls"] == coarse["calls"]
    # ... and it is one per slice, not one per chunk.
    assert fine["calls"] <= 8


def test_results_are_independent_of_the_chunk_size(setup):
    """The threshold is per site, so chunking cannot change which sites
    contribute -- only the order in which their contributions are summed,
    which at the configured precision is a round-off level difference.
    """
    coarse = _run(setup, scatter_max_batch=64)
    fine = _run(setup, scatter_max_batch=1)

    tolerance = 1e-5 if coarse.dtype == np.float32 else 1e-10
    assert np.allclose(coarse, fine, rtol=tolerance, atol=0)
