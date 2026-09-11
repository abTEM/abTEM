"""Transition potentials travel as single graph nodes, not per-task copies."""

import cloudpickle
import numpy as np
import pytest

import abtem
from abtem.core import config

from utils import synthetic_transition_potential


@pytest.fixture(autouse=True)
def on_cpu():
    """Every test in this module exercises the CPU graph machinery."""
    with config.set({"device": "cpu"}):
        yield


def _synthetic_tp(gpts=(64, 64), extent=(8.0, 8.0), n_transitions=4, seed=0):
    return synthetic_transition_potential(
        Z=5, gpts=gpts, extent=extent, energy=60e3,
        n_transitions=n_transitions, seed=seed,
    )


def _setup(gpts=(64, 64)):
    import ase

    atoms = ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True
    )
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=2.0)
    tp = _synthetic_tp(gpts=gpts, extent=potential.extent)
    probe = abtem.Probe(semiangle_cutoff=32, energy=60e3)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(4, 4), fractional=True, potential=potential
    )
    return potential, tp, probe, scan, atoms[atoms.numbers == 5]


def _scan(probe, potential, tp, scan, sites, lazy=True, threshold=1.0, **kwargs):
    # threshold=1.0 disables the per-task intensity filter: the surviving
    # site set is then independent of how the scan is chunked, which is what
    # makes eager and lazy results bit-comparable.
    return probe.transition_potential_scan(
        scan=scan, potential=potential, detectors=abtem.FlexibleAnnularDetector(),
        transition_potentials=tp, double_channel=False, sites=sites,
        max_batch=2, threshold=threshold, lazy=lazy, **kwargs,
    )


def test_graph_carries_the_transition_potential_once():
    """A many-task lazy scan's graph must hold exactly one payload copy."""
    potential, tp, probe, scan, sites = _setup()
    measurement = _scan(probe, potential, tp, scan, sites)

    graph = dict(measurement.array.__dask_graph__())
    payload = tp.array.nbytes
    sizes = [len(cloudpickle.dumps(value)) for value in graph.values()]

    assert len(graph) > 10  # genuinely a multi-task graph
    big = [size for size in sizes if size > payload / 2]
    assert len(big) == 1, f"expected one payload-sized key, got {len(big)}"
    # Total task payload stays ~one copy, not one per scan chunk.
    assert sum(sizes) < 2.5 * payload


def test_lazy_threads_and_eager_agree():
    potential, tp, probe, scan, sites = _setup()

    lazy = _scan(probe, potential, tp, scan, sites)
    result = np.asarray(
        lazy.compute(progress_bar=False, scheduler="threads").to_cpu().array
    )
    reference = np.asarray(
        _scan(probe, potential, tp, scan, sites, lazy=False).to_cpu().array
    )

    assert np.array_equal(result, reference)


def test_threaded_and_synchronous_schedulers_agree():
    """The same graph under concurrent threads must match the synchronous
    run bit for bit -- the regression this guards is concurrent tasks racing
    on a shared transition potential's grid/accelerator state."""
    potential, tp, probe, scan, sites = _setup()
    lazy = _scan(probe, potential, tp, scan, sites, threshold=0.5)

    threaded = np.asarray(
        lazy.copy().compute(
            progress_bar=False, scheduler="threads", num_workers=8
        ).to_cpu().array
    )
    synchronous = np.asarray(
        lazy.compute(progress_bar=False, scheduler="synchronous").to_cpu().array
    )

    assert np.array_equal(threaded, synchronous)


def test_graph_computes_on_a_distributed_cluster_and_survives_client_loss():
    """The graph is self-contained: it computes on a cluster, and the same
    graph still computes locally after that cluster is gone."""
    distributed = pytest.importorskip("distributed")

    potential, tp, probe, scan, sites = _setup()
    lazy = _scan(probe, potential, tp, scan, sites)

    with distributed.LocalCluster(
        n_workers=2, processes=True, threads_per_worker=1,
        dashboard_address=None,
    ) as cluster, distributed.Client(cluster):
        on_cluster = np.asarray(
            lazy.copy().compute(progress_bar=False).to_cpu().array
        )

    # The client is closed now; the identical graph must still compute with
    # the local scheduler -- the graph is self-contained.
    local = np.asarray(
        lazy.compute(progress_bar=False, scheduler="threads").to_cpu().array
    )

    assert np.array_equal(on_cluster, local)


def test_prism_lazy_and_eager_agree():
    potential, tp, _, scan, sites = _setup()
    s_matrix = abtem.SMatrix(
        potential=potential, energy=60e3, semiangle_cutoff=32, interpolation=1
    )

    def run(lazy):
        m = s_matrix.transition_potential_scan(
            transition_potentials=tp, scan=scan,
            detectors=abtem.FlexibleAnnularDetector(), sites=sites,
            double_channel=False, lazy=lazy,
        )
        if lazy:
            m = m.compute(progress_bar=False)
        return np.asarray(m.to_cpu().array)

    assert np.array_equal(run(lazy=True), run(lazy=False))


def test_task_local_view_shields_the_shared_object():
    """The driver's grid/accelerator matching must not leak into the shared
    transition potential object."""
    tp = _synthetic_tp()

    view = tp._task_local()
    assert view.array is tp.array  # payload shared
    assert view._local_potential is tp._local_potential
    assert view._grid is not tp._grid
    assert view._accelerator is not tp._accelerator

    view.accelerator.energy = 300e3
    view.grid.sampling = (0.5, 0.5)
    assert tp.energy == 60e3
    assert tp.extent == (8.0, 8.0)


def test_scan_leaves_the_users_object_unmutated():
    """End to end: after lazy compute, the user's transition potential still
    has exactly the state it was built with."""
    potential, tp, probe, scan, sites = _setup()
    energy_before = tp.energy
    extent_before = tp.extent
    array_before = tp.array

    _scan(probe, potential, tp, scan, sites).compute(progress_bar=False)

    assert tp.energy == energy_before
    assert tp.extent == extent_before
    assert tp.array is array_before
