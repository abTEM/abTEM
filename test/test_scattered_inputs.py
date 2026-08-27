"""Tests for inputs scattered onto workers instead of embedded in the graph."""

import pickle

import cloudpickle
import numpy as np
import pytest

from abtem.core.backend import (
    ScatteredInput,
    resolve_scattered,
    scatter_to_workers,
)


@pytest.fixture(scope="module")
def client():
    """One cluster for the whole module: repeated LocalCluster startup and
    teardown in a single process is flaky when tests are randomly ordered."""
    distributed = pytest.importorskip("distributed")

    with distributed.LocalCluster(
        n_workers=2, processes=True, threads_per_worker=1, dashboard_address=None
    ) as cluster, distributed.Client(cluster) as running_client:
        yield running_client


def _sum_payload(index, payload=None):
    """Mimics abTEM: the payload rides in the task's *function*, not its args."""
    value = resolve_scattered(payload)
    return float(np.asarray(value).sum()) + index


def test_resolve_scattered_passes_through_plain_objects():
    array = np.ones((4, 4))
    assert resolve_scattered(array) is array
    assert resolve_scattered(None) is None
    assert resolve_scattered("not scattered") == "not scattered"


def test_scattered_handle_pickles_small_and_keeps_metadata(client):
    big = np.ones((512, 512), dtype=np.complex64)  # 2 MB
    handle = scatter_to_workers(big, client=client)

    assert isinstance(handle, ScatteredInput)
    assert len(pickle.dumps(handle)) < 10_000  # vs 2 MB embedded
    assert resolve_scattered(handle) is not None


def test_resolution_survives_a_worker_restart(client):
    """A restarted worker must still find the value.

    This is the failure that killed a production run with client.scatter:
    losing a worker lost the scattered data unrecoverably. A worker plugin's
    setup runs again on the replacement.
    """
    array = np.arange(9, dtype=np.float64)
    handle = scatter_to_workers(array, client=client)

    client.restart()

    from functools import partial

    func = partial(_sum_payload, payload=handle)
    results = client.gather([client.submit(func, i, pure=False) for i in range(4)])
    assert results == [float(array.sum()) + i for i in range(4)]


def test_missing_handle_raises_a_clear_error():
    handle = ScatteredInput("abtem-input-never-registered")
    with pytest.raises(RuntimeError, match="not available in this process"):
        resolve_scattered(handle)


def test_scattered_input_resolves_inside_tasks(client):
    from functools import partial

    array = np.arange(16, dtype=np.float64).reshape(4, 4)
    handle = scatter_to_workers(array, client=client)

    func = partial(_sum_payload, payload=handle)
    results = client.gather([client.submit(func, i, pure=False) for i in range(6)])

    assert results == [float(array.sum()) + i for i in range(6)]


def test_metadata_survives_on_the_handle(client):
    class WithMetadata:
        def __init__(self):
            self.metadata = {"Z": 5, "n": 1, "l": 0}

    handle = scatter_to_workers(WithMetadata(), client=client)
    assert handle.metadata == {"Z": 5, "n": 1, "l": 0}


def _small_eels_setup():
    import ase

    import abtem
    from abtem.inelastic.core_loss import SubshellTransitions

    atoms = ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True
    )
    potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
    transitions = SubshellTransitions(Z=5, n=1, l=0, xc="PBE", order=1, epsilon=10)
    potentials = transitions.get_transition_potentials(energy=60e3)
    potentials.grid.match(potential)
    probe = abtem.Probe(semiangle_cutoff=32, energy=60e3)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(0, 0), end=(0.5, 0.5), gpts=(2, 2), fractional=True, potential=potential
    )
    return potential, potentials.build(), probe, scan, atoms[atoms.numbers == 5]


def _run_scan(probe, potential, tp, scan, sites):
    import abtem
    import numpy as np

    measurement = probe.transition_potential_scan(
        scan=scan, potential=potential, detectors=abtem.FlexibleAnnularDetector(),
        transition_potentials=tp, double_channel=False, sites=sites,
        max_batch=2, threshold=0.5, lazy=True,
    ).integrate_radial(inner=0, outer=40)
    return np.asarray(measurement.compute(progress_bar=False).to_cpu().array)


def test_scattered_transition_potential_matches_embedded(client):
    from abtem.core import config

    potential, tp, probe, scan, sites = _small_eels_setup()

    with config.set({"device": "cpu"}):
        reference = _run_scan(probe, potential, tp, scan, sites)
        handle = scatter_to_workers(tp, client=client)
        scattered = _run_scan(probe, potential, handle, scan, sites)

    assert np.allclose(reference, scattered, rtol=1e-6, atol=0)


def test_scattering_shrinks_the_task_graph(client):
    import abtem
    from abtem.core import config

    potential, tp, probe, scan, sites = _small_eels_setup()

    def graph_size(transition_potentials):
        measurement = probe.transition_potential_scan(
            scan=scan, potential=potential, detectors=abtem.FlexibleAnnularDetector(),
            transition_potentials=transition_potentials, double_channel=False,
            sites=sites, max_batch=2, threshold=0.5, lazy=True,
        )
        # dask serializes graphs with cloudpickle, which handles the local
        # closures abTEM's ensemble machinery creates.
        return len(cloudpickle.dumps(measurement.array.__dask_graph__()))

    with config.set({"device": "cpu"}):
        embedded = graph_size(tp)
        scattered = graph_size(scatter_to_workers(tp, client=client))

    assert scattered < embedded - tp.array.nbytes / 2


def test_large_inputs_are_scattered_automatically(client):
    """The user passes the object; abTEM places it on the workers itself."""
    from abtem.core import config
    from abtem.core.backend import ScatteredInput, maybe_scatter_large_input

    big = np.ones((2048, 2048), dtype=np.complex64)  # 32 MB

    class Holder:
        def __init__(self, array):
            self.array = array
            self.metadata = {"Z": 5}

    handle = maybe_scatter_large_input(Holder(big))
    assert isinstance(handle, ScatteredInput)
    assert handle.metadata == {"Z": 5}

    # Small inputs are left alone.
    small = Holder(np.ones((8, 8), dtype=np.complex64))
    assert maybe_scatter_large_input(small) is small

    # And the behaviour is switchable.
    with config.set({"dask.scatter-large-inputs": False}):
        holder = Holder(big)
        assert maybe_scatter_large_input(holder) is holder



def test_scan_scatters_without_user_involvement(client):
    """Same call as always -- no handle, no helper, no new arguments."""
    from abtem.core import config
    from abtem.core.backend import _worker_inputs

    potential, tp, probe, scan, sites = _small_eels_setup()

    with config.set({"device": "cpu", "dask.scatter-large-inputs": "1 kB"}):
        before = len(_worker_inputs)
        result = _run_scan(probe, potential, tp, scan, sites)
        assert len(_worker_inputs) > before  # abTEM scattered it itself

    with config.set({"device": "cpu", "dask.scatter-large-inputs": False}):
        reference = _run_scan(probe, potential, tp, scan, sites)

    assert np.allclose(result, reference, rtol=1e-6, atol=0)
