"""Tests for shared_constant_arg / _ScatterCandidate / _prescatter_shared_constants
(abtem/core/ensemble.py) and their hook into abtem/array.py's compute path.

Background: shared_constant_arg hands a large constant to dask via
dask.delayed(_wrap_with_array)(x, ndims=0) + da.from_delayed, which embeds x
as a literal value in the low-level task graph -- the client pickles and
sends it to the *scheduler* as part of graph submission, a different, slower
channel than client.scatter()'s worker-to-worker transfer. Traced directly
(this fix's own session) to confirm shared_constant_arg -- called from
MultisliceTransform._partition_args (multislice.py) and FrozenPhonons
(phonons.py) -- is the exact mechanism behind dask's own "large graph"
warning on a real core-loss scan.

An earlier version of this fix had shared_constant_arg decide whether to
scatter at graph-construction time. That doesn't work: abTEM builds its lazy
graphs (e.g. transition_potential_scan(lazy=True)) long before any client
exists -- abTEM's own multi-GPU cluster (ensure_cuda_cluster) only starts
inside _resolve_gpu_scheduler, at .compute() time. So shared_constant_arg now
always wraps x in _ScatterCandidate, and _prescatter_shared_constants -- called
from abtem/array.py's _compute() and the bulk-write path, right before their
dask.compute() calls, once the real client (if any) is finally known -- finds
and replaces any _ScatterCandidate still in the graph with a scattered Future,
in a COPY of the graph. It must be a copy: mutating a wrapper's own stored
.array would break test_graph_computes_on_a_distributed_cluster_and_survives_
client_loss (test_transition_potential_transport.py), which this file also
re-verifies is unaffected by re-running it here at the unit level.

Uses broadcast=False (the default), not True: measured directly on a real
4-GPU Perlmutter dask-cuda cluster that broadcast=True's wait-for-every-
worker-to-confirm serialized each worker's own first `import abtem` (~1.7 s/
worker, needed just to unpickle a TransitionPotentialArray) in front of
every run. Traded away deliberately: broadcast=False's placement on fewer
workers means losing the (possibly sole) worker holding a memoized payload
before another worker independently fetches a copy loses it permanently --
scattered data has no lineage to recompute from, unlike a normal task
result. test_losing_the_only_worker_with_broadcast_false_is_unrecoverable
below documents this tradeoff directly, alongside
test_broadcast_true_survives_losing_one_worker showing the same kill is
harmless under broadcast=True -- kept only as a reference/contrast, not
exercised by the shipped code path.
"""
import gc
import weakref

import dask.array as da
import numpy as np
import pytest
from ase import Atoms

distributed = pytest.importorskip("distributed")

from abtem.core.ensemble import (
    _prescatter_shared_constants,
    _ScatterCandidate,
    _wrap_with_array,
    shared_constant_arg,
)


@pytest.fixture
def cluster_client():
    with distributed.LocalCluster(
        n_workers=2, processes=True, threads_per_worker=1, dashboard_address=None,
    ) as cluster, distributed.Client(cluster) as client:
        yield client


def _payload():
    """Matches shared_constant_arg's real call sites (an ase.Atoms instance,
    e.g. FrozenPhonons.atoms) -- not a bare ndarray: _wrap_with_array's
    np.zeros((), dtype=object) + itemset assignment tries to broadcast an
    array-shaped x into the 0-d wrapper instead of storing it as one opaque
    object, a pre-existing quirk unrelated to this fix (no real caller passes
    a bare ndarray)."""
    return Atoms("H2O", positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0)])


def test_no_client_returns_arrays_unchanged():
    arr = shared_constant_arg(_payload(), lazy=True)
    result = _prescatter_shared_constants([arr], None)
    assert result == [arr]


def test_client_replaces_the_literal_and_computes_correctly(cluster_client):
    x = _payload()
    arr = shared_constant_arg(x, lazy=True)
    [new_arr] = _prescatter_shared_constants([arr], cluster_client)

    assert new_arr is not arr
    assert new_arr.compute().item() == x


def test_original_array_graph_is_untouched(cluster_client):
    """The critical guard: prescattering must not mutate the array it was
    given, or a wrapper's own stored .array would lose the self-contained,
    literal-embedding graph a later recompute (e.g. after the client closes)
    depends on."""
    x = _payload()
    arr = shared_constant_arg(x, lazy=True)
    _prescatter_shared_constants([arr], cluster_client)

    # arr itself must still compute correctly with NO client at all.
    assert arr.compute().item() == x


def test_repeated_calls_reuse_the_same_future(cluster_client):
    x = _payload()
    arr_a = shared_constant_arg(x, lazy=True)
    arr_b = shared_constant_arg(x, lazy=True)  # a second, independent graph, same x

    [new_a] = _prescatter_shared_constants([arr_a], cluster_client)
    [new_b] = _prescatter_shared_constants([arr_b], cluster_client)

    future_a = next(iter(new_a.__dask_graph__().values())).args[0]
    future_b = next(
        v.args[0] for v in new_b.__dask_graph__().values() if getattr(v, "args", None)
    )
    assert future_a is future_b
    assert len(cluster_client._abtem_scatter_memo) == 1


def test_distinct_objects_get_distinct_futures(cluster_client):
    arr_a = shared_constant_arg(_payload(), lazy=True)
    arr_b = shared_constant_arg(_payload(), lazy=True)
    _prescatter_shared_constants([arr_a, arr_b], cluster_client)
    assert len(cluster_client._abtem_scatter_memo) == 2


def test_memo_entry_is_released_when_the_object_is_collected(cluster_client):
    def make():
        arr = shared_constant_arg(_payload(), lazy=True)
        _prescatter_shared_constants([arr], cluster_client)

    make()
    gc.collect()
    assert len(cluster_client._abtem_scatter_memo) == 0


def test_a_new_object_after_collection_gets_a_fresh_scatter_not_a_stale_one(
    cluster_client,
):
    """Guards the actual point of using a weakref instead of a bare id(x) key:
    id() can be reused after garbage collection, which would otherwise let a
    new, unrelated object silently reuse a stale Future for a different one."""
    first = Atoms("H", positions=[(0, 0, 0)])
    arr_first = shared_constant_arg(first, lazy=True)
    [new_first] = _prescatter_shared_constants([arr_first], cluster_client)
    future_first = next(iter(new_first.__dask_graph__().values())).args[0]
    del first, arr_first, new_first
    gc.collect()

    second = Atoms("He", positions=[(0, 0, 0)])
    arr_second = shared_constant_arg(second, lazy=True)
    [new_second] = _prescatter_shared_constants([arr_second], cluster_client)
    future_second = next(iter(new_second.__dask_graph__().values())).args[0]

    assert future_second is not future_first
    assert new_second.compute().item() == second


def test_non_weakrefable_object_still_scatters_without_crashing(cluster_client):
    x = (1, 2, 3)  # a bare tuple does not support weak references
    with pytest.raises(TypeError):
        weakref.ref(x)  # confirms the premise of this test on this Python

    arr = shared_constant_arg(x, lazy=True)
    [new_arr] = _prescatter_shared_constants([arr], cluster_client)
    assert new_arr.compute().item() == x


def test_scatter_places_data_on_at_least_one_worker(cluster_client):
    arr = shared_constant_arg(_payload(), lazy=True)
    [new_arr] = _prescatter_shared_constants([arr], cluster_client)
    future = next(iter(new_arr.__dask_graph__().values())).args[0]

    who_has = cluster_client.who_has(future)
    workers_holding_it = next(iter(who_has.values()))
    assert len(workers_holding_it) >= 1


def test_a_worker_without_the_data_fetches_it_from_a_peer(cluster_client):
    """The actual justification for broadcast=False: a worker that doesn't
    already have the scattered payload still gets it correctly when a task
    needs it there, via distributed's normal (always-on) dependency fetch --
    not something that can simply fail to happen."""
    x = _payload()
    arr = shared_constant_arg(x, lazy=True)
    [new_arr] = _prescatter_shared_constants([arr], cluster_client)
    future = next(iter(new_arr.__dask_graph__().values())).args[0]

    who_has_initially = cluster_client.who_has(future)
    initial_worker = next(iter(who_has_initially.values()))[0]
    other_workers = [
        w for w in cluster_client.scheduler_info()["workers"] if w != initial_worker
    ]
    assert other_workers, "test needs at least 2 workers"

    # Force the task computing new_arr onto a worker that does NOT already
    # have the data, to exercise the cross-worker fetch rather than the
    # trivial same-worker case.
    result = cluster_client.compute(
        new_arr, workers=other_workers[0], allow_other_workers=False
    ).result()
    assert result.item() == x


def _kill_worker_holding(client, address):
    """Abruptly close the Nanny for the worker at `address` -- not a graceful
    retire_workers(), which proactively migrates data away first and so
    would trivially "survive" any worker loss regardless of broadcast,
    proving nothing. This is meant to model an actual crash (a CUDA error,
    an OOM kill): the process is just gone, nothing gets a chance to move
    its data first."""
    cluster = client.cluster
    match = next(
        key for key, w in cluster.workers.items()
        if getattr(w, "worker_address", getattr(w, "address", None)) == address
    )
    cluster.sync(cluster.workers[match].close)


def test_losing_the_only_worker_with_broadcast_false_is_unrecoverable(cluster_client):
    """Documents the reliability tradeoff accepted by using broadcast=False:
    scattered data has no lineage, so if the single worker holding a
    memoized payload dies before another worker fetches its own copy, every
    later reuse of that memo entry fails -- there is nothing left to
    recompute from. Contrast with test_broadcast_true_survives_losing_one_
    worker below, which is not what the shipped code does."""
    x = _payload()
    arr = shared_constant_arg(x, lazy=True)
    [new_arr] = _prescatter_shared_constants([arr], cluster_client)
    future = next(iter(new_arr.__dask_graph__().values())).args[0]

    who_has = cluster_client.who_has(future)
    sole_worker = next(iter(who_has.values()))[0]
    _kill_worker_holding(cluster_client, sole_worker)

    with pytest.raises(Exception):
        new_arr.compute()


def test_broadcast_true_survives_losing_one_worker(cluster_client):
    """Reference/contrast only -- not the shipped code path (which always
    calls scatter() with the default broadcast=False). Shows the same kind
    of abrupt worker loss above is harmless when the payload was placed on
    every worker instead of just one."""
    x = _payload()
    future = cluster_client.scatter([x], broadcast=True)[0]

    who_has = cluster_client.who_has(future)
    a_worker = next(iter(who_has.values()))[0]
    _kill_worker_holding(cluster_client, a_worker)

    assert future.result() == x


def test_arrays_without_a_candidate_pass_through_unchanged(cluster_client):
    """Prescattering must be a no-op for ordinary dask arrays -- it should
    only ever touch shared_constant_arg's own _ScatterCandidate nodes."""
    arr = da.ones((4, 4), chunks=2)
    [result] = _prescatter_shared_constants([arr], cluster_client)
    assert result is arr


def test_shared_constant_arg_computes_to_the_same_value_without_prescattering():
    """_wrap_with_array must still unwrap a _ScatterCandidate that was never
    prescattered (no client was ever active) -- shared_constant_arg's lazy
    output must remain directly computable on its own, matching pre-fix
    behaviour exactly."""
    x = _payload()
    lazy_result = shared_constant_arg(x, lazy=True)
    computed = lazy_result.compute()
    eager_result = _wrap_with_array(x, ndims=0)
    assert computed.item() == eager_result.item() == x


def test_prescatter_removes_the_large_literal_from_the_graph(cluster_client):
    """The actual defect this guards: before prescattering, the graph carries
    x's bytes as a literal task argument; after, it must not."""
    import cloudpickle

    x = Atoms(2000 * "H", positions=np.random.rand(2000, 3) * 10)  # a large-ish payload
    arr = shared_constant_arg(x, lazy=True)

    before_sizes = [len(cloudpickle.dumps(v)) for v in dict(arr.__dask_graph__()).values()]
    assert max(before_sizes) > 50_000, "test payload isn't actually large -- fix the test"

    [new_arr] = _prescatter_shared_constants([arr], cluster_client)
    after_sizes = [len(cloudpickle.dumps(v)) for v in dict(new_arr.__dask_graph__()).values()]
    assert max(after_sizes) < 50_000, (
        "a task in the prescattered graph is still large -- the literal is "
        "still being embedded instead of scattered"
    )
