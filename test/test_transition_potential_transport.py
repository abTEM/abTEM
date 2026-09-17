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


def _synthetic_tp(gpts=(64, 64), extent=(8.0, 8.0), energy=60e3, n_transitions=4,
                  seed=0):
    return synthetic_transition_potential(
        Z=5, gpts=gpts, extent=extent, energy=energy,
        n_transitions=n_transitions, seed=seed,
    )


def _setup(gpts=(64, 64)):
    import ase

    atoms = ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True
    )
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=2.0)
    # Matched to the probe: a built TransitionPotentialArray whose energy
    # disagrees with the waves is refused outright by the driver's guard
    # (transition_potential_multislice_and_detect, abtem/multislice.py), so a
    # shared fixture used by tests that do not care about that guard must not
    # trip it. The two tests that specifically exercise the guard's
    # private-view isolation build their own mismatched tp instead.
    tp = _synthetic_tp(gpts=gpts, extent=potential.extent, energy=60e3)
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
    """The same graph under concurrent threads must match the synchronous run
    bit for bit.

    Note what this does and does not catch: the grid/accelerator matching is
    idempotent, so concurrent tasks racing on a shared transition potential
    converge to the same values and the numbers alone cannot see it. The
    assertions that the caller's object is left unmatched are what actually
    guard that race (see the two tests below); this one guards the broader
    invariant that concurrency does not perturb the result.
    """
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

    # Not bit-for-bit: the default FFTW backend plans with FFTW_MEASURE, which
    # picks different algorithms in differently loaded processes, so results
    # that cross a process boundary agree only to float32 round-off. The
    # same-process comparisons above do assert exact equality.
    assert np.allclose(on_cluster, local, rtol=1e-5, atol=0)


def test_prism_lazy_and_eager_agree():
    """Uses its own energy-matched tp rather than `_setup()`'s: that fixture
    deliberately mismatches energy to exercise the multislice driver's
    private-view matching (see `_setup`'s own comment), which
    `_prism_eels_common_setup` now refuses outright for a built transition
    potential (abtem/inelastic/core_loss.py) -- correctly, since its array
    was never computed for any other energy. Lazy/eager agreement is
    orthogonal to that; a mismatch here would only make both branches raise
    identically, testing nothing."""
    potential, tp, _, scan, sites = _setup()
    tp = _synthetic_tp(gpts=potential.gpts, extent=potential.extent)  # energy=60e3
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


def test_scan_refuses_a_mismatch_rather_than_rematching_in_place():
    """The end-to-end guard that actually bites: a transition potential
    carrying a different energy from the probe used to be silently re-matched
    -- on a private view, never on the caller's object, but silently all the
    same. transition_potential_multislice_and_detect (abtem/multislice.py)
    now refuses it outright, before _task_local's match ever runs, the same
    guard and the same reason as the PRISM-EELS driver's own
    accelerator.check_match. Concurrency is part of what needs guarding: a
    refusal only some threads see, or that mutates the caller before raising,
    would be worse than no guard."""
    potential, _, probe, scan, sites = _setup()
    tp = _synthetic_tp(gpts=potential.gpts, extent=potential.extent, energy=80e3)
    assert tp.energy == 80e3  # fixture precondition: a genuine mismatch

    with pytest.raises(RuntimeError, match="Inconsistent energies"):
        _scan(probe, potential, tp, scan, sites).compute(
            progress_bar=False, scheduler="threads", num_workers=8
        )

    assert tp.energy == 80e3, "the refused scan still mutated the caller's object"


def test_scan_leaves_the_users_object_unmutated_even_when_refused():
    """End to end: a refused scan raises before touching anything, so the
    user's transition potential keeps exactly the state it was built with --
    stronger than just its energy, since _task_local's private view shares
    the payload and only privatises grid and accelerator."""
    potential, _, probe, scan, sites = _setup()
    tp = _synthetic_tp(gpts=potential.gpts, extent=potential.extent, energy=80e3)
    energy_before = tp.energy
    extent_before = tp.extent
    array_before = tp.array
    assert energy_before == 80e3  # the guard has to actually have something to refuse

    with pytest.raises(RuntimeError, match="Inconsistent energies"):
        _scan(probe, potential, tp, scan, sites).compute(progress_bar=False)

    assert tp.energy == energy_before
    assert tp.extent == extent_before
    assert tp.array is array_before


def test_prism_threaded_and_synchronous_schedulers_refuse_a_mismatch_alike():
    """The PRISM-EELS driver shares the materialized transition potential
    through its own mechanism (a delayed ``map_blocks`` kwarg), so its
    guards need race coverage of their own.

    Concurrency requires more than one task, and a plain potential gives
    PRISM exactly one block -- hence the frozen-phonon ensemble, which puts
    one block per configuration.

    This used to assert the opposite: that a built, energy-mismatched
    transition potential's accelerator was silently matched on a private
    view, leaving the caller's object untouched, under both schedulers.
    ``_prism_eels_common_setup`` now refuses that case outright
    (abtem/inelastic/core_loss.py) -- correctly, since a built array's form
    factors were computed for the transition potential's own energy and
    cannot be silently reinterpreted as another one's. What still needs race
    coverage is that every concurrent task hits the same refusal: a refusal
    that only some threads see, or that corrupts partial state before
    raising, would be worse than no guard at all.
    """
    import ase

    atoms = ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True
    )
    phonons = abtem.FrozenPhonons(atoms, num_configs=4, sigmas=0.05, seed=11)
    potential = abtem.Potential(phonons, gpts=(64, 64), slice_thickness=2.0)
    # Energy differs from the S-matrix: every task's guard must fire.
    tp = _synthetic_tp(gpts=(64, 64), extent=potential.extent, energy=80e3)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(4, 4), fractional=True, potential=potential
    )
    s_matrix = abtem.SMatrix(
        potential=potential, energy=60e3, semiangle_cutoff=32, interpolation=1
    )

    lazy = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=scan,
        detectors=abtem.FlexibleAnnularDetector(), sites=None,
        double_channel=False, lazy=True,
    )
    assert len(lazy.array.__dask_graph__()) > 1  # the test must have concurrency

    with pytest.raises(RuntimeError, match="Inconsistent energies"):
        lazy.copy().compute(progress_bar=False, scheduler="threads", num_workers=8)
    with pytest.raises(RuntimeError, match="Inconsistent energies"):
        lazy.compute(progress_bar=False, scheduler="synchronous")

    assert tp.energy == 80e3, "the refused scan still mutated the caller's object"


def test_prism_graph_carries_the_transition_potential_once():
    """The PRISM path's transport needs its own assertion: without the
    delayed wrapper dask would embed one copy per ensemble block, and the
    result-only tests above would not notice."""
    import ase

    atoms = ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True
    )
    phonons = abtem.FrozenPhonons(atoms, num_configs=6, sigmas=0.05, seed=3)
    potential = abtem.Potential(phonons, gpts=(64, 64), slice_thickness=2.0)
    tp = _synthetic_tp(extent=potential.extent, energy=80e3)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(4, 4), fractional=True, potential=potential
    )
    s_matrix = abtem.SMatrix(
        potential=potential, energy=60e3, semiangle_cutoff=32, interpolation=1
    )

    lazy = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=scan,
        detectors=abtem.FlexibleAnnularDetector(), sites=None,
        double_channel=False, lazy=True,
    )
    graph = dict(lazy.array.__dask_graph__())
    payload = tp.array.nbytes
    sizes = [len(cloudpickle.dumps(value)) for value in graph.values()]

    assert len(graph) > 6  # one block per configuration, plus structure
    big = [size for size in sizes if size > payload / 2]
    assert len(big) == 1, f"expected one payload-sized key, got {len(big)}"


def test_reconstructor_without_its_graph_node_args_fails_clearly():
    """Calling the partial from _from_partitioned_args with only the
    potential's args used to die inside the potential's own reconstructor,
    naming a function the caller never invoked."""
    import ase

    from abtem.multislice import (
        MultisliceTransform,
        transition_potential_multislice_and_detect,
    )

    atoms = ase.Atoms("BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)],
                      cell=(8, 8, 4), pbc=True)
    potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
    tp = _synthetic_tp(extent=potential.extent, energy=80e3)
    transform = MultisliceTransform(
        potential=potential, detectors=abtem.FlexibleAnnularDetector(),
        multislice_func=transition_potential_multislice_and_detect,
        transition_potential=tp, threshold=1.0,
    )

    assert transform._graph_node_keys() == ("transition_potential",)
    potential_args = potential._partition_args(lazy=False)
    with pytest.raises(ValueError, match="partitioned arguments"):
        transform._from_partitioned_args()(*potential_args)

    # With the full set from _partition_args it round-trips.
    rebuilt = transform._from_partitioned_args()(
        *transform._partition_args(lazy=False)
    ).item()
    assert rebuilt._multislice_func_kwargs["transition_potential"] is tp


def test_prism_matches_the_grid_before_building():
    """`build()` evaluates form factors on `self.gpts`, so an unbuilt
    transition potential must be matched to the waves first.

    The multislice driver and `TransitionPotential.scatter` both do that;
    `_prism_eels_common_setup` used to build first, so PRISM rejected an
    unbuilt transition potential that multislice accepts. Uses a stub rather
    than real `SubshellTransitions` so it runs without GPAW.
    """
    import ase

    from abtem.inelastic.core_loss import TransitionPotential

    class _RecordingTransitionPotential(TransitionPotential):
        """Records the grid it saw when build() was called."""

        def __init__(self, **kwargs):
            # A list, not a scalar: the driver calls build() on the private
            # view from _task_local (a shallow copy), which shares this
            # object by reference, so the original still sees the record.
            self.builds = []
            super().__init__(Z=5, transitions=(), **kwargs)

        def __len__(self):
            return 4

        def build(self):
            self.builds.append(self.gpts)
            assert self.gpts is not None, "build() called before the grid was matched"
            return _synthetic_tp(gpts=self.gpts, extent=self.extent, energy=self.energy)

    atoms = ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True
    )
    potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
    )
    s_matrix = abtem.SMatrix(
        potential=potential, energy=60e3, semiangle_cutoff=32, interpolation=1
    )

    unbuilt = _RecordingTransitionPotential()  # no extent, no gpts, no energy
    assert unbuilt.gpts is None

    s_matrix.transition_potential_scan(
        transition_potentials=unbuilt, scan=scan,
        detectors=abtem.FlexibleAnnularDetector(), sites=None,
        double_channel=False, lazy=False,
    )

    assert unbuilt.builds == [potential.gpts]
