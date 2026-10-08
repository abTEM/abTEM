"""Multi-output blocks are released promptly on a distributed cluster.

A scan with several detectors computes one packed block per multislice task (every
detector's output, exit waves included), and one extract task per detector pulls its
output out. The packed block is released only when all of its extracts have run. The
extracts of a block with several outputs carry a scheduler priority so that the
distributed scheduler runs them as soon as the block exists, instead of letting the
blocks pile up behind an extract that only feeds a final output. A block with a single
output is not annotated: it fuses with its extract and everything downstream, and the
priority would then reach the multislice work itself. dask's low-level fusion drops
annotations, so abTEM's compute keeps it off, and keeps annotated layers unfused, while
a distributed client runs an annotated graph.
"""

import collections
import warnings

import ase.build
import dask
import dask.array as da
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.array import (
    _EXTRACT_PRIORITY,
    ComputableList,
    _keep_annotations_guard,
)
from abtem.core.backend import asnumpy

distributed = pytest.importorskip("distributed")

FUSE_KEYS = ("optimization.fuse.active", "optimization.annotations.fuse")


def _layer(key):
    return (key[0] if isinstance(key, tuple) else key).rsplit("-", 1)[0]


class _Scheduler(distributed.diagnostics.plugin.SchedulerPlugin):
    """Scheduler-side record of the packed blocks held in memory at once, and of the
    priorities the scheduler assigned; runs no code in the tasks."""

    name = "packed-blocks"

    def __init__(self):
        self.peak = 0
        self.total = 0
        self.priorities = collections.defaultdict(set)

    async def start(self, scheduler):
        self.scheduler = scheduler

    def update_graph(self, scheduler, *args, **kwargs):
        for ts in scheduler.tasks.values():
            self.priorities[_layer(ts.key)].add(ts.priority[0])

    def transition(self, key, start, finish, *args, **kwargs):
        if finish == "memory" and _layer(key) == "apply_transform":
            self.total += 1
            held = sum(
                ts.state == "memory" and _layer(ts.key) == "apply_transform"
                for ts in self.scheduler.tasks.values()
            )
            self.peak = max(self.peak, held)


def _scan(device, detectors):
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    frozen_phonons = abtem.FrozenPhonons(
        atoms, num_configs=4, sigmas=0.08, seed=1, ensemble_mean=False
    )
    potential = abtem.Potential(
        frozen_phonons, gpts=64, slice_thickness=2, device=device
    )
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=20, device=device)
    scan = abtem.GridScan(
        (0, 0), (1, 1), gpts=(4, 4), fractional=True, potential=potential
    )
    return probe.scan(potential, scan=scan, detectors=detectors, max_batch=2)


def _haadf_and_waves(device):
    return _scan(device, [abtem.AnnularDetector(40, 90), abtem.WavesDetector()])


@pytest.fixture(params=["cpu", gpu])
def cluster(request):
    """A distributed cluster of single-threaded workers, the layout of abTEM's dask-cuda
    cluster; one worker on a GPU, since in-process workers share a CUDA context."""
    device = request.param
    n_workers = 2 if device == "cpu" else 1
    with (
        abtem.config.set({"device": device}),
        distributed.LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
        ) as local_cluster,
        distributed.Client(local_cluster) as client,
    ):
        plugin = _Scheduler()
        client.register_plugin(plugin)
        yield device, n_workers, client, local_cluster.scheduler.plugins[plugin.name]


def test_extract_priority_reaches_the_distributed_scheduler(cluster):
    device, _, _, plugin = cluster
    image, waves = _haadf_and_waves(device)

    ComputableList([image, waves]).compute(progress_bar=False)

    assert plugin.priorities["_extract_blockwise_multi_output"] == {-_EXTRACT_PRIORITY}
    assert plugin.priorities["apply_transform"] == {0}


def test_a_single_output_block_is_not_annotated(cluster):
    """With one detector, the multislice task fuses with its extract and everything
    downstream; a priority there would reach the multislice work itself."""
    device, _, _, plugin = cluster
    waves = _scan(device, abtem.WavesDetector())

    waves.diffraction_patterns(max_angle=60).mean(0).compute(progress_bar=False)

    assert plugin.priorities
    assert all(priorities == {0} for priorities in plugin.priorities.values())


def test_packed_blocks_are_released_as_they_are_extracted(cluster):
    """With a HAADF detector next to a WavesDetector, the distributed scheduler held
    15-20 of the 32 packed blocks at once (2 CPU workers, 6 runs) without the extract
    priority, and 3-4 with it. The bound is per worker: a block being extracted and one
    being computed."""
    device, n_workers, _, plugin = cluster

    def elastic_and_total():
        image, waves = _haadf_and_waves(device)
        patterns = waves.diffraction_patterns(max_angle=60, return_complex=True)
        return ComputableList(
            [
                image,
                patterns.intensity().mean(axis=0),
                patterns.mean(axis=0).intensity(),
            ]
        )

    computed = elastic_and_total().compute(progress_bar=False)

    assert plugin.total == 32
    assert plugin.peak <= 3 * n_workers
    # The reference is computed locally on purpose, next to the fixture's client;
    # dask up to 2025.3.0 warns about any local scheduler named while one is active.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Running on a single-machine scheduler", UserWarning
        )
        reference = elastic_and_total().compute(
            progress_bar=False, scheduler="synchronous"
        )
    for result, ref in zip(computed, reference):
        result, ref = asnumpy(result.array), asnumpy(ref.array)
        np.testing.assert_allclose(result, ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def _annotated_and_plain():
    x = da.ones((4, 4), chunks=2)
    with dask.annotate(priority=1):
        annotated = x.map_blocks(lambda block: block + 1, dtype=float)
    return annotated, x.map_blocks(lambda block: block + 1, dtype=float)


def _fusion_inside(arrays, kwargs, config=None):
    with dask.config.set(config or {}), _keep_annotations_guard(arrays, kwargs):
        return tuple(dask.config.get(key, None) for key in FUSE_KEYS)


def test_keep_annotations_guard_scope(cluster):
    """Fusion changes only while a distributed client runs an annotated graph, whichever
    way the client is named; a local scheduler, an unannotated graph and an explicit
    setting are left alone, and the configuration is restored afterwards."""
    _, _, client, _ = cluster
    annotated, plain = _annotated_and_plain()
    off, untouched = (False, False), (None, True)

    for kwargs in (
        {},
        {"scheduler": client},
        {"scheduler": "distributed"},
        {"scheduler": "dask.distributed"},
        {"scheduler": client.get},
    ):
        assert _fusion_inside([annotated], kwargs) == off

    assert _fusion_inside([annotated], {"scheduler": "synchronous"}) == untouched
    assert _fusion_inside([annotated], {}, {"scheduler": "threads"}) == untouched
    assert _fusion_inside([plain], {}) == untouched
    # An already computed item next to a lazy one plays no part.
    assert _fusion_inside([np.ones(3), annotated], {}) == off
    assert _fusion_inside([np.ones(3), plain], {}) == untouched
    explicit = {"optimization.fuse.active": True}
    assert _fusion_inside([annotated], {}, explicit) == (True, True)
    assert tuple(dask.config.get(key, None) for key in FUSE_KEYS) == untouched


def test_keep_annotations_guard_without_a_client():
    annotated, _ = _annotated_and_plain()

    assert _fusion_inside([annotated], {}) == (None, True)
    assert _fusion_inside([np.ones(3), annotated], {}) == (None, True)
    assert _fusion_inside([np.ones(3)], {}) == (None, True)


def test_a_list_mixing_computed_and_lazy_objects_computes():
    """A ComputableList may hold objects that are already computed next to lazy
    ones, including a multi-detector scan whose graph carries annotations."""
    potential = abtem.Potential(
        ase.build.mx2("WSe2", vacuum=2), gpts=64, slice_thickness=2, device="cpu"
    )

    def exit_wave():
        return abtem.PlaneWave(energy=60e3, device="cpu").multislice(potential)

    def haadf_and_waves():
        probe = abtem.Probe(energy=60e3, semiangle_cutoff=20, device="cpu")
        scan = abtem.GridScan(
            (0, 0), (1, 1), gpts=(2, 3), fractional=True, potential=potential
        )
        return probe.scan(
            potential,
            scan=scan,
            detectors=[abtem.AnnularDetector(40, 90), abtem.WavesDetector()],
        )

    computed = exit_wave().compute(progress_bar=False)
    haadf, waves = ComputableList(haadf_and_waves()).compute(
        progress_bar=False, scheduler="synchronous"
    )
    reference = [computed.array, computed.array, haadf.array, waves.array]

    for items in (
        [computed, exit_wave(), *haadf_and_waves()],
        [exit_wave(), computed, *haadf_and_waves()],
    ):
        results = ComputableList(items).compute(progress_bar=False)
        for result, ref in zip(results, reference):
            np.testing.assert_allclose(
                result.array, ref, rtol=0, atol=1e-6 * np.abs(ref).max()
            )
