"""Multi-output blocks are released promptly on a distributed cluster.

A scan with several detectors computes one packed block per multislice task (every
detector's output, exit waves included), and one extract task per detector pulls its
output out. The packed block is released only when all of its extracts have run. The
extracts carry a scheduler priority so that the distributed scheduler runs them as soon as
the block exists, instead of letting the blocks pile up behind an extract that only feeds a
final output. dask's low-level fusion drops annotations, so abTEM's compute keeps it off
while a distributed client runs the graph.
"""

import collections

import ase.build
import dask
import numpy as np
import pytest

import abtem
from abtem.array import (
    _EXTRACT_PRIORITY,
    ComputableList,
    _keep_annotations_guard,
)

distributed = pytest.importorskip("distributed")

N_WORKERS = 2


def _layer(key):
    return (key[0] if isinstance(key, tuple) else key).rsplit("-", 1)[0]


class _PackedBlocks(distributed.diagnostics.plugin.SchedulerPlugin):
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


def _haadf_and_waves_scan():
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    frozen_phonons = abtem.FrozenPhonons(
        atoms, num_configs=4, sigmas=0.08, seed=1, ensemble_mean=False
    )
    potential = abtem.Potential(frozen_phonons, gpts=64, slice_thickness=2)
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=20)
    scan = abtem.GridScan(
        (0, 0), (1, 1), gpts=(4, 4), fractional=True, potential=potential
    )
    return probe.scan(
        potential,
        scan=scan,
        detectors=[abtem.AnnularDetector(40, 90), abtem.WavesDetector()],
        max_batch=2,
    )


@pytest.fixture
def cluster_client():
    with distributed.LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=False,
        dashboard_address=":0",
    ) as cluster, distributed.Client(cluster) as client:
        plugin = _PackedBlocks()
        client.register_plugin(plugin)
        yield client, cluster.scheduler.plugins[plugin.name]


def test_extract_priority_reaches_the_distributed_scheduler(cluster_client):
    client, plugin = cluster_client
    image, waves = _haadf_and_waves_scan()

    ComputableList([image, waves]).compute(progress_bar=False)

    assert plugin.priorities["_extract_blockwise_multi_output"] == {-_EXTRACT_PRIORITY}
    assert plugin.priorities["apply_transform"] == {0}


def test_packed_blocks_are_released_as_they_are_extracted(cluster_client):
    """With a HAADF detector next to a WavesDetector, the distributed scheduler held 15-20
    of the 32 packed blocks at once (6 runs) before the extracts had a priority, and 3-4
    with it. The bound is per worker: a block being extracted and one being computed."""
    client, plugin = cluster_client

    def elastic_and_total():
        image, waves = _haadf_and_waves_scan()
        patterns = waves.diffraction_patterns(max_angle=60, return_complex=True)
        return ComputableList(
            [image, patterns.intensity().mean(axis=0), patterns.mean(axis=0).intensity()]
        )

    computed = elastic_and_total().compute(progress_bar=False)

    assert plugin.total == 32
    assert plugin.peak <= 3 * N_WORKERS
    reference = elastic_and_total().compute(progress_bar=False, scheduler="synchronous")
    for result, ref in zip(computed, reference):
        np.testing.assert_allclose(
            result.array, ref.array, rtol=0, atol=1e-5 * np.abs(ref.array).max()
        )


def test_keep_annotations_guard_scope(cluster_client):
    """Low-level fusion is switched off only while a distributed client runs the
    compute, and is left alone for a named scheduler or an explicit setting."""

    def fuse_inside(kwargs, config=None):
        with dask.config.set(config or {}), _keep_annotations_guard(kwargs):
            return dask.config.get("optimization.fuse.active", None)

    assert fuse_inside({}) is False
    assert fuse_inside({"scheduler": "synchronous"}) is None
    assert fuse_inside({}, {"optimization.fuse.active": True}) is True
    assert dask.config.get("optimization.fuse.active", None) is None


def test_keep_annotations_guard_without_a_client():
    with _keep_annotations_guard({}):
        assert dask.config.get("optimization.fuse.active", None) is None
