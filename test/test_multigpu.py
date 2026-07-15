"""Multi-GPU integration tests for abTEM's dask-cuda support.

Skipped unless the machine has >=2 GPUs *and* dask-cuda installed. Each test
runs a real ``dask_cuda.LocalCUDACluster`` (one worker per GPU) via abTEM's own
``ensure_cuda_cluster`` and checks that GPU computations actually distribute and
that the result does not depend on how many GPUs were used.
"""

import numpy as np
import pytest
from utils import requires_multigpu

import abtem
from abtem.core.backend import asnumpy, is_gpu_dask_client

# All tests here need multiple GPUs; ignore the noisy dask/cupy runtime warnings
# (the suite otherwise runs with filterwarnings=error).
pytestmark = [pytest.mark.multigpu, requires_multigpu, pytest.mark.filterwarnings("ignore")]


# --------------------------------------------------------------------------
# deterministic workloads (fixed seed -> identical frozen-phonon configs)
# --------------------------------------------------------------------------


def _atoms():
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 3)


def _exit_waves(n_configs):
    """Exit waves keep the frozen-phonon axis: n_configs independent dask blocks."""
    frozen_phonons = abtem.FrozenPhonons(_atoms(), num_configs=n_configs, sigmas=0.1, seed=1)
    potential = abtem.Potential(frozen_phonons, sampling=0.1)
    return abtem.PlaneWave(energy=100e3).multislice(potential)


def _haadf(n_configs=4):
    frozen_phonons = abtem.FrozenPhonons(_atoms(), num_configs=n_configs, sigmas=0.1, seed=1)
    potential = abtem.Potential(frozen_phonons, sampling=0.1)
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    scan = abtem.GridScan(start=(0, 0), end=(2.715, 2.715), sampling=0.3)
    return probe.scan(potential, scan=scan, detectors=abtem.AnnularDetector(inner=40, outer=100))


def _worker_gpu_pci():
    """PCI address of a worker's pinned GPU (unique per physical device)."""
    import cupy

    p = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    return (p["pciDomainID"], p["pciBusID"], p["pciDeviceID"])


@pytest.fixture(scope="module")
def cuda_client():
    """One dask-cuda cluster spanning all GPUs, built via abTEM's ensure_cuda_cluster."""
    from abtem.core import backend

    backend._cuda_cluster_client = None
    client = backend.ensure_cuda_cluster()
    yield client
    cluster = getattr(client, "cluster", None)
    client.close()
    if cluster is not None:
        cluster.close()
    backend._cuda_cluster_client = None


# --------------------------------------------------------------------------
# cluster properties
# --------------------------------------------------------------------------


def test_cluster_is_gpu_appropriate(cuda_client):
    assert is_gpu_dask_client(cuda_client)
    nthreads = cuda_client.nthreads()
    assert len(nthreads) >= 2
    assert all(t == 1 for t in nthreads.values())


def test_workers_pinned_to_distinct_gpus(cuda_client):
    pcis = cuda_client.run(_worker_gpu_pci)
    assert len(set(pcis.values())) == len(cuda_client.nthreads())


def test_ensure_cuda_cluster_is_memoized(cuda_client):
    from abtem.core import backend

    assert backend.ensure_cuda_cluster() is cuda_client
    assert backend._cuda_cluster_client is cuda_client


# --------------------------------------------------------------------------
# distribution + correctness
# --------------------------------------------------------------------------


def test_gpu_compute_distributes_over_all_workers(cuda_client):
    from distributed import get_task_stream

    n = len(cuda_client.nthreads())
    with abtem.config.set({"device": "gpu"}):
        waves = _exit_waves(n_configs=2 * n)
        with get_task_stream(cuda_client) as ts:
            result = waves.compute(progress_bar=False)

    arr = asnumpy(result.array)
    assert arr.shape[0] == 2 * n
    assert np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag))

    workers = {r.get("worker") for r in ts.data if isinstance(r, dict) and r.get("worker")}
    assert len(workers) == n, f"work ran on {len(workers)}/{n} GPU workers"


def test_haadf_image_independent_of_gpu_distribution(cuda_client):
    """The HAADF image must be the same computed on one device or across all GPUs."""
    with abtem.config.set({"device": "gpu"}):
        # single device (synchronous scheduler bypasses the cluster)
        reference = _haadf(n_configs=4).compute(progress_bar=False, scheduler="synchronous")
        # distributed across every GPU via the active cluster
        distributed_result = _haadf(n_configs=4).compute(progress_bar=False)

    np.testing.assert_allclose(
        asnumpy(distributed_result.array),
        asnumpy(reference.array),
        rtol=1e-4,
        atol=1e-8,
    )


def test_automatic_multigpu_config_distributes(cuda_client):
    """config['dask.multi-gpu'] with the active cluster still distributes correctly."""
    from distributed import get_task_stream

    n = len(cuda_client.nthreads())
    with abtem.config.set({"device": "gpu", "dask.multi-gpu": True}):
        waves = _exit_waves(n_configs=2 * n)
        with get_task_stream(cuda_client) as ts:
            waves.compute(progress_bar=False)

    workers = {r.get("worker") for r in ts.data if isinstance(r, dict) and r.get("worker")}
    assert len(workers) == n
