"""Logic tests for abTEM's multi-GPU support that need no real multi-GPU hardware.

Covers the client-suitability check, the memoized cluster lifecycle, and the
scheduler-selection branches of ``_compute`` -- via mocks / a threaded CPU
cluster -- so the behaviour is exercised on ordinary (single-GPU or CPU) CI.
"""

import sys
import types
from unittest import mock

import pytest

import abtem
import abtem.array
from abtem.core import backend
from abtem.core.backend import cp, is_gpu_dask_client


# --------------------------------------------------------------------------
# is_gpu_dask_client  (needs only distributed; no cupy / GPU)
# --------------------------------------------------------------------------


def test_is_gpu_dask_client_none():
    assert is_gpu_dask_client(None) is False


def test_is_gpu_dask_client_closed():
    assert is_gpu_dask_client(mock.Mock(status="closed")) is False


def test_is_gpu_dask_client_single_threaded_running():
    client = mock.Mock(status="running")
    client.nthreads.return_value = {"w1": 1, "w2": 1}
    assert is_gpu_dask_client(client) is True


def test_is_gpu_dask_client_rejects_multithreaded():
    client = mock.Mock(status="running")
    client.nthreads.return_value = {"w1": 2}
    assert is_gpu_dask_client(client) is False


def test_is_gpu_dask_client_rejects_empty():
    client = mock.Mock(status="running")
    client.nthreads.return_value = {}
    assert is_gpu_dask_client(client) is False


@pytest.mark.filterwarnings("ignore")
def test_is_gpu_dask_client_rejects_real_cpu_cluster():
    """A plain threaded LocalCluster (opened for the dashboard) is unsafe for CuPy."""
    from distributed import Client, LocalCluster

    with LocalCluster(
        processes=False, n_workers=1, threads_per_worker=2, dashboard_address=":0"
    ) as cluster:
        with Client(cluster) as client:
            assert is_gpu_dask_client(client) is False


# --------------------------------------------------------------------------
# ensure_cuda_cluster  (mocked dask_cuda; no GPU)
# --------------------------------------------------------------------------


def _fake_dask_cuda_module():
    m = types.ModuleType("dask_cuda")
    m.LocalCUDACluster = type("LocalCUDACluster", (), {})
    return m


def test_ensure_cuda_cluster_memoizes(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)
    monkeypatch.setitem(sys.modules, "dask_cuda", _fake_dask_cuda_module())
    created = []

    def fake_client(cluster):
        created.append(1)
        return mock.Mock(status="running")

    monkeypatch.setattr("distributed.Client", fake_client)

    c1 = backend.ensure_cuda_cluster()
    c2 = backend.ensure_cuda_cluster()
    assert c1 is c2
    assert len(created) == 1  # created once, then reused


def test_ensure_cuda_cluster_recreates_when_dead(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", mock.Mock(status="closed"))
    monkeypatch.setitem(sys.modules, "dask_cuda", _fake_dask_cuda_module())
    created = []

    def fake_client(cluster):
        created.append(1)
        return mock.Mock(status="running")

    monkeypatch.setattr("distributed.Client", fake_client)

    client = backend.ensure_cuda_cluster()
    assert client.status == "running"
    assert len(created) == 1  # stale/closed client discarded and rebuilt


def test_ensure_cuda_cluster_requires_dask_cuda(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)
    monkeypatch.setitem(sys.modules, "dask_cuda", None)  # force ImportError on import
    with pytest.raises(RuntimeError, match="dask-cuda"):
        backend.ensure_cuda_cluster()


# --------------------------------------------------------------------------
# _compute scheduler selection on the GPU path  (needs cupy to build gpu meta)
# --------------------------------------------------------------------------


def _no_client(*args, **kwargs):
    raise ValueError("no global client")


def _build_lazy_gpu():
    return abtem.Probe(energy=100e3, semiangle_cutoff=20, gpts=32, extent=5).build(lazy=True)


@pytest.mark.skipif(cp is None, reason="no gpu")
@pytest.mark.filterwarnings("ignore")
def test_gpu_no_client_forces_synchronous(monkeypatch):
    import distributed

    monkeypatch.setattr(distributed, "get_client", _no_client)
    captured = {}

    def fake_compute(*args, **kwargs):
        captured.update(kwargs)
        return ([None],)

    monkeypatch.setattr(abtem.array.dask, "compute", fake_compute)
    with abtem.config.set({"device": "gpu", "dask.multi-gpu": False}):
        _build_lazy_gpu().compute(progress_bar=False)
    assert captured.get("scheduler") == "synchronous"
    # the removed dead kwargs must not reappear
    assert "num_workers" not in captured and "threads_per_worker" not in captured


@pytest.mark.skipif(cp is None, reason="no gpu")
@pytest.mark.filterwarnings("ignore")
def test_multigpu_config_starts_cluster(monkeypatch):
    import distributed

    monkeypatch.setattr(distributed, "get_client", _no_client)
    started = []
    monkeypatch.setattr(
        abtem.array, "ensure_cuda_cluster",
        lambda: (started.append(1), mock.Mock(status="running"))[1],
    )
    monkeypatch.setattr(abtem.array, "is_gpu_dask_client", lambda c: c is not None)
    monkeypatch.setattr(abtem.array.cp.cuda.runtime, "getDeviceCount", lambda: 2)
    monkeypatch.setattr(abtem.array.dask, "compute", lambda *a, **k: ([None],))

    with abtem.config.set({"device": "gpu", "dask.multi-gpu": True}):
        _build_lazy_gpu().compute(progress_bar=False)
    assert started == [1]


@pytest.mark.skipif(cp is None, reason="no gpu")
@pytest.mark.filterwarnings("ignore")
def test_single_gpu_multigpu_config_no_cluster(monkeypatch):
    import distributed

    monkeypatch.setattr(distributed, "get_client", _no_client)
    started = []
    monkeypatch.setattr(abtem.array, "ensure_cuda_cluster", lambda: started.append(1))
    monkeypatch.setattr(abtem.array.cp.cuda.runtime, "getDeviceCount", lambda: 1)
    captured = {}

    def fake_compute(*args, **kwargs):
        captured.update(kwargs)
        return ([None],)

    monkeypatch.setattr(abtem.array.dask, "compute", fake_compute)
    with abtem.config.set({"device": "gpu", "dask.multi-gpu": True}):
        _build_lazy_gpu().compute(progress_bar=False)
    assert started == []  # only one GPU -> no cluster
    assert captured.get("scheduler") == "synchronous"


@pytest.mark.skipif(cp is None, reason="no gpu")
@pytest.mark.filterwarnings("ignore")
def test_explicit_scheduler_skips_cluster(monkeypatch):
    import distributed

    monkeypatch.setattr(distributed, "get_client", _no_client)
    started = []
    monkeypatch.setattr(abtem.array, "ensure_cuda_cluster", lambda: started.append(1))
    monkeypatch.setattr(abtem.array.cp.cuda.runtime, "getDeviceCount", lambda: 2)
    captured = {}

    def fake_compute(*args, **kwargs):
        captured.update(kwargs)
        return ([None],)

    monkeypatch.setattr(abtem.array.dask, "compute", fake_compute)
    with abtem.config.set({"device": "gpu", "dask.multi-gpu": True}):
        _build_lazy_gpu().compute(progress_bar=False, scheduler="synchronous")
    assert started == []  # user asked for a scheduler -> no cluster spun up
    assert captured.get("scheduler") == "synchronous"
