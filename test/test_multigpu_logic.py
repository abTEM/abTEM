"""Logic tests for abTEM's multi-GPU support that need no real multi-GPU hardware.

Covers the client-suitability check, the memoized cluster lifecycle, and the
GPU execution-context resolution shared by ``_compute`` and the ``to_zarr``
save path -- via mocks / a threaded CPU cluster -- so the behaviour is
exercised on ordinary (single-GPU or CPU) CI.
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
    # ensure_cuda_cluster may pass memory_limit=..., so accept **kwargs
    m.LocalCUDACluster = type("LocalCUDACluster", (), {"__init__": lambda self, **kw: None})
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


def _fake_dask_cuda_capturing(captured):
    m = types.ModuleType("dask_cuda")

    def init(self, **kwargs):
        captured.update(kwargs)

    m.LocalCUDACluster = type("LocalCUDACluster", (), {"__init__": init})
    return m


def test_ensure_cuda_cluster_forwards_rmm_pool_and_devices(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)
    captured = {}
    monkeypatch.setitem(sys.modules, "dask_cuda", _fake_dask_cuda_capturing(captured))
    monkeypatch.setattr(
        "distributed.Client", lambda cluster: mock.Mock(status="running")
    )
    with abtem.config.set(
        {"dask.multi-gpu-rmm-pool": "20 GB", "dask.multi-gpu-devices": [0, 2]}
    ):
        backend.ensure_cuda_cluster()
    assert captured.get("rmm_pool_size") == "20 GB"
    assert captured.get("CUDA_VISIBLE_DEVICES") == "0,2"


def test_ensure_cuda_cluster_defaults_omit_optional_kwargs(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)
    captured = {}
    monkeypatch.setitem(sys.modules, "dask_cuda", _fake_dask_cuda_capturing(captured))
    monkeypatch.setattr(
        "distributed.Client", lambda cluster: mock.Mock(status="running")
    )
    backend.ensure_cuda_cluster()
    assert "rmm_pool_size" not in captured
    assert "CUDA_VISIBLE_DEVICES" not in captured


def test_ensure_cuda_cluster_explains_missing_main_guard(monkeypatch):
    """The cryptic multiprocessing spawn error is translated into guidance."""
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)

    def raise_bootstrapping(self, **kwargs):
        raise RuntimeError(
            "An attempt has been made to start a new process before the "
            "current process has finished its bootstrapping phase."
        )

    m = types.ModuleType("dask_cuda")
    m.LocalCUDACluster = type("LocalCUDACluster", (), {"__init__": raise_bootstrapping})
    monkeypatch.setitem(sys.modules, "dask_cuda", m)
    with pytest.raises(RuntimeError, match="__main__"):
        backend.ensure_cuda_cluster()


def test_ensure_cuda_cluster_reraises_other_runtime_errors(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)

    def raise_other(self, **kwargs):
        raise RuntimeError("something else entirely")

    m = types.ModuleType("dask_cuda")
    m.LocalCUDACluster = type("LocalCUDACluster", (), {"__init__": raise_other})
    monkeypatch.setitem(sys.modules, "dask_cuda", m)
    with pytest.raises(RuntimeError, match="something else entirely"):
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


# --------------------------------------------------------------------------
# _resolve_gpu_scheduler  (shared by ArrayObject.compute and the save paths)
# --------------------------------------------------------------------------


def _patch_no_cupy_check(monkeypatch):
    monkeypatch.setattr(abtem.array, "check_cupy_is_installed", lambda: None)


def test_resolve_gpu_scheduler_no_client_forces_synchronous(monkeypatch):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(distributed, "get_client", _no_client)
    with abtem.config.set({"dask.multi-gpu": False}):
        kwargs = abtem.array._resolve_gpu_scheduler({})
    assert kwargs.get("scheduler") == "synchronous"


def test_resolve_gpu_scheduler_cuda_client_left_in_charge(monkeypatch):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(
        distributed, "get_client", lambda *a, **k: mock.Mock(status="running")
    )
    monkeypatch.setattr(abtem.array, "is_gpu_dask_client", lambda c: True)
    kwargs = abtem.array._resolve_gpu_scheduler({})
    assert "scheduler" not in kwargs


def test_resolve_gpu_scheduler_explicit_scheduler_untouched(monkeypatch):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(distributed, "get_client", _no_client)
    started = []
    monkeypatch.setattr(abtem.array, "ensure_cuda_cluster", lambda: started.append(1))
    with abtem.config.set({"dask.multi-gpu": True}):
        kwargs = abtem.array._resolve_gpu_scheduler({"scheduler": "synchronous"})
    assert started == []  # user asked for a scheduler -> no cluster spun up
    assert kwargs["scheduler"] == "synchronous"


def test_resolve_gpu_scheduler_starts_cluster(monkeypatch):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(distributed, "get_client", _no_client)
    fake_client = mock.Mock(status="running")
    started = []
    monkeypatch.setattr(
        abtem.array,
        "ensure_cuda_cluster",
        lambda: (started.append(1), fake_client)[1],
    )
    monkeypatch.setattr(abtem.array, "is_gpu_dask_client", lambda c: c is fake_client)
    fake_cp = mock.Mock()
    fake_cp.cuda.runtime.getDeviceCount.return_value = 2
    monkeypatch.setattr(abtem.array, "cp", fake_cp)
    with abtem.config.set({"dask.multi-gpu": True}):
        kwargs = abtem.array._resolve_gpu_scheduler({})
    assert started == [1]
    assert "scheduler" not in kwargs  # the cluster client is left in charge


def test_resolve_gpu_scheduler_warns_when_single_gpu(monkeypatch):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(distributed, "get_client", _no_client)
    started = []
    monkeypatch.setattr(abtem.array, "ensure_cuda_cluster", lambda: started.append(1))
    fake_cp = mock.Mock()
    fake_cp.cuda.runtime.getDeviceCount.return_value = 1
    monkeypatch.setattr(abtem.array, "cp", fake_cp)
    with abtem.config.set({"dask.multi-gpu": True}):
        with pytest.warns(UserWarning, match="only one GPU"):
            kwargs = abtem.array._resolve_gpu_scheduler({})
    assert started == []  # declined; and no longer silently
    assert kwargs["scheduler"] == "synchronous"


def test_resolve_gpu_scheduler_warns_on_unsuitable_client(monkeypatch):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(
        distributed, "get_client", lambda *a, **k: mock.Mock(status="running")
    )
    monkeypatch.setattr(abtem.array, "is_gpu_dask_client", lambda c: False)
    with abtem.config.set({"dask.multi-gpu": True}):
        with pytest.warns(UserWarning, match="not a single-threaded"):
            kwargs = abtem.array._resolve_gpu_scheduler({})
    assert kwargs["scheduler"] == "synchronous"


# --------------------------------------------------------------------------
# get_cuda_cluster_client  (public, side-effect-free accessor)
# --------------------------------------------------------------------------


def test_get_cuda_cluster_client_returns_running(monkeypatch):
    running = mock.Mock(status="running")
    monkeypatch.setattr(backend, "_cuda_cluster_client", running)
    assert backend.get_cuda_cluster_client() is running


def test_get_cuda_cluster_client_none_when_absent_or_dead(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_cluster_client", None)
    assert backend.get_cuda_cluster_client() is None
    monkeypatch.setattr(backend, "_cuda_cluster_client", mock.Mock(status="closed"))
    assert backend.get_cuda_cluster_client() is None


# --------------------------------------------------------------------------
# to_zarr resolves the GPU execution context (the save path must distribute
# like .compute() does; previously it silently forced single-GPU synchronous
# and never consulted dask.multi-gpu)
# --------------------------------------------------------------------------


def _build_lazy_cpu():
    return abtem.Probe(energy=100e3, semiangle_cutoff=20, gpts=32, extent=5).build(
        lazy=True
    )


@pytest.mark.filterwarnings("ignore")
def test_to_zarr_starts_cluster_when_multigpu(monkeypatch, tmp_path):
    import distributed

    _patch_no_cupy_check(monkeypatch)
    monkeypatch.setattr(distributed, "get_client", _no_client)
    started = []
    monkeypatch.setattr(
        abtem.array,
        "ensure_cuda_cluster",
        lambda: (started.append(1), mock.Mock(status="running"))[1],
    )
    # Report the (mock) cluster unsuitable so the write still runs on the
    # local synchronous scheduler with the actually-CPU-backed test arrays.
    monkeypatch.setattr(abtem.array, "is_gpu_dask_client", lambda c: False)
    fake_cp = mock.Mock()
    fake_cp.cuda.runtime.getDeviceCount.return_value = 2
    monkeypatch.setattr(abtem.array, "cp", fake_cp)

    waves = _build_lazy_cpu()
    with abtem.config.set({"device": "gpu", "dask.multi-gpu": True}):
        waves.to_zarr(str(tmp_path / "waves.zarr"), progress_bar=False)
    assert started == [1]
    assert (tmp_path / "waves.zarr").exists()


def test_to_zarr_roundtrip_cpu(tmp_path):
    waves = _build_lazy_cpu()
    url = str(tmp_path / "waves.zarr")
    waves.to_zarr(url, progress_bar=False)
    loaded = abtem.from_zarr(url)
    import numpy as np

    np.testing.assert_allclose(
        loaded.compute().array, waves.compute().array, rtol=1e-6
    )


def test_to_zarr_compute_false_returns_delayed(tmp_path):
    from dask.delayed import Delayed

    waves = _build_lazy_cpu()
    delayed = waves.to_zarr(str(tmp_path / "waves.zarr"), compute=False)
    assert isinstance(delayed, Delayed)
    delayed.compute()
    assert (tmp_path / "waves.zarr").exists()
