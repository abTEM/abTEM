"""Tests that the abTEM configuration reaches the workers of a distributed client."""

import numpy as np
import pytest

import abtem
from abtem.core import backend, config

distributed = pytest.importorskip("distributed")


def _worker_precision():
    from abtem.core import config

    return config.get("precision")


def _worker_config_snapshot():
    import copy

    from abtem.core.config import config as config_dict

    return copy.deepcopy(config_dict)


def _process_cluster(n_workers=1):
    return distributed.LocalCluster(
        n_workers=n_workers,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":0",
    )


def test_push_config_to_workers():
    backend._pushed_config_token = None
    with _process_cluster() as cluster, distributed.Client(cluster) as client:
        assert set(client.run(_worker_precision).values()) == {"float32"}

        with config.set({"precision": "float64"}):
            backend.push_config_to_workers(client)
            assert set(client.run(_worker_precision).values()) == {"float64"}

            # An unchanged configuration is not pushed again.
            token = backend._pushed_config_token
            backend.push_config_to_workers(client)
            assert backend._pushed_config_token is token


def test_config_plugin_does_not_overwrite_the_clients_own_config():
    # In-process workers (LocalCluster(processes=False)) share the client's
    # config dict. A worker that joins later runs the plugin's setup, which must
    # not replay the old snapshot over configuration changed after the push.
    import copy

    from abtem.core.config import config as config_dict

    snapshot = copy.deepcopy(config_dict)
    snapshot["precision"] = "float64"
    plugin = backend._make_config_plugin(snapshot)

    with config.set({"precision": "float32"}):
        plugin.setup()
        assert config.get("precision") == "float32"

        # A worker in another process still receives the snapshot.
        plugin._origin_pid = -1
        plugin.setup()
        assert config.get("precision") == "float64"


def test_workers_see_identical_config():
    from abtem.core.config import config as client_config_dict

    backend._pushed_config_token = None
    with _process_cluster() as cluster, distributed.Client(cluster) as client:
        with config.set({"precision": "float64", "fftw.threads": 3}):
            backend.push_config_to_workers(client)

            # Every worker must hold a config identical to the client's merged
            # configuration -- not just the keys we changed.
            for snapshot in client.run(_worker_config_snapshot).values():
                assert snapshot == client_config_dict

            # The snapshot travels as a named worker plugin, so workers that
            # join later (or are restarted by a Nanny) also receive it.
            cluster.scale(2)
            client.wait_for_workers(2)
            snapshots = client.run(_worker_config_snapshot).values()
            assert len(snapshots) == 2
            for snapshot in snapshots:
                assert snapshot == client_config_dict

            # A subsequent change must invalidate the push token and
            # propagate on the next dispatch.
            with config.set({"precision": "float32"}):
                backend.push_config_to_workers(client)
                for snapshot in client.run(_worker_config_snapshot).values():
                    assert snapshot == client_config_dict
                    assert snapshot["precision"] == "float32"


def test_distributed_compute_pushes_config():
    backend._pushed_config_token = None
    with _process_cluster() as cluster, distributed.Client(cluster) as client:
        with config.set({"precision": "float64"}):
            waves = abtem.PlaneWave(energy=100e3, gpts=32, extent=5).build(lazy=True)
            assert waves.array.dtype == np.complex128

            computed = waves.compute(progress_bar=False)

            assert computed.array.dtype == np.complex128
            assert set(client.run(_worker_precision).values()) == {"float64"}


def test_distributed_to_zarr_pushes_config(tmp_path):
    backend._pushed_config_token = None
    with _process_cluster() as cluster, distributed.Client(cluster) as client:
        with config.set({"precision": "float64"}):
            waves = abtem.PlaneWave(energy=100e3, gpts=32, extent=5).build(lazy=True)
            waves.to_zarr(str(tmp_path / "waves.zarr"), progress_bar=False)

            loaded = abtem.from_zarr(str(tmp_path / "waves.zarr")).compute()

            assert loaded.array.dtype == np.complex128
