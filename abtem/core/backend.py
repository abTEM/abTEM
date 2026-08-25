"""Module for handling the array backend (NumPy, CuPy, Dask, etc.) of the library."""

from __future__ import annotations

import logging
import warnings
from numbers import Number
from types import ModuleType
from typing import Union

import dask.array as da
import numpy as np
import scipy  # type: ignore
import scipy.ndimage  # type: ignore

from abtem.core.config import config
from abtem.core.config import get as _config_get

try:
    import cupy as cp  # type: ignore
except ModuleNotFoundError:
    cp = None
except ImportError:
    if config.get("device") == "gpu":
        warnings.warn(
            "The CuPy library could not be imported. Please check your installation, or"
            " change your configuration to use CPU."
        )
    cp = None


try:
    import cupyx  # type: ignore
except ImportError:
    assert cp is None
    cupyx = None


try:
    import cupyx.scipy.ndimage as cupyx_ndimage  # type: ignore
except ImportError:
    assert cupyx is None
    cupyx_ndimage = None


ArrayModule = Union[ModuleType, str]

logger = logging.getLogger(__name__)


def check_cupy_is_installed():
    """
    Check if CuPy is installed, raise an error if not.
    """
    if cp is None:
        raise RuntimeError("CuPy is not installed, GPU calculations disabled")


_cuda_cluster_client = None


def ensure_cuda_cluster():
    """
    Start a dask-cuda cluster spanning all visible GPUs and return its client.

    The cluster assigns one worker process to each GPU, allowing dask to distribute
    computations across all of them. It is created once per process and reused on
    subsequent calls. Requires the optional dask-cuda package.

    Returns
    -------
    distributed.Client
        The client connected to the dask-cuda cluster.
    """
    global _cuda_cluster_client

    if _cuda_cluster_client is not None:
        if getattr(_cuda_cluster_client, "status", None) == "running":
            return _cuda_cluster_client
        # The previous cluster was shut down; discard it and start a new one.
        _cuda_cluster_client = None

    try:
        from dask_cuda import LocalCUDACluster  # type: ignore
    except ImportError:
        raise RuntimeError(
            "The dask-cuda package is required to distribute computations across "
            "multiple GPUs. Please install it (see "
            "https://docs.rapids.ai/api/dask-cuda/stable/install/), or set the "
            "configuration option 'dask.multi-gpu' to false."
        )

    from distributed import Client

    # Cap each worker's memory to a share of the cgroup/job limit so a
    # memory-constrained allocation (e.g. a Slurm cgroup) spills instead of being
    # OOM-killed. dask-cuda otherwise sizes workers from the node total, which
    # over-commits when the cgroup grants less than the node has.
    cluster_kwargs: dict = {}
    try:
        from distributed.system import MEMORY_LIMIT

        n_gpus = cp.cuda.runtime.getDeviceCount() if cp is not None else 0
        if n_gpus > 0:
            cluster_kwargs["memory_limit"] = int(0.85 * MEMORY_LIMIT / n_gpus)
    except Exception:  # noqa: BLE001 -- fall back to dask-cuda's default sizing
        pass

    # Optional RMM memory pool per worker (e.g. "20 GB"), forwarded to
    # dask-cuda; a pre-grown pool avoids allocator churn on memory-intensive
    # workloads.
    rmm_pool = _config_get("dask.multi-gpu-rmm-pool", None)
    if rmm_pool:
        cluster_kwargs["rmm_pool_size"] = rmm_pool

    # Optional subset of GPUs to span, as a list of device indices or a
    # comma-separated string. By default the cluster spans all visible GPUs.
    devices = _config_get("dask.multi-gpu-devices", None)
    if devices:
        if not isinstance(devices, str):
            devices = ",".join(str(d) for d in devices)
        cluster_kwargs["CUDA_VISIBLE_DEVICES"] = devices

    try:
        _cuda_cluster_client = Client(LocalCUDACluster(**cluster_kwargs))
    except RuntimeError as exc:
        # dask-cuda starts worker processes with the 'spawn' method (fork is
        # unsafe with a live CUDA context), so the workers re-import the main
        # module. Without an entry-point guard that re-runs the script in every
        # worker, which multiprocessing reports with a cryptic bootstrapping
        # error (typically alongside a port-8787-in-use complaint).
        if "bootstrapping phase" in str(exc):
            raise RuntimeError(
                "Starting the multi-GPU cluster failed because the worker "
                "processes re-imported the main module before it finished "
                "executing. dask-cuda starts workers with the 'spawn' method, "
                "so the script's entry point must be guarded with "
                "'if __name__ == \"__main__\":'."
            ) from exc
        raise

    logger.info(
        "dask.multi-gpu: started a dask-cuda LocalCUDACluster; computations "
        "will be distributed with one worker per visible GPU."
    )

    return _cuda_cluster_client


def get_cuda_cluster_client():
    """
    Return the dask-cuda cluster client started by ``ensure_cuda_cluster``.

    Returns the running client, or None when no cluster has been started or the
    previous one was shut down. Unlike ``ensure_cuda_cluster`` this never starts
    a cluster, which makes it suitable for inspecting whether multi-GPU
    execution is active (e.g. from benchmark or verification scripts).

    Returns
    -------
    distributed.Client or None
        The client connected to the running dask-cuda cluster, if any.
    """
    if (
        _cuda_cluster_client is not None
        and getattr(_cuda_cluster_client, "status", None) == "running"
    ):
        return _cuda_cluster_client
    return None


_pushed_config_token = None


def _apply_config_snapshot(snapshot):
    from abtem.core.config import config as config_dict

    config_dict.clear()
    config_dict.update(snapshot)


def push_config_to_workers(client):
    """Mirror this process's abTEM configuration onto the client's workers.

    abTEM resolves configuration inside tasks, in the worker process --
    ``get_dtype`` reads ``precision`` at call time, for example -- but worker
    processes only ever see the defaults: ``abtem.config.set`` in the client
    does not reach them, silently changing results (a float64 computation
    dispatched to default-configured workers runs in float32). Push a snapshot
    of the merged configuration before dispatching work; repeated pushes of an
    unchanged configuration are skipped.
    """
    global _pushed_config_token

    import copy

    snapshot = copy.deepcopy(config)
    token = (id(client), repr(snapshot))
    if token == _pushed_config_token:
        return
    client.run(_apply_config_snapshot, snapshot)
    _pushed_config_token = token


def is_gpu_dask_client(client) -> bool:
    """
    Check whether a distributed client can safely execute CuPy computations.

    Only a client whose workers are each single-threaded — as produced by
    ``dask_cuda.LocalCUDACluster``, which additionally pins one GPU per worker — is
    considered suitable. The threaded scheduler and multi-threaded workers share a
    single CUDA context per process, which cannot be used with CuPy.

    Parameters
    ----------
    client : distributed.Client or None
        The client to check.

    Returns
    -------
    bool
        True if the client is running and all of its workers are single-threaded.
    """
    if client is None:
        return False

    if getattr(client, "status", None) != "running":
        return False

    try:
        nthreads = client.nthreads()
    except Exception:
        return False

    return len(nthreads) > 0 and all(n == 1 for n in nthreads.values())


def validate_device(device: str | None = None) -> str:
    """
    Validate the device string.

    Parameters
    ----------
    device : str, None
        The device string to validate. Must be either 'cpu' or 'gpu'. If None, the
        device from the configuration is used.

    Returns
    -------
    str
        The validated device string.
    """

    if device is None:
        device = config.get("device")
        assert isinstance(device, str)
        return device

    return device


def get_array_module(
    x: ModuleType | np.ndarray | da.core.Array | str | None = None,
) -> ModuleType:
    """
    Get the array module (NumPy or CuPy) for a given array or string.

    Parameters
    ----------
    x : numpy.ndarray, cupy.ndarray, dask.array.Array, str, None
        The array or string to get the array module for. If None, the default device is
        used.

    Returns
    -------
    numpy or cupy
        The array module.
    """

    if x is None:
        return get_array_module(config.get("device"))

    if isinstance(x, da.Array):
        return get_array_module(x._meta)

    if isinstance(x, str):
        if x.lower() in ("numpy", "cpu"):
            return np

        if x.lower() in ("cupy", "gpu"):
            check_cupy_is_installed()
            return cp

    if isinstance(x, np.ndarray):
        return np

    if x is np:
        return np

    if isinstance(x, Number):
        return np

    if cp is not None:
        if isinstance(x, cp.ndarray):
            return cp

        if x is cp:
            return cp

    raise ValueError(f"array module specification {x} not recognized")


def device_name_from_array_module(xp: ArrayModule) -> str:
    """
    Get the device string from the array module. The array module must be either NumPy
    or CuPy.

    Parameters
    ----------
    xp : numpy or cupy
        The array module.

    Returns
    -------
    str
        The device string.
    """
    if xp is np:
        return "cpu"

    if xp is cp:
        return "gpu"

    raise ValueError(f"array module must be NumPy or CuPy, not {xp}")


def get_scipy_module(x: ModuleType | np.ndarray | da.core.Array | str | None = None):
    """
    Get the SciPy module for a given array or device string.

    Parameters
    ----------
    x : numpy.ndarray, cupy.ndarray, dask.array.Array, str, None
        The array or string to get the SciPy module for. If None, the default device is
        used.

    Returns
    -------
    scipy or cupyx.scipy
        The SciPy module.
    """

    xp = get_array_module(x)

    if xp is np:
        return scipy

    elif xp is cp:
        return cupyx.scipy  # type: ignore

    else:
        raise ValueError(f"array module must be NumPy or CuPy, not {xp}")


def get_ndimage_module(
    x: ModuleType | np.ndarray | da.core.Array | str | None = None,
) -> ModuleType:
    """
    Get the ndimage module for a given array or device string.

    Parameters
    ----------
    x : numpy.ndarray, cupy.ndarray, dask.array.Array, str, None
        The array or string to get the ndimage module for. If None, the default device
        is used.

    Returns
    -------
    scipy.ndimage or cupyx.ndimage
        The ndimage module.
    """
    xp = get_array_module(x)

    if xp is np:
        return scipy.ndimage

    if xp is cp:
        return cupyx_ndimage  # type: ignore

    raise RuntimeError("Invalid array module")


def asnumpy(array: np.ndarray | da.Array):
    """
    Convert an array to NumPy.

    Parameters
    ----------
    array : numpy.ndarray, dask.array.Array
        The array to convert.

    Returns
    -------
    numpy.ndarray
        The array converted to NumPy.
    """
    if cp is None:
        return array

    if isinstance(array, da.core.Array):  # pyright: ignore[reportAttributeAccessIssue]
        return da.map_blocks(asnumpy, array)

    return cp.asnumpy(array)


def copy_to_device(
    array: np.ndarray | da.core.Array,
    device: ModuleType | np.ndarray | da.core.Array | str | None = None,
):
    """
    Copy an array to a different device (CPU or GPU) using CuPy.

    Parameters
    ----------
    array : numpy.ndarray
        The array to copy.
    device : str
        The device to copy to. Either 'cpu' or 'gpu'.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        The array copied to the specified device.
    """
    old_xp = get_array_module(array)
    new_xp = get_array_module(device)

    if old_xp is new_xp:
        return array

    if isinstance(array, da.core.Array):
        return da.map_blocks(
            copy_to_device,
            array,
            meta=new_xp.array((), dtype=array.dtype),
            device=device,
        )

    if new_xp is np:
        return cp.asnumpy(array)

    if new_xp is cp:
        return cp.asarray(array)

    raise RuntimeError("Invalid device specified")
