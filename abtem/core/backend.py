"""Module for handling the array backend (NumPy, CuPy, Dask, etc.) of the library."""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
import threading
import warnings
from numbers import Number
from types import ModuleType
from typing import Any, Union

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
    # this can fail even though cupyx itself imported, if the CUDA/ROCm
    # libraries it eagerly pulls in aren't on the loader path. GPU code that
    # needs cupyx_ndimage then fails at use time instead of blocking the abtem
    # import.
    cupyx_ndimage = None


def _is_macos() -> bool:
    """Whether this machine can have a Metal (MPS) device.

    Apple silicon and Intel Macs with an AMD GPU both can; whether this one
    does is for PyTorch to answer.
    """
    return sys.platform == "darwin"


def _preload_torch_openmp() -> bool:
    """Load PyTorch's own OpenMP runtime ahead of pyfftw's, without torch.

    PyTorch and pyfftw each ship a copy of libomp.dylib, and the order the two
    enter the process matters: if pyfftw's comes first, torch's work on the
    CPU later segfaults -- including the dtype cast it makes before uploading
    a double-precision host array to Metal, and its CPU fallback for large
    eigendecompositions. Importing all of torch up front would win that
    race, but costs about two seconds and 180 MB on every import of abTEM,
    whether or not Metal is ever used. Only torch's runtime has to be first,
    and loading that alone takes a few milliseconds -- after which pyfftw can
    load as usual and torch can wait until something asks for the 'mps'
    device. This module is imported before abtem.core.fft imports pyfftw.

    Returns
    -------
    bool
        Whether torch can safely be imported later. False only when it is too
        late: pyfftw's runtime is already in the process and torch's is not.
    """
    if not _is_macos():
        return True  # no Metal device to load torch for
    if "torch" in sys.modules:
        return True  # its runtime is already in, ahead of whatever follows
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return True
    if spec is None or not spec.submodule_search_locations:
        return True  # not installed; using Metal will say so
    runtime = os.path.join(spec.submodule_search_locations[0], "lib", "libomp.dylib")
    if not os.path.exists(runtime):
        return True  # no runtime of its own to clash with
    if "pyfftw" in sys.modules:
        return False
    try:
        ctypes.CDLL(runtime, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # importing torch will report whatever is wrong with it
    return True


_TORCH_IMPORT_IS_SAFE = _preload_torch_openmp()

# The Metal (MPS) array namespace and its array type, loaded on first use of
# the 'mps' device and None until then. Read them through the module
# (``backend.tp``) rather than binding them by value, so every caller sees the
# namespace once it is loaded. No Metal array can exist before then, so
# ``backend.tp is None`` correctly answers "not a Metal array" either way.
tp: Any = None
TorchNDArray: Any = None
_TORCH_LOAD_LOCK = threading.Lock()


ArrayModule = Union[ModuleType, str]

logger = logging.getLogger(__name__)


def check_cupy_is_installed():
    """
    Check if CuPy is installed, raise an error if not.
    """
    if cp is None:
        raise RuntimeError("CuPy is not installed, GPU calculations disabled")


def check_mps_is_available():
    """
    Load the Metal (MPS) array namespace, raising if it is unusable.

    PyTorch is imported here, on the first request for the 'mps' device,
    rather than with abTEM: see ``_preload_torch_openmp``.

    Returns
    -------
    module
        The Metal array namespace, as returned by ``get_array_module('mps')``.
    """
    _load_torch()

    if config.get("precision") != "float32":
        raise RuntimeError(
            "Metal (MPS) is a single-precision backend, but the configured "
            f"precision is '{config.get('precision')}'. Set "
            "abtem.config.set({'precision': 'float32'}), or run on the 'cpu' or "
            "'gpu' device for double precision."
        )

    return tp


_METAL_DEVICE_NAMES = frozenset({"mps", "metal", "torch"})
_kernels_checked = False


def _check_kernels_once() -> None:
    global _kernels_checked
    if not _kernels_checked:
        check_metal_kernels_are_available()
        _kernels_checked = True


def emulates_float64() -> bool:
    """
    Whether float64 work runs on the Metal GPU in emulated double precision.

    That is the case on the device 'mps' with precision 'float64', which Metal
    cannot hold in arrays and so gets from :mod:`abtem.core._double_float`: the
    data stay float64 on the host and the heavy loops run in double-float on the
    GPU. The configuration key ``torch.emulate-float64`` asks for the same with
    the device 'cpu'. Neither does anything in single precision.
    """
    if config.get("precision") != "float64":
        return False
    return bool(_config_get("torch.emulate-float64")) or (
        str(config.get("device", "")).lower() in _METAL_DEVICE_NAMES
    )


def check_metal_kernels_are_available():
    """
    Load PyTorch for abTEM's own Metal kernels, raising if they cannot run.

    The emulated double precision of ``torch.emulate-float64``
    (:mod:`abtem.core._double_float`) runs float64 work on the GPU, so unlike
    :func:`check_mps_is_available` it takes any precision; it needs the real
    Metal device and a PyTorch that compiles Metal shaders. Loading goes
    through the same guards: importing PyTorch any other way would skip the
    refusal to load it after pyfftw, after which torch's CPU work crashes the
    process.

    Returns
    -------
    module
        ``abtem.core._torch``.
    """
    _load_torch()

    from abtem.core import _torch

    if _torch.DEVICE != "mps":
        raise RuntimeError(
            "Emulated float64 (device 'mps' with precision 'float64', or "
            "torch.emulate-float64) runs Metal kernels, so it needs the 'mps' torch "
            f"device, but the configuration key 'torch.device' is "
            f"'{_torch.DEVICE}'."
        )
    if not hasattr(_torch.torch.mps, "compile_shader"):
        raise RuntimeError(
            "Emulated float64 on Metal compiles Metal kernels with "
            "torch.mps.compile_shader, which PyTorch "
            f"{_torch.torch.__version__} does not have. Upgrade PyTorch."
        )
    return _torch


def _load_torch():
    """Import PyTorch and the Metal array namespace once, behind the guards
    against the crashes an unguarded import can cause."""
    global tp, TorchNDArray

    if tp is None:
        with _TORCH_LOAD_LOCK:
            if tp is None:
                if not _TORCH_IMPORT_IS_SAFE:
                    raise RuntimeError(
                        "The Metal (MPS) backend cannot be loaded: pyfftw was "
                        "imported before abTEM, so its OpenMP runtime is already "
                        "loaded, and importing PyTorch after it would crash the "
                        "process. Import abtem (or torch) before pyfftw."
                    )

                torch_device = _config_get("torch.device")
                if not isinstance(torch_device, str) or torch_device.lower() not in (
                    "mps",
                    "cpu",
                ):
                    raise ValueError(
                        "The configuration key 'torch.device' must be 'mps' or "
                        f"'cpu', got {torch_device!r}."
                    )
                torch_device = torch_device.lower()

                # Importing PyTorch can break CuPy's runtime kernel compilation
                # in the same process (observed with ROCm), so a machine that
                # cannot have a Metal device is refused before the import.
                if torch_device == "mps" and not _is_macos():
                    raise RuntimeError(
                        "The Metal (MPS) backend is not available on this machine. "
                        "Metal requires macOS; change the device to 'cpu'."
                    )

                from abtem.core import _torch

                _torch.DEVICE = torch_device

                _torch._check_available()

                # tp last: other threads test it without the lock, and must
                # not see it before TorchNDArray is in place too.
                TorchNDArray = _torch.TorchNDArray
                tp = _torch.torch_numpy


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

_CONFIG_PLUGIN_NAME = "abtem-config"


def _apply_config_snapshot(snapshot):
    import copy

    from abtem.core.config import config as config_dict
    from abtem.core.config import config_lock

    snapshot = copy.deepcopy(snapshot)
    with config_lock:
        # Update before pruning: readers that do not take the lock (config.get)
        # then always observe a fully-populated dict, never the empty window a
        # clear-then-update would open to a concurrently executing task.
        config_dict.update(snapshot)
        for key in [k for k in config_dict if k not in snapshot]:
            del config_dict[key]


def _make_config_plugin(snapshot):
    from distributed.diagnostics.plugin import WorkerPlugin

    class _AbtemConfigPlugin(WorkerPlugin):
        """Apply the client's abTEM configuration snapshot on every worker.

        Plugin ``setup`` runs on all current workers at registration time and
        on every worker that joins or is restarted later (e.g. by a Nanny) --
        coverage a one-shot ``client.run`` cannot provide.
        """

        name = _CONFIG_PLUGIN_NAME

        def __init__(self, snapshot):
            self._snapshot = snapshot

        def setup(self, worker=None):
            _apply_config_snapshot(self._snapshot)

    return _AbtemConfigPlugin(snapshot)


def push_config_to_workers(client):
    """Mirror this process's abTEM configuration onto the client's workers.

    abTEM resolves configuration inside tasks, in the worker process --
    ``get_dtype`` reads ``precision`` at call time, for example -- but worker
    processes only ever see the defaults: ``abtem.config.set`` in the client
    does not reach them, silently changing results (a float64 computation
    dispatched to default-configured workers runs in float32).

    The snapshot is carried by a named worker plugin, so workers that join or
    restart later also receive it; when the configuration changes,
    re-registering under the same name replaces the plugin and re-runs its
    setup on all workers. Repeated pushes of an unchanged configuration to the
    same client (keyed on ``client.id``) are skipped.
    """
    global _pushed_config_token

    import copy

    snapshot = copy.deepcopy(config)
    token = (getattr(client, "id", None) or id(client), repr(snapshot))
    if token == _pushed_config_token:
        return
    plugin = _make_config_plugin(snapshot)
    try:
        client.register_plugin(plugin, name=_CONFIG_PLUGIN_NAME)
    except AttributeError:  # distributed without Client.register_plugin
        client.register_worker_plugin(plugin, name=_CONFIG_PLUGIN_NAME)
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

        if x.lower() in ("torch", "mps", "metal"):
            if config.get("precision") == "float64":
                # Metal has no float64 arrays: double precision on 'mps' is the
                # emulation, whose data are float64 NumPy arrays on the host (see
                # emulates_float64), so this device is NumPy's with Metal kernels.
                _check_kernels_once()
                return np
            return check_mps_is_available()

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

    if tp is not None and (isinstance(x, (TorchNDArray, tp.Tensor)) or x is tp):
        return tp

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

    if tp is not None and xp is tp:
        return "mps"

    raise ValueError(
        f"array module must be NumPy, CuPy or the Metal namespace, not {xp}"
    )


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

    elif tp is not None and xp is tp:
        from abtem.core import _torch

        return _torch.scipy

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

    if tp is not None and xp is tp:
        from abtem.core import _torch

        return _torch.ndimage

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
    if tp is not None:
        if isinstance(array, (TorchNDArray, tp.Tensor)):
            return tp.asnumpy(array)

        # A lazy array whose chunks live on Metal still has to be brought
        # across. Without this it falls to the cp-less short-circuit below and
        # is handed back untouched -- still device-backed, so the caller's next
        # reduction runs against Metal chunks rather than host memory, having
        # asked for NumPy.
        if isinstance(array, da.core.Array) and get_array_module(array) is tp:
            return da.map_blocks(asnumpy, array)

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

    if tp is not None and old_xp is tp:
        array = tp.asnumpy(array)
        old_xp = np

    if new_xp is np:
        return array if old_xp is np else cp.asnumpy(array)

    if new_xp is cp:
        return cp.asarray(array)

    if tp is not None and new_xp is tp:
        return tp.asarray(array)

    raise RuntimeError("Invalid device specified")
