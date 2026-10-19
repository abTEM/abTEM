"""Module for handling the array backend (NumPy, CuPy, Dask, etc.) of the library."""

from __future__ import annotations

import warnings
from numbers import Number
from types import ModuleType
from typing import Union

import dask.array as da
import numpy as np
import scipy  # type: ignore
import scipy.ndimage  # type: ignore

from abtem.core.config import config

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


ArrayModule = Union[ModuleType, str]


def check_cupy_is_installed():
    """
    Check if CuPy is installed, raise an error if not.
    """
    if cp is None:
        raise RuntimeError("CuPy is not installed, GPU calculations disabled")


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
