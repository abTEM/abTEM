"""Tests for the ``torch.device`` configuration key of the torch backend.

The key is read when the backend loads, so each test reloads it from scratch with
the backend's state put back afterwards. Needs PyTorch and either a Mac or
``ABTEM_TORCH__DEVICE=cpu``: the key's ``'cpu'`` value runs the backend on torch's
CPU device.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import requires_mps

import abtem
from abtem.core import backend
from abtem.core.backend import asnumpy, get_array_module

# Selected with the other tests of the torch backend (see test_mps.py), and run
# only where it loads: the fixture below imports PyTorch, which a machine that
# cannot use Metal must not do partway through a test run (see test_backend.py).
pytestmark = requires_mps


@pytest.fixture
def unloaded_backend(monkeypatch):
    """Put the backend's state back after the test; returns ``abtem.core._torch``."""
    from abtem.core import _torch

    monkeypatch.setattr(backend, "tp", None)
    monkeypatch.setattr(backend, "TorchNDArray", None)
    monkeypatch.setattr(backend, "_kernels_checked", False)
    monkeypatch.setattr(_torch, "DEVICE", "mps")
    return _torch


def test_cpu_runs_a_multislice_on_torch_cpu_tensors(unloaded_backend):
    _torch = unloaded_backend
    # Not the canonical spelling: the key is case-insensitive, and the one
    # .lower() is all that differs between "cpu", "CPU" and "Cpu".
    value = "CPU"
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 3)

    with abtem.config.set(
        {"torch.device": value, "dask.lazy": False, "precision": "float32"}
    ):
        arrays = []
        for device in ("cpu", "mps"):
            potential = abtem.Potential(atoms, gpts=128, device=device)
            waves = abtem.PlaneWave(energy=100e3, device=device).multislice(potential)
            arrays.append(waves.array)

    assert _torch.DEVICE == "cpu"
    assert _torch.is_available()
    assert isinstance(arrays[1], backend.TorchNDArray)
    assert arrays[1]._tensor.device.type == "cpu"
    assert np.allclose(
        asnumpy(arrays[1]), arrays[0], atol=1e-3 * np.abs(arrays[0]).max()
    )


def test_invalid_device_is_rejected_naming_the_key(unloaded_backend):
    # A non-string, an unknown name and the empty string: the two halves of the
    # check (isinstance, membership) plus the falsy edge.
    for value in (None, "cuda", ""):
        with abtem.config.set({"torch.device": value}):
            with pytest.raises(ValueError, match="torch.device.*'mps' or 'cpu'"):
                get_array_module("mps")

        assert backend.tp is None, value


def test_double_precision_on_mps_needs_the_real_metal_device(unloaded_backend):
    # Double precision on 'mps' is the emulation: allowed to be configured, but
    # its kernels need a real Metal device, which torch's CPU device is not.
    with abtem.config.set({"torch.device": "cpu"}):
        with abtem.config.set({"device": "mps", "precision": "float64"}):
            with pytest.raises(RuntimeError, match="Metal kernels"):
                get_array_module("mps")

        xp = get_array_module("mps")
        with pytest.raises(RuntimeError, match="single-precision"):
            xp.asarray(np.zeros(4), dtype=np.float64)
