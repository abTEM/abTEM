"""Tests for the ``torch.device`` configuration key of the torch backend.

The key is read when the backend loads, so each test reloads it from scratch with
the backend's state put back afterwards. Needs PyTorch, but not Apple silicon:
the key's ``'cpu'`` value runs the backend on torch's CPU device.
"""

import importlib.util

import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem.core import backend
from abtem.core.backend import asnumpy, get_array_module

# Selected with the other tests of the torch backend (see test_mps.py). PyTorch is
# not imported at module level: collecting this file must leave it unimported
# (see test_backend.py).
pytestmark = [
    pytest.mark.torch,
    pytest.mark.mps,
    pytest.mark.metal,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None, reason="requires PyTorch"
    ),
]


@pytest.fixture
def unloaded_backend(monkeypatch):
    """Put the backend's state back after the test; returns ``abtem.core._torch``."""
    from abtem.core import _torch

    monkeypatch.setattr(backend, "tp", None)
    monkeypatch.setattr(backend, "TorchNDArray", None)
    monkeypatch.setattr(_torch, "DEVICE", "mps")
    return _torch


@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu"])
def test_cpu_runs_a_multislice_on_torch_cpu_tensors(unloaded_backend, value):
    _torch = unloaded_backend
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


@pytest.mark.parametrize("value", [None, -1, "cuda", "", 1.5, ["cpu"]])
def test_invalid_device_is_rejected_naming_the_key(unloaded_backend, value):
    with abtem.config.set({"torch.device": value}):
        with pytest.raises(ValueError, match="torch.device.*'mps' or 'cpu'"):
            get_array_module("mps")

    assert backend.tp is None


def test_double_precision_is_still_refused_on_cpu(unloaded_backend):
    with abtem.config.set({"torch.device": "cpu"}):
        with pytest.raises(ValueError, match="single-precision"):
            abtem.config.set({"device": "mps", "precision": "float64"})

        with abtem.config.set({"precision": "float64"}):
            with pytest.raises(RuntimeError, match="single-precision"):
                get_array_module("mps")

        xp = get_array_module("mps")
        with pytest.raises(RuntimeError, match="single-precision"):
            xp.asarray(np.zeros(4), dtype=np.float64)
