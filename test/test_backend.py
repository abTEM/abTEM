import ast
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

import abtem
from abtem.core import backend


# Functions in the Metal backend that touch torch without taking _TORCH_LOCK,
# each for a reason that keeps it safe: it launches no Metal work, or it is
# only ever called from inside a function that already holds the lock.
_UNLOCKED_BY_DESIGN = {
    "is_available": "queries the backend, launches nothing",
    "_check_available": "queries the backend, launches nothing",
    "_wrap": "an isinstance check",
    "iscomplexobj": "a dtype query",
    "_unwrap_key": "only called under the lock, from __getitem__/__setitem__",
    "_resolve_reversed_slices": "only called under the lock, from __getitem__",
    "_common_dtype": "only called under the lock, from the contractions",
}


def test_metal_backend_serializes_every_torch_call():
    """PyTorch's MPS backend is not thread-safe, and dask's threaded scheduler
    reaches it from several threads at once, so every function that launches
    Metal work has to hold _TORCH_LOCK. A missing @_serialized is invisible
    to any single-threaded test, and in a threaded run shows up only as an
    intermittent abort or hang -- so check the source for it instead.

    Needs neither torch nor Apple silicon: it only parses the module.
    """
    source_path = Path(__file__).parents[1] / "abtem" / "core" / "_torch.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    def is_serialized(function):
        return any(
            isinstance(decorator, ast.Name) and decorator.id == "_serialized"
            for decorator in function.decorator_list
        )

    def calls_torch(function):
        return any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            for node in ast.walk(function)
        )

    def returns_serialized_closure(function):
        # the factories (_elementwise, _fft_func, ...) wrap what they build
        return "return _serialized(" in ast.get_source_segment(source, function)

    unlocked = sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and calls_torch(node)
        and not is_serialized(node)
        and not returns_serialized_closure(node)
        and node.name not in _UNLOCKED_BY_DESIGN
    )

    assert not unlocked, (
        f"these functions reach torch without @_serialized: {unlocked}. "
        "Decorate them, or add them to _UNLOCKED_BY_DESIGN with the reason "
        "they are safe."
    )


# Importing PyTorch can break CuPy's kernel compilation in the same process
# (observed with a ROCm build), so on a machine that cannot have a Metal device
# neither asking for 'mps' nor collecting the tests may import it. Whether it is
# imported is process-wide state, hence the fresh processes.
_not_macos = pytest.mark.skipif(
    backend._is_macos(), reason="the Metal backend is loaded on macOS"
)


def _run_in_fresh_process(code):
    # The device is set explicitly: the torch-CPU job runs with it set to 'cpu'.
    env = {**os.environ, "ABTEM_TORCH__DEVICE": "mps"}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@_not_macos
def test_mps_is_refused_off_macos_without_importing_torch():
    result = _run_in_fresh_process("""
import sys

import abtem
from abtem.core import backend

try:
    backend.check_mps_is_available()
except RuntimeError as error:
    print(error)
else:
    raise SystemExit("check_mps_is_available() did not raise")

print("TORCH_IMPORTED", "torch" in sys.modules or "abtem.core._torch" in sys.modules)
""")

    assert result.returncode == 0, result.stderr
    assert "Metal requires macOS" in result.stdout
    assert "TORCH_IMPORTED False" in result.stdout


@_not_macos
@pytest.mark.skipif(
    str(abtem.config.get("torch.device")).lower() != "mps",
    reason="torch is the configured backend (torch.device='cpu'), so loading it is intended",
)
def test_collecting_the_tests_does_not_import_torch(request):
    """Read off this run's own collection (see pytest_collection_finish in
    conftest.py) rather than collecting the suite again in a subprocess,
    which took ~15 s: it covers the tests this run collected -- all of them
    in a full run."""
    assert not request.config._abtem_torch_imported_by_collection


@pytest.mark.parametrize("machine", ["arm64", "x86_64"])
def test_a_mac_is_left_to_torch_to_answer(monkeypatch, machine):
    """Apple silicon and Intel Macs with an AMD GPU both have PyTorch's MPS
    backend, so on macOS the request reaches it whatever the architecture."""

    class FakeTorchBackend:
        DEVICE = "mps"
        TorchNDArray = object
        torch_numpy = object()
        asked = False

        @classmethod
        def _check_available(cls):
            cls.asked = True
            raise RuntimeError("torch's answer")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(backend, "tp", None)
    monkeypatch.setattr(backend, "TorchNDArray", None)
    monkeypatch.setattr(abtem.core, "_torch", FakeTorchBackend, raising=False)
    monkeypatch.setitem(sys.modules, "abtem.core._torch", FakeTorchBackend)

    with abtem.config.set({"torch.device": "mps"}):
        with pytest.raises(RuntimeError, match="torch's answer"):
            backend.check_mps_is_available()

    assert FakeTorchBackend.asked
