import ast
import os
import platform
import subprocess
import sys
from pathlib import Path

import numba
import pytest

import abtem
from abtem.core import backend
from abtem.core.backend import _cap_numba_threads_to_omp_num_threads


class TestCapNumbaThreadsToOmpNumThreads:
    """``_cap_numba_threads_to_omp_num_threads`` runs once as an import-time
    side effect (see ``abtem/core/backend.py``), so these call the function
    directly rather than reimporting the module -- reimporting would not
    even exercise a fresh call in the same process, since Python caches
    modules after their first import.

    Every test pins numba's launch-time ceiling, ``numba.config.NUMBA_NUM_THREADS``,
    via the ``ceiling`` fixture: the real value is fixed when numba is first
    imported and depends on the test process's own environment -- e.g. an
    inherited ``OMP_NUM_THREADS=1`` (which some GPAW versions set on import,
    and which pytest-xdist workers then inherit from the controller) makes
    ``abtem.core._numba_threads`` seed a ceiling of 1, which would clamp the
    value under test.
    """

    @pytest.fixture(autouse=True)
    def ceiling(self, monkeypatch):
        monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 8)
        return 8

    def test_applies_omp_num_threads_when_numba_num_threads_is_unset(
        self, monkeypatch
    ):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        calls = []
        monkeypatch.setattr(
            "abtem.core.backend.numba.set_num_threads", calls.append
        )

        _cap_numba_threads_to_omp_num_threads()

        assert calls == [3]

    def test_does_not_override_an_explicit_numba_num_threads(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.setenv("NUMBA_NUM_THREADS", "7")
        calls = []
        monkeypatch.setattr(
            "abtem.core.backend.numba.set_num_threads", calls.append
        )

        _cap_numba_threads_to_omp_num_threads()

        assert calls == []

    def test_does_nothing_when_omp_num_threads_is_unset(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        calls = []
        monkeypatch.setattr(
            "abtem.core.backend.numba.set_num_threads", calls.append
        )

        _cap_numba_threads_to_omp_num_threads()

        assert calls == []

    @pytest.mark.parametrize("bad_value", ["0", "-1", "not-a-number", ""])
    def test_invalid_or_non_positive_omp_num_threads_is_a_no_op(
        self, monkeypatch, bad_value
    ):
        monkeypatch.setenv("OMP_NUM_THREADS", bad_value)
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        calls = []
        monkeypatch.setattr(
            "abtem.core.backend.numba.set_num_threads", calls.append
        )

        _cap_numba_threads_to_omp_num_threads()

        assert calls == []

    def test_omp_num_threads_above_numbas_launch_ceiling_is_clamped(
        self, monkeypatch, ceiling
    ):
        """set_num_threads raises ValueError above numba.config.NUMBA_NUM_THREADS
        (the launch-time ceiling derived from the visible CPU count) -- a
        larger OMP_NUM_THREADS, plausible on a misconfigured or constrained
        allocation, must be clamped rather than crash the abtem import.
        """
        monkeypatch.setenv("OMP_NUM_THREADS", str(ceiling + 100))
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        calls = []
        monkeypatch.setattr(
            "abtem.core.backend.numba.set_num_threads", calls.append
        )

        _cap_numba_threads_to_omp_num_threads()

        assert calls == [ceiling]


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
def test_collecting_the_tests_does_not_import_torch():
    result = _run_in_fresh_process(f"""
import sys

import pytest

exit_code = pytest.main(
    ["--collect-only", "-q", "-p", "no:cacheprovider", {str(Path(__file__).parent)!r}]
)
print("COLLECT_EXIT", int(exit_code))
print("TORCH_IMPORTED", "torch" in sys.modules or "abtem.core._torch" in sys.modules)
""")

    assert "COLLECT_EXIT 0" in result.stdout, result.stdout[-2000:] + result.stderr
    assert "TORCH_IMPORTED False" in result.stdout


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
