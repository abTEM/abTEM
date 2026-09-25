import ast
from pathlib import Path

import numba
import pytest

from abtem.core.backend import _cap_numba_threads_to_omp_num_threads


class TestCapNumbaThreadsToOmpNumThreads:
    """``_cap_numba_threads_to_omp_num_threads`` runs once as an import-time
    side effect (see ``abtem/core/backend.py``), so these call the function
    directly rather than reimporting the module -- reimporting would not
    even exercise a fresh call in the same process, since Python caches
    modules after their first import.
    """

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
        self, monkeypatch
    ):
        """set_num_threads raises ValueError above numba.config.NUMBA_NUM_THREADS
        (the launch-time ceiling derived from the visible CPU count) -- a
        larger OMP_NUM_THREADS, plausible on a misconfigured or constrained
        allocation, must be clamped rather than crash the abtem import.
        """
        monkeypatch.setenv("OMP_NUM_THREADS", str(numba.config.NUMBA_NUM_THREADS + 100))
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        calls = []
        monkeypatch.setattr(
            "abtem.core.backend.numba.set_num_threads", calls.append
        )

        _cap_numba_threads_to_omp_num_threads()

        assert calls == [numba.config.NUMBA_NUM_THREADS]


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
