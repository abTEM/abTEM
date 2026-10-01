import numba
import pytest

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
