import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from abtem.core._numba_threads import _seed_numba_num_threads_from_omp

# Numba's default thread count, which the seeded value has to differ from.
_AVAILABLE_CPUS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count() or 1
)


class TestSeedNumbaNumThreadsFromOmp:
    """Unit tests for the seeding logic itself. It writes to ``os.environ``
    for real (real thread-pool sizing happens later, at numba's own first
    import) -- ``os.environ`` is swapped for a disposable copy for the
    duration of each test (not just the individual keys these tests care
    about), so the function's real writes can never leak into the actual
    process environment. That matters here specifically: numba re-validates
    ``NUMBA_NUM_THREADS`` against its already-launched thread pool on every
    new compilation (``config.reload_config()``, called from
    ``numba/core/compiler.py``), so a leaked value from one of these tests
    reaching a real numba compilation anywhere else in the same pytest
    process raises ``RuntimeError: Cannot set NUMBA_NUM_THREADS to a
    different value once the threads have been launched`` -- confirmed as
    the actual cause of a real CI failure from an earlier version of these
    tests that used ``monkeypatch.setenv``/``delenv`` on the real
    environment instead.
    """

    @pytest.fixture(autouse=True)
    def _isolated_environ(self, monkeypatch):
        monkeypatch.setattr(os, "environ", os.environ.copy())
        # The seeding then takes the branch for a process that has not imported
        # numba; the tests of the other branch install a stand-in module.
        monkeypatch.delitem(sys.modules, "numba", raising=False)

    def test_sets_numba_num_threads_from_omp_num_threads(self):
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        _seed_numba_num_threads_from_omp()

        assert os.environ["NUMBA_NUM_THREADS"] == "3"

    def test_does_not_override_an_explicit_numba_num_threads(self):
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ["NUMBA_NUM_THREADS"] = "7"

        _seed_numba_num_threads_from_omp()

        assert os.environ["NUMBA_NUM_THREADS"] == "7"

    def test_does_nothing_when_omp_num_threads_is_unset(self):
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ.pop("NUMBA_NUM_THREADS", None)

        _seed_numba_num_threads_from_omp()

        assert "NUMBA_NUM_THREADS" not in os.environ

    @pytest.mark.parametrize("bad_value", ["0", "-1", "not-a-number", ""])
    def test_invalid_or_non_positive_omp_num_threads_is_a_no_op(self, bad_value):
        os.environ["OMP_NUM_THREADS"] = bad_value
        os.environ.pop("NUMBA_NUM_THREADS", None)

        _seed_numba_num_threads_from_omp()

        assert "NUMBA_NUM_THREADS" not in os.environ

    def test_omp_num_threads_above_cpu_count_is_clamped(self):
        """Numba's own import-time handling of NUMBA_NUM_THREADS is not
        bounds-checked against the visible core count, and an implausibly
        large value segfaults the process the moment a parallel kernel
        first launches -- verified directly (not asserted here, since that
        would crash the test process too): NUMBA_NUM_THREADS=99999 dumps
        core on first kernel use on a 24-core box.
        """
        os.environ["OMP_NUM_THREADS"] = "99999"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        _seed_numba_num_threads_from_omp()

        assert int(os.environ["NUMBA_NUM_THREADS"]) <= (os.cpu_count() or 1)

    @staticmethod
    def _install_numba(monkeypatch, pool_started):
        """A stand-in for an already imported numba that records the calls the
        seeding makes: ``threading_layer`` raises ``ValueError`` until the pool
        has started, and ``config.reload_config`` records the environment
        variable it would re-read."""
        reloads = []

        def threading_layer():
            if not pool_started:
                raise ValueError("Threading layer is not initialized.")
            return "workqueue"

        config = SimpleNamespace(
            reload_config=lambda: reloads.append(os.environ.get("NUMBA_NUM_THREADS"))
        )
        numba = SimpleNamespace(threading_layer=threading_layer, config=config)
        monkeypatch.setitem(sys.modules, "numba", numba)
        return reloads

    def test_imported_numba_without_a_started_pool_gets_the_value_at_once(
        self, monkeypatch
    ):
        reloads = self._install_numba(monkeypatch, pool_started=False)
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        _seed_numba_num_threads_from_omp()

        assert os.environ["NUMBA_NUM_THREADS"] == "3"
        assert reloads == ["3"]

    def test_imported_numba_with_a_started_pool_is_left_alone(self, monkeypatch):
        reloads = self._install_numba(monkeypatch, pool_started=True)
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        _seed_numba_num_threads_from_omp()

        assert "NUMBA_NUM_THREADS" not in os.environ
        assert reloads == []


class TestNumbaThreadsReachSpawnedWorkerThreads:
    """The property that actually matters, per review: the cap must hold
    where abTEM's parallel=True kernels actually execute, not just on
    whichever thread happened to import abtem.

    numba.set_num_threads() only
    rebinds the *calling* thread's own active count -- a spawned worker
    thread (exactly what dask's threaded scheduler uses for a CPU-side
    lazy computation) sees Numba's unmodified default instead. Only
    seeding NUMBA_NUM_THREADS before Numba's own first import (this
    module) applies process-wide, so this needs a fresh subprocess: by
    the time this test file itself runs, numba is already imported in
    this process with its pool sized without the seeding, and re-importing
    it here would not trigger a second read of the env var.
    """

    def _run(self, env):
        script = (
            "import abtem\n"
            "import numba\n"
            "from numba import njit, prange\n"
            "from numba.np.ufunc.parallel import get_thread_id\n"
            "import numpy as np\n"
            "import concurrent.futures\n"
            "\n"
            "@njit(parallel=True)\n"
            "def kernel(n):\n"
            "    ids = np.zeros(n, dtype=np.int64)\n"
            "    for i in prange(n):\n"
            "        ids[i] = get_thread_id()\n"
            "    return ids\n"
            "\n"
            "main_ids = set(kernel(64))\n"
            "with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:\n"
            "    worker_ids = set(ex.submit(lambda: set(kernel(64))).result())\n"
            "print(len(main_ids), len(worker_ids))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        main_count, worker_count = (int(x) for x in result.stdout.split())
        return main_count, worker_count

    def test_cap_reaches_a_spawned_worker_thread(self):
        env = os.environ.copy()
        env.pop("NUMBA_NUM_THREADS", None)
        env["OMP_NUM_THREADS"] = "2"
        main_count, worker_count = self._run(env)
        assert main_count == 2
        assert worker_count == 2


@pytest.mark.skipif(
    _AVAILABLE_CPUS < 3, reason="needs more available CPUs than the seeded value"
)
class TestNumbaImportedBeforeAbtem:
    """With numba imported before abtem (as pytest does through the
    ``filterwarnings`` entry naming a numba warning class), the seeded thread
    count must reach numba at ``import abtem`` and stay fixed afterwards.
    Needs fresh interpreters, because the result depends on the import order.
    """

    @staticmethod
    def _run(script):
        env = os.environ.copy()
        env.pop("NUMBA_NUM_THREADS", None)
        env["OMP_NUM_THREADS"] = "2"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.split()

    def test_the_seeded_value_applies_at_import_and_survives_a_compilation(self):
        script = (
            "import numba\n"
            "import abtem\n"
            "after_import = numba.config.NUMBA_NUM_THREADS\n"
            "numba.njit(lambda x: x + 1)(1)\n"
            "print(after_import, numba.config.NUMBA_NUM_THREADS)\n"
        )
        assert self._run(script) == ["2", "2"]

    def test_a_compilation_after_import_works_when_the_pool_already_runs(self):
        script = (
            "import numba\n"
            "import numpy as np\n"
            "\n"
            "@numba.njit(parallel=True)\n"
            "def total(x):\n"
            "    s = 0.0\n"
            "    for i in numba.prange(x.size):\n"
            "        s += x[i]\n"
            "    return s\n"
            "\n"
            "total(np.ones(100))\n"
            "launched = numba.config.NUMBA_NUM_THREADS\n"
            "import abtem\n"
            "numba.njit(lambda x: x + 1)(1)\n"
            "print(launched, numba.config.NUMBA_NUM_THREADS)\n"
        )
        launched, after = self._run(script)
        assert launched == after
