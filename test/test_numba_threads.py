import os
import subprocess
import sys

import pytest

from abtem.core._numba_threads import _seed_numba_num_threads_from_omp


class TestSeedNumbaNumThreadsFromOmp:
    """Unit tests for the seeding logic itself. Since it only writes to
    ``os.environ`` (real thread-pool sizing happens later, at numba's own
    first import), these call the function directly with monkeypatched
    environment variables rather than spawning a subprocess -- that is
    reserved for ``TestNumbaThreadsReachSpawnedWorkerThreads`` below, which
    tests the property that actually depends on process/import order.
    """

    def test_sets_numba_num_threads_from_omp_num_threads(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)

        _seed_numba_num_threads_from_omp()

        assert os.environ["NUMBA_NUM_THREADS"] == "3"

    def test_does_not_override_an_explicit_numba_num_threads(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.setenv("NUMBA_NUM_THREADS", "7")

        _seed_numba_num_threads_from_omp()

        assert os.environ["NUMBA_NUM_THREADS"] == "7"

    def test_does_nothing_when_omp_num_threads_is_unset(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)

        _seed_numba_num_threads_from_omp()

        assert "NUMBA_NUM_THREADS" not in os.environ

    @pytest.mark.parametrize("bad_value", ["0", "-1", "not-a-number", ""])
    def test_invalid_or_non_positive_omp_num_threads_is_a_no_op(
        self, monkeypatch, bad_value
    ):
        monkeypatch.setenv("OMP_NUM_THREADS", bad_value)
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)

        _seed_numba_num_threads_from_omp()

        assert "NUMBA_NUM_THREADS" not in os.environ

    def test_omp_num_threads_above_cpu_count_is_clamped(self, monkeypatch):
        """Numba's own import-time handling of NUMBA_NUM_THREADS is not
        bounds-checked against the visible core count, and an implausibly
        large value segfaults the process the moment a parallel kernel
        first launches -- verified directly (not asserted here, since that
        would crash the test process too): NUMBA_NUM_THREADS=99999 dumps
        core on first kernel use on a 24-core box.
        """
        monkeypatch.setenv("OMP_NUM_THREADS", "99999")
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)

        _seed_numba_num_threads_from_omp()

        assert int(os.environ["NUMBA_NUM_THREADS"]) <= (os.cpu_count() or 1)


class TestNumbaThreadsReachSpawnedWorkerThreads:
    """The property that actually matters, per review: the cap must hold
    where abTEM's parallel=True kernels actually execute, not just on
    whichever thread happened to import abtem.

    numba.set_num_threads() (abtem/core/backend.py's fallback) only
    rebinds the *calling* thread's own active count -- a spawned worker
    thread (exactly what dask's threaded scheduler uses for a CPU-side
    lazy computation) sees Numba's unmodified default instead. Only
    seeding NUMBA_NUM_THREADS before Numba's own first import (this
    module) applies process-wide, so this needs a fresh subprocess: by
    the time this test file itself runs, numba is already imported in
    this process without the seeding, and re-importing it here would not
    trigger a second read of the env var.
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

    def test_uncapped_worker_thread_uses_every_visible_core(self):
        """Documents the failure mode this fix addresses: without
        OMP_NUM_THREADS set, both threads use every visible core -- the
        baseline this repository's own multi-worker test runs oversubscribe
        against when many such processes share one node.
        """
        env = os.environ.copy()
        env.pop("OMP_NUM_THREADS", None)
        env.pop("OPENBLAS_NUM_THREADS", None)
        env.pop("NUMBA_NUM_THREADS", None)
        main_count, worker_count = self._run(env)
        cpu_count = os.cpu_count() or 1
        assert main_count == cpu_count
        assert worker_count == cpu_count
