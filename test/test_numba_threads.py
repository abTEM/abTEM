import os
import subprocess
import sys

import numba
import pytest

from abtem.core import _numba_threads
from abtem.core._numba_threads import (
    _seed_numba_num_threads_from_omp,
    _numba_thread_cap,
    _usable_cpus,
)
from abtem.integrals import _interpolation_threads

# Numba's default thread count, which the seeded value has to differ from.
_AVAILABLE_CPUS = _usable_cpus()


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
        # numba; the test of the other branch installs the module again.
        monkeypatch.delitem(sys.modules, "numba", raising=False)

    def test_sets_numba_num_threads_from_omp_num_threads(self, monkeypatch):
        monkeypatch.setattr(_numba_threads, "_usable_cpus", lambda: 8)
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        assert _seed_numba_num_threads_from_omp() == 3

        assert os.environ["NUMBA_NUM_THREADS"] == "3"

    def test_does_not_override_an_explicit_numba_num_threads(self):
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ["NUMBA_NUM_THREADS"] = "7"

        assert _seed_numba_num_threads_from_omp() is None

        assert os.environ["NUMBA_NUM_THREADS"] == "7"

    def test_does_nothing_when_omp_num_threads_is_unset(self):
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ.pop("NUMBA_NUM_THREADS", None)

        assert _seed_numba_num_threads_from_omp() is None

        assert "NUMBA_NUM_THREADS" not in os.environ

    @pytest.mark.parametrize("bad_value", ["0", "-1", "not-a-number", ""])
    def test_invalid_or_non_positive_omp_num_threads_is_a_no_op(self, bad_value):
        os.environ["OMP_NUM_THREADS"] = bad_value
        os.environ.pop("NUMBA_NUM_THREADS", None)

        assert _seed_numba_num_threads_from_omp() is None

        assert "NUMBA_NUM_THREADS" not in os.environ

    def test_omp_num_threads_above_the_usable_cpus_is_clamped(self, monkeypatch):
        """Numba's own import-time handling of NUMBA_NUM_THREADS is not
        bounds-checked against the visible core count, and an implausibly
        large value segfaults the process the moment a parallel kernel
        first launches -- verified directly (not asserted here, since that
        would crash the test process too): NUMBA_NUM_THREADS=99999 dumps
        core on first kernel use on a 24-core box. The clamp is the number of
        CPUs the process may run on, the count Numba's own default is
        computed from.
        """
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False
        )
        os.environ["OMP_NUM_THREADS"] = "99999"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        assert _seed_numba_num_threads_from_omp() == 3

        assert os.environ["NUMBA_NUM_THREADS"] == "3"

    def test_without_sched_getaffinity_the_clamp_is_the_cpu_count(self, monkeypatch):
        """macOS and Windows have no ``os.sched_getaffinity``."""
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 5)
        os.environ["OMP_NUM_THREADS"] = "99999"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        assert _seed_numba_num_threads_from_omp() == 5

    def test_imported_numba_keeps_the_environment_and_gets_the_cap(self, monkeypatch):
        """Numba's pool has its size from its import, and a changed variable
        would be applied, or refused, at its next compilation."""
        monkeypatch.setattr(_numba_threads, "_usable_cpus", lambda: 8)
        monkeypatch.setitem(sys.modules, "numba", numba)
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ.pop("NUMBA_NUM_THREADS", None)

        assert _seed_numba_num_threads_from_omp() == 3

        assert "NUMBA_NUM_THREADS" not in os.environ


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
    ``filterwarnings`` entry naming a numba warning class), importing abtem
    leaves numba's configuration as it is, so no compilation changes it.
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

    def test_importing_abtem_leaves_numbas_configuration_alone(self):
        script = (
            "import numba\n"
            "before = numba.config.NUMBA_NUM_THREADS\n"
            "import abtem\n"
            "after_import = numba.config.NUMBA_NUM_THREADS\n"
            "numba.njit(lambda x: x + 1)(1)\n"
            "print(before, after_import, numba.config.NUMBA_NUM_THREADS)\n"
        )
        assert len(set(self._run(script))) == 1

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


class TestNumbaThreadCap:
    """The context manager around each ``parallel=True`` kernel launch, on this
    process's numba, with the module's recorded cap replaced. It is the one
    ``numba_thread_cap`` is bound to when numba was imported before abtem and
    there is a cap; the subprocess tests below check the binding in each import
    order and the thread count that the launched kernels use."""

    def test_caps_inside_the_block_and_restores_the_count_after_it(self, monkeypatch):
        monkeypatch.setattr(_numba_threads, "thread_cap", 1)
        before = numba.get_num_threads()

        with _numba_thread_cap():
            assert numba.get_num_threads() == 1

        assert numba.get_num_threads() == before

    def test_the_count_is_restored_when_the_kernel_raises(self, monkeypatch):
        monkeypatch.setattr(_numba_threads, "thread_cap", 1)
        before = numba.get_num_threads()

        with pytest.raises(ZeroDivisionError):
            with _numba_thread_cap():
                1 / 0

        assert numba.get_num_threads() == before

    def test_a_count_within_the_cap_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(_numba_threads, "thread_cap", 10**6)
        sets = []
        monkeypatch.setattr(numba, "set_num_threads", sets.append)

        with _numba_thread_cap():
            pass

        assert sets == []


class TestInterpolationThreads:
    def test_is_the_cap_when_there_is_one(self, monkeypatch):
        monkeypatch.setattr(_numba_threads, "thread_cap", 1)
        assert _interpolation_threads() == 1

    def test_is_the_configured_count_without_a_cap(self, monkeypatch):
        monkeypatch.setattr(_numba_threads, "thread_cap", None)
        assert _interpolation_threads() == max(int(numba.config.NUMBA_NUM_THREADS), 1)


def _run_python(script, extra_env=None):
    env = os.environ.copy()
    env.pop("NUMBA_NUM_THREADS", None)
    env["OMP_NUM_THREADS"] = "2"
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


# Each snippet defines `call`, which launches one real abTEM kernel with a tiny
# input, and records in `used` the thread count at the launch: the kernel is
# wrapped by a shim that reads numba's count in the launching thread and calls the
# real kernel. The sizes along the stacked, row and column axes differ.
_KERNEL_CALLS = {
    "real_space_laplacian": (
        "import abtem.finite_difference as module\n"
        "real = module._laplace_stencil_cpu_kernel\n"
        "module._laplace_stencil_cpu_kernel = (\n"
        "    lambda *args: used.append(numba.get_num_threads()) or real(*args)\n"
        ")\n"
        "stencil = module._laplace_operator_stencil(\n"
        "    4, 1.0, mode='none', dtype=np.complex64\n"
        ")\n"
        "call = lambda: stencil(np.ones((3, 16, 20), dtype=np.complex64))\n"
    ),
}

_START_POOL = (
    "@numba.njit(parallel=True)\n"
    "def total(x):\n"
    "    s = 0.0\n"
    "    for i in numba.prange(x.size):\n"
    "        s += x[i]\n"
    "    return s\n"
    "\n"
    "total(np.ones(100))\n"
)

_NUMBA_FIRST = "import numba\nimport numpy as np\nimport abtem\n"
_POOL_RUNNING = f"import numba\nimport numpy as np\n{_START_POOL}import abtem\n"
_ABTEM_FIRST = "import abtem\nimport numba\nimport numpy as np\n"

# scenario: (imports, environment, threads the thread has before the launch,
# or None for the pool size, threads the kernel is expected to launch with).
# OMP_NUM_THREADS is 2.
# The process may run on 2 CPUs while OMP_NUM_THREADS is 3: the cap is the usable CPUs.
_RESTRICT = "import os\nos.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])\n"
_THREE = {"OMP_NUM_THREADS": "3"}
_SCENARIOS = {
    "numba_first": (_NUMBA_FIRST, {}, None, 2),
    "numba_first_pool_running": (_POOL_RUNNING, {}, None, 2),
    "abtem_first": (_ABTEM_FIRST, {}, None, 2),
    "explicit_numba_num_threads": (_NUMBA_FIRST, {"NUMBA_NUM_THREADS": "3"}, None, 3),
    "numba_first_application_chose_3": (_NUMBA_FIRST, {}, 3, 2),
    "numba_first_application_chose_1": (_NUMBA_FIRST, {}, 1, 1),
    "restricted_cpus_numba_first": (_RESTRICT + _NUMBA_FIRST, _THREE, None, 2),
    "restricted_cpus_pool_running": (_RESTRICT + _POOL_RUNNING, _THREE, None, 2),
    "restricted_cpus_abtem_first": (_RESTRICT + _ABTEM_FIRST, _THREE, None, 2),
}
_NEEDS_AFFINITY = pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="needs os.sched_setaffinity"
)
_SCENARIO_PARAMS = [
    pytest.param(name, marks=_NEEDS_AFFINITY) if name.startswith("restricted") else name
    for name in _SCENARIOS
]


@pytest.mark.skipif(
    _AVAILABLE_CPUS < 3, reason="needs more available CPUs than the seeded value"
)
class TestKernelsFollowTheCap:
    """``OMP_NUM_THREADS=2`` reaches the ``parallel=True`` kernels whichever of
    numba and abtem is imported first, also when numba's pool already runs.
    Numba sets the thread count per thread, so the kernel is launched from a
    spawned thread, as dask's threaded scheduler does. The count the kernel
    launches with is read there, and the thread's own count has to be back
    afterwards. An explicit ``NUMBA_NUM_THREADS`` is not overridden, a count
    the application set on the thread is kept, and the cap does not exceed the
    CPUs the process may use."""

    @pytest.mark.parametrize("kernel", _KERNEL_CALLS)
    @pytest.mark.parametrize("scenario", _SCENARIO_PARAMS)
    def test_the_kernel_launches_with_the_cap_and_the_thread_keeps_its_count(
        self, scenario, kernel
    ):
        imports, extra_env, chosen, expected = _SCENARIOS[scenario]
        script = (
            f"{imports}"
            "import threading\n"
            "used = []\n"
            f"{_KERNEL_CALLS[kernel]}"
            "out = []\n"
            "def run():\n"
            f"    {'numba.set_num_threads(%d)' % chosen if chosen else 'pass'}\n"
            "    before = numba.get_num_threads()\n"
            "    call()\n"
            "    out.extend([before, used[0], numba.get_num_threads()])\n"
            "thread = threading.Thread(target=run)\n"
            "thread.start()\n"
            "thread.join()\n"
            "print(*out)\n"
        )
        before, launched, after = (int(x) for x in _run_python(script, extra_env))
        assert launched == expected
        assert after == before
        if chosen is None and scenario.startswith("numba_first"):
            assert before > expected  # the pool, which the cap is below
        elif chosen is not None:
            assert before == chosen


@pytest.mark.skipif(
    _AVAILABLE_CPUS < 3, reason="needs more available CPUs than the seeded value"
)
class TestInterpolationThreadsFollowTheCap:
    """The interpolation pool of the potential follows the same cap as the
    kernels, so the potential does not depend on the import order. The thread
    count changes the order of the float32 sums in the last bits."""

    @staticmethod
    def _in_every_import_order(omp, restrict=""):
        build = (
            "import hashlib\n"
            "from ase.build import bulk\n"
            "from abtem.integrals import _interpolation_threads\n"
            "abtem.config.set({'fft': 'numpy'})\n"
            "atoms = bulk('Si', cubic=True) * (2, 2, 2)\n"
            "potential = abtem.Potential(\n"
            "    atoms, gpts=64, projection='finite', slice_thickness=2\n"
            ").build(lazy=False).array\n"
            "digest = hashlib.sha1(np.ascontiguousarray(potential).tobytes())\n"
            "print(_interpolation_threads(), digest.hexdigest())\n"
        )
        return [
            _run_python(restrict + imports + build, {"OMP_NUM_THREADS": omp})
            for imports in (_NUMBA_FIRST, _POOL_RUNNING, _ABTEM_FIRST)
        ]

    @pytest.mark.parametrize("omp", ["2", "3"])
    def test_the_potential_is_bitwise_the_same_in_every_import_order(self, omp):
        results = self._in_every_import_order(omp)
        assert {threads for threads, _ in results} == {omp}
        assert len({digest for _, digest in results}) == 1

    @_NEEDS_AFFINITY
    def test_the_count_is_the_usable_cpus_in_every_import_order(self):
        results = self._in_every_import_order("3", restrict=_RESTRICT)
        assert {threads for threads, _ in results} == {"2"}
        assert len({digest for _, digest in results}) == 1
