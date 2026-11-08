import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numba
import pytest

from abtem.core import _numba_threads
from abtem.core._numba_threads import (
    _numba_thread_cap,
    _seed_numba_num_threads_from_omp,
    _usable_cpus,
)
from abtem.integrals import _interpolation_threads


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

# The imports of each process, up to and including `import abtem`, which leave
# numba's configured thread count before the import in `config_before`.
_ABTEM_FIRST = "import abtem\nimport numba\nimport numpy as np\nconfig_before = None\n"
_NUMBA_FIRST = (
    "import numba\n"
    "import numpy as np\n"
    "config_before = numba.config.NUMBA_NUM_THREADS\n"
    "import abtem\n"
)
_POOL_RUNNING = (
    "import numba\n"
    "import numpy as np\n"
    f"{_START_POOL}"
    "config_before = numba.config.NUMBA_NUM_THREADS\n"
    "import abtem\n"
)
# The process may run on two CPUs from the import of abtem on, while
# OMP_NUM_THREADS is 3: the cap is the usable CPUs.
_RESTRICTED = (
    "import numba\n"
    "import numpy as np\n"
    "import os\n"
    "config_before = numba.config.NUMBA_NUM_THREADS\n"
    "os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])\n"
    "import abtem\n"
)

# Launches the real-space Laplacian stencil from spawned threads, as dask's
# threaded scheduler does, with the thread count numba has there recorded at the
# kernel launch, builds a small potential, and prints the findings as JSON.
# Numba sets the thread count per thread, so one thread keeps numba's count and
# another has chosen a count of its own (the integer `chosen`, 0 for none). The
# stencil has different sizes along the stacked, row and column axes.
_BODY = """
import contextlib
import hashlib
import json
import threading

import dask
from ase.build import bulk

import abtem.core._numba_threads as numba_threads
import abtem.finite_difference as finite_difference
import abtem.integrals as integrals

abtem.config.set({"fft": "numpy"})
result = {
    "cap_is_noop": numba_threads.numba_thread_cap is contextlib.nullcontext,
    "config_before": config_before,
    "interpolation_threads": integrals._interpolation_threads(),
}

used = []
kernel = finite_difference._laplace_stencil_cpu_kernel
finite_difference._laplace_stencil_cpu_kernel = (
    lambda *args: used.append(numba.get_num_threads()) or kernel(*args)
)
stencil = finite_difference._laplace_operator_stencil(
    4, 1.0, mode="none", dtype=np.complex64
)


def launch(chosen):
    out = {}

    def run():
        if chosen:
            numba.set_num_threads(chosen)
        out["before"] = numba.get_num_threads()
        stencil(np.ones((3, 16, 20), dtype=np.complex64))
        out["launched"] = used[-1]
        out["after"] = numba.get_num_threads()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return out


result["default"] = launch(0)
result["chosen"] = launch(chosen) if chosen else None

# The number of interpolation chunks of each call of the threaded interpolation.
calls = []
interpolate = integrals.interpolate_radial_functions
integrals.interpolate_radial_functions = (
    lambda *args: calls.append(1) or interpolate(*args)
)
chunks = []
threaded = integrals._threaded_interpolate_radial_functions


def count_chunks(*args, **kwargs):
    calls_before = len(calls)
    threaded(*args, **kwargs)
    chunks.append(len(calls) - calls_before)


integrals._threaded_interpolate_radial_functions = count_chunks
with dask.config.set(scheduler="synchronous"):
    potential = abtem.Potential(
        bulk("Si", cubic=True), gpts=32, projection="finite", slice_thickness=2
    ).build(lazy=False).array
result["max_chunks"] = max(chunks)
result["digest"] = hashlib.sha1(np.ascontiguousarray(potential).tobytes()).hexdigest()

numba.njit(lambda x: x + 1)(1)
result["config_after"] = numba.config.NUMBA_NUM_THREADS
result["interpolation_threads_after"] = integrals._interpolation_threads()
print(json.dumps(result))
"""


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
    return json.loads(result.stdout.splitlines()[-1])


# name: (imports, OMP_NUM_THREADS, the count the second thread chooses, the
# CPUs the scenario needs: numba's pool has to be larger than the cap of 2 for
# the cap to differ from it). The cap is 2 in every scenario.
_SCENARIOS = {
    "abtem_first": (_ABTEM_FIRST, "2", 0, 1),
    "numba_first": (_NUMBA_FIRST, "2", 3, 3),
    "numba_first_pool_running": (_POOL_RUNNING, "2", 3, 3),
    "numba_first_restricted_cpus": (_RESTRICTED, "3", 3, 3),
}


@pytest.fixture(scope="module")
def runs():
    """The findings of one interpreter per import order, run side by side."""
    names = [name for name, v in _SCENARIOS.items() if v[3] <= _usable_cpus()]

    def run(name):
        imports, omp, chosen, _ = _SCENARIOS[name]
        script = f"chosen = {chosen}\n{imports}{_BODY}"
        return _run_python(script, {"OMP_NUM_THREADS": omp})

    with ThreadPoolExecutor(len(names)) as pool:
        return dict(zip(names, pool.map(run, names)))


def _scenario(name, runs):
    if name not in runs:
        pytest.skip(f"needs {_SCENARIOS[name][3]} usable CPUs")
    return runs[name]


@pytest.mark.parametrize("name", _SCENARIOS)
class TestEveryImportOrder:
    """``OMP_NUM_THREADS`` reaches the ``parallel=True`` kernels, the
    interpolation pool of the potential and numba's configuration the same way
    whichever of numba and abtem is imported first, also when numba's pool
    already runs, and when the process may use fewer CPUs than
    ``OMP_NUM_THREADS``. Only the import order needs a fresh interpreter."""

    def test_numba_thread_cap_does_nothing_with_abtem_imported_first(self, name, runs):
        assert _scenario(name, runs)["cap_is_noop"] == (name == "abtem_first")

    def test_the_stencil_launches_with_the_cap_and_the_thread_keeps_its_count(
        self, name, runs
    ):
        found = _scenario(name, runs)["default"]
        assert found["launched"] == 2
        assert found["after"] == found["before"]
        if name != "abtem_first":
            assert found["before"] > 2  # numba's pool, which the cap is below

    def test_a_count_the_application_chose_is_kept(self, name, runs):
        found = _scenario(name, runs)["chosen"]
        if name == "abtem_first":
            assert found is None
        else:
            assert found == {"before": 3, "launched": 2, "after": 3}

    def test_the_interpolation_pool_has_the_cap_before_and_after_a_compilation(
        self, name, runs
    ):
        found = _scenario(name, runs)
        assert found["interpolation_threads"] == 2
        assert found["interpolation_threads_after"] == 2
        assert found["max_chunks"] == 2

    def test_numbas_configuration_is_the_pool_it_has(self, name, runs):
        """It is the cap with abtem imported first, and what numba had before
        abtem was imported otherwise, through a compilation as well."""
        found = _scenario(name, runs)
        expected = 2 if name == "abtem_first" else found["config_before"]
        assert found["config_after"] == expected

    def test_the_potential_is_the_same_in_every_import_order(self, name, runs):
        """The thread count changes the order of the float32 sums in the last
        bits."""
        assert _scenario(name, runs)["digest"] == runs["abtem_first"]["digest"]
