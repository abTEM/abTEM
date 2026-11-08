"""Cap Numba's thread count at ``OMP_NUM_THREADS``: seed Numba's thread pool
size before Numba's own first import anywhere in this package, and scope the
cap to each launch of the ``parallel=True`` kernels when the application
imported Numba first.

Kept as its own module, imported as the very first statement in
``abtem/__init__.py``, rather than inlined there or placed in
``abtem/core/backend.py``: it must run -- and this module must
not itself import numba -- before anything else in this package does,
including the first import that pulls numba in (``abtem.core.backend``,
reached transitively via ``abtem.distributions``).

The cap is ``OMP_NUM_THREADS``, at most the number of CPUs usable by the
process, when ``NUMBA_NUM_THREADS`` is unset and ``OMP_NUM_THREADS`` is a
positive integer; ``thread_cap`` holds it, or ``None``. With abTEM imported
before Numba, the seeded ``NUMBA_NUM_THREADS`` sizes Numba's whole pool at the
cap, for every ``parallel=True`` kernel in the process, and the interpolation
pool of the potential has the cap's size. With Numba imported first its pool
has its own size, and the cap reaches the interpolation pool and abTEM's
kernels only: ``numba_thread_cap`` sets it on the launching thread for the
duration of each launch, and other threads of the process keep Numba's count.

One call site in this package caps a ``parallel=True`` kernel with
``numba_thread_cap``: the real-space Laplacian stencil
(``_laplace_stencil_cpu_batch`` in ``finite_difference.py``, reached only
through ``RealSpaceMultislice``, not the default Fourier multislice path, which
is FFT-bound and never touches Numba's thread pool). Three more kernels are
compiled with ``parallel=True`` and have no caller in this package, so with
Numba imported first they are not capped, nor is a kernel that a user calls
directly: ``central_difference_gradient_pbc`` and
``central_difference_gradient_cbc`` (both in ``magnetism/pauli.py``;
``apply_A_xy_dot_nabla_xy`` is their only wrapper and has no caller) and
``reduce_beamlets_nearest_no_interpolation``
(``prism/_partitioned_s_matrix.py``).

**HPC schedulers routinely export ``OMP_NUM_THREADS=1`` by default, and a
single abTEM process then runs these kernels on that many threads
instead of on every visible core, which is a real change for a
single-process run.** Measured directly
(10-core/20-thread Xeon and a 24-core box, both memory-bandwidth-bound
kernels): the cost of that is real but modest and does not scale with
core count -- roughly 15-36%
slower than the best setting found (which peaked around 8-12 threads on
the 24-core box and *degraded slightly* beyond that, never at the full
core count) -- not a factor of how many cores are visible. Under many
concurrent dask worker threads, which is abTEM's actual typical execution
shape rather than one isolated call, the cost disappears entirely:
measured with 24 concurrent callers each repeatedly invoking the
Laplacian stencil, total throughput was statistically indistinguishable
across ``NUMBA_NUM_THREADS`` from 1 to 24 -- once that many independent
tasks already saturate the node, giving each one its own internal thread
pool on top has nothing left to parallelize into. A single-process,
single-call run (a small script, not a scan under dask) is the one shape
where this is worth checking before assuming it is free.

The escape hatch, for a single-process run that turns out to be
unexpectedly slow under an inherited ``OMP_NUM_THREADS=1``: set
``NUMBA_NUM_THREADS`` explicitly, which this module always leaves alone.

.. code-block:: bash

    export NUMBA_NUM_THREADS=8   # overrides an inherited OMP_NUM_THREADS=1
"""

import os
import sys
from contextlib import contextmanager


def _seed_numba_num_threads_from_omp() -> int | None:
    """Set ``NUMBA_NUM_THREADS`` from ``OMP_NUM_THREADS``, if neither is
    already set.

    ``NUMBA_NUM_THREADS`` is read by Numba itself the first time it
    is imported, and applies process-wide -- every thread Numba's parallel
    threading layer ever launches (whichever it selects: ``workqueue`` by
    default when neither TBB nor OpenMP is available, ``tbb`` or ``omp``
    otherwise -- verified directly that this holds even when the active
    layer is genuinely ``omp``, i.e. real libomp threads underneath, since
    Numba still governs how many of them a given call may use through its
    own config rather than deferring to ``OMP_NUM_THREADS`` at the libomp
    level), not just the thread that happened to trigger the import. That
    is a stronger guarantee than a single call of
    ``numba.set_num_threads()`` at import: ``set_num_threads`` only
    rebinds the *calling* thread's own active count. A CPU-side lazy
    abTEM computation, which dask's threaded scheduler executes on
    spawned worker threads rather than the caller's own thread, is
    exactly the case ``set_num_threads`` does not reach -- verified
    directly: after ``set_num_threads(2)`` on the main thread, the same
    ``@njit(parallel=True)`` kernel launched from a spawned worker thread
    still ran with the full visible core count of threads, not 2.

    Left untouched when ``NUMBA_NUM_THREADS`` is already set explicitly,
    so a deliberate, Numba-specific choice is never overridden. Clamped
    to the number of CPUs usable by the process, which is also what Numba's
    own default thread count is computed from: Numba's import-time handling of
    ``NUMBA_NUM_THREADS`` is not bounds-checked against the visible core
    count, and an implausibly large value segfaults the process outright
    the moment a parallel kernel first launches (verified directly:
    ``NUMBA_NUM_THREADS=99999`` dumps core on first use, not a catchable
    exception), rather than raising cleanly the way ``set_num_threads``
    does above its own launch ceiling.

    When the application imported numba before abTEM, Numba has already read
    its configuration and would apply a new value only at its next
    compilation, in the middle of whatever abTEM is computing. The value is
    therefore applied at once by reloading Numba's configuration. When Numba's
    threads already run, the variable is left unset, because Numba raises at
    its next compilation when it changes after that point.

    Returns the cap, also when Numba's pool already runs and the variable is
    left unset, or ``None`` when no cap applies: ``NUMBA_NUM_THREADS`` is set,
    or ``OMP_NUM_THREADS`` is unset, not an integer or not positive.
    """
    if "NUMBA_NUM_THREADS" in os.environ:
        return None

    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is None:
        return None

    try:
        n = int(omp_num_threads)
    except ValueError:
        return None

    if n <= 0:
        return None

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    usable_cpus = (
        len(sched_getaffinity(0)) if sched_getaffinity else os.cpu_count() or 1
    )
    cap = min(n, usable_cpus)

    numba = sys.modules.get("numba")
    if numba is not None:
        try:
            numba.threading_layer()
        except ValueError:  # the thread pool has not started
            pass
        else:
            return cap

    os.environ["NUMBA_NUM_THREADS"] = str(cap)
    if numba is not None:
        numba.config.reload_config()
    return cap


thread_cap = _seed_numba_num_threads_from_omp()

# The cap is scoped to each launch only with Numba imported first: with abTEM
# first, the pool the seeding sized is the cap.
_numba_imported_first = "numba" in sys.modules


@contextmanager
def numba_thread_cap():
    """Run a ``parallel=True`` kernel launch on at most ``thread_cap``
    threads, when Numba was imported before abTEM and there is a cap.

    Entered on the thread that launches the kernel, because
    ``numba.set_num_threads`` applies to the calling thread only and dask runs
    the kernels on worker threads. The thread's own count is restored on exit,
    so a thread that also runs a kernel of the application keeps its count,
    and nothing is set when that count is already within the cap.
    """
    if thread_cap is None or not _numba_imported_first:
        yield
        return
    import numba

    previous = numba.get_num_threads()
    if previous <= thread_cap:
        yield
        return
    numba.set_num_threads(thread_cap)
    try:
        yield
    finally:
        numba.set_num_threads(previous)
