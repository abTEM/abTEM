"""Seed Numba's thread pool size from ``OMP_NUM_THREADS``, before Numba's
own first import anywhere in this package.

Kept as its own module, imported as the very first statement in
``abtem/__init__.py``, rather than inlined there or placed alongside the
fallback in ``abtem/core/backend.py``: it must run -- and this module must
not itself import numba -- before anything else in this package does,
including the first import that pulls numba in (``abtem.core.backend``,
reached transitively via ``abtem.distributions``).

Only three kernels in this package are compiled with ``parallel=True`` and
therefore affected at all: the real-space Laplacian stencil
(``finite_difference.py``, reached only through ``RealSpaceMultislice``,
not the default Fourier multislice path, which is FFT-bound and never
touches Numba's thread pool), the non-collinear magnetism gradient kernels
(``magnetism/pauli.py``), and the partitioned S-matrix kernel
(``prism/_partitioned_s_matrix.py``).

**HPC schedulers routinely export ``OMP_NUM_THREADS=1`` by default, and a
single abTEM process now runs those three kernels single-threaded there
instead of on every visible core, which is a real change for a
single-process run.** Measured directly (10-core/20-thread Xeon and a
24-core box, both memory-bandwidth-bound kernels): the cost of that is
real but modest and does not scale with core count -- roughly 15-36%
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


def _seed_numba_num_threads_from_omp() -> None:
    """Set ``NUMBA_NUM_THREADS`` from ``OMP_NUM_THREADS``, if neither is
    already set.

    ``NUMBA_NUM_THREADS`` is read once, by Numba itself, the first time it
    is imported, and applies process-wide -- every thread Numba's parallel
    threading layer ever launches (whichever it selects: ``workqueue`` by
    default when neither TBB nor OpenMP is available, ``tbb`` or ``omp``
    otherwise -- verified directly that this holds even when the active
    layer is genuinely ``omp``, i.e. real libomp threads underneath, since
    Numba still governs how many of them a given call may use through its
    own config rather than deferring to ``OMP_NUM_THREADS`` at the libomp
    level), not just the thread that happened to trigger the import. That
    is a stronger guarantee than
    ``numba.set_num_threads()`` (see ``abtem/core/backend.py``'s own call,
    kept as a fallback for a caller that imports numba before abtem, when
    this env var can no longer take effect): ``set_num_threads`` only
    rebinds the *calling* thread's own active count. A CPU-side lazy
    abTEM computation, which dask's threaded scheduler executes on
    spawned worker threads rather than the caller's own thread, is
    exactly the case ``set_num_threads`` does not reach -- verified
    directly: after ``set_num_threads(2)`` on the main thread, the same
    ``@njit(parallel=True)`` kernel launched from a spawned worker thread
    still ran with the full visible core count of threads, not 2.

    Left untouched when ``NUMBA_NUM_THREADS`` is already set explicitly,
    so a deliberate, Numba-specific choice is never overridden. Clamped
    to ``os.cpu_count()``: Numba's own import-time handling of
    ``NUMBA_NUM_THREADS`` is not bounds-checked against the visible core
    count, and an implausibly large value segfaults the process outright
    the moment a parallel kernel first launches (verified directly:
    ``NUMBA_NUM_THREADS=99999`` dumps core on first use, not a catchable
    exception), rather than raising cleanly the way ``set_num_threads``
    does above its own launch ceiling.
    """
    if "NUMBA_NUM_THREADS" in os.environ:
        return

    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is None:
        return

    try:
        n = int(omp_num_threads)
    except ValueError:
        return

    if n > 0:
        os.environ["NUMBA_NUM_THREADS"] = str(min(n, os.cpu_count() or 1))


_seed_numba_num_threads_from_omp()
