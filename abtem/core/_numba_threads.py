"""Seed Numba's thread pool size from ``OMP_NUM_THREADS``, before Numba's
own first import anywhere in this package.

Kept as its own module, imported as the very first statement in
``abtem/__init__.py``, rather than inlined there or placed alongside the
fallback in ``abtem/core/backend.py``: it must run -- and this module must
not itself import numba -- before anything else in this package does,
including the first import that pulls numba in (``abtem.core.backend``,
reached transitively via ``abtem.distributions``).
"""

import os


def _seed_numba_num_threads_from_omp() -> None:
    """Set ``NUMBA_NUM_THREADS`` from ``OMP_NUM_THREADS``, if neither is
    already set.

    ``NUMBA_NUM_THREADS`` is read once, by Numba itself, the first time it
    is imported, and applies process-wide -- every thread Numba's default
    "workqueue" layer ever launches, not just the thread that happened to
    trigger the import. That is a stronger guarantee than
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
