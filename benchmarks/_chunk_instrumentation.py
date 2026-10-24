"""
Shared instrumentation for abTEM's two GPU memory-budget estimators:
``estimate_potential_chunk_size`` and ``estimate_scan_batch_size``
(both in ``abtem.core.chunks``).

Both functions decide a size ONCE per call by reading live VRAM at that
moment (``cp.cuda.Device().mem_info`` + the CuPy pool's ``used_bytes()``),
then hand back a fixed integer that the caller uses for a whole loop
(potential-chunk sizing: every chunk in one ``generate_chunked_slices``
call; probe-batch sizing: the whole scan). Neither call re-checks VRAM
again until it is invoked afresh. This module exists to answer, from real
runs, two questions that reasoning alone can't settle:

1. What are the actual internal numbers (free_mem, pool_used,
   effective_free, budget_bytes, effective_per_slice/per_probe) at the
   moment each estimator is called -- not just the integer it returns?
2. How do those numbers, and the pool's live/dead-block split, drift
   *within* a single call's loop and *across* successive calls (successive
   rotations, successive probe batches, successive energies)?

Usage
-----
Call ``instrument_chunk_estimators()`` once, early, before building any
lazy graph. It monkeypatches both estimators in place (transparent to
callers -- return values are unchanged) and prints one line per call. Call
``instrument_chunk_estimators(uninstall=True)`` to restore the originals.

This is diagnostic-only: it duplicates each estimator's internal formula
in a read-only side channel purely for logging, then calls the real
function for the actual (unaffected) return value. It must be run on the
GPU machine (needs cupy); on CPU or without cupy it prints a warning once
and becomes a no-op.
"""

import sys
import time

import numpy as np

_call_counter = {"potential_chunk": 0, "scan_batch": 0}
_installed = {"potential_chunk": None, "scan_batch": None}
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _fmt_gb(nbytes):
    return f"{nbytes / 1e9:.3f}GB"


def _log_potential_chunk_call(gpts, device, dtype, chunk_size):
    _call_counter["potential_chunk"] += 1
    n = _call_counter["potential_chunk"]

    if device != "gpu":
        print(f"[t={_elapsed():7.2f}s] potential_chunk_size call #{n}: "
              f"device={device} gpts={gpts} -> {chunk_size} (no VRAM formula on this device)")
        sys.stdout.flush()
        return

    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        free_mem, total_mem = cp.cuda.Device().mem_info
        pool_used = pool.used_bytes()
        pool_free = pool.free_bytes()
        pool_total = pool.total_bytes()
        # Mirrors the real formula in abtem.core.chunks.estimate_potential_chunk_size
        # (as of commits c24b92fb / affdc1bc): counts the pool's own idle cache
        # (pool_free -- already reserved from CUDA but not currently live) as
        # available, and uses a 2x per-slice factor calibrated from real peak
        # pool.used_bytes() measurements rather than the original 5x.
        effective_free = min(free_mem + pool_free, total_mem - pool_used)

        if dtype is None:
            itemsize = 4
        else:
            itemsize = np.dtype(dtype).itemsize
        slice_bytes = gpts[0] * gpts[1] * itemsize
        effective_per_slice = slice_bytes * 2
        budget_bytes = int(effective_free * 0.35)

        print(
            f"[t={_elapsed():7.2f}s] potential_chunk_size call #{n}: "
            f"gpts={gpts} device={device}\n"
            f"    cuda_free={_fmt_gb(free_mem)} cuda_total={_fmt_gb(total_mem)} "
            f"pool_used={_fmt_gb(pool_used)} pool_free_cached={_fmt_gb(pool_free)} "
            f"pool_total_reserved={_fmt_gb(pool_total)}\n"
            f"    effective_free=min(cuda_free+pool_free_cached, cuda_total-pool_used)={_fmt_gb(effective_free)}\n"
            f"    slice_bytes={_fmt_gb(slice_bytes)} effective_per_slice(x2)={_fmt_gb(effective_per_slice)} "
            f"budget_bytes(35%% of effective_free)={_fmt_gb(budget_bytes)}\n"
            f"    -> resolved chunk_size = {chunk_size}"
        )
    except Exception as e:  # pragma: no cover - diagnostic path only
        print(f"[t={_elapsed():7.2f}s] potential_chunk_size call #{n}: "
              f"instrumentation failed ({type(e).__name__}: {e}); resolved={chunk_size}")
    sys.stdout.flush()


def _log_scan_batch_call(gpts, dtype, device, n_probes):
    _call_counter["scan_batch"] += 1
    n = _call_counter["scan_batch"]

    if device != "gpu":
        print(f"[t={_elapsed():7.2f}s] scan_batch_size call #{n}: "
              f"device={device} gpts={gpts} -> {n_probes} (no VRAM formula on this device)")
        sys.stdout.flush()
        return

    try:
        import cupy as cp

        from abtem.core.fft import is_fast_fft_size

        pool = cp.get_default_memory_pool()
        free_mem, total_mem = cp.cuda.Device().mem_info
        pool_used = pool.used_bytes()
        pool_free = pool.free_bytes()
        pool_total = pool.total_bytes()
        # Mirrors the real formula in abtem.core.chunks.estimate_scan_batch_size
        # (as of commit c24b92fb): counts the pool's own idle cache as available.
        # The per-probe overhead factor (6x / 12x for Bluestein) is unchanged --
        # only estimate_potential_chunk_size's per-slice factor was recalibrated.
        effective_free = min(free_mem + pool_free, total_mem - pool_used)

        overhead = 6 if all(is_fast_fft_size(g) for g in gpts) else 12
        per_probe_bytes = int(np.prod(gpts)) * np.dtype(dtype).itemsize
        per_probe_effective = max(1, int(per_probe_bytes * overhead))
        probe_budget = int(effective_free * 0.50)

        print(
            f"[t={_elapsed():7.2f}s] scan_batch_size call #{n}: "
            f"gpts={gpts} device={device} dtype={dtype}\n"
            f"    cuda_free={_fmt_gb(free_mem)} cuda_total={_fmt_gb(total_mem)} "
            f"pool_used={_fmt_gb(pool_used)} pool_free_cached={_fmt_gb(pool_free)} "
            f"pool_total_reserved={_fmt_gb(pool_total)}\n"
            f"    effective_free=min(cuda_free+pool_free_cached, cuda_total-pool_used)={_fmt_gb(effective_free)}\n"
            f"    per_probe_bytes={_fmt_gb(per_probe_bytes)} overhead={overhead}x "
            f"per_probe_effective={_fmt_gb(per_probe_effective)} "
            f"probe_budget(50%% of effective_free)={_fmt_gb(probe_budget)}\n"
            f"    -> resolved n_probes (pre power-of-two rounding target) = {n_probes}"
        )
    except Exception as e:  # pragma: no cover - diagnostic path only
        print(f"[t={_elapsed():7.2f}s] scan_batch_size call #{n}: "
              f"instrumentation failed ({type(e).__name__}: {e}); resolved={n_probes}")
    sys.stdout.flush()


def instrument_chunk_estimators(uninstall=False):
    """Monkeypatch estimate_potential_chunk_size / estimate_scan_batch_size
    to log their internal free-VRAM/budget numbers on every call, without
    changing their return values.

    Idempotent: calling this twice without ``uninstall=True`` in between is
    a no-op (it will not double-wrap). Call with ``uninstall=True`` to
    restore the untouched originals.
    """
    from abtem.core import chunks as _chunks_mod

    if uninstall:
        if _installed["potential_chunk"] is not None:
            _chunks_mod.estimate_potential_chunk_size = _installed["potential_chunk"]
            _installed["potential_chunk"] = None
        if _installed["scan_batch"] is not None:
            _chunks_mod.estimate_scan_batch_size = _installed["scan_batch"]
            _installed["scan_batch"] = None
        return

    if _installed["potential_chunk"] is None:
        _orig_potential = _chunks_mod.estimate_potential_chunk_size

        def _wrapped_potential(gpts, device="cpu", dtype=None):
            result = _orig_potential(gpts, device, dtype)
            _log_potential_chunk_call(gpts, device, dtype, result)
            return result

        _installed["potential_chunk"] = _orig_potential
        _chunks_mod.estimate_potential_chunk_size = _wrapped_potential

    if _installed["scan_batch"] is None:
        _orig_scan = _chunks_mod.estimate_scan_batch_size

        def _wrapped_scan(gpts, dtype, device):
            result = _orig_scan(gpts, dtype, device)
            _log_scan_batch_call(gpts, dtype, device, result)
            return result

        _installed["scan_batch"] = _orig_scan
        _chunks_mod.estimate_scan_batch_size = _wrapped_scan

    # Re-point every module-level alias already imported elsewhere (e.g.
    # ``from abtem.core.chunks import estimate_potential_chunk_size`` inside
    # abtem.waves / abtem.potentials.iam) at the wrapped versions too, since
    # those call sites hold their own reference captured at import time.
    import abtem.waves as _waves_mod

    if hasattr(_waves_mod, "estimate_potential_chunk_size"):
        _waves_mod.estimate_potential_chunk_size = _chunks_mod.estimate_potential_chunk_size
