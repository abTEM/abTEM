"""
Profile memory allocations during a scan multislice computation.

Measures peak RSS (Resident Set Size) at key points to understand
where memory goes during probe.scan() → compute(). Runs on CPU so
we can use tracemalloc for precise allocation tracking.

Usage:
    python benchmarks/profile_scan_memory.py
    python benchmarks/profile_scan_memory.py --gpts 4096
"""

import argparse
import gc
import os
import sys
import tracemalloc
import time

os.environ["TQDM_DISABLE"] = "1"

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from ase.build import bulk

import abtem
from abtem import AnnularDetector, Potential, Probe
from abtem.core import config
from abtem.scan import GridScan


def get_rss_mb():
    """Get current RSS in MB (macOS/Linux)."""
    try:
        import resource
        # maxrss is in bytes on Linux, KB on macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / 1e6  # macOS: bytes
        return rss / 1e3  # Linux: KB
    except ImportError:
        return 0


def fmt_mb(nbytes):
    return f"{nbytes / 1e6:.1f} MB"


def run_profile(gpts_size=2048, reps_z=60, scan_gpts=(4, 4), max_batch=2):
    config.set({"device": "cpu", "dask.lazy": False})

    reps = (10, 10, reps_z)
    gpts = (gpts_size, gpts_size)

    print(f"{'=' * 70}")
    print(f"Memory Profile — CPU, gpts={gpts}, reps={reps}")
    print(f"scan={scan_gpts}, max_batch={max_batch}")
    print(f"{'=' * 70}")

    # --- Sizes ---
    atoms = bulk("Si", cubic=True) * reps
    potential = Potential(atoms, gpts=gpts, slice_thickness=2.0, device="cpu")
    num_slices = len(potential)
    slice_bytes = gpts[0] * gpts[1] * np.dtype(np.float32).itemsize
    total_pot_bytes = num_slices * slice_bytes

    print(f"\nPotential: {num_slices} slices, {fmt_mb(total_pot_bytes)} total")
    print(f"  Slice: {gpts[0]}x{gpts[1]} float32 = {fmt_mb(slice_bytes)}")

    n_positions = scan_gpts[0] * scan_gpts[1]
    wave_bytes = gpts[0] * gpts[1] * np.dtype(np.complex128).itemsize
    batch_wave_bytes = max_batch * wave_bytes
    print(f"  Wave (1 probe): {fmt_mb(wave_bytes)} complex128")
    print(f"  Wave batch ({max_batch}): {fmt_mb(batch_wave_bytes)}")
    print(f"  Scan positions: {n_positions}, batches: {(n_positions + max_batch - 1) // max_batch}")

    # Transmission function: complex128, same gpts as potential
    tf_bytes = gpts[0] * gpts[1] * np.dtype(np.complex128).itemsize
    print(f"  Transmission func: {fmt_mb(tf_bytes)} complex128")
    print(f"  Propagator kernel: {fmt_mb(tf_bytes)} complex128 (cached)")

    # --- Theoretical peak for different chunk sizes ---
    print(f"\n  Theoretical peak per batch (wave + chunk + transmission + FFT workspace):")
    for cs in [1, 10, 50, "all"]:
        n = num_slices if cs == "all" else cs
        chunk_mem = n * slice_bytes  # potential chunk (float32)
        # During propagation: wave (complex128) + transmission (complex128) + FFT of wave
        # fft2_convolve does: fft2(wave) → wave *= kernel → ifft2(wave), all in-place if overwrite_x=True
        # But propagator calls with in_place=True which means overwrite_x=True in fft2_convolve
        # transmission_function: complex_exponential(sigma * slice) → new complex128 array
        # transmit: waves._array *= tf_array → in-place
        # So per slice during propagation, simultaneously in memory:
        #   wave_batch + propagator_kernel + chunk_array + current_transmission_func
        #   + FFT workspace (fft2 of wave_batch, in-place)
        prop_peak = batch_wave_bytes + tf_bytes + chunk_mem + tf_bytes
        # FFT workspace: cupy/numpy fft2 with overwrite_x=True may or may not allocate
        # conservatively add 1x wave for FFT
        prop_peak += batch_wave_bytes
        print(f"    chunk={str(cs):>4s}: chunk={fmt_mb(chunk_mem):>10s}  "
              f"propagation_peak={fmt_mb(prop_peak):>10s}")

    gc.collect()

    # --- tracemalloc profiling ---
    for scheduler_label, scheduler in [("threaded (default)", None), ("synchronous", "synchronous")]:
        print(f"\n--- Running with tracemalloc [{scheduler_label}] ---")

        for chunk_size in [1, 10, 50, "auto"]:
            gc.collect()
            tracemalloc.start()

            probe = Probe(energy=200e3, semiangle_cutoff=20, device="cpu")
            detector = AnnularDetector(inner=40, outer=200)
            scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

            t0 = time.perf_counter()
            lazy_result = probe.scan(
                potential, scan=scan, detectors=detector, lazy=True,
                max_batch=max_batch, potential_chunk_size=chunk_size,
            )
            compute_kwargs = {}
            if scheduler:
                compute_kwargs["scheduler"] = scheduler
            lazy_result.compute(**compute_kwargs)
            elapsed = time.perf_counter() - t0

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            del lazy_result, probe, detector, scan
            gc.collect()

            print(f"  chunk={str(chunk_size):>4s}: {elapsed:7.1f}s  "
                  f"peak_mem={fmt_mb(peak):>10s}  current={fmt_mb(current):>10s}")

    # --- Also profile just the build path ---
    print(f"\n--- Build-only profiling (generate_chunked_slices) ---")

    for chunk_size in [1, 10, 50, "auto"]:
        gc.collect()
        tracemalloc.start()

        t0 = time.perf_counter()
        for chunk in potential.generate_chunked_slices(chunk_size=chunk_size):
            pass  # discard immediately
        elapsed = time.perf_counter() - t0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()

        print(f"  chunk={str(chunk_size):>4s}: {elapsed:7.1f}s  "
              f"peak_mem={fmt_mb(peak):>10s}  current={fmt_mb(current):>10s}")

    # --- Snapshot for top allocations at chunk=auto ---
    print(f"\n--- Top allocations during chunk=auto scan ---")
    gc.collect()
    tracemalloc.start()

    probe = Probe(energy=200e3, semiangle_cutoff=20, device="cpu")
    detector = AnnularDetector(inner=40, outer=200)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

    lazy_result = probe.scan(
        potential, scan=scan, detectors=detector, lazy=True,
        max_batch=max_batch, potential_chunk_size="auto",
    )
    lazy_result.compute()

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Top 15 by size
    stats = snapshot.statistics("lineno")
    print(f"  Top 15 allocations by line:")
    for stat in stats[:15]:
        print(f"    {stat}")

    # Top 10 by traceback
    print(f"\n  Top 10 allocations by file:")
    stats_file = snapshot.statistics("filename")
    for stat in stats_file[:10]:
        print(f"    {stat}")

    del lazy_result, probe, detector, scan
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpts", type=int, default=2048)
    parser.add_argument("--reps-z", type=int, default=60)
    parser.add_argument("--scan-gpts", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--max-batch", type=int, default=2)
    args = parser.parse_args()

    run_profile(
        gpts_size=args.gpts,
        reps_z=args.reps_z,
        scan_gpts=tuple(args.scan_gpts),
        max_batch=args.max_batch,
    )


if __name__ == "__main__":
    main()
