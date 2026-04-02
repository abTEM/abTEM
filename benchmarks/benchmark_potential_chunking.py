"""
Benchmark: Potential chunking performance for large multislice simulations.

Designed to run on a workstation with an NVIDIA RTX 4090 (24 GB VRAM).
Tests both CPU and GPU with various potential sizes and chunk strategies.

Usage:
    python benchmarks/benchmark_potential_chunking.py
    python benchmarks/benchmark_potential_chunking.py --device gpu
    python benchmarks/benchmark_potential_chunking.py --device cpu
    python benchmarks/benchmark_potential_chunking.py --device gpu --quick
"""

import argparse
import gc
import os
import sys
import time
import threading

# Ensure the repo root is on sys.path so we import the local abtem, not an
# editable install pointing at a different checkout.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from dataclasses import dataclass, field

import numpy as np
from ase.build import bulk

import abtem
from abtem import PlaneWave, Potential
from abtem.core import config


@dataclass
class BenchmarkResult:
    label: str
    device: str
    gpts: tuple[int, int]
    num_slices: int
    chunk_size: str | int
    potential_bytes_mb: float
    time_seconds: float
    extra: dict = field(default_factory=dict)


def make_large_potential(
    gpts: tuple[int, int],
    repetitions: tuple[int, int, int],
    device: str = "cpu",
    slice_thickness: float = 2.0,
) -> Potential:
    """Create a large silicon potential for benchmarking."""
    atoms = bulk("Si", cubic=True) * repetitions
    potential = Potential(
        atoms,
        gpts=gpts,
        slice_thickness=slice_thickness,
        device=device,
    )
    return potential


def estimate_potential_memory_mb(potential: Potential) -> float:
    """Estimate total potential memory in MB (float32)."""
    num_slices = len(potential)
    gpts = potential.gpts
    dtype_bytes = np.dtype(np.float32).itemsize
    return num_slices * gpts[0] * gpts[1] * dtype_bytes / 1e6


def warmup_gpu():
    """Run a small computation to warm up the GPU."""
    try:
        import cupy as cp

        a = cp.ones((256, 256), dtype=np.complex64)
        cp.fft.fft2(a)
        cp.cuda.Stream.null.synchronize()
        del a
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass


class PeakVRAMTracker:
    """Context manager that samples GPU memory in a background thread.

    Captures the peak cupy memory-pool usage during the lifetime of the
    context, rather than only measuring after the computation finishes
    (when temporary allocations have already been freed).
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample_loop(self):
        import cupy as cp
        pool = cp.get_default_memory_pool()
        while not self._stop.is_set():
            used = pool.used_bytes()
            if used > self.peak_bytes:
                self.peak_bytes = used
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_bytes = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1e6


def benchmark_multislice(
    potential: Potential,
    chunk_size: int | str,
    device: str,
    energy: float = 200e3,
    n_repeats: int = 3,
) -> BenchmarkResult:
    """Benchmark a single multislice run with given chunk size."""
    waves = PlaneWave(energy=energy, gpts=potential.gpts, device=device)
    waves.grid.match(potential)

    label = f"gpts={potential.gpts}, slices={len(potential)}, chunk={chunk_size}"
    mem_mb = estimate_potential_memory_mb(potential)

    # Warmup run (not timed)
    try:
        result = waves.multislice(
            potential, potential_chunk_size=chunk_size, lazy=False
        )
        del result
    except Exception as e:
        return BenchmarkResult(
            label=label,
            device=device,
            gpts=potential.gpts,
            num_slices=len(potential),
            chunk_size=chunk_size,
            potential_bytes_mb=mem_mb,
            time_seconds=float("nan"),
            extra={"error": f"{type(e).__name__}: {e}"},
        )

    gc.collect()
    if device == "gpu":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    # Timed runs
    times = []
    peak_vram_mb = 0.0
    for _ in range(n_repeats):
        if device == "gpu":
            import cupy as cp

            cp.cuda.Stream.null.synchronize()

        if device == "gpu":
            tracker = PeakVRAMTracker(interval=0.002)
        else:
            tracker = None

        t0 = time.perf_counter()
        if tracker:
            tracker.__enter__()

        result = waves.multislice(
            potential, potential_chunk_size=chunk_size, lazy=False
        )

        if device == "gpu":
            import cupy as cp

            cp.cuda.Stream.null.synchronize()

        if tracker:
            tracker.__exit__(None, None, None)
            peak_vram_mb = max(peak_vram_mb, tracker.peak_mb)

        t1 = time.perf_counter()
        times.append(t1 - t0)
        del result

        gc.collect()
        if device == "gpu":
            cp.get_default_memory_pool().free_all_blocks()

    extra = {"all_times": times}
    if peak_vram_mb > 0:
        extra["peak_vram_mb"] = peak_vram_mb

    return BenchmarkResult(
        label=label,
        device=device,
        gpts=potential.gpts,
        num_slices=len(potential),
        chunk_size=chunk_size,
        potential_bytes_mb=mem_mb,
        time_seconds=np.median(times),
        extra=extra,
    )


def benchmark_build_only(
    potential: Potential,
    chunk_size: int | str,
    device: str,
    n_repeats: int = 3,
) -> BenchmarkResult:
    """Benchmark just the potential building (no multislice propagation)."""
    label = f"build-only gpts={potential.gpts}, slices={len(potential)}, chunk={chunk_size}"
    mem_mb = estimate_potential_memory_mb(potential)

    # Warmup
    for chunk in potential.generate_chunked_slices(chunk_size=chunk_size):
        pass

    gc.collect()
    if device == "gpu":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for chunk in potential.generate_chunked_slices(chunk_size=chunk_size):
            pass
        t1 = time.perf_counter()
        times.append(t1 - t0)

        gc.collect()
        if device == "gpu":
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()

    return BenchmarkResult(
        label=label,
        device=device,
        gpts=potential.gpts,
        num_slices=len(potential),
        chunk_size=chunk_size,
        potential_bytes_mb=mem_mb,
        time_seconds=np.median(times),
        extra={"all_times": times},
    )


def print_result(result: BenchmarkResult):
    err = result.extra.get("error")
    if err:
        print(
            f"  {result.label:<55s}  "
            f"ERROR: {err}"
        )
    else:
        times_str = ", ".join(f"{t:.3f}" for t in result.extra.get("all_times", []))
        vram = result.extra.get("peak_vram_mb")
        vram_str = f"  vram={vram / 1000:5.1f}GB" if vram else ""
        print(
            f"  {result.label:<55s}  "
            f"{result.time_seconds:8.3f}s"
            f"{vram_str}  "
            f"[{times_str}]"
        )


def run_benchmarks(device: str, quick: bool = False):
    print(f"\n{'=' * 90}")
    print(f"Potential Chunking Benchmark — device={device}")
    print(f"abTEM version: {abtem.__version__}")
    print(f"{'=' * 90}")

    if device == "gpu":
        try:
            import cupy as cp

            dev = cp.cuda.Device()
            free, total = dev.mem_info
            print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            print(f"VRAM: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
            warmup_gpu()
        except ImportError:
            print("ERROR: cupy not available, cannot run GPU benchmarks")
            return

    # ─── Benchmark configurations ───
    #
    # Each config: (gpts, repetitions, description)
    # Repetitions control the number of slices (z-direction).
    # gpts controls lateral resolution and memory per slice.
    #
    # Memory per slice (float32): gpts^2 * 4 bytes
    #   512x512   =   1 MB/slice
    #  1024x1024  =   4 MB/slice
    #  2048x2048  =  16 MB/slice
    #  4096x4096  =  64 MB/slice
    #
    if quick:
        configs = [
            ((256, 256), (2, 2, 10), "small (256x256, ~50 slices)"),
            ((512, 512), (2, 2, 10), "medium (512x512, ~50 slices)"),
        ]
    else:
        configs = [
            ((512, 512), (2, 2, 10), "small (512x512, ~50 slices)"),
            ((1024, 1024), (2, 2, 20), "medium (1024x1024, ~100 slices)"),
            ((2048, 2048), (2, 2, 20), "large (2048x2048, ~100 slices, ~1.7 GB)"),
            ((2048, 2048), (10, 10, 60), "xlarge (2048x2048, ~300 slices, ~5 GB)"),
        ]
        if device == "gpu":
            configs.extend([
                # These are designed to approach or exceed 24 GB RTX 4090 VRAM.
                # build(lazy=False) should OOM; chunked multislice should succeed.
                ((2048, 2048), (10, 10, 200), "xxlarge (2048x2048, ~1000 slices, ~17 GB)"),
                ((4096, 4096), (20, 20, 60), "VRAM-exceed (4096x4096, ~300 slices, ~20 GB)"),
                ((4096, 4096), (20, 20, 120), "VRAM-exceed-2x (4096x4096, ~600 slices, ~40 GB)"),
            ])

    # Chunk sizes to test
    chunk_sizes: list[int | str] = [1, 10, 50, "auto"]

    for gpts, reps, desc in configs:
        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_mb = estimate_potential_memory_mb(potential)

        mem_gb = mem_mb / 1000
        print(f"\n── {desc}: {num_slices} slices, {mem_gb:.2f} GB potential ──")

        if device == "gpu":
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

        # First benchmark: build the full potential at once (reference)
        print("\n  [Full build reference]")
        full_build_ok = False
        try:
            t0 = time.perf_counter()
            full = potential.build(lazy=False)
            t1 = time.perf_counter()
            print(f"  build(lazy=False) all {num_slices} slices:  {t1 - t0:.3f}s")
            full_build_ok = True
            del full
        except (MemoryError, Exception) as e:
            ename = type(e).__name__
            print(f"  build(lazy=False) FAILED ({ename}): {e}")
            print(f"  >>> This is expected — potential ({mem_gb:.1f} GB) exceeds available memory.")

        gc.collect()
        if device == "gpu":
            cp.get_default_memory_pool().free_all_blocks()

        # Benchmark: potential building only (no multislice)
        print("\n  [Potential build (generate_chunked_slices)]")
        for cs in chunk_sizes:
            if isinstance(cs, int) and cs > num_slices:
                continue
            result = benchmark_build_only(potential, cs, device, n_repeats=3)
            print_result(result)

        # Benchmark: full multislice
        print("\n  [Full multislice (PlaneWave, 200 keV)]")

        # Reference: pre-built potential (if it fits in memory)
        if full_build_ok:
            try:
                prebuilt = potential.build(lazy=False)
                result = benchmark_multislice(prebuilt, "auto", device, n_repeats=3)
                result.label = f"pre-built PotentialArray, chunk=auto"
                print_result(result)
                del prebuilt
            except (MemoryError, Exception) as e:
                print(f"  pre-built PotentialArray: OOM ({type(e).__name__})")
        else:
            print(f"  pre-built PotentialArray: SKIPPED (full build failed above)")

        gc.collect()
        if device == "gpu":
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()

        # Chunked: unbuilt Potential with various chunk sizes
        for cs in chunk_sizes:
            if isinstance(cs, int) and cs > num_slices:
                continue
            result = benchmark_multislice(potential, cs, device, n_repeats=3)
            print_result(result)

        gc.collect()
        if device == "gpu":
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 90}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark potential chunking for abTEM multislice"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Device to benchmark (default: both)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick subset of benchmarks",
    )
    args = parser.parse_args()

    # Disable dask lazy evaluation for benchmarking
    config.set({"dask.lazy": False})

    if args.device in ("cpu", "both"):
        run_benchmarks("cpu", quick=args.quick)

    if args.device in ("gpu", "both"):
        run_benchmarks("gpu", quick=args.quick)


if __name__ == "__main__":
    main()
