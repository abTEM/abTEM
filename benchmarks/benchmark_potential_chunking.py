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
            extra={"error": str(e)},
        )

    gc.collect()
    if device == "gpu":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    # Timed runs
    times = []
    for _ in range(n_repeats):
        if device == "gpu":
            import cupy as cp

            cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        result = waves.multislice(
            potential, potential_chunk_size=chunk_size, lazy=False
        )
        if device == "gpu":
            import cupy as cp

            cp.cuda.Stream.null.synchronize()

        t1 = time.perf_counter()
        times.append(t1 - t0)
        del result

        gc.collect()
        if device == "gpu":
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
            f"  {result.label:<60s}  "
            f"ERROR: {err}"
        )
    else:
        times_str = ", ".join(f"{t:.3f}" for t in result.extra.get("all_times", []))
        print(
            f"  {result.label:<60s}  "
            f"{result.time_seconds:8.3f}s  "
            f"(pot={result.potential_bytes_mb:7.1f} MB)  "
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
    if quick:
        configs = [
            ((256, 256), (2, 2, 10), "small (256x256, ~50 slices)"),
            ((512, 512), (2, 2, 10), "medium (512x512, ~50 slices)"),
        ]
    else:
        configs = [
            ((256, 256), (2, 2, 10), "small (256x256, ~50 slices)"),
            ((512, 512), (2, 2, 10), "medium (512x512, ~50 slices)"),
            ((512, 512), (2, 2, 40), "large (512x512, ~200 slices)"),
            ((1024, 1024), (2, 2, 20), "xlarge (1024x1024, ~100 slices)"),
        ]
        if device == "gpu":
            # This one may exceed VRAM if built all at once
            configs.append(
                ((1024, 1024), (2, 2, 60), "xxlarge (1024x1024, ~300 slices)"),
            )
            configs.append(
                ((2048, 2048), (2, 2, 20), "huge (2048x2048, ~100 slices)"),
            )

    # Chunk sizes to test
    chunk_sizes: list[int | str] = [1, 5, 10, 25, 50, "auto"]

    for gpts, reps, desc in configs:
        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_mb = estimate_potential_memory_mb(potential)

        print(f"\n── {desc}: {num_slices} slices, {mem_mb:.1f} MB potential ──")

        # First benchmark: build the full potential at once (reference)
        print("\n  [Full build reference]")
        try:
            t0 = time.perf_counter()
            full = potential.build(lazy=False)
            t1 = time.perf_counter()
            print(f"  build(lazy=False) all {num_slices} slices:  {t1 - t0:.3f}s")
            del full
        except Exception as e:
            print(f"  build(lazy=False) FAILED: {e}")

        gc.collect()
        if device == "gpu":
            import cupy as cp
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
        try:
            prebuilt = potential.build(lazy=False)
            result = benchmark_multislice(prebuilt, "auto", device, n_repeats=3)
            result.label = f"pre-built PotentialArray, chunk=auto"
            print_result(result)
            del prebuilt
        except Exception as e:
            print(f"  pre-built PotentialArray: FAILED ({e})")

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
