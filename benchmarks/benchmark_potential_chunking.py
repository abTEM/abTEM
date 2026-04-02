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

# Suppress tqdm progress bars (from dask/abtem internals)
os.environ["TQDM_DISABLE"] = "1"

# Ensure the repo root is on sys.path so we import the local abtem, not an
# editable install pointing at a different checkout.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from dataclasses import dataclass, field

import numpy as np
from ase.build import bulk

import abtem
from abtem import AnnularDetector, PlaneWave, Potential, Probe
from abtem.core import config
from abtem.scan import GridScan


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


def benchmark_scan(
    potential: Potential,
    scan_gpts: tuple[int, int],
    device: str,
    prebuilt: bool,
    potential_chunk_size: int | str = "auto",
    max_batch: int = 8,
    energy: float = 200e3,
    n_repeats: int = 2,
) -> BenchmarkResult:
    """Benchmark a Probe + GridScan measurement.

    If prebuilt=True, builds the potential eagerly once, then runs scan with
    lazy=False (pre-built GPU arrays can't stream through dask→zarr due to
    dtype serialization issues, but the result fits in memory anyway).

    If prebuilt=False (unbuilt Potential), uses the realistic lazy→zarr path:
    build the dask graph lazily, then stream results to a zarr store block by
    block without ever materializing the full result array.

    max_batch controls how many probe positions are processed at once,
    bounding the wave array memory (n_batch * gpts_y * gpts_x * 8 bytes).
    """
    import shutil
    import tempfile

    probe = Probe(energy=energy, semiangle_cutoff=20, device=device)
    detector = AnnularDetector(inner=40, outer=200)

    scan = GridScan(
        start=(0, 0),
        end=potential.extent,
        gpts=scan_gpts,
    )

    n_positions = scan_gpts[0] * scan_gpts[1]
    n_batches = (n_positions + max_batch - 1) // max_batch
    cs_label = "pre-built" if prebuilt else f"chunk={potential_chunk_size}"
    label = (
        f"{cs_label}, scan={scan_gpts}, batch={max_batch}, "
        f"slices={len(potential)}, gpts={potential.gpts}"
    )
    mem_mb = estimate_potential_memory_mb(potential)
    use_zarr = not prebuilt
    zarr_dir = None

    if prebuilt:
        try:
            pot = potential.build(lazy=False)
        except Exception as e:
            return BenchmarkResult(
                label=label,
                device=device,
                gpts=potential.gpts,
                num_slices=len(potential),
                chunk_size="N/A",
                potential_bytes_mb=mem_mb,
                time_seconds=float("nan"),
                extra={"error": f"build failed: {type(e).__name__}: {e}"},
            )
    else:
        pot = potential
        zarr_dir = tempfile.mkdtemp(prefix="abtem_bench_")

    # Warmup
    try:
        if use_zarr:
            zarr_path = os.path.join(zarr_dir, "warmup.zarr")
            lazy_result = probe.scan(
                pot, scan=scan, detectors=detector, lazy=True,
                max_batch=max_batch, potential_chunk_size=potential_chunk_size,
            )
            lazy_result.to_zarr(zarr_path, overwrite=True)
            del lazy_result
        else:
            result = probe.scan(
                pot, scan=scan, detectors=detector, lazy=False,
                max_batch=max_batch, potential_chunk_size=potential_chunk_size,
            )
            del result
    except Exception as e:
        if zarr_dir:
            shutil.rmtree(zarr_dir, ignore_errors=True)
        return BenchmarkResult(
            label=label,
            device=device,
            gpts=potential.gpts,
            num_slices=len(potential),
            chunk_size="pre-built" if prebuilt else potential_chunk_size,
            potential_bytes_mb=mem_mb,
            time_seconds=float("nan"),
            extra={"error": f"{type(e).__name__}: {e}"},
        )

    gc.collect()
    if device == "gpu":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    times = []
    peak_vram_mb = 0.0
    for i in range(n_repeats):
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

        if use_zarr:
            zarr_path = os.path.join(zarr_dir, f"run_{i}.zarr")
            lazy_result = probe.scan(
                pot, scan=scan, detectors=detector, lazy=True,
                max_batch=max_batch, potential_chunk_size=potential_chunk_size,
            )
            lazy_result.to_zarr(zarr_path, overwrite=True)
            del lazy_result
        else:
            result = probe.scan(
                pot, scan=scan, detectors=detector, lazy=False,
                max_batch=max_batch, potential_chunk_size=potential_chunk_size,
            )
            del result

        if device == "gpu":
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

        if tracker:
            tracker.__exit__(None, None, None)
            peak_vram_mb = max(peak_vram_mb, tracker.peak_mb)

        t1 = time.perf_counter()
        times.append(t1 - t0)

        gc.collect()
        if device == "gpu":
            cp.get_default_memory_pool().free_all_blocks()

    if zarr_dir:
        shutil.rmtree(zarr_dir, ignore_errors=True)

    extra = {"all_times": times}
    if peak_vram_mb > 0:
        extra["peak_vram_mb"] = peak_vram_mb

    if prebuilt:
        del pot

    return BenchmarkResult(
        label=label,
        device=device,
        gpts=potential.gpts,
        num_slices=len(potential),
        chunk_size="pre-built" if prebuilt else potential_chunk_size,
        potential_bytes_mb=mem_mb,
        time_seconds=np.median(times),
        extra=extra,
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
            ((512, 512), (2, 2, 10)),
        ]
    else:
        configs = [
            ((2048, 2048), (10, 10, 60)),
        ]
        if device == "gpu":
            configs.extend([
                # These are designed to approach or exceed 24 GB RTX 4090 VRAM.
                # build(lazy=False) should OOM; chunked multislice should succeed.
                ((2048, 2048), (10, 10, 200)),
                ((4096, 4096), (20, 20, 60)),
                ((4096, 4096), (20, 20, 120)),
            ])

    # Chunk sizes to test
    chunk_sizes: list[int | str] = [1, 10, 50, "auto"]

    for gpts, reps in configs:
        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_mb = estimate_potential_memory_mb(potential)
        mem_gb = mem_mb / 1000

        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB potential ──")

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

    # ─── Scan benchmark: pre-built vs unbuilt potential (lazy → zarr) ───
    #
    # The realistic workflow for memory-hungry scans: build the dask graph
    # lazily, then stream results to a zarr store block by block — the full
    # result array never lives in memory.
    #
    # With an unbuilt Potential, chunks are rebuilt from atoms for every
    # scan-position batch. Pre-building once avoids this redundant work
    # (but requires the full potential to fit in memory).
    #
    print(f"\n{'=' * 90}")
    print(f"Scan Benchmark (Probe + GridScan, lazy → zarr) — device={device}")
    print(f"{'=' * 90}")

    # Each config: (gpts, repetitions, scan_gpts, max_batch, description)
    #
    # max_batch controls probe positions per batch, bounding wave memory:
    #   wave_bytes = max_batch * gpts_y * gpts_x * 8 (complex64)
    #   e.g. max_batch=8, gpts=1024: 8 * 1024^2 * 8 = 64 MB
    #
    # More scan positions with small max_batch = more batches = more
    # potential rebuilds for the unbuilt case.
    #
    # Each config: (gpts, repetitions, scan_gpts, max_batch)
    #
    # max_batch controls probe positions per batch, bounding wave memory:
    #   wave_bytes = max_batch * gpts_y * gpts_x * 8 (complex64)
    #
    # n_batches = ceil(n_positions / max_batch) — each batch triggers a
    # full potential rebuild for the unbuilt case.
    #
    if quick:
        scan_configs = [
            ((128, 128), (2, 2, 4), (4, 4), 4),
        ]
    else:
        scan_configs = [
            # Increasing scan positions — shows scaling with rebuild count
            ((512, 512), (2, 2, 10), (16, 16), 8),
            ((512, 512), (2, 2, 10), (32, 32), 8),
            # Larger slices — higher per-rebuild cost
            ((1024, 1024), (2, 2, 10), (16, 16), 8),
        ]
        if device == "gpu":
            scan_configs.extend([
                # Potential exceeds VRAM — pre-build impossible
                ((2048, 2048), (10, 10, 60), (8, 8), 4),
                ((4096, 4096), (20, 20, 60), (4, 4), 1),
            ])

    for gpts, reps, scan_gpts, max_batch in scan_configs:
        # Aggressive cleanup before each config to prevent VRAM leaks
        gc.collect()
        if device == "gpu":
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_mb = estimate_potential_memory_mb(potential)
        mem_gb = mem_mb / 1000
        n_positions = scan_gpts[0] * scan_gpts[1]
        n_batches = (n_positions + max_batch - 1) // max_batch

        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB, "
              f"scan={scan_gpts[0]}x{scan_gpts[1]}, batch={max_batch} ──")
        print(f"  {n_positions} positions, {n_batches} batches")

        if device == "gpu":
            free, total = cp.cuda.Device().mem_info
            print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

        # Pre-built: build once, scan reuses (reference)
        result = benchmark_scan(potential, scan_gpts, device, prebuilt=True,
                                max_batch=max_batch, n_repeats=3)
        print_result(result)

        gc.collect()
        if device == "gpu":
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()

        # Unbuilt: test different potential chunk sizes
        scan_chunk_sizes: list[int | str] = [1, 10, "auto"]
        # Add a larger chunk size if the potential has enough slices
        if num_slices > 50:
            scan_chunk_sizes.insert(2, 50)

        for cs in scan_chunk_sizes:
            result = benchmark_scan(potential, scan_gpts, device, prebuilt=False,
                                    potential_chunk_size=cs,
                                    max_batch=max_batch, n_repeats=3)
            print_result(result)

            gc.collect()
            if device == "gpu":
                cp.cuda.Stream.null.synchronize()
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
