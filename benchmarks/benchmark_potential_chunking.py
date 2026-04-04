"""
Benchmark: Potential chunking performance for large multislice simulations.

Designed to run on a workstation with an NVIDIA RTX 4090 (24 GB VRAM).
Tests both CPU and GPU with various potential sizes and chunk strategies.

Each scan configuration runs in a **separate subprocess** so that OOM
errors in one config don't pollute VRAM for subsequent configs.

Usage:
    python benchmarks/benchmark_potential_chunking.py
    python benchmarks/benchmark_potential_chunking.py --device gpu
    python benchmarks/benchmark_potential_chunking.py --device cpu
    python benchmarks/benchmark_potential_chunking.py --device gpu --quick
"""

import argparse
import gc
import json
import os
import subprocess
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

import numpy as np
from ase.build import bulk

import abtem
from abtem import AnnularDetector, PlaneWave, Potential, Probe
from abtem.core import config
from abtem.scan import GridScan


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_large_potential(
    gpts: tuple[int, int],
    repetitions: tuple[int, int, int],
    device: str = "cpu",
    slice_thickness: float = 2.0,
) -> Potential:
    """Create a large silicon potential for benchmarking."""
    atoms = bulk("Si", cubic=True) * repetitions
    return Potential(
        atoms, gpts=gpts, slice_thickness=slice_thickness, device=device,
    )


def estimate_potential_memory_mb(potential: Potential) -> float:
    """Estimate total potential memory in MB (float32)."""
    return (
        len(potential) * potential.gpts[0] * potential.gpts[1]
        * np.dtype(np.float32).itemsize / 1e6
    )


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
    """Context manager that samples GPU memory in a background thread."""

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


def _gpu_cleanup():
    gc.collect()
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Benchmark functions
# ──────────────────────────────────────────────────────────────────────

def benchmark_multislice(
    potential: Potential,
    chunk_size: int | str,
    device: str,
    energy: float = 200e3,
) -> dict:
    """Benchmark a single multislice run with given chunk size."""
    waves = PlaneWave(energy=energy, gpts=potential.gpts, device=device)
    waves.grid.match(potential)

    label = f"gpts={potential.gpts}, slices={len(potential)}, chunk={chunk_size}"

    gc.collect()
    if device == "gpu":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    tracker = PeakVRAMTracker(interval=0.002) if device == "gpu" else None

    t0 = time.perf_counter()
    if tracker:
        tracker.__enter__()

    try:
        result = waves.multislice(
            potential, potential_chunk_size=chunk_size, lazy=False
        )
    except Exception as e:
        if tracker:
            tracker.__exit__(None, None, None)
        msg = f"{type(e).__name__}: {e}"
        e.__traceback__ = None
        del e
        _gpu_cleanup()
        return {"label": label, "error": msg}

    if device == "gpu":
        cp.cuda.Stream.null.synchronize()
    if tracker:
        tracker.__exit__(None, None, None)

    elapsed = time.perf_counter() - t0
    del result
    _gpu_cleanup()

    out = {"label": label, "time": elapsed}
    if tracker:
        out["peak_vram_mb"] = tracker.peak_mb
    return out


def benchmark_build_only(
    potential: Potential,
    chunk_size: int | str,
    device: str,
) -> dict:
    """Benchmark just the potential building (no multislice propagation)."""
    label = f"build-only gpts={potential.gpts}, slices={len(potential)}, chunk={chunk_size}"

    gc.collect()
    if device == "gpu":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()

    t0 = time.perf_counter()
    for chunk in potential.generate_chunked_slices(chunk_size=chunk_size):
        pass
    elapsed = time.perf_counter() - t0

    _gpu_cleanup()
    return {"label": label, "time": elapsed}


def benchmark_scan(
    potential: Potential,
    scan_gpts: tuple[int, int],
    device: str,
    prebuilt: bool,
    potential_chunk_size: int | str = "auto",
    max_batch: int = 8,
    energy: float = 200e3,
    to_zarr: bool = False,
) -> dict:
    """Benchmark a Probe + GridScan measurement using the lazy path."""
    import shutil
    import tempfile

    probe = Probe(energy=energy, semiangle_cutoff=20, device=device)
    detector = AnnularDetector(inner=40, outer=200)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

    if prebuilt:
        cs_label = "pre-built"
    else:
        cs_label = f"chunk={potential_chunk_size}"
    if to_zarr:
        cs_label += " (zarr)"
    label = f"{cs_label}, scan={scan_gpts}, batch={max_batch}"

    if prebuilt:
        try:
            pot = potential.build(lazy=False)
        except Exception as e:
            e.__traceback__ = None
            _gpu_cleanup()
            return {"label": label, "error": f"build failed: {type(e).__name__}: {e}"}
    else:
        pot = potential

    zarr_dir = tempfile.mkdtemp(prefix="abtem_bench_") if to_zarr else None

    _gpu_cleanup()

    tracker = PeakVRAMTracker(interval=0.002) if device == "gpu" else None
    run_path = os.path.join(zarr_dir, "run.zarr") if zarr_dir else None

    t0 = time.perf_counter()
    if tracker:
        tracker.__enter__()

    try:
        lazy_result = probe.scan(
            pot, scan=scan, detectors=detector, lazy=True,
            max_batch=max_batch, potential_chunk_size=potential_chunk_size,
        )
        if to_zarr:
            lazy_result.to_zarr(run_path, overwrite=True)
        else:
            lazy_result.compute()
        del lazy_result
    except Exception as e:
        if tracker:
            tracker.__exit__(None, None, None)
        e.__traceback__ = None
        _gpu_cleanup()
        if zarr_dir:
            shutil.rmtree(zarr_dir, ignore_errors=True)
        return {"label": label, "error": f"{type(e).__name__}: {e}"}

    if device == "gpu":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    if tracker:
        tracker.__exit__(None, None, None)

    elapsed = time.perf_counter() - t0

    if prebuilt:
        del pot
    _gpu_cleanup()
    if zarr_dir:
        shutil.rmtree(zarr_dir, ignore_errors=True)

    out = {"label": label, "time": elapsed}
    if tracker:
        out["peak_vram_mb"] = tracker.peak_mb
    return out


# ──────────────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────

def print_result(r: dict):
    if "error" in r:
        print(f"  {r['label']:<55s}  ERROR: {r['error']}")
    else:
        vram = r.get("peak_vram_mb")
        vram_str = f"  vram={vram / 1000:5.1f}GB" if vram else ""
        print(f"  {r['label']:<55s}  {r['time']:8.3f}s{vram_str}")


# ──────────────────────────────────────────────────────────────────────
# Configuration lists
# ──────────────────────────────────────────────────────────────────────

def get_planewave_configs(device: str, quick: bool):
    if quick:
        return [((512, 512), (2, 2, 10))]
    configs = []
    if device == "gpu":
        configs.extend([
            # ~22 GB — just fits in 24 GB VRAM
            ((4096, 4096), (20, 20, 120)),
            # ~36 GB — exceeds 24 GB VRAM; build(lazy=False) OOMs,
            # chunked multislice succeeds.
            ((4096, 4096), (20, 20, 200)),
        ])
    else:
        configs.append(((2048, 2048), (10, 10, 60)))
    return configs


def get_scan_configs(device: str, quick: bool):
    """Return list of (gpts, repetitions, scan_gpts, max_batch)."""
    if quick:
        return [((128, 128), (2, 2, 4), (4, 4), 4)]
    configs = []
    if device == "gpu":
        configs.extend([
            # ~11 GB potential — fits in VRAM, pre-built should work
            ((2048, 2048), (10, 10, 60), (4, 4), 4),
            # ~22 GB potential — tight fit, pre-built may OOM
            ((4096, 4096), (20, 20, 120), (4, 4), 4),
            # ~36 GB potential — exceeds VRAM, must chunk
            ((4096, 4096), (20, 20, 200), (4, 4), 2),
        ])
    else:
        configs.append(((1024, 1024), (2, 2, 10), (8, 8), 4))
    return configs


# ──────────────────────────────────────────────────────────────────────
# Planewave benchmarks (run in-process — OOMs are less sticky here)
# ──────────────────────────────────────────────────────────────────────

def run_planewave_benchmarks(device: str, quick: bool = False):
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

    chunk_sizes: list[int | str] = [1, 10, 50, "auto"]

    for gpts, reps in get_planewave_configs(device, quick):
        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_gb = estimate_potential_memory_mb(potential) / 1000

        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB potential ──")

        if device == "gpu":
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

        # Full build reference
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
            print(f"  build(lazy=False) FAILED ({type(e).__name__}): {e}")
            print(f"  >>> This is expected — potential ({mem_gb:.1f} GB) exceeds available memory.")
            e.__traceback__ = None
            del e
        _gpu_cleanup()

        # Build only
        print("\n  [Potential build (generate_chunked_slices)]")
        for cs in chunk_sizes:
            if isinstance(cs, int) and cs > num_slices:
                continue
            print_result(benchmark_build_only(potential, cs, device))

        # Full multislice
        print("\n  [Full multislice (PlaneWave, 200 keV)]")
        if full_build_ok:
            try:
                prebuilt = potential.build(lazy=False)
                r = benchmark_multislice(prebuilt, "auto", device)
                r["label"] = "pre-built PotentialArray, chunk=auto"
                print_result(r)
                del prebuilt
            except (MemoryError, Exception) as e:
                print(f"  pre-built PotentialArray: OOM ({type(e).__name__})")
        else:
            print(f"  pre-built PotentialArray: SKIPPED (full build failed above)")
        _gpu_cleanup()

        for cs in chunk_sizes:
            if isinstance(cs, int) and cs > num_slices:
                continue
            print_result(benchmark_multislice(potential, cs, device))
        _gpu_cleanup()


# ──────────────────────────────────────────────────────────────────────
# Scan benchmarks — each config runs in a SEPARATE SUBPROCESS
# ──────────────────────────────────────────────────────────────────────

def run_single_scan_config(device: str, config_index: int, quick: bool):
    """Run one scan config (called in a subprocess). Prints results to stdout."""
    config.set({"dask.lazy": False})

    scan_configs = get_scan_configs(device, quick)
    gpts, reps, scan_gpts, max_batch = scan_configs[config_index]

    if device == "gpu":
        warmup_gpu()

    potential = make_large_potential(gpts, reps, device=device)
    num_slices = len(potential)
    mem_gb = estimate_potential_memory_mb(potential) / 1000
    n_positions = scan_gpts[0] * scan_gpts[1]
    n_batches = (n_positions + max_batch - 1) // max_batch

    print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB, "
          f"scan={scan_gpts[0]}x{scan_gpts[1]}, batch={max_batch} ──")
    print(f"  {n_positions} positions, {n_batches} batches")

    if device == "gpu":
        import cupy as cp
        free, total = cp.cuda.Device().mem_info
        print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    scan_chunk_sizes: list[int | str] = [1, 10, "auto"]

    # --- lazy → .compute() ---
    print("  [lazy → compute]")

    print_result(benchmark_scan(
        potential, scan_gpts, device, prebuilt=True, max_batch=max_batch))
    _gpu_cleanup()

    for cs in scan_chunk_sizes:
        print_result(benchmark_scan(
            potential, scan_gpts, device, prebuilt=False,
            potential_chunk_size=cs, max_batch=max_batch))
        _gpu_cleanup()

    # --- lazy → zarr ---
    print("  [lazy → zarr]")

    for cs in scan_chunk_sizes:
        print_result(benchmark_scan(
            potential, scan_gpts, device, prebuilt=False,
            potential_chunk_size=cs, to_zarr=True, max_batch=max_batch))
        _gpu_cleanup()

    sys.stdout.flush()


def run_scan_benchmarks_via_subprocesses(device: str, quick: bool = False):
    """Spawn a fresh subprocess for each scan config."""
    print(f"\n{'=' * 90}")
    print(f"Scan Benchmark (Probe + GridScan, lazy path) — device={device}")
    print(f"Each config runs in a fresh subprocess for clean VRAM.")
    print(f"{'=' * 90}")

    scan_configs = get_scan_configs(device, quick)
    script = os.path.abspath(__file__)

    for i, (gpts, reps, scan_gpts, max_batch) in enumerate(scan_configs):
        cmd = [
            sys.executable, script,
            "--device", device,
            "--_run-scan-config", str(i),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "TQDM_DISABLE": "1"},
        )

        # Print subprocess stdout directly
        if result.stdout:
            print(result.stdout, end="")

        if result.returncode != 0:
            print(f"  [subprocess exited with code {result.returncode}]")
            if result.stderr:
                # Show last few lines of stderr (skip tracebacks)
                lines = result.stderr.strip().split("\n")
                for line in lines[-5:]:
                    print(f"  stderr: {line}")

    print(f"\n{'=' * 90}")
    print("Done.")


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark potential chunking for abTEM multislice"
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu", "both"], default="both",
        help="Device to benchmark (default: both)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick subset of benchmarks",
    )
    # Internal: used by subprocesses to run a single scan config
    parser.add_argument("--_run-scan-config", type=int, default=None)

    args = parser.parse_args()

    config.set({"dask.lazy": False})

    # If we're a subprocess running a single scan config, do just that and exit
    if args._run_scan_config is not None:
        run_single_scan_config(args.device, args._run_scan_config, args.quick)
        return

    # Otherwise, run the full benchmark suite
    if args.device in ("cpu", "both"):
        run_planewave_benchmarks("cpu", quick=args.quick)
        run_scan_benchmarks_via_subprocesses("cpu", quick=args.quick)

    if args.device in ("gpu", "both"):
        run_planewave_benchmarks("gpu", quick=args.quick)
        run_scan_benchmarks_via_subprocesses("gpu", quick=args.quick)


if __name__ == "__main__":
    main()
