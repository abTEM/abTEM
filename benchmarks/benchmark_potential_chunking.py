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


class GPUUtilTracker:
    """Context manager that samples GPU compute utilization in a background thread.

    Uses the ``pynvml`` API from ``nvidia-ml-py`` (``pip install nvidia-ml-py``).
    If unavailable, all samples are ``None``.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None

    def _sample_loop(self):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.samples.append(util.gpu)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def mean_util(self) -> float | None:
        return (sum(self.samples) / len(self.samples)) if self.samples else None

    @property
    def peak_util(self) -> int | None:
        return max(self.samples) if self.samples else None


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

    label = f"gpts={potential.gpts}, slices={len(potential)}, {_resolve_chunk_label(chunk_size, potential, device)}"

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
    max_batch: int | str = "auto",
    energy: float = 200e3,
    to_zarr: bool = False,
) -> dict:
    """Benchmark a Probe + GridScan measurement using the lazy path."""
    import shutil
    import tempfile

    probe = Probe(energy=energy, semiangle_cutoff=20, device=device)
    detector = AnnularDetector(inner=40, outer=200)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

    # Resolve the effective batch size for display before building the graph.
    # Use the potential's gpts as a proxy for the probe grid size.
    n_pos = scan_gpts[0] * scan_gpts[1]
    if max_batch == "auto":
        if device == "gpu":
            from abtem.core.chunks import estimate_scan_batch_size
            effective_batch = min(
                estimate_scan_batch_size(potential.gpts, np.dtype("complex64"), "gpu"),
                n_pos,
            )
        else:
            from dask.utils import parse_bytes
            chunk_bytes = parse_bytes(config.get("dask.chunk-size"))
            wave_bytes = int(np.prod(potential.gpts)) * np.dtype("complex64").itemsize
            effective_batch = min(max(1, chunk_bytes // wave_bytes), n_pos)
        batch_label = f"auto({effective_batch})"
    else:
        batch_label = str(max_batch)

    if prebuilt:
        cs_label = "pre-built"
    else:
        cs_label = _resolve_chunk_label(potential_chunk_size, potential, device)
    if to_zarr:
        cs_label += " (zarr)"
    label = f"{cs_label}, scan={scan_gpts}, batch={batch_label}"

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

    vram_tracker = PeakVRAMTracker(interval=0.002) if device == "gpu" else None
    util_tracker = GPUUtilTracker(interval=0.05) if device == "gpu" else None
    run_path = os.path.join(zarr_dir, "run.zarr") if zarr_dir else None

    t0 = time.perf_counter()
    if vram_tracker:
        vram_tracker.__enter__()
    if util_tracker:
        util_tracker.__enter__()

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
        if vram_tracker:
            vram_tracker.__exit__(None, None, None)
        if util_tracker:
            util_tracker.__exit__(None, None, None)
        e.__traceback__ = None
        _gpu_cleanup()
        if zarr_dir:
            shutil.rmtree(zarr_dir, ignore_errors=True)
        return {"label": label, "error": f"{type(e).__name__}: {e}"}

    if device == "gpu":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    if vram_tracker:
        vram_tracker.__exit__(None, None, None)
    if util_tracker:
        util_tracker.__exit__(None, None, None)

    elapsed = time.perf_counter() - t0

    if prebuilt:
        del pot
    _gpu_cleanup()
    if zarr_dir:
        shutil.rmtree(zarr_dir, ignore_errors=True)

    out = {"label": label, "time": elapsed}
    if vram_tracker:
        out["peak_vram_mb"] = vram_tracker.peak_mb
    if util_tracker:
        out["mean_gpu_util"] = util_tracker.mean_util
    return out


# ──────────────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────

def _resolve_chunk_label(chunk_size: int | str, potential, device: str) -> str:
    """Return a display label for the chunk size, resolving 'auto' to 'auto(NxM)'
    where N is the number of chunks and M is the (maximum) slices per chunk."""
    import math
    from abtem.core.chunks import estimate_potential_chunk_size
    n_slices = len(potential)
    if chunk_size == "auto":
        budget = min(estimate_potential_chunk_size(potential.gpts, device), n_slices)
        n_chunks = math.ceil(n_slices / budget)
        actual_cs = math.ceil(n_slices / n_chunks)
        return f"chunk=auto({n_chunks}x{actual_cs})"
    return f"chunk={chunk_size}"


def print_result(r: dict):
    if "error" in r:
        print(f"  {r['label']:<60s}  ERROR: {r['error']}")
    else:
        vram = r.get("peak_vram_mb")
        util = r.get("mean_gpu_util")
        vram_str = f"  vram={vram / 1000:5.1f}GB" if vram else ""
        util_str = f"  gpu={util:3.0f}%" if util is not None else ""
        print(f"  {r['label']:<60s}  {r['time']:8.3f}s{vram_str}{util_str}")


# ──────────────────────────────────────────────────────────────────────
# Configuration lists
# ──────────────────────────────────────────────────────────────────────

def get_planewave_configs(device: str, quick: bool):
    if quick:
        return [((512, 512), (2, 2, 10))]
    configs = []
    if device == "gpu":
        configs.extend([
            # ~3 GB — small, fits easily
            ((2048, 2048), (10, 10, 60)),
            # ~14 GB — tight fit
            ((4096, 4096), (20, 20, 75)),
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
    """Return list of (gpts, repetitions, scan_gpts) tuples.

    Batch size is determined automatically at runtime via
    ``estimate_scan_batch_size`` so that available VRAM is fully exploited.
    """
    if quick:
        return [((128, 128), (2, 2, 4), (4, 4))]
    configs = []
    if device == "gpu":
        configs.extend([
            # ~3 GB potential — fits in VRAM, shows pre-built advantage
            ((2048, 2048), (10, 10, 60), (8, 8)),
            # ~14 GB potential — tight fit, pre-built OOMs
            ((4096, 4096), (20, 20, 75), (8, 8)),
            # ~22 GB potential — exceeds VRAM, must chunk
            ((4096, 4096), (20, 20, 120), (8, 8)),
            # ~36 GB potential — well exceeds VRAM, heavy chunking required
            ((4096, 4096), (20, 20, 200), (8, 8)),
        ])
    else:
        configs.extend([
            # ~0.7 GB potential, 64 positions
            ((1024, 1024), (4, 4, 20), (8, 8)),
            # ~0.7 GB potential, 256 positions — larger scan to stress potential-first
            ((1024, 1024), (4, 4, 20), (16, 16)),
        ])
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

    for gpts, reps in get_planewave_configs(device, quick):
        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_gb = estimate_potential_memory_mb(potential) / 1000

        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB potential ──")

        if device == "gpu":
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

        print_result(benchmark_multislice(potential, "auto", device))
        _gpu_cleanup()


# ──────────────────────────────────────────────────────────────────────
# Scan benchmarks — each config runs in a SEPARATE SUBPROCESS
# ──────────────────────────────────────────────────────────────────────

def run_single_scan_benchmark(
    device: str, config_index: int, quick: bool,
    prebuilt: bool, chunk_size: int | str, to_zarr: bool,
    batch_size: int | str = "auto",
):
    """Run ONE scan benchmark (called in a subprocess). Prints one result line."""
    config.set({"dask.lazy": False})

    scan_configs = get_scan_configs(device, quick)
    gpts, reps, scan_gpts = scan_configs[config_index]

    if device == "gpu":
        warmup_gpu()

    potential = make_large_potential(gpts, reps, device=device)

    print_result(benchmark_scan(
        potential, scan_gpts, device,
        prebuilt=prebuilt,
        potential_chunk_size=chunk_size,
        max_batch=batch_size,
        to_zarr=to_zarr,
    ))
    sys.stdout.flush()


def _run_scan_subprocess(
    script: str, device: str, config_index: int, quick: bool,
    prebuilt: bool, chunk_size: int | str, to_zarr: bool,
    batch_size: int | str = "auto",
):
    """Spawn a fresh subprocess for one scan benchmark and print its output."""
    cmd = [
        sys.executable, script,
        "--device", device,
        "--_run-scan-bench", str(config_index),
        "--_prebuilt" if prebuilt else "--_no-prebuilt",
        "--_chunk-size", str(chunk_size),
        "--_batch-size", str(batch_size),
    ]
    if to_zarr:
        cmd.append("--_to-zarr")
    if quick:
        cmd.append("--quick")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )

    if result.stdout:
        print(result.stdout, end="")
    elif result.returncode != 0:
        # Process died without printing anything (e.g. OOM killed by kernel
        # before Python exception handling could run).
        label = f"chunk={chunk_size}, batch={batch_size}"
        print(f"  {label:<60s}  KILLED (OOM or crash, exit {result.returncode})")

    if result.returncode != 0 and result.stderr:
        lines = result.stderr.strip().split("\n")
        for line in lines[-3:]:
            print(f"  stderr: {line}")


def run_scan_benchmarks_via_subprocesses(device: str, quick: bool = False):
    """Spawn a fresh subprocess for EACH scan benchmark."""
    print(f"\n{'=' * 90}")
    print(f"Scan Benchmark (Probe + GridScan, lazy path) — device={device}")
    print(f"Each benchmark runs in a fresh subprocess for clean VRAM.")
    print(f"{'=' * 90}")

    scan_configs = get_scan_configs(device, quick)
    script = os.path.abspath(__file__)

    for i, (gpts, reps, scan_gpts) in enumerate(scan_configs):
        potential = make_large_potential(gpts, reps, device=device)
        num_slices = len(potential)
        mem_gb = estimate_potential_memory_mb(potential) / 1000
        del potential

        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB, "
              f"scan={scan_gpts[0]}x{scan_gpts[1]} ──")

        _run_scan_subprocess(script, device, i, quick,
                             prebuilt=False, chunk_size="auto", to_zarr=False,
                             batch_size="auto")

    print(f"\n{'=' * 90}")
    print("Done.")


# ──────────────────────────────────────────────────────────────────────
# Single-slice stress test
# ──────────────────────────────────────────────────────────────────────

def run_single_slice_stress_test(device: str, quick: bool = False):
    """Stress-test where a single potential slice barely fits in device memory.

    A 16384² grid produces ~1.07 GB per float32 slice.  On a 25 GB GPU the
    VRAM budget for potential chunks is ~6.2 GB, and with 5× fragmentation
    overhead ``estimate_potential_chunk_size`` returns 1 — so the auto path
    naturally processes one slice at a time without any manual override.

    Note: at 16384² the wavefunction FFT workspace alone consumes ~24 GB,
    leaving no room for the potential slice.  This test is therefore expected
    to OOM — it documents the upper limit of what fits on a 25 GB GPU rather
    than asserting success.

    Only meaningful on GPU; on CPU the potential is a single chunk by default.
    """
    print(f"\n{'=' * 90}")
    print(f"Single-slice stress test — device={device}")
    print(f"{'=' * 90}")

    if quick:
        # 4096² still gives auto chunk_size ~18, not 1 — use explicit override
        # in quick mode just to exercise the code path quickly
        gpts, reps = (4096, 4096), (1, 1, 4)
        scan_gpts = (2, 2)
    else:
        # 16384²: one slice ≈ 1.07 GB → auto chunk_size = 1 on a 25 GB GPU
        gpts, reps = (16384, 16384), (1, 1, 5)
        scan_gpts = (2, 2)  # 4 positions → 1 scan batch, potential built once

    potential = make_large_potential(gpts, reps, device=device)
    num_slices = len(potential)
    mem_gb = estimate_potential_memory_mb(potential) / 1000

    if device == "gpu":
        import cupy as cp
        free, total = cp.cuda.Device().mem_info
        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB potential ──")
        print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    else:
        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB potential ──")

    print("\n  [PlaneWave multislice]")
    print_result(benchmark_multislice(potential, chunk_size="auto", device=device))
    _gpu_cleanup()

    print("\n  [Probe scan (2×2 positions)]")
    # Run in-process: 2×2 is tiny, and _gpu_cleanup() above cleared any
    # residual allocations from the PlaneWave run.
    print_result(benchmark_scan(
        potential, scan_gpts, device,
        prebuilt=False,
        potential_chunk_size="auto",
        max_batch="auto",
    ))
    _gpu_cleanup()

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
    # Internal: used by subprocesses to run a single scan benchmark
    parser.add_argument("--_run-scan-bench", type=int, default=None)
    parser.add_argument("--_prebuilt", action="store_true", default=False)
    parser.add_argument("--_no-prebuilt", action="store_true", default=False)
    parser.add_argument("--_chunk-size", type=str, default="auto")
    parser.add_argument("--_batch-size", type=str, default="auto")
    parser.add_argument("--_to-zarr", action="store_true", default=False)

    args = parser.parse_args()

    config.set({"dask.lazy": False})

    # If we're a subprocess running a single scan benchmark
    if args._run_scan_bench is not None:
        cs: int | str = args._chunk_size
        try:
            cs = int(cs)
        except ValueError:
            pass  # keep as string ("auto")
        bs: int | str = args._batch_size
        try:
            bs = int(bs)
        except ValueError:
            pass  # keep as string ("auto")
        run_single_scan_benchmark(
            args.device, args._run_scan_bench, args.quick,
            prebuilt=args._prebuilt, chunk_size=cs, to_zarr=args._to_zarr,
            batch_size=bs,
        )
        return

    # Otherwise, run the full benchmark suite
    if args.device in ("cpu", "both"):
        run_planewave_benchmarks("cpu", quick=args.quick)
        run_scan_benchmarks_via_subprocesses("cpu", quick=args.quick)

    if args.device in ("gpu", "both"):
        run_planewave_benchmarks("gpu", quick=args.quick)
        run_scan_benchmarks_via_subprocesses("gpu", quick=args.quick)
        run_single_slice_stress_test("gpu", quick=args.quick)


if __name__ == "__main__":
    main()
