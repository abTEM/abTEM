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
    python benchmarks/benchmark_potential_chunking.py --stress-only
    python benchmarks/benchmark_potential_chunking.py --stress-only --quick
"""

import argparse
import gc
import json
import os
import re
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

# Accumulates all benchmark results for optional JSON output (--output-json).
_collected_results: list[dict] = []
from abtem import AnnularDetector, PlaneWave, Potential, Probe
from abtem.core import config
from abtem.potentials.iam import CrystalPotential
from abtem.scan import GridScan


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_large_potential(
    gpts: tuple[int, int],
    repetitions: tuple[int, int, int],
    device: str = "cpu",
    slice_thickness: float = 2.0,
    projection: str = "infinite",
) -> Potential:
    """Create a large silicon potential for benchmarking."""
    atoms = bulk("Si", cubic=True) * repetitions
    return Potential(
        atoms, gpts=gpts, slice_thickness=slice_thickness, device=device,
        projection=projection,
    )


def _nearest_divisor(n: int, target: int) -> int:
    """Return the largest divisor of *n* that is <= *target*."""
    best = 1
    for d in range(1, target + 1):
        if n % d == 0:
            best = d
    return best


def make_crystal_potential(
    gpts: tuple[int, int],
    repetitions: tuple[int, int, int],
    device: str = "cpu",
    slice_thickness: float = 2.0,
    projection: str = "infinite",
) -> CrystalPotential:
    """Create a CrystalPotential by tiling a single Si unit cell.

    The xy repetitions are adjusted to the nearest divisor of *gpts* so that
    the unit-cell gpts are integers.  The z repetitions are kept as-is.
    """
    rx = _nearest_divisor(gpts[0], repetitions[0])
    ry = _nearest_divisor(gpts[1], repetitions[1])
    reps = (rx, ry, repetitions[2])
    unit_gpts = (gpts[0] // rx, gpts[1] // ry)

    atoms = bulk("Si", cubic=True)
    unit = Potential(
        atoms, gpts=unit_gpts, slice_thickness=slice_thickness,
        device=device, projection=projection,
    )
    return CrystalPotential(unit, repetitions=reps)


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
    projection: str = "infinite",
    precision: str = "float32",
) -> dict:
    """Benchmark a single multislice run with given chunk size."""
    prec_label = "f64" if precision == "float64" else "f32"
    waves = PlaneWave(energy=energy, gpts=potential.gpts, device=device)
    waves.grid.match(potential)

    label = (
        f"gpts={potential.gpts}, slices={len(potential)}, "
        f"{_resolve_chunk_label(chunk_size, potential, device)}, "
        f"proj={projection}, prec={prec_label}"
    )

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
        with config.set({"precision": precision}):
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
    projection: str = "infinite",
    precision: str = "float32",
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
    prec_label = "f64" if precision == "float64" else "f32"
    label = f"{cs_label}, scan={scan_gpts}, batch={batch_label}, proj={projection}, prec={prec_label}"

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
        with config.set({"precision": precision}):
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


def _format_error(msg: str) -> str:
    """Shorten error messages for display; keep OOM size info concise."""
    if "Out of memory" in msg or "OutOfMemoryError" in msg:
        sizes = re.findall(r"[\d,]+(?= bytes)", msg)
        if len(sizes) >= 2:
            tried_gb = int(sizes[0].replace(",", "")) / 1e9
            used_gb  = int(sizes[1].replace(",", "")) / 1e9
            return f"OOM (tried {tried_gb:.1f} GB, {used_gb:.1f} GB in use)"
    # Fallback: first line, at most 100 chars
    return msg.split("\n")[0][:100]


def _parse_result_line(line: str) -> dict | None:
    """Parse a single ``print_result`` output line back into a result dict.

    Used by subprocess wrappers to capture results into ``_collected_results``
    without requiring a separate IPC channel.
    """
    line = line.strip()
    if not line:
        return None
    # Success: "  {label:<72}  {time:8.3f}s[  vram={gb:5.1f}GB][  gpu={util:3.0f}%]"
    m = re.match(
        r"^(.+?)\s{2,}([\d.]+)s(?:\s+vram=([ \d.]+)GB)?(?:\s+gpu=\s*(\d+)%)?$", line
    )
    if m:
        r: dict = {"label": m.group(1).strip(), "time": float(m.group(2))}
        if m.group(3):
            r["peak_vram_mb"] = float(m.group(3).strip()) * 1000
        if m.group(4):
            r["mean_gpu_util"] = float(m.group(4))
        return r
    # Error
    m = re.match(r"^(.+?)\s{2,}ERROR:\s*(.+)$", line)
    if m:
        return {"label": m.group(1).strip(), "error": m.group(2).strip()}
    # Killed
    m = re.match(r"^(.+?)\s{2,}KILLED\s+(.+)$", line)
    if m:
        return {"label": m.group(1).strip(), "error": f"KILLED {m.group(2).strip()}"}
    return None


def _stress_gpts(stress_size: str, quick: bool) -> tuple[int, int]:
    """Return the grid size used by a stress-test subprocess."""
    if stress_size == "medium":
        return (2048, 2048) if quick else (8192, 8192)
    else:  # "large"
        return (4096, 4096) if quick else (16384, 16384)


def _classify_subprocess_error(
    stderr: str,
    returncode: int,
    mem_hint: tuple[float, float] | None = None,
) -> str:
    """Turn subprocess stderr + exit code into a concise one-line error string.

    Recognises common CUDA faults and OOM patterns so callers don't need to
    show a separate ``stderr:`` line for every known error type.

    *mem_hint* is an optional ``(tried_gb, in_use_gb)`` pair that is appended
    to OOM messages when available, matching the format produced by Python-level
    ``OutOfMemoryError`` messages.
    """
    if not stderr:
        return f"KILLED (exit {returncode})"

    # Hard GPU faults — CUDA error codes embedded in the traceback.
    # Both ILLEGAL_ADDRESS and OUT_OF_MEMORY are treated as OOM: in a
    # memory-pressure context ILLEGAL_ADDRESS arises when an allocation
    # partially succeeds and a kernel then touches the guard region.
    m = re.search(r"(CUDA_ERROR_\w+)", stderr)
    if m:
        code = m.group(1)
        if code in ("CUDA_ERROR_ILLEGAL_ADDRESS", "CUDA_ERROR_OUT_OF_MEMORY"):
            if mem_hint:
                tried_gb, in_use_gb = mem_hint
                return f"OOM (tried ~{tried_gb:.1f} GB, ~{in_use_gb:.1f} GB in use)"
            return "OOM"
        return f"CUDA error: {code}"

    # Python-level OOM (cupy / torch style)
    if "OutOfMemoryError" in stderr or "Out of memory" in stderr:
        return _format_error(stderr)

    # Generic fallback
    for line in stderr.splitlines():
        line = line.strip()
        if "Error" in line or "error" in line:
            return line[:100]
    return f"KILLED (exit {returncode})"


def print_result(r: dict):
    _collected_results.append(r)
    if "error" in r:
        print(f"  {r['label']:<72s}  ERROR: {_format_error(r['error'])}")
    else:
        vram = r.get("peak_vram_mb")
        util = r.get("mean_gpu_util")
        vram_str = f"  vram={vram / 1000:5.1f}GB" if vram else ""
        util_str = f"  gpu={util:3.0f}%" if util is not None else ""
        print(f"  {r['label']:<72s}  {r['time']:8.3f}s{vram_str}{util_str}")


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
    print(f"PlaneWave Benchmark (chunked multislice) — device={device}")
    print(f"{'=' * 90}")

    for gpts, reps in get_planewave_configs(device, quick):
        potential_inf = make_large_potential(gpts, reps, device=device, projection="infinite")
        num_slices = len(potential_inf)
        mem_gb = estimate_potential_memory_mb(potential_inf) / 1000

        print(f"\n── {gpts[0]}x{gpts[1]}, {num_slices} slices, {mem_gb:.2f} GB potential ──")

        if device == "gpu":
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

        crystal_inf = make_crystal_potential(gpts, reps, device=device, projection="infinite")
        creps = (crystal_inf.repetitions[0], crystal_inf.repetitions[1])
        unit_g = crystal_inf.potential_unit.gpts
        print(f"  [CrystalPotential: {creps[0]}×{creps[1]} tiles of {unit_g[0]}×{unit_g[1]}]")
        for prec in ("float32", "float64"):
            print_result(benchmark_multislice(crystal_inf, "auto", device,
                                              projection="crystal(inf)", precision=prec))
            _gpu_cleanup()

        crystal_fin = make_crystal_potential(gpts, reps, device=device, projection="finite")
        for prec in ("float32", "float64"):
            print_result(benchmark_multislice(crystal_fin, "auto", device,
                                              projection="crystal(fin)", precision=prec))
            _gpu_cleanup()

        for prec in ("float32", "float64"):
            print_result(benchmark_multislice(potential_inf, "auto", device,
                                              projection="infinite", precision=prec))
            _gpu_cleanup()

        potential_fin = make_large_potential(gpts, reps, device=device, projection="finite")
        for prec in ("float32", "float64"):
            print_result(benchmark_multislice(potential_fin, "auto", device,
                                              projection="finite", precision=prec))
            _gpu_cleanup()


# ──────────────────────────────────────────────────────────────────────
# Scan benchmarks — each config runs in a SEPARATE SUBPROCESS
# ──────────────────────────────────────────────────────────────────────

def run_single_scan_benchmark(
    device: str, config_index: int, quick: bool,
    prebuilt: bool, chunk_size: int | str, to_zarr: bool,
    batch_size: int | str = "auto",
    projection: str = "infinite",
    use_crystal: bool = False,
    precision: str = "float32",
):
    """Run ONE scan benchmark (called in a subprocess). Prints one result line."""
    config.set({"dask.lazy": False})

    scan_configs = get_scan_configs(device, quick)
    gpts, reps, scan_gpts = scan_configs[config_index]

    if device == "gpu":
        warmup_gpu()

    if use_crystal:
        potential = make_crystal_potential(gpts, reps, device=device, projection=projection)
        projection = f"crystal({projection[:3]})"
    else:
        potential = make_large_potential(gpts, reps, device=device, projection=projection)

    print_result(benchmark_scan(
        potential, scan_gpts, device,
        prebuilt=prebuilt,
        potential_chunk_size=chunk_size,
        max_batch=batch_size,
        to_zarr=to_zarr,
        projection=projection,
        precision=precision,
    ))
    sys.stdout.flush()


def _run_scan_subprocess(
    script: str, device: str, config_index: int, quick: bool,
    prebuilt: bool, chunk_size: int | str, to_zarr: bool,
    batch_size: int | str = "auto",
    projection: str = "infinite",
    use_crystal: bool = False,
    precision: str = "float32",
):
    """Spawn a fresh subprocess for one scan benchmark and print its output."""
    cmd = [
        sys.executable, script,
        "--device", device,
        "--_run-scan-bench", str(config_index),
        "--_prebuilt" if prebuilt else "--_no-prebuilt",
        "--_chunk-size", str(chunk_size),
        "--_batch-size", str(batch_size),
        "--_projection", projection,
        "--_precision", precision,
    ]
    if to_zarr:
        cmd.append("--_to-zarr")
    if use_crystal:
        cmd.append("--_use-crystal")
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
        for line in result.stdout.splitlines():
            r = _parse_result_line(line)
            if r is not None:
                _collected_results.append(r)
        # Partial run — subprocess printed something but still crashed.
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.splitlines():
                line = line.strip()
                if "Error" in line or "error" in line:
                    print(f"  stderr: {line}")
                    break
    elif result.returncode != 0:
        # Process died before printing anything — classify the error from stderr.
        prec_label = "f64" if precision == "float64" else "f32"
        proj_label = f"crystal({projection[:3]})" if use_crystal else projection
        label = f"chunk={chunk_size}, batch={batch_size}, proj={proj_label}, prec={prec_label}"
        error_msg = _classify_subprocess_error(result.stderr, result.returncode)
        _collected_results.append({"label": label, "error": error_msg})
        print(f"  {label:<72s}  ERROR: {error_msg}")


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

        for projection in ("infinite", "finite"):
            for prec in ("float32", "float64"):
                _run_scan_subprocess(script, device, i, quick,
                                     prebuilt=False, chunk_size="auto", to_zarr=False,
                                     batch_size="auto", projection=projection,
                                     use_crystal=True, precision=prec)
            for prec in ("float32", "float64"):
                _run_scan_subprocess(script, device, i, quick,
                                     prebuilt=False, chunk_size="auto", to_zarr=False,
                                     batch_size="auto", projection=projection,
                                     precision=prec)



# ──────────────────────────────────────────────────────────────────────
# Single-slice stress test
# ──────────────────────────────────────────────────────────────────────

def run_stress_scan_subprocess(
    script: str, device: str, batch: int | str, quick: bool,
    projection: str = "infinite",
    use_crystal: bool = False,
    precision: str = "float32",
    stress_size: str = "large",
):
    """Spawn a fresh subprocess for one stress-test scan attempt.

    A hard GPU fault (CUDA_ERROR_ILLEGAL_ADDRESS) corrupts the CUDA context
    and cannot be recovered in-process, so each attempt runs isolated.
    """
    cmd = [
        sys.executable, script,
        "--device", device,
        "--_run-stress-scan", str(batch),
        "--_projection", projection,
        "--_precision", precision,
        "--_stress-size", stress_size,
    ]
    if use_crystal:
        cmd.append("--_use-crystal")
    if quick:
        cmd.append("--quick")

    # Estimate memory figures before spawning so we can annotate OOM messages.
    mem_hint: tuple[float, float] | None = None
    if device == "gpu":
        try:
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            in_use_gb = (total - free) / 1e9
            g = _stress_gpts(stress_size, quick)
            wf_bytes = g[0] * g[1] * (16 if precision == "float64" else 8)
            mem_hint = (wf_bytes / 1e9, in_use_gb)
        except Exception:
            pass

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )

    if result.stdout:
        print(result.stdout, end="")
        for line in result.stdout.splitlines():
            r = _parse_result_line(line)
            if r is not None:
                _collected_results.append(r)
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.splitlines():
                line = line.strip()
                if "Error" in line or "error" in line:
                    print(f"  stderr: {line}")
                    break
    elif result.returncode != 0:
        prec_label = "f64" if precision == "float64" else "f32"
        proj_label = f"crystal({projection[:3]})" if use_crystal else projection
        label = f"chunk=auto, batch={batch}, proj={proj_label}, prec={prec_label}"
        error_msg = _classify_subprocess_error(result.stderr, result.returncode, mem_hint)
        _collected_results.append({"label": label, "error": error_msg})
        print(f"  {label:<72s}  ERROR: {error_msg}")


def run_single_scan_for_stress_test(device: str, batch: int | str, quick: bool,
                                    projection: str = "infinite",
                                    use_crystal: bool = False,
                                    precision: str = "float32",
                                    stress_size: str = "large"):
    """Run one stress-test scan attempt (called in a subprocess)."""
    config.set({"dask.lazy": False})

    # Disable the cuFFT plan cache so workspace is freed after every FFT call.
    if device == "gpu":
        try:
            import cupy as cp
            cp.fft.config.get_plan_cache().set_size(0)
        except Exception:
            pass
        warmup_gpu()

    if stress_size == "medium":
        gpts, reps, scan_gpts = ((2048, 2048), (1, 1, 4), (2, 2)) if quick else ((8192, 8192), (1, 1, 5), (2, 2))
    else:
        gpts, reps, scan_gpts = ((4096, 4096), (1, 1, 4), (2, 2)) if quick else ((16384, 16384), (1, 1, 5), (2, 2))

    if use_crystal:
        if stress_size == "medium":
            reps = (2, 2, 4) if quick else (8, 8, 5)    # unit_gpts = (1024, 1024)
        else:
            reps = (4, 4, 4) if quick else (16, 16, 5)  # unit_gpts = (1024, 1024)
        potential = make_crystal_potential(gpts, reps, device=device,
                                          slice_thickness=6.0, projection=projection)
        proj_label = f"crystal({projection[:3]})"
    else:
        potential = make_large_potential(gpts, reps, device=device, slice_thickness=6.0,
                                         projection=projection)
        proj_label = projection

    print_result(benchmark_scan(
        potential, scan_gpts, device,
        prebuilt=False,
        potential_chunk_size="auto",
        max_batch=batch,
        projection=proj_label,
        precision=precision,
    ))
    sys.stdout.flush()


def run_stress_plane_subprocess(
    script: str, device: str, quick: bool, projection: str = "infinite",
    use_crystal: bool = False,
    precision: str = "float32",
    stress_size: str = "large",
):
    """Spawn a fresh subprocess for one stress-test PlaneWave attempt."""
    cmd = [
        sys.executable, script,
        "--device", device,
        "--_run-stress-plane",
        "--_projection", projection,
        "--_precision", precision,
        "--_stress-size", stress_size,
    ]
    if use_crystal:
        cmd.append("--_use-crystal")
    if quick:
        cmd.append("--quick")

    mem_hint: tuple[float, float] | None = None
    if device == "gpu":
        try:
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            in_use_gb = (total - free) / 1e9
            g = _stress_gpts(stress_size, quick)
            wf_bytes = g[0] * g[1] * (16 if precision == "float64" else 8)
            mem_hint = (wf_bytes / 1e9, in_use_gb)
        except Exception:
            pass

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )

    if result.stdout:
        print(result.stdout, end="")
        for line in result.stdout.splitlines():
            r = _parse_result_line(line)
            if r is not None:
                _collected_results.append(r)
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.splitlines():
                line = line.strip()
                if "Error" in line or "error" in line:
                    print(f"  stderr: {line}")
                    break
    elif result.returncode != 0:
        prec_label = "f64" if precision == "float64" else "f32"
        proj_label = f"crystal({projection[:3]})" if use_crystal else projection
        label = f"chunk=auto, proj={proj_label}, prec={prec_label}"
        error_msg = _classify_subprocess_error(result.stderr, result.returncode, mem_hint)
        _collected_results.append({"label": label, "error": error_msg})
        print(f"  {label:<72s}  ERROR: {error_msg}")


def run_single_plane_for_stress_test(device: str, quick: bool,
                                     projection: str = "infinite",
                                     use_crystal: bool = False,
                                     precision: str = "float32",
                                     stress_size: str = "large"):
    """Run one stress-test PlaneWave attempt (called in a subprocess)."""
    config.set({"dask.lazy": False})

    if device == "gpu":
        try:
            import cupy as cp
            cp.fft.config.get_plan_cache().set_size(0)
        except Exception:
            pass
        warmup_gpu()

    if stress_size == "medium":
        gpts, reps = ((2048, 2048), (1, 1, 4)) if quick else ((8192, 8192), (1, 1, 5))
    else:  # "large"
        gpts, reps = ((4096, 4096), (1, 1, 4)) if quick else ((16384, 16384), (1, 1, 5))

    if use_crystal:
        if stress_size == "medium":
            reps = (2, 2, 4) if quick else (8, 8, 5)    # unit_gpts = (1024, 1024)
        else:  # "large"
            reps = (4, 4, 4) if quick else (16, 16, 5)  # unit_gpts = (1024, 1024)
        potential = make_crystal_potential(gpts, reps, device=device,
                                          slice_thickness=6.0, projection=projection)
        proj_label = f"crystal({projection[:3]})"
    else:
        potential = make_large_potential(gpts, reps, device=device, slice_thickness=6.0,
                                         projection=projection)
        proj_label = projection

    print_result(benchmark_multislice(potential, chunk_size="auto", device=device,
                                      projection=proj_label, precision=precision))
    sys.stdout.flush()


def run_single_slice_stress_test(device: str, quick: bool = False):
    """Stress-test where a single potential slice barely fits in device memory.

    A 16384² grid produces ~1.07 GB per float32 slice.  On a 25 GB GPU the
    VRAM budget for potential chunks is ~6.2 GB, and with 5× fragmentation
    overhead ``estimate_potential_chunk_size`` returns 1 — so the auto path
    naturally processes one slice at a time without any manual override.

    Two cases are tested:
    - 1 slice (slice_thickness > cell z-extent): no chunking required at all;
      reveals whether the wavefunction FFT workspace alone fits in VRAM.
    - Several slices (slice_thickness=2.0): exercises the chunk_size=1 path.

    Only meaningful on GPU; on CPU the potential is a single chunk by default.
    """
    print(f"\n{'=' * 90}")
    print(f"Single-slice stress test — device={device}")
    print(f"{'=' * 90}")

    # Disable the cuFFT plan cache so workspace is freed after every FFT call.
    # This is essential for grids where the persistent workspace would OOM.
    if device == "gpu":
        try:
            import cupy as cp
            cp.fft.config.get_plan_cache().set_size(0)
        except Exception:
            pass

    # Both PlaneWave and scan run in subprocesses: after the infinite run
    # the Numba CUDA compiler and cuFFT context retain device memory that
    # free_all_blocks() cannot reclaim, leaving insufficient headroom for
    # the finite projection's large disk_indices array.
    script = os.path.abspath(__file__)

    # ── Medium stress test (8192² non-quick / 2048² quick) ────────────
    # One slice at 8192² ≈ 0.27 GB fp32 / 0.54 GB fp64 — fits fp64 within
    # 24 GB VRAM; use this section to confirm fp64 works correctly.
    if quick:
        m_gpts, m_reps = (2048, 2048), (1, 1, 4)
    else:
        m_gpts, m_reps = (8192, 8192), (1, 1, 5)

    m_potential = make_large_potential(m_gpts, m_reps, device=device, slice_thickness=6.0)
    m_num_slices = len(m_potential)
    m_mem_gb = estimate_potential_memory_mb(m_potential) / 1000
    del m_potential

    print(f"\n── {m_gpts[0]}x{m_gpts[1]} (medium / fp64-capable), "
          f"{m_num_slices} slice(s), {m_mem_gb:.2f} GB ──")

    print("\n  [PlaneWave multislice — medium]")
    for projection in ("infinite", "finite"):
        for prec in ("float32", "float64"):
            run_stress_plane_subprocess(script, device, quick, projection=projection,
                                        use_crystal=True, precision=prec,
                                        stress_size="medium")
        for prec in ("float32", "float64"):
            run_stress_plane_subprocess(script, device, quick, projection=projection,
                                        precision=prec, stress_size="medium")

    print("\n  [Probe scan (2×2 positions) — medium]")
    for projection in ("infinite", "finite"):
        for prec in ("float32", "float64"):
            run_stress_scan_subprocess(script, device, "auto", quick, projection=projection,
                                       use_crystal=True, precision=prec, stress_size="medium")
        for prec in ("float32", "float64"):
            run_stress_scan_subprocess(script, device, "auto", quick, projection=projection,
                                       precision=prec, stress_size="medium")

    # ── Large stress test (16384² non-quick / 4096² quick) ────────────
    # One slice at 16384² ≈ 1.07 GB fp32 → auto chunk_size = 1 on a 25 GB GPU.
    # Si cubic cell is 5.43 Å in z; slice_thickness=6.0 Å → exactly 1 slice.
    # fp64 is expected to OOM at this size.
    if quick:
        gpts, reps, scan_gpts = (4096, 4096), (1, 1, 4), (2, 2)
    else:
        gpts, reps, scan_gpts = (16384, 16384), (1, 1, 5), (2, 2)

    potential = make_large_potential(gpts, reps, device=device, slice_thickness=6.0)
    num_slices = len(potential)
    mem_gb = estimate_potential_memory_mb(potential) / 1000
    del potential

    if device == "gpu":
        import cupy as cp
        free, total = cp.cuda.Device().mem_info
        pool = cp.get_default_memory_pool()
        cache = cp.fft.config.get_plan_cache()
        print(f"\n── {gpts[0]}x{gpts[1]} (large), {num_slices} slice(s), {mem_gb:.2f} GB ──")
        print(f"  GPU VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
        print(f"  CuPy pool: used={pool.used_bytes()/1e9:.2f} GB  free={pool.free_bytes()/1e9:.2f} GB")
        print(f"  cuFFT plan cache: {cache}")
    else:
        print(f"\n── {gpts[0]}x{gpts[1]} (large), {num_slices} slice(s), {mem_gb:.2f} GB ──")

    print("\n  [PlaneWave multislice]")
    for projection in ("infinite", "finite"):
        for prec in ("float32", "float64"):
            run_stress_plane_subprocess(script, device, quick, projection=projection,
                                        use_crystal=True, precision=prec)
        for prec in ("float32", "float64"):
            run_stress_plane_subprocess(script, device, quick, projection=projection,
                                        precision=prec)

    print(f"\n  [Probe scan ({scan_gpts[0]}×{scan_gpts[1]} positions)]")
    for projection in ("infinite", "finite"):
        for prec in ("float32", "float64"):
            run_stress_scan_subprocess(script, device, "auto", quick, projection=projection,
                                       use_crystal=True, precision=prec)
        for prec in ("float32", "float64"):
            run_stress_scan_subprocess(script, device, "auto", quick, projection=projection,
                                       precision=prec)


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
    parser.add_argument(
        "--stress-only", action="store_true",
        help="Run only the single-slice stress test (GPU only)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None, metavar="PATH",
        help="Write all benchmark results to a JSON file for later comparison",
    )
    # Internal: used by subprocesses to run a single scan benchmark
    parser.add_argument("--_run-scan-bench", type=int, default=None)
    parser.add_argument("--_prebuilt", action="store_true", default=False)
    parser.add_argument("--_no-prebuilt", action="store_true", default=False)
    parser.add_argument("--_chunk-size", type=str, default="auto")
    parser.add_argument("--_batch-size", type=str, default="auto")
    parser.add_argument("--_to-zarr", action="store_true", default=False)
    parser.add_argument("--_projection", type=str, default="infinite")
    parser.add_argument("--_precision", type=str, default="float32")
    parser.add_argument("--_use-crystal", action="store_true", default=False)
    # Internal: used by subprocesses to run one stress-test scan attempt
    parser.add_argument("--_run-stress-scan", type=str, default=None)
    # Internal: used by subprocesses to run one stress-test PlaneWave attempt
    parser.add_argument("--_run-stress-plane", action="store_true", default=False)
    # Internal: stress test size ("large" = 16384², "medium" = 8192²)
    parser.add_argument("--_stress-size", type=str, default="large")

    args = parser.parse_args()

    config.set({"dask.lazy": False})

    # If we're a subprocess running a stress-test PlaneWave attempt
    if args._run_stress_plane:
        run_single_plane_for_stress_test(args.device, args.quick,
                                         projection=args._projection,
                                         use_crystal=args._use_crystal,
                                         precision=args._precision,
                                         stress_size=args._stress_size)
        return

    # If we're a subprocess running a stress-test scan attempt
    if args._run_stress_scan is not None:
        batch: int | str = args._run_stress_scan
        try:
            batch = int(batch)
        except ValueError:
            pass  # keep as string ("auto")
        run_single_scan_for_stress_test(args.device, batch, args.quick,
                                        projection=args._projection,
                                        use_crystal=args._use_crystal,
                                        precision=args._precision,
                                        stress_size=args._stress_size)
        return

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
            batch_size=bs, projection=args._projection,
            use_crystal=args._use_crystal, precision=args._precision,
        )
        return

    # Otherwise, run the full benchmark suite
    print(f"abTEM version: {abtem.__version__}")
    if args.device in ("gpu", "both"):
        try:
            import cupy as cp
            free, total = cp.cuda.Device().mem_info
            print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            print(f"VRAM: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
            warmup_gpu()
        except ImportError:
            print("ERROR: cupy not available, cannot run GPU benchmarks")
            return

    if args.stress_only:
        run_single_slice_stress_test("gpu", quick=args.quick)
    else:
        if args.device in ("cpu", "both"):
            run_planewave_benchmarks("cpu", quick=args.quick)
            run_scan_benchmarks_via_subprocesses("cpu", quick=args.quick)

        if args.device in ("gpu", "both"):
            run_planewave_benchmarks("gpu", quick=args.quick)
            run_single_slice_stress_test("gpu", quick=args.quick)
            run_scan_benchmarks_via_subprocesses("gpu", quick=args.quick)

    print(f"\n{'=' * 90}")
    print("Done.")

    if args.output_json:
        meta = {
            "abtem_version": abtem.__version__,
            "device": args.device,
            "quick": args.quick,
        }
        output = {"meta": meta, "results": _collected_results}
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
