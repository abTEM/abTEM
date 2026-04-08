"""
Benchmark: new (chunked) vs old (monolithic) potential handling in abTEM.

The key change on the potential_chunking branch is that multislice now iterates
through the potential in memory-bounded chunks instead of building the entire
potential into a single in-memory array first.  This script measures that
overhead/benefit across a range of grid sizes and potential depths.

Usage
-----
Run against the branch or against main (or a pip install), then compare:

    # on branch
    python benchmarks/benchmark_comparison.py --output results_branch.json

    # on main / pip
    python benchmarks/benchmark_comparison.py --output results_main.json

    # print a side-by-side comparison
    python benchmarks/benchmark_comparison.py \\
        --compare results_main.json results_branch.json

Flags
-----
--device   cpu | gpu          (default: cpu)
--output   PATH               write JSON results to PATH
--compare  A.json B.json      compare two result files instead of running
--quick                       use smaller grids for a fast smoke test
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time

import numpy as np
from ase.build import bulk

os.environ.setdefault("TQDM_DISABLE", "1")

# Ensure we import from the repo root when run directly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import abtem
from abtem import AnnularDetector, PlaneWave, Potential, Probe
from abtem.scan import GridScan


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

def _has_chunking() -> bool:
    """Return True if this abtem build supports potential_chunk_size."""
    from abtem.multislice import multislice_and_detect
    return "potential_chunk_size" in inspect.signature(multislice_and_detect).parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_potential(gpts, repetitions, device="cpu", slice_thickness=2.0):
    atoms = bulk("Si", cubic=True) * repetitions
    return Potential(atoms, gpts=gpts, slice_thickness=slice_thickness, device=device)


def _gpu_cleanup():
    gc.collect()
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass


def _time(fn, repeats=3):
    """Run fn() *repeats* times and return the mean elapsed seconds."""
    times = []
    for _ in range(repeats):
        _gpu_cleanup()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------

def bench_planewave_prebuilt(potential, device, repeats=3):
    """Reference: build the full potential first, then run multislice."""
    waves = PlaneWave(energy=200e3, gpts=potential.gpts, device=device)
    waves.grid.match(potential)
    built = potential.build(lazy=False)

    def run():
        waves.multislice(built, lazy=False)

    return _time(run, repeats)


def bench_planewave_unbuilt_default(potential, device, repeats=3):
    """Unbuilt potential with default settings (new: chunked; old: monolithic)."""
    waves = PlaneWave(energy=200e3, gpts=potential.gpts, device=device)
    waves.grid.match(potential)

    def run():
        waves.multislice(potential, lazy=False)

    return _time(run, repeats)


def bench_planewave_unbuilt_chunked(potential, device, chunk_size, repeats=3):
    """Unbuilt potential with an explicit chunk size (branch only)."""
    waves = PlaneWave(energy=200e3, gpts=potential.gpts, device=device)
    waves.grid.match(potential)

    def run():
        waves.multislice(potential, potential_chunk_size=chunk_size, lazy=False)

    return _time(run, repeats)


def bench_probe_scan(potential, scan_gpts, device, max_batch="auto", repeats=3):
    """Probe scan via Probe.scan()."""
    probe = Probe(energy=200e3, semiangle_cutoff=20, device=device)
    detector = AnnularDetector(inner=40, outer=200)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

    def run():
        result = probe.scan(potential, scan=scan, detectors=detector,
                            lazy=True, max_batch=max_batch)
        result.compute()

    return _time(run, repeats)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_benchmarks(device: str, quick: bool, repeats: int = 3) -> dict:
    """Run all benchmarks and return a results dict."""
    chunking = _has_chunking()

    if quick:
        grid_configs = [
            # (gpts, repetitions)
            ((512, 512),  (5, 5, 8)),   # ~40 slices, ~0.2 GB
        ]
        scan_configs = [
            # (gpts, repetitions, scan_gpts)
            ((512, 512),  (5, 5, 8), (4, 4)),
        ]
    elif device == "gpu":
        grid_configs = [
            ((2048, 2048), (10, 10, 30),  ),   # ~3 GB
            ((4096, 4096), (20, 20, 75),  ),   # ~14 GB
        ]
        scan_configs = [
            ((2048, 2048), (10, 10, 30),  (8, 8)),
            ((4096, 4096), (20, 20, 75),  (8, 8)),
        ]
    else:
        grid_configs = [
            ((512,  512),  (5,  5,  8)),   # ~40 slices, ~0.2 GB
            ((1024, 1024), (8,  8, 15)),   # ~60 slices, ~1.5 GB
            ((2048, 2048), (10, 10, 20)),  # ~80 slices, ~6 GB
        ]
        scan_configs = [
            ((512,  512),  (5,  5,  8),  (8, 8)),
            ((1024, 1024), (8,  8, 15),  (8, 8)),
        ]

    results = {
        "abtem_version": abtem.__version__,
        "has_chunking": chunking,
        "device": device,
        "planewave": [],
        "probe_scan": [],
    }

    print(f"\nabTEM {abtem.__version__}  |  chunking={'yes' if chunking else 'no'}  |  device={device}")
    print("=" * 70)

    # ── PlaneWave multislice ──────────────────────────────────────────────
    print("\n[PlaneWave multislice]")
    for cfg in grid_configs:
        gpts, reps = cfg
        pot = _make_potential(gpts, reps, device=device)
        n_slices = len(pot)
        mem_gb = n_slices * gpts[0] * gpts[1] * 4 / 1e9

        if chunking:
            from abtem.core.chunks import estimate_potential_chunk_size
            auto_cs = estimate_potential_chunk_size(gpts, device)
        else:
            auto_cs = None

        cs_note = f"  auto chunk_size={auto_cs}" if auto_cs is not None else ""
        print(f"\n  {gpts[0]}×{gpts[1]}, {n_slices} slices, {mem_gb:.2f} GB{cs_note}")

        row = {"gpts": list(gpts), "n_slices": n_slices, "mem_gb": mem_gb,
               "auto_chunk_size": auto_cs, "times": {}}

        try:
            t = bench_planewave_prebuilt(pot, device, repeats)
            row["times"]["prebuilt"] = t
            print(f"    pre-built (reference):   {t:7.3f} s")
        except Exception as e:
            row["times"]["prebuilt"] = None
            print(f"    pre-built (reference):   FAILED ({e})")

        try:
            t = bench_planewave_unbuilt_default(pot, device, repeats)
            label = "unbuilt/chunked(auto)" if chunking else "unbuilt/monolithic"
            row["times"]["unbuilt_default"] = t
            print(f"    {label:<28s} {t:7.3f} s")
        except Exception as e:
            row["times"]["unbuilt_default"] = None
            print(f"    unbuilt default:         FAILED ({e})")

        if chunking:
            for cs in [1, 5]:
                if cs >= n_slices:
                    continue
                try:
                    t = bench_planewave_unbuilt_chunked(pot, device, cs, repeats)
                    row["times"][f"unbuilt_chunk{cs}"] = t
                    print(f"    unbuilt chunk={cs:<3d}:         {t:7.3f} s")
                except Exception as e:
                    row["times"][f"unbuilt_chunk{cs}"] = None
                    print(f"    unbuilt chunk={cs}:           FAILED ({e})")

        results["planewave"].append(row)
        del pot
        _gpu_cleanup()

    # ── Probe scan ───────────────────────────────────────────────────────
    print("\n[Probe scan]")
    for cfg in scan_configs:
        gpts, reps, scan_gpts = cfg
        pot = _make_potential(gpts, reps, device=device)
        n_slices = len(pot)
        mem_gb = n_slices * gpts[0] * gpts[1] * 4 / 1e9
        n_pos = scan_gpts[0] * scan_gpts[1]

        print(f"\n  {gpts[0]}×{gpts[1]}, {n_slices} slices, {mem_gb:.2f} GB, "
              f"scan {scan_gpts[0]}×{scan_gpts[1]} ({n_pos} pos)")

        row = {"gpts": list(gpts), "n_slices": n_slices, "mem_gb": mem_gb,
               "scan_gpts": list(scan_gpts), "times": {}}

        try:
            t = bench_probe_scan(pot, scan_gpts, device, max_batch="auto",
                                 repeats=repeats)
            row["times"]["auto"] = t
            print(f"    max_batch=auto:          {t:7.3f} s")
        except Exception as e:
            row["times"]["auto"] = None
            print(f"    max_batch=auto:          FAILED ({e})")

        results["probe_scan"].append(row)
        del pot
        _gpu_cleanup()

    return results


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def _pct(a, b):
    """Return '±X%' string for b relative to a (a=reference/old)."""
    if a is None or b is None or a == 0:
        return "   n/a"
    pct = (b - a) / a * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def compare(path_a: str, path_b: str):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print(f"\nComparison")
    print(f"  A (reference): abTEM {a['abtem_version']}  chunking={a['has_chunking']}")
    print(f"  B (new):       abTEM {b['abtem_version']}  chunking={b['has_chunking']}")
    print(f"  device: {a['device']}")
    print()

    def _row(label, ta, tb, width=32):
        ta_s = f"{ta:.3f}s" if ta is not None else " n/a "
        tb_s = f"{tb:.3f}s" if tb is not None else " n/a "
        pct = _pct(ta, tb)
        sign = "faster" if (ta and tb and tb < ta) else ("slower" if (ta and tb and tb > ta) else "")
        print(f"    {label:<{width}} {ta_s:>8}  →  {tb_s:>8}  {pct:>8}  {sign}")

    print("[PlaneWave multislice]")
    for ra, rb in zip(a["planewave"], b["planewave"]):
        g = ra["gpts"]
        print(f"\n  {g[0]}×{g[1]}, {ra['n_slices']} slices:")
        all_keys = sorted(set(ra["times"]) | set(rb["times"]))
        for k in all_keys:
            _row(k, ra["times"].get(k), rb["times"].get(k))

    print("\n[Probe scan]")
    for ra, rb in zip(a["probe_scan"], b["probe_scan"]):
        g = ra["gpts"]
        sg = ra["scan_gpts"]
        print(f"\n  {g[0]}×{g[1]}, scan {sg[0]}×{sg[1]}:")
        all_keys = sorted(set(ra["times"]) | set(rb["times"]))
        for k in all_keys:
            _row(k, ra["times"].get(k), rb["times"].get(k))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output", metavar="PATH",
                        help="Write JSON results to this file")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two result JSON files")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller grids for a fast smoke test")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    results = run_benchmarks(args.device, args.quick, repeats=1 if args.quick else 3)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
