"""
Benchmark: time and memory savings from sharing the diffraction-pattern FFT
across multiple angular detectors in a *regular* (elastic) STEM multislice
scan -- as opposed to the transition-potential/core-loss scenario in
``benchmark_shared_diffraction_pattern.py``.

Mirrors a typical everyday STEM simulation: one ``Potential``, one
``GridScan``, and a handful of angular detectors evaluated together (e.g. a
bright-field disk, one or two annular dark-field rings, and a segmented
detector) via ``Probe.scan(..., detectors=[...])`` ->
``multislice_and_detect``.

Each configuration is measured in its own subprocess (see
``benchmark_potential_chunking.py`` for the same pattern) so that one
config's allocations/caches never leak into the peak-RSS measurement of the
next.

Usage:
    python benchmarks/benchmark_stem_multi_detector.py
    python benchmarks/benchmark_stem_multi_detector.py --device gpu
    python benchmarks/benchmark_stem_multi_detector.py --gpts 256 --scan 32
"""

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


CONFIGS = {
    "one_annular": 1,
    "two_annular": 2,
    "three_annular": 3,
    "bf_adf_segmented": "mixed",
}


def _build_detectors(config, max_angle):
    """Build detectors with angles scaled to fractions of ``max_angle``
    (the largest angle the simulation grid can resolve), so the detector
    ranges stay valid across different ``--gpts``/probe settings.
    """
    from abtem import AnnularDetector, SegmentedDetector

    a = max_angle

    if config == 1:
        return [AnnularDetector(inner=0, outer=0.3 * a)]
    if config == 2:
        return [
            AnnularDetector(inner=0, outer=0.3 * a),
            AnnularDetector(inner=0.5 * a, outer=0.9 * a),
        ]
    if config == 3:
        return [
            AnnularDetector(inner=0, outer=0.2 * a),
            AnnularDetector(inner=0.3 * a, outer=0.6 * a),
            AnnularDetector(inner=0.6 * a, outer=0.95 * a),
        ]
    if config == "mixed":
        # A realistic everyday combination: bright field, one ADF ring, and
        # a segmented (e.g. DPC-style) detector, all reading the same
        # diffraction pattern.
        return [
            AnnularDetector(inner=0, outer=0.2 * a),  # BF
            AnnularDetector(inner=0.5 * a, outer=0.9 * a),  # ADF
            SegmentedDetector(
                inner=0.2 * a, outer=0.5 * a, nbins_radial=2, nbins_azimuthal=4
            ),
        ]
    raise ValueError(config)


def _disable_shared_diffraction_pattern_cache():
    """Monkeypatch away the shared-FFT optimization so the same installed
    abtem also gives a true pre-optimization baseline (see the equivalent
    helper in ``benchmark_shared_diffraction_pattern.py`` for why this
    particular monkeypatch point -- ``abtem.waves``'s already-bound
    ``get_shared_diffraction_pattern_fft`` -- is the one that actually takes
    effect everywhere ``Waves.diffraction_patterns`` is called from)."""
    import abtem.waves as _waves_mod

    def _always_recompute(waves, normalize, compute):
        return compute()

    _waves_mod.get_shared_diffraction_pattern_fft = _always_recompute


def _run_subprocess_worker(device, config, gpts, scan_gpts, num_slices, no_cache):
    import resource

    import numpy as np
    from ase.build import bulk

    import abtem
    from abtem import Potential, Probe, GridScan

    if no_cache:
        _disable_shared_diffraction_pattern_cache()

    atoms = bulk("Au", cubic=True) * (
        max(1, gpts // 20), max(1, gpts // 20), 1
    )
    slice_thickness = atoms.cell[2, 2] / num_slices if num_slices else atoms.cell[2, 2]

    potential = Potential(
        atoms, gpts=gpts, slice_thickness=slice_thickness, device=device,
    )
    probe = Probe(energy=200e3, semiangle_cutoff=25, device=device)
    probe.grid.match(potential)

    scan = GridScan(
        start=(0, 0), end=(1, 1), fractional=True, potential=potential,
        endpoint=False, gpts=(scan_gpts, scan_gpts),
    )

    max_angle = float(np.floor(min(probe.cutoff_angles)))
    detectors = _build_detectors(config, max_angle)

    # Warm-up: build FFTW plans / JIT paths outside the timed region.
    warm_scan = GridScan(
        start=(0, 0), end=(0.25, 0.25), fractional=True, potential=potential,
        endpoint=False, gpts=(2, 2),
    )
    probe.scan(scan=warm_scan, potential=potential, detectors=detectors, lazy=False)

    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start = time.perf_counter()
    result = probe.scan(scan=scan, potential=potential, detectors=detectors, lazy=False)
    if hasattr(result, "compute"):
        result = result.compute()
    elapsed = time.perf_counter() - start

    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
    rss_unit_bytes = 1 if sys.platform == "darwin" else 1024

    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "peak_rss_mb": end_rss * rss_unit_bytes / (1024 ** 2),
                "rss_growth_mb": (end_rss - start_rss) * rss_unit_bytes / (1024 ** 2),
            }
        )
    )


def _run_one_config(device, config, gpts, scan_gpts, num_slices, no_cache=False):
    cmd = [
        sys.executable,
        __file__,
        "--subprocess-worker",
        "--device", device,
        "--config", str(config),
        "--gpts", str(gpts),
        "--scan", str(scan_gpts),
        "--slices", str(num_slices),
    ]
    if no_cache:
        cmd.append("--no-cache")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"subprocess failed for config={config}")

    # The JSON result is the last line of stdout (tqdm/warning noise, if
    # any, precedes it).
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpts", type=int, default=192)
    parser.add_argument("--scan", type=int, default=16, help="scan_gpts per side")
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)

    # Internal flags used to re-invoke this script as a measurement worker.
    parser.add_argument("--subprocess-worker", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    if args.subprocess_worker:
        config = args.config
        if config not in ("mixed",):
            config = int(config)
        _run_subprocess_worker(
            args.device, config, args.gpts, args.scan, args.slices, args.no_cache
        )
        return

    if args.device == "gpu":
        try:
            import cupy  # noqa: F401
        except ImportError:
            print("cupy not available, skipping GPU benchmark")
            return

    print(
        f"device={args.device} gpts={args.gpts}x{args.gpts} "
        f"scan={args.scan}x{args.scan} slices={args.slices}\n"
    )

    def _measure(no_cache):
        rows = {}
        for label, config in CONFIGS.items():
            timings, rss_growths, peak_rsses = [], [], []
            for _ in range(args.repeats):
                r = _run_one_config(
                    args.device, config, args.gpts, args.scan, args.slices,
                    no_cache=no_cache,
                )
                timings.append(r["elapsed_s"])
                rss_growths.append(r["rss_growth_mb"])
                peak_rsses.append(r["peak_rss_mb"])
            rows[label] = (min(timings), min(rss_growths), max(peak_rsses), timings)
        return rows

    print("-- pre-cache baseline (each detector recomputes its own FFT) --")
    baseline = _measure(no_cache=True)
    for label, (best_t, best_rss, peak_rss, timings) in baseline.items():
        print(
            f"  {label:20s} time best={best_t:.4f}s "
            f"(all={['%.4f' % t for t in timings]})  "
            f"rss_growth best={best_rss:.1f} MB  peak_rss={peak_rss:.1f} MB"
        )

    print("\n-- with shared-FFT cache (current behavior) --")
    cached = _measure(no_cache=False)
    for label, (best_t, best_rss, peak_rss, timings) in cached.items():
        print(
            f"  {label:20s} time best={best_t:.4f}s "
            f"(all={['%.4f' % t for t in timings]})  "
            f"rss_growth best={best_rss:.1f} MB  peak_rss={peak_rss:.1f} MB"
        )

    labels = list(CONFIGS)
    print()
    print(f"  {'':20s} {'baseline':>10s} {'cached':>10s} {'speedup':>9s}  "
          f"{'rss (baseline->cached)':>26s}")
    for label in labels:
        bt, brss, _, _ = baseline[label]
        ct, crss, _, _ = cached[label]
        print(
            f"  {label:20s} {bt:9.4f}s {ct:9.4f}s {bt / ct:8.2f}x  "
            f"{brss:8.1f} -> {crss:6.1f} MB ({crss - brss:+.1f})"
        )

    print()
    bt1, _, _, _ = baseline["one_annular"]
    bt2, _, _, _ = baseline["two_annular"]
    bt3, _, _, _ = baseline["three_annular"]
    ct1, _, _, _ = cached["one_annular"]
    ct2, _, _, _ = cached["two_annular"]
    ct3, _, _, _ = cached["three_annular"]

    print("  marginal TIME cost of an extra detector, relative to 1 detector's own time:")
    print(
        f"    baseline : 2nd detector {(bt2 - bt1) / bt1 * 100:+.1f}%   "
        f"3rd detector {(bt3 - bt2) / bt1 * 100:+.1f}%"
    )
    print(
        f"    cached   : 2nd detector {(ct2 - ct1) / ct1 * 100:+.1f}%   "
        f"3rd detector {(ct3 - ct2) / ct1 * 100:+.1f}%"
    )

    bt_mix, brss_mix, _, _ = baseline["bf_adf_segmented"]
    ct_mix, crss_mix, _, _ = cached["bf_adf_segmented"]
    print()
    print(
        f"  realistic BF+ADF+segmented pass: "
        f"{bt_mix:.4f}s -> {ct_mix:.4f}s ({bt_mix / ct_mix:.2f}x), "
        f"RSS growth {brss_mix:.1f} -> {crss_mix:.1f} MB"
    )


if __name__ == "__main__":
    main()
