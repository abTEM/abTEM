"""Benchmark: regular multislice vs PRISM for plasmon (energy-loss) STEM scans.

Compares wall time and peak memory of a STEM scan computed with

* regular multislice, ``FourierMultislice`` (the default),
* regular multislice, ``RealSpaceMultislice`` (finite difference),
* PRISM (``SMatrix``) at interpolation factors ``f = 1, 2, 4``,

for the plain elastic calculation and for order-resolved plasmon scattering
(``PhaseScramblePlasmons(max_loss_order=1, 2, 3)``).

Each cell is run in its own subprocess so the peak-memory measurement is clean
(host RSS above the imported-library baseline on CPU; CuPy memory-pool peak on
GPU). FFTW is given all physical cores on CPU; on GPU the FFTs run on the device.

Usage
-----
Full matrix (spawns one subprocess per cell, prints a Markdown table)::

    python benchmarks/plasmon_multislice_vs_prism.py --matrix --device cpu
    python benchmarks/plasmon_multislice_vs_prism.py --matrix --device gpu

A single cell (prints ``<seconds> <MB>``)::

    # method f max_loss_order algorithm
    python benchmarks/plasmon_multislice_vs_prism.py --cell multislice 1 elastic realspace --device gpu
    python benchmarks/plasmon_multislice_vs_prism.py --cell prism 2 3 fourier --device cpu

``max_loss_order`` is ``elastic`` (no plasmons), ``none`` (unresolved plasmons),
or an integer. The system (cell, sampling, scan, optics) is set by the options
below; the defaults give ~0.064 Å sampling, adequate for physical convergence.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import warnings


def _build_system(args):
    import ase.build

    import abtem
    from abtem import GridScan, PixelatedDetector, Potential

    atoms = ase.build.bulk("Si", cubic=True) * (3, 3, args.cells_z)
    extent = atoms.cell.lengths()[:2]
    potential = Potential(
        atoms, gpts=args.gpts, slice_thickness=args.slice_thickness, device=args.device
    )
    detector = PixelatedDetector(max_angle="valid")
    scan = GridScan(start=(0, 0), end=tuple(extent), gpts=(args.scan, args.scan))
    return potential, detector, scan


def _plasmons(max_loss_order):
    """``elastic`` -> None; ``none`` -> unresolved; int -> order-resolved."""
    if max_loss_order == "elastic":
        return None
    from abtem import PhaseScramblePlasmons

    mlo = None if max_loss_order == "none" else int(max_loss_order)
    return PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=7,
        max_loss_order=mlo,
    )


def run_cell(args):
    warnings.filterwarnings("ignore")
    import psutil

    import abtem
    from abtem import SMatrix, Probe
    from abtem.multislice import FourierMultislice, RealSpaceMultislice

    method, f, max_loss_order, algo_name = args.cell
    f = int(f)

    # Use the whole machine: FFTW threads = physical cores on CPU.
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count()
    if args.device == "cpu":
        abtem.config.set({"fftw.threads": physical})
    abtem.config.set({"device": args.device})

    algorithm = (
        RealSpaceMultislice() if algo_name == "realspace" else FourierMultislice()
    )
    potential, detector, scan = _build_system(args)

    def once():
        if method == "multislice":
            probe = Probe(
                energy=args.energy, semiangle_cutoff=args.semiangle, device=args.device
            )
            probe.grid.match(potential)
            probe.multislice(
                potential,
                scan=scan,
                detectors=detector,
                plasmons=_plasmons(max_loss_order),
                algorithm=algorithm,
            ).compute()
        else:
            s_matrix = SMatrix(
                potential=potential,
                energy=args.energy,
                semiangle_cutoff=args.semiangle,
                interpolation=f,
                plasmons=_plasmons(max_loss_order),
                device=args.device,
            ).build(lazy=False)
            s_matrix.reduce(scan=scan, detectors=detector).compute()

    # Memory sampler: GPU memory pool (gpu) or host RSS (cpu).
    if args.device == "gpu":
        import cupy

        pool = cupy.get_default_memory_pool()

        def mem_now():
            return pool.used_bytes()

    else:
        proc = psutil.Process(os.getpid())

        def mem_now():
            return proc.memory_info().rss

    if not args.no_warmup:
        once()
        gc.collect()

    peak = [mem_now()]
    baseline = peak[0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], mem_now())
            time.sleep(0.005)

    sampler = threading.Thread(target=sample)
    sampler.start()
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
        gc.collect()
    stop.set()
    sampler.join()

    if args.device == "gpu":
        megabytes = peak[0] / 1e6
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            rss *= 1024  # Linux reports KiB, macOS bytes
        megabytes = (rss - baseline) / 1e6

    print(f"{min(times):.2f} {megabytes:.0f}")


def run_matrix(args):
    orders = [
        ("elastic", "elastic (no plasmons)"),
        ("1", "+ plasmons, `max_loss_order=1`"),
        ("2", "+ plasmons, `max_loss_order=2`"),
        ("3", "+ plasmons, `max_loss_order=3`"),
    ]
    # (key, header, method, f, algorithm, reps): real-space is slow -> a single run.
    columns = [
        ("fourier", "multislice (Fourier)", "multislice", "1", "fourier", args.reps),
        ("realspace", "multislice (real-space)", "multislice", "1", "realspace", 1),
        ("prism1", "PRISM f=1", "prism", "1", "fourier", args.reps),
        ("prism2", "PRISM f=2", "prism", "2", "fourier", args.reps),
        ("prism4", "PRISM f=4", "prism", "4", "fourier", args.reps),
    ]

    base = [
        sys.executable,
        os.path.abspath(__file__),
        "--device", args.device,
        "--gpts", str(args.gpts),
        "--cells-z", str(args.cells_z),
        "--slice-thickness", str(args.slice_thickness),
        "--scan", str(args.scan),
        "--semiangle", str(args.semiangle),
        "--energy", str(args.energy),
    ]

    results = {}
    for mlo, _ in orders:
        for key, _, method, f, algo, reps in columns:
            cmd = base + ["--reps", str(reps), "--cell", method, f, mlo, algo]
            if reps == 1:
                cmd.append("--no-warmup")
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
                seconds, megabytes = line.split()
                results[(mlo, key)] = (float(seconds), float(megabytes))
                print(f"{mlo}/{key}: {seconds} s, {megabytes} MB", flush=True)
            except Exception as exc:  # noqa: BLE001 - report and continue
                results[(mlo, key)] = None
                print(f"{mlo}/{key}: FAILED ({exc})", flush=True)

    header = "| Calculation | " + " | ".join(c[1] for c in columns) + " |"
    rule = "|---" + "|--:" * len(columns) + "|"
    print("\n" + header)
    print(rule)
    for mlo, label in orders:
        cells = []
        for key, *_ in columns:
            r = results.get((mlo, key))
            cells.append(f"{r[0]:.1f} s / {r[1] / 1000:.2f} GB" if r else "n/a")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--matrix",
        action="store_true",
        help="run the full method x loss-order matrix and print a Markdown table",
    )
    mode.add_argument(
        "--cell",
        nargs=4,
        metavar=("METHOD", "F", "MLO", "ALGO"),
        help="run one cell: METHOD in {multislice, prism}, F interpolation, "
        "MLO in {elastic, none, <int>}, ALGO in {fourier, realspace}",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpts", type=int, default=256)
    parser.add_argument("--cells-z", type=int, default=12, help="Si cells along z")
    parser.add_argument("--slice-thickness", type=float, default=2.0)
    parser.add_argument("--scan", type=int, default=16, help="scan is SCAN x SCAN")
    parser.add_argument("--semiangle", type=float, default=20.0, help="mrad")
    parser.add_argument("--energy", type=float, default=200e3, help="eV")
    parser.add_argument("--reps", type=int, default=3, help="best-of timing repeats")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if args.matrix:
        run_matrix(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
