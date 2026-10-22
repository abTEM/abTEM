"""Benchmark: emulated float64 on Apple Metal against float64 on the CPU.

Run it on an Apple-silicon Mac with PyTorch installed (``uv sync --extra mps``) and
paste the Markdown it prints into the pull request or issue. Two parts:

1. The 2D FFT on its own: double-float on the GPU, float32 on the GPU, and float64
   on the CPU (SciPy, all cores), with the accuracy of each against float64.
2. A multislice of a Si crystal, the same simulation in float64 on the CPU and with
   ``torch.emulate-float64`` on, from an unbuilt potential, as in a real run: the
   warm time, the first (cold) time, which includes compiling the Metal kernels, and
   the largest difference between the two results.

Usage
-----
    python benchmarks/benchmark_metal_float64.py            # about 3 minutes
    python benchmarks/benchmark_metal_float64.py --quick    # under a minute
    python benchmarks/benchmark_metal_float64.py --output result.md
    python benchmarks/benchmark_metal_float64.py --grids 256 480 --probes 1 64
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

import ase.build
import numpy as np

import abtem
from abtem.core import config

REPEATS = 3


def _run(*command):
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- a missing tool only costs a line of the report
        return "unknown"


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment(torch):
    commit = _run(
        "git", "-C", os.path.dirname(abtem.__file__), "rev-parse", "--short", "HEAD"
    )
    memory = _run("sysctl", "-n", "hw.memsize")
    return [
        ("chip", _run("sysctl", "-n", "machdep.cpu.brand_string")),
        ("cores", f"{os.cpu_count()}"),
        ("memory", f"{int(memory) / 2**30:.0f} GiB" if memory.isdigit() else memory),
        ("macOS", platform.mac_ver()[0]),
        ("python", platform.python_version()),
        ("torch", torch.__version__),
        ("numpy / scipy", f"{np.__version__} / {_version('scipy')}"),
        ("pyfftw", _version("pyFFTW")),
        ("abtem", f"{abtem.__version__} ({commit})"),
        ("fft / fftw threads", f"{config.get('fft')} / {config.get('fftw.threads')}"),
    ]


def best_of(function, repeats, synchronize=None):
    function()
    if synchronize:
        synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if synchronize:
            synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


def table(header, rows):
    widths = [
        max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))
    ]

    def line(row):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"

    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), separator] + [line(row) for row in rows])


def benchmark_fft(torch, grids, repeats):
    import scipy.fft as scipy_fft

    from abtem.core import _double_float as double_float

    rng = np.random.default_rng(0)
    rows = []
    for n in grids:
        batch = max(1, (1 << 22) // (n * n))
        x = rng.standard_normal((batch, n, n)) + 1j * rng.standard_normal((batch, n, n))
        device_x = double_float.from_numpy(x)
        seen = double_float.to_numpy(device_x)  # x as the kernels see it
        reference = scipy_fft.fft2(seen, axes=(-2, -1), workers=-1)

        error = np.abs(
            double_float.to_numpy(double_float.fft2(device_x)) - reference
        ).max()
        error /= np.abs(reference).max()
        single = torch.tensor(seen.astype(np.complex64), device="mps")
        error32 = np.abs(torch.fft.fft2(single).cpu().numpy() - reference).max()
        error32 /= np.abs(reference).max()

        sync = torch.mps.synchronize
        t_double = best_of(lambda: double_float.fft2(device_x), repeats, sync)
        t_single = best_of(lambda: torch.fft.fft2(single), repeats, sync)
        t_cpu = best_of(
            lambda: scipy_fft.fft2(seen, axes=(-2, -1), workers=-1), repeats
        )
        rows.append(
            (
                f"{batch} x {n}²",
                f"{t_double * 1e3:.1f}",
                f"{t_single * 1e3:.1f}",
                f"{t_cpu * 1e3:.1f}",
                f"{error:.1e}",
                f"{error32:.1e}",
            )
        )
        print(f"  fft {n}² done", file=sys.stderr, flush=True)
    return table(
        (
            "batch x grid",
            "double-float GPU [ms]",
            "float32 GPU [ms]",
            "float64 CPU [ms]",
            "double-float error",
            "float32 error",
        ),
        rows,
    )


def simulation(grid, probes, slices):
    """The simulation, unbuilt, so that building the potential is part of it, and
    its number of slices."""
    repeats = max(1, round(slices / 5.431))
    cell = ase.build.bulk("Si", "diamond", a=5.431, cubic=True) * (2, 2, repeats)
    potential = abtem.Potential(cell, gpts=grid, slice_thickness=1.0)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=20, defocus=50)
    probe.grid.match(potential)
    side = int(round(probes**0.5))
    scan = abtem.GridScan(
        start=(0, 0), end=(5, 5), gpts=(side, side), potential=potential
    )
    detector = abtem.PixelatedDetector(max_angle=60)

    def run():
        return np.asarray(probe.scan(potential, scan, detector).compute().array)

    return run, potential.num_slices


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def benchmark_multislice(grids, probe_counts, slices, repeats):
    """The table, and the number of slices the crystal came out with."""
    rows, num_slices = [], slices
    for grid in grids:
        for probes in probe_counts:
            with config.set({"precision": "float64", "torch.emulate-float64": False}):
                run, num_slices = simulation(grid, probes, slices)
                reference, _ = timed(run)  # warm FFTW plans, caches
                t_cpu = min(timed(run)[1] for _ in range(repeats))
            with config.set({"precision": "float64", "torch.emulate-float64": True}):
                run, _ = simulation(grid, probes, slices)
                emulated, t_cold = timed(run)
                t_warm = min(timed(run)[1] for _ in range(repeats))
            difference = np.abs(emulated - reference).max() / np.abs(reference).max()
            rows.append(
                (
                    f"{grid}²",
                    probes,
                    f"{t_cpu:.2f}",
                    f"{t_warm:.2f}",
                    f"{t_cpu / t_warm:.1f}x",
                    f"{t_cold:.2f}",
                    f"{difference:.1e}",
                )
            )
            print(
                f"  multislice {grid}², {probes} probes done",
                file=sys.stderr,
                flush=True,
            )
    return num_slices, table(
        (
            "grid",
            "probes",
            "float64 CPU [s]",
            "emulated Metal [s]",
            "speedup",
            "emulated, first call [s]",
            "max difference",
        ),
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small grids, one repeat")
    parser.add_argument("--grids", type=int, nargs="+", help="multislice grid sizes")
    parser.add_argument("--fft-grids", type=int, nargs="+", help="FFT grid sizes")
    parser.add_argument(
        "--probes", type=int, nargs="+", help="probe positions (squares)"
    )
    parser.add_argument(
        "--slices", type=int, default=136, help="slices of the crystal, roughly"
    )
    parser.add_argument("--repeats", type=int, help="repetitions, the best is reported")
    parser.add_argument("--skip-fft", action="store_true")
    parser.add_argument("--skip-multislice", action="store_true")
    parser.add_argument("--output", help="also write the report to this Markdown file")
    args = parser.parse_args()

    from abtem.core.backend import check_metal_kernels_are_available

    try:
        torch = check_metal_kernels_are_available().torch
    except RuntimeError as error:
        sys.exit(f"This benchmark needs abTEM's Metal kernels: {error}")

    config.set(
        {"diagnostics.progress_bar": False, "local_diagnostics.progress_bar": False}
    )
    repeats = args.repeats or (1 if args.quick else REPEATS)
    grids = args.grids or ([128, 256] if args.quick else [128, 256, 512])
    fft_grids = args.fft_grids or (
        [128, 256] if args.quick else [128, 256, 480, 512, 1024]
    )
    probes = args.probes or ([16] if args.quick else [1, 16])

    sections = [
        "## Environment\n\n"
        + table(("setting", "value"), [list(row) for row in environment(torch)])
    ]
    if not args.skip_multislice:
        num_slices, result = benchmark_multislice(grids, probes, args.slices, repeats)
        sections.append(
            f"## Multislice\n\nSi, {num_slices} slices, 200 keV probe scan with a "
            "`PixelatedDetector`, unbuilt potential. Warm times are the best of "
            f"{repeats} runs; the first call at a grid size also compiles the Metal "
            "kernels for it, unless macOS has them cached.\n\n" + result
        )
    if not args.skip_fft:
        sections.append(
            "## 2D FFT\n\nBest of {} runs. Errors are the largest difference to "
            "SciPy's float64 result, relative to its largest value.\n\n".format(repeats)
            + benchmark_fft(torch, fft_grids, repeats)
        )
    report = "\n\n".join(sections) + "\n"
    print(report)
    if args.output:
        with open(args.output, "w") as file:
            file.write(report)


if __name__ == "__main__":
    main()
