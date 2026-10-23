"""Benchmark the two-way solver against the multislice methods that approximate it.

For a plane wave on a thin crystal, every method gives the transmitted wave; the
two-way solver (``TwoWaySolver``) is exact for the Helmholtz equation, so it is the
reference for the accuracy of the others, and its cost is compared with theirs.

Each method runs in its own subprocess, so that its peak memory is its own. For every
method the table lists

* the wall time, from a ``Potential`` and a plane wave to the exit wave (the potential
  is built inside, as in a simulation; the best of ``--repeats``),
* the peak resident memory of the process above that of the imported packages
  (and, on a GPU, the memory taken by the CuPy pool),
* the error [%] of the intensity |psi_g|^2 summed in three bands of the scattering
  angle against the two-way solution, ``low`` (0 to 1/3 of the cutoff), ``mid`` and
  ``high`` (up to 0.98 of the cutoff); ``--weighting flux`` weights the beams of both
  by kz / k (the flux, which the solver conserves) instead,
* the sum of |psi_g|^2 and the flux of the exit wave.

``RealSpaceMultislice(expansion_scope="full")`` always applies the backscatter
correction. The solver and ``order="exact"`` (abTEM#629) are left out of the methods of
a checkout that does not have them (see ``--list``); without the solver there is no
reference, and the table has only the times and the memory. The same holds if
``--methods`` leaves ``twoway`` out.

Examples
--------
Graphene at 5 keV, every method::

    python benchmarks/benchmark_twoway_vs_multislice.py --system graphene --energy 5000

How the cost grows with the thickness of SrTiO3 (the cost of multislice is linear in
it; so is that of the solver, the number of beams stays the same)::

    python benchmarks/benchmark_twoway_vs_multislice.py --system srtio3 \\
        --energy 30000 --cells 1 2 4 --methods twoway rs-full-o1

How the cost grows with the in-plane size, where the solver is cubic in the number of
beams: a perfect crystal is reduced by its symmetry, so break it (``--rattle``) or
switch the reduction off (``--no-symmetry``) to see the cost of a general structure::

    python benchmarks/benchmark_twoway_vs_multislice.py --system graphene \\
        --energy 5000 --lateral 1 2 3 --rattle 0.05 --methods twoway fourier

On a GPU (the solver and the multislice methods run on CUDA devices)::

    python benchmarks/benchmark_twoway_vs_multislice.py --device gpu --energy 5000

Memory is reported in MB (10^6 bytes), times in s, lengths in Å as everywhere in abTEM.
Set ``--threads`` (or the thread environment variables) when comparing runs: a BLAS
that starts one thread per core on the small blocks of the solver is much slower.
"""

import argparse
import dataclasses
import json
import os
import resource
import subprocess
import sys
import time

# the progress bars of dask and abTEM would flood the table
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

BANDS = ("low", "mid", "high")


def _peak_rss_mb():
    """Peak resident memory of this process [MB] (ru_maxrss is bytes on macOS and
    kilobytes on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak * 1e3 / 1e6


def make_atoms(system, cells, lateral, rattle, seed):
    """The slab and the grid of its potential, per lateral repeat the grid that
    keeps the sampling below 0.046 Å (cutoff 5.4 Å^-1) and divides the lattice."""
    import ase
    import ase.build

    import abtem

    if system == "graphene":
        atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
        atoms = atoms * (lateral, lateral, 1)
        atoms.center(vacuum=6.0, axis=2)
        gpts = (64 * lateral, 112 * lateral)
    elif system == "mos2":
        atoms = abtem.orthogonalize_cell(
            ase.build.mx2(formula="MoS2", kind="2H", a=3.18, thickness=3.19, vacuum=3.0)
        )
        atoms = atoms * (lateral, lateral, 1)
        atoms.center(vacuum=3.0, axis=2)
        gpts = (80 * lateral, 128 * lateral)
    elif system == "srtio3":
        a = 3.905
        unit = ase.Atoms(
            "SrTiO3",
            scaled_positions=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ],
            cell=[a, a, a],
            pbc=True,
        )
        atoms = unit * (2 * lateral, 2 * lateral, cells)
        atoms.center(vacuum=2.0, axis=2)
        gpts = (192 * lateral, 192 * lateral)
    else:
        raise ValueError(system)
    atoms.pbc = [True, True, False]
    if rattle:
        rng = np.random.default_rng(seed)
        atoms.positions[:, :2] += rng.normal(0.0, rattle, (len(atoms), 2))
    return atoms, gpts


def make_potential(atoms, gpts, thickness, device):
    import abtem

    n = max(1, int(round(atoms.cell[2, 2] / thickness)))
    return abtem.Potential(
        atoms,
        gpts=gpts,
        slice_thickness=atoms.cell[2, 2] / n,
        projection="finite",
        device=device,
    )


def available_methods():
    """name -> description of the methods this checkout has."""
    from abtem.multislice import RealSpaceMultislice

    names = {}
    try:
        import abtem.twoway  # noqa: F401

        names["twoway"] = "TwoWaySolver"
    except ImportError:  # a checkout without the solver: times and memory only
        pass
    names["fourier"] = "FourierMultislice"
    for order in (1, 2, 3):
        names[f"rs-full-o{order}"] = f"RealSpaceMultislice full, order {order}"
    fields = {f.name for f in dataclasses.fields(RealSpaceMultislice)}
    if "lanczos_tolerance" in fields:  # order="exact", abTEM#629
        names["rs-full-exact"] = "RealSpaceMultislice full, order 'exact'"
    return names


def make_algorithm(name):
    from abtem.multislice import FourierMultislice, RealSpaceMultislice

    if name == "fourier":
        return FourierMultislice()
    order = name.removeprefix("rs-full-")
    return RealSpaceMultislice(
        expansion_scope="full", order="exact" if order == "exact" else int(order[1:])
    )


def band_edges(cutoff):
    return [0.0, cutoff / 3, 2 * cutoff / 3, 0.98 * cutoff]


def band_sums(array, extent, energy, edges):
    """Intensity |psi_g|^2 and flux |psi_g|^2 kz / k of the beams of an exit wave,
    summed over the bands of |g| and in total: (bands, total, flux bands, total flux).

    The exit wave of every method is the wave function, so the beams of two methods
    compare as |psi_g|^2 against |psi_g|^2. The flux (what the solver conserves) weights
    the beams by kz / k, which is small at high angles and zero for evanescent beams: it
    must weight both methods, never one."""
    from abtem.core.energy import energy2wavelength

    wavelength = energy2wavelength(energy)
    nx, ny = array.shape[-2:]
    g = np.hypot(
        np.fft.fftfreq(nx, extent[0] / nx)[:, None],
        np.fft.fftfreq(ny, extent[1] / ny)[None, :],
    )
    kz = np.sqrt(np.maximum(1 - (wavelength * g) ** 2, 0.0))
    intensity = np.abs(np.fft.fft2(array) / (nx * ny)) ** 2

    def bands(values):
        return [
            float(values[(g >= low) & (g < high)].sum())
            for low, high in zip(edges[:-1], edges[1:])
        ]

    return (
        bands(intensity),
        float(intensity.sum()),
        bands(intensity * kz),
        float((intensity * kz).sum()),
    )


def worker(args):
    """Run one method and print its result as a JSON line."""
    import abtem
    from abtem.core.backend import asnumpy
    from abtem.core.energy import energy2wavelength

    baseline = _peak_rss_mb()
    abtem.config.set({"precision": args.precision})
    device = args.device
    wavelength = energy2wavelength(args.energy)
    cutoff = args.cutoff or min(5.4, 0.95 / wavelength)
    atoms, gpts = make_atoms(
        args.system, args.cells[0], args.lateral[0], args.rattle, args.seed
    )
    wave = abtem.PlaneWave(energy=args.energy, device=device)

    def synchronize():
        if device == "gpu":
            import cupy

            cupy.cuda.Stream.null.synchronize()

    if args.worker == "twoway":
        from abtem.twoway import TwoWaySolver

        thickness = 0.25 * wavelength / (2 * np.pi)
        solver = TwoWaySolver(
            cutoff=cutoff,
            symmetry=not args.no_symmetry,
            precision=args.solver_precision,
        )

        def run():
            potential = make_potential(atoms, gpts, thickness, device)
            transmitted, reflected = solver.scatter(wave, potential)
            return transmitted, potential, reflected

    else:
        algorithm = make_algorithm(args.worker)

        def run():
            potential = make_potential(atoms, gpts, args.slice_thickness, device)
            exit_wave = wave.multislice(potential, algorithm=algorithm, lazy=False)
            return exit_wave, potential, None

    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result, potential, reflected = run()
        synchronize()
        times.append(time.perf_counter() - start)
    array = np.asarray(asnumpy(result.array))
    bands, total, flux_bands, flux = band_sums(
        array, potential.extent, args.energy, band_edges(cutoff)
    )
    out = {
        "method": args.worker,
        "times": times,
        "peak_mb": _peak_rss_mb(),
        "baseline_mb": baseline,
        "bands": bands,
        "total": total,
        "flux_bands": flux_bands,
        "flux": flux,
        "slices": int(potential.num_slices),
        "gpts": list(array.shape[-2:]),
        "reflected": None
        if reflected is None
        else float(np.mean(np.abs(np.asarray(asnumpy(reflected.array))) ** 2)),
    }
    if device == "gpu":
        import cupy

        out["gpu_pool_mb"] = cupy.get_default_memory_pool().total_bytes() / 1e6
    print("RESULT " + json.dumps(out))


def run_worker(method, args, cells, lateral):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        method,
        "--system",
        args.system,
        "--energy",
        str(args.energy),
        "--cells",
        str(cells),
        "--lateral",
        str(lateral),
        "--slice-thickness",
        str(args.slice_thickness),
        "--precision",
        args.precision,
        "--solver-precision",
        args.solver_precision,
        "--repeats",
        str(args.repeats),
        "--rattle",
        str(args.rattle),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.cutoff:
        command += ["--cutoff", str(args.cutoff)]
    if args.no_symmetry:
        command.append("--no-symmetry")
    env = dict(os.environ)
    if args.threads:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            env[name] = str(args.threads)
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    for line in done.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    return {"method": method, "error": (done.stderr.strip().splitlines() or ["?"])[-1]}


def format_row(label, result, reference, weighting="intensity"):
    if "error" in result:
        return f"{label:<26s} failed: {result['error'][:70]}"
    best = min(result["times"])
    extra = result["peak_mb"] - result["baseline_mb"]
    cells = [f"{label:<26s}", f"{best:9.1f}", f"{extra:9.0f}"]
    if "gpu_pool_mb" in result:
        cells.append(f"{result['gpu_pool_mb']:9.0f}")
    if reference is None or "error" in reference:
        cells += ["        -"] * 3
    else:
        key = "bands" if weighting == "intensity" else "flux_bands"
        for value, exact in zip(result[key], reference[key]):
            cells.append(f"{100 * (value / exact - 1):+9.2f}")
    cells.append(f"{result['total']:8.4f}")
    cells.append(f"{result['flux']:8.4f}")
    cells.append(f"{result['slices']:7d}")
    return " ".join(cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--system", choices=("graphene", "mos2", "srtio3"), default="graphene"
    )
    parser.add_argument("--energy", type=float, default=5000.0, help="energy [eV]")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="beam cutoff of the solver [1/Å] (default min(5.4, 0.95/wavelength))",
    )
    parser.add_argument(
        "--cells", type=int, nargs="+", default=[1], help="unit cells along z (srtio3)"
    )
    parser.add_argument(
        "--lateral", type=int, nargs="+", default=[1], help="repeats in the plane"
    )
    parser.add_argument(
        "--slice-thickness",
        type=float,
        default=0.05,
        help="slices of the multislice methods [Å]",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="default: all available; the error columns need twoway among them",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="abtem.config['precision']",
    )
    parser.add_argument(
        "--solver-precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--no-symmetry", action="store_true", help="solve in the full beam basis"
    )
    parser.add_argument(
        "--rattle",
        type=float,
        default=0.0,
        help="random in-plane displacement of the atoms, standard deviation [Å]",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="timed repeats (the first includes compilation)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads of BLAS/OpenMP/Numba in each process",
    )
    parser.add_argument(
        "--weighting",
        choices=("intensity", "flux"),
        default="intensity",
        help="the bands of the error table: sums of |psi_g|^2 (the wave function, "
        "default) or of |psi_g|^2 kz / k (the flux)",
    )
    parser.add_argument("--json", default=None, help="write all results to this file")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--list", action="store_true", help="list the available methods"
    )
    args = parser.parse_args()

    if args.worker:
        worker(args)
        return

    methods = available_methods()
    if args.list:
        for name, description in methods.items():
            print(f"{name:<20s} {description}")
        return
    selected = args.methods or list(methods)
    unknown = [m for m in selected if m not in methods]
    if unknown:
        raise SystemExit(f"unavailable methods {unknown}; available: {list(methods)}")
    # the errors are against the two-way solution, which therefore runs first, if it is
    # among the methods (it is the slowest: leave it out for times and memory only)
    if "twoway" in selected:
        selected.remove("twoway")
        selected.insert(0, "twoway")
    else:
        print("no two-way solution among the methods: only times and memory")

    everything = []
    for lateral in args.lateral:
        for cells in args.cells:
            print(
                f"\n{args.system}, {args.energy:g} eV, {cells} cell(s) along z, "
                f"{lateral}x{lateral} in the plane, {args.device}, "
                f"multislice slices {args.slice_thickness} Å, precision "
                f"{args.precision} (solver {args.solver_precision})"
                + (f", rattle {args.rattle} Å" if args.rattle else "")
                + (", no symmetry" if args.no_symmetry else "")
            )
            header = (
                f"{'method':<26s} {'time [s]':>9s} {'mem [MB]':>9s} "
                + (f"{'pool [MB]':>9s} " if args.device == "gpu" else "")
                + " ".join(f"{band + ' [%]':>9s}" for band in BANDS)
                + f" {'sum|psi|2':>9s} {'flux':>8s} {'slices':>7s}"
            )
            print(header)
            reference = None
            for method in selected:
                result = run_worker(method, args, cells, lateral)
                if method == "twoway":
                    reference = result
                print(format_row(method, result, reference, args.weighting), flush=True)
                everything.append({"cells": cells, "lateral": lateral, **result})
            if reference and "error" not in reference:
                print(
                    f"(errors against the two-way solution; its reflection is "
                    f"{reference['reflected']:.1e}, its grid {reference['gpts']})"
                )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(everything, handle, indent=1)


if __name__ == "__main__":
    main()
