"""Benchmark the paraxial Pauli multislice (abtem.magnetism.multislice) for
non-collinear magnetic scattering.

Times both per-slice evolution schemes -- RealSpaceMultislice (per-slice-exact
Taylor series, the default) and FourierMultislice (Strang split-step, ~7-15x
faster but with a slice-thickness-dependent splitting error) -- on CPU and,
when CuPy is available, on GPU. Cross-checks GPU against CPU (same math,
different backend, so results should agree to near machine precision) and
reports the physical FePt spin magnetic signal both methods converge to.

Examples
--------
Run the default benchmark (both devices if CuPy is available, both
algorithms, small grid)::

    python benchmarks/benchmark_pauli_multislice.py

CPU only, larger (Fig. 8c-sized) grid::

    python benchmarks/benchmark_pauli_multislice.py --device cpu --xy-reps 23 --n-slices 50

GPU only, split-step algorithm only::

    python benchmarks/benchmark_pauli_multislice.py --device gpu --algorithm split

Sweep grid size on both devices, both algorithms::

    python benchmarks/benchmark_pauli_multislice.py --grid-sweep
"""

import argparse
import os
import time

# Suppress tqdm progress bars (from dask/abtem internals) for clean output.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from ase import Atoms

import abtem
from abtem.core.energy import energy2wavelength
from abtem.detectors import WavesDetector
from abtem.magnetism.iam import MagneticField, VectorPotential
from abtem.magnetism.multislice import pauli_multislice
from abtem.magnetism.utils import set_magnetic_moments
from abtem.multislice import FourierMultislice, RealSpaceMultislice

ENERGY = 100e3
A_LAT, C_LAT = 2.71, 3.72  # FePt (L1_0) lattice parameters [PRB 94, 174414]

ALGORITHMS = {"series": RealSpaceMultislice(), "split": FourierMultislice()}


def to_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def synchronize(device: str):
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def make_atoms(xy_reps: int, n_slices: int):
    unit = Atoms(
        "FePt",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[A_LAT, A_LAT, C_LAT],
        pbc=True,
    )
    atoms = unit * (xy_reps, xy_reps, n_slices)
    moments = np.zeros((len(atoms), 3))
    moments[atoms.get_atomic_numbers() == 26] = [0, 0, 3.0]
    set_magnetic_moments(atoms, moments)
    return atoms


def run(atoms, device: str, algorithm: str, xy_reps: int, gpts_cell: int):
    gpts = gpts_cell * xy_reps
    A = VectorPotential(atoms, gpts=gpts, slice_thickness=C_LAT, device=device).build(
        lazy=False
    )
    B = MagneticField(atoms, gpts=gpts, slice_thickness=C_LAT, device=device).build(
        lazy=False
    )
    potential = abtem.Potential(
        atoms, gpts=gpts, slice_thickness=C_LAT, device=device
    ).build(lazy=False)
    probe = abtem.Probe(
        semiangle_cutoff=15,
        energy=ENERGY,
        extent=A_LAT * xy_reps,
        gpts=gpts,
        device=device,
    )
    position = [(xy_reps // 2 * A_LAT,) * 2]

    def one_spin(polarization):
        spinor = probe.build(scan=position, lazy=False).to_spinor(polarization)
        out = pauli_multislice(
            spinor,
            potential,
            vector_potential=A,
            magnetic_field=B,
            average_field=(0, 0, 1.34),
            detectors=WavesDetector(),
            algorithm=ALGORITHMS[algorithm],
        )
        return to_numpy(out.ensure_reciprocal_space().array)

    one_spin((1, 0))  # warm-up: JIT compile / FFT plan creation
    synchronize(device)
    start = time.perf_counter()
    up = one_spin((1, 0))
    down = one_spin((0, 1))
    synchronize(device)
    elapsed = time.perf_counter() - start

    wavelength = energy2wavelength(ENERGY)
    extent = A_LAT * xy_reps
    freq = np.fft.fftfreq(gpts, d=extent / gpts)
    k_mrad = np.hypot(freq[:, None], freq[None]) * wavelength * 1e3
    intensity_up = (np.abs(up) ** 2).sum(0)
    intensity_down = (np.abs(down) ** 2).sum(0)
    intensity_up /= intensity_up.sum()
    intensity_down /= intensity_down.sum()
    mask = k_mrad <= 14.0
    spin_signal = intensity_up[mask].sum() - intensity_down[mask].sum()

    return elapsed, up, down, spin_signal


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument(
        "--algorithm", choices=("series", "split", "both"), default="both"
    )
    parser.add_argument("--xy-reps", type=int, default=11, help="in-plane unit-cell repeats")
    parser.add_argument("--gpts-cell", type=int, default=12, help="grid points per unit cell")
    parser.add_argument("--n-slices", type=int, default=10, help="unit cells in z")
    parser.add_argument(
        "--grid-sweep",
        action="store_true",
        help="sweep (xy_reps, n_slices) instead of a single grid size",
    )
    args = parser.parse_args()

    abtem.config.set({"precision": "float64"})  # magnetic signals ~1e-4 to 1e-8

    devices = []
    if args.device in ("cpu", "both"):
        devices.append("cpu")
    if args.device in ("gpu", "both"):
        try:
            import cupy  # noqa: F401

            devices.append("gpu")
        except ImportError:
            if args.device == "gpu":
                raise SystemExit("GPU requested but CuPy is not installed.")
            print("CuPy not installed; running CPU only.")

    algorithms = list(ALGORITHMS) if args.algorithm == "both" else [args.algorithm]

    # Report which install is being measured: `import abtem` resolves to the
    # installed package, which may not be the checkout this script lives in.
    print(f"abTEM {abtem.__version__} from {abtem.__file__}")

    grids = (
        [(11, 10), (23, 50)] if args.grid_sweep else [(args.xy_reps, args.n_slices)]
    )

    print(
        f"\n{'grid':>16s} {'algorithm':>10s} {'device':>8s} {'time (s)':>10s} "
        f"{'spin signal':>14s}"
    )
    results_by_grid = {}
    for xy_reps, n_slices in grids:
        atoms = make_atoms(xy_reps, n_slices)
        gpts = args.gpts_cell * xy_reps
        grid_label = f"{gpts}^2 x {n_slices}"
        for algorithm in algorithms:
            results = {}
            for device in devices:
                elapsed, up, down, signal = run(
                    atoms, device, algorithm, xy_reps, args.gpts_cell
                )
                results[device] = (elapsed, up, down, signal)
                print(
                    f"{grid_label:>16s} {algorithm:>10s} {device:>8s} "
                    f"{elapsed:10.2f} {signal:14.3e}"
                )
            results_by_grid[(xy_reps, n_slices, algorithm)] = results

    if "cpu" in devices and "gpu" in devices:
        print("\nGPU vs CPU correctness (same math, different backend):")
        all_ok = True
        for (xy_reps, n_slices, algorithm), results in results_by_grid.items():
            t_cpu, up_cpu, down_cpu, sig_cpu = results["cpu"]
            t_gpu, up_gpu, down_gpu, sig_gpu = results["gpu"]
            wave_dev = max(
                np.abs(up_cpu - up_gpu).max() / np.abs(up_cpu).max(),
                np.abs(down_cpu - down_gpu).max() / np.abs(down_cpu).max(),
            )
            sig_dev = abs(sig_gpu - sig_cpu) / abs(sig_cpu)
            speedup = t_cpu / t_gpu
            ok = wave_dev < 1e-8 and sig_dev < 1e-6
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"  {algorithm:>6s} @ {xy_reps}x{xy_reps}x{n_slices}: "
                f"wave rel dev={wave_dev:.2e}, signal rel dev={sig_dev:.2e}, "
                f"speedup={speedup:.1f}x  [{status}]"
            )
        if not all_ok:
            raise SystemExit("FAIL: GPU and CPU results disagree beyond tolerance.")
        print("PASS")


if __name__ == "__main__":
    main()
