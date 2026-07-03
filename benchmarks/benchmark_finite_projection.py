"""Benchmark the finite-projection potential integrals (QuadratureProjectionIntegrals).

Times ``Potential(..., projection="finite").build()`` on CPU and, when CuPy is
available, on GPU, and cross-checks the two results against each other. Also
supports sweeping ``cutoff_tolerance`` to quantify its speed/accuracy tradeoff.

Examples
--------
Run the default benchmark (both devices if CuPy is available)::

    python benchmarks/benchmark_finite_projection.py

CPU only, larger crystal, double precision::

    python benchmarks/benchmark_finite_projection.py --device cpu --reps 8 --precision float64

Sweep the cutoff tolerance (compares against a tight 1e-6 reference)::

    python benchmarks/benchmark_finite_projection.py --tolerance-sweep
"""

import argparse
import os
import time

# Suppress tqdm progress bars (from dask/abtem internals) for clean output.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from ase.spacegroup import crystal

import abtem
from abtem.integrals import QuadratureProjectionIntegrals
from abtem.potentials.iam import Potential


def make_atoms(reps: int):
    sto = crystal(
        ("Sr", "Ti", "O"),
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    )
    return sto * (reps, reps, reps)


def synchronize(device: str):
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def build(atoms, device: str, sampling: float, slice_thickness: float, tolerance: float):
    integrator = QuadratureProjectionIntegrals(cutoff_tolerance=tolerance)
    potential = Potential(
        atoms,
        sampling=sampling,
        slice_thickness=slice_thickness,
        projection="finite",
        integrator=integrator,
        device=device,
    )
    array = potential.build(lazy=False).array
    synchronize(device)
    return array


def to_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array, dtype=np.float64)


def time_build(atoms, device, sampling, slice_thickness, tolerance, repeats):
    build(atoms, device, sampling, slice_thickness, tolerance)  # jit/kernel warmup
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        array = build(atoms, device, sampling, slice_thickness, tolerance)
        times.append(time.perf_counter() - start)
    return min(times), array


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--reps", type=int, default=6, help="crystal repetitions")
    parser.add_argument("--sampling", type=float, default=0.05, help="sampling [A]")
    parser.add_argument("--slice-thickness", type=float, default=1.0)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats (min taken)")
    parser.add_argument(
        "--tolerance-sweep",
        action="store_true",
        help="sweep cutoff_tolerance on CPU and report deviation vs a 1e-6 reference",
    )
    args = parser.parse_args()

    abtem.config.set({"precision": args.precision})

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

    atoms = make_atoms(args.reps)
    probe_potential = Potential(
        atoms, sampling=args.sampling, slice_thickness=args.slice_thickness,
        projection="finite",
    )
    # Report which install is being measured: `import abtem` resolves to the
    # installed package, which may not be the checkout this script lives in.
    print(f"abTEM {abtem.__version__} from {abtem.__file__}")
    print(
        f"{len(atoms)} atoms | gpts {probe_potential.gpts} | "
        f"{probe_potential.num_slices} slices | precision {args.precision}"
    )

    if args.tolerance_sweep:
        reference = to_numpy(
            build(atoms, "cpu", args.sampling, args.slice_thickness, 1e-6)
        )
        print(f"\n{'tolerance':>10s} {'time':>8s} {'max rel dev vs tol=1e-6':>24s}")
        for tolerance in (1e-4, 1e-3, 1e-2):
            elapsed, array = time_build(
                atoms, "cpu", args.sampling, args.slice_thickness, tolerance,
                args.repeats,
            )
            deviation = np.abs(to_numpy(array) - reference).max() / reference.max()
            print(f"{tolerance:10.0e} {elapsed:7.2f}s {deviation:24.3e}")
        return

    results = {}
    print(f"\n{'device':>8s} {'time':>8s}")
    for device in devices:
        elapsed, array = time_build(
            atoms, device, args.sampling, args.slice_thickness, 1e-4, args.repeats
        )
        results[device] = to_numpy(array)
        print(f"{device:>8s} {elapsed:7.2f}s")

    if len(results) == 2:
        deviation = (
            np.abs(results["gpu"] - results["cpu"]).max() / results["cpu"].max()
        )
        print(f"\nGPU vs CPU max relative deviation: {deviation:.3e}")
        # The CPU kernel skips pixels whose contribution is below
        # cutoff_tolerance (lateral disk truncation); the GPU kernel does not
        # yet, so agreement at the ~1e-6 level is expected rather than exact.
        if deviation > 1e-4:
            raise SystemExit("FAIL: GPU and CPU potentials disagree beyond tolerance.")
        print("PASS")


if __name__ == "__main__":
    main()
