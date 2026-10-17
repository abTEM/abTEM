"""Benchmark the native CUDA kernel for finite projections on skewed grids (#599).

Builds a skewed (hexagonal) ``Potential(..., projection="finite")`` on the GPU with
the native kernel (``abtem.integrals.NATIVE_SKEW_CUDA_KERNEL = True``) and with the
host build copied to the device (``False``), and on the CPU for reference. Reports
the best wall time of each and the largest difference of the GPU builds from the
CPU build, relative to its maximum.

Examples
--------
On a CUDA machine::

    python benchmarks/benchmark_skew_finite_projection.py

A larger supercell, double precision::

    python benchmarks/benchmark_skew_finite_projection.py --repeat 8 8 --precision float64
"""

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from ase.build import graphene

import abtem
import abtem.integrals
from abtem.core.backend import asnumpy, cp
from abtem.potentials.iam import Potential


def _build(atoms, device, native, reps, **kwargs):
    abtem.integrals.NATIVE_SKEW_CUDA_KERNEL = native
    times, array = [], None
    for _ in range(reps):
        start = time.perf_counter()
        potential = Potential(atoms, device=device, **kwargs).build(lazy=False)
        array = potential.array
        if device == "gpu":
            cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - start)
    return min(times), asnumpy(array)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--sampling", type=float, default=0.05)
    parser.add_argument("--slice-thickness", type=float, default=1.0)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    if cp is None:
        raise SystemExit("CuPy is not available")
    abtem.config.set({"precision": args.precision})

    atoms = graphene(formula="BN", a=2.504, vacuum=2.0)
    atoms = atoms * (*args.repeat, 1)
    stack = atoms.copy()
    for layer in range(1, args.layers):
        shifted = atoms.copy()
        shifted.positions[:, 2] += 3.33 * layer
        stack += shifted
    stack.cell[2, 2] = 4.0 + 3.33 * args.layers
    stack.pbc = True

    kwargs = dict(
        sampling=args.sampling,
        slice_thickness=args.slice_thickness,
        projection="finite",
    )
    probe = Potential(stack, **kwargs)
    print(
        f"{len(stack)} atoms, gpts {probe.gpts}, {probe.num_slices} slices, "
        f"skewed: {probe.non_orthogonal}, {args.precision}"
    )

    # warm up the kernels and caches once per path
    for device, native in (("cpu", True), ("gpu", True), ("gpu", False)):
        _build(stack, device, native, 1, **kwargs)

    cpu_time, reference = _build(stack, "cpu", True, args.reps, **kwargs)
    scale = np.abs(reference).max()
    print(f"{'cpu':>22}: {cpu_time:8.3f} s")
    for native in (False, True):
        gpu_time, array = _build(stack, "gpu", native, args.reps, **kwargs)
        label = "gpu native kernel" if native else "gpu host build"
        print(
            f"{label:>22}: {gpu_time:8.3f} s  "
            f"max |gpu - cpu| / max = {np.abs(array - reference).max() / scale:.2e}"
        )
    abtem.integrals.NATIVE_SKEW_CUDA_KERNEL = True


if __name__ == "__main__":
    main()
