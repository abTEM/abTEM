"""
Diagnostic: trace CuPy pool state and both GPU memory-budget estimators
(``estimate_potential_chunk_size`` and ``estimate_scan_batch_size``)
through a scan computation, to see whether Probe+GridScan workloads carry
a tighter VRAM margin than a plain PlaneWave/rotation-series multislice.

A scan stacks TWO single-shot VRAM estimates instead of one:
  1. ``estimate_scan_batch_size`` sizes the probe batch once, at graph-
     construction time, before the potential's per-chunk arrays exist on
     device.
  2. ``estimate_potential_chunk_size`` then sizes potential chunks once
     per probe batch, at computation time, for whatever VRAM the probe
     batch leaves behind -- then holds that size fixed for every chunk in
     that batch's ``generate_chunked_slices`` loop.
Neither estimate is revisited within its own loop, so if real resident
memory drifts (CuPy pool fragmentation across many chunk/batch cycles)
faster than either loop re-checks, a config that "succeeds" briefly can
still run out of memory deeper into the same run -- the same failure mode
already confirmed for the plain rotation-series case (chunk sizes that
looked safe in a short test OOM'd over a longer one).

This script uses ``chunk_size="auto"`` / ``max_batch="auto"`` (the real
production defaults) rather than hand-picked fixed values, logs the full
internal numbers behind every estimator call via
``_chunk_instrumentation.py``, and logs CuPy pool state at every
potential-chunk transition -- so both the joint-budget interaction and
any within-run drift are visible directly, not inferred.

Must run on the GPU machine (needs cupy). Not runnable in this repo's
local dev environment (no cupy/GPU here) -- this file is meant to be
copied to / run on the GPU workstation.

Usage
-----
    python benchmarks/diagnose_scan_vram.py
    python benchmarks/diagnose_scan_vram.py --gpts 4096 4096 --repetitions 20 20 200 \\
        --scan-gpts 4 4 --energies 100e3 150e3 200e3 250e3 300e3
    python benchmarks/diagnose_scan_vram.py --chunk-size 10 --max-batch 1  # old fixed-size behaviour
"""

import argparse
import gc
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from ase.build import bulk

import abtem
from abtem import AnnularDetector, Potential, Probe
from abtem.core import config
from abtem.scan import GridScan

from _chunk_instrumentation import instrument_chunk_estimators

config.set({"dask.lazy": False})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpts", type=int, nargs=2, default=[4096, 4096], metavar=("NY", "NX"))
    p.add_argument("--repetitions", type=int, nargs=3, default=[20, 20, 200], metavar=("RX", "RY", "RZ"),
                    help="Si cubic-cell repetitions (default gives a ~36 GB potential, "
                         "deliberately larger than a single GPU's VRAM to exercise chunking).")
    p.add_argument("--slice-thickness", type=float, default=2.0)
    p.add_argument("--scan-gpts", type=int, nargs=2, default=[4, 4], metavar=("NY", "NX"))
    p.add_argument("--energies", type=float, nargs="+", default=[200e3],
                    help="one or more energies [eV]; pass several for an energy ensemble "
                         "(Probe supports the same multi-energy ensemble axis as PlaneWave).")
    p.add_argument("--chunk-size", default="auto", help="potential.slice-chunk-size ('auto' or an int)")
    p.add_argument("--max-batch", default="auto", help="probe scan batch size ('auto' or an int)")
    args = p.parse_args()

    def _maybe_int(v):
        try:
            return int(v)
        except ValueError:
            return v

    args.chunk_size = _maybe_int(args.chunk_size)
    args.max_batch = _maybe_int(args.max_batch)
    return args


def pool_str(label=""):
    import cupy as cp
    pool = cp.get_default_memory_pool()
    used = pool.used_bytes() / 1e9
    free = pool.free_bytes() / 1e9
    total = pool.total_bytes() / 1e9
    free_cuda, total_cuda = cp.cuda.Device().mem_info
    cuda_free = free_cuda / 1e9
    tag = f"[{label}] " if label else ""
    return (f"{tag}pool used={used:.2f}GB free={free:.2f}GB total={total:.2f}GB | "
            f"cuda_free={cuda_free:.2f}GB")


def install_chunk_loop_tracer(max_chunks_logged=5):
    """Patch _FieldBuilderFromAtoms.generate_chunked_slices to log CuPy pool
    state at every chunk boundary, so within-a-loop drift is visible.

    Separate from ``_chunk_instrumentation.instrument_chunk_estimators()``,
    which logs the *estimators'* internal numbers at the moment they are
    called -- this logs the pool's actual state as the loop those estimates
    govern runs, so the two together show both the prediction and the
    reality it was trying to predict.
    """
    from abtem.potentials.iam import _FieldBuilderFromAtoms

    orig = _FieldBuilderFromAtoms.generate_chunked_slices
    batch_counter = [0]

    def patched(self, first_slice=0, last_slice=None, chunk_size="auto"):
        batch_counter[0] += 1
        batch = batch_counter[0]
        chunk_counter = 0
        print(f"\n  [Batch {batch}] START generate_chunked_slices  {pool_str()}")
        sys.stdout.flush()

        n_chunks_total = 0
        for chunk in orig(self, first_slice=first_slice, last_slice=last_slice, chunk_size=chunk_size):
            chunk_counter += 1
            n_chunks_total += 1
            verbose = chunk_counter <= max_chunks_logged
            if verbose:
                print(f"    [Batch {batch} Chunk {chunk_counter}] yield  {pool_str()}")
                sys.stdout.flush()
            elif chunk_counter == max_chunks_logged + 1:
                print(f"    [Batch {batch}] ... suppressing further per-chunk lines ...")
                sys.stdout.flush()
            yield chunk
            if verbose:
                print(f"    [Batch {batch} Chunk {chunk_counter}] post-propagate  {pool_str()}")
                sys.stdout.flush()

        print(f"  [Batch {batch}] END generate_chunked_slices  "
              f"({n_chunks_total} chunks total)  {pool_str()}")
        sys.stdout.flush()

    _FieldBuilderFromAtoms.generate_chunked_slices = patched


def main():
    args = parse_args()

    import cupy as cp

    instrument_chunk_estimators()
    install_chunk_loop_tracer()

    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    free, total = cp.cuda.Device().mem_info
    print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")

    # Warmup
    a = cp.ones((256, 256), dtype=np.complex64)
    cp.fft.fft2(a)
    cp.cuda.Device().synchronize()
    del a
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\nAfter warmup: {pool_str()}")

    gpts = tuple(args.gpts)
    reps = tuple(args.repetitions)
    scan_gpts = tuple(args.scan_gpts)
    energy = args.energies if len(args.energies) > 1 else args.energies[0]

    atoms = bulk("Si", cubic=True) * reps
    potential = Potential(atoms, gpts=gpts, slice_thickness=args.slice_thickness, device="gpu")
    probe = Probe(energy=energy, semiangle_cutoff=20, device="gpu")
    probe.grid.match(potential)
    detector = AnnularDetector(inner=40, outer=200)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

    n_slices = len(potential)
    print(f"\nPotential: {n_slices} slices, {n_slices * gpts[0] * gpts[1] * 4 / 1e9:.2f} GB "
          f"(gpts={gpts}, chunk_size={args.chunk_size})")
    print(f"Scan: {scan.gpts} = {int(np.prod(scan.gpts))} positions, max_batch={args.max_batch}")
    print(f"Energies: {args.energies} ({'ensemble' if len(args.energies) > 1 else 'single'})")

    gc.collect()
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    print(f"Before scan: {pool_str()}")

    try:
        lazy = probe.scan(
            potential, scan=scan, detectors=detector, lazy=True,
            max_batch=args.max_batch, potential_chunk_size=args.chunk_size,
        )
        lazy.compute()
        print("\nScan completed successfully!")
    except Exception:
        import traceback
        print("\nScan FAILED:")
        traceback.print_exc()
    finally:
        cp.cuda.Device().synchronize()
        print(f"\nAfter scan: {pool_str()}")


if __name__ == "__main__":
    main()
