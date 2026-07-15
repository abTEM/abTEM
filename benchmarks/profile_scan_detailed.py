"""
Detailed memory profiling of the scan multislice path.

Instruments key functions to log memory usage at each step.
"""

import gc
import os
import sys
import time
import tracemalloc

os.environ["TQDM_DISABLE"] = "1"

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from ase.build import bulk

import abtem
from abtem import AnnularDetector, Potential, Probe
from abtem.core import config
from abtem.scan import GridScan


def fmt_mb(nbytes):
    return f"{nbytes / 1e6:.1f} MB"


def mem_snapshot(label):
    """Print current and peak tracemalloc memory."""
    current, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={fmt_mb(current)}  peak={fmt_mb(peak)}")


def main():
    config.set({"device": "cpu", "dask.lazy": False})

    gpts = (1024, 1024)
    reps = (10, 10, 30)
    scan_gpts = (4, 4)
    max_batch = 2

    print(f"Detailed Memory Profile — CPU, gpts={gpts}")
    print(f"{'=' * 60}")

    atoms = bulk("Si", cubic=True) * reps
    potential = Potential(atoms, gpts=gpts, slice_thickness=2.0, device="cpu")
    num_slices = len(potential)
    print(f"Potential: {num_slices} slices")

    # Monkey-patch key functions to log memory
    from abtem import multislice as ms

    _orig_multislice_and_detect = ms.multislice_and_detect
    _orig_conventional_step = ms.conventional_multislice_step

    call_count = [0]

    def traced_multislice_and_detect(waves, potential, **kwargs):
        mem_snapshot(f"multislice_and_detect START, waves.shape={waves.shape}")
        result = _orig_multislice_and_detect(waves, potential, **kwargs)
        mem_snapshot(f"multislice_and_detect END")
        return result

    step_count = [0]
    def traced_conventional_step(waves, potential_slice, **kwargs):
        step_count[0] += 1
        if step_count[0] <= 3 or step_count[0] % 20 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"    step {step_count[0]}: current={fmt_mb(current)}  peak={fmt_mb(peak)}")
        return _orig_conventional_step(waves, potential_slice, **kwargs)

    ms.multislice_and_detect = traced_multislice_and_detect
    # Can't easily patch the inner step since it's a closure, but let's trace the chunked slices

    # Patch generate_chunked_slices to log chunk builds
    from abtem.potentials import iam
    _orig_generate_chunked = iam._FieldBuilderFromAtoms.generate_chunked_slices

    def traced_generate_chunked(self, first_slice=0, last_slice=None, chunk_size="auto"):
        for i, chunk in enumerate(_orig_generate_chunked(self, first_slice, last_slice, chunk_size)):
            mem_snapshot(f"  chunk {i} built ({len(chunk)} slices)")
            yield chunk
            mem_snapshot(f"  chunk {i} consumed")

    iam._FieldBuilderFromAtoms.generate_chunked_slices = traced_generate_chunked

    # Run
    gc.collect()
    tracemalloc.start()

    probe = Probe(energy=200e3, semiangle_cutoff=20, device="cpu")
    detector = AnnularDetector(inner=40, outer=200)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=scan_gpts)

    mem_snapshot("before scan()")
    lazy_result = probe.scan(
        potential, scan=scan, detectors=detector, lazy=True,
        max_batch=max_batch, potential_chunk_size=1,
    )
    mem_snapshot("after scan(), before compute()")
    lazy_result.compute()
    mem_snapshot("after compute()")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nFinal: peak={fmt_mb(peak)}")


if __name__ == "__main__":
    main()
