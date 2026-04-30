"""
Diagnostic: trace CuPy pool state through a 36 GB scan computation.

Patches generate_chunked_slices and multislice_and_detect to log pool
used_bytes / total_bytes at each key transition point. Run on GPU only.

Usage:
    python benchmarks/diagnose_scan_vram.py
"""

import os, sys, gc
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from ase.build import bulk
import cupy as cp

import abtem
from abtem import AnnularDetector, Potential, Probe
from abtem.core import config
from abtem.scan import GridScan

config.set({"dask.lazy": False})


# ──────────────────────────────────────────────────────────────────────
# Pool helpers
# ──────────────────────────────────────────────────────────────────────

def pool_str(label=""):
    pool = cp.get_default_memory_pool()
    used = pool.used_bytes() / 1e9
    free = pool.free_bytes() / 1e9
    total = pool.total_bytes() / 1e9
    free_cuda, total_cuda = cp.cuda.Device().mem_info
    cuda_free = free_cuda / 1e9
    tag = f"[{label}] " if label else ""
    return (f"{tag}pool used={used:.2f}GB free={free:.2f}GB total={total:.2f}GB | "
            f"cuda_free={cuda_free:.2f}GB")


# ──────────────────────────────────────────────────────────────────────
# Monkey-patch generate_chunked_slices to log pool state
# ──────────────────────────────────────────────────────────────────────

from abtem.potentials.iam import _FieldBuilderFromAtoms
_orig_generate_chunked_slices = _FieldBuilderFromAtoms.generate_chunked_slices

_batch_counter = [0]
_chunk_counter = [0]

def _patched_generate_chunked_slices(self, first_slice=0, last_slice=None, chunk_size="auto"):
    _batch_counter[0] += 1
    batch = _batch_counter[0]
    _chunk_counter[0] = 0
    print(f"\n  [Batch {batch}] START generate_chunked_slices  {pool_str()}")
    sys.stdout.flush()

    for chunk in _orig_generate_chunked_slices(self, first_slice=first_slice, last_slice=last_slice, chunk_size=chunk_size):
        _chunk_counter[0] += 1
        print(f"    [Batch {batch} Chunk {_chunk_counter[0]}] yield  {pool_str()}")
        sys.stdout.flush()
        yield chunk
        print(f"    [Batch {batch} Chunk {_chunk_counter[0]}] post-propagate  {pool_str()}")
        sys.stdout.flush()

    print(f"  [Batch {batch}] END generate_chunked_slices  {pool_str()}")
    sys.stdout.flush()

_FieldBuilderFromAtoms.generate_chunked_slices = _patched_generate_chunked_slices


# ──────────────────────────────────────────────────────────────────────
# Run a minimal scan
# ──────────────────────────────────────────────────────────────────────

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

# 36 GB potential, 4x4 scan, batch=1, chunk=10
atoms = bulk("Si", cubic=True) * (20, 20, 200)
potential = Potential(atoms, gpts=(4096, 4096), slice_thickness=2.0, device="gpu")
probe = Probe(energy=200e3, semiangle_cutoff=20, device="gpu")
probe.grid.match(potential)
detector = AnnularDetector(inner=40, outer=200)
scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))

print(f"\nPotential: {len(potential)} slices, {len(potential)*4096*4096*4/1e9:.2f} GB")
print(f"Scan: {scan.gpts} = {np.prod(scan.gpts)} positions, batch=1")

gc.collect()
cp.cuda.Device().synchronize()
cp.get_default_memory_pool().free_all_blocks()
print(f"Before scan: {pool_str()}")

try:
    lazy = probe.scan(potential, scan=scan, detectors=detector, lazy=True,
                      max_batch=1, potential_chunk_size=10)
    lazy.compute()
    print("\nScan completed successfully!")
except Exception as e:
    print(f"\nScan FAILED: {type(e).__name__}: {e}")
finally:
    cp.cuda.Device().synchronize()
    print(f"\nAfter scan: {pool_str()}")
