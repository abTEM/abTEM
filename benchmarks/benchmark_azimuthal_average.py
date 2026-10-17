"""Benchmark the step-weighted ``DiffractionPatterns.azimuthal_average`` (#602).

Compares the single-pass average (``abtem.measurements.AZIMUTHAL_STEP_SINGLE_PASS =
True``: the pixels of every bin gathered once and summed in one pass) with the
bin-by-bin average (``False``: a pattern-sized mask and masked sum per bin). For
each pattern size it reports:

- the peak host memory of building the step bounds, before #602 (a meshgrid and
  its temporaries) and now (``tracemalloc``);
- the best wall time of an eager and of a lazy (``--blocks`` blocks) average on
  the chosen device, bin by bin and single pass, and the CuPy memory-pool total
  when the device is a GPU;
- the largest difference between the two averages, relative to their maximum.

Examples
--------
On a CUDA machine::

    python benchmarks/benchmark_azimuthal_average.py --device gpu

CPU only, smaller patterns::

    python benchmarks/benchmark_azimuthal_average.py --device cpu --sizes 512 1024
"""

import argparse
import os
import time
import tracemalloc

os.environ.setdefault("TQDM_DISABLE", "1")

import dask
import numpy as np

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import asnumpy, cp
import abtem.measurements
from abtem.measurements import DiffractionPatterns

CURRENT = DiffractionPatterns._azimuthal_step_bounds


def _previous_step_bounds(coordinates, radial_sampling, width, device="cpu"):
    """The bounds before #602, for their host memory: a meshgrid and its
    temporaries."""
    x, y = np.meshgrid(*coordinates, indexing="ij")
    r = np.sqrt(x**2 + y**2)
    del x, y
    first = np.floor(np.round((r - width / 2) / radial_sampling + 1, 7))
    first = first.astype(np.int32)
    last = np.floor(np.round((r + width / 2) / radial_sampling, 7))
    last = last.astype(np.int32)
    return first, last


def _patterns(n, members, device, lazy, blocks):
    rng = np.random.default_rng(0)
    array = rng.random((members, n, n), dtype=np.float32)
    patterns = DiffractionPatterns(
        array,
        sampling=1 / 20.0,
        fftshift=True,
        metadata={"energy": 200e3},
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(members)))],
    ).copy_to_device(device)
    if lazy:
        patterns = patterns.ensure_lazy(chunks=(max(members // blocks, 1), -1, -1))
    return patterns


def _host_peak(bounds, n):
    coordinates = tuple(np.fft.fftshift(np.fft.fftfreq(n, 1 / n)) for _ in range(2))
    tracemalloc.start()
    bounds(coordinates, 1.0, 1.0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def _run(single_pass, patterns, device, reps):
    abtem.measurements.AZIMUTHAL_STEP_SINGLE_PASS = single_pass
    try:
        times, result = [], None
        if device == "gpu":
            cp.get_default_memory_pool().free_all_blocks()
        for _ in range(reps):
            start = time.perf_counter()
            profiles = patterns.azimuthal_average()
            if profiles.is_lazy:
                with dask.config.set(scheduler="synchronous"):
                    profiles = profiles.compute()
            result = asnumpy(profiles.array)
            if device == "gpu":
                cp.cuda.Device().synchronize()
            times.append(time.perf_counter() - start)
        pool_peak = None
        if device == "gpu":
            pool_peak = cp.get_default_memory_pool().total_bytes()
        return min(times), pool_peak, result
    finally:
        abtem.measurements.AZIMUTHAL_STEP_SINGLE_PASS = True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    if args.device == "gpu" and cp is None:
        raise SystemExit("CuPy is not available; use --device cpu")
    abtem.config.set({"device": args.device})

    print(
        f"{'n':>6} {'mode':>6} {'bounds host MB before/now':>26}"
        f" {'time loop/single s':>20} {'gpu pool loop/single MB':>24} max rel diff"
    )
    for n in args.sizes:
        host = [_host_peak(b, n) / 1e6 for b in (_previous_step_bounds, CURRENT)]
        for lazy in (False, True):
            patterns = _patterns(n, args.members, args.device, lazy, args.blocks)
            old = _run(False, patterns, args.device, args.reps)
            new = _run(True, patterns, args.device, args.reps)
            pool = (
                f"{old[1] / 1e6:9.1f}/{new[1] / 1e6:<9.1f}"
                if args.device == "gpu"
                else f"{'-':>21}"
            )
            difference = np.abs(new[2] - old[2]).max() / np.abs(old[2]).max()
            print(
                f"{n:6d} {'lazy' if lazy else 'eager':>6} "
                f"{host[0]:12.1f}/{host[1]:<12.1f} "
                f"{old[0]:9.3f}/{new[0]:<9.3f} {pool:>24} "
                f"{difference:.1e}"
            )


if __name__ == "__main__":
    main()
