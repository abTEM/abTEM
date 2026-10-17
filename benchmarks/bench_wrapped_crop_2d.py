"""Old vs new ``wrapped_crop_2d`` (abtem/prism/utils.py) on the GPU.

The old implementation cut the window from up to four slices and concatenated
them, falling back to a modulo-index gather only for windows wider than one
period. The new one always gathers by modulo index, which copies even a window
that lies inside the array. This times both, plus a variant that returns the
slice view when the window does not wrap, at the sizes the two callers use:

* ``SMatrixArray.reduce`` (prism/s_matrix.py): the whole S-matrix chunk,
  (n_waves, H, W), cropped once per position batch -- including the
  window == grid case of interpolation 1, where the old code returned a view.
* PRISM-EELS (inelastic/core_loss.py): many small windows of
  (n_k, H, W) S-matrix waves and of 2-D transmission functions, one per site.

Each case is first checked to give identical arrays in all variants.

Run on a CUDA machine with CuPy installed::

    python benchmarks/bench_wrapped_crop_2d.py            # cupy
    python benchmarks/bench_wrapped_crop_2d.py --device cpu

The script is self-contained: it does not import abTEM, so it runs unchanged
on any branch.
"""

import argparse
import itertools

import numpy as np


def _old_wrapped_slices(start, stop, n):
    size = stop - start
    if size > n:
        raise RuntimeError("exceeds the period")
    start %= n
    stop = start + size
    if stop <= n:
        return slice(start, stop), slice(0, 0)
    return slice(start, None), slice(0, stop - n)


def _gather(xp, array, corner, size):
    i = (corner[0] + xp.arange(size[0])) % array.shape[-2]
    j = (corner[1] + xp.arange(size[1])) % array.shape[-1]
    return array[..., i[:, None], j[None, :]]


def old_wrapped_crop_2d(xp, array, corner, size):
    upper_corner = (corner[0] + size[0], corner[1] + size[1])
    try:
        a, c = _old_wrapped_slices(corner[0], upper_corner[0], array.shape[-2])
        b, d = _old_wrapped_slices(corner[1], upper_corner[1], array.shape[-1])
    except RuntimeError:
        return _gather(xp, array, corner, size)

    A = array[..., a, b]
    B = array[..., c, b]
    D = array[..., c, d]
    C = array[..., a, d]

    if A.size == 0:
        AB = B
    elif B.size == 0:
        AB = A
    else:
        AB = xp.concatenate([A, B], axis=-2)
    if C.size == 0:
        CD = D
    elif D.size == 0:
        CD = C
    else:
        CD = xp.concatenate([C, D], axis=-2)
    if CD.size == 0:
        return AB
    if AB.size == 0:
        return CD
    return xp.concatenate([AB, CD], axis=-1)


def new_wrapped_crop_2d(xp, array, corner, size):
    return _gather(xp, array, corner, size)


def new_with_view_wrapped_crop_2d(xp, array, corner, size):
    """The new gather, but a slice view when the window does not wrap."""
    n0, n1 = array.shape[-2:]
    if 0 <= corner[0] <= n0 - size[0] and 0 <= corner[1] <= n1 - size[1]:
        return array[
            ..., corner[0] : corner[0] + size[0], corner[1] : corner[1] + size[1]
        ]
    return _gather(xp, array, corner, size)


VARIANTS = {
    "old": old_wrapped_crop_2d,
    "new (gather)": new_wrapped_crop_2d,
    "new + view": new_with_view_wrapped_crop_2d,
}


# (label, array shape, list of (corner, size), consumer)
# The consumer mimics what the caller does next, so a view is not timed as free
# when the caller then reads every element anyway.
def cases():
    return [
        (
            "S-matrix reduce, interp=1 (window == grid)",
            (400, 256, 256),
            [((0, 0), (256, 256))],
            "sum",
        ),
        (
            "S-matrix reduce, interp=2, in range",
            (400, 256, 256),
            [((40, 70), (160, 150))],
            "sum",
        ),
        (
            "S-matrix reduce, interp=2, wraps both axes",
            (400, 256, 256),
            [((-40, 200), (160, 150))],
            "sum",
        ),
        (
            "PRISM-EELS s_waves, 64 sites, 64x64 window",
            (200, 256, 256),
            [((r, c), (64, 64)) for r, c in itertools.product(range(-20, 236, 32), repeat=2)],
            "sum",
        ),
        (
            "PRISM-EELS transmission, 64 sites, 64x64 window",
            (256, 256),
            [((r, c), (64, 64)) for r, c in itertools.product(range(-20, 236, 32), repeat=2)],
            "sum",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    if args.device == "gpu":
        import cupy as xp
        from cupyx.profiler import benchmark
    else:
        xp = np
        benchmark = None

    import time

    rng = np.random.default_rng(0)
    print(f"device: {args.device}, repeat: {args.repeat}")
    print(f"{'case':52s} " + " ".join(f"{name:>14s}" for name in VARIANTS))

    for label, shape, crops, consumer in cases():
        host = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex64
        )
        array = xp.asarray(host)

        reference = [old_wrapped_crop_2d(xp, array, c, s) for c, s in crops]
        for name, function in VARIANTS.items():
            for (corner, size), expected in zip(crops, reference):
                got = function(xp, array, corner, size)
                assert got.shape == expected.shape, (name, label, corner)
                assert bool(xp.array_equal(got, expected)), (name, label, corner)

        def run(function):
            total = 0
            for corner, size in crops:
                total = total + function(xp, array, corner, size).sum()
            return total

        timings = []
        for function in VARIANTS.values():
            if benchmark is not None:
                result = benchmark(run, (function,), n_repeat=args.repeat, n_warmup=3)
                timings.append(float(np.median(result.gpu_times)) * 1e3)
            else:
                run(function)
                times = []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    run(function)
                    times.append(time.perf_counter() - start)
                timings.append(float(np.median(times)) * 1e3)

        print(f"{label:52s} " + " ".join(f"{t:11.3f} ms" for t in timings))

    if args.device == "gpu":
        # Peak extra memory of one crop of the largest case, gather vs view.
        array = xp.zeros((400, 256, 256), dtype=np.complex64)
        pool = xp.get_default_memory_pool()
        for name, function in VARIANTS.items():
            pool.free_all_blocks()
            before = pool.used_bytes()
            crop = function(xp, array, (0, 0), (256, 256))
            print(
                f"window == grid, {name:14s}: extra device memory "
                f"{(pool.used_bytes() - before) / 2**20:8.1f} MiB"
            )
            del crop


if __name__ == "__main__":
    main()
