"""Old vs new ``wrapped_crop_2d`` (abtem/prism/utils.py), on the GPU or the CPU.

Three implementations of the periodic window crop are timed:

* ``old``: up to four slices joined by three concatenations (two passes over a
  window that wraps both axes), falling back to a modulo-index gather for
  windows wider than one period. A window that does not wrap is a view.
* ``gather``: the modulo-index gather alone (``a[..., i[:, None], j[None, :]]``),
  tried and reverted (commit f4ee9a47): it copies every window, also the
  window == grid of interpolation 1, and many small gathers are slow on the GPU.
* ``new``: a view when the window does not wrap; otherwise slice copies of the
  contiguous runs into one preallocated array (one pass, no index arrays).

Each case is timed together with what its caller does next with the crop, so
that a view is not timed as free when the caller then reads it, and a copy is
not charged when the caller would copy anyway:

* ``SMatrixArray._reduce_to_waves`` (prism/s_matrix.py) crops the whole
  S-matrix chunk (n_waves, H, W) once per batch of positions and contracts it
  with the position coefficients (``tensordot``, which copies a strided view).
* PRISM-EELS (inelastic/core_loss.py) crops a small window of the (n_k, H, W)
  S-matrix waves per site and multiplies it with the transition potentials, and
  crops the 2-D transmission function per site and stacks the crops.

The last block isolates why the old code looked 15x slower than the gather on
an in-range interpolation-2 crop in the first GPU run of this script: there the
crop was followed by ``.sum()``, and a full reduction of a *strided* view runs
far slower on CuPy than one of a contiguous array (CUB handles only contiguous
input; the fallback kernel reduces to one value in a single block). The crop
itself is a view and costs nothing; adding a contiguous copy before the sum, or
the caller's own ``tensordot``, removes the difference.

Every case first checks that all implementations give identical arrays.

Run on a CUDA machine with CuPy installed::

    python benchmarks/bench_wrapped_crop_2d.py               # cupy
    python benchmarks/bench_wrapped_crop_2d.py --device cpu  # numpy

``--consumer sum`` follows every crop with ``.sum()`` instead of the caller's
operation, as the first version of this script did.

The script is self-contained: it does not import abTEM, so it runs unchanged
on any branch.
"""

import argparse
import itertools
import time
import tracemalloc

import numpy as np

# --- old: origin/dev before this change --------------------------------------


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


# --- gather: the reverted attempt (f4ee9a47) ----------------------------------


def gather_wrapped_crop_2d(xp, array, corner, size):
    return _gather(xp, array, corner, size)


# --- new: this branch ---------------------------------------------------------


def _periodic_runs(start, size, n):
    runs = []
    source = start % n
    done = 0
    while done < size:
        length = min(n - source, size - done)
        runs.append((slice(source, source + length), slice(done, done + length)))
        done += length
        source = 0
    return runs


def new_wrapped_crop_2d(xp, array, corner, size):
    rows = _periodic_runs(corner[0], size[0], array.shape[-2])
    columns = _periodic_runs(corner[1], size[1], array.shape[-1])

    if len(rows) == 1 and len(columns) == 1:
        return array[..., rows[0][0], columns[0][0]]

    cropped = xp.empty_like(array, shape=array.shape[:-2] + tuple(size))
    for source_rows, target_rows in rows:
        for source_columns, target_columns in columns:
            cropped[..., target_rows, target_columns] = array[
                ..., source_rows, source_columns
            ]
    return cropped


VARIANTS = {
    "old": old_wrapped_crop_2d,
    "gather": gather_wrapped_crop_2d,
    "new": new_wrapped_crop_2d,
}


# --- what the callers do with the crop ----------------------------------------


def consume_sum(xp, crops, aux):
    total = 0
    for crop in crops:
        total = total + crop.sum()
    return total


def consume_contiguous_sum(xp, crops, aux):
    total = 0
    for crop in crops:
        total = total + xp.ascontiguousarray(crop).sum()
    return total


def consume_tensordot(xp, crops, aux):
    # SMatrixArray._reduce_to_waves: (n_positions, n_waves) x (n_waves, h, w)
    return [xp.tensordot(aux, crop, axes=[-1, -3]) for crop in crops]


def consume_multiply(xp, crops, aux):
    # PRISM-EELS _scatter_at_site: (n_T, h, w)[:, None] * (n_k, h, w)[None]
    return [aux[:, None] * crop[None] for crop in crops]


def consume_stack(xp, crops, aux):
    # PRISM-EELS double channel: the per-site transmission crops, stacked
    return xp.stack(crops)[:, None]


def eels_corners():
    return [
        ((r, c), (64, 64)) for r, c in itertools.product(range(-20, 236, 32), repeat=2)
    ]


# (label, array shape, list of (corner, size), consumer, auxiliary operand shape)
def cases():
    smatrix = (400, 256, 256)
    coefficients = (16, 400)
    return [
        (
            "S-matrix reduce, interp=1 (window == grid)",
            smatrix,
            [((0, 0), (256, 256))],
            consume_tensordot,
            coefficients,
        ),
        (
            "S-matrix reduce, interp=2, in range",
            smatrix,
            [((40, 70), (160, 150))],
            consume_tensordot,
            coefficients,
        ),
        (
            "S-matrix reduce, interp=2, wraps one axis",
            smatrix,
            [((40, 200), (160, 150))],
            consume_tensordot,
            coefficients,
        ),
        (
            "S-matrix reduce, interp=2, wraps both axes",
            smatrix,
            [((-40, 200), (160, 150))],
            consume_tensordot,
            coefficients,
        ),
        (
            "S-matrix reduce, box wider than the grid",
            smatrix,
            [((-40, -30), (300, 290))],
            consume_tensordot,
            coefficients,
        ),
        (
            "PRISM-EELS s_waves, 64 sites, 64x64 window",
            (200, 256, 256),
            eels_corners(),
            consume_multiply,
            (3, 64, 64),
        ),
        (
            "PRISM-EELS transmission, 64 sites, 64x64 window",
            (256, 256),
            eels_corners(),
            consume_stack,
            None,
        ),
    ]


def cause_cases():
    smatrix = (400, 256, 256)
    crop = [((40, 70), (160, 150))]
    return [
        ("interp=2 in range + sum", smatrix, crop, consume_sum, None),
        (
            "interp=2 in range + contiguous copy + sum",
            smatrix,
            crop,
            consume_contiguous_sum,
            None,
        ),
        ("interp=2 in range + tensordot", smatrix, crop, consume_tensordot, (16, 400)),
    ]


def random_complex(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--consumer",
        choices=["caller", "sum"],
        default="caller",
        help="follow each crop with what its caller does (default), or with .sum()",
    )
    args = parser.parse_args()

    if args.device == "gpu":
        import cupy as xp
        from cupyx.profiler import benchmark
    else:
        xp = np
        benchmark = None

    rng = np.random.default_rng(0)

    def time_case(label, shape, crops, consumer, aux_shape):
        array = xp.asarray(random_complex(rng, shape))
        aux = None if aux_shape is None else xp.asarray(random_complex(rng, aux_shape))

        reference = [old_wrapped_crop_2d(xp, array, c, s) for c, s in crops]
        for name, function in VARIANTS.items():
            for (corner, size), expected in zip(crops, reference):
                got = function(xp, array, corner, size)
                assert got.shape == expected.shape, (name, label, corner)
                assert bool(xp.array_equal(got, expected)), (name, label, corner)

        def run(function):
            return consumer(xp, [function(xp, array, c, s) for c, s in crops], aux)

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

    header = f"{'case':52s} " + " ".join(f"{name:>14s}" for name in VARIANTS)
    print(f"device: {args.device}, repeat: {args.repeat}, median times")
    then = "what the caller does next" if args.consumer == "caller" else ".sum()"
    print(f"\ncrop + {then}")
    print(header)
    for label, shape, crops, consumer, aux_shape in cases():
        if args.consumer == "sum":
            consumer, aux_shape = consume_sum, None
        time_case(label, shape, crops, consumer, aux_shape)

    print("\nthe old in-range interp=2 slowness: the same crop, three consumers")
    print(header)
    for case in cause_cases():
        time_case(*case)

    # Extra memory of one crop of the window == grid case: a view costs none.
    print()
    array = xp.zeros((400, 256, 256), dtype=np.complex64)
    for name, function in VARIANTS.items():
        if args.device == "gpu":
            pool = xp.get_default_memory_pool()
            pool.free_all_blocks()
            before = pool.used_bytes()
            crop = function(xp, array, (0, 0), (256, 256))
            extra = pool.used_bytes() - before
        else:
            tracemalloc.start()
            crop = function(xp, array, (0, 0), (256, 256))
            extra = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
        print(
            f"window == grid, {name:7s}: extra {args.device} memory "
            f"{extra / 2**20:8.1f} MiB (array {array.nbytes / 2**20:.1f} MiB)"
        )
        del crop


if __name__ == "__main__":
    main()
