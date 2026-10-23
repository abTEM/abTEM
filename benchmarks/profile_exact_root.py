"""Profile the parts of one Lanczos iteration of RealSpaceMultislice(order="exact").

One slice of the exact root with the backscatter correction runs three Lanczos
iterations of about ten steps each. A step applies the operator (stencil, FFT, band,
inverse FFT), orthogonalises the result against the Krylov basis, normalises it and
moves two numbers to the host. This script times each part on a batch of waves of the
shape of a benchmark (graphene, 64 x 112 points by default), and a whole run.

For each part it reports

* ``dispatch``: the time that the host needs to queue the part (the loop of calls
  returns, nothing waited for),
* ``total``: the time until the device has finished,
* ``data``: the size of the array that the part reads and writes.

On a GPU, ``dispatch`` close to ``total`` means the part is limited by the host (the
number of calls), ``total`` well above ``dispatch`` that the device is the limit (the
arithmetic or the memory bandwidth). On the CPU the two are the same.

Examples
--------
::

    python benchmarks/profile_exact_root.py --device gpu --batch 256
    python benchmarks/profile_exact_root.py --device gpu --batch 1
    python benchmarks/profile_exact_root.py --device cpu --batch 8
"""

import argparse
import time

import numpy as np


def synchronizer(device):
    if device == "gpu":
        import cupy

        return cupy.cuda.Stream.null.synchronize
    return lambda: None


def measure(function, repeats, sync):
    """(dispatch, total) time per call [ms]."""
    function()
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    dispatched = time.perf_counter()
    sync()
    done = time.perf_counter()
    return (
        1e3 * (dispatched - start) / repeats,
        1e3 * (done - start) / repeats,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch", type=int, default=1, help="waves in the batch")
    parser.add_argument("--energy", type=float, default=1000.0, help="energy [eV]")
    parser.add_argument("--gpts", type=int, nargs=2, default=(64, 112))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--basis", type=int, default=8, help="vectors in the Krylov basis (typical: 8)"
    )
    args = parser.parse_args()

    import ase.build

    import abtem
    from abtem._lanczos import _lanczos_device, _lanczos_host, exponential_of_root
    from abtem.core.backend import asnumpy, cp, get_array_module
    from abtem.core.energy import energy2sigma, energy2wavelength
    from abtem.core.fft import fft2, ifft2
    from abtem.finite_difference import LaplaceOperator, _propagating_band

    device = args.device
    sync = synchronizer(device)
    xp = get_array_module(device)
    nx, ny = args.gpts
    batch, size = args.batch, args.gpts[0] * args.gpts[1]

    atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
    atoms.center(vacuum=3.0, axis=2)
    potential = abtem.Potential(
        atoms,
        gpts=tuple(args.gpts),
        slice_thickness=0.05,
        projection="finite",
        device=device,
    ).build(lazy=False)
    rng = np.random.default_rng(0)
    array = rng.normal(size=(batch, nx, ny)) + 1j * rng.normal(size=(batch, nx, ny))
    a = xp.asarray(array.astype(np.complex64))
    # one wave defines the grid, the sampling and the energy of the stencil and the band
    waves = abtem.Waves(a[0], extent=potential.extent, energy=args.energy)
    wavelength = energy2wavelength(args.energy)
    k = 2 * np.pi / wavelength
    sigma = energy2sigma(args.energy)
    thickness = potential.slice_thickness[0]
    local = potential.array[0] * sigma / thickness
    stencil = LaplaceOperator(6).get_stencil(waves, device=device)
    band = _propagating_band(waves, 0.95, (0.0, 0.0), xp)

    def project(x):
        return ifft2(fft2(x) * band)

    def apply(x):
        return project(stencil(x) / (2 * k) + local * x)

    # the Krylov basis and a residual of the shape of an iteration
    m = args.basis
    basis = xp.asarray(
        (
            rng.normal(size=(batch, m, size)) + 1j * rng.normal(size=(batch, m, size))
        ).astype(np.complex64)
    )
    w = xp.asarray(
        (rng.normal(size=(batch, size)) + 1j * rng.normal(size=(batch, size))).astype(
            np.complex64
        )
    )

    if cp is not None and xp is cp:

        def project_out():
            overlaps = xp.conj(xp.matmul(basis, xp.conj(w)[:, :, None]))[:, :, 0]
            return w - xp.matmul(overlaps[:, None, :], basis)[:, 0]

    else:

        def project_out():
            overlaps = xp.sum(xp.conj(basis) * w[:, None, :], axis=-1)
            return w - xp.sum(overlaps[:, :, None] * basis, axis=1)

    def norm_and_transfer():
        beta = xp.sqrt(xp.sum(xp.abs(w) ** 2, axis=-1))
        return asnumpy(xp.stack([beta.real, beta])).astype(np.float64)

    def normalise():
        beta = xp.sqrt(xp.sum(xp.abs(w) ** 2, axis=-1))
        basis[:, 1] = w / xp.where(beta > 1e-14, beta, 1.0)[:, None]

    tridiagonal = rng.normal(size=(batch, m, m))
    tridiagonal = tridiagonal + tridiagonal.transpose(0, 2, 1)
    function = exponential_of_root(k, thickness)

    def host_work():
        values, vectors = np.linalg.eigh(tridiagonal / np.sqrt(m) + 40.0 * np.eye(m))
        return np.einsum(
            "bij,bj,bj->bi", vectors, function(values), vectors[:, 0, :].conj()
        )

    megabytes = lambda x: x.nbytes / 1e6  # noqa: E731
    parts = [
        ("stencil", lambda: stencil(a), megabytes(a)),
        ("FFT, band, inverse FFT", lambda: project(a), megabytes(a)),
        (
            "potential times the wave, add (2 kernels)",
            lambda: local * a + a,
            megabytes(a),
        ),
        ("operator (the whole apply)", lambda: apply(a), megabytes(a)),
        (f"orthogonalise against {m} vectors", project_out, megabytes(basis)),
        ("norm and one transfer to the host", norm_and_transfer, megabytes(w)),
        ("normalise and store", normalise, megabytes(w)),
        ("host: eigh and the function", host_work, 0),
    ]

    print(
        f"{device}, batch {batch}, grid {nx} x {ny}, {args.energy:g} eV, "
        f"basis {m}; one wave is {a[0].nbytes / 1e3:.0f} kB\n"
    )
    print(f"{'part':42s} {'dispatch [ms]':>14s} {'total [ms]':>11s} {'data [MB]':>10s}")
    results = {}
    for name, function_, data in parts:
        dispatch, total = measure(function_, args.repeats, sync)
        results[name] = (dispatch, total)
        print(f"{name:42s} {dispatch:14.3f} {total:11.3f} {data:10.1f}")

    # the whole iteration, from the pieces (two passes of the orthogonalisation), and
    # a whole Lanczos run, counted in iterations
    estimate = (
        results["operator (the whole apply)"][1]
        + 2 * results[f"orthogonalise against {m} vectors"][1]
        + results["norm and one transfer to the host"][1]
        + results["normalise and store"][1]
        + results["host: eigh and the function"][1]
    )
    iterations = []

    def counted(x):
        iterations.append(1)
        return apply(x)

    lanczos = _lanczos_host if xp is np else _lanczos_device
    run = lambda: lanczos(a, counted, function, 1e-8, 60, xp)  # noqa: E731
    run()
    sync()
    iterations.clear()
    dispatch, total = measure(run, max(3, args.repeats // 5), sync)
    # (measure() runs the function once before the timed loop)
    count = len(iterations) / (max(3, args.repeats // 5) + 1)
    print(
        f"\nsum of the parts for one iteration: {estimate:.3f} ms "
        "(the orthogonalisation counted twice)"
    )
    print(
        f"one Lanczos run: {total:.2f} ms total, {dispatch:.2f} ms dispatch, "
        f"{count:.1f} iterations: {total / count:.3f} ms per iteration"
    )
    print(
        "a slice has three runs, so a slice of the exact root with the correction "
        f"takes about {3 * total:.1f} ms"
    )
    # the bytes of one iteration, for the bandwidth that they would need
    traffic = (6 * megabytes(a) + 4 * megabytes(basis) + 8 * megabytes(w)) * 1e6
    print(
        f"memory traffic of one iteration is about {traffic / 1e6:.0f} MB; at 1 TB/s "
        f"that is {traffic / 1e12 * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    main()
