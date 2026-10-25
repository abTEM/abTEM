"""Benchmark: the real transforms (`rfft2`, `irfft2`) against the complex ones for the
potential of the default (infinite) projection, on a CPU, a CUDA GPU or Metal.

Two parts: the transforms of one slice of the integrator alone (what the potential does
per element and slice: deltas -> FFT -> multiply by a real factor -> inverse FFT), and
the build of a potential of a few slices (where the transforms are most of the time).
The complex path is forced by making `real_fft2_available` return False, so both run the
same code otherwise; the results are compared. The device is synchronised around every
timing, the variants are interleaved, and the median of the repeats is reported.

Usage
-----
    python benchmarks/bench_real_fft_potential.py [options]

    --device cpu|gpu|mps   (default mps if Metal is usable, else gpu if CUDA is, else cpu)
    --gpts N               grid points per side (default 2048)
    --slices N             slices of the potential (default 8)
    --repeats N            (default 5)
    --fft numpy|fftw|mkl   the CPU library; the real transforms need numpy, the others
                           keep the complex ones (default numpy)
    --precision float32|float64   (default float32; Metal has no float64)
"""

import argparse
import statistics
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--device", choices=["cpu", "gpu", "mps"])
parser.add_argument("--gpts", type=int, default=2048)
parser.add_argument("--slices", type=int, default=8)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--fft", default="numpy")
parser.add_argument("--precision", default="float32", choices=["float32", "float64"])
args = parser.parse_args()

import abtem  # noqa: E402
import abtem.integrals as integrals  # noqa: E402
from abtem.core import backend  # noqa: E402
from abtem.core.backend import asnumpy, get_array_module  # noqa: E402
from abtem.core.fft import fft2, ifft2, irfft2, rfft2  # noqa: E402


def default_device():
    try:
        backend.check_mps_is_available()
        return "mps"
    except RuntimeError:
        pass
    try:
        backend.check_cupy_is_installed()
        import cupy

        if cupy.cuda.runtime.getDeviceCount():
            return "gpu"
    except Exception:
        pass
    return "cpu"


device = args.device or default_device()
abtem.config.set(
    {
        "device": device,
        "precision": args.precision,
        "fft": args.fft,
        "diagnostics.progress_bar": False,
        "diagnostics.task_progress": False,
    }
)
xp = get_array_module(device)
dtype = np.dtype(args.precision)
complex_dtype = np.result_type(dtype, np.complex64)


def synchronize():
    if device == "gpu":
        import cupy

        cupy.cuda.Stream.null.synchronize()
    elif device == "mps":
        import torch

        torch.mps.synchronize()


def timed(function, repeats):
    function()  # warm-up: kernels, FFT plans
    synchronize()
    seconds = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        function()
        synchronize()
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds), min(seconds), max(seconds)


def interleaved(variants, repeats):
    """Median, minimum and maximum of each variant, the variants taking turns."""
    for function in variants.values():
        function()
    synchronize()
    seconds = {name: [] for name in variants}
    for _ in range(repeats):
        for name, function in variants.items():
            synchronize()
            start = time.perf_counter()
            function()
            synchronize()
            seconds[name].append(time.perf_counter() - start)
    return {name: (statistics.median(v), min(v), max(v)) for name, v in seconds.items()}


n = args.gpts
print(
    f"device {device}, grid {n} x {n}, {args.slices} slices, {args.precision}, "
    f"fft {args.fft}, abtem {abtem.__version__}",
    flush=True,
)

# --- part 1: the transforms of one slice of one element ------------------------------
rng = np.random.default_rng(0)
deltas = xp.asarray(
    (rng.random((n, n)) < 2e-4).astype(dtype)
)  # a sparse real array, like the atoms of a slice
factor = xp.asarray(
    np.exp(-np.hypot(*np.meshgrid(*[np.fft.fftfreq(n)] * 2)) * 5).astype(dtype)
)


def complex_transforms():
    spectrum = fft2(deltas.astype(complex_dtype), overwrite_x=True)
    spectrum *= factor
    return ifft2(spectrum, overwrite_x=True).real


def real_transforms():
    spectrum = rfft2(deltas)
    spectrum *= factor[:, : spectrum.shape[-1]]
    return irfft2(spectrum, s=(n, n))


difference = float(
    np.abs(asnumpy(complex_transforms()) - asnumpy(real_transforms())).max()
    / np.abs(asnumpy(complex_transforms())).max()
)
results = interleaved(
    {
        "complex (fft2, ifft2)": complex_transforms,
        "real (rfft2, irfft2)": real_transforms,
    },
    args.repeats,
)
print("\none slice of one element: deltas -> FFT -> factor -> inverse FFT")
for name, (median, low, high) in results.items():
    print(f"  {name:<24}{1e3 * median:9.2f} ms  ({1e3 * low:.2f}-{1e3 * high:.2f})")
speedup = results["complex (fft2, ifft2)"][0] / results["real (rfft2, irfft2)"][0]
print(
    f"  real is {speedup:.2f}x; largest difference {difference:.1e} of the largest value"
)

# --- part 2: the build of a potential --------------------------------------------------
from ase.spacegroup import crystal  # noqa: E402

unit = crystal(
    ["Zr", "O", "O"],
    basis=[
        (0.2754, 0.0395, 0.2083),
        (0.0700, 0.3317, 0.3447),
        (0.4423, 0.7569, 0.4792),
    ],
    spacegroup=14,
    cellpar=[5.151, 5.212, 5.317, 90, 99.23, 90],
)
# a box of about 0.1 A per pixel, as many unit cells as the grid and slices need
repeats = (int(np.ceil(0.1 * n / 5.2)), int(np.ceil(0.1 * n / 5.2)), 2)
atoms = unit * repeats
atoms.set_cell([0.1 * n, 0.1 * n, args.slices * 1.0])
atoms.set_pbc(False)
atoms = atoms[
    (atoms.positions[:, 0] < 0.1 * n)
    & (atoms.positions[:, 1] < 0.1 * n)
    & (atoms.positions[:, 2] < args.slices * 1.0)
]


def build(real):
    original = integrals.real_fft2_available
    if not real:
        integrals.real_fft2_available = lambda array: False
    try:
        potential = abtem.Potential(
            atoms,
            gpts=(n, n),
            slice_thickness=1.0,
            device=device,
            projection="infinite",
        )
        return potential.build(lazy=False).array
    finally:
        integrals.real_fft2_available = original


reference = asnumpy(build(False))
real_result = asnumpy(build(True))
difference = float(np.abs(reference - real_result).max() / np.abs(reference).max())
results = interleaved(
    {
        "complex transforms": lambda: build(False),
        "real transforms": lambda: build(True),
    },
    max(args.repeats // 2, 3),
)
print(f"\nbuild of a potential of {args.slices} slices, {len(atoms)} atoms")
for name, (median, low, high) in results.items():
    print(
        f"  {name:<24}{median:9.3f} s  ({low:.3f}-{high:.3f})   "
        f"{1e3 * median / args.slices:8.1f} ms per slice"
    )
speedup = results["complex transforms"][0] / results["real transforms"][0]
print(
    f"  real is {speedup:.2f}x; largest difference {difference:.1e} of the largest value"
)
