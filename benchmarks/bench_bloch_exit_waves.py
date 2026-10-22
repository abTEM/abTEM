"""Benchmark: separable plane-wave synthesis of Bloch-wave exit waves.

Compares `plane_wave_expansion` (two matmuls over the beams) with the formula it
replaced, which formed the (thickness, beam, x, y) array of plane waves and summed it.
Two measurements per precision:
  1. the synthesis alone, for one orientation;
  2. `calculate_exit_waves` over a rotation series (eager), with the old formula
     patched in for the "old" rows.

Time is the minimum over repeats, after a warm-up call and a device synchronisation.
Peak memory is tracemalloc (cpu), the peak of the CuPy memory pool in use (gpu) or not measured (mps).
The difference column is the largest absolute difference between old and new relative to
the largest value, which in float32 is dominated by the rounding of the phases.

Usage
-----
    python benchmarks/bench_bloch_exit_waves.py [device] [g_max] [gpts] [n_rotations]
    python benchmarks/bench_bloch_exit_waves.py gpu 4.0 256 8
"""

import sys
import time
import tracemalloc

import numpy as np
from ase.build import bulk

import abtem
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch import dynamical
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.utils import get_dtype

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
g_max = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
n_rotations = int(sys.argv[4]) if len(sys.argv) > 4 else 8
thicknesses = np.linspace(20, 100, 5)
repeats = 5


def old_plane_wave_expansion(amplitudes, g, x, y, z):
    """The plane wave basis and reduction that `plane_wave_expansion` replaced."""
    xp = get_array_module(x)
    g = xp.asarray(g)
    plane_waves_x = complex_exponential(
        2 * np.pi * g[None, :, 0, None, None] * x[None, None, :, None]
    )
    plane_waves_y = complex_exponential(
        2 * np.pi * g[None, :, 1, None, None] * y[None, None, None, :]
    )
    plane_waves_z = complex_exponential(
        2 * np.pi * g[None, :, 2, None, None] * z[..., None, None, None]
    )
    basis = plane_waves_x * plane_waves_y * plane_waves_z
    if not z.ndim:
        basis = basis[0]
    return (amplitudes[..., None, None] * basis).sum(-3)


new_plane_wave_expansion = dynamical.plane_wave_expansion


def synchronize():
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif device == "mps":
        import torch

        torch.mps.synchronize()


def peak_memory(function):
    """Peak memory of one call in MB, or nan where it is not measured."""
    if device == "cpu":
        tracemalloc.start()
        function()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak / 1e6
    if device == "gpu":
        import cupy as cp

        pool = cp.get_default_memory_pool()

        class PeakHook(cp.cuda.MemoryHook):
            peak = 0

            def malloc_postprocess(self, **kwargs):
                self.peak = max(self.peak, pool.used_bytes())

        synchronize()
        pool.free_all_blocks()
        baseline = pool.used_bytes()  # the inputs, which both versions hold
        hook = PeakHook()
        hook.peak = baseline
        with hook:
            function()
            synchronize()
        return (hook.peak - baseline) / 1e6
    return float("nan")


def measure(function):
    result = function()  # warm-up: compiles kernels, fills caches
    synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize()
        times.append(time.perf_counter() - start)
    return min(times), peak_memory(function), result


def report(label, results):
    (t_old, m_old, old), (t_new, m_new, new) = results
    old, new = asnumpy(old), asnumpy(new)
    difference = np.abs(old - new).max() / np.abs(old).max()
    print(
        f"  {label:<32} old {t_old * 1e3:9.1f} ms {m_old:9.0f} MB   "
        f"new {t_new * 1e3:9.1f} ms {m_new:9.0f} MB   "
        f"speedup {t_old / t_new:6.1f}x   difference {difference:.1e}"
    )


atoms = bulk("Si", cubic=True)
print(f"device={device} g_max={g_max} gpts={n}x{n} rotations={n_rotations}")

for precision in ("float64", "float32"):
    if device == "mps" and precision == "float64":
        continue  # emulated float64 is a different benchmark

    with abtem.config.set({"precision": precision}):
        structure_factor = StructureFactor(
            atoms,
            g_max=g_max,
            parametrization="lobato",
            centering="F",
            device=device,
        )
        bloch_waves = BlochWaves(
            structure_factor=structure_factor,
            energy=100e3,
            sg_max=0.5,
            device=device,
        )
        print(f"{precision}: {len(bloch_waves.hkl)} beams")

        # 1. the synthesis alone
        xp = get_array_module(bloch_waves.device)
        dtype = get_dtype()
        # on the device, as the amplitudes are in a calculation on a CUDA device
        amplitudes = xp.asarray(
            bloch_waves._calculate_array(thicknesses.astype(dtype), lazy=False)
        )
        extent = tuple(bloch_waves.cell[i, i] for i in range(2))
        x = xp.asarray(np.linspace(0, extent[0], n, endpoint=False), dtype=dtype)
        y = xp.asarray(np.linspace(0, extent[1], n, endpoint=False), dtype=dtype)
        z = xp.asarray(thicknesses, dtype=dtype)
        g = xp.asarray(bloch_waves.g_vec, dtype=dtype)
        report(
            "synthesis, 5 thicknesses",
            [
                measure(lambda f=f: f(amplitudes, g, x, y, z))
                for f in (old_plane_wave_expansion, new_plane_wave_expansion)
            ],
        )
        z0 = xp.asarray(thicknesses[0], dtype=dtype)
        amplitudes0 = amplitudes[0]
        report(
            "synthesis, scalar thickness",
            [
                measure(lambda f=f: f(amplitudes0, g, x, y, z0))
                for f in (old_plane_wave_expansion, new_plane_wave_expansion)
            ],
        )

        # 2. the whole calculation over a rotation series
        rotated = bloch_waves.rotate("x", np.linspace(0, 0.02, n_rotations))
        results = []
        for f in (old_plane_wave_expansion, new_plane_wave_expansion):
            dynamical.plane_wave_expansion = f
            results.append(
                measure(
                    lambda: rotated.calculate_exit_waves(
                        thicknesses, gpts=(n, n), lazy=False
                    ).array
                )
            )
        dynamical.plane_wave_expansion = new_plane_wave_expansion
        report(f"calculate_exit_waves, {n_rotations} rot.", results)
