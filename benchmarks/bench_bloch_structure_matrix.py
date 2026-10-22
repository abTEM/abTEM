"""Benchmark: building and solving the Bloch-wave structure matrix for one orientation.

Compares three ways of going from the structure factors to the beam amplitudes:
  old      the structure matrix looked up through pandas on the host from the
           (beams**2, 3) array of Miller index differences, then `xp.linalg.eigh`,
           which copies the matrix (the code before issue #632);
  complex  the matrix gathered on the device and factorized in place;
  real     as complex, for a centrosymmetric crystal: the real symmetric matrix of
           the crystal moved to its inversion center (the default when there is one).

For each: the time to build the matrix and to solve it (minimum over repeats, after a
warm-up and with the device synchronised around each), the peak memory of building and
solving together, and the largest difference of the amplitudes from the old ones,
relative to the largest amplitude. Peak memory is the peak of the CuPy memory pool in
use (gpu) or tracemalloc (cpu, which misses LAPACK's workspace: read it as a lower
bound). On the CPU single precision is solved in double precision by numpy, so the times
there differ from the GPU's in kind.

Usage
-----
    python benchmarks/bench_bloch_structure_matrix.py [device] [g_max] [sg_max]
        [precision] [cif] [old]
    python benchmarks/bench_bloch_structure_matrix.py gpu 6.0 1.0 float32
    python benchmarks/bench_bloch_structure_matrix.py gpu 8.0 1.0 float32 ZrO2.cif 0

The default crystal is monoclinic ZrO2 (baddeleyite, P2_1/c). Pass `old` = 0 to skip the
old version, whose dense eigh needs about 6 N**2 complex numbers on the GPU.
"""

import sys
import time
import tracemalloc

import numpy as np
from ase.io import read
from ase.spacegroup import crystal

import abtem
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.dynamical import (
    _metric,
    calculate_dynamical_scattering,
    calculate_structure_matrix,
)
from abtem.bloch.utils import excitation_errors, ravel_hkl
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.constants import kappa
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.utils import get_dtype

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
g_max = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
sg_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
precision = sys.argv[4] if len(sys.argv) > 4 else "float32"
cif = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
with_old = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
energy = 200e3
thicknesses = np.linspace(50, 500, 10)
repeats = 3


def old_structure_matrix(structure_factor, hkl_source, hkl, cell, gpts):
    """The structure matrix as it was built before #632 ('exact' wave equation)."""
    import pandas as pd

    xp = get_array_module(structure_factor)
    gmh = (hkl[None] - hkl[:, None]).reshape(-1, 3)
    series = pd.Series(asnumpy(structure_factor), index=ravel_hkl(hkl_source, gpts))
    A = series.loc[ravel_hkl(gmh, gpts)].to_numpy(copy=True)
    A = xp.asarray(A.reshape((len(hkl),) * 2), dtype=get_dtype(complex=True))
    prefactor = energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)
    sg = excitation_errors(hkl @ cell.reciprocal(), energy, use_wave_eq="exact")
    diag = xp.asarray(2 / energy2wavelength(energy) * sg, dtype=get_dtype())
    Mii = xp.asarray(_metric(hkl, cell, energy, "exact"), dtype=get_dtype())
    A = A * prefactor * Mii[None] * Mii[:, None]
    xp.fill_diagonal(A, diag)
    return A


def old_dynamical_scattering(A, hkl):
    """The solve as it was before #632: eigh copies the matrix."""
    xp = get_array_module(A)
    v, C = xp.linalg.eigh(A)
    gamma = v * energy2wavelength(energy) / 2.0
    initial = xp.asarray(np.all(hkl == 0, axis=1), dtype=get_dtype(complex=True))
    alpha = xp.conjugate(C.T) @ initial
    z = xp.asarray(thicknesses, dtype=gamma.dtype)
    phases = xp.exp(2.0j * xp.pi * z[None] * gamma[:, None])
    return (C @ (phases * alpha[:, None])).T


def synchronize():
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def peak_memory(function):
    """Peak memory of one call in MB."""
    if device == "gpu":
        import cupy as cp

        pool = cp.get_default_memory_pool()

        class PeakHook(cp.cuda.MemoryHook):
            peak = 0

            def malloc_postprocess(self, **kwargs):
                self.peak = max(self.peak, pool.used_bytes())

        synchronize()
        pool.free_all_blocks()
        baseline = pool.used_bytes()  # the structure factors, which all versions hold
        hook = PeakHook()
        hook.peak = baseline
        with hook:
            function()
            synchronize()
        return (hook.peak - baseline) / 1e6
    tracemalloc.start()
    function()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1e6


def timed(function, *args):
    synchronize()
    start = time.perf_counter()
    result = function(*args)
    synchronize()
    return time.perf_counter() - start, result


atoms = (
    read(cif)
    if cif
    else crystal(
        ["Zr", "O", "O"],
        basis=[
            (0.2754, 0.0395, 0.2083),
            (0.0700, 0.3317, 0.3447),
            (0.4496, 0.7569, 0.4792),
        ],
        spacegroup=14,
        cellpar=[5.1505, 5.2116, 5.3173, 90, 99.23, 90],
    )
)

with abtem.config.set({"precision": precision}):
    structure_factor = StructureFactor(atoms, g_max=2 * g_max, device=device)
    built = structure_factor.build(lazy=False)
    bloch_waves = BlochWaves(built, energy=energy, sg_max=sg_max, device=device)
    hkl, cell = bloch_waves.hkl, bloch_waves.cell
    n = len(hkl)
    itemsize = np.dtype(get_dtype(complex=True)).itemsize
    print(
        f"device={device} {precision} {atoms.get_chemical_formula()} g_max={g_max} "
        f"sg_max={sg_max}: {n} beams, complex matrix {n**2 * itemsize / 1e9:.2f} GB, "
        f"inversion center {built.inversion_center}"
    )

    def new_build(center):
        return calculate_structure_matrix(
            built.array, built.hkl, hkl, cell, energy, built.gpts,
            inversion_center=center,
        )

    def new_solve(A, center):
        return calculate_dynamical_scattering(
            A, hkl, cell, energy, thicknesses, inversion_center=center,
            overwrite_structure_matrix=True,
        )

    variants = {}
    if with_old:
        variants["old"] = (
            lambda: old_structure_matrix(built.array, built.hkl, hkl, cell, built.gpts),
            lambda A: old_dynamical_scattering(A, hkl),
        )
    variants["complex"] = (lambda: new_build(None), lambda A: new_solve(A, None))
    if built.inversion_center is not None:
        center = built.inversion_center
        variants["real"] = (lambda: new_build(center), lambda A: new_solve(A, center))

    reference = None
    for label, (build, solve) in variants.items():
        solve(build())  # warm-up
        build_times, solve_times = [], []
        for _ in range(repeats):
            t_build, A = timed(build)
            t_solve, amplitudes = timed(solve, A)
            del A
            build_times.append(t_build)
            solve_times.append(t_solve)
        peak = peak_memory(lambda: solve(build()))
        amplitudes = asnumpy(amplitudes)
        if reference is None:
            reference = amplitudes
        difference = np.abs(amplitudes - reference).max() / np.abs(reference).max()
        print(
            f"  {label:<8} build {min(build_times):8.3f} s   solve "
            f"{min(solve_times):8.3f} s   peak {peak:9.0f} MB "
            f"({peak * 1e6 / (n**2 * itemsize):4.1f} complex matrices)   "
            f"difference {difference:.1e}"
        )
