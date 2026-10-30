"""Benchmark: Bloch-wave propagation by band reduction and a Chebyshev expansion
(`abtem.bloch.band_chebyshev`, the solver 'band'), against the eigendecomposition
('eigh'), stage by stage.

The amplitudes psi(z) = M D^H exp(i c_z A) e_0, c_z = pi lambda z, are computed from the
structure matrix A without its eigenvectors:

  1. bounds on the spectrum from Gershgorin discs of A (rigorous, and close: the
     couplings are small next to the excitation errors);
  2. A = Q B Q^H with B banded (bandwidth b), in place: Householder QR of each panel
     below the band, and a two-sided update of the trailing matrix by matrix products
     (compute-bound, unlike the one-stage tridiagonalization of eigh, which is limited
     by memory bandwidth). The reflectors are kept below the band, where the panel was
     annihilated, so the matrix is the only large array. The first b rows are never
     transformed: with the direct beam first, Q^H e_0 = e_0;
  3. exp(i c B) e_0 for every thickness at once, by a Chebyshev expansion of the band
     matrix: one three-term recurrence (a fused kernel per term on the GPU);
  4. Q applied to the few result vectors with the stored block reflectors.

Each stage is timed (device synchronised around it), with the peak memory of the whole
and the difference from the eigendecomposition (`eigh`, as in
`calculate_dynamical_scattering`), relative to the largest amplitude. With `reference` =
1 both are also compared with the eigendecomposition in double precision, computed once
per case, on the CPU when it does not fit on the device (minutes, and tens of GB of host
memory, at tens of thousands of beams). Both methods are first run on a small problem,
so that the timings leave out compiling kernels and initialising the libraries. An
eigendecomposition that runs out of memory is reported and the band route still runs.
`kinds` selects the solves: "real" (the centrosymmetric form, when the crystal has an
inversion center), "complex" or both (default).

The settings follow the py3DED configurations of #632 (see
bench_bloch_structure_matrix.py): `g_max` is that of the structure factors, the
paraxial wave equation, 0.1 Å thermal displacements, 200 keV, thicknesses to 1000 Å.

Usage
-----
    python benchmarks/bench_bloch_band_chebyshev.py [device] [g_max] [sg_max]
        [precision] [crystal] [bandwidth] [reference] [kinds]
    python benchmarks/bench_bloch_band_chebyshev.py gpu 8.0 1.0 float32 ZrO2.cif 128 1
    python benchmarks/bench_bloch_band_chebyshev.py gpu 8 1 float32 A.cif 128 1 real
"""

import sys
import time

import numpy as np
from ase.io import read
from ase.spacegroup import crystal

import abtem
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.band_chebyshev import (
    apply_q,
    band_reduce,
    chebyshev_degree,
    chebyshev_propagate,
    gershgorin_bounds,
    lower_band,
)
from abtem.bloch.dynamical import (
    _metric,
    calculate_dynamical_scattering,
    calculate_structure_matrix,
)
from abtem.core.backend import asnumpy
from abtem.core.energy import energy2wavelength

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
g_max = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
sg_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
precision = sys.argv[4] if len(sys.argv) > 4 else "float32"
structure = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else "baddeleyite"
bandwidth = int(sys.argv[6]) if len(sys.argv) > 6 else 128
with_reference = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
kinds = sys.argv[8].split(",") if len(sys.argv) > 8 else ["real", "complex"]
energy = 200e3
use_wave_eq = True
thermal_sigma = 0.1
thicknesses = np.linspace(100, 1000, 10)


def synchronize():
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def free_device_pool():
    if device == "gpu":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def peak_memory_hook():
    """A context manager recording the peak of the CuPy memory pool in use, or None."""
    if device != "gpu":
        return None
    import cupy as cp

    pool = cp.get_default_memory_pool()

    class PeakHook(cp.cuda.MemoryHook):
        peak = 0

        def malloc_postprocess(self, **kwargs):
            self.peak = max(self.peak, pool.used_bytes())

    hook = PeakHook()
    hook.peak = pool.used_bytes()
    return hook


if structure == "baddeleyite":
    atoms = crystal(
        ["Zr", "O", "O"],
        basis=[
            (0.2754, 0.0395, 0.2083),
            (0.0700, 0.3317, 0.3447),
            (0.4496, 0.7569, 0.4792),
        ],
        spacegroup=14,
        cellpar=[5.1505, 5.2116, 5.3173, 90, 99.23, 90],
    )
else:
    atoms = read(structure)

with abtem.config.set({"precision": precision}):
    sf = StructureFactor(atoms, g_max=g_max, thermal_sigma=thermal_sigma, device=device)
    sf = sf.build(lazy=False)
    bw = BlochWaves(
        sf, energy=energy, sg_max=sg_max, use_wave_eq=use_wave_eq, device=device
    )
    # the direct beam first, so that the band reduction leaves it alone
    order = np.argsort(~np.all(bw.hkl == 0, axis=1), kind="stable")
    hkl, cell = bw.hkl[order], bw.cell
    n = len(hkl)
    scales = np.pi * energy2wavelength(energy) * thicknesses
    print(
        f"device={device} {precision} {atoms.get_chemical_formula()} g_max={g_max} "
        f"sg_max={sg_max}: {n} beams, bandwidth {bandwidth}, inversion center "
        f"{sf.inversion_center}"
    )

    def build(center, beams=None):
        return calculate_structure_matrix(
            sf.array,
            sf.hkl,
            hkl if beams is None else hkl[:beams],
            cell,
            energy,
            sf.gpts,
            use_wave_eq=use_wave_eq,
            inversion_center=center,
        )

    def eigh_amplitudes(A, center, beams=None):
        selected = hkl if beams is None else hkl[:beams]
        return calculate_dynamical_scattering(
            A,
            selected,
            cell,
            energy,
            thicknesses,
            use_wave_eq=use_wave_eq,
            inversion_center=center,
            overwrite_structure_matrix=True,
        )

    def band_chebyshev(A, fused):
        times = {}
        synchronize()
        start = time.perf_counter()
        bounds = gershgorin_bounds(A)
        synchronize()
        times["bounds"] = time.perf_counter() - start
        start = time.perf_counter()
        factors = band_reduce(A, bandwidth)
        synchronize()
        times["band"] = time.perf_counter() - start
        start = time.perf_counter()
        band = lower_band(A, bandwidth)
        y = chebyshev_propagate(band, scales, bounds, fused=fused)
        degree = chebyshev_degree(scales, bounds, band.real.dtype)
        synchronize()
        times["Chebyshev"] = time.perf_counter() - start
        start = time.perf_counter()
        y = apply_q(A, factors, len(band) - 1, y)
        synchronize()
        times["apply Q"] = time.perf_counter() - start
        return y, degree, times

    def float64_reference(center):
        """The eigendecomposition in double precision, on the device or else on the
        CPU; returns (amplitudes, where, seconds) or Nones."""
        for where in dict.fromkeys((device, "cpu")):
            try:
                with abtem.config.set({"precision": "float64"}):
                    sf64 = StructureFactor(
                        atoms, g_max=g_max, thermal_sigma=thermal_sigma, device=where
                    ).build(lazy=False)
                    start = time.perf_counter()
                    amplitudes = calculate_dynamical_scattering(
                        calculate_structure_matrix(
                            sf64.array,
                            sf64.hkl,
                            hkl,
                            cell,
                            energy,
                            sf64.gpts,
                            use_wave_eq=use_wave_eq,
                            inversion_center=center,
                        ),
                        hkl,
                        cell,
                        energy,
                        thicknesses,
                        use_wave_eq=use_wave_eq,
                        inversion_center=center,
                        overwrite_structure_matrix=True,
                    )
                    amplitudes = asnumpy(amplitudes).T
                return amplitudes, where, time.perf_counter() - start
            except MemoryError:
                free_device_pool()
        return None, None, None

    centers = []
    if sf.inversion_center is not None and "real" in kinds:
        centers.append(("real", sf.inversion_center))
    if "complex" in kinds:
        centers.append(("complex", None))

    for kind, center in centers:
        metric = _metric(hkl, cell, energy, use_wave_eq)[:, None]
        phase = (
            np.exp(-2j * np.pi * (hkl @ center))[:, None] if center is not None else 1.0
        )
        variants = [True, False] if device == "gpu" else [False]

        # warm-up on a small problem: kernels compiled, libraries initialised
        small = min(n, 1024)
        eigh_amplitudes(build(center, small), center, small)
        for fused in list(variants):
            try:
                band_chebyshev(build(center, small), fused)
            except Exception as error:  # a fused kernel that does not compile
                print(f"  {kind} fused: failed, {error!r}")
                variants.remove(fused)
        synchronize()

        try:
            synchronize()
            start = time.perf_counter()
            eigh_result = eigh_amplitudes(build(center), center)
            synchronize()
            t_eigh = time.perf_counter() - start
            eigh_result = asnumpy(eigh_result).T
            eigh_label = f"eigh {t_eigh:.3f} s"
        except MemoryError as error:
            free_device_pool()
            eigh_result = None
            eigh_label = f"eigh out of memory: {str(error).splitlines()[0]}"

        reference = None
        if with_reference:
            reference, where, seconds = float64_reference(center)
            if reference is None:
                print(f"  {kind}: float64 reference out of memory on {device} and CPU")
            else:
                print(f"  {kind}: float64 reference on {where} ({seconds:.0f} s)")

        # the production path, calculate_dynamical_scattering(solver="band"),
        # without the stage synchronisations
        synchronize()
        start = time.perf_counter()
        calculate_dynamical_scattering(
            build(center),
            hkl,
            cell,
            energy,
            thicknesses,
            use_wave_eq=use_wave_eq,
            inversion_center=center,
            overwrite_structure_matrix=True,
            solver="band",
        )
        synchronize()
        print(
            f"  {kind:<7} solver='band': {time.perf_counter() - start:.3f} s "
            f"({eigh_label})"
        )

        for fused in variants:
            A = build(center)
            matrix_bytes = A.nbytes
            hook = peak_memory_hook()
            if hook is not None:
                hook.__enter__()
            try:
                y, degree, times = band_chebyshev(A, fused)
            finally:
                if hook is not None:
                    hook.__exit__(None, None, None)
            del A
            y = asnumpy(y) * metric * phase
            total = sum(times.values())
            stages = "  ".join(f"{name} {value:.3f} s" for name, value in times.items())
            # in units of the complex matrix, as bench_bloch_structure_matrix.py
            complex_matrix = matrix_bytes * (2 if kind == "real" else 1)
            peak = (
                f"  peak {hook.peak / 1e6:.0f} MB "
                f"({hook.peak / complex_matrix:.1f} complex matrices)"
                if hook is not None
                else ""
            )
            comparison = ""
            if eigh_result is not None:
                difference = np.abs(y - eigh_result).max() / np.abs(eigh_result).max()
                comparison = f"  difference from eigh {difference:.1e}"
            print(
                f"  {kind:<7} {'fused' if fused else 'plain'}: total {total:.3f} s "
                f"({eigh_label})  [{stages}]  degree {degree}{peak}{comparison}"
            )
            if reference is not None:
                scale = np.abs(reference).max()
                errors = f"band+Chebyshev {np.abs(y - reference).max() / scale:.1e}"
                if eigh_result is not None:
                    errors += (
                        f", eigh {np.abs(eigh_result - reference).max() / scale:.1e}"
                    )
                print(f"           error against float64: {errors}")
