"""Benchmark (experiment): Bloch-wave propagation by band reduction and a Chebyshev
expansion, against the eigendecomposition.

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
import scipy.special
from ase.io import read
from ase.spacegroup import crystal

import abtem
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.dynamical import (
    _metric,
    calculate_dynamical_scattering,
    calculate_structure_matrix,
)
from abtem.core.backend import asnumpy, get_array_module
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


def gershgorin_bounds(A, block=1024):
    """Bounds on the spectrum of the Hermitian A, from column sums in blocks (no
    temporary of the size of A)."""
    xp = get_array_module(A)
    n = len(A)
    low, high = np.inf, -np.inf
    for start in range(0, n, block):
        columns = A[:, start : start + block]
        diagonal = xp.diagonal(A[start : start + block, start : start + block]).real
        radius = xp.abs(columns).sum(axis=0) - xp.abs(diagonal)
        low = min(low, float((diagonal - radius).min()))
        high = max(high, float((diagonal + radius).max()))
    return low, high


def _gemm(transa, transb, alpha, a, b, beta, c):
    """c = alpha op(a) op(b) + beta c for Fortran-ordered 2D arrays or column views of
    them, in place (cupy.cublas.gemm copies an output view)."""
    from cupy.cuda import device as cuda_device
    from cupy_backends.cuda.libs import cublas

    function = {
        "f": cublas.sgemm,
        "d": cublas.dgemm,
        "F": cublas.cgemm,
        "D": cublas.zgemm,
    }[c.dtype.char]
    operations = {
        "N": cublas.CUBLAS_OP_N,
        "T": cublas.CUBLAS_OP_T,
        "C": cublas.CUBLAS_OP_C,
    }
    for array in (a, b, c):
        assert array.strides[0] == array.itemsize, "column-major arrays only"
    m = c.shape[0]
    n = c.shape[1]
    k = a.shape[1] if transa == "N" else a.shape[0]
    alpha = np.array(alpha, dtype=c.dtype)
    beta = np.array(beta, dtype=c.dtype)
    handle = cuda_device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    function(
        handle,
        operations[transa],
        operations[transb],
        m,
        n,
        k,
        alpha.ctypes.data,
        a.data.ptr,
        max(1, a.strides[1] // a.itemsize),
        b.data.ptr,
        max(1, b.strides[1] // b.itemsize),
        beta.ctypes.data,
        c.data.ptr,
        max(1, c.strides[1] // c.itemsize),
    )


def _block_reflector(h, tau, k):
    """V (unit lower trapezoidal) and T of Q = H_1 ... H_k = I - V T V^H, from the
    `mode="raw"` QR of a panel, with T^-1 = diag(1/tau) + strict_upper(V^H V)
    (a reflector with tau = 0 is the identity: its column of V is zeroed)."""
    xp = get_array_module(h)
    V = xp.tril(h.T[:, :k], -1)
    index = xp.arange(k)
    V[index, index] = 1
    tau = tau[:k]
    identity = tau == 0
    tau = xp.where(identity, 1, tau)
    V[:, identity] = 0
    inverse = xp.triu(V.conj().T @ V, 1)
    inverse[index, index] = 1 / tau
    T = xp.linalg.inv(inverse)
    return xp.asfortranarray(V), xp.asfortranarray(xp.triu(T)), identity


def band_reduce(A, b):
    """Reduce the Fortran-ordered Hermitian A in place to band form B = Q^H A Q.

    Returns the T factors of the block reflectors; the reflectors V are left in A below
    the band (the strictly lower triangle of each panel's R), as LAPACK's geqrf leaves
    them. Only the lower band of A holds B afterwards."""
    xp = get_array_module(A)
    assert A.flags.f_contiguous
    n = len(A)
    factors = []
    for j in range(0, n, b):
        r0 = j + b
        m = n - r0
        if m <= 1:
            break
        panel = A[r0:, j : j + b]
        h, tau = xp.linalg.qr(panel, mode="raw")  # h is the transposed geqrf output
        k = min(m, b)
        V, T, identity = _block_reflector(h, tau, k)
        panel[...] = h.T
        trailing = A[r0:, r0:]
        # trailing <- Q^H trailing Q with Q = I - V T V^H:
        # W = trailing V T - 1/2 V (T^H V^H trailing V T); trailing -= V W^H + W V^H
        VT = xp.asfortranarray(V @ T)
        if xp is np:
            W = trailing @ VT
            W -= 0.5 * V @ (T.conj().T @ (V.conj().T @ W))
            trailing -= V @ W.conj().T + W @ V.conj().T
        else:
            W = xp.empty((m, k), dtype=A.dtype, order="F")
            _gemm("N", "N", 1, trailing, VT, 0, W)
            W -= 0.5 * V @ (T.conj().T @ (V.conj().T @ W))
            W = xp.asfortranarray(W)
            _gemm("N", "C", -1, V, W, 1, trailing)
            _gemm("N", "C", -1, W, V, 1, trailing)
        factors.append((j, k, T, identity))
    return factors


def lower_band(A, b):
    """The lower band of A as a (b + 1, n) array: band[o, i] = A[i + o, i]."""
    xp = get_array_module(A)
    n = len(A)
    band = xp.zeros((b + 1, n), dtype=A.dtype)
    for o in range(b + 1):
        band[o, : n - o] = xp.diagonal(A, -o)
    return band


def apply_q(A, factors, b, y):
    """Q y, with the reflectors stored in A below the band."""
    xp = get_array_module(A)
    for j, k, T, identity in reversed(factors):
        r0 = j + b
        V = xp.tril(A[r0:, j : j + k], -1)
        index = xp.arange(k)
        V[index, index] = 1
        V[:, identity] = 0  # as in _block_reflector
        y[r0:] -= V @ (T @ (V.conj().T @ y[r0:]))
    return y


_KERNEL = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void chebyshev_step(
    const SCALAR* band, const SCALAR* t_current, const SCALAR* t_previous,
    SCALAR* t_next, const complex<REAL>* coefficients, complex<REAL>* result,
    const int n, const int b, const int nz, const REAL scale, const REAL shift)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    SCALAR product = band[i] * t_current[i];
    for (int o = 1; o <= b; ++o) {
        if (i + o < n) product += CONJ(band[o * n + i]) * t_current[i + o];
        if (i >= o) product += band[o * n + i - o] * t_current[i - o];
    }
    const SCALAR next = scale * product - shift * t_current[i] - t_previous[i];
    t_next[i] = next;
    for (int z = 0; z < nz; ++z) result[i * nz + z] += coefficients[z] * next;
}
"""


def _fused_step_kernel(dtype):
    import cupy as cp

    real = "double" if np.dtype(dtype) in (np.float64, np.complex128) else "float"
    complex_band = np.issubdtype(dtype, np.complexfloating)
    source = (
        _KERNEL.replace("SCALAR", f"complex<{real}>" if complex_band else real)
        .replace("REAL", real)
        .replace("CONJ", "conj" if complex_band else "")
    )
    return cp.RawKernel(source, "chebyshev_step")


def chebyshev_coefficients(scales, bounds, tolerance):
    """The coefficients of exp(i s x), x in `bounds`, truncated where they all fall
    below `tolerance`."""
    center, half = (bounds[1] + bounds[0]) / 2, (bounds[1] - bounds[0]) / 2
    arguments = np.asarray(scales) * half
    degree = int(arguments.max())
    while np.abs(scipy.special.jv(degree, arguments)).max() > tolerance:
        degree += 10
    k = np.arange(degree + 1)
    coefficients = (
        np.where(k == 0, 1.0, 2.0)[:, None]
        * (1j**k)[:, None]
        * scipy.special.jv(k[:, None], arguments[None])
        * np.exp(1j * np.asarray(scales) * center)[None]
    )
    return coefficients, center, half


def chebyshev_propagate(band, scales, bounds, fused=True):
    """exp(i s B) e_0 for each s, from the lower band of B."""
    xp = get_array_module(band)
    b, n = band.shape[0] - 1, band.shape[1]
    complex_dtype = np.result_type(band.dtype, np.complex64)
    real_dtype = np.finfo(band.dtype).dtype
    # truncation well below the rounding of the working precision
    tolerance = np.finfo(real_dtype).eps / 100
    coefficients, center, half = chebyshev_coefficients(scales, bounds, tolerance)
    coefficients = xp.asarray(coefficients, dtype=complex_dtype)
    degree = len(coefficients) - 1

    if xp is np:
        import scipy.sparse as sparse
    else:
        import cupyx.scipy.sparse as sparse
    # B from its lower band: the upper diagonals are the conjugated lower ones
    matrix = sparse.diags(
        [band[o, : n - o] for o in range(b, 0, -1)]
        + [band[0]]
        + [band[o, : n - o].conj() for o in range(1, b + 1)],
        offsets=[-o for o in range(b, 0, -1)] + [0] + list(range(1, b + 1)),
        format="csr",
        dtype=band.dtype,
    )

    t_previous = xp.zeros(n, dtype=band.dtype)
    t_previous[0] = 1
    t_current = (matrix @ t_previous - center * t_previous) / half
    result = xp.outer(t_previous, coefficients[0]) + xp.outer(
        t_current, coefficients[1]
    )
    result = xp.ascontiguousarray(result)

    if fused and xp is not np:
        kernel = _fused_step_kernel(band.dtype)
        band = xp.ascontiguousarray(band)
        t_next = xp.empty_like(t_current)
        threads = 256
        blocks = (n + threads - 1) // threads
        scale, shift = real_dtype.type(2 / half), real_dtype.type(2 * center / half)
        for k in range(2, degree + 1):
            kernel(
                (blocks,),
                (threads,),
                (
                    band,
                    t_current,
                    t_previous,
                    t_next,
                    coefficients[k],
                    result,
                    np.int32(n),
                    np.int32(b),
                    np.int32(len(scales)),
                    scale,
                    shift,
                ),
            )
            t_previous, t_current, t_next = t_current, t_next, t_previous
    else:
        for k in range(2, degree + 1):
            t_previous, t_current = (
                t_current,
                (2 / half) * (matrix @ t_current)
                - (2 * center / half) * t_current
                - t_previous,
            )
            result += xp.outer(t_current, coefficients[k])
    return result, degree


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
        y, degree = chebyshev_propagate(band, scales, bounds, fused=fused)
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
