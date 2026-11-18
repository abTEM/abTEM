"""Bloch-wave propagation by band reduction and a Chebyshev expansion.

The beam amplitudes of a Bloch-wave calculation are exp(i c A) e_0 for the Hermitian
structure matrix A, the incident (direct) beam e_0 and c = pi lambda z for each
thickness z. Instead of the eigendecomposition of A, this module

  1. bounds the spectrum of A by its Gershgorin discs (rigorous, and close: the
     couplings are small next to the excitation errors on the diagonal);
  2. reduces A in place to band form, A = Q B Q^H, by a Householder QR of each panel
     below the band and a two-sided update of the trailing matrix with matrix
     products. Unlike the one-stage tridiagonalization in an eigendecomposition, which
     is limited by memory bandwidth, this is limited by arithmetic and fast on a GPU.
     The reflectors are kept below the band, where each panel was annihilated, so the
     matrix is the only large array. The first rows are never transformed: with the
     direct beam first, Q^H e_0 = e_0;
  3. expands exp(i c B) e_0 in Chebyshev polynomials of the band matrix, one
     three-term recurrence for every thickness at once (a fused kernel per term on a
     GPU);
  4. applies Q to the few result vectors with the stored block reflectors.

Measured on an RTX 4090 in single precision (issue #632): 13,903 beams take 0.71 s
instead of 2.90 s for the eigendecomposition (real symmetric) and 1.13 s instead of
5.77 s (complex Hermitian), and 28,898 beams 2.56 s instead of 24.3 s, with about a
quarter of the memory and five to thirteen times smaller errors. The expansion costs
about 0.3 s per 1000 Å of the largest thickness, and on the CPU the eigendecomposition
is faster.
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.special  # type: ignore

from abtem.core.backend import cp, get_array_module

# The expansion is truncated where every coefficient falls below this fraction of the
# rounding of the working precision.
_TRUNCATION = 0.01


def gershgorin_bounds(A: np.ndarray, block: int = 1024) -> tuple[float, float]:
    """Bounds on the spectrum of the Hermitian matrix A from its Gershgorin discs.

    The column sums are taken in blocks of columns, with no temporary array of the size
    of A.
    """
    xp = get_array_module(A)
    low, high = np.inf, -np.inf
    for start in range(0, len(A), block):
        stop = start + block
        diagonal = xp.diagonal(A[start:stop, start:stop]).real
        radius = xp.abs(A[:, start:stop]).sum(axis=0) - xp.abs(diagonal)
        low = min(low, float((diagonal - radius).min()))
        high = max(high, float((diagonal + radius).max()))
    return low, high


def _gemm(transa: str, transb: str, alpha, a, b, beta, c) -> None:
    """c = alpha op(a) op(b) + beta c on the GPU, in place, for column-major arrays and
    column-major views of them (`cupy.cublas.gemm` copies a view it writes to)."""
    from cupy.cuda import device  # type: ignore
    from cupy_backends.cuda.libs import cublas  # type: ignore

    function = {
        "f": cublas.sgemm,
        "d": cublas.dgemm,
        "F": cublas.cgemm,
        "D": cublas.zgemm,
    }[c.dtype.char]
    operations = {"N": cublas.CUBLAS_OP_N, "C": cublas.CUBLAS_OP_C}
    for array in (a, b, c):
        assert array.strides[0] == array.itemsize, "column-major arrays only"
    alpha = np.array(alpha, dtype=c.dtype)
    beta = np.array(beta, dtype=c.dtype)
    handle = device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    function(
        handle,
        operations[transa],
        operations[transb],
        c.shape[0],
        c.shape[1],
        a.shape[1] if transa == "N" else a.shape[0],
        alpha.ctypes.data,
        a.data.ptr,
        max(1, a.strides[1] // a.itemsize),
        b.data.ptr,
        max(1, b.strides[1] // b.itemsize),
        beta.ctypes.data,
        c.data.ptr,
        max(1, c.strides[1] // c.itemsize),
    )


def _reflectors(stored: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """The unit lower trapezoidal V of a panel from the strictly lower part of its QR,
    with the columns of identity reflectors zeroed."""
    xp = get_array_module(stored)
    k = stored.shape[1]
    V = xp.tril(stored, -1)
    V[xp.arange(k), xp.arange(k)] = 1
    V[:, identity] = 0
    return xp.asfortranarray(V)


def _block_reflector(
    stored: np.ndarray, tau: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V, T and the identity reflectors of Q = H_1 ... H_k = I - V T V^H.

    T^-1 = diag(1 / tau) + strict_upper(V^H V), for complex reflectors too. A reflector
    with tau = 0 is the identity; its column of V is zeroed and its tau set to one.
    """
    xp = get_array_module(stored)
    identity = tau == 0
    V = _reflectors(stored, identity)
    k = len(tau)
    inverse = xp.triu(V.conj().T @ V, 1)
    inverse[xp.arange(k), xp.arange(k)] = 1 / xp.where(identity, 1, tau)
    T = xp.asfortranarray(xp.triu(xp.linalg.inv(inverse)))
    return V, T, identity


def band_reduce(A: np.ndarray, bandwidth: int) -> list:
    """Reduce the Fortran-ordered Hermitian matrix A in place to band form B = Q^H A Q.

    Afterwards the lower band of A holds B, and the reflectors are below the band, in
    the strictly lower triangle of each panel (as LAPACK's geqrf leaves them). Returns
    (column, number of reflectors, T, identity reflectors) for each panel.
    """
    xp = get_array_module(A)
    assert A.flags.f_contiguous
    n = len(A)
    factors = []
    for column in range(0, n, bandwidth):
        start = column + bandwidth
        m = n - start
        if m <= 1:
            break
        panel = A[start:, column : column + bandwidth]
        stored, tau = xp.linalg.qr(panel, mode="raw")  # transposed, as NumPy's
        panel[...] = stored.T
        k = min(m, bandwidth)
        V, T, identity = _block_reflector(panel[:, :k], tau[:k])
        # trailing <- Q^H trailing Q, Q = I - V T V^H: with W = trailing V T and
        # W <- W - 1/2 V (T^H V^H W), trailing <- trailing - V W^H - W V^H
        trailing = A[start:, start:]
        VT = xp.asfortranarray(V @ T)
        if xp is np:
            W = trailing @ VT
            W -= 0.5 * V @ (T.conj().T @ (V.conj().T @ W))
            trailing -= V @ W.conj().T
            trailing -= W @ V.conj().T
        else:
            W = xp.empty((m, k), dtype=A.dtype, order="F")
            _gemm("N", "N", 1, trailing, VT, 0, W)
            W -= 0.5 * V @ (T.conj().T @ (V.conj().T @ W))
            _gemm("N", "C", -1, V, W, 1, trailing)
            _gemm("N", "C", -1, W, V, 1, trailing)
        factors.append((column, k, T, identity))
    return factors


def lower_band(A: np.ndarray, bandwidth: int) -> np.ndarray:
    """The lower band of A as a (bandwidth + 1, n) array, band[o, i] = A[i + o, i]."""
    xp = get_array_module(A)
    n = len(A)
    band = xp.zeros((bandwidth + 1, n), dtype=A.dtype)
    for offset in range(bandwidth + 1):
        band[offset, : n - offset] = xp.diagonal(A, -offset)
    return band


def apply_q(A: np.ndarray, factors: list, bandwidth: int, y: np.ndarray) -> np.ndarray:
    """Q y in place, with the reflectors stored in A by `band_reduce`."""
    for column, k, T, identity in reversed(factors):
        start = column + bandwidth
        V = _reflectors(A[start:, column : column + k], identity)
        y[start:] -= V @ (T @ (V.conj().T @ y[start:]))
    return y


def chebyshev_coefficients(
    scales: np.ndarray, bounds: tuple[float, float], tolerance: float
) -> tuple[np.ndarray, float, float]:
    """The Chebyshev coefficients of exp(i s x) for x in `bounds` and each s, truncated
    where every coefficient falls below `tolerance`; with the center and half-width of
    the bounds.

    exp(i s x) = exp(i s center) sum_k eps_k i^k J_k(s half) T_k(y), with
    x = center + half y, eps_0 = 1 and eps_k = 2.
    """
    center, half = (bounds[1] + bounds[0]) / 2, (bounds[1] - bounds[0]) / 2
    # a degenerate spectrum, B = center I: any half-width gives the same result
    half = half if half > 0 else 1.0
    arguments = np.asarray(scales, dtype=float) * half
    degree = max(int(arguments.max(initial=0.0)), 1)
    while np.abs(scipy.special.jv(degree, arguments)).max(initial=0.0) > tolerance:
        degree += 10
    k = np.arange(degree + 1)
    coefficients = (
        np.where(k == 0, 1.0, 2.0)[:, None]
        * (1j**k)[:, None]
        * scipy.special.jv(k[:, None], arguments[None])
        * np.exp(1j * np.asarray(scales, dtype=float) * center)[None]
    )
    return coefficients, center, half


def chebyshev_degree(scales: np.ndarray, bounds: tuple[float, float], dtype) -> int:
    """The number of terms of the expansion in the precision of `dtype`."""
    tolerance = float(np.finfo(dtype).eps) * _TRUNCATION
    return len(chebyshev_coefficients(scales, bounds, tolerance)[0]) - 1


_STEP_KERNEL = r"""
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


@functools.lru_cache(maxsize=None)
def _step_kernel(dtype_char: str):
    """One term of the recurrence, t_next = 2 B~ t_current - t_previous with
    B~ = (B - center) / half, fused with its contribution to every thickness."""
    real = "double" if dtype_char in "dD" else "float"
    complex_band = dtype_char in "FD"
    source = (
        _STEP_KERNEL.replace("SCALAR", f"complex<{real}>" if complex_band else real)
        .replace("REAL", real)
        .replace("CONJ", "conj" if complex_band else "")
    )
    return cp.RawKernel(source, "chebyshev_step")


def chebyshev_propagate(
    band: np.ndarray,
    scales: np.ndarray,
    bounds: tuple[float, float],
    fused: bool = True,
) -> np.ndarray:
    """exp(i s B) e_0 for each s, as an (n, len(scales)) array, from the lower band of
    the Hermitian band matrix B (see `lower_band`) and bounds on its spectrum.

    On a GPU each term is one fused kernel unless `fused` is False.
    """
    xp = get_array_module(band)
    bandwidth, n = band.shape[0] - 1, band.shape[1]
    real_dtype = np.finfo(band.dtype).dtype
    complex_dtype = np.result_type(band.dtype, np.complex64)
    coefficients, center, half = chebyshev_coefficients(
        scales, bounds, float(np.finfo(real_dtype).eps) * _TRUNCATION
    )
    coefficients = xp.asarray(coefficients, dtype=complex_dtype)

    if xp is np:
        import scipy.sparse as sparse  # type: ignore
    else:
        import cupyx.scipy.sparse as sparse  # type: ignore
    # B from its lower band: the upper diagonals are the conjugated lower ones
    matrix = sparse.diags(
        [band[o, : n - o] for o in range(bandwidth, 0, -1)]
        + [band[0]]
        + [band[o, : n - o].conj() for o in range(1, bandwidth + 1)],
        offsets=list(range(-bandwidth, bandwidth + 1)),
        format="csr",
        dtype=band.dtype,
    )

    t_previous = xp.zeros(n, dtype=band.dtype)
    t_previous[0] = 1
    t_current = (matrix @ t_previous - center * t_previous) / half
    result = xp.outer(t_previous, coefficients[0])
    result += xp.outer(t_current, coefficients[1])

    if fused and xp is not np:
        kernel = _step_kernel(np.dtype(band.dtype).char)
        band = xp.ascontiguousarray(band)
        t_next = xp.empty_like(t_current)
        threads = 256
        blocks = (n + threads - 1) // threads
        scale = real_dtype.type(2 / half)
        shift = real_dtype.type(2 * center / half)
        for k in range(2, len(coefficients)):
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
                    np.int32(bandwidth),
                    np.int32(len(scales)),
                    scale,
                    shift,
                ),
            )
            t_previous, t_current, t_next = t_current, t_next, t_previous
    else:
        for k in range(2, len(coefficients)):
            t_previous, t_current = (
                t_current,
                (2 / half) * (matrix @ t_current)
                - (2 * center / half) * t_current
                - t_previous,
            )
            result += xp.outer(t_current, coefficients[k])
    return result


def propagate_incident_beam(
    A: np.ndarray,
    scales: np.ndarray,
    direct: int = 0,
    bandwidth: int = 128,
    bounds: tuple[float, float] | None = None,
    overwrite_a: bool = False,
    fused: bool = True,
) -> np.ndarray:
    """exp(i s A) e_direct for each s, by band reduction and a Chebyshev expansion.

    Parameters
    ----------
    A : numpy.ndarray or cupy.ndarray
        The Hermitian (or real symmetric) matrix, N x N.
    scales : sequence of float
        The values of s.
    direct : int
        The index of the incident beam.
    bandwidth : int
        The bandwidth of the reduction.
    bounds : tuple of float, optional
        Bounds on the spectrum of A; Gershgorin bounds if not given.
    overwrite_a : bool
        If True, A is destroyed (its memory holds the band matrix and the reflectors);
        otherwise it is copied first.
    fused : bool
        On a GPU, run each term of the expansion as one fused kernel.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        The (N, len(scales)) complex result, in the order of A.
    """
    xp = get_array_module(A)
    if xp is not np and (cp is None or xp is not cp):
        raise NotImplementedError("the band solver needs NumPy or CuPy arrays")
    n = len(A)
    A = xp.asfortranarray(A) if overwrite_a else xp.array(A, order="F", copy=True)
    if bounds is None:
        bounds = gershgorin_bounds(A)
    # the direct beam first: the reduction never transforms the first row
    if direct:
        A[[0, direct]] = A[[direct, 0]]
        A[:, [0, direct]] = A[:, [direct, 0]]
    bandwidth = max(0, min(bandwidth, n - 1))
    factors = band_reduce(A, bandwidth) if bandwidth else []
    band = lower_band(A, bandwidth)
    y = chebyshev_propagate(band, np.asarray(scales, dtype=float), bounds, fused)
    del band
    y = apply_q(A, factors, bandwidth, y)
    if direct:
        y[[0, direct]] = y[[direct, 0]]
    return y


# Measured on an RTX 4090 in single precision at 13,903 beams (issue #632): the
# eigendecomposition and the band reduction both scale as N**3 (the complex Hermitian
# forms take twice as long), and a term of the expansion takes about 40 µs.
_EIGH_SECONDS = 2.9
_BAND_SECONDS = 0.35
_REFERENCE_BEAMS = 13903
_TERM_SECONDS = 40e-6


def band_is_faster(n: int, degree: int) -> bool:
    """Whether the band solver is expected to be faster than the eigendecomposition on
    a GPU, for N beams and an expansion of `degree` terms.

    The eigendecomposition grows as N**3 and does not depend on the thickness; the band
    reduction also grows as N**3 but about eight times slower, and the expansion as
    the thickness. Small matrices, or very thick samples, are left to the
    eigendecomposition.
    """
    volume = (n / _REFERENCE_BEAMS) ** 3
    return _BAND_SECONDS * volume + _TERM_SECONDS * degree < _EIGH_SECONDS * volume
