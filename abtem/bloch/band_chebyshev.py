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

import ctypes
import functools

import numpy as np
import scipy.special  # type: ignore
from numba import njit  # type: ignore

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


class _HostBlas:
    """The Hermitian (symmetric) matrix product and rank-2k update of the BLAS that
    SciPy is built with, called with a leading dimension, so that the trailing matrix
    of the band reduction, a block of a larger array, is neither copied nor
    transposed, and only its lower triangle is read and written: two thirds of the
    arithmetic of the three general products it replaces.

    The functions are those of `scipy.linalg.cython_blas`, whose C pointers SciPy
    exposes in capsules; they are called by ctypes with their Fortran signature.
    """

    def __init__(self, dtype):
        import scipy.linalg.cython_blas as cython_blas  # type: ignore

        capi = cython_blas.__pyx_capi__
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        def function(name, *types):
            capsule = capi[name]
            address = get_pointer(capsule, get_name(capsule))
            return ctypes.CFUNCTYPE(None, *types)(address)

        pointer, integer, char = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        self.complex = np.dtype(dtype) == np.complex128
        scalar = ctypes.c_double * 2 if self.complex else ctypes.c_double
        self._scalar = scalar
        if self.complex:
            self._multiply = function(
                "zhemm", char, char, integer, integer, pointer, pointer, integer,
                pointer, integer, pointer, pointer, integer,
            )
            self._update = function(
                "zher2k", char, char, integer, integer, pointer, pointer, integer,
                pointer, integer, ctypes.POINTER(ctypes.c_double), pointer, integer,
            )
        else:
            self._multiply = function(
                "dsymm", char, char, integer, integer, pointer, pointer, integer,
                pointer, integer, pointer, pointer, integer,
            )
            self._update = function(
                "dsyr2k", char, char, integer, integer, pointer, pointer, integer,
                pointer, integer, pointer, pointer, integer,
            )

    @staticmethod
    def _leading_dimension(array: np.ndarray) -> int:
        assert array.flags.f_contiguous or array.strides[0] == array.itemsize
        return max(1, array.strides[1] // array.itemsize)

    def _value(self, value: float):
        if self.complex:
            return self._scalar(value, 0.0)
        return self._scalar(value)

    def multiply(self, a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
        """out = a b for the Hermitian `a`, of which the lower triangle is read."""
        n, k = out.shape
        ld = self._leading_dimension
        self._multiply(
            b"L", b"L", ctypes.byref(ctypes.c_int(n)), ctypes.byref(ctypes.c_int(k)),
            ctypes.byref(self._value(1.0)), a.ctypes.data, ctypes.byref(ctypes.c_int(ld(a))),
            b.ctypes.data, ctypes.byref(ctypes.c_int(ld(b))),
            ctypes.byref(self._value(0.0)), out.ctypes.data, ctypes.byref(ctypes.c_int(ld(out))),
        )

    def update(self, alpha: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """c += alpha (a b^H + b a^H) for the Hermitian `c`, of which the lower
        triangle is written."""
        n, k = c.shape[0], a.shape[1]
        ld = self._leading_dimension
        beta = ctypes.c_double(1.0) if self.complex else self._value(1.0)
        self._update(
            b"L", b"N", ctypes.byref(ctypes.c_int(n)), ctypes.byref(ctypes.c_int(k)),
            ctypes.byref(self._value(alpha)), a.ctypes.data, ctypes.byref(ctypes.c_int(ld(a))),
            b.ctypes.data, ctypes.byref(ctypes.c_int(ld(b))),
            ctypes.byref(beta), c.ctypes.data, ctypes.byref(ctypes.c_int(ld(c))),
        )


@functools.lru_cache(maxsize=None)
def _host_blas(dtype_char: str):
    """The `_HostBlas` of a dtype ('D' complex128, 'd' float64), or None where SciPy
    does not give access to its BLAS."""
    if dtype_char not in ("D", "d"):
        return None
    try:
        return _HostBlas(np.dtype(dtype_char))
    except Exception:  # noqa: BLE001 -- a build without cython_blas capsules
        return None


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
        blas = _host_blas(A.dtype.char) if xp is np else None
        if blas is not None:
            # the Hermitian product and update read and write the lower triangle
            # only: the upper one of the trailing matrix is stale from here on
            W = np.empty((m, k), dtype=A.dtype, order="F")
            blas.multiply(trailing, VT, W)
            W -= 0.5 * V @ (T.conj().T @ (V.conj().T @ W))
            blas.update(-1.0, V, np.asfortranarray(W), trailing)
        elif xp is np:
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


@njit(nogil=True, fastmath=True)
def _chebyshev_recurrence(lre, lim, cre, cim, scale1, shift1, scale, shift, ore, oim):
    """The Chebyshev recurrence on the host, in float64: `lre`, `lim` the real and
    imaginary parts of the lower band (bandwidth + 1, n), `cre`, `cim` the
    coefficients (terms, thicknesses), `ore`, `oim` the result (thicknesses, n).

    Loops over the offsets of the band with the rows inside, which the compiler
    makes into vector operations; one thread, and `nogil`, so that it can run next to
    the GPU work of another frame."""
    b = lre.shape[0] - 1
    n = lre.shape[1]
    terms, nz = cre.shape
    pre = np.zeros(n)
    pim = np.zeros(n)
    cur_re = np.zeros(n)
    cur_im = np.zeros(n)
    cur_re[0] = 1.0
    nre = np.zeros(n)
    nim = np.zeros(n)
    sre = np.zeros(n)
    sim = np.zeros(n)
    for z in range(nz):
        ore[z, 0] += cre[0, z]
        oim[z, 0] += cim[0, z]
    for k in range(1, terms):
        sc = scale1 if k == 1 else scale
        sh = shift1 if k == 1 else shift
        for i in range(n):
            sre[i] = lre[0, i] * cur_re[i] - lim[0, i] * cur_im[i]
            sim[i] = lre[0, i] * cur_im[i] + lim[0, i] * cur_re[i]
        for o in range(1, b + 1):
            m = n - o
            for i in range(m):
                cr = lre[o, i]
                ci = lim[o, i]
                tr = cur_re[i + o]
                ti = cur_im[i + o]
                sre[i] += cr * tr + ci * ti
                sim[i] += cr * ti - ci * tr
            for i in range(m):
                cr = lre[o, i]
                ci = lim[o, i]
                tr = cur_re[i]
                ti = cur_im[i]
                sre[i + o] += cr * tr - ci * ti
                sim[i + o] += cr * ti + ci * tr
        for i in range(n):
            nre[i] = sc * sre[i] - sh * cur_re[i] - pre[i]
            nim[i] = sc * sim[i] - sh * cur_im[i] - pim[i]
        for z in range(nz):
            ar = cre[k, z]
            ai = cim[k, z]
            for i in range(n):
                ore[z, i] += ar * nre[i] - ai * nim[i]
                oim[z, i] += ar * nim[i] + ai * nre[i]
        pre, cur_re, nre = cur_re, nre, pre
        pim, cur_im, nim = cur_im, nim, pim


def chebyshev_propagate_host(
    band: np.ndarray, scales: np.ndarray, bounds: tuple[float, float]
) -> np.ndarray:
    """`chebyshev_propagate` in float64 on the host, by a compiled kernel: the band
    of a complex128 or float64 matrix, the result complex128."""
    n = band.shape[1]
    coefficients, center, half = chebyshev_coefficients(
        scales, bounds, float(np.finfo(np.float64).eps) * _TRUNCATION
    )
    real = np.zeros((len(scales), n))
    imaginary = np.zeros((len(scales), n))
    _chebyshev_recurrence(
        np.ascontiguousarray(band.real),
        np.ascontiguousarray(band.imag),
        np.ascontiguousarray(coefficients.real),
        np.ascontiguousarray(coefficients.imag),
        1.0 / half,
        center / half,
        2.0 / half,
        2.0 * center / half,
        real,
        imaginary,
    )
    return (real + 1j * imaginary).T


def chebyshev_propagate(
    band: np.ndarray,
    scales: np.ndarray,
    bounds: tuple[float, float],
    fused: bool = True,
) -> np.ndarray:
    """exp(i s B) e_0 for each s, as an (n, len(scales)) array, from the lower band of
    the Hermitian band matrix B (see `lower_band`) and bounds on its spectrum.

    On a GPU each term is one fused kernel unless `fused` is False; on the CPU, a
    double-precision band goes to a compiled kernel (`chebyshev_propagate_host`).
    """
    xp = get_array_module(band)
    if xp is np and band.dtype in (np.float64, np.complex128):
        return chebyshev_propagate_host(band, scales, bounds)
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
    bandwidth: int | None = None,
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
    bandwidth : int, optional
        The bandwidth of the reduction: 128 on a GPU, 32 on the CPU, where a wider
        band makes the recurrence dearer faster than it makes the reduction cheaper
        (1299 beams, an Apple M2: 0.11 s and 0.16 s for 32, 0.11 s and 0.30 s for 64).
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
    if bandwidth is None:
        bandwidth = 32 if xp is np else 128
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
