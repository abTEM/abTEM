"""The band solver of :mod:`abtem.bloch.band_chebyshev` in emulated double precision
on Apple Metal (see :mod:`abtem.core._double_float`).

The beam amplitudes exp(i s A) e_0 of a Bloch-wave calculation are found as there:
the Hermitian structure matrix A is reduced to band form, A = Q B Q^H, by a blocked
Householder QR of the panels below the band and a two-sided update of the trailing
matrix; exp(i s B) e_0 is expanded in Chebyshev polynomials of the band matrix; and Q
is applied to the few result vectors. Here the matrix lives on the GPU as
double-float, with ~1e-14 relative accuracy, and

  * the trailing matrix is updated by two double-float matrix products per panel,
    which is nearly all of the arithmetic;
  * the panels, the block reflectors, the small products of the update and the
    application of Q are float64 on the host (the panels are N x b, and the block
    reflectors are applied to N x few vectors: both are small);
  * each term of the Chebyshev recurrence is one kernel launch, all the thicknesses
    sharing the recurrence.
"""

from __future__ import annotations

import numpy as np

from abtem.bloch.band_chebyshev import (
    _TRUNCATION,
    _block_reflector,
    chebyshev_coefficients,
    chebyshev_propagate_host,
    gershgorin_bounds,
)
from abtem.core import _double_float as df

_GEMM_TILE = 32  # output tile, in rows and columns; 8 x 8 threads of 4 x 4 each
_GEMM_DEPTH = 16

# C = A B (or C - A B), A and B complex double-float matrices addressed by offset and
# row stride, B optionally as the conjugate transpose of the matrix stored. An element
# is float4(re_hi, re_lo, im_hi, im_lo).
#
# The multiply-add is lean: a product of two double-floats is the exact product of the
# high parts (an FMA gives its rounding error) and the two cross terms, without the
# product of the low parts, and it is added into an accumulator (hi, lo) by a two-sum
# of the high parts with the rest collected in the low part, which is renormalised
# once per step of the depth. That is 12 flops per real multiply-add instead of the 21
# of the general double-float multiplication and addition, at the same accuracy, and
# made the products 2.2 times faster on an Apple M2 (74 to 164 GFLOP/s; 4 x 4
# elements per thread and a depth of 16 were the fastest of the tiles measured).
_GEMM = f"""
#define BM {_GEMM_TILE}u
#define BN {_GEMM_TILE}u
#define BK {_GEMM_DEPTH}u
inline float4 conj4(float4 q) {{ return float4(q.x, q.y, -q.z, -q.w); }}
inline void mac(thread float& ah, thread float& al,
                float xh, float xl, float yh, float yl, float sign) {{
  float p = xh * yh;
  float e = fma(xh, yh, -p);
  e = fma(xh, yl, e);
  e = fma(xl, yh, e);
  p *= sign;
  e *= sign;
  float s = ah + p;
  float bb = s - ah;
  al += ((ah - (s - bb)) + (p - bb)) + e;
  ah = s;
}}
kernel void gemm_df(
    device const float4* A, device const float4* B, device float4* C,
    constant uint& M, constant uint& N, constant uint& K,
    constant uint& base_a, constant uint& lda,
    constant uint& base_b, constant uint& ldb,
    constant uint& base_c, constant uint& ldc,
    constant uint& conj_transpose_b, constant uint& subtract, constant uint& lower_only,
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {{
  // a Hermitian result needs only the tiles on and below the diagonal
  if (lower_only != 0u && tg.x * BN >= (tg.y + 1u) * BM) return;
  threadgroup float4 As[BM * BK];
  threadgroup float4 Bs[BK * BN];
  uint t = tid.y * 8u + tid.x;
  uint row0 = tg.y * BM, col0 = tg.x * BN;
  float rh[4][4], rl[4][4], ih[4][4], il[4][4];
  for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {{
    rh[i][j] = 0.0f; rl[i][j] = 0.0f; ih[i][j] = 0.0f; il[i][j] = 0.0f;
  }}
  for (uint k0 = 0; k0 < K; k0 += BK) {{
    for (uint i = t; i < BM * BK; i += 64u) {{
      uint r = i / BK, kk = i % BK;
      As[i] = (row0 + r < M && k0 + kk < K)
          ? A[base_a + (row0 + r) * lda + k0 + kk] : float4(0.0f);
    }}
    if (conj_transpose_b == 0u) {{
      for (uint i = t; i < BK * BN; i += 64u) {{
        uint kk = i / BN, c = i % BN;
        Bs[i] = (k0 + kk < K && col0 + c < N)
            ? B[base_b + (k0 + kk) * ldb + col0 + c] : float4(0.0f);
      }}
    }} else {{
      for (uint i = t; i < BK * BN; i += 64u) {{
        uint c = i / BK, kk = i % BK;
        Bs[kk * BN + c] = (k0 + kk < K && col0 + c < N)
            ? conj4(B[base_b + (col0 + c) * ldb + k0 + kk]) : float4(0.0f);
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk++) {{
      for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {{
        float4 a = As[(tid.y * 4u + i) * BK + kk];
        float4 b = Bs[kk * BN + tid.x * 4u + j];
        mac(rh[i][j], rl[i][j], a.x, a.y, b.x, b.y, 1.0f);
        mac(rh[i][j], rl[i][j], a.z, a.w, b.z, b.w, -1.0f);
        mac(ih[i][j], il[i][j], a.x, a.y, b.z, b.w, 1.0f);
        mac(ih[i][j], il[i][j], a.z, a.w, b.x, b.y, 1.0f);
        float2 q = quick_two_sum(rh[i][j], rl[i][j]);
        rh[i][j] = q.x; rl[i][j] = q.y;
        float2 r = quick_two_sum(ih[i][j], il[i][j]);
        ih[i][j] = r.x; il[i][j] = r.y;
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) {{
    uint r = row0 + tid.y * 4u + i, c = col0 + tid.x * 4u + j;
    if (r < M && c < N) {{
      cdf v = cdf{{float2(rh[i][j], rl[i][j]), float2(ih[i][j], il[i][j])}};
      uint g = base_c + r * ldc + c;
      if (subtract != 0u) v = csub(cload(C[g]), v);
      C[g] = float4(v.re.x, v.re.y, v.im.x, v.im.y);
    }}
  }}
}}
"""

# acc += c x for the complex double-floats c, x (as float4) by four lean
# multiply-adds, renormalised after each, in the accumulator (re_hi, re_lo, im_hi, im_lo)
_BAND_MAC = """
inline void band_mac(thread float4& acc, float4 c, float4 x) {
  float rh = acc.x, rl = acc.y, ih = acc.z, il = acc.w;
  mac(rh, rl, c.x, c.y, x.x, x.y, 1.0f);
  mac(rh, rl, c.z, c.w, x.z, x.w, -1.0f);
  mac(ih, il, c.x, c.y, x.z, x.w, 1.0f);
  mac(ih, il, c.z, c.w, x.x, x.y, 1.0f);
  float2 re = quick_two_sum(rh, rl);
  float2 im = quick_two_sum(ih, il);
  acc = float4(re.x, re.y, im.x, im.y);
}
"""

# One term of the recurrence, t_next = scale * (B t) - shift * t - t_previous, B given
# by its lower band, band[o * n + i] = B[i + o, i], with its contribution to the
# result of each thickness, result[i, z] += coefficient[k, z] * t_next[i].
# The upper triangle of a Hermitian matrix from its lower one, beyond the diagonal tiles
# that `gemm_df` with lower_only computes whole: the elements (r, c) of the n x n
# matrix at base with row stride ld, c in a later tile than r.
_MIRROR = f"""
kernel void mirror_df(
    device float4* C, constant uint& n, constant uint& base, constant uint& ld,
    uint2 position [[thread_position_in_grid]]) {{
  uint c = position.x, r = position.y;
  if (c >= n || r >= n || c / {_GEMM_TILE}u <= r / {_GEMM_TILE}u) return;
  C[base + r * ld + c] = conj4(C[base + c * ld + r]);
}}
"""

_CHEBYSHEV = """
kernel void chebyshev_step(
    device const float4* band, device const float4* t_current,
    device const float4* t_previous, device float4* t_next,
    device const float4* coefficients, device float4* result,
    constant uint& n, constant uint& b, constant uint& nz, constant uint& k,
    constant float& scale_hi, constant float& scale_lo,
    constant float& shift_hi, constant float& shift_lo,
    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float4 sum = float4(0.0f);
  band_mac(sum, band[i], t_current[i]);
  for (uint o = 1; o <= b; o++) {
    if (i + o < n) band_mac(sum, conj4(band[o * n + i]), t_current[i + o]);
    if (i >= o) band_mac(sum, band[o * n + i - o], t_current[i - o]);
  }
  cdf acc = cload(sum);
  cdf v = cscale(acc, float2(scale_hi, scale_lo));
  v = csub(v, cscale(cload(t_current[i]), float2(shift_hi, shift_lo)));
  v = csub(v, cload(t_previous[i]));
  t_next[i] = float4(v.re.x, v.re.y, v.im.x, v.im.y);
  for (uint z = 0; z < nz; z++) {
    cdf r = cadd(cload(result[i * nz + z]), cmul(cload(coefficients[k * nz + z]), v));
    result[i * nz + z] = float4(r.re.x, r.re.y, r.im.x, r.im.y);
  }
}
"""

_LIBRARIES: dict = {}


def _library():
    if "band" not in _LIBRARIES:
        source = (
            df._HELPERS
            + df._COMPLEX
            + _GEMM
            + _BAND_MAC
            + _MIRROR
            + _CHEBYSHEV
        )
        _LIBRARIES["band"] = df.torch.mps.compile_shader(source)
    return _LIBRARIES["band"]


def gemm(
    a,
    b,
    c,
    shape: tuple[int, int, int],
    a_view: tuple[int, int],
    b_view: tuple[int, int],
    c_view: tuple[int, int],
    conjugate_transpose_b: bool = False,
    subtract: bool = False,
    lower_only: bool = False,
) -> None:
    """``c = a b`` or ``c -= a b`` for the (M, N, K) = `shape` product, the operands
    being views `(offset, row stride)`, in elements, of the double-float arrays
    `a`, `b`, `c`; with `conjugate_transpose_b`, `b` is stored as N x K and its
    conjugate transpose is the factor. With `lower_only`, the product is Hermitian
    and only the tiles that touch its lower triangle are computed (`mirror` fills
    in the rest)."""
    m, n, k = shape
    tile = _GEMM_TILE
    gx, gy = -(-n // tile), -(-m // tile)
    _library().gemm_df(
        a.data,
        b.data,
        c.data,
        m,
        n,
        k,
        a_view[0],
        a_view[1],
        b_view[0],
        b_view[1],
        c_view[0],
        c_view[1],
        1 if conjugate_transpose_b else 0,
        1 if subtract else 0,
        1 if lower_only else 0,
        threads=(gx * 8, gy * 8),
        group_size=(8, 8),
    )


def mirror(c, n: int, view: tuple[int, int]) -> None:
    """Set the upper triangle, beyond the diagonal tiles, of the n x n Hermitian matrix
    at the view `(offset, row stride)` of `c` to the conjugate of its lower one."""
    _library().mirror_df(c.data, n, view[0], view[1], threads=(n, n), group_size=(32, 8))


# From this many beams the recurrence runs on the GPU: measured on an Apple M2 (1299
# beams: 0.16 s on the host, 0.24 s on the GPU; 2195: host 0.4 s slower overall;
# 3233: the GPU 1.6 s faster), the host being one core with the band in its cache
# and the GPU a kernel per term over rows that fill it only for thousands of beams.
GPU_RECURRENCE_MIN_BEAMS = 1600


def default_bandwidth(n: int) -> int:
    """The bandwidth of the reduction for `n` beams. A wider band makes the
    reduction cheaper (fewer, larger panels) and the recurrence dearer (a term costs
    2 b + 1 multiply-adds a row); 32 was the fastest, or within 5% of it, for 1299,
    2195 and 3233 beams on an Apple M2 (16 to 96 tried)."""
    return 32


def _split_constant(value: float) -> tuple[float, float]:
    hi = float(np.float32(value))
    return hi, float(np.float32(value - hi))


@df._on_metal
def reduce_to_band(
    A: np.ndarray, bandwidth: int
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Reduce the Hermitian matrix `A` (complex128, N x N, C-ordered, on the host) to
    band form on the GPU: returns the lower band, ``band[o, i] = B[i + o, i]`` as a
    complex128 array of shape (bandwidth + 1, N), and, for each panel, its first
    row, the reflectors V and the triangular factor T of the block reflector."""
    torch = df.torch
    n = len(A)
    matrix = df.from_numpy(A)
    band = np.zeros((bandwidth + 1, n), dtype=np.complex128)
    factors = []
    column = 0
    while column < n:
        start = column + bandwidth
        m = n - start
        if m <= 1:
            # the end: what is left is a block of at most bandwidth + 1 columns
            rest = df.to_numpy(df.DoubleFloatArray(matrix.data[column:, column:]))
            for o in range(len(rest)):
                band[o, column : n - o] = np.diagonal(rest, -o)
            break
        # the columns of the panel, from the diagonal block down
        block = df.to_numpy(
            df.DoubleFloatArray(matrix.data[column:, column:start].contiguous())
        )
        diagonal_block, panel = block[:bandwidth], block[bandwidth:]
        # the diagonal block is final: its lower triangle is band
        for j in range(bandwidth):
            for i in range(j, bandwidth):
                band[i - j, column + j] = diagonal_block[i, j]
        stored, tau = np.linalg.qr(panel, mode="raw")  # transposed, as NumPy's
        stored = np.asfortranarray(stored.T)
        k = min(m, bandwidth)
        V, T, identity = _block_reflector(stored[:, :k], tau[:k])
        # R, the upper triangle of the first k rows, sits in the band below the block
        for j in range(bandwidth):
            for i in range(min(j, k - 1) + 1):
                band[bandwidth + i - j, column + j] = stored[i, j]
        # trailing <- Q^H trailing Q, Q = I - V T V^H: with W = trailing V T and
        # W <- W - 1/2 V (T^H V^H W), trailing <- trailing - V W^H - W V^H
        vt = df.from_numpy(np.ascontiguousarray(V @ T))
        w = df.DoubleFloatArray(
            torch.empty((m, k, 4), dtype=torch.float32, device="mps")
        )
        gemm(matrix, vt, w, (m, k, m), (start * n + start, n), (0, k), (0, k))
        host_w = df.to_numpy(w)
        host_w -= 0.5 * V @ (T.conj().T @ (V.conj().T @ host_w))
        # the update C -= [V W] [W V]^H in one product of depth 2k
        left = df.from_numpy(np.ascontiguousarray(np.concatenate([V, host_w], axis=1)))
        right = df.from_numpy(
            np.ascontiguousarray(np.concatenate([host_w, V], axis=1))
        )
        gemm(
            left,
            right,
            matrix,
            (m, m, 2 * k),
            (0, 2 * k),
            (0, 2 * k),
            (start * n + start, n),
            conjugate_transpose_b=True,
            subtract=True,
            lower_only=True,
        )
        mirror(matrix, m, (start * n + start, n))
        factors.append((start, V, T))
        column = start
    return band, factors


@df._on_metal
def chebyshev_propagate(
    band: np.ndarray,
    scales: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    """exp(i s B) e_0 for each s, as an (n, len(scales)) complex128 array, from the
    lower band of the Hermitian band matrix B and bounds on its spectrum; the
    recurrence, with its sum over the terms, runs on the GPU in double-float."""
    bandwidth, n = band.shape[0] - 1, band.shape[1]
    nz = len(scales)
    coefficients, center, half = chebyshev_coefficients(
        scales, bounds, float(np.finfo(np.float64).eps) * _TRUNCATION
    )
    terms = len(coefficients)
    band_device = df.from_numpy(np.ascontiguousarray(band))
    coefficients_device = df.from_numpy(np.ascontiguousarray(coefficients))
    result = np.zeros((n, nz), dtype=np.complex128)
    result[0] = coefficients[0]  # e_0 times the coefficient of T_0
    result_device = df.from_numpy(result)

    def vector(index=None):
        host = np.zeros(n, dtype=np.complex128)
        if index is not None:
            host[index] = 1.0
        return df.from_numpy(host)

    library = _library()
    first = (1.0 / half, center / half)
    later = (2.0 / half, 2.0 * center / half)
    t_previous, t_current, t_next = vector(), vector(0), vector()
    for k in range(1, terms):
        scale, shift = first if k == 1 else later
        (scale_hi, scale_lo), (shift_hi, shift_lo) = (
            _split_constant(scale),
            _split_constant(shift),
        )
        library.chebyshev_step(
            band_device.data,
            t_current.data,
            t_previous.data,
            t_next.data,
            coefficients_device.data,
            result_device.data,
            n,
            bandwidth,
            nz,
            k,
            scale_hi,
            scale_lo,
            shift_hi,
            shift_lo,
            threads=n,
        )
        t_previous, t_current, t_next = t_current, t_next, t_previous
    return df.to_numpy(result_device)


def propagate_incident_beam(
    A: np.ndarray,
    scales: np.ndarray,
    direct: int = 0,
    bandwidth: int | None = None,
    bounds: tuple[float, float] | None = None,
    recurrence: str = "auto",
) -> np.ndarray:
    """exp(i s A) e_direct for each s, as `abtem.bloch.band_chebyshev` finds it, with
    the band reduction on the GPU in emulated double precision.

    Parameters
    ----------
    A : numpy.ndarray
        The Hermitian matrix, N x N, on the host. It is left alone.
    scales : sequence of float
        The values of s.
    direct : int
        The index of the incident beam.
    bandwidth : int, optional
        The bandwidth of the reduction (see `default_bandwidth`).
    bounds : tuple of float, optional
        Bounds on the spectrum of A; Gershgorin bounds if not given.
    recurrence : {'auto', 'host', 'gpu'}
        Where the Chebyshev recurrence runs, in float64 on the host or in
        double-float on the GPU. 'auto' takes the host for fewer than
        `GPU_RECURRENCE_MIN_BEAMS` beams, where its band stays in the cache of the
        CPU and the launch of a kernel per term costs more than a term.

    Returns
    -------
    numpy.ndarray
        The (N, len(scales)) complex128 result, in the order of A.
    """
    A = np.array(A, dtype=np.complex128, order="C", copy=True)
    n = len(A)
    if bounds is None:
        bounds = gershgorin_bounds(A)
    # the direct beam first: the reduction never transforms the first row
    if direct:
        A[[0, direct]] = A[[direct, 0]]
        A[:, [0, direct]] = A[:, [direct, 0]]
    bandwidth = max(1, min(default_bandwidth(n) if bandwidth is None else bandwidth, n - 1))
    band, factors = reduce_to_band(A, bandwidth)
    if recurrence == "auto":
        recurrence = "gpu" if n >= GPU_RECURRENCE_MIN_BEAMS else "host"
    propagate = chebyshev_propagate if recurrence == "gpu" else chebyshev_propagate_host
    y = propagate(band, np.asarray(scales, dtype=float), bounds)
    for start, V, T in reversed(factors):
        y[start:] -= V @ (T @ (V.conj().T @ y[start:]))
    if direct:
        y[[0, direct]] = y[[direct, 0]]
    return y
