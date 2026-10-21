"""Emulated double precision on Apple Metal, by double-float arithmetic.

Metal has no float64, so a float64 number is carried as an unevaluated sum
``hi + lo`` of two float32 numbers, with ``|lo| <= ulp(hi) / 2``. Arithmetic on
such pairs, built from error-free transforms (Knuth's two-sum and an
FMA-based two-product), reaches about 44 bits of mantissa: ~1e-14 relative
accuracy against float32's ~1e-7, with the exponent range of float32.

Only what the multislice inner loop needs is provided -- a complex array type
(four float32 planes), an FFT over the last two axes, and an elementwise
product with a broadcast kernel. Everything around it (the transmission
function, the Fresnel propagator, detectors) stays on the host in float64 and
meets this module only at the boundaries, through :func:`from_numpy` and
:func:`to_numpy`.

The FFT is a mixed-radix (2, 3, 4, 5, 7) Stockham transform that runs a whole 1D
transform in one threadgroup, in threadgroup memory. That bounds the transformed
axes to lengths with no prime factor above 7, up to :data:`MAX_FFT_SIZE`;
anything else raises, as the alternative is a silent fall back to float32
accuracy.
"""

from __future__ import annotations

import numpy as np

from abtem.core._torch import DEVICE, _serialized, torch

# One threadgroup transforms one row, ping-ponging between two buffers of four
# float32 planes each: 2 * 4 * 4 bytes per element, against Metal's 32 KiB of
# threadgroup memory.
MAX_FFT_SIZE = 1024

_HELPERS = """
#include <metal_stdlib>
using namespace metal;
// The error-free transforms are only valid if the compiler keeps the order of
// operations. torch's compile options leave fast math off, but the pragma makes
// that independent of them.
#pragma METAL fp math_mode(safe)

inline float2 two_sum(float a, float b) {
  float s = a + b; float bb = s - a;
  return float2(s, (a - (s - bb)) + (b - bb));
}
inline float2 quick_two_sum(float a, float b) {
  float s = a + b; return float2(s, b - (s - a));
}
inline float2 two_prod(float a, float b) {
  float p = a * b; return float2(p, fma(a, b, -p));
}
inline float2 df_add(float2 a, float2 b) {
  float2 s = two_sum(a.x, b.x); float2 t = two_sum(a.y, b.y);
  s.y += t.x; s = quick_two_sum(s.x, s.y); s.y += t.y;
  return quick_two_sum(s.x, s.y);
}
inline float2 df_sub(float2 a, float2 b) { return df_add(a, float2(-b.x, -b.y)); }
inline float2 df_mul(float2 a, float2 b) {
  float2 p = two_prod(a.x, b.x); p.y += a.x * b.y + a.y * b.x;
  return quick_two_sum(p.x, p.y);
}
"""

_MULTIPLY = """
// y = x * k, with k broadcast over the leading axis: x and y hold nk-sized
// blocks and k holds one.
kernel void multiply_df(
    device const float* xrh, device const float* xrl,
    device const float* xih, device const float* xil,
    device const float* krh, device const float* krl,
    device const float* kih, device const float* kil,
    device float* yrh, device float* yrl, device float* yih, device float* yil,
    constant uint& nk, uint i [[thread_position_in_grid]]) {
  uint j = i % nk;
  float2 xr = float2(xrh[i], xrl[i]), xi = float2(xih[i], xil[i]);
  float2 kr = float2(krh[j], krl[j]), ki = float2(kih[j], kil[j]);
  float2 yr = df_sub(df_mul(xr, kr), df_mul(xi, ki));
  float2 yi = df_add(df_mul(xr, ki), df_mul(xi, kr));
  yrh[i] = yr.x; yrl[i] = yr.y; yih[i] = yi.x; yil[i] = yi.y;
}
"""


_COMPLEX = """
struct cdf { float2 re; float2 im; };
inline cdf cadd(cdf a, cdf b) { return cdf{df_add(a.re, b.re), df_add(a.im, b.im)}; }
inline cdf csub(cdf a, cdf b) { return cdf{df_sub(a.re, b.re), df_sub(a.im, b.im)}; }
inline cdf cmul(cdf a, cdf b) {
  return cdf{df_sub(df_mul(a.re, b.re), df_mul(a.im, b.im)),
             df_add(df_mul(a.re, b.im), df_mul(a.im, b.re))};
}
inline cdf cscale(cdf a, float2 c) { return cdf{df_mul(a.re, c), df_mul(a.im, c)}; }
inline float2 fsign(float2 v, float s) { return float2(s * v.x, s * v.y); }
// i * s * a, with s = +-1: exact.
inline cdf cmuli(cdf a, float s) { return cdf{fsign(a.im, -s), fsign(a.re, s)}; }

inline void bfly2(thread cdf* x, float s) {
  cdf a = x[0], b = x[1]; x[0] = cadd(a, b); x[1] = csub(a, b);
}
inline void bfly4(thread cdf* x, float s) {
  cdf e = cadd(x[0], x[2]), f = csub(x[0], x[2]);
  cdf g = cadd(x[1], x[3]), h = cmuli(csub(x[1], x[3]), s);
  x[0] = cadd(e, g); x[2] = csub(e, g); x[1] = cadd(f, h); x[3] = csub(f, h);
}
"""


def _df_literal(value: float) -> str:
    hi = np.float32(value)
    lo = np.float32(value - float(hi))
    return f"float2({float(hi)!r}f, {float(lo)!r}f)"


def _odd_butterfly(r: int) -> str:
    """The length-``r`` DFT, for an odd ``r``, in place on ``x``.

    The inputs pair up as ``x[j] +- x[r - j]``, so each output pair needs only
    real-constant products: ``X[k], X[r-k] = A[k] -+ i s B[k]``, with ``A`` the
    cosine-weighted sums of the pair sums and ``B`` the sine-weighted sums of
    the pair differences.
    """
    h = (r - 1) // 2
    lines = [f"inline void bfly{r}(thread cdf* x, float s) {{", "  cdf x0 = x[0];"]
    for j in range(1, h + 1):
        lines.append(
            f"  cdf p{j} = cadd(x[{j}], x[{r - j}]); "
            f"cdf d{j} = csub(x[{j}], x[{r - j}]);"
        )
    total = "x0"
    for j in range(1, h + 1):
        total = f"cadd({total}, p{j})"
    for k in range(1, h + 1):
        a, b = "x0", None
        for j in range(1, h + 1):
            angle = 2 * np.pi * ((j * k) % r) / r
            a = f"cadd({a}, cscale(p{j}, {_df_literal(np.cos(angle))}))"
            term = f"cscale(d{j}, {_df_literal(np.sin(angle))})"
            b = term if b is None else f"cadd({b}, {term})"
        lines.append(f"  cdf a{k} = {a}; cdf b{k} = {b};")
        lines.append(
            f"  x[{k}] = cdf{{df_sub(a{k}.re, fsign(b{k}.im, s)), "
            f"df_add(a{k}.im, fsign(b{k}.re, s))}};"
        )
        lines.append(
            f"  x[{r - k}] = cdf{{df_add(a{k}.re, fsign(b{k}.im, s)), "
            f"df_sub(a{k}.im, fsign(b{k}.re, s))}};"
        )
    lines.append(f"  x[0] = {total};")
    lines.append("}")
    return "\n".join(lines)


SUPPORTED_RADICES = (7, 5, 3, 4, 2)


def radices(n: int) -> list[int] | None:
    """The radices of the passes that transform ``n`` points, or None when ``n``
    has a prime factor other than 2, 3, 5 and 7."""
    remaining, factors = n, []
    for prime in (7, 5, 3):
        while remaining % prime == 0:
            factors.append(prime)
            remaining //= prime
    twos = 0
    while remaining % 2 == 0:
        twos += 1
        remaining //= 2
    if remaining != 1 or n < 2:
        return None
    return factors + [4] * (twos // 2) + [2] * (twos % 2)


def _fft_source(n: int) -> str:
    passes = radices(n)
    threads = _width(n)
    body, length, stride = [], n, 1
    for index, r in enumerate(passes):
        m = length // r
        body.append(f"""
  // pass {index}: radix {r}, length {length}, stride {stride}
  for (uint bt = t; bt < {n // r}u; bt += {threads}u) {{
    uint p = bt / {stride}u, q = bt - p * {stride}u;
    cdf x[{r}];
    for (uint j = 0; j < {r}u; j++) {{
      uint i = q + {stride}u * (p + j * {m}u);
      x[j] = cdf{{float2(src[0*N+i], src[1*N+i]), float2(src[2*N+i], src[3*N+i])}};
    }}
    bfly{r}(x, sign);
    for (uint k = 0; k < {r}u; k++) {{
      cdf v = x[k];
      if (k > 0) {{
        uint w = p * k * {n // length}u;
        cdf tw = cdf{{float2(wrh[w], wrl[w]), float2(wih[w], wil[w])}};
        if (sign > 0.0f) {{ tw.im = fsign(tw.im, -1.0f); }}
        v = cmul(v, tw);
      }}
      uint o = q + {stride}u * ({r}u * p + k);
      dst[0*N+o] = v.re.x; dst[1*N+o] = v.re.y;
      dst[2*N+o] = v.im.x; dst[3*N+o] = v.im.y;
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {{ threadgroup float* tmp = src; src = dst; dst = tmp; }}""")
        length, stride = m, stride * r

    # Each radix's butterfly once, however many passes use it.
    butterflies = "\n".join(
        _odd_butterfly(r) for r in sorted({r for r in passes if r in (3, 5, 7)})
    )
    return _HELPERS + _MULTIPLY + _COMPLEX + butterflies + f"""
#define N {n}u
// A Stockham mixed-radix FFT of one line per threadgroup, {threads} threads
// wide. Lines are addressed as base + i * es with base = b * bs + m * ms, so
// one kernel transforms the rows (es = 1) and the columns (es = row length) of
// a batch of images without a transpose. sign < 0 is the forward transform.
// `scale`, a double-float constant, multiplies the output. The
// twiddle table holds exp(-2 pi i k / N) for k < N.
kernel void fft_df(
    device const float* xrh, device const float* xrl,
    device const float* xih, device const float* xil,
    device float* yrh, device float* yrl, device float* yih, device float* yil,
    device const float* wrh, device const float* wrl,
    device const float* wih, device const float* wil,
    constant uint& M, constant uint& bs, constant uint& ms, constant uint& es,
    constant float& sign, constant float& scale_hi, constant float& scale_lo,
    uint tgid [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]]) {{
  threadgroup float buf_a[4 * N];
  threadgroup float buf_b[4 * N];
  uint b = tgid / M, m = tgid - b * M;
  uint base = b * bs + m * ms;
  for (uint i = t; i < N; i += {threads}u) {{
    uint g = base + i * es;
    buf_a[0*N+i] = xrh[g]; buf_a[1*N+i] = xrl[g];
    buf_a[2*N+i] = xih[g]; buf_a[3*N+i] = xil[g];
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* src = buf_a;
  threadgroup float* dst = buf_b;
{"".join(body)}
  for (uint i = t; i < N; i += {threads}u) {{
    uint g = base + i * es;
    float2 sc = float2(scale_hi, scale_lo);
    float2 re = df_mul(float2(src[0*N+i], src[1*N+i]), sc);
    float2 im = df_mul(float2(src[2*N+i], src[3*N+i]), sc);
    yrh[g] = re.x; yrl[g] = re.y; yih[g] = im.x; yil[g] = im.y;
  }}
}}
"""


# Compiled kernels and twiddle tables, by transform length. Compiling a Metal
# library costs milliseconds and is done once per length; the tables are
# float64 on the host, split on the way up so their accuracy is the pair's.
_LIBRARIES: dict = {}
_TWIDDLES: dict = {}


def _library(n: int):
    if n not in _LIBRARIES:
        _LIBRARIES[n] = torch.mps.compile_shader(_fft_source(n))
    return _LIBRARIES[n]


def _multiply_library():
    if "multiply" not in _LIBRARIES:
        _LIBRARIES["multiply"] = torch.mps.compile_shader(_HELPERS + _MULTIPLY)
    return _LIBRARIES["multiply"]


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = x.astype(np.float32)
    return hi, (x - hi.astype(np.float64)).astype(np.float32)


def _twiddles(n: int):
    if n not in _TWIDDLES:
        w = np.exp(-2j * np.pi * np.arange(n) / n)
        _TWIDDLES[n] = tuple(
            torch.from_numpy(plane).to(DEVICE)
            for plane in (*_split(w.real), *_split(w.imag))
        )
    return _TWIDDLES[n]


class DoubleFloatArray:
    """A complex array as four float32 planes on the device: the high and low
    parts of the real and of the imaginary part.

    Not a NumPy-like array: it exists to carry data across the device loop and
    supports only the operations in this module.
    """

    __slots__ = ("planes",)

    def __init__(self, planes: tuple):
        self.planes = tuple(planes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.planes[0].shape)

    def empty_like(self) -> "DoubleFloatArray":
        return DoubleFloatArray(tuple(torch.empty_like(p) for p in self.planes))


# Metal caps the threads of a threadgroup at what the kernel's registers allow,
# 448 for these kernels; a transform wider than this loops over its butterflies.
_MAX_THREADS = 256


def _width(n: int) -> int:
    """Threads per threadgroup: one per butterfly of the smallest radix, up to
    :data:`_MAX_THREADS`."""
    return min(n // min(radices(n)), _MAX_THREADS)


def _check_axis(n: int) -> None:
    if radices(n) is None or n > MAX_FFT_SIZE:
        raise ValueError(
            "emulated double precision on Metal transforms axes whose length "
            f"has no prime factor above 7 and is at most {MAX_FFT_SIZE}, got "
            f"{n}. Choose a grid of that size, or use the 'cpu' or 'gpu' device "
            "for other sizes."
        )


@_serialized
def from_numpy(x: np.ndarray) -> DoubleFloatArray:
    """Upload a float64 or complex128 array, keeping all of its precision."""
    x = np.ascontiguousarray(x)
    if x.dtype not in (np.float64, np.complex128):
        x = x.astype(np.complex128)
    real = np.ascontiguousarray(x.real, dtype=np.float64)
    imag = (
        np.ascontiguousarray(x.imag, dtype=np.float64)
        if np.iscomplexobj(x)
        else np.zeros_like(real)
    )
    return DoubleFloatArray(
        tuple(torch.from_numpy(p).to(DEVICE) for p in (*_split(real), *_split(imag)))
    )


@_serialized
def to_numpy(x: DoubleFloatArray) -> np.ndarray:
    """Download as a complex128 array."""
    rh, rl, ih, il = (p.cpu().numpy().astype(np.float64) for p in x.planes)
    return (rh + rl) + 1j * (ih + il)


@_serialized
def multiply(x: DoubleFloatArray, kernel: DoubleFloatArray) -> DoubleFloatArray:
    """The product ``x * kernel``, with the (2D) kernel broadcast over the
    leading axes of ``x``."""
    nk = int(np.prod(kernel.shape))
    if x.shape[-len(kernel.shape) :] != kernel.shape:
        raise ValueError(
            f"kernel of shape {kernel.shape} does not broadcast over {x.shape}"
        )
    y = x.empty_like()
    _multiply_library().multiply_df(
        *x.planes, *kernel.planes, *y.planes, nk, threads=int(np.prod(x.shape))
    )
    return y


@_serialized
def fft2(x: DoubleFloatArray, inverse: bool = False) -> DoubleFloatArray:
    """The 2D FFT over the last two axes, normalised like ``numpy.fft``.

    The input is left untouched.
    """
    *batch_shape, n0, n1 = x.shape
    _check_axis(n0)
    _check_axis(n1)

    batch = int(np.prod(batch_shape)) if batch_shape else 1
    sign = 1.0 if inverse else -1.0
    # 1 / (n0 n1) is not a float32 unless the grid is a power of two: scale in
    # double-float too, or the inverse would be good to float32 only.
    scale = 1.0 / (n0 * n1) if inverse else 1.0
    scale_hi = float(np.float32(scale))
    scale_lo = float(np.float32(scale - scale_hi))

    rows = x.empty_like()
    out = x.empty_like()
    # Along the last axis: n0 lines per image, contiguous.
    _library(n1).fft_df(
        *x.planes,
        *rows.planes,
        *_twiddles(n1),
        n0,
        n0 * n1,
        n1,
        1,
        sign,
        1.0,
        0.0,
        threads=batch * n0 * _width(n1),
        group_size=_width(n1),
    )
    # Along the first: n1 lines per image, strided by the row length.
    _library(n0).fft_df(
        *rows.planes,
        *out.planes,
        *_twiddles(n0),
        n1,
        n0 * n1,
        1,
        n1,
        sign,
        scale_hi,
        scale_lo,
        threads=batch * n1 * _width(n0),
        group_size=_width(n0),
    )
    return out
