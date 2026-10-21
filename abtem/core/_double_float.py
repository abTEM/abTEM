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

The FFT is a radix-2 Stockham transform that runs a whole 1D transform in one
threadgroup, in threadgroup memory. That bounds the transformed axes to powers
of two up to :data:`MAX_FFT_SIZE`; anything else raises, as the alternative is a
silent fall back to float32 accuracy.
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


def _fft_source(n: int) -> str:
    return _HELPERS + _MULTIPLY + f"""
#define N {n}u
// A Stockham radix-2 FFT of one line per threadgroup, N / 2 threads wide. Lines
// are addressed as base + i * es with base = b * bs + m * ms, so one kernel
// transforms the rows (es = 1) and the columns (es = row length) of a batch of
// images without a transpose. sign < 0 is the forward transform. `scale` is
// applied to the output; with a power of two it is exact.
kernel void fft_df(
    device const float* xrh, device const float* xrl,
    device const float* xih, device const float* xil,
    device float* yrh, device float* yrl, device float* yih, device float* yil,
    device const float* wrh, device const float* wrl,
    device const float* wih, device const float* wil,
    constant uint& M, constant uint& bs, constant uint& ms, constant uint& es,
    constant float& sign, constant float& scale,
    uint tgid [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]]) {{
  threadgroup float buf_a[4 * N];
  threadgroup float buf_b[4 * N];
  uint b = tgid / M, m = tgid - b * M;
  uint base = b * bs + m * ms;
  uint i0 = t, i1 = t + (N >> 1);
  uint g0 = base + i0 * es, g1 = base + i1 * es;
  buf_a[0*N+i0] = xrh[g0]; buf_a[1*N+i0] = xrl[g0];
  buf_a[2*N+i0] = xih[g0]; buf_a[3*N+i0] = xil[g0];
  buf_a[0*N+i1] = xrh[g1]; buf_a[1*N+i1] = xrl[g1];
  buf_a[2*N+i1] = xih[g1]; buf_a[3*N+i1] = xil[g1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* src = buf_a;
  threadgroup float* dst = buf_b;
  uint n = N, s = 1;
  while (n > 1) {{
    uint half_n = n >> 1;
    uint p = t / s, q = t - p * s;
    uint ia = q + s * p, ib = q + s * (p + half_n);
    float2 ar = float2(src[0*N+ia], src[1*N+ia]), ai = float2(src[2*N+ia], src[3*N+ia]);
    float2 br = float2(src[0*N+ib], src[1*N+ib]), bi = float2(src[2*N+ib], src[3*N+ib]);
    uint k = p * (N / n);
    float2 wr = float2(wrh[k], wrl[k]), wi = float2(wih[k], wil[k]);
    if (sign > 0.0f) {{ wi = float2(-wi.x, -wi.y); }}
    float2 sr = df_add(ar, br), si = df_add(ai, bi);
    float2 dr = df_sub(ar, br), di = df_sub(ai, bi);
    float2 tr = df_sub(df_mul(dr, wr), df_mul(di, wi));
    float2 ti = df_add(df_mul(dr, wi), df_mul(di, wr));
    uint oa = q + s * (2 * p), ob = q + s * (2 * p + 1);
    dst[0*N+oa] = sr.x; dst[1*N+oa] = sr.y; dst[2*N+oa] = si.x; dst[3*N+oa] = si.y;
    dst[0*N+ob] = tr.x; dst[1*N+ob] = tr.y; dst[2*N+ob] = ti.x; dst[3*N+ob] = ti.y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* tmp = src; src = dst; dst = tmp;
    n = half_n; s <<= 1;
  }}
  yrh[g0] = src[0*N+i0] * scale; yrl[g0] = src[1*N+i0] * scale;
  yih[g0] = src[2*N+i0] * scale; yil[g0] = src[3*N+i0] * scale;
  yrh[g1] = src[0*N+i1] * scale; yrl[g1] = src[1*N+i1] * scale;
  yih[g1] = src[2*N+i1] * scale; yil[g1] = src[3*N+i1] * scale;
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
        w = np.exp(-2j * np.pi * np.arange(n // 2) / n)
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


def _check_axis(n: int) -> None:
    if n < 2 or n & (n - 1) or n > MAX_FFT_SIZE:
        raise ValueError(
            f"emulated double precision on Metal transforms axes whose length is "
            f"a power of two up to {MAX_FFT_SIZE}, got {n}. Choose a grid of that "
            "size, or use the 'cpu' or 'gpu' device for other sizes."
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
    # The unitary-in-a-power-of-two scale of the inverse transform is exact.
    scale = 1.0 / (n0 * n1) if inverse else 1.0

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
        threads=batch * n0 * (n1 // 2),
        group_size=n1 // 2,
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
        scale,
        threads=batch * n1 * (n0 // 2),
        group_size=n0 // 2,
    )
    return out
