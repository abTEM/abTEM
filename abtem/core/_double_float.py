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
transform in one threadgroup, in threadgroup memory, which holds up to
:data:`MAX_FFT_SIZE` points; a longer axis is split into two such transforms
(Bailey's four-step algorithm). Axes are limited to lengths with no prime factor
above 7 that split in two, which is every such length up to
:data:`MAX_FFT_SIZE` squared; anything else raises, as the alternative is a
silent fall back to float32 accuracy.
"""

from __future__ import annotations

import functools

import numpy as np

from abtem.core.fft import warn_if_slow_emulated_fft

# PyTorch and abtem.core._torch, loaded on first use by _load(): importing them
# here would bypass the guards of abtem.core.backend._load_torch.
torch = None
_TORCH_MODULE = None


def _load():
    global torch, _TORCH_MODULE
    if _TORCH_MODULE is None:
        from abtem.core.backend import check_metal_kernels_are_available

        module = check_metal_kernels_are_available()
        torch = module.torch
        _TORCH_MODULE = module
    return _TORCH_MODULE


def _on_metal(func):
    """Run ``func`` with PyTorch loaded and holding the Metal lock, which also
    covers the kernel and table caches (see ``abtem.core._torch._TORCH_LOCK``)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _load()._TORCH_LOCK:
            return func(*args, **kwargs)

    return wrapper


# One threadgroup transforms a line in place in one buffer of four float32
# planes: 16 bytes per element, against Metal's 32 KiB of threadgroup memory.
MAX_FFT_SIZE = 2048

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
// The "sloppy" sum: the low parts are added plainly, not through a second
// two-sum. Its error is about one unit in the last place of the larger operand
// (2^-46 of |a| + |b|), where the careful sum's is relative to the result; the
// transforms and products here carry errors of that size anyway, as a transform
// does of its norm, and measured end to end the results are as accurate (a
// 136-slice multislice agrees with float64 to 1.5e-13 either way), at 11% less
// time.
inline float2 df_add(float2 a, float2 b) {
  float2 s = two_sum(a.x, b.x);
  s.y += a.y + b.y;
  return quick_two_sum(s.x, s.y);
}
inline float2 df_sub(float2 a, float2 b) { return df_add(a, float2(-b.x, -b.y)); }
inline float2 df_mul(float2 a, float2 b) {
  float2 p = two_prod(a.x, b.x); p.y += a.x * b.y + a.y * b.x;
  return quick_two_sum(p.x, p.y);
}
"""

_MULTIPLY = """
// y = x * k, with k broadcast over the leading axis: x and y hold n elements per
// plane, k holds nk. An array is its four planes, one after the other.
kernel void multiply_df(
    device const float* x, device const float* k, device float* y,
    constant uint& n, constant uint& nk, uint i [[thread_position_in_grid]]) {
  uint j = i % nk;
  cdf a = cdf{float2(x[i], x[n + i]), float2(x[2*n + i], x[3*n + i])};
  cdf b = cdf{float2(k[j], k[nk + j]), float2(k[2*nk + j], k[3*nk + j])};
  cdf r = cmul(a, b);
  y[i] = r.re.x; y[n + i] = r.re.y; y[2*n + i] = r.im.x; y[3*n + i] = r.im.y;
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


def _fft_source(n: int, group: int = 1) -> str:
    passes = radices(n)
    threads = _width(n, group)
    body, length, stride = [], n, 1
    for index, r in enumerate(passes):
        m = length // r
        count = n // r  # butterflies in the pass, per line
        per_thread = -(-count * group // threads)
        guard = "" if count * group % threads == 0 else f"if (bt < {count * group}u) "
        body.append(f"""
  // pass {index}: radix {r}, length {length}, stride {stride}. The inputs of
  // this thread's butterflies are read into registers before anything is
  // written, so the pass works in place in one buffer.
  {{
    cdf x[{per_thread}][{r}];
    for (uint u = 0; u < {per_thread}u; u++) {{
      uint bt = t + u * {threads}u;
      {guard}{{
        uint l = bt / {count}u, f = bt - l * {count}u;
        uint p = f / {stride}u, q = f - p * {stride}u;
        for (uint j = 0; j < {r}u; j++) {{
          uint i = l * N + q + {stride}u * (p + j * {m}u);
          x[u][j] = cdf{{float2(buf[0*NL+i], buf[1*NL+i]),
                        float2(buf[2*NL+i], buf[3*NL+i])}};
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = 0; u < {per_thread}u; u++) {{
      uint bt = t + u * {threads}u;
      {guard}{{
        uint l = bt / {count}u, f = bt - l * {count}u;
        uint p = f / {stride}u, q = f - p * {stride}u;
        bfly{r}(x[u], sign);
        for (uint k = 0; k < {r}u; k++) {{
          cdf v = x[u][k];
          if (k > 0) {{
            uint w = p * k * {n // length}u;
            cdf tw = cdf{{float2(wt[w], wt[WL + w]),
                         float2(wt[2*WL + w], wt[3*WL + w])}};
            if (sign > 0.0f) {{ tw.im = fsign(tw.im, -1.0f); }}
            v = cmul(v, tw);
          }}
          uint o = l * N + q + {stride}u * ({r}u * p + k);
          buf[0*NL+o] = v.re.x; buf[1*NL+o] = v.re.y;
          buf[2*NL+o] = v.im.x; buf[3*NL+o] = v.im.y;
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}""")
        length, stride = m, stride * r

    # Each radix's butterfly once, however many passes use it.
    butterflies = "\n".join(
        _odd_butterfly(r) for r in sorted({r for r in passes if r in (3, 5, 7)})
    )
    return _HELPERS + _COMPLEX + _MULTIPLY + butterflies + f"""
#define N {n}u
#define L {group}u
#define NL {n * group}u
// A Stockham mixed-radix FFT of L lines per threadgroup, {threads} threads
// wide; several lines let the loads and stores of strided lines (columns)
// touch neighbouring addresses together. A line is addressed as base + i * es,
// with the base of line number tgid * L + l = (b * C + c) * J + j being
// b * bs + c * cs + j * js, and the input and the output addressed apart. So
// one kernel transforms the rows (es = 1) and the columns (es = row length) of a
// batch of images without a transpose, and, with J > 1, a pass of a four-step
// transform. sign < 0 is the forward
// transform. `scale`, a double-float constant, multiplies the output.
//
// Three optional factors ride on the loads and the stores, each a table of
// one image or one axis, so the passes of a multislice step need no kernel of
// their own for the products between them: with pre_on, the loaded element is
// multiplied by pre at its position in the image; with post_on, the stored one
// by post; with tw_on, element i of line j by exp(-2 pi i i j / N') from the
// table v, N' the length of the whole four-step axis. The table w holds
// exp(-2 pi i k / N) for k < N. An array is its four planes one after the
// other; PL, WL, VL, PQ and OQ are the lengths of the planes of x and y, w, v,
// pre and post. The factor post may hold several images, one per entry of the
// batch of x or shared by several: with pmap_on, image pmap[b] of post, of OI
// elements each, is the factor of image b of x.
kernel void fft_df(
    device const float* x, device float* y,
    device const float* wt, device const float* vt,
    device const float* pre, device const float* post, device const int* pmap,
    constant uint& PL, constant uint& WL, constant uint& VL,
    constant uint& PQ, constant uint& OQ, constant uint& OI,
    constant uint& tw_on, constant uint& pre_on, constant uint& post_on,
    constant uint& pmap_on,
    constant uint& C, constant uint& J, constant uint& bs, constant uint& cs,
    constant uint& js_in, constant uint& js_out,
    constant uint& es_in, constant uint& es_out,
    constant float& sign, constant float& scale_hi, constant float& scale_lo,
    uint tgid [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]]) {{
  threadgroup float buf[4 * NL];
  uint jl[L], bl[L], local_in[L], local_out[L], base_in[L], base_out[L];
  for (uint l = 0; l < L; l++) {{
    uint line = tgid * L + l;
    uint j = line % J, rest = line / J;
    uint c = rest % C, b = rest / C;
    jl[l] = j;
    bl[l] = b;
    local_in[l] = c * cs + j * js_in;
    local_out[l] = c * cs + j * js_out;
    base_in[l] = b * bs + local_in[l];
    base_out[l] = b * bs + local_out[l];
  }}
  // Element i of line l at l * N + i in the buffer; consecutive threads take
  // consecutive lines, which for columns are neighbours in memory.
  for (uint e = t; e < NL; e += {threads}u) {{
    uint l = e % L, i = e / L;
    uint g = base_in[l] + i * es_in;
    cdf v = cdf{{float2(x[g], x[PL + g]), float2(x[2*PL + g], x[3*PL + g])}};
    if (pre_on != 0u) {{
      uint q = local_in[l] + i * es_in;
      v = cmul(v, cdf{{float2(pre[q], pre[PQ + q]),
                       float2(pre[2*PQ + q], pre[3*PQ + q])}});
    }}
    uint a = l * N + i;
    buf[0*NL+a] = v.re.x; buf[1*NL+a] = v.re.y;
    buf[2*NL+a] = v.im.x; buf[3*NL+a] = v.im.y;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
{"".join(body)}
  for (uint e = t; e < NL; e += {threads}u) {{
    uint l = e % L, i = e / L;
    uint g = base_out[l] + i * es_out;
    uint a = l * N + i;
    cdf v = cdf{{float2(buf[0*NL+a], buf[1*NL+a]),
                float2(buf[2*NL+a], buf[3*NL+a])}};
    if (tw_on != 0u) {{
      uint w = i * jl[l];
      cdf tw = cdf{{float2(vt[w], vt[VL + w]), float2(vt[2*VL + w], vt[3*VL + w])}};
      if (sign > 0.0f) {{ tw.im = fsign(tw.im, -1.0f); }}
      v = cmul(v, tw);
    }}
    if (post_on != 0u) {{
      uint q = local_out[l] + i * es_out;
      if (pmap_on != 0u) {{ q += uint(pmap[bl[l]]) * OI; }}
      v = cmul(v, cdf{{float2(post[q], post[OQ + q]),
                       float2(post[2*OQ + q], post[3*OQ + q])}});
    }}
    float2 sc = float2(scale_hi, scale_lo);
    float2 re = df_mul(v.re, sc);
    float2 im = df_mul(v.im, sc);
    y[g] = re.x; y[PL + g] = re.y; y[2*PL + g] = im.x; y[3*PL + g] = im.y;
  }}
}}
"""


# Compiled kernels and twiddle tables, by transform length. Compiling a Metal
# library costs milliseconds and is done once per length; the tables are
# float64 on the host, split on the way up so their accuracy is the pair's.
_LIBRARIES: dict = {}
_TWIDDLES: dict = {}


def _library(n: int, group: int = 1):
    if (n, group) not in _LIBRARIES:
        _LIBRARIES[n, group] = torch.mps.compile_shader(_fft_source(n, group))
    return _LIBRARIES[n, group]


def _multiply_library():
    if "multiply" not in _LIBRARIES:
        _LIBRARIES["multiply"] = torch.mps.compile_shader(
            _HELPERS + _COMPLEX + _MULTIPLY
        )
    return _LIBRARIES["multiply"]


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = x.astype(np.float32)
    return hi, (x - hi.astype(np.float64)).astype(np.float32)


def _twiddles(n: int):
    """The table ``exp(-2 pi i k / n)``, ``k < n``, as a double-float array."""
    if n not in _TWIDDLES:
        _TWIDDLES[n] = _upload(np.exp(-2j * np.pi * np.arange(n) / n))
    return _TWIDDLES[n]


def _upload(x: np.ndarray) -> "DoubleFloatArray":
    """Split a float64 or complex128 array into its four float32 planes.

    One pass per plane, and no temporaries: the ufuncs convert on the fly
    (``dtype=float64`` computes in float64, ``out`` takes it as float32), where
    ``astype`` and the arithmetic around it made seven passes and as many
    arrays, which was half the time of a thickness series.
    """
    planes = np.zeros((4, *x.shape), dtype=np.float32)
    for part, (high, low) in zip(
        (x.real, x.imag) if np.iscomplexobj(x) else (x,), ((0, 1), (2, 3))
    ):
        np.copyto(planes[high], part, casting="same_kind")
        np.subtract(
            part, planes[high], out=planes[low], dtype=np.float64, casting="same_kind"
        )
    return DoubleFloatArray(torch.from_numpy(planes).to("mps"))


class DoubleFloatArray:
    """A complex array as four float32 planes on the device, in one tensor of
    shape ``(4, *shape)``: the high and low parts of the real part, then of the
    imaginary part.

    Not a NumPy-like array: it exists to carry data across the device loop and
    supports only the operations in this module.
    """

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape[1:])

    def empty_like(self) -> "DoubleFloatArray":
        return DoubleFloatArray(torch.empty_like(self.data))


# Metal caps the threads of a threadgroup at what the kernel's registers allow,
# 448 for these kernels; a transform wider than this loops over its butterflies.
_MAX_THREADS = 256


def _width(n: int, group: int = 1) -> int:
    """Threads per threadgroup: one per butterfly of the smallest radix, up to
    :data:`_MAX_THREADS`."""
    return min(group * n // min(radices(n)), _MAX_THREADS)


# Lines per threadgroup for strided lines, the first that fits is used. Two was
# measured fastest on Apple M2: 15% off the column pass against one, while four
# and eight lose more to the larger threadgroup memory than they gain.
GROUP_LINES = (2,)


def _group(n: int, lines: int, strides) -> int:
    """The lines per threadgroup of a launch: one for contiguous lines, several
    for lines whose elements are apart but that are each other's neighbours
    (columns), as long as the threadgroup memory holds them."""
    bs, cs, js_in, js_out, es_in, es_out = strides
    # Lines of 1024 points or more fill the threadgroup memory of a pair, which
    # leaves one group per GPU core and was measured slower than single lines.
    if es_in == 1 or cs != 1 or n >= 1024:
        return 1
    for group in GROUP_LINES:
        if lines % group == 0 and 16 * n * group <= 32768:
            return group
    return 1


def _split_axis(n: int) -> tuple[int, int] | None:
    """The factors ``n = n1 * n2`` of a four-step transform of an axis too long
    for one threadgroup, as close to ``sqrt(n)`` as the divisors allow, or None
    when no split leaves both within :data:`MAX_FFT_SIZE`."""
    best = None
    for d in range(2, MAX_FFT_SIZE + 1):
        if n % d == 0 and 2 <= n // d <= MAX_FFT_SIZE:
            if best is None or abs(d - n**0.5) < abs(best - n**0.5):
                best = d
    return None if best is None else (best, n // best)


def check_transform(shape) -> None:
    """Raise a ``ValueError`` unless :func:`fft2` can transform images of the last
    two lengths of ``shape``, in arrays of the whole shape."""
    _check_axis(int(shape[-2]))
    _check_axis(int(shape[-1]))
    if int(np.prod(shape)) > _MAX_ELEMENTS:
        raise ValueError(f"an array of shape {tuple(shape)} is too large")


def _check_axis(n: int) -> None:
    if radices(n) is None or (n > MAX_FFT_SIZE and _split_axis(n) is None):
        raise ValueError(
            "emulated double precision on Metal transforms axes whose length "
            "has no prime factor above 7 and is at most "
            f"{MAX_FFT_SIZE * MAX_FFT_SIZE}, got {n}. Choose a grid of that "
            "size, or use the 'cpu' or 'gpu' device for other sizes."
        )


# The kernels address the four planes of an array with 32-bit unsigned
# integers, the last at 3 * plane + i: an array of 2**30 elements or more would
# wrap around silently.
_MAX_ELEMENTS = 2**30 - 1


def _check_size(x: DoubleFloatArray) -> None:
    if int(np.prod(x.shape)) > _MAX_ELEMENTS:
        raise ValueError(
            f"emulated double precision on Metal handles arrays of up to "
            f"{_MAX_ELEMENTS} elements, got shape {x.shape}. Use smaller chunks."
        )


@_on_metal
def from_numpy(x: np.ndarray) -> DoubleFloatArray:
    """Upload a float64 or complex128 array, keeping all of its precision."""
    x = np.asarray(x)
    if x.dtype not in (np.float64, np.complex128):
        x = x.astype(np.complex128)
    return _upload(x)


@_on_metal
def to_numpy(x: DoubleFloatArray) -> np.ndarray:
    """Download as a complex128 array."""
    planes = x.data.cpu().numpy()
    out = np.empty(planes.shape[1:], dtype=np.complex128)
    # Straight into the real and imaginary parts of the result, one pass each.
    np.add(planes[0], planes[1], out=out.real, dtype=np.float64)
    np.add(planes[2], planes[3], out=out.imag, dtype=np.float64)
    return out


@_on_metal
def multiply(x: DoubleFloatArray, kernel: DoubleFloatArray) -> DoubleFloatArray:
    """The product ``x * kernel``, with the (2D) kernel broadcast over the
    leading axes of ``x``."""
    nk = int(np.prod(kernel.shape))
    if x.shape[-len(kernel.shape) :] != kernel.shape:
        raise ValueError(
            f"kernel of shape {kernel.shape} does not broadcast over {x.shape}"
        )
    _check_size(x)
    y = x.empty_like()
    n = int(np.prod(x.shape))
    _multiply_library().multiply_df(x.data, kernel.data, y.data, n, nk, threads=n)
    return y


_NO_MAPS: dict = {}
_FACTOR_MAPS: dict = {}


def _empty_map():
    """An index table the kernel is told not to read."""
    if "map" not in _NO_MAPS:
        _NO_MAPS["map"] = torch.zeros(1, dtype=torch.int32, device="mps")
    return _NO_MAPS["map"]


def _factor_map(factor_batch, batch):
    """The image of a factor with leading shape ``factor_batch`` that goes with
    each image of a batch of shape ``batch``, as NumPy would broadcast them, as
    a device table of one index per image."""
    key = (tuple(factor_batch), tuple(batch))
    if key not in _FACTOR_MAPS:
        if len(factor_batch) > len(batch):
            extra = factor_batch[: len(factor_batch) - len(batch)]
            if any(n != 1 for n in extra):
                raise ValueError(
                    f"a factor with leading shape {factor_batch} does not "
                    f"broadcast over a batch of shape {batch}"
                )
            factor_batch = factor_batch[len(extra) :]
        padded = (1,) * (len(batch) - len(factor_batch)) + tuple(factor_batch)
        try:
            index = np.broadcast_to(
                np.arange(int(np.prod(padded))).reshape(padded), batch
            )
        except ValueError as error:
            raise ValueError(
                f"a factor with leading shape {factor_batch} does not broadcast "
                f"over a batch of shape {batch}"
            ) from error
        if len(_FACTOR_MAPS) > 64:
            _FACTOR_MAPS.clear()
        _FACTOR_MAPS[key] = torch.from_numpy(
            np.ascontiguousarray(index.reshape(-1), dtype=np.int32)
        ).to("mps")
    return _FACTOR_MAPS[key]


def _launch(
    n,
    src,
    dst,
    *,
    sign,
    scale,
    lines,
    strides,
    twiddle_length=None,
    pre=None,
    post=None,
    post_map=None,
):
    """One kernel launch: ``lines = (C, J)`` lines per image of length ``n`` with
    the strides ``(bs, cs, js_in, js_out, es_in, es_out)``, ``twiddle_length`` the
    length of the whole axis of a four-step pass, and ``pre`` and ``post`` the
    factors multiplied on at the loads and the stores, ``post`` with a ``post_map``
    from the batch of ``src`` to the images of ``post`` if it holds several."""
    batch = src.data.shape[1]
    scale_hi = float(np.float32(scale))
    scale_lo = float(np.float32(scale - scale_hi))
    tables = _twiddles(n)
    fused = _twiddles(twiddle_length) if twiddle_length else tables
    # A table the kernel is told not to read still has to be bound.
    first, last = pre or tables, post or tables
    image_post = int(np.prod(post.shape[-2:])) if post else 1
    c, j = lines
    group = _group(n, batch * c * j, strides) if j == 1 else 1
    _library(n, group).fft_df(
        src.data,
        dst.data,
        tables.data,
        fused.data,
        first.data,
        last.data,
        post_map if post_map is not None else _empty_map(),
        src.data[0].numel(),
        tables.data[0].numel(),
        fused.data[0].numel(),
        first.data[0].numel(),
        last.data[0].numel(),
        image_post,
        1 if twiddle_length else 0,
        1 if pre else 0,
        1 if post else 0,
        1 if post_map is not None else 0,
        c,
        j,
        *strides,
        sign,
        scale_hi,
        scale_lo,
        threads=batch * c * j // group * _width(n, group),
        group_size=_width(n, group),
    )


def _transform_axis(
    src,
    dst,
    length,
    element,
    lines,
    image,
    sign,
    scale,
    pre=None,
    post=None,
    post_map=None,
):
    """Transform ``lines`` lines of ``length`` points per image, each element of
    a line ``element`` apart and the lines laid out by ``lines = (count, stride)``.

    A length too long for one threadgroup is done in four steps (Bailey): with
    ``length = n1 * n2``, ``n1``-point transforms down the columns of the line
    seen as an ``n1`` by ``n2`` matrix, a twiddle multiply, then ``n2``-point
    transforms along its rows, written transposed. The twiddle multiply rides on
    the first pass, ``pre`` on the loads of the first and ``post`` on the stores
    of the last.
    """
    count, line_stride = lines
    split = _split_axis(length) if length > MAX_FFT_SIZE else None
    flat = lambda x: DoubleFloatArray(x.data.reshape(4, -1, image))  # noqa: E731
    src, dst = flat(src), flat(dst)

    if split is None:
        _launch(
            length,
            src,
            dst,
            sign=sign,
            scale=scale,
            lines=(count, 1),
            strides=(image, line_stride, 0, 0, element, element),
            pre=pre,
            post=post,
            post_map=post_map,
        )
        return

    n1, n2 = split
    middle = flat(src.empty_like())
    # Step 1: for each n2, the n1 points {n1' * n2 + n2'} ...
    _launch(
        n1,
        src,
        middle,
        sign=sign,
        scale=1.0,
        lines=(count, n2),
        strides=(image, line_stride, element, element, n2 * element, n2 * element),
        twiddle_length=length,
        pre=pre,
    )
    # Step 2: for each k1, the n2 points, to k1 + n1 * k2.
    _launch(
        n2,
        middle,
        dst,
        sign=sign,
        scale=scale,
        lines=(count, n1),
        strides=(image, line_stride, n2 * element, element, element, n1 * element),
        post=post,
        post_map=post_map,
    )


@_on_metal
def fft2(
    x: DoubleFloatArray,
    inverse: bool = False,
    pre: DoubleFloatArray | None = None,
    post: DoubleFloatArray | None = None,
) -> DoubleFloatArray:
    """The 2D FFT over the last two axes, normalised like ``numpy.fft``.

    ``pre`` and ``post`` are multiplied on in the same kernels, to the input before
    the transform and to the result after it: ``fft2(x, pre=a, post=b)`` is
    ``b * fft2(a * x)`` without the two products as passes of their own over the
    data. ``pre`` is an array of one image; ``post`` may have leading axes too,
    broadcast against those of ``x`` as NumPy does, so that each image of a batch
    can have a factor of its own (a Fresnel propagator for every tilt).

    The input is left untouched.
    """
    *batch_shape, n0, n1 = x.shape
    _check_axis(n0)
    _check_axis(n1)
    _check_size(x)
    warn_if_slow_emulated_fft((n0, n1))
    if pre is not None and pre.shape != (n0, n1):
        raise ValueError(
            f"a factor of shape {pre.shape} does not fit images of shape {(n0, n1)}"
        )
    post_map = None
    if post is not None:
        if post.shape[-2:] != (n0, n1):
            raise ValueError(
                f"a factor of shape {post.shape} does not fit images of "
                f"shape {(n0, n1)}"
            )
        if post.shape != (n0, n1):
            post_map = _factor_map(post.shape[:-2], tuple(batch_shape))

    sign = 1.0 if inverse else -1.0
    # 1 / (n0 n1) is not a float32 unless the grid is a power of two: scale in
    # double-float too, or the inverse would be good to float32 only.
    scale = 1.0 / (n0 * n1) if inverse else 1.0

    rows = x.empty_like()
    out = x.empty_like()
    image = n0 * n1
    # Along the last axis: n0 lines per image, contiguous.
    _transform_axis(x, rows, n1, 1, (n0, n1), image, sign, 1.0, pre=pre)
    # Along the first: n1 lines per image, strided by the row length.
    _transform_axis(
        rows, out, n0, n1, (n1, 1), image, sign, scale, post=post, post_map=post_map
    )
    return out
