import threading
import types

import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.core import config
from abtem.core import fft as abtem_fft
from abtem.core.backend import cp
from abtem.core.fft import fft_crop, fft_interpolate
from abtem.waves import Waves


class TestFftCropInterpolateEmptyNewShape:
    """`fft_crop`/`fft_interpolate` sliced off the batch dimensions with
    `array.shape[: -len(new_shape)]` / `array.shape[-len(new_shape) :]`, the
    same -0 bug as abtem/array.py's base-less ArrayObject sites: for
    `new_shape == ()` (every dimension is a batch dimension, none are being
    resized), `-len(new_shape)` is `-0`, which collapses to the wrong end.

    Reachable from public API: `WavesDetector(gpts=())` builds `new_shape =
    waves.shape[:-2] + gpts`, which is `()` for a plain 2D `Waves` with no
    ensemble axes, since `gpts` itself contributes nothing. Before the fix
    this crashed inside `fft_crop` with an opaque
    `TypeError: only length-1 arrays can be converted to Python scalars`,
    three frames below the `gpts=()` that caused it.
    """

    @staticmethod
    def _array():
        rng = np.random.default_rng(0)
        return (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(
            complex
        )

    def test_fft_crop_with_empty_new_shape_is_a_no_op(self):
        array = self._array()
        assert np.array_equal(fft_crop(array, ()), array)

    def test_fft_interpolate_with_empty_new_shape_is_a_no_op(self):
        array = self._array()
        out = fft_interpolate(array, ())
        assert out.shape == array.shape
        assert np.allclose(out, array)

    def test_wavesdetector_empty_gpts_matches_none(self):
        """The end-to-end case: `gpts=()` is falsy but `is not None`, so it
        reaches `_calculate_new_array`'s `if self._gpts is not None:` guard
        and used to crash there. It now degenerates to the same no-resample
        behaviour as the documented `gpts=None` default, rather than either
        crashing or silently returning something else."""
        with abtem.config.set({"fft": "numpy"}):
            array = self._array().astype("complex64")
            waves = Waves(array, energy=100e3, extent=(10, 10))

            none_out = abtem.WavesDetector(gpts=None).detect(waves)
            empty_out = abtem.WavesDetector(gpts=()).detect(waves)

        assert np.array_equal(np.asarray(none_out.array), np.asarray(empty_out.array))
        assert np.array_equal(np.asarray(empty_out.array), array)


pyfftw = abtem_fft.pyfftw
requires_pyfftw = pytest.mark.skipif(pyfftw is None, reason="needs pyfftw")


def _random_convolution_inputs(rng, batch=3, gpts=32, dtype=np.complex64):
    real = np.dtype(dtype).type(0).real.dtype
    array = (
        rng.random((batch, gpts, gpts), dtype=real)
        + 1j * rng.random((batch, gpts, gpts), dtype=real)
    ).astype(dtype)
    kernel = (
        rng.random((gpts, gpts), dtype=real) + 1j * rng.random((gpts, gpts), dtype=real)
    ).astype(dtype)
    return array, kernel


def _copy_at_byte_offset(array, offset):
    """A C-contiguous copy of ``array`` starting ``offset`` bytes past a 64-byte
    boundary. At ``offset=16`` it is aligned as malloc guarantees, but not for
    FFTW's SIMD codelets wherever ``pyfftw.simd_alignment`` is 32 (x86 AVX). On
    Apple Silicon ``simd_alignment`` is 4 and every buffer counts as aligned, so
    there the tests using this pass trivially; they bite on x86 (CI)."""
    raw = pyfftw.empty_aligned(array.nbytes + 64, dtype=np.uint8, n=64)
    copy = raw[offset : offset + array.nbytes].view(array.dtype).reshape(array.shape)
    copy[...] = array
    return copy


def _assert_misaligned_for_fftw(array):
    if pyfftw.simd_alignment > 16:
        assert not pyfftw.is_byte_aligned(array)


@requires_pyfftw
@pytest.mark.parametrize("overwrite_x", [True, False])
@pytest.mark.parametrize("func", ["fft2", "ifft2"])
def test_fftw_result_is_independent_of_buffer_alignment(func, overwrite_x):
    """The numbers must not depend on where the input buffer happens to land.

    Regression test: ``get_fftw_object`` planned for the buffer it was given, and
    pyfftw makes an ``FFTW_UNALIGNED`` plan, which rounds differently (~1e-7
    relative), for a buffer that is only 16-byte aligned. Two builds of the same
    frozen-phonon configuration then differed in the last bit depending on where
    malloc had placed a buffer, failing CI intermittently on x86.
    """
    rng = np.random.default_rng(7)
    array, _ = _random_convolution_inputs(rng, gpts=64)
    aligned = _copy_at_byte_offset(array, 0)
    offset = _copy_at_byte_offset(array, 16)
    _assert_misaligned_for_fftw(offset)

    transform = getattr(abtem_fft, func)
    with config.set({"fft": "fftw"}):
        expected = transform(aligned, overwrite_x=overwrite_x)
        result = transform(offset, overwrite_x=overwrite_x)

    assert np.array_equal(
        result, expected
    ), f"FFTW {func} result depends on input buffer alignment"
    reference = getattr(np.fft, func)(array.astype(np.complex128))
    assert np.allclose(result, reference, atol=1e-4)
    if not overwrite_x:
        assert np.array_equal(offset, array)


@requires_pyfftw
@pytest.mark.parametrize("name", ["fft2", "ifft2"])
def test_fftw_plans_for_an_unaligned_buffer_as_for_an_aligned_one(name):
    """An unaligned buffer is aligned by copying, never given its own plan."""
    rng = np.random.default_rng(8)
    array, _ = _random_convolution_inputs(rng, gpts=48)
    offset = _copy_at_byte_offset(array, 16)
    _assert_misaligned_for_fftw(offset)

    aligned_plan = abtem_fft.get_fftw_object(_copy_at_byte_offset(array, 0), name)
    offset_plan = abtem_fft.get_fftw_object(offset, name)

    assert offset_plan.flags == aligned_plan.flags
    assert offset_plan.simd_aligned == aligned_plan.simd_aligned


@requires_pyfftw
@pytest.mark.parametrize("overwrite_x", [True, False])
@pytest.mark.parametrize("gpts", [4, 6, 8, 12])
def test_cached_fftw_convolution_plans_for_an_aligned_buffer(gpts, overwrite_x):
    """The scratch array a plan is made with must be aligned too.

    Regression test: it came from ``np.zeros_like``, so whenever malloc placed it
    16- rather than 32-byte aligned (likely for the small arrays here) the
    plans were ``FFTW_UNALIGNED``, and every convolution in that thread rounded
    differently from a thread whose scratch array had landed aligned.
    """
    rng = np.random.default_rng(9)
    array, kernel = _random_convolution_inputs(rng, batch=1, gpts=gpts)
    reference = pyfftw.FFTW(
        *(2 * (_copy_at_byte_offset(array, 0),)),
        axes=(-2, -1),
        flags=(config.get("fftw.planning_effort"),),
    )

    convolution = abtem_fft.CachedFFTWConvolution()
    result = convolution(_copy_at_byte_offset(array, 16), kernel, overwrite_x)

    for plan in convolution._fftw_objects.values():
        assert plan.simd_aligned == reference.simd_aligned
    expected = np.fft.ifft2(np.fft.fft2(array.astype(np.complex128)) * kernel)
    assert np.allclose(result, expected, atol=1e-4)


@requires_pyfftw
@pytest.mark.parametrize("func", ["fft2", "ifft2"])
def test_fftw_transforms_a_non_contiguous_array_in_place(func):
    # A strided view is copied to the aligned C-contiguous layout the plans are
    # made for, so the caller's array is left alone and the result is returned.
    rng = np.random.default_rng(10)
    array, _ = _random_convolution_inputs(rng, gpts=64)
    view = array[:, ::2, :].transpose(0, 2, 1)
    before = view.copy()

    with config.set({"fft": "fftw"}):
        result = getattr(abtem_fft, func)(view, overwrite_x=True)

    reference = getattr(np.fft, func)(before.astype(np.complex128))
    assert np.allclose(result, reference, atol=1e-4)
    assert np.array_equal(view, before)


@requires_pyfftw
def test_fftw_fftn_over_three_axes():
    """Wisdom must be made for the axes being transformed.

    Regression test: it was always made for the last two axes, so an ``fftn``
    over three never found its plan and fell back to ``pyfftw.builders``, which
    was not passed the axes either and transformed the batch axis too.
    """
    rng = np.random.default_rng(11)
    array = (rng.random((2, 6, 8, 10)) + 1j * rng.random((2, 6, 8, 10))).astype(
        np.complex64
    )
    axes = (-3, -2, -1)

    with config.set({"fft": "fftw"}):
        result = abtem_fft.fftn(array, axes=axes)
        plan = abtem_fft.get_fftw_object(array, "fftn", axes=axes)

    assert isinstance(plan, pyfftw.FFTW)
    assert np.allclose(result, np.fft.fftn(array, axes=axes), atol=1e-4)


@requires_pyfftw
@pytest.mark.parametrize("allow_fallback", [True, False])
def test_fftw_without_wisdom_falls_back_or_raises(monkeypatch, allow_fallback):
    # With no wisdom for a plan and none allowed to be made, `get_fftw_object`
    # builds a plan with `pyfftw.builders` if fallback is allowed, and else
    # raises. The fallback must transform the requested axes.
    pyfftw.forget_wisdom()
    monkeypatch.setattr(abtem_fft, "_new_fftw_object", lambda *a, **k: None)

    rng = np.random.default_rng(12)
    array = (rng.random((2, 6, 8, 10)) + 1j * rng.random((2, 6, 8, 10))).astype(
        np.complex64
    )
    offset = _copy_at_byte_offset(array, 16)
    axes = (-3, -2, -1)

    with config.set({"fftw.allow_fallback": allow_fallback}):
        if not allow_fallback:
            with pytest.raises(RuntimeError, match="No FFTW wisdom"):
                abtem_fft.get_fftw_object(offset, "fftn", axes=axes)
            return
        result = abtem_fft.get_fftw_object(offset, "fftn", axes=axes)()

    assert np.allclose(result, np.fft.fftn(array, axes=axes), atol=1e-4)


_LAZY_CONVOLVE_CASES = [
    # the kernel spans the single block of x along every axis
    ((3, 16, 20), (3, 16, 20), (3, 16, 20)),
    ((3, 16, 20), (1, 16, 20), (16, 20)),
    ((3, 16, 20), (1, 16, 20), (1, 16, 20)),
    # the kernel spans several blocks of x
    ((3, 16, 20), (1, 16, 20), (3, 16, 20)),
    ((3, 16, 20), (2, 16, 20), (3, 1, 20)),
    ((2, 3, 16, 20), (1, 2, 16, 20), (3, 16, 20)),
    ((2, 3, 16, 20), (1, 3, 16, 20), (2, 1, 16, 20)),
    # x broadcasts against a longer kernel axis
    ((1, 16, 20), (1, 16, 20), (3, 16, 20)),
]


def _random_complex(rng, shape, dtype=np.complex64):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("shape, chunks, kernel_shape", _LAZY_CONVOLVE_CASES)
def test_lazy_fft2_convolve_matches_eager(shape, chunks, kernel_shape, device):
    """A lazy `fft2_convolve` gives the eager result, values and shape, also when
    the kernel has an axis longer than a block of x. Each block of x used to be
    convolved with the whole kernel, so it took the kernel's length along that
    axis, and the computed array was longer than the declared one."""
    import dask.array as da

    from abtem.core.backend import asnumpy, copy_to_device
    from abtem.core.fft import fft2_convolve

    rng = np.random.default_rng(0)
    x = copy_to_device(_random_complex(rng, shape), device)
    kernel = copy_to_device(_random_complex(rng, kernel_shape), device)

    # On the CPU the NumPy FFT, so that the blocks and the whole array are
    # transformed alike whatever their alignment in memory, and the results are
    # compared exactly; a device FFT may differ in the last bits by batch size.
    with config.set({"fft": "numpy"}):
        expected = asnumpy(fft2_convolve(x.copy(), kernel))
        lazy = fft2_convolve(da.from_array(x, chunks=chunks), kernel)
        assert lazy.shape == expected.shape
        result = asnumpy(lazy.compute(scheduler="synchronous"))

    assert result.shape == expected.shape
    if device == "cpu":
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("shape, chunks, kernel_shape", _LAZY_CONVOLVE_CASES)
def test_lazy_fft2_convolve_declares_the_dtype_it_computes(
    shape, chunks, kernel_shape, device
):
    """With a complex128 kernel and complex64 x, the product is taken in place,
    keeping complex64, unless x is the side that broadcasts; then it is promoted
    to complex128, as eagerly. The lazy result declares what its blocks give."""
    import dask.array as da

    from abtem.core.backend import copy_to_device
    from abtem.core.fft import fft2_convolve

    rng = np.random.default_rng(0)
    x = copy_to_device(_random_complex(rng, shape), device)
    kernel = copy_to_device(_random_complex(rng, kernel_shape, np.complex128), device)

    lazy = fft2_convolve(da.from_array(x, chunks=chunks), kernel)
    computed = lazy.compute(scheduler="synchronous")

    assert lazy.dtype == computed.dtype == fft2_convolve(x.copy(), kernel).dtype


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("kernel_shape", [(16, 20), (1, 16, 20), (3, 16, 20)])
def test_lazy_fft2_convolve_of_unknown_chunk_sizes(kernel_shape, device):
    """An x whose leading axis has unknown chunk sizes (NaN, as after filtering)
    is convolved block by block, with the kernel whole where its blocks cannot be
    matched to those of x."""
    import dask.array as da

    from abtem.core.backend import asnumpy, copy_to_device
    from abtem.core.fft import fft2_convolve

    rng = np.random.default_rng(0)
    array = _random_complex(rng, (3, 16, 20))
    kernel = _random_complex(rng, kernel_shape)
    if len(kernel_shape) == 3 and kernel_shape[0] == 3:
        # A kernel axis longer than 1 against the unknown axis must fit every
        # block, so x is in a single block along it.
        chunks = (3, 16, 20)
    else:
        chunks = (1, 16, 20)
    known = copy_to_device(da.from_array(array, chunks=chunks), device)
    x = da.Array(
        known.dask,
        known.name,
        chunks=((np.nan,) * len(known.chunks[0]),) + known.chunks[1:],
        meta=known._meta,
    )
    assert np.isnan(x.shape[0])
    kernel = copy_to_device(kernel, device)

    with config.set({"fft": "numpy"}):
        expected = asnumpy(fft2_convolve(copy_to_device(array, device), kernel))
        result = asnumpy(fft2_convolve(x, kernel).compute(scheduler="synchronous"))

    np.testing.assert_allclose(
        result, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


class _FakePlanCache:
    def __init__(self):
        self.size = 16
        self.memsize = -1

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def get_memsize(self):
        return self.memsize

    def set_memsize(self, memsize):
        self.memsize = memsize


class _FakeCupy:
    """As much of CuPy as the plan-cache configuration touches. Like CuPy's, the
    plan cache is per thread."""

    TOTAL_MEMORY = 8 * 10**9

    class ndarray:
        shape = (8, 8)

    def __init__(self):
        self._local = threading.local()
        self.calls = []
        self.cuda = types.SimpleNamespace(Device=self._device)
        self.fft = types.SimpleNamespace(
            config=types.SimpleNamespace(get_plan_cache=self._plan_cache),
            fft2=self._fft2,
        )

    def _device(self):
        return types.SimpleNamespace(
            id=0, mem_info=(self.TOTAL_MEMORY // 2, self.TOTAL_MEMORY)
        )

    def _plan_cache(self):
        if not hasattr(self._local, "cache"):
            self._local.cache = _FakePlanCache()
        return self._local.cache

    def _fft2(self, x, **kwargs):
        # what the cache looked like when the transform ran
        self.calls.append((threading.current_thread().name, self._plan_cache().memsize))
        return x


@pytest.fixture
def fake_cupy(monkeypatch):
    fake = _FakeCupy()
    monkeypatch.setattr(abtem_fft, "cp", fake)
    monkeypatch.setattr(abtem_fft, "get_array_module", lambda x: np)
    monkeypatch.setattr(abtem_fft, "check_cupy_is_installed", lambda: None)
    monkeypatch.setattr(abtem_fft, "_warned_plan_cache_bypass", False)
    abtem_fft._reset_cufft_cache_state()
    yield fake
    abtem_fft._reset_cufft_cache_state()


def _in_a_thread(function):
    outcome = {}

    def run():
        try:
            outcome["value"] = function()
        except BaseException as error:  # reported in the calling thread
            outcome["error"] = error

    thread = threading.Thread(target=run, name="worker")
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def test_cufft_cache_is_bounded_by_default():
    assert config.get("cupy.fft-cache-size") == "auto"


def test_cufft_cache_auto_is_a_quarter_of_the_device_memory(fake_cupy):
    abtem_fft._configure_cufft_cache()

    cache = fake_cupy.fft.config.get_plan_cache()
    assert cache.get_memsize() == _FakeCupy.TOTAL_MEMORY // 4
    assert cache.get_size() == 16  # the entry count is CuPy's own


def test_cufft_cache_is_bounded_in_every_thread(fake_cupy):
    # CuPy keeps one plan cache per thread, and the transforms of a lazy
    # computation run in dask's worker threads, not in the thread that
    # configured anything.
    abtem_fft._configure_cufft_cache()
    in_thread = _in_a_thread(
        lambda: (
            abtem_fft._configure_cufft_cache(),
            fake_cupy.fft.config.get_plan_cache().get_memsize(),
        )[1]
    )

    assert in_thread == _FakeCupy.TOTAL_MEMORY // 4


@pytest.mark.parametrize(
    "value, size, memsize",
    [
        ("1 GB", 16, 10**9),
        (2 * 10**9, 16, 2 * 10**9),
        ("0 MB", 0, -1),
        (-1, 16, -1),
        (None, 16, -1),
    ],
)
def test_cufft_cache_config_values(fake_cupy, value, size, memsize):
    with config.set({"cupy.fft-cache-size": value}):
        abtem_fft._configure_cufft_cache()

    cache = fake_cupy.fft.config.get_plan_cache()
    assert (cache.get_size(), cache.get_memsize()) == (size, memsize)


def test_cufft_cache_follows_the_config_at_runtime(fake_cupy):
    cache = fake_cupy.fft.config.get_plan_cache()

    with config.set({"cupy.fft-cache-size": "0 MB"}):
        abtem_fft._configure_cufft_cache()
    assert cache.get_size() == 0

    # a bound re-enables a cache that was disabled
    with config.set({"cupy.fft-cache-size": "1 GB"}):
        abtem_fft._configure_cufft_cache()
    assert (cache.get_size(), cache.get_memsize()) == (16, 10**9)

    # and "unlimited" undoes a bound
    with config.set({"cupy.fft-cache-size": -1}):
        abtem_fft._configure_cufft_cache()
    assert cache.get_memsize() == -1


def test_cufft_cache_is_left_alone_while_the_config_is_unchanged(fake_cupy):
    abtem_fft._configure_cufft_cache()
    cache = fake_cupy.fft.config.get_plan_cache()
    cache.set_memsize(123)  # tuned through CuPy's own API after the first call

    abtem_fft._configure_cufft_cache()

    assert cache.get_memsize() == 123


def test_cupy_fft_dispatch_bounds_the_cache_in_the_thread_that_transforms(fake_cupy):
    x = fake_cupy.ndarray()
    bound = _FakeCupy.TOTAL_MEMORY // 4

    abtem_fft.fft2(x)
    _in_a_thread(lambda: abtem_fft.fft2(x))

    assert fake_cupy.calls == [("MainThread", bound), ("worker", bound)]


def test_oversized_plan_bypasses_the_cache(fake_cupy):
    cache = fake_cupy.fft.config.get_plan_cache()
    cache.set_memsize(10**9)
    calls = []

    def fft(x, **kwargs):
        calls.append(cache.get_size())
        if len(calls) == 1:
            raise RuntimeError("The plan memsize is too large.")
        return x

    x = fake_cupy.ndarray()
    with pytest.warns(UserWarning, match="uncached"):
        out = abtem_fft._cupy_fft_with_cache_fallback(fft, x)

    assert out is x
    assert calls == [16, 0]  # the retry ran with the cache disabled
    assert (cache.get_size(), cache.get_memsize()) == (16, 10**9)  # and it was restored


def test_unrelated_runtime_error_propagates(fake_cupy):
    def fft(x, **kwargs):
        raise RuntimeError("something else entirely")

    with pytest.raises(RuntimeError, match="something else"):
        abtem_fft._cupy_fft_with_cache_fallback(fft, fake_cupy.ndarray())


@pytest.mark.skipif(cp is None, reason="no gpu")
@pytest.mark.parametrize("gpts", [(160, 147), (128, 128), (256, 256)])
def test_cupy_fft_in_a_worker_thread_after_one_in_the_main_thread(gpts):
    # With CuPy's default plan cache, cuFFT (CuPy 14.2) fails with
    # CUFFT_EXEC_FAILED for the transform in the worker thread, which is what
    # dask runs a lazy computation in.
    rng = np.random.default_rng(0)
    array = (rng.standard_normal(gpts) + 1j * rng.standard_normal(gpts)).astype(
        np.complex64
    )
    x = cp.asarray(array)

    abtem_fft.fft2(x)
    cp.cuda.Stream.null.synchronize()

    def transform():
        y = abtem_fft.fft2(x)
        cp.cuda.Stream.null.synchronize()
        return y

    assert np.allclose(_in_a_thread(transform).get(), np.fft.fft2(array), atol=1e-2)
