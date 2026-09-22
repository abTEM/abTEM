"""Tests for FFT helpers, in particular fast-radix transform-size handling."""

import threading
import warnings

import numpy as np
import pytest

from utils import requires_gpu

from abtem.core import config
from abtem.core import fft as abtem_fft
from abtem.core.fft import (
    _warn_slow_fft_size,
    _warned_slow_fft_shapes,
    is_fast_fft_size,
    next_fast_fft_size,
)


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, True),
        (2048, True),  # 2^11
        (2625, True),  # 3 * 5^3 * 7
        (2688, True),  # 2^7 * 3 * 7
        (2304, True),  # 2^8 * 3^2
        (2623, False),  # 43 * 61 -> Bluestein
        (2271, False),  # 3 * 757 -> Bluestein
        (11, False),
        (0, False),
    ],
)
def test_is_fast_fft_size(n, expected):
    assert is_fast_fft_size(n) is expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, 1),
        (2048, 2048),  # already fast
        (2623, 2625),  # 3 * 5^3 * 7
        (2271, 2304),  # 2^8 * 3^2
    ],
)
def test_next_fast_fft_size(n, expected):
    assert next_fast_fft_size(n) == expected


def test_warn_slow_fft_size_warns_once_per_shape():
    _warned_slow_fft_shapes.clear()
    with pytest.warns(UserWarning, match="Bluestein"):
        _warn_slow_fft_size((4, 2623, 2271))
    # memoized: the same shape does not warn again
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_slow_fft_size((4, 2623, 2271))


def test_warn_slow_fft_size_silent_for_fast_shapes():
    # Large enough to clear the size gate, so it is the fastness check that
    # has to keep this silent: 2048 = 2**11, 2100 = 2**2 * 3 * 5**2 * 7.
    _warned_slow_fft_shapes.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_slow_fft_size((2048, 2048))
        _warn_slow_fft_size((2100, 2100))


def test_warn_slow_fft_size_memoizes_only_warned_shapes():
    # The memo exists to warn once per shape, so it must not accumulate an
    # entry for every distinct shape the GPU ever transforms -- and a run on a
    # fast grid must not take the lock on the FFT hot path at all.
    _warned_slow_fft_shapes.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2048, 2100, 2160, 2187, 2240):  # all fast, all large
            _warn_slow_fft_size((n, n))
        assert not _warned_slow_fft_shapes

        _warn_slow_fft_size((1031, 1033))  # slow and large: recorded
        assert _warned_slow_fft_shapes == {(1031, 1033)}


def test_warn_slow_fft_size_uses_the_transformed_axes():
    # fftn over all axes of a 3D structure-factor grid (Bloch waves) must
    # report every transformed length, not just the trailing two; and the
    # "set gpts instead of the sampling" remedy does not apply to that grid,
    # which follows from g_max and the cell.
    _warned_slow_fft_shapes.clear()
    with pytest.warns(UserWarning, match=r"FFT size 133 x 133 x 523") as record:
        _warn_slow_fft_size((133, 133, 523), "fftn", {})
    assert "gpts" not in str(record[0].message)

    # A leading slow axis is invisible when only shape[-2:] is inspected.
    _warned_slow_fft_shapes.clear()
    with pytest.warns(UserWarning, match=r"FFT size 1031 x 2048 x 2048"):
        _warn_slow_fft_size((1031, 2048, 2048), "fftn", {})

    # 2D transforms keep the gpts remedy.
    _warned_slow_fft_shapes.clear()
    with pytest.warns(UserWarning, match="setting gpts explicitly"):
        _warn_slow_fft_size((4, 2623, 2271), "fft2", {})


def test_transform_lengths_ignores_batch_axes_and_tolerates_none_in_s():
    from abtem.core.fft import _transform_lengths

    assert _transform_lengths((32, 652, 652), "fft2", {}) == (652, 652)
    assert _transform_lengths((133, 133, 523), "fftn", {}) == (133, 133, 523)
    assert _transform_lengths((4, 2623, 2271), "fftn", {"axes": (0, 1)}) == (4, 2623)
    assert _transform_lengths((8, 64), "fft", {}) == (64,)
    # `s` overrides the array lengths; None entries keep theirs. A raise here
    # would break the transform itself, not merely skip a diagnostic.
    assert _transform_lengths((4, 64, 64), "fftn", {"s": (None, 32, 32)}) == (4, 32, 32)


def test_warn_slow_fft_size_silent_for_small_shapes():
    """Small transforms (e.g. interpolated measurements) never warn, even when
    their lengths would force the Bluestein path."""
    _warned_slow_fft_shapes.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_slow_fft_size((36, 29))
        _warn_slow_fft_size((4, 11, 20))


def test_fast_fft_sizes_stricter_than_scipy():
    # scipy.fft.next_fast_len targets pocketfft (radix kernels up to 11) and
    # returns 11-smooth lengths; cuFFT's documented fast path is 7-smooth.
    # These helpers must stay 7-smooth or GPU grids regress to Bluestein.
    import scipy.fft

    assert scipy.fft.next_fast_len(121, real=False) == 121  # 11**2: fine for scipy
    assert not is_fast_fft_size(121)
    assert next_fast_fft_size(121) == 125  # 5**3

    assert scipy.fft.next_fast_len(4619, real=False) == 4620  # contains 11
    assert next_fast_fft_size(4619) == 4704  # 2**5 * 3 * 7**2

    # Where the sets agree, results agree (the PR's own headline case).
    assert scipy.fft.next_fast_len(2623, real=False) == next_fast_fft_size(2623) == 2625


def test_fast_fft_sizes_agree_with_cupy():
    # CuPy made the same call for the same reason: cupyx.scipy.fft.next_fast_len
    # is 7-smooth, and its docstring notes it deliberately differs from scipy
    # ("pocketfft's prime factors are different from cuFFT's"). Pin the
    # agreement so drift on either side fails loudly.
    cupyx_fft = pytest.importorskip("cupyx.scipy.fft")

    for n in (2, 11, 13, 121, 169, 335, 2623, 4619, 10007):
        assert cupyx_fft.next_fast_len(n) == next_fast_fft_size(n)


def test_warn_slow_fft_size_thread_safe():
    # _fft_dispatch runs concurrently in dask worker threads; the warn-once
    # bookkeeping must not double-warn when threads race on a new shape.
    import threading
    import time
    from unittest import mock

    from abtem.core import fft as abtem_fft

    shape = (1031, 1033)  # both prime, > 1024*1024 elements

    class SlowMembershipSet(set):
        """Widens the check-then-add window so an unlocked race is certain."""

        def __contains__(self, item):
            # Sleep AFTER reading membership, so every racing thread observes
            # the same "not present" answer and an unlocked check-then-add
            # really does double-warn.
            present = super().__contains__(item)
            time.sleep(0.01)
            return present

    calls = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        _warn_slow_fft_size(shape)

    with mock.patch.object(
        abtem_fft, "_warned_slow_fft_shapes", SlowMembershipSet()
    ), mock.patch.object(
        abtem_fft.warnings, "warn", side_effect=lambda *a, **k: calls.append(a)
    ):
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(calls) == 1


def test_cufft_cache_auto_resolves_device_relative():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no GPU available")
    except Exception:
        pytest.skip("no usable CUDA/HIP runtime")

    from abtem.core import config
    from abtem.core import fft as abtem_fft

    expected = cp.cuda.Device().mem_info[1] // 4

    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": "auto"}):
        abtem_fft._configure_cufft_cache()
        assert getattr(abtem_fft._CUFFT_CACHE_STATE, "token", None) is not None
        assert abtem_fft._CUFFT_CACHE_STATE.limit == expected
        assert cp.fft.config.get_plan_cache().get_memsize() == expected

    # -1 must still mean unlimited (no memsize bound applied).
    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": -1}):
        abtem_fft._configure_cufft_cache()
        assert abtem_fft._CUFFT_CACHE_STATE.limit == -1


@requires_gpu
def test_oversized_plan_bypasses_cache():
    cp = pytest.importorskip("cupy")

    from abtem.core import fft as abtem_fft

    calls = []

    def fake_fft(x, **kwargs):
        calls.append(cp.fft.config.get_plan_cache().get_size())
        if len(calls) == 1:
            raise RuntimeError("The plan memsize is too large.")
        return x

    abtem_fft._warned_plan_cache_bypass = False
    x = cp.zeros((2, 8, 8), dtype="complex64")
    with pytest.warns(UserWarning, match="uncached"):
        out = abtem_fft._cupy_fft_with_cache_fallback(fake_fft, x)

    assert out is x
    assert len(calls) == 2
    assert calls[1] == 0  # the retry ran with the cache disabled
    assert cp.fft.config.get_plan_cache().get_size() > 0  # and it was restored


def test_unrelated_runtime_error_propagates():
    pytest.importorskip("cupy")

    from abtem.core import fft as abtem_fft

    def fake_fft(x, **kwargs):
        raise RuntimeError("something else entirely")

    with pytest.raises(RuntimeError, match="something else"):
        abtem_fft._cupy_fft_with_cache_fallback(fake_fft, object())


def test_cufft_cache_config_edge_values():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no GPU available")
    except Exception:
        pytest.skip("no usable CUDA/HIP runtime")

    from abtem.core import config
    from abtem.core import fft as abtem_fft

    cache = cp.fft.config.get_plan_cache()

    # -1 must undo an earlier bound (the oversized-plan warning recommends it).
    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": "auto"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_memsize() > 0
    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": -1}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_memsize() == -1

    # null means "no bound" rather than crashing.
    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": None}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_memsize() == -1

    # A positive bound re-enables a previously disabled cache.
    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": "0 MB"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() == 0
    abtem_fft._reset_cufft_cache_state()
    with config.set({"cupy.fft-cache-size": "1 GB"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() > 0
        assert cache.get_memsize() == 10**9

    # Restore the shipped default for subsequent tests.
    abtem_fft._reset_cufft_cache_state()
    abtem_fft._configure_cufft_cache()


def test_plan_cache_entry_limit():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no GPU available")
    except Exception:
        pytest.skip("no usable CUDA/HIP runtime")

    from abtem.core import config
    from abtem.core import fft as abtem_fft

    cache = cp.fft.config.get_plan_cache()

    try:
        # CuPy's own default of 16 thrashes for varying batch shapes.
        cache.set_size(16)
        abtem_fft._reset_cufft_cache_state()
        with config.set({"cupy.fft-cache-entries": 64}):
            abtem_fft._configure_cufft_cache()
            assert cache.get_size() == 64

        # Raised, never lowered: an externally tuned larger cache is kept.
        cache.set_size(256)
        abtem_fft._reset_cufft_cache_state()
        with config.set({"cupy.fft-cache-entries": 64}):
            abtem_fft._configure_cufft_cache()
            assert cache.get_size() == 256

        # Invalid or opt-out values leave the entry count alone; they must
        # never raise, as this runs ahead of every GPU FFT dispatch.
        for bad in (None, -1, "not-a-number"):
            cache.set_size(32)
            abtem_fft._reset_cufft_cache_state()
            with config.set({"cupy.fft-cache-entries": bad}):
                abtem_fft._configure_cufft_cache()
                assert cache.get_size() == 32, f"entries={bad!r}"

        # Disabling the cache still wins over the entry count.
        abtem_fft._reset_cufft_cache_state()
        with config.set(
            {"cupy.fft-cache-size": "0 MB", "cupy.fft-cache-entries": 64}
        ):
            abtem_fft._configure_cufft_cache()
            assert cache.get_size() == 0
    finally:
        # Reapply the process configuration whatever happened above.
        abtem_fft._reset_cufft_cache_state()
        abtem_fft._configure_cufft_cache()


pyfftw = abtem_fft.pyfftw
requires_pyfftw = pytest.mark.skipif(
    pyfftw is None, reason="CachedFFTWConvolution needs pyfftw"
)


def _convolution_reference(array, kernel):
    """The convolution ``CachedFFTWConvolution`` computes, via numpy."""
    return np.fft.ifft2(np.fft.fft2(array.astype(np.complex128)) * kernel)


def _random_convolution_inputs(rng, batch=3, gpts=32, dtype=np.complex64):
    real = np.dtype(dtype).type(0).real.dtype
    array = (
        rng.random((batch, gpts, gpts), dtype=real)
        + 1j * rng.random((batch, gpts, gpts), dtype=real)
    ).astype(dtype)
    kernel = (
        rng.random((gpts, gpts), dtype=real)
        + 1j * rng.random((gpts, gpts), dtype=real)
    ).astype(dtype)
    return array, kernel


@pytest.fixture
def count_fftw_plans(monkeypatch):
    """Count the pyfftw plans built while the fixture is active."""
    built = []
    original = abtem_fft._new_fftw_object

    def counted(array, name, flags=()):
        built.append(name)
        return original(array, name, flags=flags)

    monkeypatch.setattr(abtem_fft, "_new_fftw_object", counted)
    return built


@requires_pyfftw
def test_cached_fftw_convolution_reuses_plans(count_fftw_plans):
    # The plan pair must be built once and then reused. Regression test: the
    # shape was compared against an attribute that was never assigned, so every
    # call rebuilt both plans (and zeroed a scratch array the size of the input).
    rng = np.random.default_rng(0)
    convolution = abtem_fft.CachedFFTWConvolution()

    for _ in range(4):
        array, kernel = _random_convolution_inputs(rng)
        convolution(array.copy(), kernel, True)

    assert len(count_fftw_plans) == 2


@pytest.mark.parametrize("overwrite_x", [True, False])
@requires_pyfftw
def test_cached_fftw_convolution_correct_on_a_cache_hit(overwrite_x):
    # A cached plan still points at the previous call's buffer, so a hit is only
    # correct if the plans are re-pointed at the current array every call.
    #
    # Note: this also passes unmodified against the pre-fix code, since
    # "rebuild from scratch on every call" trivially satisfies "plan matches
    # the current buffer" -- there is no cache there to get wrong. It still
    # guards a real invariant of the fixed implementation.
    rng = np.random.default_rng(1)
    convolution = abtem_fft.CachedFFTWConvolution()

    for _ in range(4):
        array, kernel = _random_convolution_inputs(rng)
        source = array.copy()

        result = convolution(source, kernel, overwrite_x)

        expected = _convolution_reference(array, kernel)
        assert np.allclose(result, expected, atol=1e-6)
        if not overwrite_x:
            assert np.array_equal(source, array)


@pytest.mark.parametrize("changed", ["shape", "dtype"])
@requires_pyfftw
def test_cached_fftw_convolution_replans_on_layout_change(changed, count_fftw_plans):
    # A plan is tied to the dtype, shape and strides it was made for --
    # ``update_arrays`` raises otherwise -- so each must invalidate the cache.
    #
    # Note: this also passes unmodified against the pre-fix code, since a
    # shape/dtype change there triggers a rebuild anyway (every call does).
    # It still guards a real invariant of the fixed implementation.
    rng = np.random.default_rng(2)
    convolution = abtem_fft.CachedFFTWConvolution()

    array, kernel = _random_convolution_inputs(rng)
    convolution(array.copy(), kernel, True)
    assert len(count_fftw_plans) == 2

    if changed == "shape":
        array, kernel = _random_convolution_inputs(rng, gpts=64)
    else:
        array, kernel = _random_convolution_inputs(rng, dtype=np.complex128)

    result = convolution(array.copy(), kernel, True)

    assert len(count_fftw_plans) == 4
    assert np.allclose(result, _convolution_reference(array, kernel), atol=1e-6)


@requires_pyfftw
def test_cached_fftw_convolution_is_thread_safe():
    # A plan points at exactly one buffer, so threads sharing a plan pair would
    # transform each other's arrays. Each thread must get its own pair.
    convolution = abtem_fft.CachedFFTWConvolution()
    failures = []

    def worker(seed):
        rng = np.random.default_rng(seed)
        array, kernel = _random_convolution_inputs(rng, gpts=64)
        expected = _convolution_reference(array, kernel)
        for _ in range(25):
            result = convolution(array.copy(), kernel, True)
            if not np.allclose(result, expected, atol=1e-6):
                failures.append(seed)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures


@requires_pyfftw
def test_cached_fftw_convolution_result_is_independent_of_buffer_alignment():
    """The numbers must not depend on where the input buffer happens to land.

    Regression test: keying the plan flags off the input's alignment made the
    convolution pick ``FFTW_UNALIGNED`` codelets for some buffers and aligned
    ones for others. Those differ by ~1e-7 relative, so two runs that differed
    only in how arrays had been allocated -- e.g. the same multislice with a
    different potential chunk size -- stopped agreeing.
    """
    rng = np.random.default_rng(3)
    array, kernel = _random_convolution_inputs(rng, gpts=64)

    # a buffer deliberately offset inside a larger allocation: still contiguous
    # and still C-ordered, but not necessarily aligned for SIMD
    spare = np.empty(array.size + 1, dtype=array.dtype)
    offset = spare[1:].reshape(array.shape)
    offset[...] = array
    assert offset.strides == array.strides

    convolution = abtem_fft.CachedFFTWConvolution()
    aligned_result = convolution(array.copy(), kernel, True)
    offset_result = convolution(offset, kernel, True)

    assert np.array_equal(aligned_result, offset_result), (
        "convolution result depends on input buffer alignment"
    )


@requires_pyfftw
@pytest.mark.parametrize(
    "setting",
    [
        {"fftw.threads": 4},
        {"fftw.planning_effort": "FFTW_ESTIMATE"},
        {"fftw.planning_timelimit": 5},
    ],
    ids=["threads", "planning_effort", "planning_timelimit"],
)
def test_cached_fftw_convolution_respects_config_changes(setting, count_fftw_plans):
    """Caching must not pin the plans to the configuration of the first call.

    Every call re-read these settings before the plans were cached, so a
    ``config.set`` block took effect immediately. A plan built under the old
    configuration must not outlive it -- a user asking for more threads would
    otherwise keep getting plans made for the old count.
    """
    rng = np.random.default_rng(4)
    array, kernel = _random_convolution_inputs(rng)
    convolution = abtem_fft.CachedFFTWConvolution()

    convolution(array.copy(), kernel, True)
    assert len(count_fftw_plans) == 2

    with config.set(setting):
        convolution(array.copy(), kernel, True)
        assert len(count_fftw_plans) == 4, f"{setting} ignored by the plan cache"
        # ...and the plans built under it are themselves reused
        convolution(array.copy(), kernel, True)
        assert len(count_fftw_plans) == 4

    # leaving the block restores the original configuration's plans
    convolution(array.copy(), kernel, True)
    assert len(count_fftw_plans) == 6


@requires_pyfftw
def test_cached_fftw_convolution_is_picklable_before_first_use():
    # threading.local is not picklable, so storing the cache in one made every
    # instance unpicklable -- including one that has never built a plan.
    # FresnelPropagator's documented `propagator=` reuse argument invites
    # sending exactly such an unused instance to a dask.distributed worker.
    cloudpickle = pytest.importorskip("cloudpickle")

    convolution = abtem_fft.CachedFFTWConvolution()
    restored = cloudpickle.loads(cloudpickle.dumps(convolution))

    rng = np.random.default_rng(5)
    array, kernel = _random_convolution_inputs(rng)
    result = restored(array.copy(), kernel, True)
    assert np.allclose(result, _convolution_reference(array, kernel), atol=1e-6)


@requires_pyfftw
def test_cached_fftw_convolution_is_picklable_after_use():
    # A plan cache built in one process/thread is not valid in another, so a
    # restored instance must drop it and transparently rebuild rather than
    # shipping a stale cache across the pickle boundary.
    cloudpickle = pytest.importorskip("cloudpickle")

    rng = np.random.default_rng(6)
    array, kernel = _random_convolution_inputs(rng)
    convolution = abtem_fft.CachedFFTWConvolution()
    convolution(array.copy(), kernel, True)

    restored = cloudpickle.loads(cloudpickle.dumps(convolution))
    assert not hasattr(restored, "_local") or not hasattr(
        restored._local, "cached"
    )

    result = restored(array.copy(), kernel, True)
    assert np.allclose(result, _convolution_reference(array, kernel), atol=1e-6)


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
        import numpy as np

        rng = np.random.default_rng(0)
        return (
            rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        ).astype(complex)

    def test_fft_crop_with_empty_new_shape_is_a_no_op(self):
        import numpy as np

        from abtem.core.fft import fft_crop

        array = self._array()
        assert np.array_equal(fft_crop(array, ()), array)

    def test_fft_interpolate_with_empty_new_shape_is_a_no_op(self):
        import numpy as np

        from abtem.core.fft import fft_interpolate

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
        import numpy as np

        import abtem
        from abtem.waves import Waves

        with abtem.config.set({"fft": "numpy"}):
            array = self._array().astype("complex64")
            waves = Waves(array, energy=100e3, extent=(10, 10))

            none_out = abtem.WavesDetector(gpts=None).detect(waves)
            empty_out = abtem.WavesDetector(gpts=()).detect(waves)

        assert np.array_equal(
            np.asarray(none_out.array), np.asarray(empty_out.array)
        )
        assert np.array_equal(np.asarray(empty_out.array), array)
