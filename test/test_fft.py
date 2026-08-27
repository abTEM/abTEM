"""Tests for FFT helpers, in particular fast-radix transform-size handling."""

import warnings

import pytest

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

    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": "auto"}):
        abtem_fft._configure_cufft_cache()
        assert abtem_fft._CUFFT_CACHE_STATE is not None
        assert abtem_fft._CUFFT_CACHE_STATE[1] == expected
        assert cp.fft.config.get_plan_cache().get_memsize() == expected

    # -1 must still mean unlimited (no memsize bound applied).
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": -1}):
        abtem_fft._configure_cufft_cache()
        assert abtem_fft._CUFFT_CACHE_STATE[1] == -1


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
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": "auto"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_memsize() > 0
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": -1}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_memsize() == -1

    # null means "no bound" rather than crashing.
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": None}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_memsize() == -1

    # A positive bound re-enables a previously disabled cache.
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": "0 MB"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() == 0
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": "1 GB"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() > 0
        assert cache.get_memsize() == 10**9

    # Restore the shipped default for subsequent tests.
    abtem_fft._CUFFT_CACHE_STATE = None
    abtem_fft._configure_cufft_cache()


def test_plan_cache_entry_limit_is_raised():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no GPU available")
    except Exception:
        pytest.skip("no usable CUDA/HIP runtime")

    from abtem.core import config
    from abtem.core import fft as abtem_fft

    cache = cp.fft.config.get_plan_cache()

    # CuPy's own default of 16 thrashes for varying batch shapes.
    abtem_fft._CUFFT_CACHE_STATE = None
    abtem_fft._configure_cufft_cache()
    assert cache.get_size() == 64

    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-entries": 128}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() == 128

    # Disabling the cache still wins over the entry count.
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": "0 MB"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() == 0

    abtem_fft._CUFFT_CACHE_STATE = None
    abtem_fft._configure_cufft_cache()
