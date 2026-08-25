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
    _warned_slow_fft_shapes.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_slow_fft_size((256, 256))


def test_warn_slow_fft_size_silent_for_small_shapes():
    """Small transforms (e.g. interpolated measurements) never warn, even when
    their lengths would force the Bluestein path."""
    _warned_slow_fft_shapes.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_slow_fft_size((36, 29))
        _warn_slow_fft_size((4, 11, 20))


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
