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
