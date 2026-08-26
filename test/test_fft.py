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
