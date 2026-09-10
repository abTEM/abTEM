"""Detector bin geometry is cached rather than rebuilt per detect call."""

import numpy as np
import pytest

import abtem
import abtem.measurements as M


BINS_KWARGS = dict(
    gpts=(128, 128),
    sampling=(0.05, 0.05),
    inner=0.0,
    outer=50.0,
    nbins_radial=10,
    nbins_azimuthal=1,
)


def test_polar_detector_bins_are_cached():
    M._polar_detector_bins_cached.cache_clear()

    first = M._polar_detector_bins(**BINS_KWARGS)
    second = M._polar_detector_bins(**BINS_KWARGS)

    assert first is second
    info = M._polar_detector_bins_cached.cache_info()
    assert info.hits == 1 and info.misses == 1


def test_cached_bins_are_read_only():
    """Callers share the cached arrays, so they must not be mutable."""
    bins = M._polar_detector_bins(**BINS_KWARGS)
    with pytest.raises(ValueError):
        bins[0, 0] = 123


def test_cached_bins_match_the_uncached_computation():
    M._polar_detector_bins_cached.cache_clear()

    cached = M._polar_detector_bins(**BINS_KWARGS)
    uncached = M._polar_detector_bins_uncached(**BINS_KWARGS)

    assert np.array_equal(cached, uncached)


def test_differing_geometry_is_not_confused():
    a = M._polar_detector_bins(**BINS_KWARGS)
    b = M._polar_detector_bins(**{**BINS_KWARGS, "outer": 30.0})
    c = M._polar_detector_bins(**{**BINS_KWARGS, "nbins_radial": 5})

    assert a is not b and a is not c
    assert not np.array_equal(a, b)


_GEOMETRY_KWARGS = dict(
    {k: v for k, v in BINS_KWARGS.items() if k != "gpts"},
    rotation=0.0,
    offset=(0.0, 0.0),
    fftshift=False,
)


def test_device_index_arrays_are_cached_per_device():
    array = np.zeros((2, *BINS_KWARGS["gpts"]))

    M._radial_binning_device_arrays_cached.cache_clear()
    flat_a, sep_a = M._radial_binning_device_arrays(array, **_GEOMETRY_KWARGS)
    flat_b, sep_b = M._radial_binning_device_arrays(array, **_GEOMETRY_KWARGS)

    assert flat_a is flat_b and sep_a is sep_b
    indices = M._polar_detector_bins(**{**BINS_KWARGS, "return_indices": True})
    assert np.array_equal(flat_a, np.concatenate(indices))
    assert int(sep_a[-1]) == sum(len(i) for i in indices)
    # Shared between callers, so they must not be mutable (cpu arrays only;
    # cupy arrays cannot be flagged).
    assert not flat_a.flags.writeable and not sep_a.flags.writeable


def test_device_cache_evicts_least_recently_used():
    """A hot entry survives inserts beyond the bound; the stalest is evicted."""
    array = np.zeros((1, *BINS_KWARGS["gpts"]))

    M._radial_binning_device_arrays_cached.cache_clear()
    hot, _ = M._radial_binning_device_arrays(array, **_GEOMETRY_KWARGS)

    maxsize = M._radial_binning_device_arrays_cached.cache_info().maxsize
    for i in range(maxsize + 3):
        M._radial_binning_device_arrays(
            array, **{**_GEOMETRY_KWARGS, "outer": 30.0 + i}
        )
        M._radial_binning_device_arrays(array, **_GEOMETRY_KWARGS)

    again, _ = M._radial_binning_device_arrays(array, **_GEOMETRY_KWARGS)
    assert again is hot
    info = M._radial_binning_device_arrays_cached.cache_info()
    assert info.currsize <= info.maxsize


def test_radial_detectors_give_unchanged_results():
    """The cache must not change what a detector produces."""
    from abtem.core import config

    with config.set({"device": "cpu"}):
        probe = abtem.Probe(
            energy=100e3, semiangle_cutoff=20, gpts=(64, 64), extent=(10.0, 10.0)
        )
        waves = probe.build(lazy=False)

        for detector in (
            abtem.FlexibleAnnularDetector(),
            abtem.SegmentedDetector(
                inner=20, outer=60, nbins_radial=2, nbins_azimuthal=4
            ),
        ):
            M._polar_detector_bins_cached.cache_clear()
            M._radial_binning_device_arrays_cached.cache_clear()
            first = np.asarray(detector.detect(waves).array)
            second = np.asarray(detector.detect(waves).array)  # served from cache
            assert np.allclose(first, second, rtol=0, atol=0)
            # The repeat call is served from the device cache without even
            # touching the host-side geometry cache again.
            info = M._polar_detector_bins_cached.cache_info()
            device_info = M._radial_binning_device_arrays_cached.cache_info()
            assert info.misses == 1 and device_info.currsize == 1


def test_device_cache_is_thread_safe_under_churn():
    """Concurrent misses beyond the bound must not race the eviction (the
    hand-rolled predecessor could KeyError under the threaded scheduler)."""
    import concurrent.futures

    array = np.zeros((1, *BINS_KWARGS["gpts"]))
    M._radial_binning_device_arrays_cached.cache_clear()

    def hammer(worker):
        for j in range(30):
            outer = 20.0 + (worker * 30 + j) % 12  # > maxsize distinct keys
            M._radial_binning_device_arrays(
                array, **{**_GEOMETRY_KWARGS, "outer": outer}
            )

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(hammer, range(8)))

    info = M._radial_binning_device_arrays_cached.cache_info()
    assert info.currsize <= info.maxsize
