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


def test_device_index_arrays_are_cached_per_device():
    indices = M._polar_detector_bins(**{**BINS_KWARGS, "return_indices": True})
    key = ((128, 128), (0.05, 0.05), 0.0, 50.0, 10, 1, 0.0, (0.0, 0.0), False)

    M._RADIAL_BINNING_DEVICE_CACHE.clear()
    flat_a, sep_a = M._radial_binning_device_arrays(np, key, indices)
    flat_b, sep_b = M._radial_binning_device_arrays(np, key, indices)

    assert flat_a is flat_b and sep_a is sep_b
    assert np.array_equal(flat_a, np.concatenate(indices))
    assert int(sep_a[-1]) == sum(len(i) for i in indices)


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
            M._RADIAL_BINNING_DEVICE_CACHE.clear()
            first = np.asarray(detector.detect(waves).array)
            second = np.asarray(detector.detect(waves).array)  # served from cache
            assert np.allclose(first, second, rtol=0, atol=0)
            assert M._polar_detector_bins_cached.cache_info().hits >= 1
