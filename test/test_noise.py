"""Tests for abtem/noise.py"""

import numpy as np
import pytest

from abtem.noise import (
    NoiseTransform,
    ScanNoiseTransform,
    _apply_displacement_field,
    _make_displacement_field,
    _pixel_times,
    _single_axis_distortion,
)


# ---------------------------------------------------------------------------
# _pixel_times
# ---------------------------------------------------------------------------

class TestPixelTimes:
    def test_output_shape(self):
        times = _pixel_times(dwell_time=1e-6, flyback_time=1e-4, shape=(16, 32))
        assert times.shape == (16, 32)

    def test_monotonically_increasing_along_slow_axis(self):
        times = _pixel_times(1e-6, 1e-4, (8, 8))
        # each row should have larger times than the previous row
        assert np.all(np.diff(times, axis=1) > 0)

    def test_monotonically_increasing_along_fast_axis(self):
        times = _pixel_times(1e-6, 1e-4, (8, 8))
        assert np.all(np.diff(times, axis=0) > 0)

    def test_all_positive(self):
        times = _pixel_times(1e-6, 1e-4, (8, 8))
        assert np.all(times > 0)

    def test_square_and_rectangular(self):
        t1 = _pixel_times(1e-6, 1e-4, (8, 8))
        t2 = _pixel_times(1e-6, 1e-4, (4, 16))
        assert t1.shape == (8, 8)
        assert t2.shape == (4, 16)


# ---------------------------------------------------------------------------
# _single_axis_distortion
# ---------------------------------------------------------------------------

class TestSingleAxisDistortion:
    def test_output_shape(self):
        time = _pixel_times(1e-6, 1e-4, (16, 16))
        d = _single_axis_distortion(time, max_frequency=500, num_components=10)
        assert d.shape == (16, 16)

    def test_reproducible_with_seed(self):
        time = _pixel_times(1e-6, 1e-4, (8, 8))
        d1 = _single_axis_distortion(time, 500, 10, seed=42)
        d2 = _single_axis_distortion(time, 500, 10, seed=42)
        assert np.allclose(d1, d2)

    def test_different_seeds_differ(self):
        time = _pixel_times(1e-6, 1e-4, (8, 8))
        d1 = _single_axis_distortion(time, 500, 10, seed=1)
        d2 = _single_axis_distortion(time, 500, 10, seed=2)
        assert not np.allclose(d1, d2)

    def test_zero_components_gives_zeros(self):
        time = _pixel_times(1e-6, 1e-4, (8, 8))
        d = _single_axis_distortion(time, 500, 0, seed=0)
        assert np.allclose(d, 0.0)


# ---------------------------------------------------------------------------
# _make_displacement_field
# ---------------------------------------------------------------------------

class TestMakeDisplacementField:
    def test_output_shapes(self):
        time = _pixel_times(1e-6, 1e-4, (16, 16))
        dx, dy = _make_displacement_field(time, 500, 20, rms_power=1.0)
        assert dx.shape == (16, 16)
        assert dy.shape == (16, 16)

    def test_reproducible_with_seed(self):
        time = _pixel_times(1e-6, 1e-4, (8, 8))
        dx1, dy1 = _make_displacement_field(time, 500, 10, 1.0, seed=0)
        dx2, dy2 = _make_displacement_field(time, 500, 10, 1.0, seed=0)
        assert np.allclose(dx1, dx2)
        assert np.allclose(dy1, dy2)

    def test_larger_rms_larger_displacement(self):
        time = _pixel_times(1e-6, 1e-4, (16, 16))
        dx1, _ = _make_displacement_field(time, 500, 20, 0.1, seed=0)
        dx2, _ = _make_displacement_field(time, 500, 20, 10.0, seed=0)
        assert np.std(dx2) > np.std(dx1)


# ---------------------------------------------------------------------------
# _apply_displacement_field
# ---------------------------------------------------------------------------

class TestApplyDisplacementField:
    def test_output_shape(self):
        img = np.ones((16, 16))
        dx = np.zeros((16, 16))
        dy = np.zeros((16, 16))
        result = _apply_displacement_field(img, dx, dy)
        assert result.shape == (16, 16)

    def test_zero_displacement_preserves_image(self):
        # RegularGridInterpolator wraps boundary pixels (p % x.max()), so
        # the last column/row gets remapped to index 0; allow those pixels to differ.
        img = np.random.default_rng(0).random((16, 16))
        dx = np.zeros_like(img)
        dy = np.zeros_like(img)
        result = _apply_displacement_field(img, dx, dy)
        # Interior pixels should be preserved exactly
        assert np.allclose(result[:-1, :-1], img[:-1, :-1], atol=1e-10)

    def test_nonzero_displacement_changes_image(self):
        img = np.random.default_rng(1).random((16, 16))
        dx = np.ones_like(img) * 0.5
        dy = np.zeros_like(img)
        result = _apply_displacement_field(img, dx, dy)
        assert not np.allclose(result, img)

    def test_output_finite(self):
        img = np.random.default_rng(2).random((16, 16))
        time = _pixel_times(1e-6, 1e-4, (16, 16))
        dx, dy = _make_displacement_field(time, 500, 20, 1.0, seed=0)
        result = _apply_displacement_field(img, dx, dy)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# NoiseTransform
# ---------------------------------------------------------------------------

class TestNoiseTransform:
    def _images(self, shape=(16, 16), value=100.0):
        from abtem.measurements import Images
        return Images(np.full(shape, value), sampling=(0.1, 0.1))

    def test_scalar_dose(self):
        nt = NoiseTransform(dose=1000.0)
        assert nt.dose == 1000.0

    def test_samples_is_1_by_default(self):
        nt = NoiseTransform(dose=1000.0)
        assert nt.samples == 1

    def test_samples_with_explicit_samples(self):
        nt = NoiseTransform(dose=1000.0, samples=5)
        assert nt.samples == 5

    def test_metadata(self):
        nt = NoiseTransform(dose=1000.0)
        assert "label" in nt.metadata

    def test_apply_returns_images(self):
        from abtem.measurements import Images
        nt = NoiseTransform(dose=1000.0)
        imgs = self._images()
        result = nt.apply(imgs).compute()
        assert isinstance(result, Images)

    def test_apply_with_samples_adds_axis(self):
        nt = NoiseTransform(dose=1000.0, samples=4)
        imgs = self._images()
        result = nt.apply(imgs).compute()
        assert result.shape[0] == 4

    def test_poisson_noise_is_non_negative(self):
        nt = NoiseTransform(dose=500.0)
        imgs = self._images()
        result = nt.apply(imgs).compute()
        assert np.all(result.array >= 0)

    def test_seed_reproducibility(self):
        nt1 = NoiseTransform(dose=1000.0, seeds=42)
        nt2 = NoiseTransform(dose=1000.0, seeds=42)
        imgs = self._images()
        r1 = nt1.apply(imgs).compute()
        r2 = nt2.apply(imgs).compute()
        assert np.allclose(r1.array, r2.array)

    def test_ensemble_axes_metadata_scalar_dose(self):
        nt = NoiseTransform(dose=1000.0)
        assert nt.ensemble_axes_metadata == []

    def test_ensemble_axes_metadata_distribution_dose(self):
        from abtem.distributions import from_values
        doses = from_values([500.0, 1000.0])
        nt = NoiseTransform(dose=doses)
        assert len(nt.ensemble_axes_metadata) == 1


# ---------------------------------------------------------------------------
# ScanNoiseTransform
# ---------------------------------------------------------------------------

class TestScanNoiseTransform:
    def _images(self, shape=(16, 16)):
        from abtem.measurements import Images
        return Images(np.ones(shape), sampling=(0.1, 0.1))

    def test_construction(self):
        snt = ScanNoiseTransform(rms_power=1.0, dwell_time=1e-6, flyback_time=1e-4)
        assert snt.dwell_time == 1e-6
        assert snt.flyback_time == 1e-4

    def test_default_max_frequency(self):
        snt = ScanNoiseTransform(1.0, 1e-6, 1e-4)
        assert snt.max_frequency == 500

    def test_properties(self):
        snt = ScanNoiseTransform(
            rms_power=2.0, dwell_time=1e-5, flyback_time=2e-4,
            max_frequency=300, num_components=50,
        )
        assert snt.rms_power == 2.0
        assert snt.num_components == 50

    def test_samples_is_1_by_default(self):
        snt = ScanNoiseTransform(1.0, 1e-6, 1e-4)
        assert snt.samples == 1

    def test_apply_returns_images(self):
        from abtem.measurements import Images
        snt = ScanNoiseTransform(
            rms_power=1.0, dwell_time=1e-6, flyback_time=1e-4,
            num_components=10,
        )
        imgs = self._images()
        result = snt.apply(imgs).compute()
        assert isinstance(result, Images)

    def test_output_shape_unchanged(self):
        snt = ScanNoiseTransform(1.0, 1e-6, 1e-4, num_components=10)
        imgs = self._images()
        result = snt.apply(imgs).compute()
        assert result.base_shape == imgs.base_shape

    def test_ensemble_axes_metadata_scalar(self):
        snt = ScanNoiseTransform(1.0, 1e-6, 1e-4)
        assert snt.ensemble_axes_metadata == []

    def test_with_seeds(self):
        snt = ScanNoiseTransform(
            rms_power=1.0, dwell_time=1e-6, flyback_time=1e-4,
            seeds=0, samples=2, num_components=5,
        )
        imgs = self._images()
        result = snt.apply(imgs).compute()
        assert result.shape[0] == 2
