"""Tests for abtem/noise.py"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

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

@given(
    rows=st.integers(4, 32),
    cols=st.integers(4, 32),
    dwell_time=st.floats(1e-7, 1e-5),
    flyback_time=st.floats(1e-5, 1e-3),
)
def test_pixel_times_properties(rows, cols, dwell_time, flyback_time):
    times = _pixel_times(dwell_time, flyback_time, (rows, cols))
    assert times.shape == (rows, cols)
    assert np.all(times > 0)
    assert np.all(np.diff(times, axis=0) > 0)
    assert np.all(np.diff(times, axis=1) > 0)


# ---------------------------------------------------------------------------
# _single_axis_distortion
# ---------------------------------------------------------------------------

def test_single_axis_distortion_shape_and_seed():
    time = _pixel_times(1e-6, 1e-4, (16, 16))
    d1 = _single_axis_distortion(time, 500, 10, seed=42)
    d2 = _single_axis_distortion(time, 500, 10, seed=42)
    assert d1.shape == (16, 16) and np.allclose(d1, d2)
    assert not np.allclose(d1, _single_axis_distortion(time, 500, 10, seed=99))


def test_single_axis_distortion_zero_components():
    time = _pixel_times(1e-6, 1e-4, (8, 8))
    assert np.allclose(_single_axis_distortion(time, 500, 0, seed=0), 0.0)


# ---------------------------------------------------------------------------
# _make_displacement_field
# ---------------------------------------------------------------------------

def test_make_displacement_field_properties():
    time = _pixel_times(1e-6, 1e-4, (16, 16))
    dx1, dy1 = _make_displacement_field(time, 500, 20, rms_power=0.1, seed=0)
    dx2, _ = _make_displacement_field(time, 500, 20, rms_power=10.0, seed=0)
    assert dx1.shape == (16, 16) and dy1.shape == (16, 16)
    # same seed → same result; larger rms_power → larger displacements
    dx1b, dy1b = _make_displacement_field(time, 500, 20, rms_power=0.1, seed=0)
    assert np.allclose(dx1, dx1b) and np.allclose(dy1, dy1b)
    assert np.std(dx2) > np.std(dx1)


# ---------------------------------------------------------------------------
# _apply_displacement_field
# ---------------------------------------------------------------------------

class TestApplyDisplacementField:
    def test_output_shape(self):
        img = np.ones((16, 16))
        result = _apply_displacement_field(img, np.zeros_like(img), np.zeros_like(img))
        assert result.shape == (16, 16)

    def test_zero_displacement_preserves_interior(self):
        # RegularGridInterpolator wraps boundary pixels (p % x.max()), so
        # the last column/row gets remapped to index 0; only check interior.
        img = np.random.default_rng(0).random((16, 16))
        result = _apply_displacement_field(img, np.zeros_like(img), np.zeros_like(img))
        assert np.allclose(result[:-1, :-1], img[:-1, :-1], atol=1e-10)

    def test_nonzero_displacement_changes_image(self):
        img = np.random.default_rng(1).random((16, 16))
        result = _apply_displacement_field(img, np.ones_like(img) * 0.5, np.zeros_like(img))
        assert not np.allclose(result, img)

    def test_output_finite_with_realistic_field(self):
        img = np.random.default_rng(2).random((16, 16))
        time = _pixel_times(1e-6, 1e-4, (16, 16))
        dx, dy = _make_displacement_field(time, 500, 20, 1.0, seed=0)
        assert np.all(np.isfinite(_apply_displacement_field(img, dx, dy)))


# ---------------------------------------------------------------------------
# NoiseTransform
# ---------------------------------------------------------------------------

class TestNoiseTransform:
    def _images(self, shape=(16, 16), value=100.0):
        from abtem.measurements import Images
        return Images(np.full(shape, value), sampling=(0.1, 0.1))

    def test_attributes(self):
        nt = NoiseTransform(dose=1000.0, samples=5)
        assert nt.dose == 1000.0 and nt.samples == 5
        assert "label" in nt.metadata

    def test_apply_returns_images_and_shape(self):
        from abtem.measurements import Images
        nt = NoiseTransform(dose=1000.0, samples=4)
        result = nt.apply(self._images()).compute()
        assert isinstance(result, Images) and result.shape[0] == 4

    def test_nonnegative_and_reproducible(self):
        imgs = self._images()
        r1 = NoiseTransform(dose=500.0, seeds=42).apply(imgs).compute()
        r2 = NoiseTransform(dose=500.0, seeds=42).apply(imgs).compute()
        assert np.all(r1.array >= 0)
        assert np.allclose(r1.array, r2.array)

    def test_ensemble_axes_metadata(self):
        from abtem.distributions import from_values
        assert NoiseTransform(dose=1000.0).ensemble_axes_metadata == []
        assert len(NoiseTransform(dose=from_values([500.0, 1000.0])).ensemble_axes_metadata) == 1


# ---------------------------------------------------------------------------
# ScanNoiseTransform
# ---------------------------------------------------------------------------

class TestScanNoiseTransform:
    def _images(self):
        from abtem.measurements import Images
        return Images(np.ones((16, 16)), sampling=(0.1, 0.1))

    def test_construction_and_properties(self):
        snt = ScanNoiseTransform(
            rms_power=2.0, dwell_time=1e-5, flyback_time=2e-4,
            max_frequency=300, num_components=50,
        )
        assert snt.rms_power == 2.0 and snt.dwell_time == 1e-5
        assert snt.max_frequency == 300 and snt.num_components == 50
        assert snt.samples == 1

    def test_apply_returns_images_with_correct_shape(self):
        from abtem.measurements import Images
        snt = ScanNoiseTransform(1.0, 1e-6, 1e-4, num_components=10)
        imgs = self._images()
        result = snt.apply(imgs).compute()
        assert isinstance(result, Images) and result.base_shape == imgs.base_shape

    def test_samples_and_seeds(self):
        snt = ScanNoiseTransform(
            rms_power=1.0, dwell_time=1e-6, flyback_time=1e-4,
            seeds=0, samples=2, num_components=5,
        )
        assert snt.apply(self._images()).compute().shape[0] == 2

    def test_ensemble_axes_metadata(self):
        assert ScanNoiseTransform(1.0, 1e-6, 1e-4).ensemble_axes_metadata == []
