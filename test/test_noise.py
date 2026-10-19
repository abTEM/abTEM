"""Tests for abtem/noise.py"""

import numpy as np
import pytest

from abtem.measurements import Images
from abtem.noise import (
    ScanNoiseTransform,
    _make_displacement_field,
    _pixel_times,
)


def make_images(shape=(32, 32), sampling=(0.1, 0.1)):
    # a non-constant image, as a distorted constant image is unchanged
    return Images(np.random.default_rng(0).random(shape), sampling=sampling)


# ---------------------------------------------------------------------------
# _pixel_times
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 32), (32, 4), (7, 5)])
def test_pixel_times_raster_model(shape):
    # Documented raster model: axis 0 is the fast (line) axis, axis 1 indexes the
    # scan lines. Along a line the beam waits `dwell` per pixel; at the end of a
    # line it spends `flyback` returning before the next line starts. Non-square
    # shapes in both orientations catch a mix-up of shape[0] and shape[1].
    dwell, flyback = 1e-6, 3.7e-5
    n_fast, n_lines = shape
    times = _pixel_times(dwell, flyback, shape)
    assert times.shape == shape

    line_time = n_fast * dwell + flyback

    # consecutive pixels along the fast axis are one dwell time apart
    np.testing.assert_allclose(np.diff(times, axis=0), dwell, rtol=1e-9)
    # the same pixel on consecutive lines is one full line (incl. flyback) apart
    np.testing.assert_allclose(np.diff(times, axis=1), line_time, rtol=1e-9)
    # line end -> start of the next line: one dwell plus the flyback
    np.testing.assert_allclose(
        times[0, 1:] - times[-1, :-1], dwell + flyback, rtol=1e-9
    )
    # the whole frame (first to last pixel, plus the last pixel's dwell and
    # flyback) takes n_lines full line times
    np.testing.assert_allclose(
        times[-1, -1] - times[0, 0] + dwell + flyback, n_lines * line_time, rtol=1e-9
    )
    # visiting the pixels in raster order (axis 0 fastest) is strictly monotonic
    assert np.all(np.diff(times.ravel(order="F")) > 0)


# ---------------------------------------------------------------------------
# _make_displacement_field
# ---------------------------------------------------------------------------


def test_make_displacement_field_seeded_profiles_independent():
    time = _pixel_times(1e-6, 1e-4, (16, 24))
    dx, dy = _make_displacement_field(time, 500, 20, rms_power=1.0, seed=0)
    # x and y distortions are separate random processes, even for a fixed seed
    assert not np.allclose(dx, dy)
    # ... but the same seed reproduces both exactly
    dx_b, dy_b = _make_displacement_field(time, 500, 20, rms_power=1.0, seed=0)
    assert np.array_equal(dx, dx_b) and np.array_equal(dy, dy_b)
    dx_c, dy_c = _make_displacement_field(time, 500, 20, rms_power=1.0, seed=1)
    assert not np.allclose(dx, dx_c) and not np.allclose(dy, dy_c)
    # seed=None stays random
    dx_1, _ = _make_displacement_field(time, 500, 20, rms_power=1.0)
    dx_2, _ = _make_displacement_field(time, 500, 20, rms_power=1.0)
    assert not np.allclose(dx_1, dx_2)


def _plant_profiles(monkeypatch, profile_x, profile_y):
    # _make_displacement_field draws the x profile first, then the y profile
    import abtem.noise

    planted = iter([profile_x, profile_y])
    monkeypatch.setattr(
        abtem.noise, "_single_axis_distortion", lambda *args, **kwargs: next(planted)
    )


# The x (y) displacement is added to the axis-0 (axis-1) coordinate in
# _apply_displacement_field, so its magnification deviation is its derivative
# along axis 0 (axis 1), in pixels per pixel. rms_power is in percent and, per
# the code's stated convention, a FWHM: the rms (sigma) of the frame
# magnification deviation must be rms_power / (100 * 2.355).
DWELL, FLYBACK, SHAPE = 1e-6, 5e-5, (6, 10)
LINE_TIME = SHAPE[0] * DWELL + FLYBACK


def test_make_displacement_field_pure_x_ramp(monkeypatch):
    # displacement_x = a * t: d/d(axis 0) = a * dwell everywhere (exact for a
    # linear ramp), no y displacement -> frame deviation is exactly a * dwell,
    # so the output is scaled to t * rms_power / (235.5 * dwell).
    time = _pixel_times(DWELL, FLYBACK, SHAPE)
    rms_power = 3.0
    # slope 0.5 px/px keeps (1 + gx) - 1 free of float cancellation
    _plant_profiles(monkeypatch, 0.5 / DWELL * time, np.zeros_like(time))
    dx, dy = _make_displacement_field(time, 500, 1, rms_power=rms_power)
    np.testing.assert_allclose(dx, time * rms_power / (235.5 * DWELL), rtol=1e-12)
    assert np.all(dy == 0)
    frame = np.gradient(dx, axis=0)
    np.testing.assert_allclose(np.sqrt(np.mean(frame**2)), rms_power / 235.5)


def test_make_displacement_field_pure_y_ramp(monkeypatch):
    # displacement_y = b * t: d/d(axis 1) = b * line_time everywhere.
    time = _pixel_times(DWELL, FLYBACK, SHAPE)
    rms_power = 3.0
    _plant_profiles(monkeypatch, np.zeros_like(time), 0.5 / LINE_TIME * time)
    dx, dy = _make_displacement_field(time, 500, 1, rms_power=rms_power)
    assert np.all(dx == 0)
    np.testing.assert_allclose(dy, time * rms_power / (235.5 * LINE_TIME), rtol=1e-12)
    frame = np.gradient(dy, axis=1)
    np.testing.assert_allclose(np.sqrt(np.mean(frame**2)), rms_power / 235.5)


def test_make_displacement_field_rms_normalisation():
    # Random x and y distortions: frame deviation (1 + gx)(1 + gy) - 1 is
    # linear in the scale factor up to the gx * gy cross term, which is of
    # relative size ~ rms_power / 235.5 ~ 4e-5 here, so rtol 1e-3 is safe.
    time = _pixel_times(DWELL, FLYBACK, (32, 48))
    rms_power = 0.01
    dx, dy = _make_displacement_field(time, 500, 50, rms_power=rms_power, seed=3)
    gx = np.gradient(dx, axis=0)
    gy = np.gradient(dy, axis=1)
    frame = (1 + gx) * (1 + gy) - 1
    np.testing.assert_allclose(np.sqrt(np.mean(frame**2)), rms_power / 235.5, rtol=1e-3)


# ---------------------------------------------------------------------------
# ScanNoiseTransform
# ---------------------------------------------------------------------------


class TestScanNoiseTransform:
    def test_samples_get_independent_reproducible_distortions(self):
        imgs = make_images((16, 24))

        def transform(seeds, samples=3):
            return ScanNoiseTransform(
                rms_power=5.0,
                dwell_time=1e-6,
                flyback_time=1e-4,
                samples=samples,
                seeds=seeds,
                num_components=20,
            )

        noisy = transform(0).apply(imgs).compute().array
        assert noisy.shape == (3, 16, 24)
        # each sample is a separate distortion realisation ...
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert not np.allclose(noisy[i], noisy[j])
        # ... the same seed reproduces all of them exactly ...
        assert np.array_equal(noisy, transform(0).apply(imgs).compute().array)
        # ... and sample k is the realisation of its own entry in `seeds`, so a
        # sample does not depend on which other samples are drawn with it
        for k, seed in enumerate(transform(0).seeds.values):
            single = transform((int(seed),), samples=None).apply(imgs).compute()
            np.testing.assert_array_equal(single.array[0], noisy[k])

    def test_unseeded_samples_are_random(self):
        imgs = make_images((16, 24))
        snt = ScanNoiseTransform(
            rms_power=5.0,
            dwell_time=1e-6,
            flyback_time=1e-4,
            samples=3,
            num_components=20,
        )
        noisy = snt.apply(imgs).compute().array
        assert noisy.shape == (3, 16, 24)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert not np.allclose(noisy[i], noisy[j])
