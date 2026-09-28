"""Tests for abtem/noise.py"""

import numpy as np
import pytest

from abtem.core.axes import ScanAxis
from abtem.measurements import DiffractionPatterns
from abtem.noise import (
    NoiseTransform,
    ScanNoiseTransform,
    _apply_displacement_field,
    _make_displacement_field,
    _pixel_times,
    _single_axis_distortion,
)
from test_measure import make_images
from utils import gpu


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
    np.testing.assert_allclose(
        np.sqrt(np.mean(frame**2)), rms_power / 235.5, rtol=1e-3
    )


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
        return make_images(shape, value=value)

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
# Poisson statistics of poisson_noise
# ---------------------------------------------------------------------------


def _assert_poisson(counts, lam):
    """Assert sample mean and variance of i.i.d. counts match Poisson(lam).

    For Poisson(lam): mean = var = lam; the standard error of the sample mean
    is sqrt(lam / n) and that of the sample variance is sqrt((mu4 - var^2) / n)
    with mu4 = lam + 3 lam^2, i.e. sqrt((lam + 2 lam^2) / n). Tolerance is 5
    standard errors (seeds are fixed, so this is also deterministic).
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    n = counts.size
    assert np.all(counts == np.round(counts)) and np.all(counts >= 0)
    assert abs(counts.mean() - lam) < 5 * np.sqrt(lam / n)
    assert abs(counts.var(ddof=1) - lam) < 5 * np.sqrt((lam + 2 * lam**2) / n)


def _to_device(measurement, device):
    return measurement.to_gpu() if device == "gpu" else measurement


def _to_cpu_array(measurement):
    array = measurement.compute().array
    return array.get() if hasattr(array, "get") else array


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lam", [0.5, 50.0])
class TestPoissonNoiseStatistics:
    def test_images_dose_per_area(self, lam, device):
        # Expected counts per pixel = intensity * dose_per_area * pixel area.
        intensity, sampling = 0.25, (0.1, 0.2)
        dose_per_area = lam / (intensity * sampling[0] * sampling[1])
        images = _to_device(
            make_images((400, 500), sampling=sampling, value=intensity), device
        )
        noisy = images.poisson_noise(dose_per_area=dose_per_area, seed=11)
        _assert_poisson(_to_cpu_array(noisy), lam)

    def test_diffraction_patterns_total_dose(self, lam, device):
        # A pattern normalised to unit total intensity, spread uniformly over
        # N detector pixels, receiving `total_dose` electrons: lam = total_dose / N.
        n_det = 512 * 400
        total_dose = lam * n_det
        dp = DiffractionPatterns(np.full((512, 400), 1.0 / n_det), sampling=0.1)
        noisy = _to_device(dp, device).poisson_noise(total_dose=total_dose, seed=12)
        _assert_poisson(_to_cpu_array(noisy), lam)

    def test_diffraction_patterns_dose_per_area(self, lam, device):
        # 4D-STEM: electrons per probe position = dose_per_area * scan pixel area;
        # a normalised uniform pattern over N detector pixels gives
        # lam = dose_per_area * dx * dy / N.
        n_det = 32 * 32
        scan_sampling = (0.2, 0.25)
        dose_per_area = lam * n_det / (scan_sampling[0] * scan_sampling[1])
        dp = DiffractionPatterns(
            np.full((20, 10, 32, 32), 1.0 / n_det),
            sampling=0.1,
            ensemble_axes_metadata=[
                ScanAxis(label="x", sampling=scan_sampling[0], units="Å"),
                ScanAxis(label="y", sampling=scan_sampling[1], units="Å"),
            ],
        )
        noisy = _to_device(dp, device).poisson_noise(
            dose_per_area=dose_per_area, seed=13
        )
        _assert_poisson(_to_cpu_array(noisy), lam)

    def test_samples_are_independent_draws(self, lam, device):
        # Each of the `samples` draws must itself be Poisson(lam), and distinct
        # draws must be uncorrelated: for independent draws the sample Pearson
        # correlation over n pixels has standard error ~1/sqrt(n).
        samples, intensity, sampling = 4, 1.0, (0.1, 0.1)
        dose_per_area = lam / (intensity * sampling[0] * sampling[1])
        images = _to_device(
            make_images((250, 200), sampling=sampling, value=intensity), device
        )
        noisy = images.poisson_noise(
            dose_per_area=dose_per_area, samples=samples, seed=14
        )
        array = _to_cpu_array(noisy)
        assert array.shape == (samples, 250, 200)
        flat = array.reshape(samples, -1).astype(np.float64)
        for draw in flat:
            _assert_poisson(draw, lam)
        corr = np.corrcoef(flat)
        off_diagonal = corr[~np.eye(samples, dtype=bool)]
        assert np.all(np.abs(off_diagonal) < 5 / np.sqrt(flat.shape[1]))


# ---------------------------------------------------------------------------
# ScanNoiseTransform
# ---------------------------------------------------------------------------

class TestScanNoiseTransform:
    def _images(self):
        return make_images((16, 16), value=1.0)

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

    def test_samples_get_independent_reproducible_distortions(self):
        # A non-constant image is needed: a distorted constant image is unchanged.
        imgs = make_images((16, 24))

        def transform(seeds, samples=3):
            return ScanNoiseTransform(
                rms_power=5.0, dwell_time=1e-6, flyback_time=1e-4,
                samples=samples, seeds=seeds, num_components=20,
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
            rms_power=5.0, dwell_time=1e-6, flyback_time=1e-4,
            samples=3, num_components=20,
        )
        noisy = snt.apply(imgs).compute().array
        assert noisy.shape == (3, 16, 24)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert not np.allclose(noisy[i], noisy[j])

    def test_ensemble_axes_metadata(self):
        assert ScanNoiseTransform(1.0, 1e-6, 1e-4).ensemble_axes_metadata == []


# ---------------------------------------------------------------------------
# ScanNoiseTransform on lazy (dask-backed) images
# ---------------------------------------------------------------------------

_SCAN_NOISE_KWARGS = dict(
    dwell_time=1e-6, flyback_time=1e-4, max_frequency=500, num_components=20
)


def _scan_noise_oracle(image, rms_powers, seeds):
    """Distort `image` directly with the module-level helpers, one realisation
    per (rms_power, seed) pair, bypassing the ensemble/chunking machinery."""
    image = np.asarray(image)
    time = _pixel_times(
        _SCAN_NOISE_KWARGS["dwell_time"], _SCAN_NOISE_KWARGS["flyback_time"],
        image.shape,
    )
    out = np.zeros((len(rms_powers), len(seeds)) + image.shape, dtype=image.dtype)
    for i, rms_power in enumerate(rms_powers):
        for j, seed in enumerate(seeds):
            dx, dy = _make_displacement_field(
                time, _SCAN_NOISE_KWARGS["max_frequency"],
                _SCAN_NOISE_KWARGS["num_components"], rms_power, seed=int(seed),
            )
            out[i, j] = _apply_displacement_field(image, dx, dy)
    return out


@pytest.mark.parametrize("device", ["cpu", gpu])
class TestLazyScanNoise:
    def _images(self, device):
        # non-constant, so that a distortion actually changes the image
        return _to_device(make_images((16, 24)), device)

    def test_ensemble_shape(self, device):
        snt = ScanNoiseTransform(rms_power=5.0, seeds=0, samples=3, **_SCAN_NOISE_KWARGS)
        assert snt.ensemble_shape == (3,)
        snt = ScanNoiseTransform(
            rms_power=np.array([2.0, 5.0]), seeds=0, samples=3, **_SCAN_NOISE_KWARGS
        )
        assert snt.ensemble_shape == (2, 3)

    def test_images_scan_noise_lazy_equals_eager(self, device):
        images = self._images(device)
        eager = images.scan_noise(rms_power=5.0, seed=7, **_SCAN_NOISE_KWARGS)
        lazy = images.ensure_lazy().scan_noise(rms_power=5.0, seed=7, **_SCAN_NOISE_KWARGS)
        assert lazy.is_lazy and not eager.is_lazy
        oracle = _scan_noise_oracle(_to_cpu_array(images), [5.0], [7])[0]
        np.testing.assert_array_equal(_to_cpu_array(eager), oracle)
        np.testing.assert_array_equal(_to_cpu_array(lazy), oracle)

    @pytest.mark.parametrize("max_batch", ["auto", 1])
    def test_samples_lazy_equals_eager(self, device, max_batch):
        images = self._images(device)
        snt = ScanNoiseTransform(rms_power=5.0, seeds=0, samples=3, **_SCAN_NOISE_KWARGS)

        eager = snt.apply(images)
        lazy = snt.apply(images.ensure_lazy(), max_batch=max_batch)
        if max_batch == 1:
            # one image per chunk: the sample axis is split across chunks
            assert len(lazy.array.chunks[0]) == 3

        oracle = _scan_noise_oracle(_to_cpu_array(images), [5.0], snt.seeds.values)[0]
        np.testing.assert_array_equal(_to_cpu_array(eager), oracle)
        np.testing.assert_array_equal(_to_cpu_array(lazy), oracle)

    @pytest.mark.parametrize("max_batch", ["auto", 1, 2])
    def test_rms_power_distribution_lazy_equals_eager(self, device, max_batch):
        images = self._images(device)
        rms_powers = np.array([2.0, 5.0])
        snt = ScanNoiseTransform(
            rms_power=rms_powers, seeds=3, samples=3, **_SCAN_NOISE_KWARGS
        )

        eager = snt.apply(images)
        lazy = snt.apply(images.ensure_lazy(), max_batch=max_batch)
        assert lazy.shape == eager.shape == (2, 3, 16, 24)
        if max_batch != "auto":
            # the sample axis is split across chunks
            assert len(lazy.array.chunks[1]) > 1

        oracle = _scan_noise_oracle(_to_cpu_array(images), rms_powers, snt.seeds.values)
        np.testing.assert_array_equal(_to_cpu_array(eager), oracle)
        np.testing.assert_array_equal(_to_cpu_array(lazy), oracle)
