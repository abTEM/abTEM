"""Tests for SpectralSlitDetector and SpectralAnnularDetector."""

import numpy as np
import pytest

import abtem
from abtem.detectors import (
    SpectralAnnularDetector,
    SpectralSlitDetector,
    _slit_detector_mask,
)
from abtem.measurements import DiffractionPatterns, momentum_resolved_spectrum


def _make_dp(n_energies=3, gpts=64, sampling=1.0, energy=300e3):
    """Create a simple DiffractionPatterns with an EnergyLossAxis for testing."""
    from abtem.core.axes import EnergyLossAxis, OrdinalAxis

    rng = np.random.default_rng(42)
    e_values = np.array([0.02, 0.05, 0.10])  # eV
    array = rng.random((n_energies, gpts, gpts)).astype(np.float32)

    energy_axis = EnergyLossAxis(values=e_values, units="eV")
    metadata = {"energy": energy, "label": "intensity", "units": "arb. unit"}

    return DiffractionPatterns(
        array,
        ensemble_axes_metadata=[energy_axis],
        metadata=metadata,
        sampling=sampling,
        fftshift=True,
    )


# ---- SpectralSlitDetector: geometry vs corners mode -------------------------


def test_slit_geometry_and_corners_same_mask():
    """corners=(0, Q, -W/2, W/2) must produce the same physical mask as
    geometry mode width=W, q_min=0, q_max=Q, angle=0."""
    Q, W = 20.0, 4.0

    geom = SpectralSlitDetector(width=W, q_min=0.0, q_max=Q, angle=0.0)
    corn = SpectralSlitDetector(corners=(0.0, Q, -W / 2, W / 2))

    assert geom._corners == pytest.approx(corn._corners, abs=1e-10)


def test_slit_geometry_and_corners_same_spectrum():
    """corners and geometry mode must return identical momentum_resolved_spectrum."""
    Q, W = 15.0, 3.0
    dp = _make_dp(gpts=64, sampling=0.5)

    det_geom = SpectralSlitDetector(width=W, q_min=0.0, q_max=Q, angle=0.0)
    det_corn = SpectralSlitDetector(corners=(0.0, Q, -W / 2, W / 2))

    spec_geom = momentum_resolved_spectrum(dp, det_geom)
    spec_corn = momentum_resolved_spectrum(dp, det_corn)

    np.testing.assert_allclose(
        spec_geom.array, spec_corn.array, rtol=1e-5,
        err_msg="geometry and corners modes give different spectra"
    )
    np.testing.assert_allclose(
        spec_geom._q_values, spec_corn._q_values,
        err_msg="geometry and corners modes give different q_values"
    )


def test_slit_q_axis_starts_at_zero():
    """With q_min=0 (default), the first q-value of the spectrum must be 0."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralSlitDetector(width=3.0, q_max=15.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(0.0, abs=1e-6)


def test_slit_q_min_nonzero_excludes_origin():
    """With q_min > 0, q=0 must not appear in the q-axis."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralSlitDetector(width=3.0, q_min=5.0, q_max=20.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(5.0, abs=1e-6)
    assert spec._q_values[-1] == pytest.approx(20.0, abs=1e-6)


# ---- SpectralAnnularDetector ------------------------------------------------


def test_annular_q_axis_starts_at_zero():
    """With q_min=0 (default), the first q-value must be 0."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralAnnularDetector(outer=2.0, q_max=15.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(0.0, abs=1e-6)


def test_annular_q_min_nonzero():
    """With q_min > 0, the first q-value must equal q_min."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralAnnularDetector(outer=2.0, q_min=4.0, q_max=16.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(4.0, abs=1e-6)


# ---- Consistent q-range between slit and annular ----------------------------


def test_slit_and_annular_same_q_range():
    """SpectralSlitDetector(width=2*r, q_max=Q) and
    SpectralAnnularDetector(outer=r, q_max=Q) must span the same q range,
    both starting at q=0 and ending at q=Q.
    """
    r, Q = 2.0, 20.0
    dp = _make_dp(gpts=64, sampling=0.5)

    det_slit = SpectralSlitDetector(width=2 * r, q_max=Q)
    det_ann = SpectralAnnularDetector(outer=r, q_max=Q)

    spec_slit = momentum_resolved_spectrum(dp, det_slit)
    spec_ann = momentum_resolved_spectrum(dp, det_ann)

    # Both start at q=0
    assert spec_slit._q_values[0] == pytest.approx(spec_ann._q_values[0], abs=1e-6)
    # Both reach q_max as the last point
    assert spec_slit._q_values[-1] == pytest.approx(Q, abs=1e-6)
    assert spec_ann._q_values[-1] == pytest.approx(Q, abs=1e-6)


# ---- q_sampling reduces the number of q-points ------------------------------


def test_annular_q_sampling_reduces_points():
    """q_sampling > outer should produce fewer q-points than the default."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det_default = SpectralAnnularDetector(outer=1.0, q_max=20.0)
    det_coarse = SpectralAnnularDetector(outer=1.0, q_max=20.0, q_sampling=4.0)

    spec_default = momentum_resolved_spectrum(dp, det_default)
    spec_coarse = momentum_resolved_spectrum(dp, det_coarse)

    assert len(spec_coarse._q_values) < len(spec_default._q_values)
    # Both reach the same endpoints
    assert spec_coarse._q_values[0] == pytest.approx(0.0, abs=1e-6)
    assert spec_coarse._q_values[-1] == pytest.approx(20.0, abs=1e-6)


def test_slit_q_sampling_reduces_points():
    """q_sampling should bin the line profile, producing fewer q-points."""
    dp = _make_dp(gpts=256, sampling=0.05)
    det_default = SpectralSlitDetector(width=3.0, q_max=6.0)
    det_coarse = SpectralSlitDetector(width=3.0, q_max=6.0, q_sampling=2.0)

    spec_default = momentum_resolved_spectrum(dp, det_default)
    spec_coarse = momentum_resolved_spectrum(dp, det_coarse)

    assert len(spec_coarse._q_values) < len(spec_default._q_values)


def test_annular_q_sampling_none_matches_default():
    """q_sampling=None should be identical to omitting it (default=outer)."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det_a = SpectralAnnularDetector(outer=2.0, q_max=20.0)
    det_b = SpectralAnnularDetector(outer=2.0, q_max=20.0, q_sampling=None)

    spec_a = momentum_resolved_spectrum(dp, det_a)
    spec_b = momentum_resolved_spectrum(dp, det_b)

    np.testing.assert_allclose(spec_a.array, spec_b.array)
    assert spec_a._q_values == spec_b._q_values


# ---- show() should not raise ------------------------------------------------


def test_spectrum_show_no_warning():
    """show() should not emit a pcolormesh monotonicity warning."""
    import warnings
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralSlitDetector(width=3.0, q_max=15.0)
    spec = momentum_resolved_spectrum(dp, det)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        import matplotlib
        matplotlib.use("Agg")
        spec.show()


# ---- rotated-slit mask correctness -------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 90.0, 137.0])
def test_slit_mask_area_matches_true_rectangle_at_any_angle(angle):
    """The detector mask must integrate the true rotated rectangle, not its
    axis-aligned bounding box (which is larger for any non-90-degree-multiple
    angle and only coincides with the true area at angle=0/90/180/270)."""
    gpts = (512, 512)
    sampling = (0.2, 0.2)
    extent, width = 40.0, 4.0

    det = SpectralSlitDetector(width=width, q_min=0.0, q_max=extent, angle=angle)
    mask = _slit_detector_mask(
        gpts,
        sampling,
        center=det._center,
        angle=det._angle,
        extent=det._extent,
        width=det._width,
        fftshift=True,
    )

    pixel_area = sampling[0] * sampling[1]
    true_area = extent * width
    aabb_area = (
        (det._corners[1] - det._corners[0]) * (det._corners[3] - det._corners[2])
    )

    mask_area = mask.sum() * pixel_area

    # Should match the true rectangle area (allowing for pixel discretization),
    # not the (generally much larger) axis-aligned bounding box.
    assert mask_area == pytest.approx(true_area, rel=0.1)
    if angle % 90 != 0:
        assert aabb_area > true_area * 1.5  # sanity: AABB is genuinely bigger here
        assert mask_area < aabb_area * 0.5  # would fail against the old AABB bug


def test_slit_corners_mode_rejects_conflicting_geometry_params():
    """corners is documented as incompatible with offset/angle/q_min/q_max/width;
    all five must actually be validated, not just q_max/width."""
    corners = (-10.0, 10.0, -2.0, 2.0)

    SpectralSlitDetector(corners=corners)  # baseline: corners alone is fine

    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, angle=45.0)
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, q_min=5.0)
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, offset=(1.0, 0.0))
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, q_max=10.0)
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, width=5.0)
