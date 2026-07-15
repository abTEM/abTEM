"""Tests for SpectralSlitDetector and SpectralAnnularDetector."""

import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.detectors import (
    SpectralAnnularDetector,
    SpectralSlitDetector,
    _slit_detector_mask,
)
from abtem.measurements import (
    DiffractionPatterns,
    MomentumResolvedSpectrum,
    momentum_resolved_spectrum,
)


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


# ---- MomentumResolvedSpectrum.show() explode ordering -----------------------


def _make_multiaxis_spectrum(seed=0):
    """A MomentumResolvedSpectrum with 3 distinct-size ensemble axes, so a
    positional (rather than axis-identity) index mix-up is detectable."""
    rng = np.random.default_rng(seed)
    q_values = np.linspace(0, 10, 5)
    e_values = np.linspace(-0.1, 0.1, 6)
    array = rng.random((2, 3, 4, 5, 6))
    ensemble_axes = [
        OrdinalAxis(label="axis0", values=(0, 1)),
        OrdinalAxis(label="axis1", values=(0, 1, 2)),
        OrdinalAxis(label="axis2", values=(0, 1, 2, 3)),
    ]
    return MomentumResolvedSpectrum(
        array,
        q_values=q_values,
        e_values=e_values,
        ensemble_axes_metadata=ensemble_axes,
    ), array


def _panel_arrays(spec_array, ensemble_shape, explode_axes):
    """Ground-truth panel data for the given explode order, computed directly
    from the backing array rather than through show()."""
    import itertools

    n_ensemble = len(ensemble_shape)
    grid_sizes = [ensemble_shape[a] for a in explode_axes]
    indices = list(itertools.product(*[range(s) for s in grid_sizes]))
    panels = []
    for idx in indices:
        axis_to_value = dict(zip(explode_axes, idx))
        full = tuple(axis_to_value.get(d, 0) for d in range(n_ensemble))
        panels.append(spec_array[full])
    return panels


@pytest.mark.filterwarnings("ignore:This figure includes Axes")
@pytest.mark.parametrize("explode_axes", [(0, 2), (2, 0), (0, 1, 2), (2, 1, 0)])
def test_show_explode_sequence_any_order(explode_axes):
    """explode as a sequence of axis indices must plot each panel's own data
    regardless of the order the axes are listed in -- not the axis positioned
    at the same index as it appears in the (ascending) array shape."""
    import matplotlib

    matplotlib.use("Agg")
    spec, array = _make_multiaxis_spectrum()
    ensemble_shape = (2, 3, 4)

    fig, axes = spec.show(explode=list(explode_axes))
    expected_panels = _panel_arrays(array, ensemble_shape, explode_axes)

    axes_flat = axes.flatten()
    n = len(expected_panels)
    for k, expected in enumerate(expected_panels):
        plotted = axes_flat[k].collections[0].get_array().reshape(
            expected.shape[1], expected.shape[0]
        )
        np.testing.assert_allclose(plotted, expected.T)
    for k in range(n, len(axes_flat)):
        assert not axes_flat[k].get_visible()


@pytest.mark.filterwarnings("ignore:This figure includes Axes")
def test_show_explode_true_matches_full_range():
    """explode=True must be equivalent to exploding every ensemble axis."""
    import matplotlib

    matplotlib.use("Agg")
    spec, array = _make_multiaxis_spectrum()
    ensemble_shape = (2, 3, 4)

    fig_true, axes_true = spec.show(explode=True)
    fig_all, axes_all = spec.show(explode=[0, 1, 2])
    assert axes_true.shape == axes_all.shape


def test_show_warns_on_ensemble_collapse():
    """A non-exploded show() on an object with ensemble axes must warn that
    only member (0, ..., 0) is shown."""
    import matplotlib

    matplotlib.use("Agg")
    spec, _ = _make_multiaxis_spectrum()
    with pytest.warns(UserWarning, match="showing member"):
        spec.show()
