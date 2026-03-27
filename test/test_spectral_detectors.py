"""Tests for SpectralSlitDetector and SpectralAnnularDetector."""

import numpy as np
import pytest

import abtem
from abtem.detectors import SpectralAnnularDetector, SpectralSlitDetector
from abtem.measurements import DiffractionPatterns, momentum_resolved_spectrum


def _make_dp(n_energies=3, gpts=64, sampling=1.0, energy=300e3):
    """Create a simple DiffractionPatterns with an EnergyAxis for testing."""
    from abtem.core.axes import EnergyAxis, OrdinalAxis

    rng = np.random.default_rng(42)
    e_values = np.array([0.02, 0.05, 0.10])  # eV
    array = rng.random((n_energies, gpts, gpts)).astype(np.float32)

    energy_axis = EnergyAxis(values=e_values, units="eV", label="energy loss")
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
    SpectralAnnularDetector(outer=r, q_max=Q) must start at the same q.

    Note on endpoints: the slit samples q continuously via linspace (inclusive
    of q_max), while the annular sweeps discrete disk positions via arange
    (exclusive of q_max, so the last value is q_max - outer).  Both use the
    same q_max value to bound the output range.
    """
    r, Q = 2.0, 20.0
    dp = _make_dp(gpts=64, sampling=0.5)

    det_slit = SpectralSlitDetector(width=2 * r, q_max=Q)
    det_ann = SpectralAnnularDetector(outer=r, q_max=Q)

    spec_slit = momentum_resolved_spectrum(dp, det_slit)
    spec_ann = momentum_resolved_spectrum(dp, det_ann)

    # Both start at q=0
    assert spec_slit._q_values[0] == pytest.approx(spec_ann._q_values[0], abs=1e-6)
    # Slit reaches q_max; annular reaches q_max - outer (arange is exclusive)
    assert spec_slit._q_values[-1] == pytest.approx(Q, abs=1e-6)
    assert spec_ann._q_values[-1] == pytest.approx(Q - r, abs=1e-6)
    # All annular q_values are within [0, q_max)
    assert np.all(np.array(spec_ann._q_values) < Q + 1e-9)


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
