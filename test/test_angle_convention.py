"""The ``angles`` configuration: an angle that you give is either wavelength * k (the
paraxial convention, today's behaviour) or the scattering angle arcsin(wavelength * k).

The oracle is a wave of plane waves at chosen spatial frequencies: which of them a
detector, aperture or crop includes follows from sin(angle) = wavelength * k by hand.
"""

import os

import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.core.backend import asnumpy
from abtem.core.energy import (
    energy2wavelength,
    to_grid_angle,
    to_scattering_angle,
    to_scattering_angle_limit,
)

# the paraxial runs give angles of up to 1000 mrad on purpose
pytestmark = pytest.mark.filterwarnings("ignore:An angle above 250 mrad")

ENERGY = 200.0
EXTENT = 40.0
GPTS = 192
# plane waves along x at n / EXTENT with amplitude 1, 2, 3, 4: with the wavelength of
# 200 eV (0.867 Å) wavelength * k = 0.217, 0.650, 0.867, 1.084, so the scattering
# angles are 218.6, 708.8, 1050 mrad and the last beam is evanescent.
BEAMS = {10: 1.0, 30: 2.0, 40: 3.0, 50: 4.0}


def plane_waves():
    x = np.arange(GPTS) * EXTENT / GPTS
    psi = sum(
        a * np.exp(2j * np.pi * n / EXTENT * x)[:, None] * np.ones(GPTS)[None]
        for n, a in BEAMS.items()
    )
    return abtem.Waves(psi.astype(np.complex64), extent=EXTENT, energy=ENERGY)


def detected(detector):
    return np.asarray(detector.detect(plane_waves()).array)


def test_the_default_is_paraxial():
    assert abtem.config.get("angles") == "paraxial"


def test_helpers_are_inverse_and_the_paraxial_one_is_the_identity():
    angles = np.array([0.0, 30.0, 500.0, 1000.0, 1500.0])
    with abtem.config.set({"angles": "paraxial"}):
        np.testing.assert_array_equal(to_grid_angle(angles), angles)
        np.testing.assert_array_equal(to_scattering_angle(angles), angles)
    with abtem.config.set({"angles": "exact"}):
        grid = to_grid_angle(angles)
        np.testing.assert_allclose(grid, 1e3 * np.sin(angles / 1e3))
        np.testing.assert_allclose(to_scattering_angle(grid), angles)


def test_the_exact_angle_beyond_90_degrees_and_beams_beyond_1_are_limits():
    with abtem.config.set({"angles": "exact"}):
        # infinite and larger than 90 degrees angles include every propagating beam
        assert to_grid_angle(np.inf) == pytest.approx(1e3)
        assert to_grid_angle(2000.0) == pytest.approx(1e3)
        # an evanescent beam has no scattering angle, but a limit stays a limit
        assert np.isnan(to_scattering_angle(1200.0))
        assert to_scattering_angle_limit(1200.0) == pytest.approx(1e3 * np.pi / 2)


def test_the_limits_of_a_pattern_beyond_90_degrees_are_90_degrees_on_both_sides():
    # the pattern reaches wavelength * k = ±2.08
    patterns = plane_waves().diffraction_patterns(max_angle="full")
    with abtem.config.set({"angles": "exact"}):
        assert to_scattering_angle_limit(-1200.0) == pytest.approx(-1e3 * np.pi / 2)
        limits = patterns.angular_limits
    np.testing.assert_allclose(limits, [(-1e3 * np.pi / 2, 1e3 * np.pi / 2)] * 2)


def test_the_large_angle_warning_points_at_the_line_that_gives_the_angle():
    # each call reaches the check at a different depth inside abTEM
    patterns = plane_waves().diffraction_patterns(max_angle="full")
    calls = (
        lambda: abtem.AnnularDetector(inner=300.0, outer=500.0),
        lambda: abtem.PixelatedDetector(max_angle=400.0),
        lambda: abtem.Probe(energy=ENERGY, semiangle_cutoff=300.0),
        lambda: patterns.crop(max_angle=400.0),
    )
    with abtem.config.set({"angles": "paraxial"}):
        for give_an_angle in calls:
            with pytest.warns(UserWarning, match="'angles' configuration") as record:
                give_an_angle()
            for warning in record:
                if "'angles' configuration" in str(warning.message):
                    assert os.path.samefile(warning.filename, __file__)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_exact_convention_detects_on_every_device(device):
    # the masks take the arcsin with the device's array module
    waves = plane_waves().copy_to_device(device)

    def measure(detector):
        return np.asarray(asnumpy(detector.detect(waves).array)).item()

    with abtem.config.set({"angles": "exact"}):
        propagating = measure(abtem.AnnularDetector(inner=0.0))
        ring = measure(abtem.AnnularDetector(inner=500.0, outer=1000.0))
        weighted = measure(
            abtem.AnnularDetector(
                inner=500.0, outer=1000.0, sensitivity=lambda alpha: alpha / 1e3
            )
        )
    # of the propagating beams (amplitudes 1, 2, 3) only the second is in the ring
    angle = np.arcsin(energy2wavelength(ENERGY) * 30 / EXTENT)
    assert ring == pytest.approx(propagating * 4 / 14, rel=1e-4)
    assert weighted == pytest.approx(ring * angle, rel=1e-4)


def test_the_angles_of_an_ebsd_detector_grid_are_the_exact_angles():
    # the grid is uniform in sin(angle), so a pixel is at arcsin(wavelength * |k|)
    from abtem.ebsd import BackscatterDetector

    energy = 20e3
    with abtem.config.set({"angles": "exact"}):
        detector = BackscatterDetector(max_angle=700.0, gpts=31)
    patterns = abtem.DiffractionPatterns(
        np.zeros((31, 31), dtype=np.float32),
        sampling=detector.reciprocal_sampling(energy),
        fftshift=True,
        metadata={"energy": energy},
    )
    polar = 1e3 * np.arccos(detector.directions[:, 2].reshape(31, 31))

    # coordinates in the configured precision: to 1 µrad
    x, y = patterns.scattering_angles()
    np.testing.assert_allclose(np.abs(x), polar[:, 15], atol=1e-3)
    np.testing.assert_allclose(np.abs(y), polar[15, :], atol=1e-3)

    grid_x, grid_y = patterns.angular_coordinates
    radius = np.hypot(grid_x[:, None], grid_y[None])
    np.testing.assert_allclose(1e3 * np.arcsin(radius / 1e3), polar, atol=1e-3)

    with abtem.config.set({"angles": "exact"}):
        assert max(patterns.max_angles) == pytest.approx(polar[0, 15], abs=1e-3)


def test_an_unknown_convention_is_refused():
    with abtem.config.set({"angles": "degrees"}):
        with pytest.raises(ValueError, match="paraxial"):
            to_grid_angle(30.0)


def test_annular_detector_selects_the_beams_by_the_convention():
    # 500-1000 mrad: the beam of 708.8 mrad (amplitude 2) in the exact convention;
    # in the paraxial one the limits are wavelength * k, which include the beams at
    # 0.650 and 0.867 (amplitudes 2 and 3).
    with abtem.config.set({"angles": "paraxial"}):
        # without an outer limit the paraxial convention includes all four beams
        unit = detected(abtem.AnnularDetector(inner=0.0)) / (1 + 4 + 9 + 16)
        ring = detected(abtem.AnnularDetector(inner=500.0, outer=1000.0))
    assert ring == pytest.approx(unit * (4 + 9), rel=1e-4)

    with abtem.config.set({"angles": "exact"}):
        ring = detected(abtem.AnnularDetector(inner=500.0, outer=1000.0))
    assert ring == pytest.approx(unit * 4, rel=1e-4)


def test_segmented_detector_bins_are_uniform_in_the_scattering_angle():
    # the beam of 708.8 mrad is in the bin 7 (700-800 mrad) of ten bins to 1000 mrad
    # in the exact convention, and wavelength * k = 650 puts it in bin 6 otherwise.
    detector = abtem.SegmentedDetector(
        nbins_radial=10, nbins_azimuthal=1, inner=0.0, outer=1000.0
    )
    for convention, expected_bins in (("paraxial", (2, 6, 8)), ("exact", (2, 7))):
        with abtem.config.set({"angles": convention}):
            bins = detected(detector)[:, 0]
        assert tuple(np.flatnonzero(bins > 1e-3 * bins.sum())) == expected_bins


def test_the_detector_regions_follow_the_convention_after_a_switch():
    # the bin geometry is cached: it must be keyed on the convention
    detector = abtem.AnnularDetector(inner=500.0, outer=1000.0)
    results = []
    for convention in ("paraxial", "exact", "paraxial"):
        with abtem.config.set({"angles": convention}):
            results.append(float(detected(detector)))
    assert results[0] == pytest.approx(results[2])
    assert results[0] > 2 * results[1]


def test_the_default_outer_limit_is_the_propagating_beams_in_the_exact_convention():
    # without an outer limit every pixel up to the cutoff counts in the paraxial
    # convention, including the evanescent beam, but only the propagating ones in the
    # exact convention
    with abtem.config.set({"angles": "paraxial"}):
        everything = detected(abtem.AnnularDetector(inner=0.0))
    with abtem.config.set({"angles": "exact"}):
        propagating = detected(abtem.AnnularDetector(inner=0.0))
    assert propagating / everything == pytest.approx((1 + 4 + 9) / 30, rel=1e-4)


def test_pixelated_detector_crops_to_the_scattering_angle():
    # the pixels are 21.7 mrad in wavelength * k; 700 mrad is 644 in that coordinate
    # in the exact convention
    waves = plane_waves()
    gpts = {}
    for convention in ("paraxial", "exact"):
        with abtem.config.set({"angles": convention}):
            patterns = abtem.PixelatedDetector(max_angle=700.0).detect(waves)
        gpts[convention] = patterns.shape[-2:]
    sampling = energy2wavelength(ENERGY) / EXTENT * 1e3

    def expected(grid_angle):
        # the pixels cover the angle, in an even number as for the 192 of the waves
        n = 2 * int(np.ceil(grid_angle / sampling)) + 1
        return (n + n % 2,) * 2

    assert gpts["paraxial"] == expected(700.0)
    assert gpts["exact"] == expected(1e3 * np.sin(0.7))
    assert gpts["exact"] < gpts["paraxial"]


def test_aperture_cuts_at_the_sine_of_its_semiangle():
    # an 800 mrad aperture is open up to wavelength * k = sin(0.8) = 0.717
    def edge():
        probe = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=800.0, extent=EXTENT, gpts=GPTS, soft=False
        )
        spectrum = np.abs(np.fft.fft2(probe.build().array))
        k = np.abs(np.fft.fftfreq(GPTS, EXTENT / GPTS))
        radius = np.hypot(k[:, None], k[None])
        return (
            energy2wavelength(ENERGY) * radius[spectrum > 1e-6 * spectrum.max()].max()
        )

    with abtem.config.set({"angles": "paraxial"}):
        assert edge() == pytest.approx(0.8, abs=0.01)
    with abtem.config.set({"angles": "exact"}):
        assert edge() == pytest.approx(np.sin(0.8), abs=0.01)


def test_diffraction_patterns_get_honest_axes_and_the_exact_angles():
    waves = plane_waves()
    with abtem.config.set({"angles": "paraxial"}):
        assert "scattering angle" in waves.reciprocal_space_axes_metadata[0].label
    with abtem.config.set({"angles": "exact"}):
        assert "λk" in waves.reciprocal_space_axes_metadata[0].label

    patterns = waves.diffraction_patterns(max_angle="full")
    x, _ = patterns.scattering_angles()
    grid_x, _ = patterns.angular_coordinates
    inside = np.abs(grid_x) <= 1e3
    # in either convention: arcsin of the axis, NaN beyond wavelength * k = 1
    np.testing.assert_allclose(x[inside], 1e3 * np.arcsin(grid_x[inside] / 1e3))
    assert np.all(np.isnan(x[~inside]))


def test_a_large_paraxial_angle_warns_and_an_exact_one_does_not():
    with abtem.config.set({"angles": "paraxial"}):
        with pytest.warns(UserWarning, match="'angles' configuration"):
            abtem.AnnularDetector(inner=300.0, outer=500.0)
        # no warning for a size derived by abTEM, or for the default of no limit
        abtem.AnnularDetector(inner=30.0)
    with abtem.config.set({"angles": "exact"}):
        abtem.AnnularDetector(inner=300.0, outer=500.0)


def test_the_spectral_slit_detector_refuses_the_exact_convention():
    detector = abtem.SpectralSlitDetector(q_min=0.0, q_max=100.0, width=10.0)
    with abtem.config.set({"angles": "exact"}):
        with pytest.raises(NotImplementedError, match="SpectralSlitDetector"):
            detector.detect(plane_waves())


def test_the_prism_eels_collection_angle_is_in_the_configured_convention():
    """``collection_angle`` is an angle that you give: the guard that refuses one
    smaller than the detector compares both in the configured convention."""
    from abtem.inelastic.core_loss import _check_collection_angle

    s_matrix = abtem.SMatrix(
        semiangle_cutoff=100.0, energy=ENERGY, extent=20.0, gpts=64
    )
    detector = abtem.AnnularDetector(inner=0.0, outer=800.0)
    wavelength = energy2wavelength(ENERGY)
    k = [np.fft.fftfreq(n, e / n) for n, e in zip(s_matrix.gpts, s_matrix.extent)]
    grid = 1e3 * wavelength * np.hypot(k[0][:, None], k[1][None])
    with abtem.config.set({"angles": "exact"}):
        alpha = to_scattering_angle(grid)
        # the largest scattering angle of a beam the detector collects
        edge = float(alpha[alpha < 800.0].max())
        with pytest.raises(ValueError, match="collection_angle"):
            _check_collection_angle(s_matrix, [detector], edge - 1e-3)
        _check_collection_angle(s_matrix, [detector], edge + 1e-3)
