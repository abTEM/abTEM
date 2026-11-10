"""The ``angles`` configuration: an angle that you give is either wavelength * k (the
paraxial convention, today's behaviour) or the scattering angle arcsin(wavelength * k).

The oracle is a wave of plane waves at chosen spatial frequencies: which of them a
detector, aperture or crop includes follows from sin(angle) = wavelength * k by hand.
"""

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
from abtem.core.utils import get_dtype

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
    return abtem.Waves(
        psi.astype(get_dtype(complex=True)), extent=EXTENT, energy=ENERGY
    )


def detected(detector, device="cpu"):
    waves = plane_waves().copy_to_device(device)
    return np.asarray(asnumpy(detector.detect(waves).array))


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


def scattering_angle(n):
    """The scattering angle [mrad] of the plane wave at n / EXTENT."""
    return 1e3 * np.arcsin(n / EXTENT * energy2wavelength(ENERGY))


# The beams are in the xz plane, so the angle between two of them is the difference
# of their scattering angles: from the beam of 708.8 mrad (n = 30) the one of
# 1050 mrad (n = 40) is 341 mrad away and the one of 218.6 mrad (n = 10) 490 mrad.
CENTRE = scattering_angle(30)
TO_40 = scattering_angle(40) - CENTRE
TO_10 = CENTRE - scattering_angle(10)


def unit_intensity():
    """The detected intensity of a beam of amplitude 1."""
    with abtem.config.set({"angles": "paraxial"}):
        return float(detected(abtem.AnnularDetector(inner=0.0))) / (1 + 4 + 9 + 16)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("along", ["x", "y"])
def test_an_offset_detector_measures_the_angle_from_its_centre(along, device):
    # centred on the beam of 708.8 mrad, 470 mrad holds it and the beam 341 mrad
    # away (4 + 9), not the one 490 mrad away; shifting the paraxial region on the
    # grid put that one at arcsin(0.650 - 0.217) = 448 mrad, inside
    waves = plane_waves()
    offset = (CENTRE, 0.0)
    if along == "y":
        waves = abtem.Waves(waves.array.T.copy(), extent=EXTENT, energy=ENERGY)
        offset = (0.0, CENTRE)
    with abtem.config.set({"angles": "exact"}):
        detector = abtem.AnnularDetector(inner=0.0, outer=470.0, offset=offset)
        value = float(asnumpy(detector.detect(waves.copy_to_device(device)).array))
    assert 470.0 < TO_10 and 470.0 > TO_40
    assert value == pytest.approx(unit_intensity() * (4 + 9), rel=1e-4)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_bins_of_an_offset_detector_are_about_its_centre(device):
    # two radial bins of 300 mrad and two azimuthal ones, turned by 90 degrees so
    # that the beams (at azimuths 0 and 180 degrees about the centre) are inside
    # them: the centre beam (4) and the one 341 mrad away (9) at azimuth 0 are in
    # the second azimuthal bin, the one 490 mrad away (1) at 180 degrees in the first
    detector = abtem.SegmentedDetector(
        nbins_radial=2,
        nbins_azimuthal=2,
        inner=0.0,
        outer=600.0,
        rotation=np.pi / 2,
        offset=(CENTRE, 0.0),
    )
    with abtem.config.set({"angles": "exact"}):
        bins = detected(detector, device)
    np.testing.assert_allclose(bins / unit_intensity(), [[0, 4], [1, 9]], atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_sensitivity_of_an_offset_detector_is_at_the_angle_from_its_centre(
    device,
):
    # a sensitivity falling linearly from 1 at the centre to 0 at 600 mrad
    detector = abtem.AnnularDetector(
        inner=0.0,
        outer=600.0,
        offset=(CENTRE, 0.0),
        sensitivity=((0.0, 600.0), (1.0, 0.0)),
    )
    with abtem.config.set({"angles": "exact"}):
        value = float(detected(detector, device))
    expected = 4 + 9 * (1 - TO_40 / 600) + 1 * (1 - TO_10 / 600)
    assert value / unit_intensity() == pytest.approx(expected, rel=1e-3)


def test_an_offset_is_a_polar_angle_and_azimuth():
    from abtem.measurements import _exact_angles_about_centre

    sampling = (10.0, 10.0)
    gpts = (240, 240)
    k = [np.fft.fftfreq(n) * n * d / 1e3 for n, d in zip(gpts, sampling)]
    nx, ny = np.meshgrid(*k, indexing="ij")
    nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, None))
    propagating = np.hypot(nx, ny) <= 1.0
    for offset in ((600.0, 0.0), (600 / np.sqrt(2), 600 / np.sqrt(2)), (-200, 450)):
        alpha, _ = _exact_angles_about_centre(gpts, sampling, offset)
        # the centre pixel is the direction at |offset| from the axis
        centre = np.unravel_index(np.nanargmin(alpha), alpha.shape)
        assert 1e3 * np.arcsin(np.hypot(nx[centre], ny[centre])) == pytest.approx(
            np.hypot(*offset), abs=2 * sampling[0]
        )
        assert alpha[centre] == 0.0
        # and every angle is the one between the directions of the beam and centre,
        # to single precision, which loses up to 0.1 mrad next to wavelength * k = 1
        c = np.array([nx[centre], ny[centre], nz[centre]])
        reference = 1e3 * np.arccos(np.clip(nx * c[0] + ny * c[1] + nz * c[2], -1, 1))
        np.testing.assert_allclose(alpha[propagating], reference[propagating], atol=0.2)
        assert np.all(np.isnan(alpha[~propagating]))


def test_a_small_offset_is_the_paraxial_shift():
    from abtem.measurements import _annular_detector_mask

    masks = []
    for convention in ("paraxial", "exact"):
        with abtem.config.set({"angles": convention}):
            masks.append(
                _annular_detector_mask(
                    (96, 96), (0.5, 0.5), 2.0, 10.0, offset=(3.0, -2.5)
                )
            )
    np.testing.assert_array_equal(*masks)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_slit_length_is_in_the_scattering_angle(device):
    # 0 to 1000 mrad along x: the beams of 218.6 and 708.8 mrad (1 + 4), and not the
    # one of 1050 mrad, which is at wavelength * k = 0.867 and counts paraxially
    detector = abtem.SpectralSlitDetector(width=20.0, q_min=0.0, q_max=1000.0)
    results = {}
    for convention in ("paraxial", "exact"):
        with abtem.config.set({"angles": convention}):
            results[convention] = float(detected(detector, device)) / unit_intensity()
    assert results["paraxial"] == pytest.approx(1 + 4 + 9, rel=1e-4)
    assert results["exact"] == pytest.approx(1 + 4, rel=1e-4)


def test_the_slit_width_is_in_the_scattering_angle():
    # the beams along y, across a slit along x at q = 0: half of 1380 mrad is 690
    # mrad, at wavelength * k = sin(0.69) = 0.637, short of the beam at 0.650 that
    # it takes paraxially
    waves = plane_waves()
    waves = abtem.Waves(waves.array.T.copy(), extent=EXTENT, energy=ENERGY)
    detector = abtem.SpectralSlitDetector(width=1380.0, q_min=0.0, q_max=10.0)
    results = {}
    for convention in ("paraxial", "exact"):
        with abtem.config.set({"angles": convention}):
            results[convention] = float(detector.detect(waves).array) / unit_intensity()
    assert results["paraxial"] == pytest.approx(1 + 4, rel=1e-4)
    assert results["exact"] == pytest.approx(1, rel=1e-4)


def test_an_off_axis_slit_is_the_same_given_by_its_corners_or_its_geometry():
    # 500 to 900 mrad along x and +-50 mrad across: the beam of 708.8 mrad (4), and
    # paraxially also that at wavelength * k = 0.867 (9)
    by_corners = abtem.SpectralSlitDetector(corners=(500.0, 900.0, -50.0, 50.0))
    by_geometry = abtem.SpectralSlitDetector(
        width=100.0, q_min=0.0, q_max=400.0, offset=(500.0, 0.0)
    )
    waves = plane_waves()
    gpts, sampling = waves.gpts, waves.angular_sampling
    for convention, expected in (("paraxial", 4 + 9), ("exact", 4)):
        with abtem.config.set({"angles": convention}):
            np.testing.assert_array_equal(
                by_corners._mask(gpts, sampling), by_geometry._mask(gpts, sampling)
            )
            value = float(detected(by_corners)) / unit_intensity()
        assert value == pytest.approx(expected, rel=1e-4)
    with abtem.config.set({"angles": "exact"}):
        # the farthest corner, (sin 0.9, sin 0.05) on the grid
        _, outer = by_corners.angular_limits(waves)
    reach = np.hypot(np.sin(0.9), np.sin(0.05))
    assert outer == pytest.approx(1e3 * np.arcsin(reach))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_slit_spectrum_is_labelled_with_the_scattering_angle(device):
    # two peaks of weights 1 and 2 at the scattering angles 300 and 900 mrad along
    # x, at wavelength * k = sin(0.3) and sin(0.9), each pixel holding the density
    # times the pixel area. The strip integration of momentum_resolved_spectrum
    # gives these sums to about 0.5 % for this sampling in either convention.
    from abtem.core.axes import EnergyLossAxis

    gpts, grid_sampling, sigma = 480, 5.0, 10.0  # mrad on the grid
    wavelength = energy2wavelength(ENERGY)
    g = (np.arange(gpts) - gpts // 2) * grid_sampling
    gx, gy = np.meshgrid(g, g, indexing="ij")
    array = np.zeros((1, gpts, gpts))
    for weight, angle in ((1.0, 300.0), (2.0, 900.0)):
        r2 = (gx - 1e3 * np.sin(angle / 1e3)) ** 2 + gy**2
        array[0] += (
            weight * np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        ) * grid_sampling**2
    patterns = abtem.DiffractionPatterns(
        array.astype(get_dtype()),
        sampling=grid_sampling / wavelength / 1e3,
        fftshift=True,
        ensemble_axes_metadata=[EnergyLossAxis(values=(0.05,), units="eV")],
        metadata={"energy": ENERGY},
    ).copy_to_device(device)
    detector = abtem.SpectralSlitDetector(width=200.0, q_min=0.0, q_max=1200.0)
    with abtem.config.set({"angles": "exact"}):
        spectrum = abtem.momentum_resolved_spectrum(patterns, detector)
    values = np.asarray(asnumpy(spectrum.array), dtype=float)[:, 0]
    q = np.asarray(spectrum.q_values)
    lower, upper = q < 600.0, q >= 600.0
    np.testing.assert_allclose(
        (values[lower].sum(), values[upper].sum()), (1.0, 2.0), rtol=1e-2
    )
    # the peaks are at their scattering angles, to the q spacing there
    for peak, angle in ((lower, 300.0), (upper, 900.0)):
        at = np.flatnonzero(peak)[np.argmax(values[peak])]
        assert q[at] == pytest.approx(angle, abs=1.5 * (q[at + 1] - q[at - 1]))
    # q is uniform on the grid, from 0 to sin(1.2), not in the angle
    assert q[0] == 0.0 and q[-1] == pytest.approx(1200.0)
    steps = np.diff(q)
    assert steps[-1] > 2 * steps[0]


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


def test_the_spectral_annular_detector_shows_the_cones_it_detects():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    from matplotlib.path import Path

    from abtem.measurements import _annular_detector_mask

    waves = plane_waves()
    outer, angle = 250.0, 30.0
    # the last cone, at 1400 mrad, reaches beyond 90 degrees
    detector = abtem.SpectralAnnularDetector(
        outer=outer, q_min=0.0, q_max=1400.0, q_sampling=350.0, angle=angle
    )
    with abtem.config.set({"angles": "paraxial"}):
        ax = detector.show(waves)
    assert all(isinstance(p, Circle) and p.radius == outer for p in ax.patches)
    plt.close(ax.figure)

    with abtem.config.set({"angles": "exact"}):
        ax = detector.show(waves)
        outlines = [p.get_xy() for p in ax.patches]
        assert all(isinstance(p, Polygon) for p in ax.patches)
        plt.close(ax.figure)
        assert len(outlines) == 5

        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        sampling = waves.angular_sampling
        gpts = waves.gpts
        k = [np.fft.fftfreq(n) * n * d for n, d in zip(gpts, sampling)]
        gx, gy = np.meshgrid(*k, indexing="ij")
        pixels = np.stack([gx.ravel(), gy.ravel()], axis=1)
        for q, outline in zip(np.linspace(0.0, 1400.0, 5), outlines):
            # every point is outer from the centre direction, or on the edge of the
            # propagating beams where the cone runs past 90 degrees
            theta = q / 1e3
            centre = [np.sin(theta) * cos_a, np.sin(theta) * sin_a, np.cos(theta)]
            g = outline / 1e3
            on_edge = np.isclose(np.hypot(*g.T), 1.0)
            n = np.column_stack([g, np.sqrt(np.clip(1 - (g**2).sum(1), 0, None))])
            angles = 1e3 * np.arccos(np.clip(n @ centre, -1, 1))
            np.testing.assert_allclose(angles[~on_edge], outer, atol=1e-6)
            assert np.all(angles[on_edge] <= outer + 1e-6)
            assert on_edge.any() == (q + outer > 1e3 * np.pi / 2)

            # and it holds the pixels the detector integrates, up to some of the
            # pixels its edge crosses (at most 14 here; a circle of radius outer
            # about the centre misses 34 to 339)
            mask = _annular_detector_mask(
                gpts, sampling, 0.0, outer, offset=(q * cos_a, q * sin_a)
            ).ravel()
            inside = Path(outline).contains_points(pixels)
            edge_pixels = 2 * np.pi * outer / min(sampling)
            assert (mask != inside).sum() < edge_pixels / 3


def test_an_offset_centre_beyond_the_grid_has_no_pixel():
    # the centre at 600 mrad is beyond a grid that reaches 320 mrad: no pixel is at
    # angle 0 (rounding its index used to wrap round to the other side of the grid,
    # which was then counted as the centre), and every angle is that from the centre
    from abtem.measurements import _exact_angles_about_centre

    sampling, gpts = (10.0, 10.0), (64, 64)
    k = [np.fft.fftfreq(n) * n * d / 1e3 for n, d in zip(gpts, sampling)]
    nx, ny = np.meshgrid(*k, indexing="ij")
    nz = np.sqrt(1 - nx**2 - ny**2)
    for offset in ((600.0, 0.0), (-100.0, 640.0)):
        alpha, _ = _exact_angles_about_centre(gpts, sampling, offset)
        radius = np.hypot(*offset)
        g = np.array(offset) / radius * np.sin(radius / 1e3)
        g = np.round(g * 1e3 / np.array(sampling)) * np.array(sampling) / 1e3
        c = np.append(g, np.sqrt(1 - g @ g))
        reference = 1e3 * np.arccos(np.clip(nx * c[0] + ny * c[1] + nz * c[2], -1, 1))
        assert alpha.min() > 100.0
        np.testing.assert_allclose(alpha, reference, atol=0.2)
