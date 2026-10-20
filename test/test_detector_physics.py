"""Independent-oracle tests for the radial detectors.

Every expected value in this file comes from first principles, never from the
detector's own mask/bin helpers:

* Pixel scattering angles are rebuilt here from ``np.fft.fftfreq`` and a
  relativistic wavelength computed from CODATA constants (``_wavelength``),
  i.e. ``alpha = lambda * |k|`` with ``k`` on the unshifted FFT grid.
* A Kronecker-delta wave function has a Fourier transform of unit modulus at
  every frequency (numpy's unnormalized forward convention, which abTEM uses
  for waves whose metadata carries no ``"normalization"`` key), so its
  diffraction pattern is exactly 1 in every pixel and any integrated signal
  is an exact pixel count.
* A discrete plane wave ``exp(2 pi i (i0 m / Nx + j0 n / Ny))`` has a Fourier
  transform that is ``Nx * Ny`` at frequency index ``(i0, j0)`` and zero
  elsewhere, so its diffraction pattern is a single bright pixel of intensity
  ``(Nx * Ny)**2`` at a known ``(kx, ky)``.

Edge conventions: annular / radial bins cover ``inner <= alpha < outer``, and
radial bin ``i`` of a FlexibleAnnularDetector covers
``[inner + i * step, inner + (i + 1) * step)``. Every geometry below is
asserted to keep all pixels at least ``EDGE_MARGIN`` (relative) away from every
edge, which makes the pixel counts exact.

Grids are deliberately non-square with anisotropic sampling (and one odd
axis) so that swapped or mis-scaled x/y handling shows up.
"""

import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.core.backend import asnumpy
from abtem.core.utils import get_dtype
from abtem.waves import Waves

ENERGY = 200e3  # [eV]
GPTS = (160, 147)  # even x odd, non-square
SAMPLING = (0.1, 0.143)  # anisotropic real-space sampling [Å]
N_PIXELS = GPTS[0] * GPTS[1]
# Relative clearance required of every pixel from every bin edge: 100x the
# float32 rounding of the angle grid (~1e-7) and 2000x the difference between
# the CODATA wavelength below and abTEM's own (~5e-9).
EDGE_MARGIN = 1e-5


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------


def _wavelength(energy: float) -> float:
    """Relativistic electron wavelength [Å] from CODATA 2018 constants:
    lambda = h c / sqrt(E (2 m c^2 + E))."""
    h = 6.62607015e-34
    c = 299792458.0
    m = 9.1093837015e-31
    e = 1.602176634e-19
    E = energy * e
    return h * c / np.sqrt(E * (2 * m * c**2 + E)) * 1e10


def _pixel_angles(gpts=GPTS, sampling=SAMPLING, energy=ENERGY):
    """Scattering angles of every pixel of the *unshifted* FFT grid.

    Returns (ax, ay) [mrad], broadcast to ``gpts``: ``ax = lambda * kx`` with
    ``kx = fftfreq(Nx, d=dx)`` [1/Å].
    """
    lam = _wavelength(energy) * 1e3
    kx = np.fft.fftfreq(gpts[0], d=sampling[0])
    ky = np.fft.fftfreq(gpts[1], d=sampling[1])
    ax = np.broadcast_to(lam * kx[:, None], gpts)
    ay = np.broadcast_to(lam * ky[None, :], gpts)
    return ax, ay


def _angular_sampling(gpts=GPTS, sampling=SAMPLING, energy=ENERGY):
    """Pixel size in scattering angle [mrad]: lambda / (N * d)."""
    lam = _wavelength(energy) * 1e3
    return lam / (gpts[0] * sampling[0]), lam / (gpts[1] * sampling[1])


def _assert_clear_of_edges(values, edges, rel=EDGE_MARGIN, atol=0.0):
    """Precondition: no pixel lies (numerically) on any edge, so the pixel
    count does not depend on float rounding in the code under test."""
    values = np.asarray(values)
    for edge in edges:
        tol = max(rel * abs(edge), atol)
        if tol == 0.0:
            continue
        closest = np.min(np.abs(values - edge))
        assert closest > tol, (
            f"test geometry is ambiguous: a pixel lies {closest:.3g} from the "
            f"edge {edge}; pick another value"
        )


def _to_device(waves, device):
    return waves.to_gpu() if device == "gpu" else waves


def _delta_waves(device="cpu"):
    """Waves whose diffraction pattern is exactly 1 in every pixel."""
    array = np.zeros(GPTS, dtype=get_dtype(complex=True))
    array[0, 0] = 1.0
    return _to_device(Waves(array, energy=ENERGY, sampling=SAMPLING), device)


def _plane_wave_array(i0, j0, gpts=GPTS):
    m = np.arange(gpts[0])[:, None]
    n = np.arange(gpts[1])[None, :]
    phase = 2 * np.pi * (i0 * m / gpts[0] + j0 * n / gpts[1])
    return np.exp(1j * phase).astype(get_dtype(complex=True))


def _plane_waves(i0, j0, device="cpu"):
    """Waves whose diffraction pattern is (Nx Ny)^2 at frequency index
    (i0, j0) and zero elsewhere."""
    waves = Waves(_plane_wave_array(i0, j0), energy=ENERGY, sampling=SAMPLING)
    return _to_device(waves, device)


def _random_waves(device="cpu", seed=0):
    rng = np.random.default_rng(seed)
    array = (rng.standard_normal(GPTS) + 1j * rng.standard_normal(GPTS)).astype(
        get_dtype(complex=True)
    )
    return _to_device(Waves(array, energy=ENERGY, sampling=SAMPLING), device)


def _values(measurement):
    measurement = measurement.compute() if measurement.is_lazy else measurement
    return np.asarray(asnumpy(measurement.array), dtype=np.float64)


_ds = _angular_sampling()

ANNULAR_CASES = [
    # (inner, outer, offset in whole pixels); radii off pixel boundaries.
    (0.0, 20.3, (0, 0)),
    (12.7, 41.9, (0, 0)),
    (7.25, 55.1, (0, 0)),
    (0.0, 17.6, (3, -2)),
    (10.4, 30.2, (-4, 5)),
]


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("inner, outer, offset_pixels", ANNULAR_CASES)
def test_annular_detector_counts_pixels_in_annulus(device, inner, outer, offset_pixels):
    """Uniform unit pattern -> signal == number of pixels with
    inner <= |alpha - offset| < outer (hand-counted from fftfreq)."""
    # Offsets are whole pixels so the (nearest-pixel) offset handling is not
    # itself under test here.
    offset = (offset_pixels[0] * _ds[0], offset_pixels[1] * _ds[1])
    ax, ay = _pixel_angles()
    alpha = np.hypot(ax - offset[0], ay - offset[1])
    _assert_clear_of_edges(alpha, (inner, outer))

    expected = np.count_nonzero((alpha >= inner) & (alpha < outer))
    assert expected > 50  # a meaningful number of pixels

    detector = abtem.AnnularDetector(inner=inner, outer=outer, offset=offset)
    signal = _values(detector.detect(_delta_waves(device)))

    assert signal == expected


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("pixel", [(-20, 5), (-4, 25), (4, -5), (12, 5), (-4, 5)])
def test_annular_detector_single_bright_pixel_with_offset(device, pixel):
    """A uniform pattern cannot tell +offset from -offset, or kx from ky (the
    pixel counts of mirrored/transposed annuli are identical); a single bright
    pixel can. Expected: full intensity iff inner <= |k - offset| < outer."""
    inner, outer = 10.4, 30.2
    offset_pixels = (-4, 5)
    offset = (offset_pixels[0] * _ds[0], offset_pixels[1] * _ds[1])
    distance = np.hypot(
        (pixel[0] - offset_pixels[0]) * _ds[0], (pixel[1] - offset_pixels[1]) * _ds[1]
    )
    _assert_clear_of_edges(distance, (inner, outer))
    inside = inner <= distance < outer

    detector = abtem.AnnularDetector(inner=inner, outer=outer, offset=offset)
    signal = _values(detector.detect(_plane_waves(*pixel, device=device)))

    full = float(N_PIXELS) ** 2
    if inside:
        np.testing.assert_allclose(signal, full, rtol=1e-5)
    else:
        assert signal < 1e-9 * full


# ---------------------------------------------------------------------------
# 2. FlexibleAnnularDetector
# ---------------------------------------------------------------------------

FLEXIBLE_CASES = [
    # (step, inner, outer)
    (2.0, 4.0, 40.0),
    (3.0, 5.5, 47.5),
]
# outer - inner not a multiple of step: the documented step must still be the
# bin width (bins are [inner + i step, inner + (i + 1) step)).
FLEXIBLE_NON_MULTIPLE_CASES = [
    (0.5, 0.0, 10.5),  # 21 bins of 0.5, but floor() was applied before /step
    (2.0, 3.0, 40.0),  # 37 / 2 -> 18 bins covering [3, 39)
    (1.0, 0.0, None),  # auto outer: the (non-integer) antialias cutoff angle
]


def _flexible_expected(alpha, step, inner, nbins):
    edges = inner + step * np.arange(nbins + 1)
    _assert_clear_of_edges(alpha, edges)
    return np.array(
        [
            np.count_nonzero((alpha >= edges[i]) & (alpha < edges[i + 1]))
            for i in range(nbins)
        ]
    )


def _check_flexible_bins(device, step, inner, outer):
    waves = _delta_waves(device)
    detector = abtem.FlexibleAnnularDetector(step_size=step, inner=inner, outer=outer)
    measurement = detector.detect(waves)

    if outer is None:
        outer = min(waves.cutoff_angles)
    nbins = int(np.floor((outer - inner) / step + 1e-9))

    assert measurement.shape == (nbins, 1)
    assert measurement.radial_sampling == pytest.approx(step)
    assert measurement.radial_offset == pytest.approx(inner)

    ax, ay = _pixel_angles()
    expected = _flexible_expected(np.hypot(ax, ay), step, inner, nbins)
    got = _values(measurement)[:, 0]

    # (b) each bin holds exactly the hand-counted pixels
    np.testing.assert_array_equal(got, expected)
    # (a) conservation: nothing double-counted or lost inside the outermost edge
    alpha = np.hypot(ax, ay)
    in_range = np.count_nonzero((alpha >= inner) & (alpha < inner + nbins * step))
    assert got.sum() == in_range


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("step, inner, outer", FLEXIBLE_CASES)
def test_flexible_annular_bins_match_hand_counted_rings(device, step, inner, outer):
    _check_flexible_bins(device, step, inner, outer)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("step, inner, outer", FLEXIBLE_NON_MULTIPLE_CASES)
def test_flexible_annular_bin_width_is_step_size(device, step, inner, outer):
    _check_flexible_bins(device, step, inner, outer)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "step, inner, outer", FLEXIBLE_CASES + FLEXIBLE_NON_MULTIPLE_CASES[:2]
)
def test_diffraction_patterns_radial_binning_bin_width_is_step_size(
    device, step, inner, outer
):
    """DiffractionPatterns.radial_binning is documented as equivalent to the
    FlexibleAnnularDetector: bins [inner + i step, inner + (i + 1) step)."""
    patterns = _delta_waves(device).diffraction_patterns(max_angle="full")
    measurement = patterns.radial_binning(step_size=step, inner=inner, outer=outer)
    nbins = int(np.floor((outer - inner) / step + 1e-9))

    assert measurement.shape == (nbins, 1)
    assert measurement.radial_sampling == pytest.approx(step)
    ax, ay = _pixel_angles()
    expected = _flexible_expected(np.hypot(ax, ay), step, inner, nbins)
    np.testing.assert_array_equal(_values(measurement)[:, 0], expected)


def test_flexible_annular_show_draws_the_detected_bins():
    """show() without waves must draw the bins detect() uses: step_size wide,
    with the trailing partial step [40, 41.3) left undetected."""
    import matplotlib.pyplot as plt

    step, inner, outer = 2.0, 0.0, 41.3
    nbins = 20
    detector = abtem.FlexibleAnnularDetector(step_size=step, inner=inner, outer=outer)
    assert detector.nbins_radial == nbins

    visualization = detector.show(
        gpts=GPTS, sampling=SAMPLING, energy=ENERGY, units="mrad"
    )
    regions = np.asarray(visualization.measurement.array, dtype=np.float64)
    plt.close("all")

    ax, ay = _pixel_angles()
    alpha = np.fft.fftshift(np.hypot(ax, ay))
    edges = inner + step * np.arange(nbins + 1)
    _assert_clear_of_edges(alpha, edges)
    expected = np.floor((alpha - inner) / step)
    expected[alpha >= edges[-1]] = np.nan
    np.testing.assert_array_equal(regions, expected)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("a, b", [(10.0, 30.0), (16.0, 34.0), (4.0, 12.0)])
def test_flexible_integrate_radial_matches_annular_detector(device, a, b):
    """(c) Cross-method: FlexibleAnnular binning (polar-bin index path) then
    integrate_radial(a, b) must equal AnnularDetector(a, b) (boolean mask on
    the uncropped pattern) on the same, non-uniform waves.

    The limits lie on bin edges (inner + k step), where the current
    left-edge-index rule and the bin-centre rule of the pending
    PolarMeasurements.integrate fix select the same bins.
    """
    ax, ay = _pixel_angles()
    _assert_clear_of_edges(np.hypot(ax, ay), (a, b))
    waves = _random_waves(device)

    flexible = abtem.FlexibleAnnularDetector(step_size=2.0, inner=2.0, outer=40.0)
    annular = abtem.AnnularDetector(inner=a, outer=b)

    from_bins = _values(flexible.detect(waves).integrate_radial(a, b))
    from_mask = _values(annular.detect(waves))

    # Same pixel set, different summation order: float32 rounding only.
    np.testing.assert_allclose(from_bins, from_mask, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. SegmentedDetector
# ---------------------------------------------------------------------------


def _segment_index(alpha, phi, inner, outer, nbins_radial, nbins_azimuthal, rotation):
    """Hand-built segment label (-1 outside), from the explicit segment
    edges rather than the code's floor/modulo arithmetic."""
    dr = (outer - inner) / nbins_radial
    dphi = 2 * np.pi / nbins_azimuthal
    labels = -np.ones(alpha.shape, dtype=int)
    for r in range(nbins_radial):
        in_ring = (alpha >= inner + r * dr) & (alpha < inner + (r + 1) * dr)
        for a in range(nbins_azimuthal):
            start = rotation + a * dphi
            # angular distance swept counter-clockwise from the segment start
            swept = np.mod(phi - start, 2 * np.pi)
            in_wedge = swept < dphi
            labels[in_ring & in_wedge] = r * nbins_azimuthal + a
    return labels


SEGMENTED = dict(nbins_radial=3, nbins_azimuthal=5, inner=8.3, outer=38.9)


def _segmented_geometry_is_unambiguous(alpha, phi, rotation, offset_mrad=(0, 0)):
    p = SEGMENTED
    dr = (p["outer"] - p["inner"]) / p["nbins_radial"]
    radial_edges = p["inner"] + dr * np.arange(p["nbins_radial"] + 1)
    _assert_clear_of_edges(alpha, radial_edges)
    dphi = 2 * np.pi / p["nbins_azimuthal"]
    in_detector = (alpha >= p["inner"]) & (alpha < p["outer"])
    swept = np.mod(phi[in_detector] - rotation, dphi)
    # distance of every detected pixel from the nearest azimuthal edge
    clearance = np.minimum(swept, dphi - swept)
    assert clearance.min() > 1e-5


@pytest.mark.parametrize("device", ["cpu", gpu])
# (27, 3) and (6, 35) have |k| > outer: they lie outside a pattern cropped to
# `outer` about k=0, but within `outer` of the offset centre.
@pytest.mark.parametrize(
    "pixel",
    [
        (13, 7),
        (-8, 3),
        (27, 3),
        (6, 35),
    ],
)
def test_segmented_detector_offset_captures_pixels_beyond_centred_crop(device, pixel):
    """With an offset centre, the detector must see pixels whose |k| exceeds
    ``outer`` (they are within ``outer`` of the offset centre). Offset is in
    whole pixels, so the hand-built label is unambiguous."""
    rotation = 0.4
    ds = _angular_sampling()
    offset_pixels = (5, 3)
    offset = (offset_pixels[0] * ds[0], offset_pixels[1] * ds[1])
    kx = (pixel[0] - offset_pixels[0]) * ds[0]
    ky = (pixel[1] - offset_pixels[1]) * ds[1]
    label = _segment_index(
        np.array([np.hypot(kx, ky)]),
        np.array([np.arctan2(ky, kx)]),
        rotation=rotation,
        **SEGMENTED,
    )[0]
    assert label >= 0

    detector = abtem.SegmentedDetector(rotation=rotation, offset=offset, **SEGMENTED)
    got = _values(detector.detect(_plane_waves(*pixel, device=device))).ravel()

    full = float(N_PIXELS) ** 2
    np.testing.assert_allclose(got[label], full, rtol=1e-5)
    assert np.delete(got, label).max() < 1e-9 * full


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lazy_segmented_detector_with_offset_matches_eager(device):
    """A lazy detection rebuilds the detector from its constructor arguments,
    which read ``offset`` back as an attribute."""
    offset = (5 * _ds[0], 3 * _ds[1])
    detector = abtem.SegmentedDetector(rotation=0.4, offset=offset, **SEGMENTED)
    waves = _random_waves(device)

    eager = _values(detector.detect(waves))
    lazy = _values(detector.detect(waves.ensure_lazy()))

    np.testing.assert_array_equal(lazy, eager)


def _shown_regions(detector, *args, **kwargs):
    import matplotlib.pyplot as plt

    visualization = detector.show(*args, units="mrad", **kwargs)
    measurement = visualization.measurement
    plt.close("all")
    return np.asarray(measurement.array, dtype=np.float64), measurement


def _shown_angles(measurement):
    """(ax, ay) [mrad] of every pixel of a drawn (fftshifted) region map."""
    n0, n1 = measurement.shape[-2:]
    d0, d1 = measurement.angular_sampling
    ax = (np.arange(n0) - n0 // 2)[:, None] * d0
    ay = (np.arange(n1) - n1 // 2)[None, :] * d1
    return np.broadcast_to(ax, (n0, n1)), np.broadcast_to(ay, (n0, n1))


def test_segmented_detector_show_draws_the_offset():
    """show() draws the segments about the offset centre, as detected."""
    rotation = 0.4
    offset_pixels = (5, 3)
    offset = (offset_pixels[0] * _ds[0], offset_pixels[1] * _ds[1])
    detector = abtem.SegmentedDetector(rotation=rotation, offset=offset, **SEGMENTED)

    regions, _ = _shown_regions(detector, gpts=GPTS, sampling=SAMPLING, energy=ENERGY)

    ax, ay = _pixel_angles()
    kx, ky = ax - offset[0], ay - offset[1]
    alpha, phi = np.hypot(kx, ky), np.arctan2(ky, kx)
    _segmented_geometry_is_unambiguous(alpha, phi, rotation)
    expected = np.fft.fftshift(
        _segment_index(alpha, phi, rotation=rotation, **SEGMENTED)
    ).astype(np.float64)
    expected[expected < 0] = np.nan

    np.testing.assert_array_equal(regions, expected)


def test_radial_detector_show_default_grid_covers_the_detector():
    """Without ``sampling`` the drawn grid reaches 10 % beyond the detector
    (it used to reach outer / 4.4, and an integer ``outer`` failed)."""
    detector = abtem.SegmentedDetector(
        nbins_radial=2, nbins_azimuthal=4, inner=10, outer=40, offset=(6.0, 0.0)
    )
    regions, measurement = _shown_regions(detector, energy=100e3, gpts=64)

    np.testing.assert_allclose(measurement.max_angles, 1.1 * 46.0, rtol=0.05)
    assert set(np.unique(regions[np.isfinite(regions)])) == set(range(8))


def test_flexible_annular_show_with_waves_matches_detection():
    """show(waves) on a detector that has not detected yet sizes the default
    outer from the waves as detect() does (it raised TypeError), and draws
    detect()'s bins."""
    step = 2.0
    waves = _delta_waves()
    detected = abtem.FlexibleAnnularDetector(step_size=step).detect(waves)
    nbins = detected.shape[0]

    regions, measurement = _shown_regions(
        abtem.FlexibleAnnularDetector(step_size=step), waves
    )
    labels = np.unique(regions[np.isfinite(regions)])
    np.testing.assert_array_equal(labels, np.arange(nbins))

    ax, ay = _shown_angles(measurement)
    alpha = np.hypot(ax, ay)
    assert alpha[np.isfinite(regions)].max() < nbins * step
    clear = np.abs(alpha / step - np.round(alpha / step)) > 1e-6
    labelled = np.isfinite(regions) & clear
    np.testing.assert_array_equal(regions[labelled], np.floor(alpha[labelled] / step))


def test_annular_detector_show_with_waves_and_default_outer():
    """AnnularDetector(inner=...).show(waves) draws [inner, cutoff) (it raised
    RuntimeError: radial_sampling is not defined when outer angle is None)."""
    waves = _delta_waves()
    detector = abtem.AnnularDetector(inner=20.0)
    inner, outer = detector.angular_limits(waves)

    regions, measurement = _shown_regions(detector, waves)

    ax, ay = _shown_angles(measurement)
    alpha = np.hypot(ax, ay)
    drawn = np.isfinite(regions)
    assert drawn.any()
    assert alpha[drawn].min() >= inner
    assert alpha[drawn].max() < outer


# Offset centre 60 pixels along +kx, radius 25.5 x-pixels: the region runs to
# x-index 85.5, past the Nyquist index 79 of the 160-pixel axis. The bins
# shifted beyond it used to wrap round onto x-indices -80..-75, so the bright
# pixels at -75 and -79 (aliases of 85 and 81, which are within the radius of
# the centre) were detected although they lie ~135 pixels from it.
_PAST_NYQUIST_OFFSET_PIXELS = (60, 0)
_PAST_NYQUIST_OUTER = 25.5 * _ds[0]


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "make_detector",
    [
        lambda outer, offset: abtem.SegmentedDetector(
            nbins_radial=1, nbins_azimuthal=1, inner=0.0, outer=outer, offset=offset
        ),
        lambda outer, offset: abtem.AnnularDetector(
            inner=0.0, outer=outer, offset=offset
        ),
    ],
    ids=["segmented", "annular"],
)
@pytest.mark.parametrize(
    "pixel, inside", [((75, 0), True), ((-75, 0), False), ((-79, 3), False)]
)
def test_offset_detector_reaching_past_nyquist_does_not_wrap(
    device, make_detector, pixel, inside
):
    offset = (
        _PAST_NYQUIST_OFFSET_PIXELS[0] * _ds[0],
        _PAST_NYQUIST_OFFSET_PIXELS[1] * _ds[1],
    )
    ax, ay = _pixel_angles()
    _assert_clear_of_edges(
        np.hypot(ax - offset[0], ay - offset[1]), (_PAST_NYQUIST_OUTER,)
    )

    detector = make_detector(_PAST_NYQUIST_OUTER, offset)
    got = _values(detector.detect(_plane_waves(*pixel, device=device))).sum()

    full = float(N_PIXELS) ** 2
    if inside:
        np.testing.assert_allclose(got, full, rtol=1e-5)
    else:
        assert got < 1e-9 * full


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda **kw: abtem.FlexibleAnnularDetector(step_size=5, inner=10, **kw),
        lambda **kw: abtem.AnnularDetector(inner=10, **kw),
    ],
    ids=["flexible", "annular"],
)
def test_detector_reused_for_waves_at_another_energy_matches_its_own_waves(
    make_detector,
):
    # Without an explicit outer angle the detector takes the antialias cutoff of the
    # waves it detects. It used to keep the angle of the first waves it ever saw, so
    # the same detector gave a different (wrong) result for waves at another energy.
    waves_100 = abtem.PlaneWave(energy=100e3, gpts=64, extent=10).build()
    waves_300 = abtem.PlaneWave(energy=300e3, gpts=64, extent=10).build()

    reused = make_detector()
    reused.detect(waves_100)
    after = _values(reused.detect(waves_300))
    fresh = _values(make_detector().detect(waves_300))

    np.testing.assert_array_equal(after, fresh)


def test_explicit_outer_angle_is_kept_for_every_waves():
    detector = abtem.FlexibleAnnularDetector(step_size=5, inner=10, outer=40)
    for energy in (100e3, 300e3):
        detector.detect(abtem.PlaneWave(energy=energy, gpts=64, extent=10).build())
        assert detector.outer == 40

    detector.outer = None
    detector.detect(abtem.PlaneWave(energy=100e3, gpts=64, extent=10).build())
    assert detector.outer != 40
