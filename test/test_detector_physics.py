"""Independent-oracle tests for the detectors.

Every expected value in this file comes from first principles, never from the
detector's own mask/bin helpers or from pasted output:

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

Edge conventions established (and relied on) here:

* Annular / radial bins: ``inner <= alpha < outer`` (inner inclusive, outer
  exclusive). Inner inclusivity is checked exactly at the DC pixel; pixels
  lying *exactly* on an edge are otherwise float-precision dependent, so every
  geometry below is asserted to keep all pixels at least ``EDGE_MARGIN``
  (relative) away from every edge -- which makes the pixel counts exact.
* Radial bin ``i`` of a FlexibleAnnularDetector covers
  ``[inner + i * step, inner + (i + 1) * step)``.
* Azimuth ``phi = atan2(ky, kx)`` (counter-clockwise from +kx). Azimuthal bin
  ``j`` of a SegmentedDetector covers ``phi in [rotation + j * dphi,
  rotation + (j + 1) * dphi)`` modulo 2 pi, i.e. a positive ``rotation``
  turns the segments counter-clockwise. This matches the output's azimuthal
  axis metadata (``offset=rotation``).
* Slit ``angle`` is in degrees, counter-clockwise from +kx; the slit covers
  ``q_min <= (k - offset) . d < q_max`` and ``-width/2 <= (k - offset) . n <
  width/2`` with ``d = (cos, sin)`` and ``n = (-sin, cos)``.
* PixelatedDetector output pixel ``p`` along an axis of length ``M`` is the
  frequency ``(p - M // 2) * dk`` (its ReciprocalSpaceAxis ``offset`` is
  ``-(M // 2) * dk`` with ``fftshift=True``).

Grids are deliberately non-square with anisotropic sampling (and one odd
axis) so that swapped or mis-scaled x/y handling shows up.
"""

import ase.build
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.core.axes import EnergyLossAxis
from abtem.core.backend import asnumpy
from abtem.core.utils import get_dtype
from abtem.measurements import DiffractionPatterns, momentum_resolved_spectrum
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


# ---------------------------------------------------------------------------
# 1. AnnularDetector: exact pixel counts on a uniform pattern
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_annular_detector_inner_is_inclusive_at_dc(device):
    """inner=0 includes the k=0 pixel (0 >= 0 exactly); any inner > 0 does
    not. A plane wave with k=0 puts all its intensity in that pixel."""
    waves = _plane_waves(0, 0, device)
    full = float(N_PIXELS) ** 2

    included = _values(abtem.AnnularDetector(inner=0.0, outer=5.0).detect(waves))
    excluded = _values(abtem.AnnularDetector(inner=0.01, outer=5.0).detect(waves))

    np.testing.assert_allclose(included, full, rtol=1e-5)
    assert excluded < 1e-9 * full


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_adjacent_annuli_partition_exactly(device):
    """[a, b) + [b, c) == [a, c) with no pixel double-counted or lost, for a
    split radius b chosen exactly on the radius of an on-axis pixel (the most
    edge-sensitive choice possible)."""
    waves = _delta_waves(device)
    a, c = 3.3, 47.7
    b = 11 * _ds[0]  # radius of pixel (11, 0)

    def signal(inner, outer):
        return _values(abtem.AnnularDetector(inner=inner, outer=outer).detect(waves))

    assert signal(a, b) + signal(b, c) == signal(a, c)


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


@pytest.mark.xfail(
    strict=True,
    reason="FlexibleAnnularDetector.nbins_radial is int(np.floor(outer - inner) "
    "/ step_size) -- floor applied before dividing -- and the bins are then "
    "spread over the full [inner, outer): when outer - inner is not an integer "
    "multiple of step_size (always, for the default auto outer = antialias "
    "cutoff) the bins are wider than step_size while radial_sampling still "
    "reports step_size.",
)
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("step, inner, outer", FLEXIBLE_NON_MULTIPLE_CASES)
def test_flexible_annular_bin_width_is_step_size(device, step, inner, outer):
    _check_flexible_bins(device, step, inner, outer)


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
@pytest.mark.parametrize("rotation", [0.4, -1.1])
def test_segmented_detector_segments_match_hand_built_polar_masks(device, rotation):
    ax, ay = _pixel_angles()
    alpha, phi = np.hypot(ax, ay), np.arctan2(ay, ax)
    _segmented_geometry_is_unambiguous(alpha, phi, rotation)

    labels = _segment_index(alpha, phi, rotation=rotation, **SEGMENTED)
    n = SEGMENTED["nbins_radial"] * SEGMENTED["nbins_azimuthal"]
    expected = np.array([np.count_nonzero(labels == i) for i in range(n)])
    expected = expected.reshape(SEGMENTED["nbins_radial"], SEGMENTED["nbins_azimuthal"])
    assert expected.min() > 10

    detector = abtem.SegmentedDetector(rotation=rotation, **SEGMENTED)
    got = _values(detector.detect(_delta_waves(device)))

    np.testing.assert_array_equal(got, expected)


# Bright pixels (frequency indices) chosen so a counter-clockwise and a
# clockwise rotation of 0.4 rad put them in *different* segments.
BRIGHT_PIXELS = [(10, 2), (-7, 12), (3, -20), (-15, -9), (18, 1)]


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("pixel", BRIGHT_PIXELS)
def test_segmented_detector_single_bright_pixel_lands_in_expected_segment(
    device, pixel
):
    rotation = 0.4
    ds = _angular_sampling()
    kx, ky = pixel[0] * ds[0], pixel[1] * ds[1]
    alpha, phi = np.hypot(kx, ky), np.arctan2(ky, kx)

    def label_for(rotation):
        return _segment_index(
            np.array([alpha]), np.array([phi]), rotation=rotation, **SEGMENTED
        )[0]

    label = label_for(rotation)
    assert label >= 0
    # The pixel discriminates the rotation sense: rotating the segments
    # clockwise instead would put it in another segment.
    assert label_for(-rotation) != label

    detector = abtem.SegmentedDetector(rotation=rotation, **SEGMENTED)
    got = _values(detector.detect(_plane_waves(*pixel, device=device))).ravel()

    full = float(N_PIXELS) ** 2
    np.testing.assert_allclose(got[label], full, rtol=1e-5)
    assert np.delete(got, label).max() < 1e-9 * full


_OFFSET_CROP_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="_AbstractRadialDetector._calculate_new_array crops the pattern to "
    "max_angle=outer about k=0 before polar_binning rolls the bins by the "
    "offset, so pixels farther than outer from k=0 are lost (and the rolled "
    "bins wrap onto the opposite edge of the cropped pattern).",
)


@pytest.mark.parametrize("device", ["cpu", gpu])
# (27, 3) and (6, 35) have |k| > outer: they lie outside a pattern cropped to
# `outer` about k=0, but within `outer` of the offset centre.
@pytest.mark.parametrize(
    "pixel",
    [
        (13, 7),
        (-8, 3),
        pytest.param((27, 3), marks=_OFFSET_CROP_XFAIL),
        pytest.param((6, 35), marks=_OFFSET_CROP_XFAIL),
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


# ---------------------------------------------------------------------------
# 4. Spectral slit (and annular-sweep) detectors
# ---------------------------------------------------------------------------


def _slit_pixel_count(angle, width, q_min, q_max, offset):
    """Pixels inside the slit, from half-plane tests along its axis d and
    normal n (no rotation of the grid, no call into the detector)."""
    ax, ay = _pixel_angles()
    theta = np.deg2rad(angle)
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-np.sin(theta), np.cos(theta)])
    rx, ry = ax - offset[0], ay - offset[1]
    along = rx * d[0] + ry * d[1]
    across = rx * n[0] + ry * n[1]
    _assert_clear_of_edges(along, (q_min, q_max), atol=1e-6)
    _assert_clear_of_edges(across, (-width / 2, width / 2), atol=1e-6)
    inside = (
        (along >= q_min)
        & (along < q_max)
        & (across >= -width / 2)
        & (across < width / 2)
    )
    return np.count_nonzero(inside)


SLIT_CASES = [
    # (angle [deg], width, q_min, q_max, offset [mrad])
    # q_min > 0 or an offset keeps the DC pixel off the slit's start edge (see
    # test_slit_detector_q_min_zero_includes_direct_beam for that edge).
    (0.0, 6.1, 1.3, 40.3, (0.0, 0.0)),
    (30.0, 4.3, 0.9, 40.1, (0.0, 0.0)),
    (137.0, 5.3, 3.1, 50.2, (0.0, 0.0)),
    (-60.0, 7.7, 0.0, 45.3, (4.7, -3.9)),
]


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("angle, width, q_min, q_max, offset", SLIT_CASES)
def test_slit_detector_counts_pixels_in_rotated_rectangle(
    device, angle, width, q_min, q_max, offset
):
    """Uniform unit pattern -> slit signal == hand-counted pixels of the
    rotated rectangle; and that count is within the discretisation bound of
    the analytic area / pixel area."""
    expected = _slit_pixel_count(angle, width, q_min, q_max, offset)

    detector = abtem.SpectralSlitDetector(
        width=width, q_min=q_min, q_max=q_max, angle=angle, offset=offset
    )
    signal = _values(detector.detect(_delta_waves(device)))
    assert signal == expected

    # Independent sanity bound: |count * A_pix - L W| <= perimeter * pixel
    # diagonal (only pixels whose cell straddles the boundary can differ).
    ds = _angular_sampling()
    area = (q_max - q_min) * width
    bound = 2 * ((q_max - q_min) + width) * np.hypot(*ds)
    assert abs(expected * ds[0] * ds[1] - area) <= bound


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "pixel, offset_pixels",
    [((14, 9), (0, 0)), ((-5, 17), (2, -3)), ((12, -16), (0, 0))],
)
def test_slit_detector_rotation_sense(device, pixel, offset_pixels):
    """A slit whose axis points at an off-axis bright pixel captures all of
    its intensity; the mirror slit at -angle captures none."""
    ds = _angular_sampling()
    offset = (offset_pixels[0] * ds[0], offset_pixels[1] * ds[1])
    kx = (pixel[0] - offset_pixels[0]) * ds[0]
    ky = (pixel[1] - offset_pixels[1]) * ds[1]
    angle = np.rad2deg(np.arctan2(ky, kx))
    q = np.hypot(kx, ky)
    width = 4.0
    # the mirror slit misses the pixel by a wide margin
    assert abs(q * np.sin(2 * np.deg2rad(angle))) > width

    waves = _plane_waves(*pixel, device=device)
    full = float(N_PIXELS) ** 2

    def captured(angle):
        detector = abtem.SpectralSlitDetector(
            width=width, q_min=0.0, q_max=q + 10.0, angle=angle, offset=offset
        )
        return _values(detector.detect(waves))

    np.testing.assert_allclose(captured(angle), full, rtol=1e-5)
    assert captured(-angle) < 1e-9 * full


@pytest.mark.xfail(
    strict=True,
    reason="SpectralSlitDetector puts the q=0 pixel exactly on the slit's start "
    "edge when q_min=0 (the default), and _slit_detector_mask tests it in the "
    "slit-centre frame: local_x = (0 - c) . d = -(q_min + q_max) / 2 * "
    "(cos^2 + sin^2), which rounds either side of -extent / 2. The direct beam "
    "is dropped for ~8% of (angle, q_max) pairs (e.g. angle=15, q_max=40.1; "
    "angle=120, q_max=20), although q_min=0 is documented to include q=0.",
)
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_slit_detector_q_min_zero_includes_direct_beam(device):
    """q_min=0 is documented to include q=0: a k=0 plane wave (all intensity
    in the DC pixel) must be fully captured whatever the slit angle/length."""
    waves = _plane_waves(0, 0, device)
    full = float(N_PIXELS) ** 2
    dropped = []
    for angle in np.arange(-180.0, 180.0, 7.5):
        for q_max in (20.0, 33.3, 40.1):
            detector = abtem.SpectralSlitDetector(width=4.0, q_max=q_max, angle=angle)
            if not np.isclose(_values(detector.detect(waves)), full, rtol=1e-5):
                dropped.append((float(angle), q_max))
    assert not dropped, f"direct beam dropped for (angle, q_max) = {dropped}"


def _single_pixel_spectral_dp(i0, j0):
    """fftshifted DiffractionPatterns (2 energy losses) that are 1 at
    frequency index (i0, j0) and 0 elsewhere."""
    unshifted = np.zeros(GPTS, dtype=get_dtype())
    unshifted[i0 % GPTS[0], j0 % GPTS[1]] = 1.0
    array = np.stack([np.fft.fftshift(unshifted)] * 2)
    reciprocal = (1 / (GPTS[0] * SAMPLING[0]), 1 / (GPTS[1] * SAMPLING[1]))
    return DiffractionPatterns(
        array,
        sampling=reciprocal,
        fftshift=True,
        ensemble_axes_metadata=[EnergyLossAxis(values=(0.02, 0.05), units="eV")],
        metadata={"energy": ENERGY},
    )


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda angle: abtem.SpectralSlitDetector(width=4.0, q_max=35.0, angle=angle),
        lambda angle: abtem.SpectralAnnularDetector(outer=3.0, q_max=35.0, angle=angle),
    ],
    ids=["slit", "annular"],
)
def test_momentum_resolved_spectrum_sweeps_toward_angle(make_detector):
    """S(q, E) from a single bright pixel at q0 along theta: the sweep at theta
    peaks at q ~= q0; the sweep at -theta is identically zero."""
    pixel = (14, 9)
    ds = _angular_sampling()
    kx, ky = pixel[0] * ds[0], pixel[1] * ds[1]
    angle = np.rad2deg(np.arctan2(ky, kx))
    q0 = np.hypot(kx, ky)
    dp = _single_pixel_spectral_dp(*pixel)

    toward = momentum_resolved_spectrum(dp, make_detector(angle))
    away = momentum_resolved_spectrum(dp, make_detector(-angle))

    toward_values = np.asarray(asnumpy(toward.array))
    q_values = np.asarray(toward._q_values)
    q_peak = q_values[np.argmax(toward_values[:, 0])]
    q_step = np.max(np.diff(q_values))

    assert toward_values.max() > 0.5
    assert abs(q_peak - q0) <= q_step
    assert np.all(np.asarray(asnumpy(away.array)) == 0.0)


# ---------------------------------------------------------------------------
# 5. PixelatedDetector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("max_angle", ["full", "cutoff", "valid", 30.0, 47.3])
def test_pixelated_detector_is_shifted_squared_fft(device, max_angle):
    """Output pixel (p, q) == |fft2(psi)|^2 at frequency index
    (p - Mx // 2, q - My // 2), from an independent numpy FFT; every pixel
    within max_angle along each axis is kept."""
    waves = _random_waves(device, seed=3)
    psi = np.asarray(asnumpy(waves.array), dtype=np.complex128)
    reference = np.abs(np.fft.fft2(psi)) ** 2

    measurement = abtem.PixelatedDetector(max_angle=max_angle).detect(waves)
    got = _values(measurement)
    mx, my = got.shape

    ix = (np.arange(mx) - mx // 2) % GPTS[0]
    iy = (np.arange(my) - my // 2) % GPTS[1]
    expected = reference[np.ix_(ix, iy)]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4 * reference.max())

    if max_angle == "full":
        assert (mx, my) == GPTS
    elif isinstance(max_angle, float):
        ds = _angular_sampling()
        needed = (int(max_angle // ds[0]), int(max_angle // ds[1]))
        assert mx // 2 >= needed[0] and (mx - 1) // 2 >= needed[0]
        assert my // 2 >= needed[1] and (my - 1) // 2 >= needed[1]

    np.testing.assert_allclose(
        measurement.sampling,
        (1 / (GPTS[0] * SAMPLING[0]), 1 / (GPTS[1] * SAMPLING[1])),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# 6. Cross-detector consistency on real multislice exit waves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["cpu", gpu])
def si_exit_waves(request):
    """STEM exit waves through a small Si slab on a non-square 3 x 2 scan.

    The cell (10.9 x 16.3 Å) and coarse, anisotropic sampling put the
    antialias cutoff at a non-integer angle (~61.7 mrad) with pixels fine
    enough (~3.4 x 2.3 mrad) that a bin-edge error of a fraction of a mrad
    moves pixels between integration regions.
    """
    device = request.param
    atoms = ase.build.bulk("Si", cubic=True) * (2, 3, 1)
    potential = abtem.Potential(
        atoms, sampling=(0.2, 0.17), slice_thickness=2.0, device=device
    )
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20.0, device=device)
    scan = abtem.GridScan(
        start=(0, 0), end=(5.43, 5.43), gpts=(3, 2), potential=potential
    )
    return probe.multislice(potential, scan=scan).compute()


_FLEXIBLE_AUTO_OUTER_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="FlexibleAnnularDetector with the default outer spreads floor(outer) "
    "bins over the non-integer [0, outer): bin i is [i, i + 1) * outer / "
    "floor(outer), not [i, i + 1) mrad as its radial_sampling=1 reports.",
)


@pytest.mark.parametrize(
    "a, b",
    [
        (0.0, 18.0),
        pytest.param(20.0, 45.0, marks=_FLEXIBLE_AUTO_OUTER_XFAIL),
        pytest.param(31.0, 57.0, marks=_FLEXIBLE_AUTO_OUTER_XFAIL),
    ],
)
def test_annular_flexible_pixelated_agree_on_multislice_exit_waves(si_exit_waves, a, b):
    """Annular(a, b) == FlexibleAnnular.integrate_radial(a, b) ==
    Pixelated.integrate_radial(a, b): three code paths (uncropped boolean
    mask; polar bin indices on a cropped pattern; mask on a cropped,
    fftshifted pattern) over the same pixel set. The FlexibleAnnular uses its
    default outer (the non-integer antialias cutoff) and step 1, so its bins
    must be exactly [i, i + 1) mrad for integrate_radial(a, b) to select the
    annulus [a, b)."""
    exit_waves = si_exit_waves
    alpha = np.hypot(
        *_pixel_angles(gpts=exit_waves.gpts, sampling=exit_waves.sampling, energy=100e3)
    )
    _assert_clear_of_edges(alpha, (a, b))
    assert b < min(exit_waves.cutoff_angles)

    annular = _values(abtem.AnnularDetector(inner=a, outer=b).detect(exit_waves))
    flexible = abtem.FlexibleAnnularDetector(step_size=1.0).detect(exit_waves)
    pixelated = abtem.PixelatedDetector(max_angle="cutoff").detect(exit_waves)

    assert annular.shape == (3, 2)
    assert annular.min() > 0
    # Same pixel sets, different summation order: float32 rounding only.
    np.testing.assert_allclose(
        _values(pixelated.integrate_radial(a, b)), annular, rtol=1e-5
    )
    np.testing.assert_allclose(
        _values(flexible.integrate_radial(a, b)), annular, rtol=1e-5
    )
