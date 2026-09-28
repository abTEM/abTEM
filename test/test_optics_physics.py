"""Independent-oracle physics tests for the optics in ``abtem/transfer.py`` and
the frozen-phonon displacements in ``abtem/inelastic/phonons.py``.

Every expected value here is computed *in the test* from an independent source:
a textbook formula evaluated by hand, a Gaussian average carried out with plain
numpy quadrature, a symmetry, or a limiting case. None of them are copied from
abTEM's output or re-derived from its implementation.

Conventions pinned by this file (from the `CTF` / `Aberrations` docstrings, which
cite Kirkland (2010), Eq. 2.22):

* The aberration function is

      chi(alpha, phi) = 2 pi / lambda * sum_{n,m} C_nm alpha^(n+1) / (n+1)
                                                  * cos(m (phi - phi_nm))

  and the phase-aberration kernel is ``exp(-i chi)``.
* ``defocus = -C10``. With this, for C10 and C30 only,
  ``chi(k) = pi lambda k^2 (Cs lambda^2 k^2 / 2 - defocus)``, which is
  Kirkland's (2010) axial form with positive defocus = underfocus.
* ``alpha = lambda k`` and ``phi = arctan2(ky, kx)``: the azimuth is measured
  counter-clockwise from +kx (array axis 0) toward +ky (array axis 1), and phi_nm
  is the azimuth of a maximum of the corresponding cos term.
* ``focal_spread`` Delta is the 1/e half-width of the defocus distribution
  p(d) ~ exp(-d^2 / Delta^2), i.e. Delta = sqrt(2) sigma. The temporal envelope is
  E_t(k) = exp(-(pi lambda Delta k^2 / 2)^2).
* ``angular_spread`` alpha_s enters as E_s = exp(-(alpha_s / 2)^2 |grad_alpha chi|^2),
  Kirkland's quasi-coherent form exp(-(pi alpha_s / lambda)^2
  (Cs lambda^3 k^3 - Delta_f lambda k)^2). The Gaussian-average test below shows
  that alpha_s is the 1/e half-width of the beam-tilt distribution
  p(beta) ~ exp(-|beta|^2 / alpha_s^2), i.e. a per-axis standard deviation of
  alpha_s / sqrt(2).
"""

import ase.build
import numpy as np
import pytest
import scipy.constants as const
from utils import devices

from abtem import FrozenPhonons
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.energy import energy2wavelength
from abtem.transfer import (
    CTF,
    Aberrations,
    Aperture,
    SpatialEnvelope,
    TemporalEnvelope,
    cartesian2polar,
    hard_aperture,
    point_resolution,
    polar2cartesian,
    scherzer_defocus,
    soft_aperture,
)

ENERGY = 200e3  # eV


def relativistic_wavelength(energy):
    """Relativistic de Broglie wavelength [Å] from CODATA constants.

    lambda = h / sqrt(2 m0 e V (1 + e V / (2 m0 c^2))), e.g. Kirkland (2010) Eq. 2.5.
    """
    eV = energy * const.e
    p = np.sqrt(2 * const.m_e * eV * (1 + eV / (2 * const.m_e * const.c**2)))
    return const.h / p * 1e10


def evaluate(transfer_function, alpha, phi, device):
    """Evaluate a transfer function on a user-supplied (alpha, phi) grid."""
    xp = get_array_module(device)
    alpha = xp.asarray(alpha)
    phi = xp.asarray(phi)
    return asnumpy(transfer_function._evaluate_from_angular_grid(alpha, phi))


def ab_kernel(transfer_function, device):
    """Evaluate a transfer function on its own FFT grid."""
    return transfer_function._evaluate_from_angular_grid(*transfer_function._angular_grid(device))


def polar_grid(max_angle=30e-3, n_alpha=31, n_phi=73):
    alpha = np.linspace(1e-3, max_angle, n_alpha)
    phi = np.linspace(0.0, 2 * np.pi, n_phi)
    return np.meshgrid(alpha, phi, indexing="ij")


def cartesian_angular_grid(gpts, sampling, energy):
    """(alpha, phi) of an FFT grid built from numpy's fftfreq, axis 0 = x."""
    wavelength = relativistic_wavelength(energy)
    kx = np.fft.fftfreq(gpts[0], sampling[0])
    ky = np.fft.fftfreq(gpts[1], sampling[1])
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    return wavelength * np.hypot(kx, ky), np.arctan2(ky, kx)


# ---------------------------------------------------------------------------
# 1. Wavelength and absolute chi(k)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "energy, textbook_wavelength",
    # Kirkland (2010), Table 2.2 (4 significant figures).
    [(100e3, 0.03701), (200e3, 0.02508), (300e3, 0.01969)],
)
def test_wavelength_matches_relativistic_formula(energy, textbook_wavelength):
    # abTEM takes its constants from ase.units (CODATA 2014), scipy from a newer
    # CODATA release: they differ at the 5e-9 level.
    assert np.isclose(
        energy2wavelength(energy), relativistic_wavelength(energy), rtol=1e-7, atol=0
    )
    assert np.isclose(energy2wavelength(energy), textbook_wavelength, rtol=2e-4, atol=0)


@devices
@pytest.mark.parametrize("energy", [80e3, 200e3, 300e3])
def test_chi_absolute_small_phase(device, energy):
    # Kirkland: chi(k) = pi lambda k^2 (Cs lambda^2 k^2 / 2 - defocus). Parameters
    # chosen so |chi| < pi: -angle(kernel) is then chi itself, not chi mod 2 pi.
    defocus, Cs = 40.0, 2e5
    wavelength = relativistic_wavelength(energy)
    alpha = np.array([1e-3, 3e-3, 5e-3, 7e-3, 9e-3])
    k = alpha / wavelength
    expected_chi = np.pi * wavelength * k**2 * (0.5 * Cs * wavelength**2 * k**2 - defocus)
    assert np.all(np.abs(expected_chi) < np.pi)

    kernel = evaluate(
        Aberrations(energy=energy, defocus=defocus, Cs=Cs), alpha, np.zeros_like(alpha), device
    )
    assert np.allclose(np.abs(kernel), 1.0, atol=1e-6)
    assert np.allclose(-np.angle(kernel), expected_chi, atol=1e-5)


@devices
@pytest.mark.parametrize("defocus", [-250.0, 0.0, 600.0])
def test_chi_absolute_large_phase(device, defocus):
    # Same analytic form at realistic Cs = 1 mm, |chi| up to ~100 rad: compare
    # exp(-i chi) directly so 2 pi wraps do not matter.
    Cs = 1e7
    wavelength = relativistic_wavelength(ENERGY)
    alpha, phi = polar_grid(max_angle=20e-3)
    k = alpha / wavelength
    expected_chi = np.pi * wavelength * k**2 * (0.5 * Cs * wavelength**2 * k**2 - defocus)

    kernel = evaluate(Aberrations(energy=ENERGY, defocus=defocus, Cs=Cs), alpha, phi, device)
    assert np.abs(kernel - np.exp(-1j * expected_chi)).max() < 2e-4


def test_ctf_profile_is_minus_sin_chi():
    # CTF.profiles() returns Im[exp(-i chi)] = -sin(chi) along phi = 0.
    defocus, Cs = 300.0, 1e6
    wavelength = relativistic_wavelength(ENERGY)
    gpts, max_angle = 201, 25.0
    profile = CTF(energy=ENERGY, defocus=defocus, Cs=Cs).profiles(gpts=gpts, max_angle=max_angle)
    k = np.linspace(0, max_angle * 1e-3, gpts) / wavelength
    chi = np.pi * wavelength * k**2 * (0.5 * Cs * wavelength**2 * k**2 - defocus)
    assert np.allclose(profile.array, -np.sin(chi), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Azimuthal aberrations
# ---------------------------------------------------------------------------


@devices
@pytest.mark.parametrize(
    "symbol, m", [("C12", 2), ("C21", 1), ("C23", 3), ("C32", 2), ("C34", 4)]
)
def test_azimuthal_aberration_rotates_with_its_angle(device, symbol, m):
    # Rotational covariance: a C_nm term at angle phi_nm is the phi_nm = 0 term
    # evaluated at phi - phi_nm.
    angle_symbol = "phi" + symbol[1:]
    phi_nm = 0.7
    alpha, phi = polar_grid()
    rotated = Aberrations(energy=ENERGY, **{symbol: 80.0, angle_symbol: phi_nm})
    reference = Aberrations(energy=ENERGY, **{symbol: 80.0})
    assert np.abs(
        evaluate(rotated, alpha, phi, device)
        - evaluate(reference, alpha, phi - phi_nm, device)
    ).max() < 1e-4


@devices
@pytest.mark.parametrize(
    "symbol, n, m, magnitude",
    [
        ("C12", 1, 2, 300.0),
        ("C21", 2, 1, 3e4),
        ("C23", 2, 3, 3e4),
        ("C32", 3, 2, 3e6),
        ("C34", 3, 4, 3e6),
    ],
)
def test_azimuthal_aberration_absolute_value(device, symbol, n, m, magnitude):
    # Kirkland (2010) Eq. 2.22, single term:
    # chi = 2 pi / lambda * C_nm alpha^(n+1) / (n+1) * cos(m (phi - phi_nm)).
    angle_symbol = "phi" + symbol[1:]
    phi_nm = 0.4
    wavelength = relativistic_wavelength(ENERGY)
    alpha, phi = polar_grid(max_angle=15e-3)
    expected_chi = (
        2 * np.pi / wavelength * magnitude * alpha ** (n + 1) / (n + 1)
        * np.cos(m * (phi - phi_nm))
    )
    ab = Aberrations(energy=ENERGY, **{symbol: magnitude, angle_symbol: phi_nm})
    assert np.abs(expected_chi).max() > 2.0  # the check is not vacuous
    kernel = evaluate(ab, alpha, phi, device)
    assert np.abs(kernel - np.exp(-1j * expected_chi)).max() < 1e-4


@devices
def test_azimuthal_symmetries(device):
    # Symmetry oracles, independent of any formula: C12 is 2-fold symmetric,
    # C23 3-fold symmetric, and C21 (odd m) is antisymmetric under phi -> phi + pi.
    alpha, phi = polar_grid(max_angle=15e-3)

    def chi(**kwargs):
        ab = Aberrations(energy=ENERGY, **kwargs)
        return lambda p: -np.angle(evaluate(ab, alpha, p, device))

    c12 = chi(C12=20.0, phi12=0.3)
    assert np.allclose(c12(phi + np.pi), c12(phi), atol=1e-4)

    c23 = chi(C23=800.0, phi23=0.3)
    assert np.allclose(c23(phi + 2 * np.pi / 3), c23(phi), atol=1e-4)
    assert not np.allclose(c23(phi + np.pi / 3), c23(phi), atol=1e-2)

    c21 = chi(C21=800.0, phi21=0.3)
    assert np.allclose(c21(phi + np.pi), -c21(phi), atol=1e-4)


@devices
def test_azimuth_direction_on_fft_grid(device):
    # Pin the azimuth direction on the real FFT grid, built independently with
    # numpy: phi = arctan2(ky, kx), axis 0 = x. Coma C21 with phi21 = pi/2 gives
    # chi ~ cos(phi - pi/2) = sin(phi): positive along +ky, zero along +kx.
    gpts, sampling = (64, 64), (0.1, 0.1)
    wavelength = relativistic_wavelength(ENERGY)
    alpha, phi = cartesian_angular_grid(gpts, sampling, ENERGY)
    C21 = 20.0
    expected_chi = 2 * np.pi / wavelength * C21 * alpha**3 / 3 * np.sin(phi)

    ab = Aberrations(energy=ENERGY, gpts=gpts, sampling=sampling, C21=C21, phi21=np.pi / 2)
    kernel = asnumpy(ab_kernel(ab, device))
    assert np.abs(kernel - np.exp(-1j * expected_chi)).max() < 1e-4
    j = 20  # pixel on the +ky axis: chi > 0
    assert -np.angle(kernel[0, j]) > 0.1 and np.isclose(np.angle(kernel[j, 0]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Polar <-> Cartesian round trips
# ---------------------------------------------------------------------------

POLAR = {
    "C10": -120.0,
    "C12": 35.0,
    "phi12": 0.6,
    "C21": 900.0,
    "phi21": -1.1,
    "C23": 700.0,
    "phi23": 0.25,
    "C30": 2e5,
    "C32": 3e4,
    "phi32": 1.3,
    "C34": 2e4,
    "phi34": -0.4,
}


@devices
def test_polar_cartesian_polar_round_trip_preserves_chi(device):
    # The polar representation is not unique (e.g. (C12, phi12) ~ (-C12, phi12 +
    # pi/2)), so compare the aberration function, not the coefficients.
    round_trip = cartesian2polar(polar2cartesian(POLAR))
    alpha, phi = polar_grid(max_angle=20e-3)
    before = evaluate(Aberrations(energy=ENERGY, **POLAR), alpha, phi, device)
    after = evaluate(Aberrations(energy=ENERGY, **round_trip), alpha, phi, device)
    assert np.abs(before - after).max() < 1e-4


def test_cartesian_polar_cartesian_round_trip():
    # Cartesian coefficients are a unique representation: they must come back.
    cartesian = polar2cartesian(POLAR)
    round_trip = polar2cartesian(cartesian2polar(cartesian))
    for key, value in cartesian.items():
        assert np.isclose(round_trip[key], value, rtol=1e-10, atol=1e-8), key


# ---------------------------------------------------------------------------
# 4. Temporal envelope
# ---------------------------------------------------------------------------


@devices
@pytest.mark.parametrize("focal_spread", [30.0, 80.0])
def test_temporal_envelope_analytic_form(device, focal_spread):
    # E_t(k) = exp(-(pi lambda Delta k^2 / 2)^2), Delta the 1/e half-width.
    wavelength = relativistic_wavelength(ENERGY)
    alpha = np.linspace(0, 35e-3, 16)
    k = alpha / wavelength
    expected = np.exp(-((np.pi * wavelength * focal_spread * k**2 / 2) ** 2))
    assert expected.min() < 0.05  # the envelope decays substantially

    envelope = evaluate(
        TemporalEnvelope(focal_spread, energy=ENERGY), alpha, np.zeros_like(alpha), device
    )
    assert np.allclose(envelope, expected, atol=1e-6)


def _gauss_hermite_1d(sigma, n=80):
    """Nodes and weights for averaging over a zero-mean normal of std sigma."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2) * sigma * x, w / np.sqrt(np.pi)


def test_temporal_envelope_equals_gaussian_defocus_average():
    # Physics cross-check: E_t is the average of exp(-i pi lambda d k^2) (the
    # defocus-d part of chi) over a Gaussian defocus spread d of standard deviation
    # sigma = focal_spread / sqrt(2). Plain numpy Gauss-Hermite quadrature.
    focal_spread = 60.0
    wavelength = relativistic_wavelength(ENERGY)
    alpha = np.linspace(0, 30e-3, 16)
    k = alpha / wavelength
    d, w = _gauss_hermite_1d(focal_spread / np.sqrt(2))
    average = np.sum(w[:, None] * np.exp(-1j * np.pi * wavelength * d[:, None] * k**2), axis=0)
    assert np.allclose(average.imag, 0.0, atol=1e-12)

    envelope = TemporalEnvelope(focal_spread, energy=ENERGY)._evaluate_from_angular_grid(
        alpha, np.zeros_like(alpha)
    )
    assert np.allclose(envelope, average.real, atol=1e-6)


def test_temporal_envelope_equals_average_of_defocused_kernels():
    # The same average, but over abTEM's own coherent Aberrations kernels at
    # defocus + d (weights from plain numpy quadrature, no abTEM distributions):
    # <exp(-i chi(defocus + d))> = exp(-i chi(defocus)) * E_t.
    focal_spread, defocus, Cs = 50.0, 200.0, 1e6
    alpha, phi = polar_grid(max_angle=25e-3, n_phi=3)
    d, w = _gauss_hermite_1d(focal_spread / np.sqrt(2), n=60)
    average = sum(
        wi * Aberrations(energy=ENERGY, defocus=defocus + di, Cs=Cs)._evaluate_from_angular_grid(alpha, phi)
        for di, wi in zip(d, w)
    )
    coherent = Aberrations(energy=ENERGY, defocus=defocus, Cs=Cs)._evaluate_from_angular_grid(alpha, phi)
    envelope = TemporalEnvelope(focal_spread, energy=ENERGY)._evaluate_from_angular_grid(alpha, phi)
    assert np.abs(average - coherent * envelope).max() < 1e-4


# ---------------------------------------------------------------------------
# 5. Spatial envelope
# ---------------------------------------------------------------------------


@devices
@pytest.mark.parametrize("defocus", [-300.0, 400.0])
def test_spatial_envelope_analytic_form(device, defocus):
    # Kirkland quasi-coherent form (Delta_f = defocus):
    # E_s(k) = exp(-(pi alpha_s / lambda)^2 (Cs lambda^3 k^3 - Delta_f lambda k)^2).
    Cs, angular_spread = 1e7, 0.1  # mrad
    wavelength = relativistic_wavelength(ENERGY)
    alpha, phi = polar_grid(max_angle=25e-3)
    k = alpha / wavelength
    a_s = angular_spread * 1e-3
    expected = np.exp(
        -((np.pi * a_s / wavelength) ** 2)
        * (Cs * wavelength**3 * k**3 - defocus * wavelength * k) ** 2
    )
    assert expected.min() < 0.05

    envelope = evaluate(
        SpatialEnvelope(angular_spread, energy=ENERGY, defocus=defocus, Cs=Cs), alpha, phi, device
    )
    assert np.allclose(envelope, expected, atol=1e-5)


def test_spatial_envelope_equals_gaussian_tilt_average():
    # Physics cross-check without the linearisation: average exp(-i [chi(a + b) -
    # chi(a)]) over 2D beam tilts b with p(b) ~ exp(-|b|^2 / alpha_s^2) (1/e
    # half-width alpha_s, per-axis std alpha_s / sqrt(2)) by 2D Gauss-Hermite
    # quadrature. The quasi-coherent envelope drops the curvature term
    # b.H.b / 2 ~ 1e-2 rad here, so agreement is to ~1e-3.
    C10, C30, angular_spread = -300.0, 1e7, 0.08  # mrad
    wavelength = relativistic_wavelength(ENERGY)
    a_s = angular_spread * 1e-3

    def chi(ax, ay):
        a2 = ax**2 + ay**2
        return 2 * np.pi / wavelength * (C10 * a2 / 2 + C30 * a2**2 / 4)

    x, w = np.polynomial.hermite.hermgauss(60)
    bx, by = a_s * x[:, None], a_s * x[None, :]
    weights = w[:, None] * w[None, :] / np.pi

    alpha = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0]) * 1e-3
    average = np.array(
        [np.abs(np.sum(weights * np.exp(-1j * (chi(a + bx, by) - chi(a, 0.0))))) for a in alpha]
    )
    assert average.min() < 0.2

    envelope = SpatialEnvelope(
        angular_spread, energy=ENERGY, C10=C10, C30=C30
    )._evaluate_from_angular_grid(alpha, np.zeros_like(alpha))
    assert np.allclose(envelope, average, atol=2e-3)


# ---------------------------------------------------------------------------
# 6. Scherzer defocus and point resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("energy, Cs", [(200e3, 1e7), (300e3, 1.2e7), (80e3, 5e6)])
def test_scherzer_defocus_textbook(energy, Cs):
    # abTEM uses the extended-Scherzer variant Delta_f = sqrt(1.5 Cs lambda)
    # (= 1.22 sqrt(Cs lambda); textbook rounding "1.2 sqrt(Cs lambda)").
    # With defocus = -C10 this must be an *underfocus*: C10 < 0 for Cs > 0.
    wavelength = relativistic_wavelength(energy)
    ctf = CTF(energy=energy, Cs=Cs, defocus="scherzer")
    assert np.isclose(ctf.defocus, np.sqrt(1.5 * Cs * wavelength), rtol=1e-6)
    assert np.isclose(ctf.C10, -1.2 * np.sqrt(Cs * wavelength), rtol=0.025)
    assert np.isclose(ctf.scherzer_defocus, ctf.defocus)


@pytest.mark.parametrize("energy, Cs", [(200e3, 1e7), (300e3, 1.2e7)])
def test_scherzer_first_zero_and_phase_minimum(energy, Cs):
    # At Delta_f = sqrt(1.5 Cs lambda), chi(k) = pi lambda k^2 (Cs lambda^2 k^2 / 2
    # - Delta_f) has its first nonzero root at 1/k0 = (Cs lambda^3 / 6)^(1/4)
    # = 0.639 Cs^(1/4) lambda^(3/4) (the "0.64-0.66" point-resolution formula),
    # and a minimum chi = -3 pi / 4 at k^2 = Delta_f / (Cs lambda^2). Unlike the
    # root, the depth of the minimum depends on the 2 pi / lambda prefactor.
    wavelength = relativistic_wavelength(energy)
    ctf = CTF(energy=energy, Cs=Cs, defocus="scherzer")

    analytic_resolution = (Cs * wavelength**3 / 6) ** 0.25
    assert np.isclose(ctf.point_resolution, analytic_resolution, rtol=1e-6)
    assert np.isclose(point_resolution(Cs, energy), analytic_resolution, rtol=1e-6)
    assert np.isclose(
        ctf.point_resolution, 0.66 * Cs**0.25 * wavelength**0.75, rtol=0.035
    )

    gpts = 4001
    max_angle = 1.5e3 * wavelength / analytic_resolution  # mrad
    profile = ctf.profiles(gpts=gpts, max_angle=max_angle).array  # = -sin(chi)
    k = np.linspace(0, max_angle * 1e-3, gpts) / wavelength
    dk = k[1]

    # -sin(chi) >= 0 on (0, k0): the first sign change is the chi = 0 root.
    first_crossing = np.where(np.diff(np.sign(profile[1:])) != 0)[0][0] + 1
    assert abs(k[first_crossing] - 1 / analytic_resolution) <= dk

    k_min = np.sqrt(ctf.defocus / (Cs * wavelength**2))
    i_min = np.argmin(np.abs(k - k_min))
    assert np.isclose(profile[i_min], np.sin(3 * np.pi / 4), atol=1e-3)
    # ... and it is a local minimum of -sin(chi) (chi passes -pi/2 twice).
    assert profile[i_min] < profile[i_min // 2] and profile.max() > 0.999


def test_scherzer_string_via_C10_symbol():
    # "scherzer" given as C10 must produce the same (underfocused) lens as
    # "scherzer" given as defocus, since defocus = -C10.
    wavelength = relativistic_wavelength(ENERGY)
    ab = Aberrations(energy=ENERGY, Cs=1e7, C10="scherzer")
    assert np.isclose(ab.C10, -np.sqrt(1.5 * 1e7 * wavelength), rtol=1e-6)


def test_scherzer_string_independent_of_keyword_order():
    # Scherzer defocus depends on Cs, so it must not matter whether Cs is given
    # before or after defocus="scherzer".
    expected = scherzer_defocus(1e7, ENERGY)
    assert np.isclose(Aberrations(energy=ENERGY, defocus="scherzer", Cs=1e7).defocus, expected)
    ctf = CTF(energy=ENERGY, aberration_coefficients={"defocus": "scherzer", "C30": 1e7})
    assert np.isclose(ctf.defocus, expected)


# ---------------------------------------------------------------------------
# 7. Apertures
# ---------------------------------------------------------------------------


@devices
def test_hard_aperture_transmits_exactly_the_disk(device):
    # Independent count on a numpy fftfreq grid: transmitted pixels are exactly
    # those with alpha < semiangle_cutoff (no pixel lies near the edge, see guard).
    gpts, sampling, cutoff = (96, 64), (0.07, 0.11), 21.3
    alpha, _ = cartesian_angular_grid(gpts, sampling, ENERGY)
    assert np.abs(alpha - cutoff * 1e-3).min() > 1e-6  # edge not ambiguous

    ap = Aperture(cutoff, soft=False, energy=ENERGY, gpts=gpts, sampling=sampling)
    kernel = asnumpy(ab_kernel(ap, device))
    assert set(np.unique(kernel)) == {0.0, 1.0}
    assert np.array_equal(kernel == 1.0, alpha < cutoff * 1e-3)


def test_hard_aperture_edge_is_inclusive():
    # Pinned convention: a pixel exactly at the cutoff angle is transmitted.
    cutoff = 0.025
    alpha = np.array([0.0, np.nextafter(cutoff, 0), cutoff, np.nextafter(cutoff, 1)])
    assert np.array_equal(hard_aperture(alpha, cutoff), [1.0, 1.0, 1.0, 0.0])


@devices
def test_soft_aperture_edge_profile_and_area(device):
    # Pinned form: a linear ramp one pixel wide along the radial direction,
    # centred on the cutoff (value 1/2 at alpha = cutoff), where the pixel width
    # along azimuth phi is d = sqrt((cos phi da_x)^2 + (sin phi da_y)^2).
    # Independent oracle for the anti-aliasing: sum(kernel) * da_x * da_y
    # approximates the disk area pi cutoff^2 far better than the hard count.
    gpts, sampling, cutoff = (256, 192), (0.1, 0.15), 20.0
    wavelength = relativistic_wavelength(ENERGY)
    alpha, phi = cartesian_angular_grid(gpts, sampling, ENERGY)
    da = wavelength / (np.array(gpts) * np.array(sampling))  # rad
    d = np.hypot(np.cos(phi) * da[0], np.sin(phi) * da[1])
    c = cutoff * 1e-3

    ap = Aperture(cutoff, soft=True, energy=ENERGY, gpts=gpts, sampling=sampling)
    kernel = asnumpy(ab_kernel(ap, device))

    assert np.all(kernel[alpha <= c - d / 2] == 1.0)
    assert np.all(kernel[alpha >= c + d / 2] == 0.0)
    edge = np.abs(alpha - c) < d / 2
    assert edge.sum() > 50
    assert np.allclose(kernel[edge], 0.5 + (c - alpha[edge]) / d[edge], atol=1e-5)
    assert kernel[0, 0] == 1.0

    disk_area = np.pi * c**2
    soft_area = kernel.sum() * da[0] * da[1]
    hard_area = (alpha <= c).sum() * da[0] * da[1]
    assert abs(soft_area - disk_area) / disk_area < 2e-3
    assert abs(soft_area - disk_area) < abs(hard_area - disk_area)

    # The standalone function gives 1/2 exactly at the cutoff for any azimuth.
    half = soft_aperture(
        np.full((2, 2), c), np.array([[0.1, 0.2], [0.3, 1.2]]), c, (1.0, 1.0)
    )
    # Element [0, 0] is forced to 1 as the DC pixel; the rest are edge pixels.
    assert np.allclose(half.ravel()[1:], 0.5)


# ---------------------------------------------------------------------------
# 8. Frozen-phonon displacements
# ---------------------------------------------------------------------------
# Oracle: the FrozenPhonons docstring. Displacements are Gaussian with the given
# standard deviation along each axis named in `directions`; other axes are left
# exactly untouched. Displacements are measured from the known input positions
# (mean zero by construction), so the rms over n samples estimates sigma with
# relative standard error 1 / sqrt(2 n); tolerances are 5 such errors.


def _displacements(frozen_phonons, atoms):
    trajectory = frozen_phonons.to_atoms_ensemble().trajectory
    return np.stack([config.positions for config in trajectory]) - atoms.positions


def _rms(displacements):
    return np.sqrt(np.mean(displacements**2))


@pytest.mark.parametrize(
    "directions, moved", [("xy", (0, 1)), ("x", (0,)), ("yz", (1, 2)), ("xyz", (0, 1, 2))]
)
def test_frozen_phonons_directions(directions, moved):
    atoms = ase.build.bulk("Au", cubic=True) * (2, 2, 2)  # 32 atoms
    sigma, num_configs = 0.08, 400
    fp = FrozenPhonons(
        atoms, num_configs=num_configs, sigmas=sigma, directions=directions, seed=7
    )
    d = _displacements(fp, atoms)

    rel_err = 1 / np.sqrt(2 * num_configs * len(atoms))  # ~0.6 %
    for axis in range(3):
        if axis in moved:
            assert abs(_rms(d[..., axis]) / sigma - 1) < 5 * rel_err, axis
        else:
            assert np.array_equal(d[..., axis], np.zeros_like(d[..., axis])), axis

    # Independent Gaussian components: the x-y sample correlation vanishes
    # (standard error 1 / sqrt(n) ~ 0.9 %).
    if moved[:2] == (0, 1):
        corr = np.corrcoef(d[..., 0].ravel(), d[..., 1].ravel())[0, 1]
        assert abs(corr) < 5 / np.sqrt(num_configs * len(atoms))


def test_frozen_phonons_anisotropic_per_element_sigmas():
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)  # 4 Na + 4 Cl
    sigmas = {"Na": (0.05, 0.10, 0.15), "Cl": (0.12, 0.03, 0.08)}
    num_configs = 1000
    symbols = np.array(atoms.symbols)
    rel_err = 1 / np.sqrt(2 * num_configs * 4)  # ~1.1 %

    fp = FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas, directions="xyz", seed=3)
    d = _displacements(fp, atoms)
    for symbol, expected in sigmas.items():
        for axis in range(3):
            measured = _rms(d[:, symbols == symbol, axis])
            assert abs(measured / expected[axis] - 1) < 5 * rel_err, (symbol, axis)

    # With directions="xy" the (nonzero) z sigmas must be ignored exactly.
    fp = FrozenPhonons(atoms, num_configs=50, sigmas=sigmas, directions="xy", seed=3)
    d = _displacements(fp, atoms)
    assert np.array_equal(d[..., 2], np.zeros_like(d[..., 2]))
    assert np.all(np.abs(d[..., :2]).max(axis=(0, 1)) > 0)
