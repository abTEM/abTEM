"""Independent-oracle physics tests for core-loss transition potentials.

Every expected value here comes from something other than abTEM's own code
path: an analytic result, a symmetry, a selection rule, an independently
implemented textbook model, or a second abTEM algorithm that shares no code
with the one under test. None of the expected values is pasted output.

The site layouts are deliberately asymmetric (a general, off-pixel position on
a non-square, anisotropically sampled grid), because x<->y-symmetric,
inversion-symmetric, on-pixel layouts cannot see an x/y swap, a sign flip of
the site position or a sign flip of a sub-pixel shift.
"""

import ast
import functools
import sys

import ase
import numpy as np
import pytest

import abtem
from abtem.core import config
from abtem.core.backend import asnumpy
from abtem.core.energy import energy2wavelength
from abtem.inelastic.core_loss import SubshellTransitions, TransitionPotential

from utils import gpu

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

pytestmark = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")

devices = pytest.mark.parametrize("device", ["cpu", gpu])

# CODATA constants, independent of abTEM and ASE.
BOHR = 0.529177210903  # Bohr radius [Å]
RYDBERG = 13.605693123  # Rydberg energy [eV]
MC2 = 510998.950  # electron rest energy [eV]

# Asymmetric test geometry: non-square gpts, anisotropic sampling
# (0.150 x 0.160 Å) and a site at a general position that sits on no pixel
# (9.133, 18.19 pixels) and has x != y.
GPTS = (48, 40)
EXTENT = (7.2, 6.4)
SITE = (1.37, 2.91)


@functools.lru_cache(maxsize=None)
def _transitions(Z, epsilon, order=1):
    """Bound and continuum states from the GPAW atomic solvers (cached)."""
    return tuple(
        SubshellTransitions(
            Z=Z, n=1, l=0, order=order, epsilon=epsilon
        ).get_transitions()
    )


def _transition_potential(Z, energy, gpts=None, extent=None, epsilon=10, order=1):
    return TransitionPotential(
        Z, list(_transitions(Z, epsilon, order)), gpts=gpts, extent=extent,
        energy=energy,
    )


def _energy_loss(Z, epsilon, order=1):
    """Total energy loss [eV]: 1s binding energy plus the continuum energy."""
    bound = _transitions(Z, epsilon, order)[0][0]
    return -bound.energy + epsilon


def _final_state(transition_potential, index):
    """(l', ml') of transition ``index``, as its user-facing axis label says.

    Parsed from the ``(l,ml)→(l',ml')`` ensemble-axis label, i.e. the
    documented labelling a user reads, not an internal attribute.
    """
    label = transition_potential.ensemble_axes_metadata[0].values[index]
    return tuple(ast.literal_eval(label.split("→")[1].strip()))


def _circular_centroid(intensity, extent):
    """Centroid of a periodic image, one axis at a time.

    Uses the phase of the first Fourier coefficient of each projected profile,
    ``arg(sum_x I(x) exp(2 pi i x / L))``. For a profile symmetric about ``x0``
    on the circle this is exactly ``x0`` -- no windowing bias and no seam at
    the cell boundary -- which a plain intensity-weighted mean is not.
    """
    centroid = []
    for axis, length in enumerate(extent):
        n = intensity.shape[axis]
        x = np.arange(n) * length / n
        profile = intensity.sum(axis=1 - axis)
        phase = np.angle((profile * np.exp(2j * np.pi * x / length)).sum())
        centroid.append((phase * length / (2 * np.pi)) % length)
    return np.array(centroid)


def _periodic_distance(a, b, extent):
    extent = np.asarray(extent)
    d = (np.asarray(a) - np.asarray(b) + extent / 2) % extent - extent / 2
    return np.abs(d)


def _kinematics(energy):
    """gamma, v^2/c^2 and T = m0 v^2 / 2 [eV] of the incident electron."""
    gamma = 1 + energy / MC2
    beta2 = 1 - 1 / gamma**2
    return gamma, beta2, MC2 * beta2 / 2


def _theta_e(energy_loss, energy):
    """Characteristic angle theta_E = E / (gamma m0 v^2) [rad].

    Egerton, *EELS in the Electron Microscope*, 3rd ed. (2011), Sec. 3.3: to
    first order in E/E0, k0 - k1 = E dk/dE = E / (hbar v); dividing by
    k0 = gamma m0 v / hbar gives theta_E = E / (gamma m0 v^2).
    """
    gamma, beta2, _ = _kinematics(energy)
    return energy_loss / (gamma * MC2 * beta2)


def _angles(gpts, extent, energy):
    """Scattering angle [rad] of every pixel, in unshifted FFT order."""
    wavelength = energy2wavelength(energy)
    kx = np.fft.fftfreq(gpts[0], extent[0] / gpts[0])
    ky = np.fft.fftfreq(gpts[1], extent[1] / gpts[1])
    return np.sqrt(kx[:, None] ** 2 + ky[None] ** 2) * wavelength


# ---------------------------------------------------------------------------
# Task 1: an asymmetric, off-pixel scattering site
# ---------------------------------------------------------------------------


@devices
def test_scattered_intensity_is_centred_on_off_pixel_site(device):
    """The inelastically scattered plane wave is centred on the site.

    Oracle (symmetry): a transition potential for a free atom is invariant
    under rotation about the atom by pi, so the scattered intensity of a
    plane wave is inversion-symmetric about the site. The circular centroid
    of such an image is exactly the site, for any grid, including one whose
    two axes differ in length and sampling. Components with ml' = 0 carry no
    phase winding and are single-peaked at the atom, so their brightest
    pixel is the pixel nearest the site. (The ml' = +-1 components are
    vortices with a node at the atom, so the ml'-summed intensity is not
    required to peak there.)

    Catches: sites swapped x<->y in ``TransitionPotentialArray.scatter``, a
    transition potential placed at -r instead of r.
    """
    tp = _transition_potential(6, 100e3, GPTS, EXTENT).build()
    waves = abtem.PlaneWave(
        energy=100e3, gpts=GPTS, extent=EXTENT, device=device
    ).build(lazy=False)

    site_pixel = np.array(SITE) / np.array(waves.sampling)
    # Precondition: the site is on no pixel, in either direction.
    assert np.all(np.abs(site_pixel - np.rint(site_pixel)) > 0.1)

    scattered = asnumpy(tp.scatter(waves, [SITE]).array)
    assert scattered.shape == (len(tp),) + GPTS

    intensity = np.abs(scattered) ** 2

    # The ml'-summed intensity, and each component separately.
    for image in [intensity.sum(0), *intensity]:
        centroid = _circular_centroid(image, EXTENT)
        np.testing.assert_array_less(
            _periodic_distance(centroid, SITE, EXTENT), 1e-3
        )

    nearest_pixel = tuple(int(i) for i in np.rint(site_pixel))
    for i in range(len(tp)):
        if _final_state(tp, i)[1] != 0:
            continue
        peak = tuple(int(j) for j in np.unravel_index(np.argmax(intensity[i]), GPTS))
        assert peak == nearest_pixel


@devices
def test_prism_eels_matches_multislice_eels_at_off_pixel_site(device):
    """PRISM-EELS at interpolation=1 equals multislice-EELS on an off-pixel site.

    Oracle (independent method): at interpolation=1 the PRISM reduction is an
    exact rewriting of the multislice calculation (Brown et al., Phys. Rev.
    Research 1, 033186 (2019)), and the two drivers place the transition
    potential by different code: PRISM with an integer crop plus a sub-pixel
    Fourier shift, multislice with one Fourier shift of the full position.
    The site is off-pixel, so the sub-pixel shift is non-zero and exercised.

    Catches: a negated PRISM sub-pixel shift; an x/y swap or -r in scatter.
    """
    energy = 100e3
    atoms = ase.Atoms(
        "C", positions=[(SITE[0], SITE[1], 1.0)],
        cell=(EXTENT[0], EXTENT[1], 2.0), pbc=True,
    )
    potential = abtem.Potential(
        atoms, gpts=GPTS, slice_thickness=2.0, device=device
    )
    site_pixel = np.array(SITE) / np.array(potential.sampling)
    assert np.all(np.abs(site_pixel - np.rint(site_pixel)) > 0.1)

    tp = _transition_potential(6, energy)
    detector = abtem.PixelatedDetector(max_angle=None, to_cpu=True)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(3, 3), fractional=True,
        potential=potential, endpoint=False,
    )

    probe = abtem.Probe(energy=energy, semiangle_cutoff=25, device=device)
    probe.grid.match(potential)
    multislice = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp, scan=scan,
        detectors=detector, double_channel=False, threshold=1.0, lazy=False,
    ).compute()

    s_matrix = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=25,
        interpolation=1, downsample=False, device=device,
    )
    prism = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=scan, detectors=detector
    )

    a = np.asarray(multislice.array)
    b = np.asarray(prism.array)
    assert a.shape == b.shape
    scale = np.abs(a).max()
    assert scale > 0
    # Float precision: limited by single-precision FFT round-off.
    assert np.abs(a - b).max() <= 1e-5 * scale


# ---------------------------------------------------------------------------
# Task 2: selection rules and rotational symmetry
# ---------------------------------------------------------------------------


def _rotate_90(array):
    """Rotate a square, unshifted-FFT-ordered array by +90 deg about index 0.

    ``(R f)(k) = f(R^-1 k)`` with ``R^-1 (kx, ky) = (ky, -kx)``, i.e.
    ``out[i, j] = f[j, -i mod N]``. Exact on a square grid with equal
    sampling, where the fftfreq lattice maps onto itself.
    """
    n = array.shape[-1]
    assert array.shape[-2] == n
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return array[..., j, (-i) % n]


@pytest.fixture(scope="module")
def square_k_edge():
    with config.set({"precision": "float64"}):
        return _transition_potential(6, 100e3, (64, 64), (8.0, 8.0)).build()


def test_k_edge_ml_summed_intensity_is_rotation_invariant(square_k_edge):
    """The ml'-summed 1s -> p intensity is cylindrically symmetric.

    Oracle (symmetry / Unsold's theorem): sum_ml' |Y_1^ml'|^2 is constant, so
    the incoherent sum over the three final-state ml' of a closed initial
    subshell is invariant under any rotation about the beam axis, in both
    reciprocal and real space. A real (Cartesian) orbital combination such as
    p_x ~ Y_1^-1 - Y_1^+1 is not: rotation by 90 deg maps p_x onto p_y.
    (Each complex-basis ml' component alone *is* cylindrically symmetric in
    intensity -- it is a vortex -- see the next test for what it must obey.)
    """
    tp = square_k_edge
    p = [i for i in range(len(tp)) if _final_state(tp, i)[0] == 1]
    assert sorted(_final_state(tp, i)[1] for i in p) == [-1, 0, 1]

    array = np.asarray(tp.array)
    reciprocal = (np.abs(array[p]) ** 2).sum(0)
    real = (np.abs(np.fft.ifft2(array[p])) ** 2).sum(0)
    for image in (reciprocal, real):
        assert np.abs(_rotate_90(image) - image).max() <= 1e-9 * image.max()

    by_m = {_final_state(tp, i)[1]: array[i] for i in p}
    for cartesian in (by_m[-1] - by_m[1], by_m[-1] + by_m[1]):
        image = np.abs(np.fft.ifft2(cartesian)) ** 2
        assert np.abs(_rotate_90(image) - image).max() > 0.5 * image.max()


def test_k_edge_ml_components_carry_angular_momentum_minus_ml(square_k_edge):
    """Each ml' component is an eigenstate of L_z with eigenvalue -ml'.

    Oracle (angular-momentum conservation about the beam axis): the atom goes
    from ml = 0 to ml', so the fast electron, initially a plane wave with no
    orbital angular momentum, leaves with -ml'. A function with phase winding
    exp(i w phi) obeys (R_90 f) = exp(-i w pi / 2) f, so the component
    labelled ml' must pick up exactly i**ml' under a +90 deg rotation. This
    pins the (l', ml') axis labels to the physics, including the sign of ml'.
    """
    tp = square_k_edge
    array = np.asarray(tp.array)
    for i in range(len(tp)):
        mlprime = _final_state(tp, i)[1]
        f = array[i]
        expected = (1j**mlprime) * f
        assert np.abs(_rotate_90(f) - expected).max() <= 1e-9 * np.abs(f).max()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "1s -> eps-s monopole channel does not vanish as q -> 0: the bound "
        "(gpaw AllElectron) and continuum (AllElectronAtom, scalar-relativistic, "
        "with the 1.02 factor in radial_schroedinger_equation) states are not "
        "orthogonal -- <eps s|1s> = -3.7e-3 for C at eps = 50 eV (+9.2e-5 "
        "without the 1.02) -- giving a spurious 1/q^4 monopole term: "
        "monopole/dipole = 0.57 (C) and 0.063 (Si) at the central pixel at "
        "300 kV. With the 1.02 set to 1.00 both cases pass"
    ),
)
@pytest.mark.parametrize("Z", [6, 14])
def test_k_edge_monopole_channel_vanishes_at_small_q(Z):
    """The dipole-forbidden 1s -> eps-s channel is negligible as q -> 0.

    Oracle (selection rule / orthogonality): for eigenstates of one atomic
    Hamiltonian <eps s|1s> = 0, so the monopole matrix element
    <eps s|exp(iq.r)|1s> is O(q^2) while the dipole one is O(q). At the
    central pixel q = q_z = k0 theta_E, with q_z a0 ~ 0.1 (C) to 0.6 (Si) at
    300 kV, the monopole-to-dipole intensity ratio is of order
    (q_z <r^2> / <r>)^2 ~ (q_z a0 / Z_s)^2, far below the 1e-2 asserted.
    """
    with config.set({"precision": "float64"}):
        tp = _transition_potential(
            Z, 300e3, (64, 64), (40.0, 40.0), epsilon=50
        ).build()
    intensity = np.abs(np.asarray(tp.array)) ** 2
    index = {_final_state(tp, i): i for i in range(len(tp))}
    monopole = intensity[index[(0, 0)], 0, 0]
    dipole = intensity[index[(1, 0)], 0, 0]
    assert monopole / dipole < 1e-2


# ---------------------------------------------------------------------------
# Task 3: absolute cross sections against Egerton's hydrogenic K-shell model
# ---------------------------------------------------------------------------


def _hydrogenic_k_shell_gos(Q, energy_loss, Z):
    """Hydrogenic K-shell generalised oscillator strength df/dE [1/eV].

    Egerton, *EELS in the Electron Microscope*, 3rd ed. (2011), Sec. 3.6.1,
    and the SIGMAK model (R. F. Egerton, Ultramicroscopy 4, 169 (1979)): the
    exact hydrogen 1s GOS (Bethe 1930; Inokuti, Rev. Mod. Phys. 43, 297
    (1971)), scaled to an effective nuclear charge Z_s = Z - 0.3 (Slater's 1s
    screening), and analytically continued below the hydrogenic threshold
    E < Z_s^2 R, which is how the model represents outer-shell screening.
    Both 1s electrons are counted.

    Parameters
    ----------
    Q : np.ndarray
        (q a0)^2.
    energy_loss : float
        Energy loss E [eV].
    """
    zs = Z - 0.3
    q = Q / zs**2
    e = energy_loss / (zs**2 * RYDBERG)
    kappa2 = e - 1.0
    x = q - kappa2 + 1.0
    if kappa2 >= 0:
        kappa = np.sqrt(kappa2)
        c = np.exp(-2.0 / kappa * np.arctan2(2 * kappa, x))
        d = 1.0 - np.exp(-2 * np.pi / kappa)
    else:
        nu = np.sqrt(-kappa2)
        c = np.exp(-1.0 / nu * np.log((x + 2 * nu) / (x - 2 * nu)))
        d = 1.0
    per_rydberg = 128.0 * e * (q + e / 3.0) * c / ((x**2 + 4 * kappa2) ** 3 * d)
    return 2 * per_rydberg / (zs**2 * RYDBERG)


def _bethe_double_differential(theta, energy_loss, Z, energy):
    """d^2 sigma / (dE d^2 theta) [Å^2 / eV / rad^2], relativistic Bethe theory.

    Inokuti (1971) with Egerton (2011) Sec. 3.6: dsigma/dE =
    (4 pi a0^2 R^2 / (E T)) int (df/dE) d(ln Q), with T = m0 v^2 / 2 and
    Q = (q a0)^2, q^2 = k0^2 (theta^2 + theta_E^2). Using
    d(ln Q) = d^2 theta / (pi (theta^2 + theta_E^2)) gives the integrand per
    unit solid angle returned here.
    """
    _, _, T = _kinematics(energy)
    k0 = 2 * np.pi / energy2wavelength(energy)
    theta_e = _theta_e(energy_loss, energy)
    Q = (k0 * BOHR) ** 2 * (theta**2 + theta_e**2)
    gos = _hydrogenic_k_shell_gos(Q, energy_loss, Z)
    return 4 * BOHR**2 * RYDBERG**2 / (energy_loss * T) * gos / (
        theta**2 + theta_e**2
    )


def _scattered_diffraction_intensity(tp, gpts, extent, energy):
    """|Psi_n(k)|^2 of the scattered waves, normalised per unit incident flux.

    A unit-amplitude plane wave (unit probability density per Å^2) is
    scattered by one atom. By Parseval, sum_r |psi_n|^2 dA =
    (dA / N) sum_k |Psi_n(k)|^2, so the returned array summed over pixels and
    channels is the cross section per unit energy loss [Å^2 / eV] -- the
    convention of the abTEM core-loss tutorial,
    dsigma/dE = sigma^2 int A(k) sum_f |psi_f(k)|^2 dk.
    """
    waves = abtem.PlaneWave(energy=energy, gpts=gpts, extent=extent).build(
        lazy=False
    )
    scattered = asnumpy(tp.scatter(waves, [[0.0, 0.0]]).array)
    n = np.prod(gpts)
    area = extent[0] * extent[1]
    return np.abs(np.fft.fft2(scattered, axes=(-2, -1))) ** 2 * area / n**2


@pytest.mark.slow
@pytest.mark.parametrize("energy", [100e3, 200e3])
@pytest.mark.parametrize("Z", [6, 14])
def test_k_edge_cross_section_matches_hydrogenic_model(Z, energy):
    """Absolute energy-differential K-shell cross section, 50 eV above the edge.

    Oracle: Egerton's hydrogenic (SIGMAK) model, implemented here from its
    published equations (``_hydrogenic_k_shell_gos``), at the same energy
    loss, incident energy and collection semi-angle (beta = 20 mrad). The
    oracle is summed on the same angular pixels as abTEM, so the comparison
    carries no quadrature error.

    Tolerance +-25%: the hydrogenic model neglects the real atomic potential
    and final-state structure; Egerton (2011, Secs. 3.6.1, 4.5.1) puts its
    agreement with Hartree-Slater K-shell cross sections at the 10-20% level
    away from threshold, which is why 50 eV above the edge is used rather
    than the near-edge region. (Commit 29555974 independently found abTEM K
    edges at 0.83-1.14 of Bote & Salvat's tabulation.)

    Catches: overlap integral x3 (intensity x9); dropping the relativistic
    mass correction in build (intensity / gamma^2: / 1.43 at 100 kV, / 1.94
    at 200 kV; test_k_edge_cross_section_relativistic_scaling also sees it,
    independently of the absolute scale).
    """
    epsilon = 50
    beta = 0.02
    gpts = (160, 160)
    extent = (20.0, 20.0)
    with config.set({"precision": "float64"}):
        tp = _transition_potential(
            Z, energy, gpts, extent, epsilon=epsilon, order=2
        ).build()
        intensity = _scattered_diffraction_intensity(tp, gpts, extent, energy)

    theta = _angles(gpts, extent, energy)
    # The aperture must lie well inside the form factors' 2/3-Nyquist cutoff.
    assert beta < 0.6 * theta.max()
    aperture = theta <= beta

    cross_section = (intensity.sum(0) * aperture).sum()

    wavelength = energy2wavelength(energy)
    solid_angle = (wavelength / extent[0]) * (wavelength / extent[1])
    energy_loss = _energy_loss(Z, epsilon, order=2)
    oracle = (
        _bethe_double_differential(theta, energy_loss, Z, energy) * aperture
    ).sum() * solid_angle

    ratio = cross_section / oracle
    assert 0.75 < ratio < 1.25, f"abTEM / hydrogenic = {ratio:.3f}"


@pytest.mark.slow
def test_k_edge_cross_section_relativistic_scaling():
    """How the dipole cross section scales with the incident energy.

    Oracle (analytic kinematics): in the dipole region the GOS is the optical
    oscillator strength, independent of q, so Bethe's
    dsigma/dE = (4 pi a0^2 R^2 / (E T)) int (df/dE) d(ln Q), T = m0 v^2 / 2,
    reduces to dsigma/dE ~ (1 / v^2) int d^2 theta / (theta^2 + theta_E^2)
    -> (pi / v^2) ln(1 + beta^2 / theta_E^2) in the continuum limit, with
    theta_E = E / (gamma m0 v^2). Everything atomic cancels in the ratio
    between two incident energies; the angular integral is summed on the
    same pixels as abTEM rather than taken in the continuum limit.

    beta = 3 mrad keeps (q_max a0 / Z_s)^2 below 0.01 at 300 kV, where the
    hydrogenic GOS falls by a few percent at the aperture edge and its effect
    on the ratio is < 0.5%, inside the 1% tolerance. Only the l' = 1
    channels are used; the l' = 0 channel is dipole-forbidden (see
    test_k_edge_monopole_channel_vanishes_at_small_q).

    Catches: dropping the relativistic mass correction (the 300/60 kV ratio
    changes by (gamma_300 / gamma_60)^2 = 2.0).
    """
    Z, epsilon, beta = 6, 50, 0.003
    gpts = (128, 128)
    extent = (100.0, 100.0)
    energy_loss = _energy_loss(Z, epsilon)

    results = {}
    for energy in (60e3, 100e3, 300e3):
        with config.set({"precision": "float64"}):
            tp = _transition_potential(
                Z, energy, gpts, extent, epsilon=epsilon
            ).build()
            intensity = _scattered_diffraction_intensity(
                tp, gpts, extent, energy
            )
        dipole = [i for i in range(len(tp)) if _final_state(tp, i)[0] == 1]

        theta = _angles(gpts, extent, energy)
        assert beta < 0.6 * theta.max()
        aperture = theta <= beta
        simulated = (intensity[dipole].sum(0) * aperture).sum()

        _, beta2, _ = _kinematics(energy)
        theta_e = _theta_e(energy_loss, energy)
        wavelength = energy2wavelength(energy)
        solid_angle = (wavelength / extent[0]) * (wavelength / extent[1])
        analytic = (aperture / (theta**2 + theta_e**2)).sum() * solid_angle / beta2
        results[energy] = (simulated, analytic)

    for energy in (60e3, 300e3):
        simulated = results[energy][0] / results[100e3][0]
        analytic = results[energy][1] / results[100e3][1]
        assert simulated / analytic == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Task 4: dipole-limit angular shape
# ---------------------------------------------------------------------------


def _fit_theta_e(intensity, theta, theta_e_guess, max_angle):
    """theta_E from 1 / I = a (theta^2 + theta_E^2) (1 + b theta^2 + ...).

    A quadratic in theta^2 absorbs the slow q-dependence of the GOS; then
    theta_E^2 = c0 / c1 to first order in that dependence.
    """
    mask = theta <= max_angle
    x = theta[mask] ** 2 / theta_e_guess**2
    y = 1 / intensity[mask]
    _, c1, c0 = np.polyfit(x, y / y.max(), 2)
    return np.sqrt(c0 / c1) * theta_e_guess


def _dipole_shape_case(Z, energy, epsilon=50):
    energy_loss = _energy_loss(Z, epsilon)
    theta_e = _theta_e(energy_loss, energy)
    # Angular pixel theta_E / 4, so ~440 pixels lie inside 3 theta_E.
    size = 4 * energy2wavelength(energy) / theta_e
    gpts = (200, 200)
    with config.set({"precision": "float64"}):
        tp = _transition_potential(
            Z, energy, gpts, (size, size), epsilon=epsilon
        ).build()
    intensity = np.abs(np.asarray(tp.array)) ** 2
    theta = _angles(gpts, (size, size), energy)
    return tp, intensity, theta, theta_e


@pytest.mark.parametrize("Z, energy", [(6, 100e3), (6, 300e3), (14, 100e3)])
def test_k_edge_dipole_intensity_is_lorentzian_in_theta(Z, energy):
    """|H(k)|^2 ~ 1 / (theta^2 + theta_E^2) at small angles, theta_E relativistic.

    Oracle (Bethe dipole limit): |H|^2 ~ |<f|exp(iq.r)|i>|^2 / q^4 ~ 1 / q^2
    as q -> 0, with q^2 = k0^2 (theta^2 + theta_E^2) and theta_E =
    E / (gamma m0 v^2) (Egerton 2011; the neglected terms are
    O(theta_E / gamma^2), < 0.1% here). The non-relativistic E / (2 E0)
    differs by 9% at 100 kV and 36% at 300 kV, so the 1% tolerance resolves
    the relativistic kinematics. Fitted over theta <= 3 theta_E, where the
    GOS changes by < 1%.
    """
    tp, intensity, theta, theta_e = _dipole_shape_case(Z, energy)
    dipole = [i for i in range(len(tp)) if _final_state(tp, i)[0] == 1]
    fitted = _fit_theta_e(intensity[dipole].sum(0), theta, theta_e, 3 * theta_e)
    assert fitted == pytest.approx(theta_e, rel=0.01)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "spurious 1s -> eps-s monopole (non-orthogonal bound and continuum "
        "states, see test_k_edge_monopole_channel_vanishes_at_small_q) adds a "
        "1/q^4 term that pulls the fitted theta_E of the full C K edge 22% "
        "low at 300 kV (12% at 100 kV)"
    ),
)
def test_k_edge_total_intensity_is_lorentzian_in_theta():
    """As above, but for the full K edge including the l' = 0 channel.

    Oracle: the l' = 0 channel is dipole-forbidden and O(q^2) relative to the
    dipole near q = 0, so it cannot change the small-angle shape.
    """
    _, intensity, theta, theta_e = _dipole_shape_case(6, 300e3)
    fitted = _fit_theta_e(intensity.sum(0), theta, theta_e, 3 * theta_e)
    assert fitted == pytest.approx(theta_e, rel=0.01)


# ---------------------------------------------------------------------------
# Task 5: scattering sites follow frozen-phonon displacements
# ---------------------------------------------------------------------------

_FP_SITE = (3.37, 2.91)


def _single_carbon_atoms():
    return ase.Atoms(
        "C", positions=[(_FP_SITE[0], _FP_SITE[1], 1.0)],
        cell=(EXTENT[0], EXTENT[1], 2.0), pbc=True,
    )


def _inelastic_and_elastic_centres(potential, device):
    """Centres of the scattered intensity and of the elastic phase, per config.

    A single transition (1s -> p, ml' = 0) is used so that the WavesDetector's
    coherent sum over the transition axis is just that one wave.
    """
    tp = _transition_potential(6, 100e3, GPTS, EXTENT).build()
    tp = tp[[i for i in range(len(tp)) if _final_state(tp, i) == (1, 0)]]

    waves = abtem.PlaneWave(
        energy=100e3, gpts=GPTS, extent=EXTENT, device=device
    ).build(lazy=False)
    inelastic = waves.transition_potential_multislice(
        potential, tp, detectors=abtem.WavesDetector(), double_channel=False
    ).compute()
    elastic = waves.multislice(potential).compute()

    inelastic = asnumpy(inelastic.array).reshape((-1,) + GPTS)
    elastic = asnumpy(elastic.array).reshape((-1,) + GPTS)
    return (
        [_circular_centroid(np.abs(a) ** 2, EXTENT) for a in inelastic],
        [_circular_centroid(np.abs(np.angle(a)), EXTENT) for a in elastic],
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Waves.transition_potential_multislice resolves sites=None once, from "
        "the whole frozen-phonon ensemble (abtem/waves.py, before "
        "_prebuild_reused_potential), and FrozenPhonons.randomize uses seed[0]: "
        "every configuration scatters at configuration 0's displaced position "
        "while its elastic potential uses its own"
    ),
)
@devices
@pytest.mark.filterwarnings("ignore:ensemble_mean=False")
def test_sites_follow_frozen_phonon_displacements_potential(device):
    """With sites=None, each configuration ionises its own displaced atom.

    Oracle (independent read-out): the displaced positions are read from the
    FrozenPhonons object's own configurations (by iterating it), and each
    configuration's elastic exit-wave phase is checked to be centred there
    too. The system is then inversion-symmetric about the displaced atom, so
    the inelastic intensity's circular centroid must sit on it.
    """
    fp = abtem.FrozenPhonons(
        _single_carbon_atoms(), num_configs=4, sigmas=0.3, directions="xy",
        ensemble_mean=False, seed=7,
    )
    displaced = [atoms.positions[0, :2].copy() for atoms in fp]
    # Precondition: the configurations are distinguishable at the tolerance.
    assert min(
        _periodic_distance(displaced[0], d, EXTENT).max() for d in displaced[1:]
    ) > 0.05

    potential = abtem.Potential(fp, gpts=GPTS, slice_thickness=2.0, device=device)
    inelastic, elastic = _inelastic_and_elastic_centres(potential, device)
    assert len(inelastic) == len(displaced) == len(elastic)

    for position, centre in zip(displaced, elastic):
        np.testing.assert_array_less(
            _periodic_distance(centre, position, EXTENT), 5e-3
        )
    for position, centre in zip(displaced, inelastic):
        np.testing.assert_array_less(
            _periodic_distance(centre, position, EXTENT), 5e-3
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CrystalPotential.get_sliced_atoms (and _extract_scattering_sites' "
        "potential_unit branch) tile potential_unit.get_transformed_atoms(), "
        "the equilibrium positions, while the elastic slices are built from "
        "the displaced pool configurations: the TP sits at (3.370, 2.910) Å "
        "while the atom it should ionise is at (3.404, 2.734) Å"
    ),
)
@devices
def test_sites_follow_frozen_phonon_displacements_crystal_potential(device):
    """A CrystalPotential of one displaced unit ionises the displaced atom.

    With a single-configuration pool and repetitions (1, 1, 1) the crystal
    *is* the displaced unit, so there is exactly one displaced realisation to
    follow. Oracle as in the Potential test above.
    """
    fp = abtem.FrozenPhonons(
        _single_carbon_atoms(), num_configs=1, sigmas=0.3, directions="xy",
        seed=7,
    )
    (displaced,) = [atoms.positions[0, :2].copy() for atoms in fp]
    assert _periodic_distance(displaced, _FP_SITE, EXTENT).max() > 0.05

    unit = abtem.Potential(fp, gpts=GPTS, slice_thickness=2.0, device=device)
    crystal = abtem.CrystalPotential(unit, repetitions=(1, 1, 1))
    (inelastic,), (elastic,) = _inelastic_and_elastic_centres(crystal, device)

    np.testing.assert_array_less(
        _periodic_distance(elastic, displaced, EXTENT), 5e-3
    )
    np.testing.assert_array_less(
        _periodic_distance(inelastic, displaced, EXTENT), 5e-3
    )
