"""Tests for Bloch-wave + Monte-Carlo inelastic (plasmon/phonon) scattering.

Tests follow the repo convention: parametrized over ``["cpu", gpu]``.
"""

import math

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem.bloch.dynamical import BlochwaveEnsemble, BlochWaves
from abtem.bloch.inelastic import (
    _deflect,
    calculate_bloch_plasmon_intensities,
    calculate_quadrature_plasmon_intensities,
)
from abtem.bloch.utils import excitation_errors
from abtem.core.energy import energy2wavelength
from abtem.inelastic.plasmons import MonteCarloPhonons, MonteCarloPlasmons


def _to_numpy(a):
    """Convert an array to NumPy, handling CuPy transparently."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


# ---------------------------------------------------------------------------
# Tilt-aware excitation errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_wave_eq", ["exact", True, False])
def test_excitation_errors_beam_frame(use_wave_eq):
    """Without a tilt every form is the textbook expression, bit for bit; with one,
    it is the same expression in the frame of the beam: rotating the beam by R
    gives the excitation errors of the reciprocal lattice rotated by R^T."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((30, 3))
    energy = 200e3
    wavelength = energy2wavelength(energy)
    g_perp2 = g[:, 0] ** 2 + g[:, 1] ** 2

    if use_wave_eq == "exact":
        x = wavelength**2 * g_perp2
        expected = -g[:, 2] - x / (wavelength * (np.sqrt(1.0 - x) + 1.0))
    elif use_wave_eq:
        expected = (-2 * g[:, 2] - wavelength * g_perp2) / 2.0
    else:
        expected = (-2 * g[:, 2] - wavelength * np.sum(g * g, axis=-1)) / 2.0

    untilted = excitation_errors(g, energy, use_wave_eq)
    along_z = excitation_errors(g, energy, use_wave_eq, beam_direction=[0, 0, 1])
    np.testing.assert_array_equal(untilted, expected)
    np.testing.assert_allclose(along_z, expected, atol=1e-12)

    R = _deflect(np.eye(3), 0.02, 0.7)  # 20 mrad
    tilted = excitation_errors(g, energy, use_wave_eq, beam_direction=R[:, 2])
    rotated = excitation_errors(g @ R, energy, use_wave_eq)
    np.testing.assert_allclose(tilted, rotated, atol=1e-12)
    assert np.abs(tilted - untilted).max() > 1e-3


# ---------------------------------------------------------------------------
# Euler-rotation deflection
# ---------------------------------------------------------------------------


def test_deflect_identity():
    """Zero deflection preserves the rotation."""
    R = np.eye(3)
    R2 = _deflect(R, 0.0, 0.0)
    np.testing.assert_allclose(R2, np.eye(3), atol=1e-15)


def test_deflect_polar_only():
    """A pure polar deflection about the initial z axis tilts the beam toward x."""
    R = np.eye(3)
    theta = 0.05
    R2 = _deflect(R, theta, 0.0)
    beam = R2 @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(beam[0], np.sin(theta), atol=1e-12)
    np.testing.assert_allclose(beam[2], np.cos(theta), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(beam), 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Zero-loss elastic limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_zero_events_matches_elastic(device):
    """With zero scattering events the driver must reproduce the pure Bloch result
    exactly (up to floating point)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 200.0

    dp_elastic = bw.calculate_diffraction_patterns(t, lazy=False)
    I_elastic = _to_numpy(dp_elastic.array)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=0, num_samples=1, ensemble_mean=True
    )
    events = mc._draw_events(thickness=t, energy=bw.energy)
    orders, I, weights = calculate_bloch_plasmon_intensities(bw, events, t)

    assert orders == [0]
    np.testing.assert_allclose(_to_numpy(I[0]), I_elastic, atol=1e-5 * I_elastic.max())


# ---------------------------------------------------------------------------
# Intensity conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_intensity_conservation(device):
    """The total intensity sum_n P(n) I^(n) must equal the Bloch normalisation times
    the truncated Poisson cumulative."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 300.0

    mc = MonteCarloPlasmons(
        830.0,
        17.0,
        19.1,
        num_excitations=2,
        num_samples=20,
        ensemble_mean=True,
        seed=42,
    )
    events = mc._draw_events(thickness=t, energy=bw.energy)
    orders, I, weights = calculate_bloch_plasmon_intensities(bw, events, t)

    total = sum(
        float(weights[i]) * float(_to_numpy(I[i]).sum()) for i in range(len(orders))
    )

    # Expected: each order's pattern is normalised (Bloch conserves total intensity),
    # so total ~ sum_n P(n) = truncated Poisson CDF.
    expected_poisson = sum(
        1 / math.factorial(n) * (t / 830.0) ** n * np.exp(-t / 830.0) for n in orders
    )
    np.testing.assert_allclose(total, expected_poisson, rtol=0.01)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_calculate_diffraction_patterns_plasmon_api(device):
    """``calculate_diffraction_patterns(plasmons=MonteCarloPlasmons)``: a scalar
    thickness gives (orders, beams) and a sequence (orders, thicknesses, beams), whose
    row of the same thickness is the scalar result (the events of a thickness are
    drawn from the seed); every order is normalized, and lazy equals eager."""
    from abtem.core.axes import PlasmonOrderAxis

    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=20, ensemble_mean=True, seed=7
    )

    scalar = bw.calculate_diffraction_patterns(300.0, plasmons=mc, lazy=False)
    several = bw.calculate_diffraction_patterns([200.0, 300.0], plasmons=mc, lazy=False)
    lazy = bw.calculate_diffraction_patterns([200.0, 300.0], plasmons=mc, lazy=True)

    assert scalar.array.shape == (2, len(bw))
    assert len(scalar.ensemble_axes_metadata) == 1
    assert isinstance(scalar.ensemble_axes_metadata[0], PlasmonOrderAxis)
    assert scalar.metadata["plasmon_orders"] == (0, 1)
    np.testing.assert_allclose(_to_numpy(scalar.array).sum(-1), 1.0, atol=1e-5)

    assert several.array.shape == (2, 2, len(bw))
    assert isinstance(several.ensemble_axes_metadata[0], PlasmonOrderAxis)
    assert several.ensemble_axes_metadata[1].label == "z"
    np.testing.assert_allclose(
        _to_numpy(several.array)[:, 1], _to_numpy(scalar.array), atol=1e-6
    )

    assert hasattr(lazy.array, "compute")
    np.testing.assert_allclose(
        _to_numpy(lazy.array.compute()), _to_numpy(several.array), atol=1e-12
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_000_decreases_with_plasmon_loss(device):
    """The normalised 000 beam intensity should decrease with increasing energy loss
    (Mendis 2024, Table 2)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 600.0

    mc = MonteCarloPlasmons(
        830.0,
        17.0,
        19.1,
        num_excitations=1,
        num_samples=30,
        ensemble_mean=True,
        seed=5,
    )
    events = mc._draw_events(thickness=t, energy=bw.energy)
    orders, I, _ = calculate_bloch_plasmon_intensities(bw, events, t)

    g0 = np.argmin(np.asarray((bw.g_vec**2).sum(1)))
    I_np = _to_numpy(I)
    fracs = [I_np[i, g0] / I_np[i].sum() for i in range(len(orders))]
    assert fracs[0] > fracs[1], (
        f"000 fraction should decrease: zero-loss={fracs[0]:.4f}, "
        f"single-plasmon={fracs[1]:.4f}"
    )


# ---------------------------------------------------------------------------
# BlochwaveEnsemble with plasmons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_ensemble_plasmon_shape(device):
    """``BlochwaveEnsemble.calculate_diffraction_patterns(plasmons=...)`` returns
    array with orientation × order × beam axes."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)

    ens = BlochwaveEnsemble(
        "z",
        np.linspace(-0.01, 0.01, 2),
        structure_factor=sf,
        energy=200e3,
        sg_max=0.1,
        g_max=2.0,
        centering="F",
        device=device,
    )
    mc = MonteCarloPlasmons(
        830.0,
        17.0,
        19.1,
        num_excitations=1,
        num_samples=50,
        ensemble_mean=True,
        seed=5,
    )
    dp = ens.calculate_diffraction_patterns(300.0, plasmons=mc)
    assert dp.array.ndim == 3  # (orientations, orders, beams)
    assert dp.array.shape[0] == 2  # 2 orientations
    assert dp.array.shape[1] == 2  # orders 0 and 1
    assert np.isfinite(_to_numpy(dp.array)).all()


# ---------------------------------------------------------------------------
# MonteCarloPhonons
# ---------------------------------------------------------------------------


def test_phonon_mean_free_path_silicon():
    """The Si phonon mean free path at 200 kV should be close to the Mendis value
    (~7724 Å)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    phonons = MonteCarloPhonons(
        atoms=si,
        thermal_sigma=0.078,
        parametrization="kirkland",
        theta_max=0.1,
        num_excitations=0,
        num_samples=1,
    )
    mfp = phonons.mean_free_path(200e3)
    np.testing.assert_allclose(mfp, 7724.0, rtol=0.02)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_phonon_bloch_integration(device):
    """Phonon events work with the Bloch driver through the ``plasmons=`` kwarg."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    phonons = MonteCarloPhonons(
        atoms=si,
        thermal_sigma=0.078,
        parametrization="kirkland",
        theta_max=0.1,
        num_excitations=1,
        num_samples=50,
        ensemble_mean=True,
        seed=42,
    )
    dp = bw.calculate_diffraction_patterns(500.0, plasmons=phonons)
    assert dp.array.shape == (2, len(bw))
    assert np.isfinite(_to_numpy(dp.array)).all()
    assert (_to_numpy(dp.array) >= 0).all()


# ---------------------------------------------------------------------------
# Diffuse background rendering (rigid-shift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffuse_pattern_shape_and_values(device):
    """``calculate_diffuse_diffraction_pattern`` returns a DiffractionPatterns with the
    right shape and non-negative finite values."""
    from abtem.measurements import DiffractionPatterns

    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0,
        17.0,
        19.1,
        num_excitations=1,
        num_samples=200,
        ensemble_mean=True,
        seed=7,
    )
    dp = bw.calculate_diffuse_diffraction_pattern(
        thickness=300.0,
        plasmons=mc,
        gpts=(128, 128),
    )
    assert isinstance(dp, DiffractionPatterns)

    orders = dp.metadata["plasmon_orders"]
    weights = dp.metadata["plasmon_weights"]
    images = _to_numpy(dp.array)

    assert images.shape == (len(orders), 128, 128)
    assert np.isfinite(images).all()
    assert (images >= 0).all()
    assert len(weights) == len(orders)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffuse_zero_order_concentrates_at_bragg(device):
    """For order 0 (no events, no tilt), all intensity should land exactly at the
    unshifted Bragg positions — the image should have non-zero pixels only at the
    spot locations."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0,
        17.0,
        19.1,
        num_excitations=1,
        num_samples=100,
        ensemble_mean=True,
        seed=3,
    )
    dp = bw.calculate_diffuse_diffraction_pattern(
        thickness=300.0,
        plasmons=mc,
        gpts=(128, 128),
    )
    images = _to_numpy(dp.array)
    orders = list(dp.metadata["plasmon_orders"])

    zero_idx = orders.index(0)
    zero_image = images[zero_idx]
    nonzero_pixels = np.count_nonzero(zero_image)
    assert nonzero_pixels <= len(bw), (
        f"order-0 image has {nonzero_pixels} non-zero pixels but only "
        f"{len(bw)} beams — no tilt broadening expected"
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffuse_higher_order_spreads(device):
    """Higher-order images should have more non-zero pixels than zero-order (the
    rigid-shift broadens the spots)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0,
        17.0,
        19.1,
        num_excitations=1,
        num_samples=50,
        ensemble_mean=True,
        seed=9,
    )
    dp = bw.calculate_diffuse_diffraction_pattern(
        thickness=500.0,
        plasmons=mc,
        gpts=(128, 128),
    )
    images = _to_numpy(dp.array)
    orders = list(dp.metadata["plasmon_orders"])
    assert orders == [0, 1]

    nz_zero = np.count_nonzero(images[orders.index(0)])
    nz_one = np.count_nonzero(images[orders.index(1)])
    assert nz_one > nz_zero, (
        f"single-plasmon image ({nz_one} pixels) should be more spread than the "
        f"zero-loss one ({nz_zero} pixels)"
    )


# ---------------------------------------------------------------------------
# Regressions: units, image axes and normalization of the Monte Carlo paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, math.pi / 2])
def test_diffuse_pattern_shifts_by_the_deflection(phi):
    """An event deflects the pattern rigidly by K sin(theta) along its azimuth. The
    polar angle is in mrad (it was applied as rad, deflecting by ~0.5 rad), and x is
    the first image axis (the images were transposed). The slab is thin enough that
    the transmitted beam keeps nearly all the intensity, so its peak is the shift."""
    from abtem.bloch.inelastic import calculate_bloch_diffuse_pattern
    from abtem.inelastic.plasmons import PlasmonScatteringEvents

    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=6.0, device="cpu")
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=1.5, device="cpu")
    thickness, theta = 5.0, 5.0  # [Å], [mrad]
    events = PlasmonScatteringEvents(
        depths=((), (thickness / 2,)),
        radial_angles=((), (theta,)),
        azimuthal_angles=((), (phi,)),
        weights=(1.0, 1.0),
        ensemble_mean=False,
    )
    gpts = (301, 301)
    _, images, _, extent = calculate_bloch_diffuse_pattern(bw, events, thickness, gpts)

    k = [np.linspace(-e, e, n) for e, n in zip(extent, gpts)]
    peak = np.unravel_index(np.argmax(images[1]), gpts)
    shift = 1 / energy2wavelength(200e3) * math.sin(theta * 1e-3)
    pixel = 2 * extent[0] / (gpts[0] - 1)
    assert abs(k[0][peak[0]] - shift * math.cos(phi)) <= pixel
    assert abs(k[1][peak[1]] - shift * math.sin(phi)) <= pixel


def test_mixed_events_draw_plasmon_angles_in_mrad():
    """The mixed plasmon-phonon events carry their polar angles in mrad, like every
    other event producer (they were appended in rad). Without phonon events the
    angles follow the Lorentzian: below theta_c, with median sqrt(theta_E theta_c)."""
    from abtem.bloch.inelastic import draw_mixed_scattering_events
    from abtem.inelastic.plasmons import characteristic_angle

    theta_e = characteristic_angle(17.0, 200e3)
    theta_c = 27.6
    grid = np.linspace(0, 0.1, 200)
    _, radials, _ = draw_mixed_scattering_events(
        1050.0,
        1e12,
        1000.0,
        5,
        theta_e * 1e-3,
        theta_c * 1e-3,
        grid,
        grid / 0.1,
        num_configs=400,
        seed=0,
    )
    angles = np.concatenate([np.ravel(r) for r in radials if len(r)])
    assert angles.max() <= theta_c
    assert np.median(angles) == pytest.approx(np.sqrt(theta_e * theta_c), rel=0.15)


@pytest.mark.parametrize("step", [0.25, 0.5, 1.0])
def test_plasmon_probability_grid_is_normalized(step):
    """The grid integrates the Lorentzian to one: the pixels near the centre hold the
    undeflected core (about a quarter of the probability at 0.5 mrad), which the
    grid used to drop (it summed to 0.70), and `max_angle_rad` truncates it."""
    from abtem.bloch.inelastic import _plasmon_probability_grid
    from abtem.inelastic.plasmons import characteristic_angle

    theta_e = characteristic_angle(17.0, 200e3) * 1e-3
    theta_c = 27.6e-3
    step = step * 1e-3
    dp_range = np.arange(-0.05, 0.05 + step / 2, step)
    P, _, _ = _plasmon_probability_grid(dp_range, step, theta_e, theta_c)
    assert P.sum() == pytest.approx(1.0, abs=2e-3)

    max_angle = 10e-3
    truncated, _, _ = _plasmon_probability_grid(
        dp_range, step, theta_e, theta_c, max_angle
    )
    expected = np.log1p((max_angle / theta_e) ** 2) / np.log1p((theta_c / theta_e) ** 2)
    assert truncated.sum() == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("model", ["quadrature", "monte_carlo"])
def test_plasmons_require_a_single_energy(model):
    """The plasmon paths run at one energy; with several they raise when called,
    naming the method, instead of answering for the first one (or, lazily, failing
    only when computed)."""
    from abtem.inelastic.plasmons import QuadraturePlasmons

    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=4.0, device="cpu")
    bw = BlochWaves(sf, energy=[100e3, 200e3], sg_max=0.1, g_max=1.0, device="cpu")
    parameters = dict(
        mean_free_path=1050.0, excitation_energy=17.0, critical_angle=27.6
    )
    if model == "quadrature":
        plasmons = QuadraturePlasmons(
            **parameters, max_loss_order=1, num_angles=1, num_azimuthal=2, num_depths=1
        )
    else:
        plasmons = MonteCarloPlasmons(**parameters, num_excitations=1, num_samples=2)
    match = r"calculate_diffraction_patterns\(plasmons=\.\.\.\) is defined for a single"
    with pytest.raises(ValueError, match=match):
        bw.calculate_diffraction_patterns(100.0, plasmons=plasmons, lazy=True)
    with pytest.raises(ValueError, match="calculate_convergent_beam_patterns"):
        bw.calculate_convergent_beam_patterns(100.0, abtem.Probe(energy=100e3))


# ---------------------------------------------------------------------------
# Deterministic (quadrature) plasmon scattering
# ---------------------------------------------------------------------------


def _quadrature_bloch(device="cpu", g_max=1.0):
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=8, device=device)
    return BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=g_max, device=device)


def _quadrature_model(**kwargs):
    from abtem.inelastic.plasmons import QuadraturePlasmons

    defaults = dict(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        max_loss_order=1,
        num_angles=4,
        num_azimuthal=6,
        num_depths=3,
        max_tilt_events=1,
    )
    return QuadraturePlasmons(**{**defaults, **kwargs})


def test_quadrature_zero_loss_matches_elastic():
    """The zero-loss channel is the plain dynamical result."""
    from abtem.bloch.inelastic import calculate_quadrature_plasmon_intensities

    bw = _quadrature_bloch()
    thickness = 300.0
    intensities = calculate_quadrature_plasmon_intensities(
        bw, thickness, _quadrature_model()
    )
    elastic = _to_numpy(bw.calculate_diffraction_patterns(thickness, lazy=False).array)
    assert intensities.shape[0] == 2
    # the working precision is single by default
    assert np.allclose(intensities[0], elastic, rtol=1e-4, atol=1e-6 * elastic.max())


def test_quadrature_conserves_intensity_per_order():
    """Every loss order is normalized to the incident electron count."""
    from abtem.bloch.inelastic import calculate_quadrature_plasmon_intensities

    bw = _quadrature_bloch()
    intensities = calculate_quadrature_plasmon_intensities(
        bw, 300.0, _quadrature_model(max_loss_order=2, max_tilt_events=2)
    )
    totals = intensities.sum(1)
    assert np.allclose(totals, 1.0, rtol=2e-3)


def test_quadrature_requires_enough_tilt_events():
    """With every excitation changing the direction, the orders cannot exceed the
    number of channeling-changing events."""
    from abtem.bloch.inelastic import calculate_quadrature_plasmon_intensities

    bw = _quadrature_bloch()
    with pytest.raises(ValueError, match="max_tilt_events"):
        calculate_quadrature_plasmon_intensities(
            bw, 300.0, _quadrature_model(max_loss_order=2, max_tilt_events=1)
        )


def test_quadrature_matches_monte_carlo():
    """The quadrature reproduces the Monte Carlo average of the same model."""
    from abtem.bloch.inelastic import (
        _prepare_bloch_matrices,
        calculate_quadrature_plasmon_intensities,
    )

    bw = _quadrature_bloch()
    thickness = 500.0
    precomputed = _prepare_bloch_matrices(bw)
    parameters = dict(
        mean_free_path=1050.0, excitation_energy=17.0, critical_angle=27.6
    )

    references = []
    for seed in (11, 22):
        events = MonteCarloPlasmons(
            **parameters, num_samples=300, num_excitations=[0, 1], seed=seed
        )._draw_events(thickness=thickness, energy=bw.energy)
        _, intensities, _ = calculate_bloch_plasmon_intensities(
            bw, events, thickness, _precomputed=precomputed
        )
        references.append(np.asarray(intensities)[1])

    reference = 0.5 * (references[0] + references[1])
    noise = np.sqrt(np.mean((references[0] - references[1]) ** 2)) / 2

    quadrature = calculate_quadrature_plasmon_intensities(
        bw,
        thickness,
        _quadrature_model(num_angles=6, num_azimuthal=8, max_angular_step=4.0),
        _precomputed=precomputed,
    )[1]

    error = np.sqrt(np.mean((quadrature - reference) ** 2))
    assert error < 3 * noise


def test_quadrature_angular_step_refines_the_outer_rings():
    """Capping the cell width adds nodes only where the rings are wide."""
    model = _quadrature_model(num_angles=6, num_azimuthal=8)
    coarse = model._angular_nodes(200e3, 0.0)["single"]
    fine = model._angular_nodes(200e3, 0.0, max_angular_step=4.0)["single"]
    assert len(fine["tilts"]) > len(coarse["tilts"])
    assert np.isclose(coarse["weights"].sum(), 1.0)
    assert np.isclose(fine["weights"].sum(), 1.0)
    radii = np.linalg.norm(fine["tilts"], axis=1)
    assert radii.max() > np.linalg.norm(coarse["tilts"], axis=1).max()


# ---------------------------------------------------------------------------
# Quadrature: dispatch, convergent-beam patterns in the laboratory frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
def test_quadrature_dispatch_from_calculate_diffraction_patterns(lazy):
    from abtem.core.axes import PlasmonOrderAxis

    bw = _quadrature_bloch()
    model = _quadrature_model()
    direct = calculate_quadrature_plasmon_intensities(bw, 300.0, model)
    dp = bw.calculate_diffraction_patterns(300.0, plasmons=model, lazy=lazy)
    array = np.asarray(dp.compute().array if lazy else dp.array)
    assert array.shape == direct.shape
    assert np.allclose(array, direct, atol=1e-6)
    assert isinstance(dp.ensemble_axes_metadata[0], PlasmonOrderAxis)
    assert dp.ensemble_axes_metadata[0].model == "quadrature"
    assert np.allclose(dp.metadata["plasmon_weights"], model.excitation_weights(300.0))

    dp = bw.calculate_diffraction_patterns([200.0, 300.0], plasmons=model, lazy=lazy)
    array = np.asarray(dp.compute().array if lazy else dp.array)
    assert array.shape == (model.num_orders, 2, len(bw))
    assert np.allclose(array[:, 1], direct, atol=1e-6)


def test_quadrature_warns_when_tilts_exceed_the_beam_set():
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=8, device="cpu")
    bw = BlochWaves(sf, energy=200e3, sg_max=0.01, g_max=2.0, device="cpu")
    with pytest.warns(UserWarning, match="sg_max"):
        calculate_quadrature_plasmon_intensities(bw, 100.0, _quadrature_model())


@pytest.fixture(scope="module")
def cbed_case():
    """Static-lattice Si [001]: a probe on the multislice grid and matching Bloch
    waves; the reciprocal lattice falls exactly on the reciprocal grid of the probe."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms = si * (2, 2, 10)
    potential = abtem.Potential(atoms, gpts=64, slice_thickness=si.cell[2, 2] / 4)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=6.0)
    probe.grid.match(potential)
    sf = abtem.StructureFactor(si, g_max=6.0, device="cpu")
    bw = BlochWaves(sf, energy=200e3, sg_max=0.15, g_max=2.0, device="cpu")
    return potential, probe, bw


def _rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def test_convergent_beam_patterns_match_multislice(cbed_case):
    potential, probe, bw = cbed_case
    detector = abtem.PixelatedDetector(max_angle="valid")
    multislice = np.asarray(probe.multislice(potential, detectors=detector).array)
    bloch = bw.calculate_convergent_beam_patterns(potential.thickness, probe)
    assert bloch.shape == multislice.shape
    assert bloch.ensemble_axes_metadata == []
    assert np.isclose(bloch.array.sum(), 1.0, atol=0.02)
    assert _rel_rms(np.asarray(bloch.array), multislice) < 0.05
    assert np.allclose(bloch.sampling, probe.build().reciprocal_space_sampling)


@pytest.mark.parametrize("lab_frame", [True, False])
def test_convergent_beam_quadrature_matches_multislice_quadrature(cbed_case, lab_frame):
    from abtem.core.axes import PlasmonOrderAxis
    from abtem.inelastic.plasmons import QuadraturePlasmons

    potential, probe, bw = cbed_case
    model = QuadraturePlasmons(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        max_loss_order=2,
        num_angles=3,
        num_azimuthal=6,
        num_depths=2,
        max_tilt_events=1,
        lab_frame=lab_frame,
    )
    detector = abtem.PixelatedDetector(max_angle="valid")
    multislice = np.asarray(
        probe.multislice(potential, detectors=detector, plasmons=model).array
    )
    bloch = bw.calculate_convergent_beam_patterns(
        potential.thickness, probe, plasmons=model
    )
    assert bloch.shape == multislice.shape
    assert isinstance(bloch.ensemble_axes_metadata[0], PlasmonOrderAxis)
    array = np.asarray(bloch.array)
    for n in range(model.num_orders):
        # every order is normalized to the incident electron and agrees with the
        # multislice quadrature at the level of the elastic engine difference
        assert np.isclose(array[n].sum(), 1.0, atol=0.02)
        assert _rel_rms(array[n], multislice[n]) < 0.05
    if lab_frame:
        # momentum transfer blurs the discs: the loss channels differ from the elastic
        assert _rel_rms(array[1], array[0]) > 0.1


def test_convergent_beam_patterns_warn_off_grid(cbed_case):
    potential, probe, bw = cbed_case
    off = abtem.Probe(energy=200e3, semiangle_cutoff=6.0, extent=13.0, gpts=64)
    with pytest.warns(UserWarning, match="reciprocal grid"):
        bw.calculate_convergent_beam_patterns(50.0, off)
