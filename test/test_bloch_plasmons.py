"""Tests for Bloch-wave + Monte-Carlo inelastic (plasmon/phonon) scattering.

Tests follow the repo convention: parametrized over ``["cpu", gpu]``.
"""

import math

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem.bloch.dynamical import BlochWaves, BlochwaveEnsemble
from abtem.bloch.inelastic import (
    _chain_one_configuration,
    _deflect,
    calculate_bloch_plasmon_intensities,
)
from abtem.bloch.utils import calculate_g_vec, excitation_errors
from abtem.core.energy import energy2wavelength
from abtem.inelastic.plasmons import MonteCarloPhonons, MonteCarloPlasmons


@pytest.fixture
def si_bloch(request):
    device = request.param
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    return bw


# ---------------------------------------------------------------------------
# Tilt-aware excitation errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_excitation_errors_zero_tilt_matches_legacy(device):
    """beam_direction=None and beam_direction=[0,0,1] must both reproduce the exact
    legacy formula (no tilt)."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((30, 3))
    energy = 200e3
    wavelength = energy2wavelength(energy)

    sg_default = excitation_errors(g, energy)
    sg_z = excitation_errors(g, energy, beam_direction=np.array([0.0, 0.0, 1.0]))
    legacy = (-2 * g[:, 2] - wavelength * np.sum(g * g, axis=-1)) / 2.0

    np.testing.assert_allclose(sg_default, legacy, atol=0, rtol=0)
    np.testing.assert_allclose(sg_z, legacy, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_excitation_errors_tilt_changes_sg(device):
    """A finite tilt must shift the excitation errors."""
    rng = np.random.default_rng(1)
    g = rng.standard_normal((20, 3))
    energy = 200e3
    theta = 0.01  # 10 mrad
    n = np.array([0.0, np.sin(theta), np.cos(theta)])
    sg_default = excitation_errors(g, energy)
    sg_tilted = excitation_errors(g, energy, beam_direction=n)
    assert not np.allclose(sg_default, sg_tilted)


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
def test_zero_events_matches_elastic(request, device):
    """With zero scattering events the driver must reproduce the pure Bloch result
    exactly (up to floating point)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 200.0

    dp_elastic = bw.calculate_diffraction_patterns(t, lazy=False)
    I_elastic = np.asarray(dp_elastic.array)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=0, num_samples=1, ensemble_mean=True
    )
    events = mc._draw_events(thickness=t, energy=bw.energy)
    orders, I, weights = calculate_bloch_plasmon_intensities(bw, events, t)

    assert orders == [0]
    np.testing.assert_allclose(np.asarray(I[0]), I_elastic, atol=5e-7)


# ---------------------------------------------------------------------------
# Intensity conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_intensity_conservation(request, device):
    """The total intensity sum_n P(n) I^(n) must equal the Bloch normalisation times
    the truncated Poisson cumulative."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 300.0

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=2, num_samples=500, ensemble_mean=True,
        seed=42,
    )
    events = mc._draw_events(thickness=t, energy=bw.energy)
    orders, I, weights = calculate_bloch_plasmon_intensities(bw, events, t)

    total = sum(
        float(weights[i]) * float(np.asarray(I[i]).sum()) for i in range(len(orders))
    )

    # Expected: each order's pattern is normalised (Bloch conserves total intensity),
    # so total ~ sum_n P(n) = truncated Poisson CDF.
    expected_poisson = sum(
        1 / math.factorial(n) * (t / 830.0) ** n * np.exp(-t / 830.0)
        for n in orders
    )
    np.testing.assert_allclose(total, expected_poisson, rtol=0.01)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_calculate_diffraction_patterns_plasmon_api(request, device):
    """The ``plasmons=`` kwarg on ``calculate_diffraction_patterns`` returns an
    ``IndexedDiffractionPatterns`` with per-order intensities."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 200.0

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=200, ensemble_mean=True,
        seed=7,
    )
    dp = bw.calculate_diffraction_patterns(t, plasmons=mc)

    assert dp.array.shape == (2, len(bw))
    assert "plasmon_orders" in dp.metadata
    assert dp.metadata["plasmon_orders"] == (0, 1)
    assert all(np.isfinite(np.asarray(dp.array)).ravel())
    assert all((np.asarray(dp.array) >= 0).ravel())


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_000_decreases_with_plasmon_loss(request, device):
    """The normalised 000 beam intensity should decrease with increasing energy loss
    (Mendis 2024, Table 2)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)
    t = 600.0

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=2, num_samples=800, ensemble_mean=True,
        seed=5,
    )
    events = mc._draw_events(thickness=t, energy=bw.energy)
    orders, I, _ = calculate_bloch_plasmon_intensities(bw, events, t)

    g0 = np.argmin(np.asarray((bw.g_vec ** 2).sum(1)))
    I_np = np.asarray(I)
    fracs = [I_np[i, g0] / I_np[i].sum() for i in range(len(orders))]
    assert fracs[0] > fracs[1], (
        f"000 fraction should decrease: zero-loss={fracs[0]:.4f}, "
        f"single-plasmon={fracs[1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Multiple thicknesses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_multi_thickness_shape(request, device):
    """A thickness sequence returns shape (orders, thicknesses, beams)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=50, ensemble_mean=True,
        seed=10,
    )
    dp = bw.calculate_diffraction_patterns([200.0, 400.0], plasmons=mc)
    assert dp.array.shape == (2, 2, len(bw))
    axes_labels = [a.label for a in dp.ensemble_axes_metadata]
    assert "energy loss" in axes_labels
    assert "z" in axes_labels


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_scalar_thickness_backward_compat(request, device):
    """A scalar thickness returns shape (orders, beams) — no thickness axis."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=50, ensemble_mean=True,
        seed=10,
    )
    dp = bw.calculate_diffraction_patterns(300.0, plasmons=mc)
    assert dp.array.shape == (2, len(bw))
    assert len(dp.ensemble_axes_metadata) == 1


# ---------------------------------------------------------------------------
# BlochwaveEnsemble with plasmons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_ensemble_plasmon_shape(request, device):
    """``BlochwaveEnsemble.calculate_diffraction_patterns(plasmons=...)`` returns
    array with orientation × order × beam axes."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)

    ens = BlochwaveEnsemble(
        "z", np.linspace(-0.01, 0.01, 2),
        structure_factor=sf, energy=200e3, sg_max=0.1, g_max=2.0, centering="F",
        device=device,
    )
    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=50, ensemble_mean=True,
        seed=5,
    )
    dp = ens.calculate_diffraction_patterns(300.0, plasmons=mc)
    assert dp.array.ndim == 3  # (orientations, orders, beams)
    assert dp.array.shape[0] == 2  # 2 orientations
    assert dp.array.shape[1] == 2  # orders 0 and 1
    assert np.isfinite(np.asarray(dp.array)).all()


# ---------------------------------------------------------------------------
# MonteCarloPhonons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_phonon_mean_free_path_silicon(request, device):
    """The Si phonon mean free path at 200 kV should be close to the Mendis value
    (~7724 Å)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    phonons = MonteCarloPhonons(
        atoms=si, thermal_sigma=0.078, parametrization="kirkland",
        theta_max=0.1, num_excitations=0, num_samples=1,
    )
    mfp = phonons.mean_free_path(200e3)
    np.testing.assert_allclose(mfp, 7724.0, rtol=0.02)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_phonon_bloch_integration(request, device):
    """Phonon events work with the Bloch driver through the ``plasmons=`` kwarg."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    phonons = MonteCarloPhonons(
        atoms=si, thermal_sigma=0.078, parametrization="kirkland",
        theta_max=0.1, num_excitations=1, num_samples=50, ensemble_mean=True, seed=42,
    )
    dp = bw.calculate_diffraction_patterns(500.0, plasmons=phonons)
    assert dp.array.shape == (2, len(bw))
    assert np.isfinite(np.asarray(dp.array)).all()
    assert (np.asarray(dp.array) >= 0).all()


# ---------------------------------------------------------------------------
# Lazy/dask execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lazy_plasmon_matches_eager(request, device):
    """``lazy=True`` must produce the same result as ``lazy=False`` for the plasmon
    path (deferred via dask)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=100, ensemble_mean=True,
        seed=42,
    )
    dp_eager = bw.calculate_diffraction_patterns(300.0, plasmons=mc, lazy=False)
    dp_lazy = bw.calculate_diffraction_patterns(300.0, plasmons=mc, lazy=True)

    assert hasattr(dp_lazy.array, "compute"), "lazy result should be a dask array"
    lazy_array = np.asarray(dp_lazy.array.compute())
    eager_array = np.asarray(dp_eager.array)
    np.testing.assert_allclose(lazy_array, eager_array, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lazy_multi_thickness(request, device):
    """``lazy=True`` works with a thickness sequence."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=50, ensemble_mean=True,
        seed=10,
    )
    dp = bw.calculate_diffraction_patterns([200.0, 400.0], plasmons=mc, lazy=True)
    assert hasattr(dp.array, "compute")
    arr = np.asarray(dp.array.compute())
    assert arr.shape == (2, 2, len(bw))
    assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# Diffuse background rendering (rigid-shift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffuse_pattern_shape_and_values(request, device):
    """``calculate_diffuse_diffraction_pattern`` returns images with the right shape and
    non-negative finite values."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=200, ensemble_mean=True,
        seed=7,
    )
    result = bw.calculate_diffuse_diffraction_pattern(
        thickness=300.0, plasmons=mc, gpts=(128, 128),
    )
    images = result["images"]
    orders = result["orders"]
    weights = result["weights"]

    assert images.shape == (len(orders), 128, 128)
    assert np.isfinite(images).all()
    assert (images >= 0).all()
    assert len(weights) == len(orders)
    assert result["extent"] is not None


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffuse_zero_order_concentrates_at_bragg(request, device):
    """For order 0 (no events, no tilt), all intensity should land exactly at the
    unshifted Bragg positions — the image should have non-zero pixels only at the
    spot locations."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=1, num_samples=100, ensemble_mean=True,
        seed=3,
    )
    result = bw.calculate_diffuse_diffraction_pattern(
        thickness=300.0, plasmons=mc, gpts=(128, 128),
    )
    images = result["images"]
    orders = result["orders"]

    zero_idx = orders.index(0)
    zero_image = images[zero_idx]
    nonzero_pixels = np.count_nonzero(zero_image)
    assert nonzero_pixels <= len(bw), (
        f"order-0 image has {nonzero_pixels} non-zero pixels but only "
        f"{len(bw)} beams — no tilt broadening expected"
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffuse_higher_order_spreads(request, device):
    """Higher-order images should have more non-zero pixels than zero-order (the
    rigid-shift broadens the spots)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    sf = abtem.StructureFactor(si, g_max=12, device=device)
    bw = BlochWaves(sf, energy=200e3, sg_max=0.1, g_max=2.0, device=device)

    mc = MonteCarloPlasmons(
        830.0, 17.0, 19.1, num_excitations=2, num_samples=500, ensemble_mean=True,
        seed=9,
    )
    result = bw.calculate_diffuse_diffraction_pattern(
        thickness=500.0, plasmons=mc, gpts=(256, 256),
    )
    images = result["images"]
    orders = result["orders"]

    if len(orders) >= 2:
        nz_zero = np.count_nonzero(images[orders.index(0)])
        nz_one = np.count_nonzero(images[orders.index(1)])
        assert nz_one >= nz_zero, (
            f"single-plasmon image ({nz_one} pixels) should be at least as spread "
            f"as zero-loss ({nz_zero} pixels)"
        )
