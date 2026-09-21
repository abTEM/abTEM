"""End-to-end integration test for parity_projection: real (tiny) graphene
multislice through EnergyResolvedAtomsEnsemble -> Potential -> multislice ->
phonon_loss_diffraction_patterns (issue #373).

Deliberately small/fast (a handful of atoms and configs) -- this checks the
plumbing end-to-end and the exact identities the channels must satisfy, not
statistics; see the issue for the physics validation on a realistic cell.
"""

import dask.array as da
import numpy as np
import pytest
from ase.build import graphene

import abtem
from abtem.core.axes import PhononParityAxis
from abtem.inelastic.phonons import EnergyResolvedAtomsEnsemble
from abtem.measurements import phonon_loss_diffraction_patterns


@pytest.fixture
def equilibrium():
    return graphene(a=2.46, size=(2, 2, 1), vacuum=1.0)


def _make_snapshots(equilibrium, n_energies, n_configs, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    snapshots = []
    for _ in range(n_energies):
        group = []
        for _ in range(n_configs):
            atoms = equilibrium.copy()
            atoms.positions[:, 0] += rng.normal(scale=scale, size=len(atoms))
            group.append(atoms)
        snapshots.append(group)
    return snapshots


def _twin(atoms, equilibrium):
    twin = atoms.copy()
    twin.positions = 2 * equilibrium.positions - atoms.positions
    return twin


def _run(equilibrium, snapshots, energies, lazy=False, **ensemble_kwargs):
    ensemble = EnergyResolvedAtomsEnsemble(snapshots, energies, **ensemble_kwargs)
    potential = abtem.Potential(
        ensemble, sampling=0.1, slice_thickness=equilibrium.cell[2, 2]
    )
    return abtem.PlaneWave(energy=100e3).multislice(potential, lazy=lazy)


def test_multislice_keeps_parity_axis_length_two(equilibrium):
    """The PhononParityAxis is exactly ("real", "twin"); nothing else is
    attached to the exit waves -- no static/equilibrium wave is needed."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
    exit_waves = _run(
        equilibrium, snapshots, [0.02, 0.05],
        equilibrium_atoms=equilibrium, parity_projection=True,
    )

    parity_axis = next(
        ax for ax in exit_waves.ensemble_axes_metadata
        if isinstance(ax, PhononParityAxis)
    )
    assert parity_axis.values == ("real", "twin")
    assert exit_waves.array.shape[0] == 2
    assert not hasattr(exit_waves, "static_exit_wave")


def test_branches_match_independent_direct_multislice(equilibrium):
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=1)
    potential_kwargs = dict(sampling=0.1, slice_thickness=equilibrium.cell[2, 2])
    exit_waves = _run(
        equilibrium, snapshots, [0.02],
        equilibrium_atoms=equilibrium, parity_projection=True,
    )

    # real branch must equal an independent multislice run on the same
    # displaced atoms
    real_atoms = snapshots[0][0]
    real_ref = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(real_atoms, **potential_kwargs), lazy=False
    )
    np.testing.assert_allclose(
        exit_waves.array[(0, 0, 0)], real_ref.array, atol=1e-10
    )

    # twin branch must equal an independent multislice run on the
    # displacement-reversed atoms
    twin_ref = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(_twin(real_atoms, equilibrium), **potential_kwargs),
        lazy=False,
    )
    np.testing.assert_allclose(
        exit_waves.array[(1, 0, 0)], twin_ref.array, atol=1e-10
    )


def test_all_equals_ordinary_tds_over_full_parity_set(equilibrium):
    """Exact identity (not a statistical one): for the symmetric set of
    2N configurations {+u_j, -u_j}, the ordinary I_incoherent - I_coherent
    estimator equals one + multi, because the odd part of the mean wave and
    the odd-even cross term in the intensity cancel pairwise. So the "all"
    slot must match a plain (non-parity) run on the real *and* twin
    snapshots to round-off. Run in float64 so the plain estimator's own
    Bragg-position cancellation does not limit the comparison."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=4, scale=0.08)
    energies = [0.02, 0.05]
    with abtem.config.set({"precision": "float64"}):
        exit_waves = _run(
            equilibrium, snapshots, energies,
            equilibrium_atoms=equilibrium, parity_projection=True,
        )
        dp = phonon_loss_diffraction_patterns(exit_waves)

        full_set = [
            group + [_twin(atoms, equilibrium) for atoms in group]
            for group in snapshots
        ]
        exit_waves_full = _run(equilibrium, full_set, energies, ensemble_mean=False)
        dp_full = phonon_loss_diffraction_patterns(exit_waves_full, component="tds")

    scale = np.abs(dp.array[1]).max()
    np.testing.assert_allclose(dp.array[0], dp_full.array, atol=1e-9 * scale, rtol=0)
    np.testing.assert_allclose(
        dp.array[0], dp.array[1] + dp.array[2], atol=1e-12 * scale, rtol=0
    )


def test_multi_channel_float32_matches_float64(equilibrium):
    """The multi-phonon channel is a variance whose Bragg-position
    cancellation single precision cannot resolve; it is therefore evaluated
    in complex128 internally regardless of the configured precision. A
    float32 session must reproduce the float64 result closely."""
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=4, scale=0.08)
    results = {}
    for precision in ("float64", "float32"):
        with abtem.config.set({"precision": precision}):
            exit_waves = _run(
                equilibrium, snapshots, [0.02],
                equilibrium_atoms=equilibrium, parity_projection=True,
            )
            results[precision] = np.asarray(
                phonon_loss_diffraction_patterns(exit_waves).array, dtype=np.float64
            )

    scale = results["float64"][1].max()
    for slot in range(3):
        np.testing.assert_allclose(
            results["float32"][slot], results["float64"][slot],
            atol=1e-4 * scale, rtol=0,
        )


def test_full_pipeline_end_to_end(equilibrium):
    """Ensemble -> Potential -> multislice -> phonon_loss_diffraction_patterns,
    all the way through, with enough configs for a non-trivial (if noisy)
    signal; and a plain (non-parity) ensemble must keep working unchanged."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=8, scale=0.08)
    exit_waves = _run(
        equilibrium, snapshots, [0.02, 0.05],
        equilibrium_atoms=equilibrium, parity_projection=True,
    )
    dp = phonon_loss_diffraction_patterns(exit_waves)

    assert dp.array.shape[0] == 3  # all, one, multi
    assert np.all(np.isfinite(dp.array))
    # "one" is a plain mean of |.|**2 and "multi" a variance, so neither can
    # go negative beyond float rounding
    tiny = 1e-6 * np.abs(dp.array[1:]).max()
    assert np.all(dp.array[1:] >= -tiny)

    exit_waves_plain = _run(equilibrium, snapshots, [0.02, 0.05], ensemble_mean=False)
    dp_plain = phonon_loss_diffraction_patterns(exit_waves_plain, component="tds")
    assert dp_plain.array.shape == dp.array.shape[1:]
    assert np.all(np.isfinite(dp_plain.array))


def test_lazy_pipeline_matches_eager(equilibrium):
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
    kwargs = dict(equilibrium_atoms=equilibrium, parity_projection=True)
    exit_waves_lazy = _run(equilibrium, snapshots, [0.02, 0.05], lazy=True, **kwargs)
    exit_waves = _run(equilibrium, snapshots, [0.02, 0.05], lazy=False, **kwargs)

    assert isinstance(exit_waves_lazy.array, da.core.Array)
    dp_lazy = phonon_loss_diffraction_patterns(exit_waves_lazy)
    assert isinstance(dp_lazy.array, da.core.Array)

    dp = phonon_loss_diffraction_patterns(exit_waves)
    np.testing.assert_allclose(
        dp_lazy.array.compute(), dp.array, atol=1e-6 * np.abs(dp.array).max(), rtol=0
    )


def _rest_fields(equilibrium, n, scale, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        atoms = equilibrium.copy()
        atoms.positions[:, :2] += rng.normal(scale=scale, size=(len(atoms), 2))
        out.append(atoms)
    return out


def test_rest_parity_damps_one_phonon_channel(equilibrium):
    """Adding the rest displacement with both signs must damp the one-phonon
    channel by the Debye-Waller factor of the rest field, exp(-Q^2 sigma^2)
    for an isotropic in-plane Gaussian of per-component sigma, at the
    one-phonon peaks, and must not conserve the total weight (the lost
    weight is a loss, not a redistribution)."""
    n_configs, sigma = 16, 0.04
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=n_configs, scale=0.05)
    rest = _rest_fields(equilibrium, n_configs, scale=sigma, seed=21)
    energies = [0.02]

    with abtem.config.set({"precision": "float64"}):
        plain = phonon_loss_diffraction_patterns(
            _run(equilibrium, snapshots, energies,
                 equilibrium_atoms=equilibrium, parity_projection=True),
            max_angle="full",
        )
        damped = phonon_loss_diffraction_patterns(
            _run(equilibrium, snapshots, energies,
                 equilibrium_atoms=equilibrium, parity_projection=True,
                 rest_snapshots=rest),
            max_angle="full",
        )

    # four runs per snapshot: only the one-phonon channel comes back
    assert damped.array.shape == plain.array.shape[1:]
    assert damped.metadata["phonon_loss_component"] == "one"
    one_plain, one_damped = plain.array[1, 0], damped.array[0]

    ny, nx = one_plain.shape
    ang = plain.angular_sampling  # mrad
    wavelength = abtem.core.energy.energy2wavelength(100e3)
    qy = (np.arange(ny) - ny // 2) * ang[0] * 1e-3 / wavelength * 2 * np.pi
    qx = (np.arange(nx) - nx // 2) * ang[1] * 1e-3 / wavelength * 2 * np.pi
    q2 = qy[:, None] ** 2 + qx[None, :] ** 2
    dw = np.exp(-q2 * sigma**2)

    # The intensity average over rest realizations captures the damping to
    # second order in u_rest; the residual is the incoherent two-rest-phonon
    # term, positive and of relative order (Q sigma)^4, so compare where it
    # is small (Q^2 sigma^2 < 0.2) and allow for finite-sample noise. The
    # tolerance was checked to be independent of sampling (0.1 to 0.025 A).
    peaks = (one_plain > 0.1 * one_plain.max()) & (q2 * sigma**2 < 0.2)
    assert peaks.sum() >= 10
    ratio = one_damped[peaks] / one_plain[peaks]
    np.testing.assert_allclose(ratio, dw[peaks], atol=0.08, rtol=0)  # per-pixel, sample noise
    assert 0 < np.mean(ratio - dw[peaks]) < 0.04  # residual is a small positive bias
    assert one_damped.sum() < 0.97 * one_plain.sum()


def test_rest_parity_multi_channel_is_the_damped_bin_multi_channel(equilibrium):
    """With rest fields the multi channel must be the bin's own two-phonon
    scattering (damped by the rest field), not the two-rest-phonon
    fluctuations, which are larger: the per-realization static reference
    removes them. Check the multi/one weight ratio, which the damping
    leaves nearly unchanged, against the bin-only run."""
    n_configs, sigma = 12, 0.04
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=n_configs, scale=0.06)
    rest = _rest_fields(equilibrium, n_configs, scale=sigma, seed=31)
    with abtem.config.set({"precision": "float64"}):
        plain = phonon_loss_diffraction_patterns(
            _run(equilibrium, snapshots, [0.02],
                 equilibrium_atoms=equilibrium, parity_projection=True),
            max_angle=60,
        )
        damped = phonon_loss_diffraction_patterns(
            _run(equilibrium, snapshots, [0.02],
                 equilibrium_atoms=equilibrium, parity_projection=True,
                 rest_snapshots=rest, rest_static_reference=True),
            max_angle=60,
        )
        one_only = phonon_loss_diffraction_patterns(
            _run(equilibrium, snapshots, [0.02],
                 equilibrium_atoms=equilibrium, parity_projection=True,
                 rest_snapshots=rest),
            max_angle=60,
        )
    assert damped.array.shape[0] == 3  # six runs: all three channels
    ratio_plain = plain.array[2, 0].sum() / plain.array[1, 0].sum()
    ratio_damped = damped.array[2, 0].sum() / damped.array[1, 0].sum()
    assert 0.7 < ratio_damped / ratio_plain < 1.3, (ratio_plain, ratio_damped)
    # and the multi weight itself decreases with the damping, never grows
    assert damped.array[2, 0].sum() < plain.array[2, 0].sum()
    # the four-run one-phonon channel is the six-run one, bit for bit: the
    # real/twin members and rest fields are the same structures
    np.testing.assert_allclose(one_only.array, damped.array[1], rtol=0, atol=0)
