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
    2N configurations {+u_j, -u_j}, the ordinary total-minus-elastic (diffuse)
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
        dp_full = phonon_loss_diffraction_patterns(exit_waves_full, components="diffuse")

    assert dp.array.shape[0] == 3  # all, one, multi
    assert dp_full.array.shape == dp.array.shape[1:]
    assert np.all(np.isfinite(dp.array)) and np.all(np.isfinite(dp_full.array))
    scale = np.abs(dp.array[1]).max()
    np.testing.assert_allclose(dp.array[0], dp_full.array, atol=1e-9 * scale, rtol=0)
    np.testing.assert_allclose(
        dp.array[0], dp.array[1] + dp.array[2], atol=1e-12 * scale, rtol=0
    )


def test_multi_channel_float32_matches_float64(equilibrium):
    """The multi-phonon channel is a variance whose Bragg-position
    cancellation single precision cannot resolve; it is therefore evaluated
    in complex128 internally regardless of the configured precision. A
    float32 session must reproduce the float64 result closely, and its "one"
    (a plain mean of |.|**2) and "multi" (a variance) channels can never go
    negative beyond rounding."""
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

    assert np.all(np.isfinite(results["float32"]))
    tiny = 1e-6 * np.abs(results["float32"][1:]).max()
    assert np.all(results["float32"][1:] >= -tiny)

    scale = results["float64"][1].max()
    for slot in range(3):
        np.testing.assert_allclose(
            results["float32"][slot], results["float64"][slot],
            atol=1e-4 * scale, rtol=0,
        )


def test_lazy_pipeline_matches_eager(equilibrium):
    """Lazy and eager runs agree, and the eager exit waves carry a parity axis
    of exactly ("real", "twin") -- no static/equilibrium wave is attached."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
    kwargs = dict(equilibrium_atoms=equilibrium, parity_projection=True)
    exit_waves_lazy = _run(equilibrium, snapshots, [0.02, 0.05], lazy=True, **kwargs)
    exit_waves = _run(equilibrium, snapshots, [0.02, 0.05], lazy=False, **kwargs)

    parity_axis = next(
        ax for ax in exit_waves.ensemble_axes_metadata
        if isinstance(ax, PhononParityAxis)
    )
    assert parity_axis.values == ("real", "twin")
    assert exit_waves.array.shape[0] == 2
    assert not hasattr(exit_waves, "static_exit_wave")

    assert isinstance(exit_waves_lazy.array, da.core.Array)
    dp_lazy = phonon_loss_diffraction_patterns(exit_waves_lazy)
    assert isinstance(dp_lazy.array, da.core.Array)

    dp = phonon_loss_diffraction_patterns(exit_waves)
    np.testing.assert_allclose(
        dp_lazy.array.compute(), dp.array, atol=1e-6 * np.abs(dp.array).max(), rtol=0
    )


def test_single_configuration_raises(equilibrium):
    """The multi-phonon channel is the variance of the even part across
    configurations, identically zero for a single configuration, so parity
    projection needs at least two per energy."""
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=1)
    with pytest.raises(ValueError, match="at least 2 frozen-phonon"):
        phonon_loss_diffraction_patterns(
            _run(equilibrium, snapshots, [0.02],
                 equilibrium_atoms=equilibrium, parity_projection=True),
            max_angle=60,
        )
