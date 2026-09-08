"""End-to-end integration test for parity_projection: real (tiny) graphene
multislice through EnergyResolvedAtomsEnsemble -> Potential -> multislice ->
phonon_loss_diffraction_patterns (issue #373).

Deliberately small/fast (a handful of atoms and configs) -- this checks the
plumbing end-to-end, not statistics; see the issue for the full physics
validation on a realistic graphene cell.
"""

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


def test_multislice_produces_expanded_parity_axis(equilibrium):
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    potential = abtem.Potential(
        ensemble, sampling=0.1, slice_thickness=equilibrium.cell[2, 2]
    )
    exit_waves = abtem.PlaneWave(energy=100e3).multislice(potential, lazy=False)

    parity_axis = next(
        ax for ax in exit_waves.ensemble_axes_metadata
        if isinstance(ax, PhononParityAxis)
    )
    assert parity_axis.values == ("real", "twin", "static")
    assert exit_waves.array.shape[0] == 3


def test_branches_match_independent_direct_multislice(equilibrium):
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=1)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02], equilibrium_atoms=equilibrium, parity_projection=True,
    )
    potential_kwargs = dict(sampling=0.1, slice_thickness=equilibrium.cell[2, 2])
    potential = abtem.Potential(ensemble, **potential_kwargs)
    exit_waves = abtem.PlaneWave(energy=100e3).multislice(potential, lazy=False)

    # static branch must equal an independent multislice run on the bare
    # equilibrium atoms
    static_ref = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(equilibrium, **potential_kwargs), lazy=False
    )
    np.testing.assert_allclose(
        exit_waves.array[(2, 0, 0)], static_ref.array, atol=1e-10
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
    twin_atoms = real_atoms.copy()
    twin_atoms.positions = 2 * equilibrium.positions - real_atoms.positions
    twin_ref = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(twin_atoms, **potential_kwargs), lazy=False
    )
    np.testing.assert_allclose(
        exit_waves.array[(1, 0, 0)], twin_ref.array, atol=1e-10
    )


def test_full_pipeline_end_to_end(equilibrium):
    """Ensemble -> Potential -> multislice -> phonon_loss_diffraction_patterns,
    all the way through, with enough configs for a non-trivial (if noisy)
    signal."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=8, scale=0.08)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    potential = abtem.Potential(
        ensemble, sampling=0.1, slice_thickness=equilibrium.cell[2, 2]
    )
    exit_waves = abtem.PlaneWave(energy=100e3).multislice(potential, lazy=False)

    dp = phonon_loss_diffraction_patterns(exit_waves)

    assert dp.array.shape[0] == 3  # all, one_phonon, multi_phonon
    assert np.all(np.isfinite(dp.array))
    # one_phonon/multi_phonon are mean(|.|**2) with no subtraction, so they
    # cannot go negative beyond ordinary float rounding -- unlike the "all"
    # (I_incoherent - I_coherent) slot, which can (that cancellation is
    # exactly what the other two slots are designed to avoid).
    tiny = 1e-6 * np.abs(dp.array[1:]).max()
    assert np.all(dp.array[1:] >= -tiny)

    # a plain (non-parity) potential built from the same "real" snapshots
    # alone must still work exactly as before (backward compatibility)
    ensemble_plain = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05], ensemble_mean=False
    )
    potential_plain = abtem.Potential(
        ensemble_plain, sampling=0.1, slice_thickness=equilibrium.cell[2, 2]
    )
    exit_waves_plain = abtem.PlaneWave(energy=100e3).multislice(
        potential_plain, lazy=False
    )
    dp_plain = phonon_loss_diffraction_patterns(exit_waves_plain, component="tds")
    np.testing.assert_allclose(dp.array[0], dp_plain.array, atol=1e-6)
