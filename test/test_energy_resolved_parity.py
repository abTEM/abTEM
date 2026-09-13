"""Tests for EnergyResolvedAtomsEnsemble's parity_projection option
(separating one-phonon from multi-phonon scattering, issue #373)."""

import numpy as np
import pytest
from ase import Atoms

from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis, PhononParityAxis
from abtem.inelastic.phonons import EnergyResolvedAtomsEnsemble


def _make_snapshots(equilibrium, n_energies=3, n_configs=4, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    snapshots = []
    for _ in range(n_energies):
        group = []
        for _ in range(n_configs):
            atoms = equilibrium.copy()
            atoms.positions += rng.normal(scale=scale, size=atoms.positions.shape)
            group.append(atoms)
        snapshots.append(group)
    return snapshots


@pytest.fixture
def equilibrium():
    return Atoms(
        "H2", positions=[[0, 0, 0], [1.0, 0, 0]], cell=[10, 10, 10], pbc=True
    )


def test_parity_projection_shape_and_axes(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    energies = [0.02, 0.05, 0.10]

    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, energies, equilibrium_atoms=equilibrium, parity_projection=True
    )

    assert ensemble.ensemble_shape == (2, 3, 4)
    assert ensemble.num_configs == 4
    assert ensemble.parity_projection is True
    assert ensemble.equilibrium_atoms is equilibrium

    axes = ensemble.ensemble_axes_metadata
    assert isinstance(axes[0], PhononParityAxis)
    assert axes[0].values == ("real", "twin")
    assert isinstance(axes[1], EnergyLossAxis)
    assert isinstance(axes[2], FrozenPhononsAxis)


def test_ensemble_mean_forced_false(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots,
        [0.02, 0.05, 0.10],
        equilibrium_atoms=equilibrium,
        parity_projection=True,
        ensemble_mean=True,  # should be silently overridden
    )
    assert ensemble.ensemble_mean is False


def test_requires_equilibrium_atoms(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    with pytest.raises(ValueError, match="requires equilibrium_atoms"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02, 0.05, 0.10], parity_projection=True
        )


def test_twin_is_displacement_reversed(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    energies = [0.02, 0.05, 0.10]
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, energies, equilibrium_atoms=equilibrium, parity_projection=True
    )

    for e in range(3):
        for c in range(4):
            real = ensemble.snapshots[0, e, c]
            twin = ensemble.snapshots[1, e, c]
            # real + twin averages to exactly the equilibrium positions
            np.testing.assert_allclose(
                (real.positions + twin.positions) / 2, equilibrium.positions,
                atol=1e-12,
            )
            # and twin is *not* just a copy of real (sanity, given nonzero
            # random displacement)
            assert not np.allclose(real.positions, twin.positions)


def test_backward_compatible_without_parity_projection(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    energies = [0.02, 0.05, 0.10]
    ensemble = EnergyResolvedAtomsEnsemble(snapshots, energies, ensemble_mean=False)

    assert ensemble.ensemble_shape == (3, 4)
    assert len(ensemble.ensemble_axes_metadata) == 2
    assert ensemble.parity_projection is False
    assert ensemble.equilibrium_atoms is None


def test_getitem_not_supported_with_parity_projection(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots,
        [0.02, 0.05, 0.10],
        equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    with pytest.raises(NotImplementedError):
        ensemble[0]


class TestEnsembleMachinery:
    """Exercise the generic Ensemble (dask-blockwise) machinery on the
    3D (parity, energy, config) shape."""

    def test_generate_blocks(self, equilibrium):
        snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
        ensemble = EnergyResolvedAtomsEnsemble(
            snapshots,
            [0.02, 0.05],
            equilibrium_atoms=equilibrium,
            parity_projection=True,
        )

        count = 0
        seen_parity_members = set()
        for indices, slics, block in ensemble.generate_blocks(chunks=1):
            block = block.item() if hasattr(block, "item") else block
            assert block.ensemble_shape == (1, 1, 1)
            seen_parity_members.add(indices[0])
            count += 1

        assert count == 2 * 2 * 3
        assert seen_parity_members == {0, 1}

    def test_ensemble_blocks_lazy(self, equilibrium):
        snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
        ensemble = EnergyResolvedAtomsEnsemble(
            snapshots,
            [0.02, 0.05],
            equilibrium_atoms=equilibrium,
            parity_projection=True,
        )
        blocks = ensemble.ensemble_blocks(chunks=1)
        assert blocks.shape == (2, 2, 3)


def test_rejects_equilibrium_with_wrong_atom_count(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    wrong = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    with pytest.raises(ValueError, match="atoms but equilibrium_atoms"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02, 0.05, 0.10], equilibrium_atoms=wrong,
            parity_projection=True,
        )


def test_rejects_equilibrium_with_different_species_order(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    wrong = equilibrium.copy()
    wrong.numbers = [1, 2]
    with pytest.raises(ValueError, match="species sequence"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02, 0.05, 0.10], equilibrium_atoms=wrong,
            parity_projection=True,
        )


def test_rejects_large_displacement_as_probable_misordering(equilibrium):
    """A snapshot whose atoms are permuted relative to equilibrium_atoms
    looks like a huge displacement; that must be caught rather than
    producing meaningless twins."""
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=2)
    permuted = snapshots[0][1].copy()
    permuted.positions = permuted.positions[::-1] + [[0, 0, 0], [3.0, 0, 0]]
    snapshots[0][1] = permuted
    with pytest.raises(ValueError, match="max_displacement"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02], equilibrium_atoms=equilibrium, parity_projection=True,
        )
    # explicit opt-outs
    EnergyResolvedAtomsEnsemble(
        snapshots, [0.02], equilibrium_atoms=equilibrium, parity_projection=True,
        max_displacement=None,
    )
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02], equilibrium_atoms=equilibrium, parity_projection=True,
        max_displacement=5.0,
    )
    assert ensemble.max_displacement == 5.0


def test_wrapped_snapshot_passes_minimum_image_check(equilibrium):
    """An atom at the cell origin displaced across the boundary and wrapped
    back into the cell has a raw displacement of ~L; the check must use the
    minimum image and accept it, and the twin must still be R_eq - u modulo
    the cell."""
    u = np.array([-0.05, 0.02, 0.0])
    atoms = equilibrium.copy()
    atoms.positions[0] += u
    atoms.wrap()
    assert atoms.positions[0, 0] > 9.0  # really wrapped

    ensemble = EnergyResolvedAtomsEnsemble(
        [[atoms, atoms]], [0.02], equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    twin = ensemble.snapshots[1, 0, 0]
    expected = equilibrium.positions[0] - u
    diff = twin.positions[0] - expected
    diff -= np.round(diff / 10.0) * 10.0
    np.testing.assert_allclose(diff, 0.0, atol=1e-12)
