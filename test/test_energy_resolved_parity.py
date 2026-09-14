"""Tests for EnergyResolvedAtomsEnsemble's parity_projection option
(separating one-phonon from multi-phonon scattering, issue #373)."""

import numpy as np
import pytest
from ase import Atoms

from abtem.core.axes import (
    EnergyLossAxis,
    FrozenPhononsAxis,
    PhononParityAxis,
    PhononRestParityAxis,
)
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


def _rest_fields(equilibrium, n, seed=11, scale=0.03):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        atoms = equilibrium.copy()
        atoms.positions += rng.normal(scale=scale, size=atoms.positions.shape)
        out.append(atoms)
    return out


def test_rest_snapshots_add_rest_parity_axis_and_members(equilibrium):
    """R_eq + s u_bin + t u_rest for s, t in (+, -), laid out as
    (parity, rest sign, energy, configuration)."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
    rest = [_rest_fields(equilibrium, 3, seed=1), _rest_fields(equilibrium, 3, seed=2)]
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
        parity_projection=True, rest_snapshots=rest,
    )
    assert ensemble.ensemble_shape == (2, 2, 2, 3)
    assert ensemble.rest_parity is True
    assert ensemble.num_configs == 3
    axes = ensemble.ensemble_axes_metadata
    assert isinstance(axes[0], PhononParityAxis)
    assert isinstance(axes[1], PhononRestParityAxis) and axes[1].values == ("plus", "minus")
    assert isinstance(axes[2], EnergyLossAxis)
    assert isinstance(axes[3], FrozenPhononsAxis)

    eq = equilibrium.positions
    for i in range(2):
        for j in range(3):
            u_bin = snapshots[i][j].positions - eq
            u_rest = rest[i][j].positions - eq
            for parity, s in enumerate((1, -1)):
                for sign_index, t in enumerate((1, -1)):
                    member = ensemble.snapshots[parity, sign_index, i, j]
                    np.testing.assert_allclose(
                        member.positions, eq + s * u_bin + t * u_rest, atol=1e-12
                    )


def test_flat_rest_snapshots_are_reused_for_every_energy(equilibrium):
    snapshots = _make_snapshots(equilibrium, n_energies=3, n_configs=2)
    rest = _rest_fields(equilibrium, 2)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05, 0.10], equilibrium_atoms=equilibrium,
        parity_projection=True, rest_snapshots=rest,
    )
    assert ensemble.ensemble_shape == (2, 2, 3, 2)
    eq = equilibrium.positions
    for i in range(3):
        for j in range(2):
            u_bin = snapshots[i][j].positions - eq
            u_rest = rest[j].positions - eq
            np.testing.assert_allclose(
                ensemble.snapshots[0, 1, i, j].positions, eq + u_bin - u_rest, atol=1e-12
            )


def test_rest_snapshots_validation(equilibrium):
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=3)
    rest = _rest_fields(equilibrium, 3)
    with pytest.raises(ValueError, match="requires parity_projection"):
        EnergyResolvedAtomsEnsemble(snapshots, [0.02, 0.05], rest_snapshots=rest)
    with pytest.raises(ValueError, match="one entry per"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
            parity_projection=True, rest_snapshots=rest[:2],
        )
    with pytest.raises(ValueError, match="same \\(energy, configuration\\) layout"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
            parity_projection=True, rest_snapshots=[rest],
        )
    wrong = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    with pytest.raises(ValueError, match="atoms but equilibrium_atoms"):
        EnergyResolvedAtomsEnsemble(
            snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
            parity_projection=True, rest_snapshots=[wrong] * 3,
        )


def test_rest_parity_ensemble_blocks_reconstruct(equilibrium):
    """The dask partition/reconstruction path must carry the 4D snapshot
    array through without re-twinning or re-applying the rest fields."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=2)
    rest = _rest_fields(equilibrium, 2)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
        parity_projection=True, rest_snapshots=rest,
    )
    blocks = ensemble.ensemble_blocks(chunks=1).compute()
    assert blocks.shape == (2, 2, 2, 2)
    member = blocks[0, 1, 1, 0]
    assert member.ensemble_shape == (1, 1, 1, 1)
    np.testing.assert_allclose(
        member.snapshots[0, 0, 0, 0].positions,
        ensemble.snapshots[0, 1, 1, 0].positions,
        atol=1e-12,
    )
