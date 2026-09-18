"""Tests for EnergyResolvedAtomsEnsemble's parity_projection option
(separating one-phonon from multi-phonon scattering, issue #373)."""

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic

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


def test_ensemble_mean_forced_false_does_not_warn(equilibrium):
    """Forcing ensemble_mean=False for parity_projection is deliberate, not
    a user oversight -- reduce_ensemble's generic "did you forget
    ensemble_mean=True" warning must not fire for it, while it must still
    fire for an ordinary (non-parity) ensemble_mean=False. Scoped to just
    the reduce_ensemble() call (rather than e.g. pytest's recwarn, which
    records the whole test) so this can't be tripped up by an unrelated
    warning from a dependency raised during fixture/object setup."""
    import warnings

    import numpy as np

    from abtem.waves import Waves, reduce_ensemble

    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=2)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02], equilibrium_atoms=equilibrium, parity_projection=True,
    )
    fp_axis = next(
        ax for ax in ensemble.ensemble_axes_metadata
        if isinstance(ax, FrozenPhononsAxis)
    )
    assert fp_axis._ensemble_mean is False
    assert fp_axis._ensemble_mean_forced is True

    waves = Waves(
        np.zeros((2, 4, 4), dtype=complex), energy=100e3, sampling=0.1,
        ensemble_axes_metadata=[fp_axis],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reduce_ensemble(waves)
    assert not any("ensemble_mean=False" in str(w.message) for w in caught)

    # an ordinary (non-parity) ensemble_mean=False axis must still warn
    ordinary_fp_axis = FrozenPhononsAxis(_ensemble_mean=False)
    assert ordinary_fp_axis._ensemble_mean_forced is False
    waves_ordinary = Waves(
        np.zeros((2, 4, 4), dtype=complex), energy=100e3, sampling=0.1,
        ensemble_axes_metadata=[ordinary_fp_axis],
    )
    with pytest.warns(UserWarning, match="ensemble_mean=False"):
        reduce_ensemble(waves_ordinary)


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


def test_getitem_ambiguous_forms_not_supported_with_parity_projection(equilibrium):
    """Bare/2D-style indexing (as used by a non-parity ensemble) is
    ambiguous once a leading parity axis exists -- ensemble[0] could mean
    "energy 0" (matching non-parity semantics) or "parity member 0"
    (matching the actual leading axis), so both remain unsupported."""
    snapshots = _make_snapshots(equilibrium)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots,
        [0.02, 0.05, 0.10],
        equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    with pytest.raises(NotImplementedError, match="leading ':'"):
        ensemble[0]
    with pytest.raises(NotImplementedError, match="leading ':'"):
        ensemble[1, 0]


def test_getitem_with_explicit_leading_colon_supported_with_parity_projection(
    equilibrium,
):
    """ensemble[:, ...] is unambiguous (explicitly keeps the parity axis
    whole) and is supported: energy/config indexing behaves the same as
    the non-parity case, applied identically to both the real and twin
    halves."""
    snapshots = _make_snapshots(equilibrium, n_energies=3, n_configs=4)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots,
        [0.02, 0.05, 0.10],
        equilibrium_atoms=equilibrium,
        parity_projection=True,
    )

    # single energy -> (2, 1, n_configs)
    sub = ensemble[:, 1]
    assert sub.ensemble_shape == (2, 1, 4)
    assert sub.parity_projection is True
    np.testing.assert_allclose(sub.energies, [0.05])
    for c in range(4):
        np.testing.assert_allclose(
            sub.snapshots[0, 0, c].positions,
            ensemble.snapshots[0, 1, c].positions,
        )
        np.testing.assert_allclose(
            sub.snapshots[1, 0, c].positions,
            ensemble.snapshots[1, 1, c].positions,
        )

    # energy slice + config slice
    sub2 = ensemble[:, 1:3, 0:2]
    assert sub2.ensemble_shape == (2, 2, 2)
    np.testing.assert_allclose(sub2.energies, [0.05, 0.10])

    # bare ensemble[:] keeps everything, unchanged
    sub3 = ensemble[:]
    assert sub3.ensemble_shape == ensemble.ensemble_shape
    np.testing.assert_allclose(sub3.energies, ensemble.energies)


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


def test_direct_3d_construction_is_validated(equilibrium):
    """Passing a pre-built 3D (parity, energy, config) snapshots array
    straight through the public constructor -- not the documented 2D
    real-configurations-only shape -- must still be checked against
    equilibrium_atoms, not silently skipped just because ndim == 3."""
    snapshots = _make_snapshots(equilibrium, n_energies=1, n_configs=2)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02], equilibrium_atoms=equilibrium, parity_projection=True,
    )
    array_3d = ensemble.snapshots.copy()

    # corrupt the twin half with a huge, unrelated displacement
    corrupted = array_3d[1, 0, 0].copy()
    corrupted.positions += [[5.0, 0, 0], [0, 0, 0]]
    array_3d[1, 0, 0] = corrupted

    with pytest.raises(ValueError, match="max_displacement"):
        EnergyResolvedAtomsEnsemble(
            array_3d, [0.02], equilibrium_atoms=equilibrium,
            parity_projection=True,
        )

    # a species mismatch on the pre-built 3D array must also be caught
    wrong_species = array_3d[1, 0, 0].copy()
    wrong_species.numbers = [1, 2]
    array_3d_species = ensemble.snapshots.copy()
    array_3d_species[1, 0, 0] = wrong_species
    with pytest.raises(ValueError, match="species sequence"):
        EnergyResolvedAtomsEnsemble(
            array_3d_species, [0.02], equilibrium_atoms=equilibrium,
            parity_projection=True,
        )


def test_reconstructed_chunk_skips_redundant_validation(equilibrium):
    """The internal _validated=True path (used when dask reconstructs a
    chunk of an already-validated ensemble) must still work transparently
    -- this is what generate_blocks/ensemble_blocks exercise."""
    snapshots = _make_snapshots(equilibrium, n_energies=2, n_configs=2)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05], equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    array_3d = ensemble.snapshots

    # explicit internal opt-out of re-validation: must not raise even
    # though max_displacement=0.0 would reject every snapshot if checked
    rebuilt = EnergyResolvedAtomsEnsemble(
        array_3d, [0.02, 0.05], equilibrium_atoms=equilibrium,
        parity_projection=True, max_displacement=0.0, _validated=True,
    )
    assert rebuilt.ensemble_shape == (2, 2, 2)

    # the default (_validated=False) does re-check, and 0.0 rejects
    with pytest.raises(ValueError, match="max_displacement"):
        EnergyResolvedAtomsEnsemble(
            array_3d, [0.02, 0.05], equilibrium_atoms=equilibrium,
            parity_projection=True, max_displacement=0.0,
        )


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


def test_wrapped_snapshot_passes_minimum_image_check_2d_material():
    """A 2D-material cell built the standard ASE way (e.g. ase.build.
    graphene()'s defaults) has a degenerate/zero out-of-plane cell vector,
    so atoms.cell.rank < 3 even though the in-plane directions are fully
    periodic. The minimum-image check must still wrap those in-plane
    directions instead of silently skipping the correction (which would
    make an ordinary in-plane PBC wrap look like a multi-angstrom
    displacement and spuriously fail max_displacement)."""
    equilibrium_2d = Atoms(
        "C2",
        positions=[[0, 0, 0], [1.42, 0, 0]],
        cell=[[2.84, 0, 0], [-1.42, 2.46, 0], [0, 0, 0]],
        pbc=(True, True, False),
    )
    assert equilibrium_2d.cell.rank < 3

    u = np.array([0.03, -0.02, 0.0])
    atoms = equilibrium_2d.copy()
    atoms.positions[0] += u
    atoms.positions[0] += atoms.cell[0]  # push across the periodic x boundary
    atoms.wrap()
    assert np.linalg.norm(atoms.positions[0] - equilibrium_2d.positions[0]) > 1.0

    # must not raise despite the >1 A raw (unwrapped) displacement
    ensemble = EnergyResolvedAtomsEnsemble(
        [[atoms, atoms]], [0.02], equilibrium_atoms=equilibrium_2d,
        parity_projection=True,
    )
    twin = ensemble.snapshots[1, 0, 0]
    expected = equilibrium_2d.positions[0] - u
    diff = twin.positions[0] - expected
    vmin, _ = find_mic(diff[None, :], equilibrium_2d.cell, pbc=equilibrium_2d.pbc)
    np.testing.assert_allclose(vmin[0], 0.0, atol=1e-10)


def test_wrapped_snapshot_passes_minimum_image_check_pbc_false():
    """A bulk cell built by hand carries ASE's default pbc=False, but abTEM
    treats it as periodic along every cell vector; the minimum-image check
    must wrap along those directions regardless of the pbc flags."""
    equilibrium_bulk = Atoms("H2", positions=[[0, 0, 0], [1.0, 0, 0]], cell=[10, 10, 10])
    assert not equilibrium_bulk.pbc.any()

    u = np.array([-0.05, 0.02, 0.0])
    atoms = equilibrium_bulk.copy()
    atoms.positions[0] += u
    atoms.wrap(pbc=True)
    assert atoms.positions[0, 0] > 9.0  # really wrapped

    ensemble = EnergyResolvedAtomsEnsemble(
        [[atoms, atoms]], [0.02], equilibrium_atoms=equilibrium_bulk,
        parity_projection=True,
    )
    assert ensemble.ensemble_shape == (2, 1, 2)


def test_getitem_rejects_array_index_with_parity_projection(equilibrium):
    snapshots = _make_snapshots(equilibrium)
    ensemble = EnergyResolvedAtomsEnsemble(
        snapshots, [0.02, 0.05, 0.10], equilibrium_atoms=equilibrium,
        parity_projection=True,
    )
    with pytest.raises(NotImplementedError, match="leading ':'"):
        ensemble[np.array([0, 1])]
