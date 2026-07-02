import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import given
from utils import gpu

from abtem.potentials.iam import CrystalPotential, Potential

# @given(atoms=abtem_st.atoms(),
#        gpts=abtem_st.gpts(),
#        num_configs=st.integers(min_value=1, max_value=3),
#        sigmas=st.floats(min_value=0., max_value=1.))
# @pytest.mark.parametrize('lazy', [True, False])
# def test_frozen_phonons_seed(atoms, gpts, lazy, num_configs, sigmas):
#     frozen_phonons = FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas, seeds=0)
#     potential1 = Potential(frozen_phonons, gpts=gpts).build(lazy=lazy).compute()
#     frozen_phonons = FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas, seeds=0)
#     potential2 = Potential(frozen_phonons, gpts=gpts).build(lazy=lazy).compute()
#     assert np.allclose(potential1.array.sum(0), potential2.array.sum(0))


@given(
    atoms=abtem_st.atoms(max_atomic_number=14),
    gpts=abtem_st.gpts(),
    slice_thickness=st.floats(min_value=1, max_value=2.0),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", [gpu, "cpu"])
@pytest.mark.parametrize("parametrization", ["kirkland", "lobato"])
@pytest.mark.parametrize("projection", ["finite", "infinite"])
def test_build(atoms, gpts, slice_thickness, lazy, device, parametrization, projection):
    potential = Potential(
        atoms,
        gpts=gpts,
        device=device,
        slice_thickness=slice_thickness,
        parametrization=parametrization,
        projection=projection,
    )
    potential_array = potential.build(lazy=lazy).compute()


@given(
    data=st.data(),
    tile=st.tuples(
        st.integers(min_value=1, max_value=2),
        st.integers(min_value=1, max_value=2),
        st.integers(min_value=1, max_value=2),
    ),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "potential_unit",
    [
        abtem_st.potential(projection="infinite", no_frozen_phonons=True),
        abtem_st.potential_array(max_ensemble_dims=0, lazy=True),
        abtem_st.potential_array(max_ensemble_dims=0, lazy=False),
    ],
)
def test_crystal_potential_builds(data, potential_unit, tile, lazy):
    potential_unit = data.draw(potential_unit)

    crystal_potential = CrystalPotential(potential_unit, tile)
    crystal_potential = crystal_potential.build(lazy=lazy).compute()

    try:
        potential_unit = potential_unit.build().compute()
    except RuntimeError:
        pass

    tiled_potential = potential_unit.compute().tile(tile)
    assert crystal_potential == tiled_potential
    assert len(crystal_potential) == len(potential_unit) * tile[2]
    assert crystal_potential.gpts == (
        potential_unit.gpts[0] * tile[0],
        potential_unit.gpts[1] * tile[1],
    )


@given(
    data=st.data(),
    num_frozen_phonons=st.integers(1, 3),
    tile=st.tuples(
        st.integers(min_value=1, max_value=2),
        st.integers(min_value=1, max_value=2),
        st.integers(min_value=1, max_value=2),
    ),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "potential_unit",
    [
        abtem_st.potential(projection="infinite"),
        abtem_st.potential_array(max_ensemble_dims=1, lazy=True),
        abtem_st.potential_array(max_ensemble_dims=1, lazy=False),
    ],
)
def test_crystal_potential_with_frozen_phonons(
    data, potential_unit, tile, num_frozen_phonons, lazy
):
    potential_unit = data.draw(potential_unit)

    crystal_potential = CrystalPotential(
        potential_unit, tile, num_frozen_phonons=num_frozen_phonons
    )

    crystal_potential = crystal_potential.build(lazy=lazy)

    assert num_frozen_phonons == crystal_potential.num_configurations

    crystal_potential.compute()

    assert num_frozen_phonons == crystal_potential.num_configurations


def test_crystal_potential_get_sliced_atoms_matches_manual_tile():
    """CrystalPotential.get_sliced_atoms tiles the unit's transformed atoms by
    the repetitions, matching a manually-tiled Potential's sliced atoms."""
    import ase
    import numpy as np

    from abtem.slicing import SliceIndexedAtoms

    unit_atoms = ase.build.bulk("Si", cubic=True)
    reps = (2, 2, 3)
    slice_thickness = float(unit_atoms.cell[2, 2])

    unit_pot = Potential(unit_atoms, gpts=(32, 32), slice_thickness=slice_thickness)
    cryst = CrystalPotential(unit_pot, repetitions=reps)

    manual = Potential(
        unit_atoms * reps, gpts=(64, 64), slice_thickness=slice_thickness
    )

    cryst_sa = cryst.get_sliced_atoms()
    manual_sa = manual.get_sliced_atoms()

    assert isinstance(cryst_sa, SliceIndexedAtoms)
    assert cryst_sa.num_slices == manual_sa.num_slices

    # Same atoms (order-independent) and same per-slice binning.
    assert np.allclose(
        np.sort(cryst_sa.atoms.positions, axis=0),
        np.sort(manual_sa.atoms.positions, axis=0),
    )
    for i in range(cryst_sa.num_slices):
        c = cryst_sa.get_atoms_in_slices(i)
        m = manual_sa.get_atoms_in_slices(i)
        assert len(c) == len(m)
        assert np.allclose(
            np.sort(c.positions, axis=0), np.sort(m.positions, axis=0)
        )


def test_crystal_potential_get_sliced_atoms_is_cached():
    """The sliced-atoms tile is non-trivial for big supercells; it must be
    cached on the instance (mirrors _FieldBuilderFromAtoms.get_sliced_atoms)."""
    import ase

    unit_pot = Potential(
        ase.build.bulk("Si", cubic=True), gpts=(16, 16), slice_thickness=5.43
    )
    cryst = CrystalPotential(unit_pot, repetitions=(2, 2, 2))
    assert cryst.get_sliced_atoms() is cryst.get_sliced_atoms()


def test_crystal_potential_get_sliced_atoms_frozen_phonons_equilibrium():
    """For a frozen-phonon CrystalPotential, get_sliced_atoms returns the
    equilibrium (un-displaced) atoms, because the ensemble draws an independent
    random unit configuration per z-repetition (no single displaced
    realisation) and column identification wants equilibrium positions."""
    import ase
    import numpy as np

    import abtem

    unit_atoms = ase.build.bulk("Si", cubic=True)
    fp = abtem.FrozenPhonons(unit_atoms, num_configs=3, sigmas=0.1, seed=7)
    unit_pot = Potential(fp, gpts=(32, 32), slice_thickness=5.43)
    # The unit already carries frozen phonons; CrystalPotential draws one of its
    # configs per repeated unit, so there is no single displaced realisation.
    cryst = CrystalPotential(unit_pot, repetitions=(2, 2, 2))

    sa = cryst.get_sliced_atoms()
    expected = (unit_atoms * (2, 2, 2)).positions
    assert np.allclose(
        np.sort(sa.atoms.positions, axis=0), np.sort(expected, axis=0)
    )


@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_eager_build_populates_all_frozen_phonon_configs(device):
    """Eager ``build(lazy=False)`` of a multi-config frozen-phonon potential must
    populate *every* ensemble member, not just the first. Regression for a bug
    where the ensemble write index was hardcoded to 0, so all configs
    overwrote config 0 and configs 1..N-1 were left as zeros -- which in turn
    made CrystalPotential (it builds its pool eagerly) reshuffle a pool of one
    real config plus N-1 vacuum slices."""
    import ase
    import numpy as np

    import abtem
    from abtem.core.backend import asnumpy

    unit_atoms = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True)
    num_configs = 4
    fp = abtem.FrozenPhonons(
        unit_atoms, num_configs=num_configs, sigmas=0.1, seed=7
    )
    potential = Potential(
        fp, gpts=(32, 32), slice_thickness=5.43 / 4, device=device
    )

    eager = potential.build(lazy=False).array
    lazy = potential.build(lazy=True).compute().array
    eager = asnumpy(eager)
    lazy = asnumpy(lazy)

    assert eager.shape[0] == num_configs
    # every config carries real (non-vacuum) potential (total mass is ~conserved
    # under displacement, so a positive sum is what distinguishes real from the
    # zero-filled vacuum slices the bug produced)
    per_config_sums = eager.reshape(num_configs, -1).sum(axis=1)
    assert np.all(per_config_sums > 0)
    # configs are genuinely distinct realisations (independent displacements) --
    # every config differs pixel-wise from config 0 (sums alone are conserved)
    for c in range(1, num_configs):
        assert np.abs(eager[c] - eager[0]).max() > 0
    # eager and lazy builds agree config-for-config
    assert np.allclose(eager, lazy)


@pytest.mark.parametrize("device", [gpu, "cpu"])
@pytest.mark.parametrize("lazy", [True, False])
def test_crystal_potential_frozen_phonons_lateral_disorder(lazy, device):
    """A frozen-phonon CrystalPotential must reproduce *lateral* (in-plane)
    disorder: each lateral repetition draws an independent configuration from
    the pool (a mosaic), so the tiles differ from one another. Regression for
    the original ``.tile()`` behaviour that replicated a single displaced unit
    across every tile -- giving zero in-plane disorder (and hence no diffuse /
    Kikuchi scattering)."""
    import ase
    import numpy as np

    import abtem
    from abtem.core.backend import asnumpy

    si = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True)
    reps = (2, 3, 2)  # asymmetric to catch tile-axis-order mistakes
    ug = 32
    fp = abtem.FrozenPhonons(si, num_configs=20, sigmas=0.1, seed=2)
    unit = Potential(fp, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    if lazy:
        unit = unit.build(lazy=True)

    cryst = CrystalPotential(unit, repetitions=reps)
    slic = next(cryst.generate_slices())
    arr = asnumpy(slic.array)[0]  # (reps[0]*ug, reps[1]*ug)
    assert arr.shape == (reps[0] * ug, reps[1] * ug)

    # reshape into the reps[0] x reps[1] lateral tiles and measure how much the
    # tiles differ at matched within-tile pixels
    tiles = arr.reshape(reps[0], ug, reps[1], ug)
    inter_tile_std = float(tiles.std(axis=(0, 2)).mean())

    # a single-config pool has no disorder to reproduce -> tiles are identical
    # copies (up to float rounding), which fixes the disorder floor to compare
    # against
    fp1 = abtem.FrozenPhonons(si, num_configs=1, sigmas=0.1, seed=2)
    unit1 = Potential(fp1, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    if lazy:
        unit1 = unit1.build(lazy=True)
    slic1 = next(CrystalPotential(unit1, repetitions=reps).generate_slices())
    arr1 = asnumpy(slic1.array)[0]
    tiles1 = arr1.reshape(reps[0], ug, reps[1], ug)
    single_config_floor = float(tiles1.std(axis=(0, 2)).mean())

    # the multi-config mosaic must show real lateral disorder, orders of
    # magnitude above the single-config (identical-tiles) rounding floor
    assert single_config_floor < 1e-2
    assert inter_tile_std > 100 * single_config_floor


@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_crystal_potential_pool_enlarged_to_avoid_lateral_duplication(device):
    """When the frozen-phonon pool is smaller than the number of lateral tiles,
    CrystalPotential enlarges it (warning) so every tile draws a distinct
    configuration and no two tiles in a layer are identical."""
    import ase
    import numpy as np

    import abtem
    from abtem.core.backend import asnumpy

    si = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True)
    reps = (5, 4, 2)  # 20 lateral tiles
    ug = 24
    n_tiles = reps[0] * reps[1]

    fp = abtem.FrozenPhonons(si, num_configs=6, sigmas=0.1, seed=0)  # pool < tiles
    unit = Potential(fp, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    cryst = CrystalPotential(unit, repetitions=reps)

    with pytest.warns(UserWarning, match="smaller than the number of lateral"):
        slic = next(cryst.generate_slices())

    arr = asnumpy(slic.array)[0]
    tiles = arr.reshape(reps[0], ug, reps[1], ug).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(n_tiles, ug, ug)
    # every lateral tile is a distinct realisation -> no duplication
    keys = {t.round(6).tobytes() for t in tiles}
    assert len(keys) == n_tiles

    # a pool already >= n_tiles is left untouched (no warning)
    fp_big = abtem.FrozenPhonons(si, num_configs=n_tiles, sigmas=0.1, seed=0)
    unit_big = Potential(fp_big, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any pool warning would fail here
        big = next(
            CrystalPotential(unit_big, repetitions=reps).generate_slices()
        )
    arr_big = asnumpy(big.array)[0]
    tiles_big = arr_big.reshape(reps[0], ug, reps[1], ug).transpose(0, 2, 1, 3)
    assert len({t.round(6).tobytes() for t in tiles_big.reshape(n_tiles, ug, ug)}) == (
        n_tiles
    )


@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_crystal_potential_balanced_pool_drawing(device):
    """Pool configurations are drawn without replacement over the WHOLE
    crystal (balanced budgets), not just within a z-layer: a pool matching
    the total number of unit-cell slots gives every slot a distinct
    configuration (statistically identical to tiling displaced atoms), and a
    smaller pool spreads reuse exactly evenly."""
    from collections import Counter

    import ase

    import abtem
    from abtem.core.backend import asnumpy

    si = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True)
    ug = 16

    # z-only pool (full-lateral pattern): pool == nz -> every z-rep distinct
    nz = 6
    fp = abtem.FrozenPhonons(si, num_configs=nz, sigmas=0.1, seed=1)
    unit = Potential(fp, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    cryst = CrystalPotential(unit, repetitions=(1, 1, nz), seeds=(3,))
    slices = [asnumpy(s.array)[0] for s in cryst.generate_slices()]
    n_sub = len(slices) // nz
    reps = {slices[i * n_sub].round(6).tobytes() for i in range(nz)}
    assert len(reps) == nz

    # mosaic: pool == n_tiles * nz -> every (tile, z) slot distinct
    tile_reps = (3, 2, 2)
    n_tiles = tile_reps[0] * tile_reps[1]
    total_slots = n_tiles * tile_reps[2]
    fp2 = abtem.FrozenPhonons(si, num_configs=total_slots, sigmas=0.1, seed=1)
    unit2 = Potential(fp2, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    cryst2 = CrystalPotential(unit2, repetitions=tile_reps, seeds=(3,))
    slices2 = [asnumpy(s.array)[0] for s in cryst2.generate_slices()]
    slots = set()
    for i in range(tile_reps[2]):
        layer = slices2[i * n_sub]
        tiles = layer.reshape(tile_reps[0], ug, tile_reps[1], ug).transpose(0, 2, 1, 3)
        slots.update(t.round(6).tobytes() for t in tiles.reshape(n_tiles, ug, ug))
    assert len(slots) == total_slots

    # pool == n_tiles: usage perfectly balanced (each config used exactly nz
    # times over the crystal) and still distinct within every layer
    fp3 = abtem.FrozenPhonons(si, num_configs=n_tiles, sigmas=0.1, seed=1)
    unit3 = Potential(fp3, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    cryst3 = CrystalPotential(unit3, repetitions=tile_reps, seeds=(3,))
    slices3 = [asnumpy(s.array)[0] for s in cryst3.generate_slices()]
    counts = Counter()
    for i in range(tile_reps[2]):
        layer = slices3[i * n_sub]
        tiles = layer.reshape(tile_reps[0], ug, tile_reps[1], ug).transpose(0, 2, 1, 3)
        layer_keys = [t.round(6).tobytes() for t in tiles.reshape(n_tiles, ug, ug)]
        assert len(set(layer_keys)) == n_tiles  # in-plane distinctness kept
        counts.update(layer_keys)
    assert set(counts.values()) == {tile_reps[2]}


def test_crystal_potential_get_sliced_atoms_raises_for_array_unit():
    """A precomputed PotentialArray unit has no atoms, so get_sliced_atoms must
    raise an actionable error rather than failing obscurely downstream."""
    import ase

    unit_pot = Potential(
        ase.build.bulk("Si", cubic=True), gpts=(16, 16), slice_thickness=5.43
    ).build(lazy=False)
    cryst = CrystalPotential(unit_pot, repetitions=(1, 1, 2))
    with pytest.raises(RuntimeError, match="get_transformed_atoms"):
        cryst.get_sliced_atoms()


# @given(data=st.data(),
#        tile=st.tuples(st.integers(min_value=1, max_value=2),
#                       st.integers(min_value=1, max_value=2),
#                       st.integers(min_value=1, max_value=2)))
# @pytest.mark.parametrize('lazy', [True, False])
# @pytest.mark.parametrize('device', [gpu, 'cpu'])
# @pytest.mark.parametrize('potential_unit', [
#     abtem_st.potential,
# ])
# def test_crystal_potential_with_frozen_phonons(data, potential, tile, lazy, device):
#     potential_unit = data.draw(abtem_st.potential(device=device,
#                                                   projection='infinite',
#                                                   ))
#
#     crystal_potential = CrystalPotential(potential_unit, tile, num_frozen_phonons=3)
#     crystal_potential = crystal_potential.build(lazy=lazy).compute()

# potential_unit = potential_unit.build(lazy=lazy).compute()

# tiled_potential = potential_unit.compute().tile(tile)
# assert crystal_potential == tiled_potential

# @settings(max_examples=2)
# @given(Z=st.integers(1, 14),
#        slice_thickness=st.floats(min_value=.5, max_value=4.)
#        )
# @pytest.mark.parametrize('parametrization', ['kirkland', 'lobato'])
# def test_finite_infinite_projected_match(Z, slice_thickness, parametrization):
#     atoms = Atoms([Z], positions=[(0., 0., 3.)], cell=[6., 6., 6.])
#     finite_potential = Potential(atoms,
#                                  sampling=0.01,
#                                  projection='finite',
#                                  slice_thickness=slice_thickness,
#                                  parametrization=parametrization)
#
#     finite_potential = finite_potential.build(lazy=False).project()
#
#     infinite_potential = Potential(atoms,
#                                    sampling=0.01,
#                                    projection='infinite',
#                                    slice_thickness=slice_thickness,
#                                    parametrization=parametrization)
#     infinite_potential = infinite_potential.build(lazy=False).project()
#
#     mask = np.ones_like(finite_potential.array, dtype=bool)
#     mask[0, 0] = 0
#     assert array_is_close(finite_potential.array, infinite_potential.array, rel_tol=.01, check_above_rel=.1, mask=mask)


# @given(Z=st.integers(1, 102),
#        slice_thickness=st.floats(min_value=2, max_value=4.),
#        sampling=st.floats(min_value=0.01, max_value=0.02))
# @pytest.mark.parametrize('parametrization', [LobatoParametrization(), KirklandParametrization()])
# def test_infinite_projected_match(Z, slice_thickness, parametrization, sampling):
#     sidelength = 8
#
#     atoms = Atoms([Z], positions=[(0., 0., sidelength / 2)], cell=[sidelength, sidelength, sidelength])
#
#     potential = Potential(atoms,
#                           slice_thickness=slice_thickness,
#                           sampling=sampling,
#                           projection='infinite',
#                           parametrization=parametrization)
#
#     r = np.linspace(0, sidelength, potential.gpts[0], endpoint=False)[1:]
#     analytical_potential = parametrization.projected_potential(Z)(r)
#
#     potential = potential.build(lazy=False).project().array[0, 1:]
#     assert array_is_close(potential, analytical_potential, rel_tol=.01, check_above_rel=.01)


# @settings(max_examples=2)
# @given(Z=st.integers(1, 50),
#        slice_thickness=st.floats(min_value=2, max_value=4.),
#        sampling=st.floats(min_value=0.025, max_value=0.05))
# @pytest.mark.parametrize('parametrization', [LobatoParametrization(), KirklandParametrization()])
# def test_finite_projected_match(Z, slice_thickness, parametrization, sampling):
#     sidelength = 6
#     atoms = Atoms([Z], positions=[(0., 0., sidelength / 2)], cell=[sidelength, sidelength, sidelength])
#
#     potential = Potential(atoms,
#                           slice_thickness=slice_thickness,
#                           sampling=sampling,
#                           projection='finite',
#                           parametrization=parametrization)
#
#     r = np.linspace(0, sidelength, potential.gpts[0], endpoint=False)[1:]
#     analytical_potential = parametrization.projected_potential(Z)(r)
#
#     potential = potential.build(lazy=False).project().array[0, 1:]
#     assert array_is_close(potential, analytical_potential, rel_tol=.01, check_above_rel=.01)
#
# # def test_atom_position():
#     from ase import Atoms
#
#     L = 8.0
#     z1 = 0
#     z2 = L / 2
#
#     atoms1 = Atoms('C', [(L / 2, L / 2, z1)], cell=(L,) * 3)
#     atoms2 = Atoms('C', [(L / 2, L / 2, z2)], cell=(L,) * 3)
#
#     potential1 = Potential(atoms1, sampling=.1, projection='finite', slice_thickness=L)
#     potential2 = Potential(atoms2, sampling=.1, projection='finite', slice_thickness=L)
#
#     # print(potential1.num_slices, potential2.num_slices)
#
#     potential1 = potential1.build(lazy=False)
#     potential2 = potential2.build(lazy=False)


# --- depth_profile tests ---


@pytest.fixture
def si_potential():
    """Build a Si 2x2x5 potential for depth profile tests."""
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (2, 2, 5)
    return Potential(atoms, slice_thickness=1.0, gpts=(32, 32))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_potential_depth_profile_shape(si_potential, device):
    pot = si_potential.build().compute()
    profile = pot.depth_profile()
    n_x = pot.gpts[0]
    n_z = pot.num_slices
    assert profile.shape == (n_x, n_z)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_potential_depth_profile_x_projection(si_potential, device):
    pot = si_potential.build().compute()
    profile = pot.depth_profile(projection_axis="x")
    n_y = pot.gpts[1]
    n_z = pot.num_slices
    assert profile.shape == (n_y, n_z)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_potential_depth_profile_sampling(si_potential, device):
    pot = si_potential.build().compute()
    profile = pot.depth_profile()

    assert np.isclose(profile.sampling[0], pot.sampling[0])

    expected_z_sampling = pot.thickness / pot.num_slices
    assert np.isclose(profile.sampling[1], expected_z_sampling)


def test_potential_depth_profile_invalid_axis(si_potential):
    pot = si_potential.build().compute()
    with pytest.raises(ValueError, match="projection_axis"):
        pot.depth_profile(projection_axis="z")


def test_potential_depth_profile_finite_depth(si_potential):
    pot = si_potential.build().compute()
    full = pot.depth_profile()
    partial = pot.depth_profile(depth=3.0)
    assert full.shape == partial.shape
    assert partial.array.sum() < full.array.sum()


def test_potential_depth_profile_lazy_delegation(si_potential):
    profile_lazy = si_potential.depth_profile()
    profile_built = si_potential.build().compute().depth_profile()
    assert profile_lazy.shape == profile_built.shape
    assert np.allclose(profile_lazy.array, profile_built.array)


def test_potential_show_depth_profile(si_potential):
    import matplotlib

    matplotlib.use("Agg")
    from abtem.visualize import Visualization

    viz = si_potential.show_depth_profile()
    assert isinstance(viz, Visualization)
