import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from hypothesis import given
from utils import gpu

from abtem.core.grid import disk_meshgrid
from abtem.integrals import (
    QuadratureProjectionIntegrals,
    _threaded_interpolate_radial_functions,
    interpolate_radial_functions,
)
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
@pytest.mark.parametrize("projection", ["finite", "infinite"])
@pytest.mark.parametrize("parametrization", ["kirkland", "lobato"])
def test_build_parametrizations(atoms, gpts, slice_thickness, parametrization, projection):
    potential = Potential(
        atoms,
        gpts=gpts,
        slice_thickness=slice_thickness,
        parametrization=parametrization,
        projection=projection,
    )
    potential.build(lazy=False).compute()


@given(
    atoms=abtem_st.atoms(max_atomic_number=14),
    gpts=abtem_st.gpts(),
    slice_thickness=st.floats(min_value=1, max_value=2.0),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_build_device_lazy(atoms, gpts, slice_thickness, lazy, device):
    potential = Potential(
        atoms,
        gpts=gpts,
        device=device,
        slice_thickness=slice_thickness,
    )
    potential.build(lazy=lazy).compute()


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
        warnings.filterwarnings("error", category=UserWarning)  # any pool warning would fail here
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


@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_crystal_potential_ensemble_members_have_independent_pools(device):
    """Ensemble members (num_frozen_phonons / seeds) all share the same
    ``potential_unit`` object, so without reseeding they would rebuild the
    identical, fixed pool of atomic snapshots and differ only in how those
    same snapshots are arranged -- not in which displacements exist. Each
    member's pool must instead be independently reseeded from that member's
    own seed, even when the pool is already at (or above) the size needed for
    a single crystal to be exact, so the ensemble does not need to be sized
    for the number of members."""
    import ase

    import abtem
    from abtem.core.backend import asnumpy

    si = ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True)
    ug = 16
    nz = 6  # pool == nz: already exact for one member (see test above)

    fp = abtem.FrozenPhonons(si, num_configs=nz, sigmas=0.1, seed=1)
    unit = Potential(fp, gpts=(ug, ug), slice_thickness=5.43 / 4, device=device)
    cryst = CrystalPotential(unit, repetitions=(1, 1, nz), num_frozen_phonons=3)
    built = cryst.build(lazy=False)
    members = [asnumpy(built.array[i]) for i in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.allclose(members[i], members[j])

    # same seeds -> bit-reproducible
    cryst_again = CrystalPotential(unit, repetitions=(1, 1, nz), seeds=cryst.seeds)
    built_again = cryst_again.build(lazy=False)
    for i in range(3):
        np.testing.assert_allclose(
            members[i], asnumpy(built_again.array[i])
        )


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


@pytest.mark.parametrize(
    "position",
    [
        (0.0, 0.0),  # atom sits exactly on a grid point
        (0.31, 0.47),  # atom offset by a sub-pixel amount in both directions
        (1.9, -1.9),  # atom offset near the edge of the truncated disk
    ],
)
def test_interpolate_radial_functions_disk_truncation_matches_full_disk(position):
    # The lateral disk-truncation optimization in QuadratureProjectionIntegrals
    # relies on interpolate_radial_functions correctly stopping at
    # disk_counts[i] once the disk is sorted by radial distance. Verify this
    # directly against calling it with the untruncated (full) disk, which is
    # the behavior prior to the optimization.
    sampling = (0.1, 0.1)
    radial_gpts = np.geomspace(0.05, 3.0, 64)
    radial_functions = np.exp(-radial_gpts)[None].astype(np.float64)
    radial_derivative = np.zeros_like(radial_functions)
    radial_derivative[:, :-1] = np.diff(radial_functions, axis=1) / np.diff(radial_gpts)

    positions = np.array([position], dtype=np.float64)

    # Deliberately oversized disk (as if this atom's slice offset were 0 but
    # a sibling atom in the same call needed a much larger disk radius), so
    # that disk_counts genuinely truncates away real, non-empty pixels rather
    # than just the ceiling-rounding pad at the disk's own edge.
    disk = disk_meshgrid(int(np.ceil(2 * radial_gpts[-1] / min(sampling))))
    disk_radii = np.hypot(disk[:, 0] * sampling[0], disk[:, 1] * sampling[1])
    order = np.argsort(disk_radii)
    disk = disk[order]
    disk_radii = disk_radii[order]

    margin = np.hypot(sampling[0], sampling[1]) / 2
    disk_counts_truncated = np.searchsorted(
        disk_radii, radial_gpts[-1] + margin, side="right"
    )
    disk_counts_full = np.array([disk.shape[0]])

    gpts = (64, 64)
    array_truncated = np.zeros(gpts, dtype=np.float64)
    array_full = np.zeros(gpts, dtype=np.float64)

    interpolate_radial_functions(
        array=array_truncated,
        positions=positions,
        disk_indices=disk,
        disk_counts=np.array([disk_counts_truncated]),
        sampling=sampling,
        radial_gpts=radial_gpts,
        radial_functions=radial_functions,
        radial_derivative=radial_derivative,
    )
    interpolate_radial_functions(
        array=array_full,
        positions=positions,
        disk_indices=disk,
        disk_counts=disk_counts_full,
        sampling=sampling,
        radial_gpts=radial_gpts,
        radial_functions=radial_functions,
        radial_derivative=radial_derivative,
    )

    assert disk_counts_truncated < disk.shape[0]
    np.testing.assert_allclose(array_truncated, array_full, atol=1e-12)


def test_threaded_interpolation_matches_serial_kernel():
    # The thread-pool wrapper deals atoms round-robin to per-thread buffers
    # and sums them; up to float summation reordering this must match calling
    # the serial kernel directly with all atoms.
    rng = np.random.default_rng(7)
    sampling = (0.1, 0.12)
    gpts = (96, 80)
    n_atoms = 37  # deliberately not divisible by typical thread counts

    radial_gpts = np.geomspace(0.05, 3.0, 64)
    radial_functions = (
        np.exp(-radial_gpts)[None] * rng.uniform(0.5, 2.0, (n_atoms, 1))
    ).astype(np.float64)
    radial_derivative = np.zeros_like(radial_functions)
    radial_derivative[:, :-1] = np.diff(radial_functions, axis=1) / np.diff(radial_gpts)

    positions = np.zeros((n_atoms, 3))
    positions[:, 0] = rng.uniform(-1.0, gpts[0] * sampling[0] + 1.0, n_atoms)
    positions[:, 1] = rng.uniform(-1.0, gpts[1] * sampling[1] + 1.0, n_atoms)

    disk = disk_meshgrid(int(np.ceil(radial_gpts[-1] / min(sampling))))
    disk_radii = np.hypot(disk[:, 0] * sampling[0], disk[:, 1] * sampling[1])
    order = np.argsort(disk_radii)
    disk = np.ascontiguousarray(disk[order])
    disk_radii = disk_radii[order]
    disk_counts = np.searchsorted(
        disk_radii, rng.uniform(1.0, 3.0, n_atoms), side="right"
    )

    array_serial = np.zeros(gpts, dtype=np.float64)
    interpolate_radial_functions(
        array_serial,
        positions,
        disk,
        disk_counts,
        sampling,
        radial_gpts,
        radial_functions,
        radial_derivative,
    )

    array_threaded = np.zeros(gpts, dtype=np.float64)
    _threaded_interpolate_radial_functions(
        array_threaded,
        positions,
        disk,
        disk_counts,
        sampling,
        radial_gpts,
        radial_functions,
        radial_derivative,
    )

    np.testing.assert_allclose(array_threaded, array_serial, rtol=1e-12, atol=1e-12)


def test_finite_projection_tolerance_matches_tight_reference():
    # Regression test for the lateral disk-truncation optimization in
    # QuadratureProjectionIntegrals.integrate_on_grid: build a potential with
    # atoms deliberately placed at a slice boundary (dz=0, the edge case for
    # the truncation formula) and off it, and check that the default
    # cutoff_tolerance still agrees closely with a much tighter tolerance.
    atoms = Atoms(
        "Au2",
        positions=[(3.0, 3.0, 2.0), (3.0, 3.0, 5.0)],
        cell=(6.0, 6.0, 6.0),
        pbc=True,
    )

    def build(tol):
        integrator = QuadratureProjectionIntegrals(cutoff_tolerance=tol)
        potential = Potential(
            atoms,
            sampling=0.1,
            slice_thickness=2.0,
            projection="finite",
            integrator=integrator,
        )
        return potential.build(lazy=False).array

    tight = build(1e-6)
    default = build(1e-4)

    max_dev = np.abs(default - tight).max() / tight.max()
    assert max_dev < 1e-2


@pytest.mark.parametrize("device", [gpu])
def test_finite_projection_gpu_matches_cpu_near_atom_core(device):
    """Regression for a GPU/CPU mismatch in the radial-table index lookup
    inside interpolate_radial_functions (abtem/core/_cuda.py): the CUDA
    kernel used int() to derive the log-spaced table index, which truncates
    toward zero, while the CPU kernel uses int(floor(...)). For pixels just
    inside the table's innermost tabulated radius the argument is a small
    negative number, and the two roundings disagree in sign for values in
    (-1, 0) -- e.g. int(floor(-0.2)) == -1 (correctly clamps to the innermost
    tabulated value) but int(-0.2) == 0 (incorrectly extrapolates from the
    innermost table point instead). This only misfires for atoms whose
    fractional pixel offset happens to place a disk pixel in that razor-thin
    annulus, so a generic small structure may not trigger it; SrTiO3 (highly
    symmetric fractional coordinates) reliably does.
    """
    from ase.spacegroup import crystal

    sto = crystal(
        ("Sr", "Ti", "O"),
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    )
    atoms = sto * (2, 2, 2)

    def build(dev):
        pot = Potential(
            atoms, sampling=0.05, slice_thickness=1.0, projection="finite",
            device=dev,
        )
        array = pot.build(lazy=False).array
        if hasattr(array, "get"):
            array = array.get()
        return np.asarray(array, dtype=np.float64)

    cpu = build("cpu")
    gpu_array = build(device)

    max_dev = np.abs(gpu_array - cpu).max() / cpu.max()
    assert max_dev < 1e-4, f"GPU vs CPU max relative deviation {max_dev:.3e}"


def test_potential_array_slicing_maps_exit_planes():
    # slicing a potential array must map its exit planes into the sliced range,
    # otherwise the exit plane can fall outside the slices and the multislice
    # algorithm silently returns an unpropagated wave function
    import abtem
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (2, 2, 8)
    potential = Potential(atoms, gpts=128, slice_thickness=2.0).build(lazy=False)

    assert potential.exit_planes == (potential.num_slices - 1,)

    for item in (slice(None, 9), slice(5, 15), slice(None, 1)):
        sliced = potential[item]
        assert sliced.exit_planes == (sliced.num_slices - 1,)
        assert max(sliced.exit_planes) < sliced.num_slices

    # splitting the multislice algorithm at a slice boundary is exact
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    whole = waves.multislice(potential)
    split = waves.multislice(potential[:9]).multislice(potential[9:])

    assert np.allclose(whole.array, split.array)


class TestIntegratorSharedAcrossEnsemble:
    """generate_blocks()/ensemble_blocks() reconstruct a fresh Potential per
    ensemble member. The integrator must be passed through by reference, not
    deep-copied, or every member silently rebuilds (and, on GPU, re-uploads)
    the per-symbol projection tables and disk caches that QuadratureProjection-
    Integrals/ScatteringFactorProjectionIntegrals exist specifically to avoid
    recomputing across repeated calls."""

    def test_integrator_identity_preserved_across_members(self):
        import abtem

        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
        fp = abtem.FrozenPhonons(atoms, num_configs=4, sigmas=0.05, seed=1)
        potential = Potential(fp, sampling=0.2, slice_thickness=1.0, projection="finite")

        original_integrator = potential.integrator
        # Populate the cache the way a real build would, so a deep copy has
        # actual cached state to (wrongly) duplicate.
        original_integrator.get_integral_table("Si", potential.sampling)

        seen_integrator_ids = set()
        seen_table_ids = set()
        n_members = 0
        for _, _, block in potential.generate_blocks():
            block = block.item()
            n_members += 1
            seen_integrator_ids.add(id(block.integrator))
            seen_table_ids.add(id(block.integrator.tables))

        assert n_members == 4
        assert seen_integrator_ids == {id(original_integrator)}
        assert seen_table_ids == {id(original_integrator.tables)}

    @pytest.mark.parametrize("projection", ["finite", "infinite"])
    def test_shared_integrator_does_not_change_results(self, projection):
        import abtem

        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
        fp = abtem.FrozenPhonons(atoms, num_configs=4, sigmas=0.05, seed=1)
        potential = Potential(
            fp, sampling=0.2, slice_thickness=1.0, projection=projection
        )

        waves = abtem.PlaneWave(energy=100e3)
        exit_waves = waves.multislice(potential, lazy=False)

        # Rebuilding member-by-member via the (shared-integrator) path used
        # by generate_blocks() must match the direct build for every member.
        for index, _, block in potential.generate_blocks():
            block = block.item()
            direct = abtem.PlaneWave(energy=100e3).multislice(block, lazy=False)
            np.testing.assert_allclose(
                exit_waves.array[index], direct.array[0], atol=1e-10
            )


class TestPotentialDoesNotMutateItsAtoms:
    """Building a potential rewrote the ``Atoms`` it was given.

    ``Potential._prepare_atoms`` wrapped in place. For ``DummyFrozenPhonons``
    -- the wrapper every plain ``Potential(atoms)`` gets --
    ``get_transformed_atoms()`` and ``randomize()`` are both the identity, so
    the write landed on the object the potential stores and ships into the task
    graph as a single shared node. Every task on a worker then wrapped the same
    ``Atoms``.

    Two of the three entry points alias the **caller's** object, not merely
    abTEM's internal copy: ``_validate_frozen_phonons`` copies a plain
    ``Atoms``, but passes a list (which becomes an ``AtomsEnsemble`` holding
    references) and a pre-built frozen-phonons object straight through.

    ``FrozenPhonons`` is unaffected -- its ``randomize`` already copies -- which
    is the oracle this fix follows. That holds only when ``get_transformed_atoms``
    takes the identity path, which is the case for a cell that is already
    orthogonal and box-matching. For any other cell, ``get_transformed_atoms``
    calls ``orthogonalize_cell`` before ``randomize`` -- or any construction
    path below -- gets a chance to copy, so ``orthogonalize_cell`` copying its
    argument once, at entry (see ``abtem/atoms.py``), is what makes every entry
    point below safe for a non-orthogonal cell too; ``TestNonOrthogonalCell``
    exercises that case.
    """

    @staticmethod
    def _atoms():
        # x = 4.2 in a 4 A cell: outside the cell, so wrapping has work to do
        # and an in-place write is visible.
        return Atoms(
            "Si2", positions=[(0.2, 0.2, 0.5), (4.2, 2.0, 1.5)], cell=(4.0, 4.0, 4.0)
        )

    @staticmethod
    def _build(potential):
        from abtem.core import config

        with config.set({"fft": "numpy"}):
            return potential.build(lazy=False)

    def test_a_plain_atoms_potential_does_not_rewrite_its_own_atoms(self):
        atoms = self._atoms()
        potential = Potential(atoms, gpts=(32, 32), slice_thickness=1.0)
        stored = potential.frozen_phonons.atoms
        before = stored.positions.copy()
        self._build(potential)
        assert np.array_equal(stored.positions, before)

    def test_a_list_of_atoms_does_not_rewrite_the_callers_objects(self):
        """`Potential([a])` keeps a reference, so the write reached the caller."""
        atoms = self._atoms()
        before = atoms.positions.copy()
        self._build(Potential([atoms], gpts=(32, 32), slice_thickness=1.0))
        assert np.array_equal(atoms.positions, before)

    def test_a_prebuilt_dummy_frozen_phonons_does_not_rewrite_the_callers_atoms(self):
        from abtem.inelastic.phonons import DummyFrozenPhonons

        atoms = self._atoms()
        before = atoms.positions.copy()
        self._build(
            Potential(
                DummyFrozenPhonons(atoms), gpts=(32, 32), slice_thickness=1.0
            )
        )
        assert np.array_equal(atoms.positions, before)

    def test_frozen_phonons_was_already_safe(self):
        """The oracle: FrozenPhonons.randomize copies, so this path never had
        the defect. Pinned so the fix cannot be 'simplified' by removing the
        copy there instead."""
        atoms = self._atoms()
        before = atoms.positions.copy()
        from abtem.inelastic.phonons import FrozenPhonons

        # sigmas must be NON-ZERO. With sigmas=0.0 randomize's displacement is
        # `positions += 0 * r`, so an in-place randomize leaves the positions
        # numerically identical and this test passes even with
        # FrozenPhonons.randomize's own copy deleted -- it would be pinning a
        # no-op. Verified: removing that copy is caught at 0.1 and missed at 0.0.
        phonons = FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=1)
        self._build(Potential(phonons, gpts=(32, 32), slice_thickness=1.0))
        assert np.array_equal(atoms.positions, before)

    def test_an_earlier_build_does_not_change_a_later_potentials_result(self):
        """The sharpest consequence: the write crosses *objects*.

        Not repeated builds of one potential -- `get_sliced_atoms` memoises
        `_sliced_atoms`, so a second build never re-enters `_prepare_atoms`,
        and `wrap_and_snap_atoms` is idempotent anyway. An earlier version of
        this test asserted that and therefore could not fail.

        What does fail is a second potential built from the SAME Atoms object:
        the first build wrapped it in place, so the second sees pre-wrapped
        atoms and slices them differently.
        """
        atoms = Atoms(
            "Si2", positions=[(0.2, 0.2, -0.5), (2.0, 2.0, 1.5)],
            cell=(4.0, 4.0, 4.0),
        )
        reference = Potential(
            [atoms.copy()], gpts=(32, 32), slice_thickness=1.0, periodic=False,
            projection="infinite",
        ).get_sliced_atoms()

        shared = Potential([atoms], gpts=(32, 32), slice_thickness=1.0)
        self._build(shared)  # wraps `atoms` in place on the unfixed code
        after = Potential(
            [atoms], gpts=(32, 32), slice_thickness=1.0, periodic=False,
            projection="infinite",
        ).get_sliced_atoms()

        counts_ref = [len(reference.get_atoms_in_slices(i)) for i in range(4)]
        counts_after = [len(after.get_atoms_in_slices(i)) for i in range(4)]
        assert counts_after == counts_ref, (
            f"an earlier build changed a later potential's slicing: "
            f"{counts_after} != {counts_ref}"
        )

    def test_the_wrapped_positions_still_reach_the_slicing(self):
        """Copying must not lose the wrap -- the potential still has to be
        built from wrapped atoms, only not by rewriting the caller's."""
        atoms = self._atoms()
        potential = Potential(atoms, gpts=(32, 32), slice_thickness=1.0)
        sliced = potential.get_sliced_atoms()
        xs = np.asarray(sliced.atoms.positions)[:, 0]
        assert np.all(xs < 4.0), f"an unwrapped x survived into the slicing: {xs}"


class TestNonOrthogonalCellDoesNotMutateItsAtoms:
    """`TestPotentialDoesNotMutateItsAtoms._atoms()` hardcodes an orthogonal,
    box-matching cell, so none of that class's tests reach
    `get_transformed_atoms`'s `orthogonalize_cell` branch -- only its
    `wrap_and_snap_atoms` call, which is a different mutation site with its
    own fix. A non-orthogonal cell takes the `orthogonalize_cell` branch
    instead, and that function mutated its argument in three places
    internally (`set_cell`/`wrap`, `translate`/`wrap` for a non-default
    origin, and `_snap_scaled_positions_to_cell_boundary` ahead of `cut()`
    in the repeat-and-cut path) before copying anywhere -- so every
    construction site that aliases the caller's `Atoms`, including
    `FrozenPhonons`, was reachable through it regardless of `randomize`
    copying, because `orthogonalize_cell` ran first and mutated in place.

    `orthogonalize_cell` now copies its argument once, at entry, rather than
    at one of the three mutating call sites, so no path through the function
    can still be missed.
    """

    @staticmethod
    def _atoms():
        # A sheared (non-orthogonal) cell with the second atom given in
        # fractional coordinates clearly outside [0, 1) along the sheared
        # lattice vector, so wrapping moves it by a whole lattice vector --
        # 3.46 A -- and an in-place write is unambiguous.
        cell = np.array([[4.0, 0.0, 0.0], [2.0, 3.4641, 0.0], [0.0, 0.0, 6.0]])
        scaled = np.array([[0.1, 0.1, 0.2], [1.3, -0.2, 0.5]])
        return Atoms("Si2", positions=scaled @ cell, cell=cell, pbc=True)

    @staticmethod
    def _build(potential):
        from abtem.core import config

        with config.set({"fft": "numpy"}):
            return potential.build(lazy=False)

    @pytest.mark.parametrize(
        "construction",
        ["list_of_atoms", "dummy_frozen_phonons", "frozen_phonons"],
    )
    def test_a_non_orthogonal_cell_does_not_rewrite_the_callers_atoms(
        self, construction
    ):
        from abtem.inelastic.phonons import DummyFrozenPhonons, FrozenPhonons

        atoms = self._atoms()
        before = atoms.positions.copy()

        if construction == "list_of_atoms":
            wrapped = [atoms]
        elif construction == "dummy_frozen_phonons":
            wrapped = DummyFrozenPhonons(atoms)
        else:
            wrapped = FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=1)

        self._build(Potential(wrapped, gpts=(32, 32), slice_thickness=1.0))
        assert np.array_equal(atoms.positions, before)

    @pytest.mark.parametrize(
        "construction",
        ["list_of_atoms", "dummy_frozen_phonons", "frozen_phonons"],
    )
    def test_sampling_auto_does_not_rewrite_the_callers_atoms_either(
        self, construction
    ):
        """`orthogonalize_cell` has a second call site: `Potential.__init__`
        itself, reached through `sampling="auto"` when the cell needs a
        transform (iam.py, `_require_cell_transform`). That call runs
        synchronously in the constructor, before `build()` is ever called --
        an independent path into the same aliasing bug, not merely the same
        bug reached twice through one call. Both call sites go through the
        same `orthogonalize_cell`, so the fix covers this one too, but
        nothing above pins it: every test in this class builds before
        checking, which only exercises the first call site.
        """
        from abtem.inelastic.phonons import DummyFrozenPhonons, FrozenPhonons

        atoms = self._atoms()
        before = atoms.positions.copy()

        if construction == "list_of_atoms":
            wrapped = [atoms]
        elif construction == "dummy_frozen_phonons":
            wrapped = DummyFrozenPhonons(atoms)
        else:
            wrapped = FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=1)

        # No .build() call: sampling="auto" must do its own damage, if any,
        # inside __init__ alone.
        Potential(wrapped, sampling="auto", slice_thickness=1.0)
        assert np.array_equal(atoms.positions, before)


class TestSliceIndexedAtomsWrapping:
    """Atoms outside the cell were binned without being wrapped.

    ``Potential._prepare_atoms`` wrapped, but the other two construction sites
    -- explicit core-loss ``sites`` and ``CrystalPotential``'s tiled atoms --
    hand ``SliceIndexedAtoms`` raw atoms. ``np.digitize`` returns 0 for any z
    below the first bin edge, including arbitrarily negative z, so such an atom
    was assigned to slice 0 whatever its true wrapped depth; one above the last
    edge was discarded by ``label_to_index``. Both silent.

    Parametrised over ``pbc``: the first attempt at this fix used
    ``Atoms.wrap``, which is a no-op along non-periodic axes, so it left the
    bug in place for ASE's default ``pbc=False`` and for every
    ``build.*(vacuum=...)`` slab -- and the boundary snap then moved those
    atoms to zero instead of wrapping them.
    """

    DZ = 2.0
    N_SLICES = 4

    def _atoms(self, pbc=True):
        import ase
        import numpy as np

        z = [1.0, 3.0, 5.0, 7.0, -0.5, 8.5]  # last two outside the cell
        return ase.Atoms(
            "B" * len(z),
            positions=[[1.0, 1.0, zz] for zz in z],
            cell=np.diag([4.0, 4.0, self.DZ * self.N_SLICES]),
            pbc=pbc,
        )

    def _expected(self, atoms):
        height = self.DZ * self.N_SLICES
        counts = [0] * self.N_SLICES
        for z in atoms.positions[:, 2]:
            counts[int((z % height) // self.DZ)] += 1
        return counts

    @staticmethod
    def _per_slice(sliced):
        return [
            len(sliced.get_atoms_in_slices(i, atomic_number=5))
            for i in range(sliced.num_slices)
        ]

    @pytest.mark.parametrize(
        "pbc", [True, (True, True, False), False], ids=["pbc", "slab", "nopbc"]
    )
    def test_out_of_cell_atoms_land_in_their_wrapped_slice(self, pbc):
        from abtem.slicing import SliceIndexedAtoms

        atoms = self._atoms(pbc)
        sliced = SliceIndexedAtoms(atoms, slice_thickness=self.DZ)
        assert self._per_slice(sliced) == self._expected(atoms)

    @pytest.mark.parametrize(
        "pbc", [True, (True, True, False), False], ids=["pbc", "slab", "nopbc"]
    )
    def test_explicit_sites_agree_with_the_potentials_own_atoms(self, pbc):
        from abtem.inelastic.core_loss import _extract_scattering_sites

        atoms = self._atoms(pbc)
        potential = Potential(atoms, gpts=(32, 32), slice_thickness=self.DZ)
        from_potential = self._per_slice(_extract_scattering_sites(potential, None))
        from_caller = self._per_slice(_extract_scattering_sites(potential, atoms))
        assert from_caller == from_potential == self._expected(atoms)

    @pytest.mark.parametrize("pbc", [True, False], ids=["pbc", "nopbc"])
    def test_in_plane_site_positions_are_wrapped_not_zeroed(self, pbc):
        """The snap must only catch values a hair below the boundary.

        Applied to an un-wrapped position it teleports the site to the cell
        origin -- a real change of ionisation site, which ``dev`` did not make.
        """
        import ase
        import numpy as np

        from abtem.inelastic.core_loss import _extract_scattering_sites

        atoms = ase.Atoms(
            "B4",
            positions=[
                [1.0, 1.0, 1.0],
                [4.3, 1.0, 1.0],
                [2.0, 1.0, 3.0],
                [2.0, 1.0, 5.0],
            ],
            cell=np.diag([4.0, 4.0, 8.0]),
            pbc=pbc,
        )
        potential = Potential(atoms, gpts=(32, 32), slice_thickness=2.0)
        sites = _extract_scattering_sites(potential, atoms)
        # 4.3 in a 4 A cell is 0.3, not 0.0.
        assert np.allclose(sites.atoms.positions[:, 0], [1.0, 0.3, 2.0, 2.0])

    def test_the_callers_atoms_are_not_modified(self):
        import numpy as np

        from abtem.slicing import SliceIndexedAtoms

        atoms = self._atoms()
        before = atoms.positions.copy()
        SliceIndexedAtoms(atoms, slice_thickness=self.DZ)
        assert np.array_equal(atoms.positions, before)

    @pytest.mark.parametrize(
        "pbc", [True, (True, True, False), False], ids=["pbc", "slab", "nopbc"]
    )
    def test_wrapping_is_idempotent(self, pbc):
        import numpy as np

        from abtem.slicing import SliceIndexedAtoms

        atoms = self._atoms(pbc)
        once = SliceIndexedAtoms(atoms, slice_thickness=self.DZ)
        twice = SliceIndexedAtoms(once.atoms, slice_thickness=self.DZ)
        # Bitwise, not approximately: a second pass must be a no-op.
        assert np.array_equal(once.atoms.positions, twice.atoms.positions)
        assert self._per_slice(once) == self._per_slice(twice)

    def test_crystal_potential_slices_match_the_tiled_unit(self):
        """A head count passes even when the atoms are in the wrong slices."""
        import ase
        import numpy as np

        z = [1.0, 3.0, -0.5, 4.5]
        unit = ase.Atoms(
            "B" * len(z),
            positions=[[1.0, 1.0, zz] for zz in z],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        reps = (1, 1, 2)
        unit_potential = Potential(unit, gpts=(32, 32), slice_thickness=2.0)
        crystal = CrystalPotential(unit_potential, repetitions=reps)

        got = self._per_slice(crystal.get_sliced_atoms())
        per_unit = self._per_slice(unit_potential.get_sliced_atoms())
        assert got == per_unit * reps[2]
        assert sum(got) == len(z) * reps[2]

    @pytest.mark.parametrize("periodic", [True, False])
    def test_wrapping_follows_the_potentials_periodicity(self, periodic):
        """``Potential(periodic=False)`` deliberately never wraps.

        Its atoms are cut from a larger repeated potential and randomised
        *after* padding, so an edge atom displaced just outside the cell
        belongs at the face it left, not the opposite one. Wrapping it
        unconditionally moved it the full height of the box -- the same depth
        corruption the wrap exists to prevent, for the other path.
        """
        import ase

        import numpy as np

        from abtem.inelastic.core_loss import _extract_scattering_sites

        # Sits just inside the entrance face, so the randomize that runs after
        # padding on the non-periodic path pushes it out.
        atoms = ase.Atoms(
            "B4",
            positions=[
                [1.0, 1.0, 0.05],
                [2.0, 2.0, 0.05],
                [3.0, 3.0, 2.0],
                [1.0, 3.0, 3.95],
            ],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        from abtem.inelastic.phonons import FrozenPhonons

        phonons = FrozenPhonons(atoms, num_configs=1, sigmas=0.25, seed=1)
        potential = Potential(
            phonons, gpts=(32, 32), slice_thickness=1.0, periodic=periodic
        )
        sliced = potential.get_sliced_atoms()
        z = sliced.atoms.positions[:, 2]

        if periodic:
            assert np.all((z >= 0.0) & (z < 4.0))
        else:
            # Unwrapped, exactly as on a potential built without this change.
            assert z.min() < 0.0 or z.max() >= 4.0

        # Explicitly passed sites must follow the same convention, so that
        # sites=<Atoms> and sites=None never disagree.
        from_potential = self._per_slice(_extract_scattering_sites(potential, None))
        from_caller = self._per_slice(
            _extract_scattering_sites(potential, sliced.atoms)
        )
        assert from_caller == from_potential

    def test_out_of_range_guard_does_not_fire_when_wrapping_is_off(self):
        """Without a wrap, an atom outside the cell is expected, not an error."""
        import ase

        import numpy as np

        from abtem.slicing import SliceIndexedAtoms

        atoms = ase.Atoms(
            "B2",
            positions=[[1.0, 1.0, 2.0], [1.0, 1.0, 4.36]],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        sliced = SliceIndexedAtoms(atoms, slice_thickness=1.0, wrap=False)
        # Dropped silently, as before this change.
        assert sum(self._per_slice(sliced)) == 1

    def test_non_orthogonal_cell_raises_before_any_wrapping(self):
        import ase

        from abtem.slicing import SliceIndexedAtoms

        atoms = ase.Atoms(
            "B", positions=[[1.0, 1.0, 1.0]], cell=[[4, 0, 0], [1, 4, 0], [0, 0, 4]],
            pbc=True,
        )
        with pytest.raises(RuntimeError, match="orthogonal"):
            SliceIndexedAtoms(atoms, slice_thickness=1.0)
