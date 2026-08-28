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


def test_skew_potential_reduces_to_orthogonal():
    """A non_orthogonal=True potential on an orthogonal crystal must match the
    standard orthogonal potential."""
    import numpy as np
    from ase import Atoms

    atoms = Atoms(
        "C2", cell=(4.0, 5.0, 6.0), pbc=True,
        positions=[(1.0, 1.0, 3.0), (3.0, 4.0, 3.0)],
    )
    ortho = Potential(atoms, gpts=(80, 100), slice_thickness=6.0).build(lazy=False).compute()
    skew = Potential(
        atoms, gpts=(80, 100), slice_thickness=6.0, non_orthogonal=True
    ).build(lazy=False).compute()
    assert skew.grid.is_orthogonal  # a diagonal cell is still orthogonal
    assert np.allclose(ortho.array, skew.array, atol=1e-4)


def test_skew_potential_non_orthogonal_cell():
    """A genuinely non-orthogonal (hexagonal) cell builds a skewed-grid potential."""
    import numpy as np
    from ase import Atoms

    a, c = 2.46, 4.0
    b = a / (2 * np.sqrt(3))
    cell = np.array(
        [[a, 0, 0], [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0], [0, 0, c]]
    )
    atoms = Atoms("C2", cell=cell, pbc=True, positions=[(0, 0, c / 2), (a / 2, b, c / 2)])

    pot = Potential(atoms, gpts=(60, 60), slice_thickness=c, non_orthogonal=True)
    assert not pot.grid.is_orthogonal
    assert pot.cell is not None
    parr = pot.build(lazy=False).compute()
    assert parr.array.shape == (1, 60, 60)
    assert np.all(np.isfinite(parr.array))


def test_non_orthogonal_requires_ab_in_plane():
    """A tilted c-axis (with a, b in the xy-plane) is supported; an a- or b-axis with a
    z-component is rejected."""
    import numpy as np
    from ase import Atoms

    # a, b in-plane (b skewed), c tilted out of z -> supported (full triclinic)
    cell = np.array([[4.0, 0, 0], [1.0, 4.0, 0], [0.5, 0.0, 6.0]])
    atoms = Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)])
    pot = Potential(atoms, gpts=(40, 40), slice_thickness=6.0, non_orthogonal=True)
    assert not pot.grid.is_orthogonal
    assert np.all(np.isfinite(pot.build(lazy=False).compute().array))

    # a-axis has a z-component -> not sliceable along the beam -> rejected
    bad = np.array([[4.0, 0, 0.5], [0.0, 4.0, 0], [0.0, 0.0, 6.0]])
    atoms_bad = Atoms("C", cell=bad, pbc=True, positions=[(0, 0, 0)])
    with pytest.raises(NotImplementedError):
        Potential(atoms_bad, gpts=(40, 40), slice_thickness=6.0, non_orthogonal=True)


def test_potential_auto_detects_non_orthogonal():
    """A non-orthogonal (z-separable) cell builds a skew potential with no flag."""
    import numpy as np
    from ase import Atoms

    a, c = 3.0, 2.0
    cell = np.array(
        [[a, 0, 0], [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0], [0, 0, c]]
    )
    atoms = Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)])

    auto = Potential(atoms, gpts=(60, 60), slice_thickness=c)
    assert not auto.grid.is_orthogonal  # auto-detected
    # explicit opt-out restores the legacy orthogonalising behaviour
    forced_ortho = Potential(atoms, gpts=(60, 60), slice_thickness=c, non_orthogonal=False)
    assert forced_ortho.grid.is_orthogonal

    # an orthogonal cell is never switched to skew
    ortho_atoms = Atoms("C", cell=(4, 5, 6), pbc=True, positions=[(0, 0, 0)])
    assert Potential(ortho_atoms, gpts=(40, 40), slice_thickness=6).grid.is_orthogonal


def test_tilted_c_axis_matches_orthogonal_supercell():
    """A crystal with a tilted c-axis (a, b orthogonal in-plane) is handled by placing
    the atoms at their true positions and propagating straight -- no tilt ramp -- so it
    reproduces the orthogonalised supercell of the same crystal exactly (this is ordinary
    multislice on the true atomic positions, hence exact at any tilt angle)."""
    import numpy as np
    from ase import Atoms

    import abtem

    L, cz = 4.0, 4.0
    basis = [(0.1, 0.15, 0.05), (0.6, 0.55, 0.45)]
    # c tilted in x by L/2 (a 26.6 deg tilt); wraps to an orthogonal cell after 2 layers
    prim = Atoms(
        "SiO", cell=[[L, 0, 0], [0, L, 0], [L / 2, 0, cz]], pbc=True,
        scaled_positions=basis,
    )

    tilted = prim * (1, 1, 2)
    pot_tilted = abtem.Potential(tilted, gpts=(128, 128), slice_thickness=0.5)
    assert pot_tilted.grid.is_orthogonal  # a, b orthogonal -> plain orthogonal grid
    assert pot_tilted.cell is None  # no spurious skew cell carried in the metadata
    w_tilted = abtem.PlaneWave(energy=100e3).multislice(pot_tilted).compute().array

    ortho = prim * (1, 1, 2)
    ortho.set_cell([[L, 0, 0], [0, L, 0], [0, 0, 2 * cz]])
    ortho.wrap()
    w_ortho = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(ortho, gpts=(128, 128), slice_thickness=0.5)
    ).compute().array

    rel = np.max(np.abs(np.asarray(w_tilted) - np.asarray(w_ortho))) / np.max(
        np.abs(np.asarray(w_ortho))
    )
    # identical up to float32 rounding (the two paths build the same potential); in
    # float64 this drops to ~1e-15
    assert rel < 1e-5


def test_finite_projection_supports_skew_grid():
    """``projection='finite'`` integrates the radial potential per atom on a
    metric-aware (Cartesian) pixel grid for non-orthogonal cells. Two checks:
    (i) the build succeeds and conserves atoms (the in-plane-integrated potential
    matches the infinite-projection result up to the projection-method difference,
    same level as on an orthogonal grid); (ii) hexagonal symmetry: the structure
    factors of symmetry-equivalent reflections agree to numerical precision."""
    import numpy as np
    from ase import Atoms

    import abtem

    a = 2.46  # graphene-like in-plane spacing
    hexcell = [
        [a, 0, 0],
        [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0],
        [0, 0, 3.35],
    ]
    atoms = Atoms(
        "C2", cell=hexcell, pbc=True,
        scaled_positions=[(0, 0, 0), (1 / 3, 1 / 3, 0.5)],
    )

    fin = np.asarray(
        abtem.Potential(atoms, gpts=(80, 80), slice_thickness=3.35,
                        projection="finite").build(lazy=False).compute().array
    )[0]
    inf = np.asarray(
        abtem.Potential(atoms, gpts=(80, 80), slice_thickness=3.35,
                        projection="infinite").build(lazy=False).compute().array
    )[0]

    assert np.all(np.isfinite(fin))

    # atom conservation: the integrated potential agrees with the infinite-projection
    # result (which uses the analytic structure factor) -- to the same ~1e-3 level seen
    # on an orthogonal grid (the residual is the projection-method difference, not skew)
    sv = np.stack([np.array(hexcell)[0, :2] / 80, np.array(hexcell)[1, :2] / 80])
    pix_area = abs(np.linalg.det(sv))
    assert abs(fin.sum() - inf.sum()) / inf.sum() < 5e-3

    # hexagonal symmetry of the structure factor: F(1, 0) == F(0, 1) == F(-1, -1)
    F = np.fft.fft2(fin)
    f10 = np.abs(F[1, 0])
    f01 = np.abs(F[0, 1])
    f_11 = np.abs(F[-1, -1])
    # hexagonal symmetry holds to float32 rounding (~1e-7); float64 would give ~1e-15
    assert abs(f10 - f01) / f10 < 1e-5
    assert abs(f10 - f_11) / f10 < 1e-5


def test_finite_projection_tilted_c_axis():
    """``projection='finite'`` on a crystal with a tilted c-axis runs end-to-end and
    matches an orthogonalised supercell of the same crystal."""
    import numpy as np
    from ase import Atoms

    import abtem

    L, cz = 4.0, 4.0
    basis = [(0.1, 0.15, 0.05), (0.6, 0.55, 0.45)]
    # c tilted in x by L/2 (26.6 deg); wraps to an orthogonal cell after 2 layers
    prim = Atoms(
        "SiO", cell=[[L, 0, 0], [0, L, 0], [L / 2, 0, cz]], pbc=True,
        scaled_positions=basis,
    )
    tilted = prim * (1, 1, 2)
    w_tilted = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(tilted, gpts=(128, 128), slice_thickness=0.25,
                        projection="finite")
    ).compute().array

    ortho = prim * (1, 1, 2)
    ortho.set_cell([[L, 0, 0], [0, L, 0], [0, 0, 2 * cz]])
    ortho.wrap()
    w_ortho = abtem.PlaneWave(energy=100e3).multislice(
        abtem.Potential(ortho, gpts=(128, 128), slice_thickness=0.25,
                        projection="finite")
    ).compute().array

    rel = np.max(np.abs(np.asarray(w_tilted) - np.asarray(w_ortho))) / np.max(
        np.abs(np.asarray(w_ortho))
    )
    # the two paths build the same potential up to float32 rounding
    assert rel < 1e-5


def test_all_modes_agree_on_skew_grid_bragg_intensities():
    """Cross-algorithm consistency check: on a non-orthogonal (hex) cell, the four
    multislice/dynamical engines must agree on the plane-wave Bragg intensities to
    their respective convergence levels: Bloch (reference), Fourier multislice,
    real-space finite-difference multislice, and the (0, 0) plane-wave column of the
    PRISM S-matrix.

    The check sweeps *every* Bloch reflection with l = 0 and intensity above a
    threshold (~30 reflections), not a hand-picked few. With c perpendicular to ab,
    the Bloch reciprocal basis vectors b_i_3D have zero z-components, so
    Bloch hkl = (h, k, 0) corresponds one-to-one to the multislice FFT pixel
    (h, k) -- a clean per-reflection comparison."""
    import numpy as np
    from ase import Atoms

    import abtem
    from abtem.multislice import FourierMultislice, RealSpaceMultislice
    from abtem.parametrizations import LobatoParametrization

    a, c = 2.46, 3.35
    hexcell = [
        [a, 0, 0],
        [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0],
        [0, 0, c],
    ]
    hexC = Atoms(
        "C2", cell=hexcell, pbc=True,
        scaled_positions=[(0, 0, 0), (1 / 3, 1 / 3, 0.5)],
    )
    energy, nc = 100e3, 12

    # Bloch reference (matched parametrization, no DW)
    sf = abtem.StructureFactor(
        hexC, g_max=12.0, parametrization="lobato", centering="auto",
        thermal_sigma=0.0,
    )
    bw = abtem.BlochWaves(sf, energy=energy, sg_max=0.6)
    hkl = np.asarray(bw.hkl)
    Ib = np.asarray(
        bw.calculate_diffraction_patterns(nc * c, lazy=False).array
    ).ravel()

    # multislice potential matched to the Bloch structure factor
    param = LobatoParametrization(sigmas={"C": 0.0})
    atoms = hexC * (1, 1, nc)
    pot = abtem.Potential(
        atoms, sampling=0.030, slice_thickness=0.1,
        projection="finite", parametrization=param,
    )
    N0, N1 = pot.gpts
    assert not pot.grid.is_orthogonal

    def bragg(w):
        I = np.abs(np.fft.fft2(np.asarray(w))) ** 2
        return I / I.sum()

    wF = abtem.PlaneWave(energy=energy).multislice(
        pot, algorithm=FourierMultislice()
    ).compute().array
    IF = bragg(wF)
    wR = abtem.PlaneWave(energy=energy).multislice(
        pot, algorithm=RealSpaceMultislice()
    ).compute().array
    IR = bragg(wR)
    sm = abtem.SMatrix(
        potential=pot, energy=energy, semiangle_cutoff=1.0,
        interpolation=1, downsample=False,
    ).build(lazy=False)
    wv = np.asarray(sm.wave_vectors)
    idx0 = int(np.argmin(np.linalg.norm(wv, axis=1)))
    wP = np.asarray(sm.array)[idx0] * (np.prod(pot.gpts) / np.prod(sm.interpolation))
    IP = bragg(wP)

    # comprehensive sweep: take every Bloch reflection with l = 0 and intensity
    # above a threshold (so we ignore noise-level reflections where any relative
    # error is dominated by discretisation noise). For each, look up the MS
    # intensity at FFT pixel (h, k).
    threshold = 1e-4
    in_plane = (hkl[:, 2] == 0) & (Ib > threshold) & (np.abs(hkl[:, 0]) < N0 // 2) & (
        np.abs(hkl[:, 1]) < N1 // 2
    )
    sel_hkl = hkl[in_plane]
    sel_I = Ib[in_plane]
    assert len(sel_hkl) >= 15, (
        f"want at least 15 in-plane reflections for a comprehensive sweep; "
        f"got {len(sel_hkl)}"
    )

    # Bloch vs each multislice variant: per-reflection rel errors. Bounds reflect
    # the fact that weak / high-g reflections naturally pick up more
    # discretisation error than strong / low-g ones (relative error inflates as
    # the absolute intensity shrinks).
    for I, name in [
        (IF, "Fourier MS"),
        (IR, "Real-space FD"),
        (IP, "PRISM S(0,0)"),
    ]:
        rels = []
        for h, k, _ in sel_hkl:
            ms_val = float(I[int(h) % N0, int(k) % N1])
            bloch_val = float(sel_I[np.all(sel_hkl == [h, k, 0], axis=1)][0])
            rels.append(abs(ms_val - bloch_val) / bloch_val)
        rels = np.array(rels)
        worst_idx = int(np.argmax(rels))
        worst_hkl = sel_hkl[worst_idx]
        # every reflection agrees to a few percent; the mean and median are much
        # smaller (only the high-g tail pushes the worst case up)
        assert rels.max() < 5e-2, (
            f"{name} disagrees with Bloch on {tuple(worst_hkl)}: "
            f"rel={rels.max():.2e} (worst of {len(rels)} reflections)"
        )
        assert rels.mean() < 2e-2, (
            f"{name} mean rel error {rels.mean():.2e} on {len(rels)} reflections"
        )

    # inter-MS-variant consistency on the full FFT pattern (every pixel with
    # intensity above the physical-reflection threshold; very weak pixels near
    # the float32 noise floor are excluded). Fourier MS == PRISM(0,0)-column to
    # float precision (same path); real-space FD agrees within its FD
    # discretisation.
    inter_threshold = 3e-4
    mask = IF > inter_threshold
    rel_FP = np.abs(IF[mask] - IP[mask]) / IF[mask]
    rel_FR = np.abs(IF[mask] - IR[mask]) / IF[mask]
    assert mask.sum() >= 15, f"want at least 15 above-threshold pixels; got {mask.sum()}"
    # bit-identical in float64 (~1e-15); float32 precision floor sets the bound
    assert rel_FP.max() < 1e-3, (
        f"Fourier MS vs PRISM disagrees on {int(mask.sum())} pixels: "
        f"max rel={rel_FP.max():.2e}"
    )
    # real-space FD discretisation level on physical reflections
    assert rel_FR.max() < 2e-2, (
        f"Real-space FD vs Fourier MS disagrees on {int(mask.sum())} pixels: "
        f"max rel={rel_FR.max():.2e}, mean={rel_FR.mean():.2e}"
    )
    assert rel_FR.mean() < 1e-2, (
        f"Real-space FD vs Fourier MS mean rel error {rel_FR.mean():.2e}"
    )

    # hexagonal-grid reflection symmetry preserved across modes: (h, k) <-> (k, h)
    # (mirror across the a/b bisector). Inversion (Friedel) symmetry is implicit
    # in the real potential.
    for I_mode in (IF, IR, IP):
        for h, k in [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]:
            I1 = float(I_mode[h % N0, k % N1])
            I2 = float(I_mode[k % N0, h % N1])
            if I1 < threshold:
                continue
            assert abs(I1 - I2) / I1 < 5e-4, (
                f"hex mirror (h,k)<->(k,h) breaks at ({h},{k}): {I1:.5f} vs {I2:.5f}"
            )


def test_all_modes_agree_on_fully_triclinic_cell():
    """A fully triclinic cell -- all three lattice vectors mutually non-orthogonal:
    a, b non-orthogonal in the xy-plane AND c with non-zero in-plane components in
    BOTH x and y -- runs end-to-end through every engine and the three multislice
    variants agree. This exercises every skew code path at once: the metric-aware
    propagator/integrator on the in-plane (a, b) parallelogram, the tilted-c atom
    wrap, the finite projection on a non-orthogonal grid, and the PRISM S-matrix
    construction on a fully triclinic supercell.

    The Bloch engine is NOT compared here -- and not because it's wrong, but because
    for a fully triclinic crystal viewed along Cartesian +z the 3D reciprocal basis
    vectors ``b_i`` have non-zero z-components (b1_3D has z = -a3_xy/V etc.), so
    Bloch ``hkl = (h, k, 0)`` is in general off the Ewald sphere and does NOT
    correspond to the multislice in-plane pixel ``(h, k)`` (which is at g_xy =
    h*b1_2D + k*b2_2D, the *xy parts* of b1_3D, b2_3D). For each in-plane g_xy the
    multislice intensity is physically a sum over the entire tower ``(h, k, l)`` of
    3D reflections projecting onto that g_xy, not a single Bloch reflection. A fair
    Bloch comparison would require summing over l. The in-plane-skew and orthogonal
    tests (where c is perpendicular to ab so b_i = (b_i_2D, 0) and the indexing is
    one-to-one) cover the Bloch cross-check."""
    import numpy as np
    from ase import Atoms

    import abtem
    from abtem.multislice import FourierMultislice, RealSpaceMultislice
    from abtem.parametrizations import LobatoParametrization

    # hex a, b at 60 deg; c tilted with non-zero cx AND cy -- mutually non-orthogonal
    a, cz = 2.5, 3.35
    cell = [
        [a, 0.0, 0.0],
        [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0.0],
        [0.7, 0.4, cz],
    ]
    # confirm no pair is orthogonal
    c0, c1, c2 = (np.asarray(v) for v in cell)
    assert abs(c0 @ c1) > 1e-3 and abs(c0 @ c2) > 1e-3 and abs(c1 @ c2) > 1e-3

    triC = Atoms(
        "C2", cell=cell, pbc=True,
        scaled_positions=[(0, 0, 0), (1 / 3, 1 / 3, 0.5)],
    )

    energy, nc = 100e3, 8

    param = LobatoParametrization(sigmas={"C": 0.0})
    atoms = triC * (1, 1, nc)
    pot = abtem.Potential(
        atoms, sampling=0.030, slice_thickness=0.1,
        projection="finite", parametrization=param,
    )
    N0, N1 = pot.gpts
    assert not pot.grid.is_orthogonal, "in-plane grid should be skewed"

    def bragg(w):
        I = np.abs(np.fft.fft2(np.asarray(w))) ** 2
        return I / I.sum()

    wF = abtem.PlaneWave(energy=energy).multislice(
        pot, algorithm=FourierMultislice()
    ).compute().array
    IF = bragg(wF)

    # the calculation should actually scatter (no silently-dropped atoms)
    assert IF[0, 0] < 0.999, "no scattering -- atoms may have been silently dropped"
    assert IF[0, 0] > 0.5, "unexpectedly thick / strong scattering"

    wR = abtem.PlaneWave(energy=energy).multislice(
        pot, algorithm=RealSpaceMultislice()
    ).compute().array
    IR = bragg(wR)

    sm = abtem.SMatrix(
        potential=pot, energy=energy, semiangle_cutoff=1.0,
        interpolation=1, downsample=False,
    ).build(lazy=False)
    wv = np.asarray(sm.wave_vectors)
    idx0 = int(np.argmin(np.linalg.norm(wv, axis=1)))
    wP = np.asarray(sm.array)[idx0] * (np.prod(pot.gpts) / np.prod(sm.interpolation))
    IP = bragg(wP)

    # comprehensive sweep: compare the full FFT pattern across every pixel with
    # intensity above a threshold (not a hand-picked few). This stresses every
    # spatial frequency the multislice produces.
    threshold = 1e-5
    mask = IF > threshold
    n = int(mask.sum())
    assert n >= 50, f"want at least 50 above-threshold pixels; got {n}"

    # Fourier MS and PRISM (0, 0) column take the same multislice path -> agreement
    # at the float precision floor (max over all above-threshold pixels).
    # 5e-4 (not tighter): the weakest above-threshold pixels sit at the float32
    # rounding floor, and run-to-run variation (threaded potential-build
    # summation order, FFTW plan selection) moves them by ~1e-4 relative.
    rel_FP = np.abs(IF[mask] - IP[mask]) / IF[mask]
    assert rel_FP.max() < 5e-4, (
        f"Fourier MS vs PRISM disagrees on {n} pixels: max rel={rel_FP.max():.2e}, "
        f"mean rel={rel_FP.mean():.2e}"
    )

    # real-space FD agrees with Fourier MS within its FD discretisation -- the
    # tilted-c potential adds an in-plane drift between slices that the metric
    # Laplacian must capture
    rel_FR = np.abs(IF[mask] - IR[mask]) / IF[mask]
    assert rel_FR.max() < 5e-2, (
        f"Real-space FD vs Fourier MS disagrees on {n} pixels: max rel={rel_FR.max():.2e}, "
        f"mean rel={rel_FR.mean():.2e}"
    )
    assert rel_FR.mean() < 1e-2, (
        f"Real-space FD vs Fourier MS mean rel error {rel_FR.mean():.2e} too large"
    )


def test_skew_cell_metadata_survives_reconstruction():
    """The non-orthogonal cell is stored in metadata and survives reconstruction."""
    import numpy as np
    from ase import Atoms

    import abtem

    a, c = 3.0, 2.0
    cell = np.array(
        [[a, 0, 0], [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0], [0, 0, c]]
    )
    atoms = Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)]) * (1, 1, 3)
    pot = abtem.Potential(atoms, gpts=(60, 60), slice_thickness=c)
    exit_waves = abtem.PlaneWave(energy=100e3).multislice(pot).compute()

    assert exit_waves.cell is not None
    assert "cell" in exit_waves.metadata
    rebuilt = abtem.Waves.from_array_and_metadata(
        exit_waves.array,
        exit_waves.ensemble_axes_metadata + exit_waves.base_axes_metadata,
        exit_waves.metadata,
    )
    assert rebuilt.cell is not None
    assert np.allclose(rebuilt.cell, exit_waves.cell)


def test_skew_supercell_potential_is_periodic():
    """A skew supercell must keep all atoms (no boundary atom dropped by wrap) and the
    projected potential must be primitive-periodic."""
    import numpy as np
    from ase import Atoms

    a, cz, rep = 3.0, 2.0, 6
    cell = np.array(
        [[a, 0, 0], [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0], [0, 0, cz]]
    )
    atoms = Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)]) * (rep, rep, 1)
    pot = Potential(atoms, gpts=(240, 240), slice_thickness=cz)

    assert len(pot.get_sliced_atoms().atoms) == rep * rep  # no atom dropped
    V = np.asarray(pot.build(lazy=False).compute().array[0])
    n = 240 // rep  # one primitive period in pixels
    assert np.allclose(V, np.roll(V, n, axis=0), atol=1e-6 * np.abs(V).max())
    assert np.allclose(V, np.roll(V, n, axis=1), atol=1e-6 * np.abs(V).max())


def test_frozen_phonons_skew():
    """Frozen phonons work on a non-orthogonal cell and the configurations differ."""
    import numpy as np
    from ase import Atoms

    import abtem

    a, cz = 3.0, 2.0
    cell = np.array(
        [[a, 0, 0], [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0], [0, 0, cz]]
    )
    atoms = Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)]) * (3, 3, 4)
    fp = abtem.FrozenPhonons(atoms, num_configs=3, sigmas=0.1, seed=0)
    pot = abtem.Potential(fp, gpts=(96, 96), slice_thickness=cz)
    assert not pot.grid.is_orthogonal
    V = np.asarray(pot.build(lazy=False).compute().array)
    assert V.shape[0] == 3 and np.all(np.isfinite(V))
    assert not np.allclose(V[0], V[1])  # thermal disorder breaks periodicity

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


# --- interpolate_line tests ---


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [True, False])
def test_potential_interpolate_line_projected_matches_project(device, lazy):
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (2, 2, 5)
    potential = Potential(
        atoms, slice_thickness=1.0, gpts=(32, 32), device=device
    ).build(lazy=lazy)

    direct = potential.interpolate_line(start=(0.0, 0.0), end=(3.0, 4.0), gpts=25)
    via_project = potential.project().interpolate_line(
        start=(0.0, 0.0), end=(3.0, 4.0), gpts=25
    )

    assert direct.shape == via_project.shape
    np.testing.assert_allclose(
        np.asarray(direct.array.compute() if lazy else direct.array),
        np.asarray(via_project.array.compute() if lazy else via_project.array),
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [True, False])
def test_potential_interpolate_line_3d_lazy_matches_eager(device, lazy):
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (2, 2, 5)
    potential_lazy = Potential(
        atoms, slice_thickness=0.5, gpts=(32, 32), device=device
    ).build(lazy=True)
    potential_eager = Potential(
        atoms, slice_thickness=0.5, gpts=(32, 32), device=device
    ).build(lazy=False)

    kwargs = dict(start=(1.0, 2.0, 0.5), end=(3.0, 3.5, 8.0), gpts=40, projected=False)
    profile_lazy = potential_lazy.interpolate_line(**kwargs)
    profile_eager = potential_eager.interpolate_line(**kwargs)

    assert profile_lazy.shape == profile_eager.shape
    np.testing.assert_allclose(
        np.asarray(profile_lazy.array.compute()),
        np.asarray(profile_eager.array),
    )


def test_potential_interpolate_line_3d_frozen_phonon_ensemble_shape():
    import abtem

    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
    fp = abtem.FrozenPhonons(atoms, num_configs=3, sigmas=0.1, seed=1)
    potential_lazy = Potential(fp, sampling=0.2, slice_thickness=0.5).build(lazy=True)
    potential_eager = Potential(fp, sampling=0.2, slice_thickness=0.5).build(
        lazy=False
    )

    projected = potential_lazy.interpolate_line(
        start=(0.0, 0.0), end=(2.0, 2.0), gpts=10
    )
    volumetric = potential_lazy.interpolate_line(
        start=(0.0, 0.0, 0.0), end=(2.0, 2.0, 2.0), gpts=10, projected=False
    )

    assert projected.shape == (3, 10)
    assert volumetric.shape == (3, 10)
    np.testing.assert_allclose(
        projected.array.compute(),
        potential_eager.interpolate_line(
            start=(0.0, 0.0), end=(2.0, 2.0), gpts=10
        ).array,
    )
    np.testing.assert_allclose(
        volumetric.array.compute(),
        potential_eager.interpolate_line(
            start=(0.0, 0.0, 0.0), end=(2.0, 2.0, 2.0), gpts=10, projected=False
        ).array,
    )


def test_potential_interpolate_line_3d_rejects_width_and_margin(si_potential):
    pot = si_potential.build(lazy=False)
    with pytest.raises(NotImplementedError):
        pot.interpolate_line(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 1.0, 1.0),
            gpts=10,
            width=1.0,
            projected=False,
        )
    with pytest.raises(NotImplementedError):
        pot.interpolate_line(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 1.0, 1.0),
            gpts=10,
            margin=1.0,
            projected=False,
        )


def test_potential_interpolate_line_fractional(si_potential):
    pot = si_potential.build(lazy=False)

    projected = pot.interpolate_line(
        start=(0.0, 0.0), end=(1.0, 1.0), fractional=True, gpts=10
    )
    expected_end = (pot.extent[0], pot.extent[1])
    explicit = pot.interpolate_line(start=(0.0, 0.0), end=expected_end, gpts=10)
    np.testing.assert_allclose(projected.array, explicit.array)

    volumetric = pot.interpolate_line(
        start=(0.0, 0.0, 0.0),
        end=(1.0, 1.0, 1.0),
        fractional=True,
        gpts=10,
        projected=False,
    )
    explicit_3d = pot.interpolate_line(
        start=(0.0, 0.0, 0.0),
        end=(pot.extent[0], pot.extent[1], pot.thickness),
        gpts=10,
        projected=False,
    )
    np.testing.assert_allclose(volumetric.array, explicit_3d.array)


def test_potential_interpolate_line_at_position(si_potential):
    pot = si_potential.build(lazy=False)
    center = (pot.extent[0] / 2, pot.extent[1] / 2)

    projected = pot.interpolate_line_at_position(
        center=center, angle=30.0, extent=5.0, gpts=20
    )
    assert projected.shape == (20,)

    volumetric = pot.interpolate_line_at_position(
        center=center + (1.0,), angle=30.0, extent=5.0, gpts=20, projected=False
    )
    assert volumetric.shape == (20,)


def test_potential_interpolate_line_lazy_delegation(si_potential):
    profile_lazy = si_potential.interpolate_line(start=(0.0, 0.0), end=(2.0, 2.0), gpts=10)
    profile_built = si_potential.build(lazy=False).interpolate_line(
        start=(0.0, 0.0), end=(2.0, 2.0), gpts=10
    )
    assert profile_lazy.shape == profile_built.shape
    np.testing.assert_allclose(profile_lazy.array, profile_built.array)


@pytest.fixture
def hex_skew_potential():
    """A graphene-like hexagonal (non-orthogonal) 2-atom cell, with atom 1 at
    fractional in-plane position (1/3, 1/3) -- a convenient, exactly-known Cartesian
    location for verifying the Cartesian<->fractional pixel mapping on a skewed grid."""
    a, c = 2.46, 3.35
    cell = [
        [a, 0, 0],
        [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0],
        [0, 0, c],
    ]
    atoms = Atoms(
        "C2", cell=cell, pbc=True, scaled_positions=[(0, 0, 0), (1 / 3, 1 / 3, 0.5)]
    )
    potential = Potential(
        atoms, gpts=(64, 64), slice_thickness=c / 10, projection="infinite"
    )
    assert not potential.grid.is_orthogonal
    return potential, np.array(cell)[:2, :2], np.array([1 / 3, 1 / 3]) @ np.array(cell)[:2, :2]


def test_potential_interpolate_line_projected_locates_atom_on_skew_grid(
    hex_skew_potential,
):
    potential, _, atom1_xy = hex_skew_potential
    built = potential.build(lazy=False)

    eps = 0.5
    start = (atom1_xy[0] - eps, atom1_xy[1])
    end = (atom1_xy[0] + eps, atom1_xy[1])
    profile = built.interpolate_line(start=start, end=end, gpts=101, endpoint=True)

    assert np.argmax(profile.array) == 50


def test_potential_interpolate_line_3d_locates_atom_on_skew_grid(hex_skew_potential):
    potential, _, atom1_xy = hex_skew_potential
    built_eager = potential.build(lazy=False)
    built_lazy = potential.build(lazy=True)
    atom1_z = 0.5 * built_eager.thickness

    eps = 0.5
    start = (atom1_xy[0] - eps, atom1_xy[1], atom1_z)
    end = (atom1_xy[0] + eps, atom1_xy[1], atom1_z)

    profile_eager = built_eager.interpolate_line(
        start=start, end=end, gpts=101, endpoint=True, projected=False
    )
    profile_lazy = built_lazy.interpolate_line(
        start=start, end=end, gpts=101, endpoint=True, projected=False
    )

    assert np.argmax(profile_eager.array) == 50
    np.testing.assert_allclose(profile_eager.array, profile_lazy.array.compute())


def test_potential_interpolate_line_fractional_skew_matches_cartesian(
    hex_skew_potential,
):
    potential, _, atom1_xy = hex_skew_potential
    built = potential.build(lazy=False)
    atom1_z = 0.5 * built.thickness

    frac_profile = built.interpolate_line(
        start=(0.0, 0.0, 0.0),
        end=(1 / 3, 1 / 3, 0.5),
        gpts=50,
        endpoint=True,
        fractional=True,
        projected=False,
    )
    cartesian_profile = built.interpolate_line(
        start=(0.0, 0.0, 0.0),
        end=(atom1_xy[0], atom1_xy[1], atom1_z),
        gpts=50,
        endpoint=True,
        projected=False,
    )
    np.testing.assert_allclose(frac_profile.array, cartesian_profile.array)


def test_potential_interpolate_line_fractional_skew_matches_cartesian_projected(
    hex_skew_potential,
):
    """Same as test_potential_interpolate_line_fractional_skew_matches_cartesian, but
    for the projected=True (2D) path -- this goes through LineScan's own fractional
    handling rather than the potential's own effective-cell conversion, and must
    likewise use the true (skewed) cell rather than a plain per-axis extent scaling."""
    potential, _, atom1_xy = hex_skew_potential
    built = potential.build(lazy=False)

    frac_profile = built.interpolate_line(
        start=(0.0, 0.0),
        end=(1 / 3, 1 / 3),
        gpts=50,
        endpoint=True,
        fractional=True,
        projected=True,
    )
    cartesian_profile = built.interpolate_line(
        start=(0.0, 0.0),
        end=(atom1_xy[0], atom1_xy[1]),
        gpts=50,
        endpoint=True,
        projected=True,
    )
    np.testing.assert_allclose(frac_profile.array, cartesian_profile.array)
    np.testing.assert_allclose(frac_profile.metadata["end"], atom1_xy)


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
    potential = abtem.Potential(atoms, gpts=128, slice_thickness=2.0).build(lazy=False)

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
