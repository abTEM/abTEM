import os
import pickle
import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from ase.build import bulk, graphene, mx2
from hypothesis import given
from utils import (
    assert_array_matches_device,
    devices,
    float64_devices,
    gpu,
    ignore_strain_warning,
    si_cubic_atoms,
    si_diamond_atoms,
)

import abtem
from abtem.atoms import best_orthogonal_cell, cut_cell, orthogonalize_cell
from abtem.core.backend import asnumpy
from abtem.core.fft import next_fast_fft_size
from abtem.core.grid import disk_meshgrid, round_auto_derived_gpts
from abtem.integrals import (
    GaussianProjectionIntegrals,
    QuadratureProjectionIntegrals,
    _threaded_interpolate_radial_functions,
    interpolate_radial_functions,
)
from abtem.inelastic.phonons import FrozenPhonons
from abtem.magnetism.gpaw import GPAWMagneticField, GPAWVectorPotential
from abtem.magnetism.iam import MagneticField
from abtem.potentials.charge_density import ChargeDensityPotential
from abtem.potentials.iam import CrystalPotential, Potential, PotentialArray


def _build_with_numpy_fft(potential):
    from abtem.core import config

    with config.set({"fft": "numpy"}):
        return potential.build(lazy=False)


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
    """The built potential must obey the k = 0 sum rule,
    int V d^3r = sum_atoms F_Z(0), with F the parametrization's projected
    scattering factor (see test_potential_physics._f0 for the units).

    Infinite projection: exact up to float32 round-off and parameter storage
    (Lobato carbon's nearly cancelling terms alone move F(0) by 1.8e-5), so
    1e-4. Finite projection: truncation at the cutoff and the taper only remove
    potential (<= 0.2 % for Z <= 14), and the log-singular core pixel costs
    <= 1.8 % of the atom's slice at dx = 0.1 A, scaling as dx^2
    (test_potential_physics); at the coarsest grid drawn here, dx ~ 0.16 A,
    that is <= 4.6 %, so 8 %. The core-pixel error is a deficit for square
    pixels but can be an excess for anisotropic ones -- the core pixel is
    clamped to V(min(sampling) / 2), above its average over a dx != dy pixel
    -- so the bound is two-sided.
    """
    from ase.data import chemical_symbols

    from abtem.core import config
    from abtem.parametrizations import validate_parametrization

    potential = Potential(
        atoms,
        gpts=gpts,
        slice_thickness=slice_thickness,
        parametrization=parametrization,
        projection=projection,
    )
    array = potential.build(lazy=False).compute().array
    total = float(array.astype(np.float64).sum()) * np.prod(potential.sampling)

    parametrization = validate_parametrization(parametrization)
    with config.set({"precision": "float64"}):
        expected = sum(
            float(
                parametrization.projected_scattering_factor(chemical_symbols[Z])(
                    np.array([0.0])
                )[0]
            )
            for Z in atoms.numbers
        )
    deficit = 1 - total / expected
    if projection == "infinite":
        assert abs(deficit) < 1e-4, deficit
    else:
        assert abs(deficit) < 8e-2, deficit


@given(
    atoms=abtem_st.atoms(max_atomic_number=14),
    gpts=abtem_st.gpts(),
    slice_thickness=st.floats(min_value=1, max_value=2.0),
)
@pytest.mark.parametrize("lazy", [True, False])
@devices
def test_build_device_lazy(atoms, gpts, slice_thickness, lazy, device):
    """Laziness and device must not change the values: compare against an
    eager CPU build. Lazy vs eager on one device runs the same kernels, so it
    must match to float32 round-off; GPU vs CPU uses different FFTs and an
    atomic scatter-add, whose round-off is bounded by ~eps log2(N) of the
    peak -- 1e-5 of the peak covers both."""
    from abtem.core.backend import asnumpy

    potential = Potential(
        atoms,
        gpts=gpts,
        device=device,
        slice_thickness=slice_thickness,
    )
    result = potential.build(lazy=lazy).compute()
    assert_array_matches_device(result.array, device)

    reference = Potential(
        atoms, gpts=gpts, device="cpu", slice_thickness=slice_thickness
    ).build(lazy=False)
    np.testing.assert_allclose(
        asnumpy(result.array),
        reference.array,
        rtol=0,
        atol=1e-5 * float(np.abs(reference.array).max()),
    )


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


def test_lazy_potential_unit_is_evaluated_once():
    """A lazily built unit is materialized once, not once per slice.

    A unit the caller has built themselves is used as-is, so a lazy one used
    to be recomputed every time a slice consumed it -- once per unit slice per
    z-repetition.
    """
    import dask.array as da
    from ase.build import bulk

    evaluations = []

    def tap(block):
        evaluations.append(None)
        return block

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    unit = Potential(atoms, gpts=32).build(lazy=True)
    # 'meta' given explicitly: without it dask infers the output type by
    # calling tap on a probe block, which the count would pick up.
    unit._array = da.map_blocks(
        tap, unit.array, meta=np.array((), dtype=unit.array.dtype)
    )
    assert unit.array.npartitions == 1
    assert not evaluations

    crystal = CrystalPotential(unit, repetitions=(2, 2, 4))
    assert len(crystal) > 1

    built = crystal.build().compute()

    assert len(evaluations) == 1
    assert built.array.shape == (len(crystal), 64, 64)


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

    crystal_potential = crystal_potential.compute()

    assert num_frozen_phonons == crystal_potential.num_configurations

    # Every repetition of the unit draws a configuration from the unit's pool
    # (CrystalPotential docstring). For a pre-built unit the pool is the
    # unit's own array, so each tile must equal one of its configurations. A
    # unit built from atoms re-draws its displacements per crystal member, so
    # there the oracle is the k = 0 sum rule: displacing atoms does not change
    # int V dA, so every tile must integrate to the undisplaced unit's total
    # (exact for the infinite projection up to float32 round-off).
    is_array_unit = isinstance(potential_unit, PotentialArray)
    unit = potential_unit if is_array_unit else potential_unit.build()
    unit_array = np.asarray(unit.compute().array)
    n_slices, gx, gy = unit_array.shape[-3:]
    unit_configurations = unit_array.reshape((-1, n_slices, gx, gy))
    crystal_array = np.asarray(crystal_potential.array).reshape(
        (-1, n_slices * tile[2], gx * tile[0], gy * tile[1])
    )
    assert len(crystal_array) == num_frozen_phonons
    unit_totals = unit_configurations.astype(np.float64).sum((1, 2, 3))
    for configuration in crystal_array:
        for i in range(tile[0]):
            for j in range(tile[1]):
                for k in range(tile[2]):
                    block = configuration[
                        k * n_slices : (k + 1) * n_slices,
                        i * gx : (i + 1) * gx,
                        j * gy : (j + 1) * gy,
                    ]
                    if is_array_unit:
                        assert any(
                            np.array_equal(block, candidate)
                            for candidate in unit_configurations
                        )
                    else:
                        total = block.astype(np.float64).sum()
                        np.testing.assert_allclose(
                            total, unit_totals, rtol=1e-4,
                            atol=1e-4 * float(np.abs(unit_totals).max()),
                        )


def test_crystal_potential_get_sliced_atoms_matches_manual_tile():
    """CrystalPotential.get_sliced_atoms tiles the unit's transformed atoms by
    the repetitions, matching a manually-tiled Potential's sliced atoms."""
    import numpy as np

    from abtem.slicing import SliceIndexedAtoms

    unit_atoms = si_cubic_atoms()
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
    unit_pot = Potential(
        si_cubic_atoms(), gpts=(16, 16), slice_thickness=5.43
    )
    cryst = CrystalPotential(unit_pot, repetitions=(2, 2, 2))
    assert cryst.get_sliced_atoms() is cryst.get_sliced_atoms()


def test_crystal_potential_get_sliced_atoms_frozen_phonons_equilibrium():
    """For a frozen-phonon CrystalPotential, get_sliced_atoms returns the
    equilibrium (un-displaced) atoms, because the ensemble draws an independent
    random unit configuration per z-repetition (no single displaced
    realisation) and column identification wants equilibrium positions."""
    import numpy as np

    import abtem

    unit_atoms = si_cubic_atoms()
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


@devices
def test_eager_build_populates_all_frozen_phonon_configs(device):
    """Eager ``build(lazy=False)`` of a multi-config frozen-phonon potential must
    populate *every* ensemble member, not just the first. Regression for a bug
    where the ensemble write index was hardcoded to 0, so all configs
    overwrote config 0 and configs 1..N-1 were left as zeros -- which in turn
    made CrystalPotential (it builds its pool eagerly) reshuffle a pool of one
    real config plus N-1 vacuum slices."""
    import numpy as np

    import abtem
    from abtem.core.backend import asnumpy

    unit_atoms = si_diamond_atoms()
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


@devices
@pytest.mark.parametrize("lazy", [True, False])
def test_crystal_potential_frozen_phonons_lateral_disorder(lazy, device):
    """A frozen-phonon CrystalPotential must reproduce *lateral* (in-plane)
    disorder: each lateral repetition draws an independent configuration from
    the pool (a mosaic), so the tiles differ from one another. Regression for
    the original ``.tile()`` behaviour that replicated a single displaced unit
    across every tile -- giving zero in-plane disorder (and hence no diffuse /
    Kikuchi scattering)."""
    import numpy as np

    import abtem
    from abtem.core.backend import asnumpy

    si = si_diamond_atoms()
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


@devices
def test_crystal_potential_pool_enlarged_to_avoid_lateral_duplication(device):
    """When the frozen-phonon pool is smaller than the number of lateral tiles,
    CrystalPotential enlarges it (warning) so every tile draws a distinct
    configuration and no two tiles in a layer are identical."""
    import numpy as np

    import abtem
    from abtem.core.backend import asnumpy

    si = si_diamond_atoms()
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


@devices
def test_crystal_potential_balanced_pool_drawing(device):
    """Pool configurations are drawn without replacement over the WHOLE
    crystal (balanced budgets), not just within a z-layer: a pool matching
    the total number of unit-cell slots gives every slot a distinct
    configuration (statistically identical to tiling displaced atoms), and a
    smaller pool spreads reuse exactly evenly."""
    from collections import Counter

    import abtem
    from abtem.core.backend import asnumpy

    si = si_diamond_atoms()
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


@devices
def test_crystal_potential_ensemble_members_have_independent_pools(device):
    """Ensemble members (num_frozen_phonons / seeds) all share the same
    ``potential_unit`` object, so without reseeding they would rebuild the
    identical, fixed pool of atomic snapshots and differ only in how those
    same snapshots are arranged -- not in which displacements exist. Each
    member's pool must instead be independently reseeded from that member's
    own seed, even when the pool is already at (or above) the size needed for
    a single crystal to be exact, so the ensemble does not need to be sized
    for the number of members."""
    import abtem
    from abtem.core.backend import asnumpy

    si = si_diamond_atoms()
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
    unit_pot = Potential(
        si_cubic_atoms(), gpts=(16, 16), slice_thickness=5.43
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
def si_potential(request):
    """Build a Si 2x2x5 potential for depth profile tests, on the device the
    test is parametrised over (cpu when it is not)."""
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device", "cpu") if callspec else "cpu"
    atoms = si_cubic_atoms() * (2, 2, 5)
    return Potential(atoms, slice_thickness=1.0, gpts=(32, 32), device=device)


@devices
def test_potential_depth_profile_shape(si_potential, device):
    pot = si_potential.build().compute()
    assert_array_matches_device(pot.array, device)
    profile = pot.depth_profile()
    n_x = pot.gpts[0]
    n_z = pot.num_slices
    assert profile.shape == (n_x, n_z)


@devices
def test_potential_depth_profile_x_projection(si_potential, device):
    pot = si_potential.build().compute()
    profile = pot.depth_profile(projection_axis="x")
    n_y = pot.gpts[1]
    n_z = pot.num_slices
    assert profile.shape == (n_y, n_z)


@devices
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


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_finite_projection_smooths_only_the_elements_given_a_sigma(device):
    """With a smoothing width for Mg only, the finite MgO slice is the smoothed Mg part
    plus the unsmoothed O part; every element without a width used to be dropped."""
    import ase.build

    from abtem.parametrizations import LobatoParametrization

    atoms = ase.build.bulk("MgO", "rocksalt", a=4.21, cubic=True)
    symbols = np.array(atoms.get_chemical_symbols())

    def build(atoms, sigmas):
        parametrization = LobatoParametrization(sigmas=sigmas)
        potential = Potential(
            atoms,
            sampling=0.1,
            slice_thickness=2.105,
            projection="finite",
            parametrization=parametrization,
            device=device,
        )
        return asnumpy(potential.build(lazy=False).array)

    both = build(atoms, {"Mg": 0.05})
    magnesium = build(atoms[symbols == "Mg"], {"Mg": 0.05})
    oxygen = build(atoms[symbols == "O"], {})

    np.testing.assert_allclose(
        both, magnesium + oxygen, rtol=0, atol=1e-5 * np.abs(both).max()
    )
    assert np.abs(oxygen).max() > 0.1 * np.abs(both).max()


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


@pytest.mark.parametrize(
    "exit_planes, expected",
    [
        (10, (-1, 9, 19)),
        (19, (-1, 18, 19)),
        # every integer up to the number of slices includes the entrance plane
        (20, (-1, 19)),
        (21, (19,)),
        (None, (19,)),
    ],
)
def test_integer_exit_planes_include_entrance_plane_up_to_num_slices(
    exit_planes, expected
):
    # issue #515: exit_planes == num_slices used to drop the thickness axis while
    # exit_planes == num_slices - 1 kept it
    from abtem.potentials.iam import _validate_exit_planes

    assert _validate_exit_planes(exit_planes, 20) == expected


def test_potential_array_slicing_maps_exit_planes():
    # slicing a potential array must map its exit planes into the sliced range,
    # otherwise the exit plane can fall outside the slices and the multislice
    # algorithm silently returns an unpropagated wave function
    import abtem

    atoms = si_cubic_atoms() * (2, 2, 8)
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

    def test_a_plain_atoms_potential_does_not_rewrite_its_own_atoms(self):
        atoms = self._atoms()
        potential = Potential(atoms, gpts=(32, 32), slice_thickness=1.0)
        stored = potential.frozen_phonons.atoms
        before = stored.positions.copy()
        _build_with_numpy_fft(potential)
        assert np.array_equal(stored.positions, before)

    def test_a_list_of_atoms_does_not_rewrite_the_callers_objects(self):
        """`Potential([a])` keeps a reference, so the write reached the caller."""
        atoms = self._atoms()
        before = atoms.positions.copy()
        _build_with_numpy_fft(Potential([atoms], gpts=(32, 32), slice_thickness=1.0))
        assert np.array_equal(atoms.positions, before)

    def test_a_prebuilt_dummy_frozen_phonons_does_not_rewrite_the_callers_atoms(self):
        from abtem.inelastic.phonons import DummyFrozenPhonons

        atoms = self._atoms()
        before = atoms.positions.copy()
        _build_with_numpy_fft(
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
        _build_with_numpy_fft(Potential(phonons, gpts=(32, 32), slice_thickness=1.0))
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
        _build_with_numpy_fft(shared)  # wraps `atoms` in place on the unfixed code
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

        _build_with_numpy_fft(Potential(wrapped, gpts=(32, 32), slice_thickness=1.0))
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

    @staticmethod
    def _edge_atoms(offset=0.0, lateral=4.0):
        """Four B atoms in a 4 A tall cell, two of them just inside a face.

        ``FrozenPhonons(sigmas=0.25, seed=1)`` then displaces one out through
        the entrance face and one out through the exit face; the tests assert
        that rather than assume it.
        """
        import ase
        import numpy as np

        return ase.Atoms(
            "B4",
            positions=[
                [offset + 1.0, offset + 1.0, 0.05],
                [offset + 2.0, offset + 2.0, 0.05],
                [offset + 3.0, offset + 3.0, 2.0],
                [offset + 1.0, offset + 3.0, 3.95],
            ],
            cell=np.diag([lateral, lateral, 4.0]),
            pbc=True,
        )

    @pytest.mark.parametrize("periodic", [True, False])
    def test_wrapping_follows_the_potentials_periodicity(self, periodic):
        """``Potential(periodic=False)`` deliberately never wraps.

        Its atoms are cut from a larger repeated potential and randomised
        *after* padding, so an edge atom displaced just outside the cell
        belongs at the face it left, not the opposite one. Wrapping it
        unconditionally moved it the full height of the box -- the same depth
        corruption the wrap exists to prevent, for the other path.

        Not wrapping must not mean dropping, though: the atom displaced out
        through the exit face used to fall outside every slice and vanish.
        """
        import numpy as np

        from abtem.inelastic.core_loss import _extract_scattering_sites
        from abtem.inelastic.phonons import FrozenPhonons

        atoms = self._edge_atoms()
        phonons = FrozenPhonons(atoms, num_configs=1, sigmas=0.25, seed=1)
        potential = Potential(
            phonons, gpts=(32, 32), slice_thickness=1.0, periodic=periodic
        )
        sliced = potential.get_sliced_atoms()
        z = sliced.atoms.positions[:, 2]

        if periodic:
            assert np.all((z >= 0.0) & (z < 4.0))
        else:
            # Unwrapped, and out through *both* faces -- otherwise this does
            # not exercise the non-periodic path at either end.
            assert z.min() < 0.0 and z.max() >= 4.0

        # Conservation: every atom is held by exactly one slice -- the one
        # containing its depth, or, for an atom displaced out of the cell, the
        # face slice it left through.
        per_slice = self._per_slice(sliced)
        expected = [0] * 4
        for depth in z:
            expected[min(max(int(np.floor(depth / 1.0)), 0), 3)] += 1
        assert sum(per_slice) == len(atoms)
        assert per_slice == expected

        # Explicitly passed sites must follow the same convention, so that
        # sites=<Atoms> and sites=None never disagree.
        from_potential = self._per_slice(_extract_scattering_sites(potential, None))
        from_caller = self._per_slice(
            _extract_scattering_sites(potential, sliced.atoms)
        )
        assert from_caller == from_potential

    def test_atoms_outside_the_cell_stay_in_the_face_slices_without_wrapping(self):
        """Without a wrap, an out-of-cell atom is kept, at its true position."""
        import ase

        import numpy as np

        from abtem.slicing import SliceIndexedAtoms

        atoms = ase.Atoms(
            "B3",
            positions=[[1.0, 1.0, -0.3], [1.0, 1.0, 2.0], [1.0, 1.0, 4.36]],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        sliced = SliceIndexedAtoms(atoms, slice_thickness=1.0, wrap=False)
        # z = -0.3 belongs to the entrance slice [0, 1), z = 2.0 to [2, 3) and
        # z = 4.36 to the exit slice [3, 4): nothing is lost, and nothing is
        # moved to the opposite face.
        assert self._per_slice(sliced) == [1, 0, 1, 1]
        assert np.array_equal(sliced.atoms.positions, atoms.positions)

    def test_atoms_far_outside_the_faces_warn_but_are_kept(self):
        """Clamping a misplaced atom into a face slice is a guess at its depth,
        so it warns; within the threshold (the test above, which runs with
        warnings as errors) it does not."""
        import ase

        import numpy as np

        from abtem.slicing import FACE_SLICE_WARNING_DISTANCE, SliceIndexedAtoms

        far = FACE_SLICE_WARNING_DISTANCE + 0.5
        atoms = ase.Atoms(
            "B4",
            positions=[
                [1.0, 1.0, -far],
                [1.0, 1.0, 2.0],
                [1.0, 1.0, 4.0 + far],
                [1.0, 1.0, 100.0],
            ],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        with pytest.warns(UserWarning, match=r"^3 atom\(s\) lie more than"):
            sliced = SliceIndexedAtoms(atoms, slice_thickness=1.0, wrap=False)

        assert self._per_slice(sliced) == [1, 0, 1, 2]

    @staticmethod
    def _displaced_far_past_the_exit_face():
        """Frozen phonons that displace the atom at z = 3.9 more than
        FACE_SLICE_WARNING_DISTANCE past the exit face of the 4 A cube (by 2.3
        and 2.5 A in the first configuration, with seed 6); every atom is given
        inside the cell."""
        atoms = Atoms(
            "B2", positions=[(2, 2, 2), (2, 2, 3.9)], cell=[4.0] * 3, pbc=True
        )
        return FrozenPhonons(atoms, num_configs=2, sigmas=3.0, seed=6)

    def test_far_outside_face_warning_points_at_the_caller(self):
        """The warning about an atom displaced far out of the cell is attributed
        to the line that builds the potential, not to a frame inside abTEM."""
        potential = Potential(
            self._displaced_far_past_the_exit_face(),
            sampling=0.2,
            slice_thickness=1.0,
            periodic=False,
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            potential.build(lazy=False)

        face = [r for r in records if "lie more than" in str(r.message)]
        assert face
        assert all(r.filename == __file__ for r in face)

    def test_far_outside_face_warning_of_a_lazy_build_is_not_inside_abtem(self):
        """A lazy build raises the warning in a dask worker thread, whose stack
        holds no frame of the caller. It is attributed to a frame outside abTEM,
        not to one inside it or to ``<sys>``."""
        potential = Potential(
            self._displaced_far_past_the_exit_face(),
            sampling=0.2,
            slice_thickness=1.0,
            periodic=False,
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            potential.build(lazy=True).compute()

        face = [r for r in records if "lie more than" in str(r.message)]
        assert face
        package = os.path.dirname(abtem.__file__) + os.sep
        for record in face:
            assert record.filename != "<sys>"
            assert not record.filename.startswith(package)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("source", ["atoms", "frozen_phonons", "trajectory"])
    def test_atoms_given_far_outside_along_z_warn_once_where_the_potential_is_made(
        self, source, lazy
    ):
        """Atoms given far outside the cell along z are folded into it, which a
        cell meant to enclose them does not intend. The warning is raised when
        the potential is constructed, so it names the caller's line for a lazy
        build too, and the blocks of an ensemble do not repeat it."""
        from abtem.inelastic.phonons import AtomsEnsemble

        atoms = Atoms(
            "B2", positions=[(2, 2, 2), (2, 2, 14.3)], cell=[4.0] * 3, pbc=True
        )
        given = {
            "atoms": atoms,
            "frozen_phonons": FrozenPhonons(atoms, num_configs=2, sigmas=0.05, seed=1),
            "trajectory": AtomsEnsemble([atoms, atoms.copy()]),
        }[source]

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            potential = Potential(
                given, sampling=0.2, slice_thickness=1.0, periodic=False
            )
            potential.build(lazy=lazy).compute()

        far = [r for r in records if "outside the cell along z" in str(r.message)]
        assert len(far) == 1
        assert str(far[0].message).startswith("1 atom(s) are given more than 2.0 Å")
        assert far[0].filename == __file__
        assert not [r for r in records if "lie more than" in str(r.message)]

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_crystal_of_a_non_periodic_frozen_phonon_unit_keeps_every_atom(
        self, device, lazy
    ):
        """A CrystalPotential draws its tiles from the unit's frozen-phonon pool,
        built by the unit's own slicing. A pool configuration that pushed the
        atom near the exit face out of the cell used to lose it, so 3 of these
        4 members were 0.75 of the expected potential."""
        import ase
        import numpy as np

        from abtem.core.backend import asnumpy
        from abtem.inelastic.phonons import FrozenPhonons

        cell = np.diag([4.0, 4.0, 4.0])
        kwargs = dict(gpts=(32, 32), slice_thickness=1.0, device=device)

        # One atom 0.01 A below the exit face: sigma = 0.2 pushes it out in
        # about half of the configurations.
        atoms = ase.Atoms(
            "B2", positions=[[1.0, 1.0, 2.0], [3.0, 3.0, 3.99]], cell=cell, pbc=True
        )
        unit = Potential(
            FrozenPhonons(atoms, num_configs=4, sigmas=0.2, seed=5),
            periodic=False,
            **kwargs,
        )
        crystal = CrystalPotential(
            unit,
            repetitions=(1, 1, 2),
            num_frozen_phonons=4,
            seeds=7,
            ensemble_mean=False,
        )
        array = asnumpy(crystal.build(lazy=lazy).compute().array)

        # Oracle: an atom's infinite projection summed over the cell is its
        # q = 0 Fourier component, independent of where the atom sits, so
        # every member must sum to (2 atoms x 2 units) single atoms.
        single = asnumpy(
            Potential(ase.Atoms("B", positions=[[2.0, 2.0, 2.0]], cell=cell), **kwargs)
            .build()
            .compute()
            .array
        ).sum()
        np.testing.assert_allclose(
            array.reshape(len(array), -1).sum(axis=1), 4 * single, rtol=1e-5
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_non_periodic_infinite_projection_conserves_every_atom(self, device):
        """An atom displaced out of a non-periodic cell used to lose all of its
        potential -- a quarter of the total for these four atoms."""
        import numpy as np

        from abtem.core.backend import asnumpy
        from abtem.inelastic.phonons import FrozenPhonons

        phonons = FrozenPhonons(self._edge_atoms(), num_configs=1, sigmas=0.25, seed=1)
        potential = Potential(
            phonons, gpts=(32, 32), slice_thickness=1.0, periodic=False, device=device
        )
        displaced = potential.get_sliced_atoms().atoms
        z = displaced.positions[:, 2]
        assert z.min() < 0.0 and z.max() >= 4.0
        projected = asnumpy(potential.build().project().compute().array)
        projected = projected.reshape((-1,) + projected.shape[-2:])[0]

        # Oracle: the infinite projection of an atom does not depend on its
        # depth, so the projected potential must equal that of the same
        # in-plane positions with every atom moved to mid-depth, inside the
        # cell, where no boundary handling is involved.
        inside = displaced.copy()
        inside.positions[:, 2] = 2.0
        reference = asnumpy(
            Potential(inside, gpts=(32, 32), slice_thickness=1.0, device=device)
            .build()
            .project()
            .compute()
            .array
        )
        np.testing.assert_allclose(
            projected, reference, rtol=1e-5, atol=1e-5 * np.abs(reference).max()
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_non_periodic_finite_projection_integrates_atoms_beyond_the_faces(
        self, device
    ):
        """Finite projection keeps out-of-cell atoms in its candidate set.

        ``SlicedAtoms`` selects every atom within the integrator cutoff of a
        slice, so an atom displaced beyond a face still contributes the tail
        of its potential that lies inside the cell.
        """
        import ase
        import numpy as np

        from abtem.core.backend import asnumpy
        from abtem.inelastic.phonons import FrozenPhonons

        # Laterally 12 A with the atoms in the middle, so no in-plane image
        # lies within the cutoff (~4.3 A for B) and the padded atom set can
        # be reused as-is below.
        phonons = FrozenPhonons(
            self._edge_atoms(offset=4.0, lateral=12.0),
            num_configs=1,
            sigmas=0.25,
            seed=1,
        )
        potential = Potential(
            phonons,
            gpts=(64, 64),
            slice_thickness=1.0,
            periodic=False,
            projection="finite",
            device=device,
        )
        held = potential.get_sliced_atoms().atoms
        z = held.positions[:, 2]
        assert z.min() < 0.0 and z.max() >= 4.0
        slices = asnumpy(potential.build().compute().array)
        slices = slices.reshape((-1,) + slices.shape[-3:])[0]

        # Oracle: translation invariance. The same atoms raised by 12 A into
        # a 28 A cell sit wholly inside it, more than a cutoff from every face
        # and from their own periodic images, so its slices 12..15 are the
        # integrals over the same absolute depths [0, 4) with no boundary
        # handling involved.
        shift = 12
        raised = ase.Atoms(
            held.numbers,
            positions=held.positions + [0.0, 0.0, shift],
            cell=np.diag([12.0, 12.0, 4.0 + 2 * shift]),
            pbc=True,
        )
        reference = asnumpy(
            Potential(
                raised,
                gpts=(64, 64),
                slice_thickness=1.0,
                projection="finite",
                device=device,
            )
            .build()
            .compute()
            .array
        )[shift : shift + 4]
        np.testing.assert_allclose(
            slices, reference, rtol=1e-5, atol=1e-5 * np.abs(reference).max()
        )

    @staticmethod
    def _outside_atoms():
        """Five B atoms in a 4 A cube, four of them already outside it: one
        through each in-plane face and one through each z face."""
        import ase
        import numpy as np

        return ase.Atoms(
            "B5",
            positions=[
                [-0.3, 2.0, 2.0],
                [2.0, 4.2, 2.0],
                [1.0, 1.0, 4.3],
                [3.0, 3.0, -0.2],
                [2.0, 2.0, 2.0],
            ],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )

    def test_pad_atoms_crops_only_along_the_repeated_axes(self):
        import numpy as np

        from abtem.atoms import pad_atoms

        atoms = self._outside_atoms()

        # Nothing is repeated, so nothing is cropped either.
        padded = pad_atoms(atoms, margins=0.0, directions="z")
        assert np.array_equal(padded.positions, atoms.positions)

        # Repeated along z only: the z crop still trims the images, but no
        # atom is cropped in-plane.
        margin = 0.5
        padded = pad_atoms(atoms, margins=margin, directions="z")
        expected = [
            position + [0.0, 0.0, shift]
            for position in atoms.positions
            for shift in (-4.0, 0.0, 4.0)
            if -margin <= position[2] + shift < 4.0 + margin
        ]
        assert sorted(map(tuple, padded.positions.round(12))) == sorted(
            map(tuple, np.array(expected).round(12))
        )

        # Margins are per axis: a zero margin in-plane repeats nothing there,
        # which is the same as padding z alone.
        per_axis = pad_atoms(atoms, margins=(0.0, 0.0, margin))
        assert np.array_equal(per_axis.positions, padded.positions)

        # A margin per direction used to be paired with `directions` by zip.
        with pytest.raises(ValueError, match="three values for x, y and z"):
            pad_atoms(atoms, margins=(margin,), directions="z")

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_non_periodic_potential_keeps_atoms_already_outside_the_cell(self, device):
        """Atoms that reach the potential already outside the cell, in-plane or
        in depth, used to be cropped before slicing and lose all of their
        potential."""
        import numpy as np

        from abtem.core.backend import asnumpy

        atoms = self._outside_atoms()
        potential = Potential(
            atoms, gpts=(32, 32), slice_thickness=1.0, periodic=False, device=device
        )
        assert len(potential.get_sliced_atoms().atoms) == len(atoms)

        # Oracle: the in-plane build is periodic and the infinite projection
        # does not depend on depth, so the projected potential must equal that
        # of the periodic potential, which wraps every atom into the cell.
        def projected(periodic):
            return asnumpy(
                Potential(
                    atoms,
                    gpts=(32, 32),
                    slice_thickness=1.0,
                    periodic=periodic,
                    device=device,
                )
                .build()
                .project()
                .compute()
                .array
            )

        reference = projected(True)
        np.testing.assert_allclose(
            projected(False),
            reference,
            rtol=1e-5,
            atol=1e-5 * np.abs(reference).max(),
        )

    @staticmethod
    def _far_in_plane(shifts=((0, 0), (0, 0), (0, 0))):
        """Three B atoms in a 4 x 5 x 4 A cell (unequal in-plane lengths), moved
        by whole cell lengths ``shifts`` [(nx, ny) per atom]. The repeated
        structure is the same for every shift."""
        atoms = Atoms(
            "B3",
            positions=[[2.0, 2.5, 2.0], [1.0, 1.0, 1.2], [1.0, 3.0, 2.7]],
            cell=np.diag([4.0, 5.0, 4.0]),
            pbc=True,
        )
        atoms.positions[:, :2] += np.array(shifts) * [4.0, 5.0]
        return atoms

    @staticmethod
    def _slices(atoms, lazy=False, **kwargs):
        potential = Potential(atoms, sampling=0.1, slice_thickness=0.5, **kwargs)
        return asnumpy(potential.build(lazy=lazy).compute().array)

    # Quadrature pads 4.29 A in-plane, repeating the cell twice to each side
    # along x (4 A) and once along y (5 A): an atom given further out than that
    # has an image missing in the cell.
    FAR_IN_PLANE = {
        "far_x": ((0, 0), (-2, 0), (0, 0)),  # x = -7
        "far_y": ((0, 0), (0, 0), (0, 2)),  # y = 13
        "far_x_and_y": ((0, 0), (2, -2), (-1, 3)),
    }

    @float64_devices
    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("case", list(FAR_IN_PLANE))
    def test_quadrature_potential_folds_atoms_far_outside_in_plane(
        self, case, lazy, device
    ):
        """The padding repeats the cell a number of times set by the cell, so the
        images of an atom further outside than that reaches never entered the
        cell. Moving an atom by whole cell lengths leaves the periodic in-plane
        build unchanged."""
        kwargs = dict(projection="finite", periodic=False, device=device)
        moved = self._slices(
            self._far_in_plane(self.FAR_IN_PLANE[case]), lazy=lazy, **kwargs
        )
        reference = self._slices(self._far_in_plane(), lazy=lazy, **kwargs)
        np.testing.assert_allclose(
            moved, reference, rtol=0, atol=1e-10 * np.abs(reference).max()
        )

    @float64_devices
    @pytest.mark.parametrize(
        "length_x, x, y",
        [(4.0, -7.0, 1.0), (4.0, 1.0, 13.0), (20.0, -5.0, 1.0), (20.0, 25.0, 1.0)],
        ids=["far_x", "far_y", "wide_below", "wide_above"],
    )
    def test_quadrature_configurations_keep_atoms_far_outside_in_plane(
        self, length_x, x, y, device
    ):
        """An atom given outside the cell is kept, wherever the padding stops:
        beyond its reach, or, in a cell wider than twice the cutoff, in the range
        it reaches but the crop of the padding drops."""
        atoms = Atoms(
            "B2",
            positions=[[2.0, 2.5, 2.0], [x, y, 1.2]],
            cell=np.diag([length_x, 5.0, 4.0]),
            pbc=True,
        )
        potential = Potential(
            FrozenPhonons(atoms, num_configs=2, sigmas=0.05, seed=1),
            sampling=0.2,
            slice_thickness=1.0,
            projection="finite",
            periodic=False,
            device=device,
        )
        for configuration in potential.to_atoms_ensemble().trajectory:
            assert len(configuration) == len(atoms)

    @float64_devices
    @pytest.mark.parametrize("x", [-0.3, -1e-16])
    def test_quadrature_configurations_leave_atoms_within_reach_unwrapped(
        self, x, device
    ):
        """An atom given just outside the cell is within the padding's reach, so
        it is displaced where it is given, not moved a cell length and drawn in
        another order."""
        atoms = Atoms(
            "B2",
            positions=[[2.0, 2.5, 2.0], [x, 1.0, 1.2]],
            cell=np.diag([4.0, 5.0, 4.0]),
            pbc=True,
        )
        potential = Potential(
            FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=3),
            sampling=0.1,
            slice_thickness=0.5,
            projection="finite",
            periodic=False,
            device=device,
        )
        for configuration in potential.to_atoms_ensemble().trajectory:
            # sigma is 0.1 A, a cell length is at least 4 A.
            displacement = configuration.positions - atoms.positions
            assert np.abs(displacement).max() < 0.5

    INTEGRATORS = {
        "infinite": {"projection": "infinite"},
        "gaussian": {"integrator": GaussianProjectionIntegrals()},
        "quadrature": {"projection": "finite"},
    }

    @staticmethod
    def _given_along_z(z):
        """Two B atoms in a 4 A cube, the second given at height `z`."""
        return Atoms(
            "B2", positions=[(2.0, 2.0, 2.0), (1.0, 1.5, z)], cell=[4.0] * 3, pbc=True
        )

    @pytest.mark.filterwarnings("ignore:.*outside the cell along z:UserWarning")
    @pytest.mark.parametrize("integrator", list(INTEGRATORS))
    @pytest.mark.parametrize("z", [14.3, -10.3])
    def test_configurations_keep_an_atom_given_far_outside_along_z(self, integrator, z):
        """The padding along z of the finite integrators did not reach an atom
        given more than a cell height outside, which was lost silently. Every
        integrator now keeps it, at its image in the cell."""
        atoms = self._given_along_z(z)
        potential = Potential(
            atoms,
            sampling=0.2,
            slice_thickness=1.0,
            periodic=False,
            **self.INTEGRATORS[integrator],
        )
        for configuration in potential.to_atoms_ensemble().trajectory:
            assert len(configuration) == len(atoms)
            assert configuration.positions[1, 2] == pytest.approx(z % 4.0)

    @float64_devices
    @pytest.mark.filterwarnings("ignore:.*outside the cell along z:UserWarning")
    @pytest.mark.parametrize("integrator", list(INTEGRATORS))
    @pytest.mark.parametrize("z", [6.3, 10.3, 14.3, -2.5, -10.3])
    def test_atom_given_far_outside_along_z_is_its_image_in_the_cell(
        self, integrator, z, device
    ):
        """abTEM #540: a non-periodic potential is cut out of the repeated
        structure, so an atom given more than FACE_SLICE_WARNING_DISTANCE
        outside the cell along z is its image in the cell for every integrator,
        at the same depth. The infinite projection used to put it in the face
        slice, and the finite integrators to lose it beyond the padding."""
        kwargs = dict(self.INTEGRATORS[integrator], periodic=False, device=device)
        given = self._slices(self._given_along_z(z), **kwargs)
        image = self._slices(self._given_along_z(z % 4.0), **kwargs)
        np.testing.assert_allclose(
            given, image, rtol=0, atol=1e-10 * np.abs(image).max()
        )

    @float64_devices
    @pytest.mark.filterwarnings("ignore:.*outside the cell along z:UserWarning")
    @pytest.mark.parametrize("integrator", list(INTEGRATORS))
    @pytest.mark.parametrize("z", [4.3, 6.3, 8.3, 10.3, 12.3, 14.3])
    def test_non_periodic_total_keeps_an_atom_given_outside_along_z(
        self, integrator, z, device
    ):
        """The table of abTEM #540: the total of the non-periodic potential is
        that of the periodic one at any distance, also within
        FACE_SLICE_WARNING_DISTANCE, where the depths differ."""
        atoms = self._given_along_z(z)
        kwargs = dict(self.INTEGRATORS[integrator], device=device)
        total = self._slices(atoms, periodic=False, **kwargs).sum()
        reference = self._slices(atoms, periodic=True, **kwargs).sum()
        assert total == pytest.approx(reference, rel=1e-4)

    @pytest.mark.filterwarnings("ignore:.*outside the cell along z:UserWarning")
    @pytest.mark.parametrize(
        "z, expected",
        [(4.3, 3), (5.9, 3), (6.1, 2), (-1.9, 0), (-2.1, 1)],
    )
    def test_infinite_projection_keeps_the_face_rule_within_the_warning_distance(
        self, z, expected
    ):
        """Within FACE_SLICE_WARNING_DISTANCE of a face the infinite projection
        puts an atom given outside the cell in the face slice, as frozen phonons
        displace atoms, so that a configuration they return builds the same
        potential when it is given back. Further out the atom is its image, so
        the slice it is put in jumps at that distance."""
        potential = Potential(
            self._given_along_z(z),
            sampling=0.2,
            slice_thickness=1.0,
            periodic=False,
            projection="infinite",
        )
        # The first atom, at z = 2, is in slice 2.
        counts = [0, 0, 1, 0]
        counts[expected] += 1
        assert self._per_slice(potential.get_sliced_atoms()) == counts

    @float64_devices
    @pytest.mark.parametrize("integrator", list(INTEGRATORS))
    @pytest.mark.parametrize("z", [4.3, -0.3])
    def test_configurations_leave_atoms_within_reach_along_z_unwrapped(
        self, integrator, z, device
    ):
        """An atom given just outside the cell along z is within the padding's
        reach and within FACE_SLICE_WARNING_DISTANCE, so it is displaced where
        it is given, not moved a cell height and drawn in another order."""
        atoms = self._given_along_z(z)
        potential = Potential(
            FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=3),
            sampling=0.2,
            slice_thickness=1.0,
            periodic=False,
            device=device,
            **self.INTEGRATORS[integrator],
        )
        for configuration in potential.to_atoms_ensemble().trajectory:
            displacement = configuration.positions - atoms.positions
            assert np.abs(displacement).max() < 0.5

    @float64_devices
    @pytest.mark.parametrize(
        "integrator",
        [
            {"projection": "infinite"},
            {"integrator": GaussianProjectionIntegrals()},
            {"projection": "finite"},
        ],
        ids=["infinite", "gaussian", "quadrature"],
    )
    def test_cut_potential_folds_atoms_far_outside_the_cell(self, integrator, device):
        """A box other than the cell sends a non-periodic potential through
        ``cut_cell``, which repeated the cell only as far as the atoms inside it
        need."""
        kwargs = dict(integrator, periodic=False, box=(8.0, 10.0, 4.0), device=device)
        moved = self._slices(
            self._far_in_plane(self.FAR_IN_PLANE["far_x_and_y"]), **kwargs
        )
        reference = self._slices(self._far_in_plane(), **kwargs)
        np.testing.assert_allclose(
            moved, reference, rtol=0, atol=1e-10 * np.abs(reference).max()
        )

    @float64_devices
    @ignore_strain_warning
    @pytest.mark.parametrize(
        "integrator",
        [{"projection": "infinite"}, {"integrator": GaussianProjectionIntegrals()}],
        ids=["infinite", "gaussian"],
    )
    def test_default_box_keeps_an_atom_just_past_an_upper_face(
        self, integrator, device
    ):
        """The default box of a hexagonal cell is cut out of the repeated
        structure. An atom 0.08 A past the upper face (0.03 of the second
        lattice vector) is as much part of it as the same atom moved into the
        cell."""

        def potential(scaled_y):
            atoms = mx2("MoS2", vacuum=3.0)
            scaled = atoms.get_scaled_positions(wrap=False)
            scaled[1, 1] = scaled_y
            atoms.set_scaled_positions(scaled)
            return asnumpy(
                Potential(
                    atoms,
                    sampling=0.1,
                    slice_thickness=0.5,
                    periodic=False,
                    device=device,
                    **integrator,
                )
                .build(lazy=False)
                .array
            )

        reference = potential(0.03)
        np.testing.assert_allclose(
            potential(1.03), reference, rtol=0, atol=1e-10 * np.abs(reference).max()
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_iterated_frozen_phonons_match_the_non_periodic_ensemble(self, device):
        """``for atoms in frozen_phonons: Potential(atoms, periodic=False)``
        must build the same configurations as
        ``Potential(frozen_phonons, periodic=False)`` on a cell the potential
        does not transform. It used to drop the atoms that the iterated
        displacement had moved out of the cell.

        Only an untransformed cell is used: there, iteration already yields
        the simulated configurations, which is the contract #462 extends to
        transformed cells. A fix for #462 should leave this test passing."""
        import numpy as np

        from abtem.core.backend import asnumpy
        from abtem.inelastic.phonons import FrozenPhonons

        atoms = self._outside_atoms()
        atoms.wrap()
        phonons = FrozenPhonons(
            atoms, num_configs=3, sigmas=0.25, seed=1, ensemble_mean=False
        )
        iterated = list(phonons)
        scaled = np.concatenate(
            [config.get_scaled_positions(wrap=False) for config in iterated]
        )
        assert np.any((scaled[:, :2] < 0.0) | (scaled[:, :2] >= 1.0))

        kwargs = dict(
            gpts=(32, 32), slice_thickness=1.0, periodic=False, device=device
        )
        ensemble = asnumpy(Potential(phonons, **kwargs).build().compute().array)
        for i, configuration in enumerate(iterated):
            np.testing.assert_allclose(
                ensemble[i],
                asnumpy(Potential(configuration, **kwargs).build().compute().array),
                rtol=1e-5,
                atol=1e-5 * np.abs(ensemble[i]).max(),
            )

    @pytest.mark.parametrize("structure", ["graphene", "Mg", "MoS2"])
    @pytest.mark.parametrize(
        "integrator", ["scattering_factor", "gaussian", "quadrature"]
    )
    def test_non_periodic_cut_cell_is_not_padded_again(self, integrator, structure):
        """A transformed non-periodic cell is cut out of the repeated structure
        with the integrator's margin already included. Padding it periodically
        on top added images of those margin atoms: the infinite projection was
        empty, and the Gaussian and quadrature potentials many times too
        large.

        hcp Mg and MoS2 also put atoms within float noise of the faces after
        the cut, at scaled -3e-17 and 1 - 1e-16. With no margin to absorb them,
        a crop keeping both ends of that pair held each such atom twice."""
        import ase.build
        import numpy as np

        from abtem.atoms import orthogonalize_cell
        from abtem.integrals import (
            GaussianProjectionIntegrals,
            ScatteringFactorProjectionIntegrals,
        )

        integrators = {
            "scattering_factor": ScatteringFactorProjectionIntegrals,
            "gaussian": GaussianProjectionIntegrals,
            "quadrature": QuadratureProjectionIntegrals,
        }

        hexagonal = {
            "graphene": lambda: ase.build.graphene(vacuum=2),
            "Mg": lambda: ase.build.bulk("Mg"),
            "MoS2": lambda: ase.build.mx2("MoS2", vacuum=2),
        }[structure]()
        orthogonal = orthogonalize_cell(hexagonal)

        def build(atoms, periodic):
            return (
                Potential(
                    atoms,
                    gpts=64,
                    slice_thickness=0.5,
                    periodic=periodic,
                    integrator=integrators[integrator](),
                )
                .build()
                .compute()
                .array
            )

        # Oracle: the orthogonal cell is commensurate with the lattice, so
        # cutting it out of the repeated hexagonal structure must give the
        # potential of the same cell built periodically.
        reference = build(orthogonal, periodic=True)
        np.testing.assert_allclose(
            build(hexagonal, periodic=False),
            reference,
            rtol=1e-5,
            atol=1e-5 * np.abs(reference).max(),
        )

    def test_non_orthogonal_cell_raises_before_any_wrapping(self):
        import ase

        from abtem.slicing import SliceIndexedAtoms

        atoms = ase.Atoms(
            "B", positions=[[1.0, 1.0, 1.0]], cell=[[4, 0, 0], [1, 4, 0], [0, 0, 4]],
            pbc=True,
        )
        with pytest.raises(RuntimeError, match="orthogonal"):
            SliceIndexedAtoms(atoms, slice_thickness=1.0)


# A potential's `box`, `origin` and auto grid follow the arguments given.


GRID = dict(sampling=0.1, slice_thickness=1.0)


def _two_atoms():
    # Different lengths along x, y, z, and atoms off every symmetry position, so
    # no two axes can be confused.
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )


def _permuted_two_atoms():
    # `_two_atoms()` with the y and z axes exchanged, as plane="xz" sees it.
    atoms = _two_atoms()
    return Atoms(
        atoms.numbers,
        positions=atoms.positions[:, [0, 2, 1]],
        cell=[4.0, 5.0, 3.0],
        pbc=True,
    )


def _array(atoms, **kwargs):
    return asnumpy(abtem.Potential(atoms, **kwargs).build(lazy=False).array)


def _assert_same(actual, expected):
    actual, expected = asnumpy(actual), asnumpy(expected)
    assert actual.shape == expected.shape
    scale = np.abs(expected).max()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10 * scale)


def _integral(potential):
    array = asnumpy(potential.build(lazy=False).array)
    return array.sum() * np.prod(potential.sampling)


def _charge_density():
    shape = (16, 12, 20)
    x, y, z = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
    return (
        0.3
        + 0.1 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        + 0.05 * np.cos(4 * np.pi * z)
    )


class _FakeCalculator:
    # The part of a GPAW calculator that the GPAW magnetics read when they are
    # constructed.
    def __init__(self, atoms):
        self.atoms = atoms

    def get_number_of_grid_points(self):
        return np.array([16, 12, 20])


@ignore_strain_warning
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(box=(8.0, 9.0, 10.0)),
        dict(box=(4.0, 3.0, 5.5)),
        dict(origin=(1.0, 0.75, 0.0)),
        dict(origin=(0.0, 0.0, 1e-3)),
    ],
    ids=["box", "box-z", "origin", "origin-z"],
)
def test_charge_density_potential_rejects_a_box_or_origin(kwargs):
    with pytest.raises(NotImplementedError, match="default box"):
        ChargeDensityPotential(_two_atoms(), _charge_density(), **GRID, **kwargs)


@float64_devices
@ignore_strain_warning
def test_charge_density_potential_accepts_its_own_box():
    atoms = _two_atoms()
    rho = _charge_density()
    potential = ChargeDensityPotential(atoms, rho, box=(4.0, 3.0, 5.0), **GRID)
    assert potential.box == (4.0, 3.0, 5.0)
    _assert_same(
        potential.build(lazy=False).array,
        ChargeDensityPotential(atoms, rho, **GRID).build(lazy=False).array,
    )


@ignore_strain_warning
def test_charge_density_potential_default_box_follows_the_plane_and_cell():
    rho = _charge_density()
    atoms = _two_atoms()

    # plane="xz": the box is the cell with y and z exchanged.
    in_plane = ChargeDensityPotential(
        atoms, rho, plane="xz", box=(4.0, 5.0, 3.0), **GRID
    )
    assert in_plane.box == (4.0, 5.0, 3.0)
    with pytest.raises(NotImplementedError, match="default box"):
        ChargeDensityPotential(atoms, rho, plane="xz", box=(4.0, 3.0, 5.0), **GRID)

    # A non-orthogonal cell: the default box is its best orthogonal cell.
    skewed = graphene(vacuum=2.0)
    default = tuple(best_orthogonal_cell(skewed.cell))
    assert ChargeDensityPotential(skewed, rho, **GRID).box == pytest.approx(default)
    assert ChargeDensityPotential(skewed, rho, box=default, **GRID).box == (
        pytest.approx(default)
    )
    with pytest.raises(NotImplementedError, match="default box"):
        ChargeDensityPotential(
            skewed, rho, box=(default[0] * 2, default[1], default[2]), **GRID
        )


@ignore_strain_warning
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
@pytest.mark.parametrize(
    "kwargs",
    [dict(box=(8.0, 9.0, 10.0)), dict(origin=(1.0, 0.75, 0.0))],
    ids=["box", "origin"],
)
def test_gpaw_magnetics_reject_a_box_or_origin(builder, kwargs):
    calculator = _FakeCalculator(_two_atoms())
    with pytest.raises(NotImplementedError, match="default box"):
        builder(calculator, **GRID, **kwargs)


@ignore_strain_warning
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_magnetics_accept_their_own_box(builder):
    calculator = _FakeCalculator(_two_atoms())
    assert builder(calculator, box=(4.0, 3.0, 5.0), **GRID).box == (4.0, 3.0, 5.0)
    assert builder(calculator, origin=(0.0, 0.0, 0.0), **GRID).box == (4.0, 3.0, 5.0)


@ignore_strain_warning
@pytest.mark.parametrize(
    "build",
    [
        lambda **kwargs: ChargeDensityPotential(
            _two_atoms(), _charge_density(), **GRID, **kwargs
        ),
        lambda **kwargs: GPAWMagneticField(
            _FakeCalculator(_two_atoms()), **GRID, **kwargs
        ),
        lambda **kwargs: GPAWVectorPotential(
            _FakeCalculator(_two_atoms()), **GRID, **kwargs
        ),
    ],
    ids=["charge-density", "magnetic-field", "vector-potential"],
)
class TestRejectingBuildersValidateTheirArguments:
    def test_origin_none_is_the_zero_origin(self, build):
        assert build(origin=None).box == (4.0, 3.0, 5.0)

    @pytest.mark.parametrize(
        "origin", [(1.0, 0.5), ("1", "0", "0"), (np.nan, 0.0, 0.0), 1.0]
    )
    def test_invalid_origin_raises(self, build, origin):
        with pytest.raises(ValueError, match="origin"):
            build(origin=origin)

    @pytest.mark.parametrize("box", [("4", "3", "5"), (4.0, 3.0), (4.0, np.nan, 5.0)])
    def test_invalid_box_raises(self, build, box):
        with pytest.raises(ValueError, match="box"):
            build(box=box)


def _supercell_cases():
    atoms = _two_atoms()
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    orthogonal_graphene = orthogonalize_cell(graphene(vacuum=2.0))
    return [
        ("orthogonal 2x3x2", atoms, (2, 3, 2)),
        ("cubic Si 2x3x1", si, (2, 3, 1)),
        ("graphene 3x1x1", graphene(vacuum=2.0), (3, 1, 1), orthogonal_graphene),
    ]


@float64_devices
@ignore_strain_warning
@pytest.mark.parametrize("projection", ["infinite", "finite"])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("case", _supercell_cases(), ids=lambda c: c[0])
def test_supercell_box_matches_repeated_atoms(case, periodic, projection):
    name, atoms, repetitions, *orthogonal = case
    repeated = (orthogonal[0] if orthogonal else atoms) * repetitions
    box = tuple(np.diag(repeated.cell))

    potential = abtem.Potential(
        atoms, box=box, periodic=periodic, projection=projection, **GRID
    )

    assert potential.box == pytest.approx(box, rel=1e-12)
    _assert_same(
        potential.build(lazy=False).array,
        _array(repeated, projection=projection, **GRID),
    )


@float64_devices
@ignore_strain_warning
@pytest.mark.parametrize("projection", ["infinite", "finite"])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("box, cells", [((8.0, 9.0, 10.0), 12), ((12.0, 3.0, 5.0), 3)])
def test_projected_potential_integral_counts_the_cells_in_the_box(
    box, cells, periodic, projection
):
    # Each atom's projected potential integrates to the same value wherever it
    # sits, so the integral over a box that holds whole cells counts them.
    atoms = _two_atoms()
    one_cell = abtem.Potential(atoms, projection=projection, **GRID)
    in_box = abtem.Potential(
        atoms, box=box, periodic=periodic, projection=projection, **GRID
    )

    assert _integral(in_box) / _integral(one_cell) == pytest.approx(cells, rel=1e-9)


@float64_devices
@ignore_strain_warning
def test_projected_potential_integral_of_a_strained_box_counts_its_periods():
    # A 9 A box holds 2 periods of the 4 A axis and 3 of the 3 A axis, strained
    # onto the box, so the integral counts 2 x 3 x 2 cells. The infinite
    # projection does not depend on where the strain puts an atom relative to a
    # slice boundary.
    atoms = _two_atoms()
    one_cell = abtem.Potential(atoms, **GRID)
    in_box = abtem.Potential(atoms, box=(9.0, 9.0, 10.0), **GRID)

    assert _integral(in_box) / _integral(one_cell) == pytest.approx(12, rel=1e-9)


@float64_devices
@ignore_strain_warning
def test_box_that_is_not_a_supercell_strains_the_atoms_onto_it():
    atoms = _two_atoms()
    box = (9.0, 9.0, 10.0)
    potential = abtem.Potential(atoms, box=box, **GRID)
    assert potential.extent == pytest.approx(box[:2])
    _assert_same(
        potential.build(lazy=False).array,
        _array(orthogonalize_cell(atoms, box=box), **GRID),
    )


@float64_devices
@ignore_strain_warning
def test_box_with_one_period_compresses_the_cell_onto_it():
    atoms = _two_atoms()
    box = (2.1, 3.0, 5.0)
    potential = abtem.Potential(atoms, box=box, **GRID)
    assert potential.box == pytest.approx(box)
    assert potential.get_transformed_atoms().cell.lengths() == pytest.approx(box)
    _assert_same(
        potential.build(lazy=False).array,
        _array(orthogonalize_cell(atoms, box=box), **GRID),
    )


@float64_devices
@ignore_strain_warning
@pytest.mark.parametrize("box", [None, (4.0, 3.0, 5.0)])
@pytest.mark.parametrize("kind", [tuple, list, np.array])
def test_origin_translates_an_orthogonal_cell(box, kind):
    atoms = _two_atoms()
    origin = (1.0, 0.5, 0.0)
    translated = atoms.copy()
    translated.translate(-np.array(origin))
    translated.wrap()

    _assert_same(
        _array(atoms, origin=kind(origin), box=box, **GRID),
        _array(translated, **GRID),
    )


@float64_devices
@ignore_strain_warning
def test_origin_with_a_plane_translates_the_permuted_atoms():
    atoms = _two_atoms()
    origin = (1.0, 0.5, 0.25)
    # The origin is given relative to the atoms as provided: they are translated
    # first, then the plane is mapped to xy.
    translated = atoms.copy()
    translated.translate(-np.array(origin))
    translated.wrap()
    permuted = Atoms(
        translated.numbers,
        positions=translated.positions[:, [0, 2, 1]],
        cell=[4.0, 5.0, 3.0],
        pbc=True,
    )

    _assert_same(
        _array(atoms, plane="xz", origin=origin, **GRID), _array(permuted, **GRID)
    )


@float64_devices
@ignore_strain_warning
def test_zero_origin_given_as_a_list_changes_nothing():
    atoms = _two_atoms()
    _assert_same(_array(atoms, origin=[0.0, 0.0, 0.0], **GRID), _array(atoms, **GRID))


@float64_devices
@ignore_strain_warning
def test_plane_and_box_together():
    atoms = _two_atoms()
    potential = abtem.Potential(atoms, plane="xz", box=(8.0, 10.0, 3.0), **GRID)
    assert potential.box == (8.0, 10.0, 3.0)
    _assert_same(
        potential.build(lazy=False).array,
        _array(_permuted_two_atoms() * (2, 2, 1), **GRID),
    )


@float64_devices
@ignore_strain_warning
def test_box_equal_to_the_rotated_cell_with_a_plane_changes_nothing():
    # The box that a plane gives by default describes the rotated cell, not the
    # cell as it is given.
    atoms = _two_atoms()
    _assert_same(
        _array(atoms, plane="xz", box=(4.0, 5.0, 3.0), **GRID),
        _array(atoms, plane="xz", **GRID),
    )


@float64_devices
@ignore_strain_warning
def test_box_equal_to_the_unrotated_cell_with_a_plane_strains_the_rotated_cell():
    # (4, 3, 5) is the cell's own diagonal, but with plane="xz" the potential's
    # axes are the cell's x, z, y: the rotated 4 x 5 x 3 A cell is strained onto
    # it.
    atoms = _two_atoms()
    box = (4.0, 3.0, 5.0)
    potential = abtem.Potential(atoms, plane="xz", box=box, **GRID)
    assert potential.box == box
    _assert_same(
        potential.build(lazy=False).array,
        _array(orthogonalize_cell(_permuted_two_atoms(), box=box), **GRID),
    )


@float64_devices
@ignore_strain_warning
def test_box_equal_to_the_cell_changes_nothing():
    atoms = _two_atoms()
    _assert_same(_array(atoms, box=(4.0, 3.0, 5.0), **GRID), _array(atoms, **GRID))


@ignore_strain_warning
def test_auto_sampling_and_slice_thickness_follow_the_box():
    atoms = _two_atoms()
    repeated = atoms * (2, 3, 2)
    in_box = abtem.Potential(
        atoms, box=(8.0, 9.0, 10.0), sampling="auto", slice_thickness="auto"
    )
    reference = abtem.Potential(repeated, sampling="auto", slice_thickness="auto")
    assert in_box.extent == pytest.approx((8.0, 9.0))
    assert in_box.gpts == reference.gpts
    assert in_box.slice_thickness == pytest.approx(reference.slice_thickness)


@ignore_strain_warning
def test_auto_sampling_follows_the_box_of_a_strained_cell():
    atoms = _two_atoms()
    box = (9.0, 9.0, 10.0)
    in_box = abtem.Potential(atoms, box=box, sampling="auto", slice_thickness="auto")
    reference = abtem.Potential(
        orthogonalize_cell(atoms, box=box), sampling="auto", slice_thickness="auto"
    )
    assert in_box.gpts == reference.gpts
    assert in_box.slice_thickness == pytest.approx(reference.slice_thickness)
    assert sum(in_box.slice_thickness) == pytest.approx(box[2])


@ignore_strain_warning
def test_auto_slice_thickness_follows_the_plane():
    atoms = _two_atoms()
    potential = abtem.Potential(atoms, plane="xz", sampling=0.1, slice_thickness="auto")
    reference = abtem.Potential(
        _permuted_two_atoms(), sampling=0.1, slice_thickness="auto"
    )
    assert potential.slice_thickness == pytest.approx(reference.slice_thickness)


@ignore_strain_warning
def test_auto_slice_thickness_of_a_primitive_fcc_cell_fills_its_box():
    # The primitive cell is non-orthogonal; the slices fill the best orthogonal
    # cell the potential is built in.
    atoms = bulk("Si", "diamond", a=5.431)
    potential = abtem.Potential(atoms, sampling=0.1, slice_thickness="auto")
    assert sum(potential.slice_thickness) == pytest.approx(potential.box[2])


@ignore_strain_warning
@pytest.mark.parametrize(
    "box", [(8.0, 9.0), (8.0, 0.0, 10.0), (8.0, -9.0, 10.0), (8.0, np.nan, 10.0), "abc"]
)
def test_invalid_box_raises(box):
    with pytest.raises(ValueError):
        abtem.Potential(_two_atoms(), box=box, **GRID)


@ignore_strain_warning
@pytest.mark.parametrize("sampling", [0.1, "auto"])
def test_periodic_box_with_no_whole_period_raises_at_construction(sampling):
    with pytest.raises(ValueError, match="no whole repetition"):
        abtem.Potential(_two_atoms(), box=(1.5, 3.0, 5.0), sampling=sampling)


@ignore_strain_warning
def test_whole_period_of_a_box_is_counted_in_the_frame_of_the_plane():
    # With plane="xz" the potential's axes are the cell's x, z, y (4, 5, 3 A), so
    # the 2 A along y holds no period of the 5 A axis it is laid over; in the
    # unrotated frame it holds a period of the 3 A axis.
    atoms = _two_atoms()
    box = (4.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="no whole repetition"):
        abtem.Potential(atoms, plane="xz", box=box, **GRID)
    assert abtem.Potential(atoms, box=box, **GRID).box == box


@float64_devices
@ignore_strain_warning
def test_non_periodic_box_with_no_whole_period_is_cut_out():
    atoms = _two_atoms()
    box = (1.5, 3.0, 5.0)
    potential = abtem.Potential(atoms, box=box, periodic=False, **GRID)
    array = potential.build(lazy=False).array
    assert array.shape == (5, 15, 30)
    _assert_same(array, _array(cut_cell(atoms, cell=box), **GRID))


@ignore_strain_warning
@pytest.mark.parametrize("box", [(1.5, 3.0, 5.0), (9.0, 9.0, 10.0), (8.0, 9.0, 10.0)])
def test_non_periodic_auto_grid_follows_the_atoms_cut_out_of_the_box(box):
    # The cut-out atoms are not strained, so the grid commensurate with them is
    # the one of the atoms cut out of the repeated structure.
    atoms = _two_atoms()
    potential = abtem.Potential(
        atoms, box=box, periodic=False, sampling="auto", slice_thickness="auto"
    )
    reference = abtem.Potential(
        cut_cell(atoms, cell=box), sampling="auto", slice_thickness="auto"
    )
    assert potential.gpts == reference.gpts
    assert potential.sampling == pytest.approx(reference.sampling)
    assert potential.slice_thickness == pytest.approx(reference.slice_thickness)


def _triclinic():
    # Lattice vectors with z components, so the default box strains z as well as
    # x and y.
    return Atoms(
        "SiO",
        scaled_positions=[(0.1, 0.2, 0.3), (0.6, 0.7, 0.85)],
        cell=[[4.0, 0.0, 0.4], [0.5, 3.0, 0.3], [0.6, 0.2, 5.0]],
        pbc=True,
    )


@pytest.mark.parametrize(
    "atoms, origin",
    [
        (graphene(formula="BN", a=2.5, vacuum=2.0) * (3, 1, 1), (0.0, 0.0, 0.0)),
        (graphene(formula="BN", a=2.5, vacuum=2.0) * (3, 1, 1), (0.7, 0.3, 0.2)),
        (_triclinic(), (0.0, 0.0, 0.0)),
    ],
    ids=["hBN-3x1x1", "hBN-3x1x1-origin", "triclinic"],
)
def test_non_periodic_auto_grid_follows_the_atoms_cut_out_of_the_default_box(
    atoms, origin
):
    # With no box, a non-periodic potential cuts its default box out of the
    # repeated structure, as it cuts a box it is given.
    potential = abtem.Potential(
        atoms, periodic=False, origin=origin, sampling="auto", slice_thickness="auto"
    )
    reference = abtem.Potential(
        cut_cell(atoms, cell=potential.box, origin=origin),
        sampling="auto",
        slice_thickness="auto",
    )
    assert (potential.gpts, len(potential.slice_thickness)) == (
        reference.gpts,
        len(reference.slice_thickness),
    )
    assert potential.sampling == pytest.approx(reference.sampling, rel=1e-12)
    assert potential.slice_thickness == pytest.approx(
        reference.slice_thickness, rel=0, abs=1e-12 * potential.box[2]
    )


@ignore_strain_warning
def test_invalid_origin_raises():
    for origin in [(1.0, 0.5), ("1", "0", "0"), (np.nan, 0.0, 0.0)]:
        with pytest.raises(ValueError, match="origin"):
            abtem.Potential(_two_atoms(), origin=origin, **GRID)


@float64_devices
@ignore_strain_warning
def test_origin_none_is_the_zero_origin():
    atoms = _two_atoms()
    _assert_same(_array(atoms, origin=None, **GRID), _array(atoms, **GRID))


@ignore_strain_warning
def test_box_of_strings_raises():
    with pytest.raises(ValueError, match="box"):
        abtem.Potential(_two_atoms(), box=("8", "9", "10"), **GRID)


@float64_devices
@ignore_strain_warning
def test_plane_box_and_origin_together():
    atoms = _two_atoms()
    origin = (1.0, 0.5, 0.25)
    # The origin translates the atoms as provided, then the plane maps y and z.
    translated = atoms.copy()
    translated.translate(-np.array(origin))
    translated.wrap()
    permuted = Atoms(
        translated.numbers,
        positions=translated.positions[:, [0, 2, 1]],
        cell=[4.0, 5.0, 3.0],
        pbc=True,
    )

    potential = abtem.Potential(
        atoms, plane="xz", box=(8.0, 10.0, 3.0), origin=origin, **GRID
    )
    assert potential.box == (8.0, 10.0, 3.0)
    _assert_same(
        potential.build(lazy=False).array, _array(permuted * (2, 2, 1), **GRID)
    )


@float64_devices
@ignore_strain_warning
def test_frozen_phonons_through_a_box():
    atoms = _two_atoms()
    frozen_phonons = FrozenPhonons(atoms, 3, sigmas=0.1, seed=1)
    potential = abtem.Potential(frozen_phonons, box=(8.0, 9.0, 10.0), **GRID)

    eager = asnumpy(potential.build(lazy=False).array)
    lazy = potential.build(lazy=True).compute().array

    assert eager.shape == (3, 10, 80, 90)
    _assert_same(lazy, eager)
    assert not np.allclose(eager[0], eager[1])

    # Each configuration holds the 12 cells of the box, displaced.
    cell = abtem.Potential(atoms, **GRID)
    for configuration in eager:
        integral = configuration.sum() * np.prod(potential.sampling)
        assert integral / _integral(cell) == pytest.approx(12, rel=1e-9)


@float64_devices
@ignore_strain_warning
def test_crystal_potential_of_a_unit_with_a_box():
    atoms = _two_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0)
    unit = abtem.Potential(atoms, box=(8.0, 9.0, 10.0), **kwargs)
    crystal = abtem.CrystalPotential(unit, repetitions=(2, 1, 1))
    reference = abtem.Potential(atoms * (4, 3, 2), **kwargs)

    assert crystal.box == pytest.approx((16.0, 9.0, 10.0))
    _assert_same(crystal.build(lazy=False).array, reference.build(lazy=False).array)


@ignore_strain_warning
def test_non_periodic_box_is_cut_out_of_the_repeated_atoms():
    potential = abtem.Potential(
        _two_atoms(),
        box=(8.0, 9.0, 10.0),
        periodic=False,
        projection="finite",
        **GRID,
    )
    assert potential.box == (8.0, 9.0, 10.0)
    assert potential.get_transformed_atoms().cell.lengths() == pytest.approx(
        (8.0, 9.0, 10.0)
    )


@float64_devices
@ignore_strain_warning
def test_multislice_through_a_box_matches_the_repeated_atoms():
    atoms = _two_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0)
    in_box = abtem.Potential(atoms, box=(8.0, 9.0, 10.0), **kwargs)
    repeated = abtem.Potential(atoms * (2, 3, 2), **kwargs)
    wave = abtem.PlaneWave(energy=100e3)
    _assert_same(
        wave.multislice(in_box, lazy=False).array,
        wave.multislice(repeated, lazy=False).array,
    )


@float64_devices
@ignore_strain_warning
def test_magnetic_field_box_matches_repeated_atoms(device):
    atoms = _two_atoms()
    atoms.set_chemical_symbols(["Fe", "O"])
    atoms.set_array("magnetic_moments", np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
    kwargs = dict(sampling=0.2, slice_thickness=1.0)
    _assert_same(
        MagneticField(atoms, box=(8.0, 9.0, 10.0), **kwargs).build(lazy=False).array,
        MagneticField(atoms * (2, 3, 2), **kwargs).build(lazy=False).array,
    )


@float64_devices
@ignore_strain_warning
@pytest.mark.parametrize("stored", [None, [0.0, 0.0, 0.0], np.zeros(3)])
def test_potential_restored_with_the_origin_as_it_was_passed_builds(stored):
    # A potential restored from a pickle skips `__init__`, so it may hold the
    # origin exactly as the user passed it.
    atoms = _two_atoms()
    potential = abtem.Potential(atoms, **GRID)
    restored = pickle.loads(pickle.dumps(potential))
    restored._origin = stored

    _assert_same(restored.build(lazy=False).array, _array(atoms, **GRID))


@float64_devices
@ignore_strain_warning
def test_potential_restored_with_an_invalid_origin_raises_on_build():
    restored = pickle.loads(pickle.dumps(abtem.Potential(_two_atoms(), **GRID)))
    restored._origin = (1.0, 0.0)
    with pytest.raises(ValueError, match="origin"):
        restored.build(lazy=False)


def _strain_warnings(records):
    return [r for r in records if str(r.message).startswith("The box")]


def _construct(*args, **kwargs):
    """The potential, and the box-strain warnings its construction gave."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = abtem.Potential(*args, **kwargs)
    return potential, _strain_warnings(records)


def _strain_from_the_transform(atoms, box):
    """Stretch of each supercell vector onto the box, and the cosines of the angles
    between them, from the affine map `orthogonalize_cell` applies: it takes the
    supercell vectors v to the box edges, v @ A = diag(box)."""
    _, transform = orthogonalize_cell(atoms, box=box, return_transform_matrix=True)
    supercell = np.diag(box) @ np.linalg.inv(transform)
    lengths = np.linalg.norm(supercell, axis=1)
    unit = supercell / lengths[:, None]
    cosines = [unit[1] @ unit[2], unit[0] @ unit[2], unit[0] @ unit[1]]
    return np.asarray(box) / lengths - 1.0, np.array(cosines), lengths


@ignore_strain_warning
def test_strain_warning_threshold_is_a_tenth_of_a_percent():
    assert abtem.atoms.BOX_STRAIN_WARNING_THRESHOLD == 1e-3


@ignore_strain_warning
def test_box_that_strains_the_atoms_warns_with_the_numbers():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    potential, records = _construct(si, box=(20.0, 5.431, 5.431), sampling=0.2)

    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)
    assert records[0].filename == __file__
    message = str(records[0].message)
    # 4 periods of 5.431 A make 21.724 A; 20 A compresses them.
    stretch = 100 * (20.0 / (4 * 5.431) - 1.0)
    assert f"{stretch:+.3f} %" in message
    assert "-7.936 %" in message
    assert "+0.000 %" in message
    assert "(4, 1, 1) periods" in message
    assert "21.724" in message
    assert "20.0" in message
    assert potential.box == (20.0, 5.431, 5.431)


@ignore_strain_warning
@pytest.mark.parametrize(
    "factor, warns",
    [(1.0011, True), (0.9989, True), (1.0009, False), (0.9991, False), (1.0, False)],
)
def test_strain_warning_threshold_applies_to_the_stretch(factor, warns):
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    box = (4 * 5.431 * factor, 5.431, 5.431)
    _, records = _construct(si, box=box, sampling=0.2)
    assert bool(records) == warns
    if warns:
        assert f"{100 * (factor - 1):+.3f} %" in str(records[0].message)


@ignore_strain_warning
def test_a_slightly_off_box_of_four_periods_is_silent():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    _, records = _construct(si, box=(21.72, 5.431, 5.431), sampling=0.2)
    assert records == []


@ignore_strain_warning
def test_strain_warning_quotes_the_shear_of_a_hexagonal_supercell():
    graphene_cell = graphene(vacuum=2.0)
    box = (20.0, 20.0, 4.0)
    _, records = _construct(graphene_cell, box=box, sampling=0.2)

    assert len(records) == 1
    message = str(records[0].message)
    stretch, cosines, lengths = _strain_from_the_transform(graphene_cell, box)
    assert cosines[2] == pytest.approx(0.064, abs=1e-3)
    for value in stretch:
        assert f"{100 * value:+.3f} %" in message
    for value in cosines:
        assert f"{value:.2e}" in message
    for value in np.degrees(np.arccos(cosines)):
        assert f"{value:.3f}°" in message
    assert str(round(float(lengths[0]), 6)) in message
    assert "[[8, 0, 0], [5, 9, 0], [0, 0, 1]]" in message


@ignore_strain_warning
def test_boxes_that_are_whole_supercells_up_to_round_off_are_silent():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    hexagonal = graphene(vacuum=2.0)
    a = 2.46
    cases = [(si, (n * 5.431, m * 5.431, 5.431)) for n in (1, 2, 3, 7) for m in (1, 3)]
    # Boxes of hexagonal supercells computed in floating point.
    cases += [
        (hexagonal, (n * a, m * a * np.sqrt(3.0), 4.0))
        for n in (1, 2, 3, 5)
        for m in (1, 2, 3)
    ]
    cases += [
        (hexagonal, (n * a, m * 3.0 * a / np.sqrt(3.0), 4.0))
        for n in (2, 4)
        for m in (1, 3)
    ]
    for atoms, box in cases:
        _, records = _construct(atoms, box=box, sampling=0.5)
        assert records == [], (box, [str(r.message)[:80] for r in records])


@ignore_strain_warning
def test_exact_default_box_of_a_non_orthogonal_cell_is_silent():
    _, records = _construct(bulk("Si", "diamond", a=5.431), sampling=0.2)
    assert records == []


@ignore_strain_warning
def test_default_box_given_explicitly_is_not_checked():
    # The default box of this supercell is itself reached by a strain of 1.4 %,
    # 0.7 % and a shear of 0.11; it is the default, however it is spelled.
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0) * (3, 1, 1)
    default = abtem.Potential(atoms, sampling=0.2).box
    assert abs(100 * (default[0] / 7.5 - 1)) > 0.5

    _, records = _construct(atoms, box=default, sampling=0.2)
    assert records == []

    _, records = _construct(atoms, box=(default[0] * 1.01, default[1], 4.0))
    assert len(records) == 1


@ignore_strain_warning
def test_non_periodic_box_is_not_strained_and_is_silent():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    _, records = _construct(si, box=(20.0, 5.431, 5.431), periodic=False, sampling=0.2)
    assert records == []


@float64_devices
@ignore_strain_warning
@pytest.mark.parametrize("builder", [abtem.Potential, MagneticField])
def test_strain_warning_is_given_once_per_construction(builder, device):
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    si.set_array("magnetic_moments", np.zeros((len(si), 3)))
    kwargs = dict(box=(20.0, 5.431, 5.431), sampling=0.2, slice_thickness=2.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = builder(si, **kwargs)
        potential.copy()
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_strain_warnings(records)) == 1


@float64_devices
@ignore_strain_warning
def test_strain_warning_is_not_repeated_by_frozen_phonons_or_lazy_blocks():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    frozen_phonons = FrozenPhonons(si, 3, sigmas=0.05, seed=1)
    kwargs = dict(box=(20.0, 5.431, 5.431), sampling=0.2, slice_thickness=2.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = abtem.Potential(frozen_phonons, **kwargs)
        assert len(_strain_warnings(records)) == 1
        lazy = potential.build(lazy=True).compute()
        eager = potential.build(lazy=False)
        wave = abtem.PlaneWave(energy=100e3)
        wave.multislice(potential, lazy=True).compute()
    assert len(_strain_warnings(records)) == 1
    assert lazy.array.shape == eager.array.shape
    assert lazy.array.shape[:3] == (3, 3, 100)


@ignore_strain_warning
def test_strain_warning_does_not_hide_an_error_for_a_box_with_no_whole_period():
    with pytest.raises(ValueError, match="no whole repetition"):
        abtem.Potential(_two_atoms(), box=(1.5, 3.0, 5.0), **GRID)


@float64_devices
@ignore_strain_warning
def test_strain_warning_is_not_repeated_by_a_crystal_potential():
    # CrystalPotential rebuilds its unit with its own frozen-phonon pool per
    # member and per enlarged pool; the unit's box was reported when it was made.
    two = _two_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        unit = abtem.Potential(
            FrozenPhonons(two, 2, sigmas=0.1, seed=1), box=(9.0, 9.0, 10.0), **kwargs
        )
        assert len(_strain_warnings(records)) == 1

        # The pool (2) is smaller than the 4 lateral tiles and is enlarged.
        tiled = abtem.CrystalPotential(unit, (2, 2, 1))
        tiled.build(lazy=False)
        tiled.build(lazy=True).compute()

        # An ensemble of members, each with its own pool.
        members = abtem.CrystalPotential(
            unit, (1, 1, 2), num_frozen_phonons=2, seeds=(5, 6)
        )
        members.build(lazy=False)
        members.build(lazy=True).compute()

    assert len(_strain_warnings(records)) == 1


@float64_devices
@ignore_strain_warning
@pytest.mark.parametrize("repetitions, reported", [((2, 3, 2), 0), ((3, 1, 1), 1)])
def test_charge_density_potential_reports_its_default_box_once(repetitions, reported):
    # The default box of BN x (3, 1, 1) is reached by a strain; the potential
    # reports it when it is constructed, and the Ewald potential it builds from
    # its own box does not repeat it.
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = ChargeDensityPotential(
            atoms,
            _charge_density(),
            sampling=0.2,
            slice_thickness=1.0,
            repetitions=repetitions,
        )
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_strain_warnings(records)) == reported


# The box that abTEM picks itself is reported when it strains the atoms.


STRAIN_GRID = dict(sampling=0.2, slice_thickness=1.0)


# Hexagonal BN repeated along x, as (nx, ny, 1), with the default box its
# repetitions give: exact for (1, 1, 1) and (2, 1, 1), strained for the others.
STRAINED = [(3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1)]


EXACT = [(1, 1, 1), (2, 1, 1), (3, 2, 1), (2, 3, 1), (4, 2, 1), (5, 2, 1), (3, 3, 1)]


def _bn(repetitions=(1, 1, 1)):
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    return atoms * repetitions


def _chosen_box_warnings(records):
    return [r for r in records if "abTEM chose" in str(r.message)]


def _construct_chosen(builder, *args, **kwargs):
    """The builder, and the warnings about the box abTEM chose that its
    construction gave."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        built = builder(*args, **kwargs)
    return built, _chosen_box_warnings(records)


@ignore_strain_warning
def test_default_box_of_bn_repeated_three_times_warns_with_the_numbers():
    atoms = _bn((3, 1, 1))
    potential, records = _construct_chosen(abtem.Potential, atoms, **STRAIN_GRID)

    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)
    assert records[0].filename == __file__
    message = str(records[0].message)
    stretch, cosines, lengths = _strain_from_the_transform(atoms, potential.box)
    assert 100 * stretch == pytest.approx([1.379, -0.660, 0.0], abs=1e-3)
    assert cosines[2] == pytest.approx(0.115, abs=1e-3)
    for value in stretch:
        assert f"{100 * value:+.3f} %" in message
    for value in cosines:
        assert f"{value:.2e}" in message
    for value in np.degrees(np.arccos(cosines)):
        assert f"{value:.3f}°" in message
    assert str(round(float(lengths[0]), 6)) in message
    assert "[[1, 0, 0], [1, 5, 0], [0, 0, 1]]" in message
    assert str(tuple(float(b) for b in potential.box)) in message


@ignore_strain_warning
def test_warning_says_who_chose_the_box_and_names_the_remedies():
    _, records = _construct_chosen(abtem.Potential, _bn((3, 1, 1)), **STRAIN_GRID)

    message = str(records[0].message)
    assert "abTEM chose because none was given" in message
    assert "Pass a `box` that is a whole supercell" in message
    assert "repeat the atoms' cell so that an orthogonal supercell" in message
    assert "at most 5 repetitions" in message
    assert f"{abtem.atoms.BOX_STRAIN_WARNING_THRESHOLD:.1e}" in message


@ignore_strain_warning
def test_default_box_of_bn_repeated_five_times_warns():
    atoms = _bn((5, 1, 1))
    potential, records = _construct_chosen(abtem.Potential, atoms, **STRAIN_GRID)

    assert len(records) == 1
    stretch, _, _ = _strain_from_the_transform(atoms, potential.box)
    assert 100 * stretch[1] == pytest.approx(-13.397, abs=1e-3)
    assert f"{100 * stretch[1]:+.3f} %" in str(records[0].message)


@ignore_strain_warning
@pytest.mark.parametrize("repetitions", STRAINED)
def test_strained_default_box_warns_once(repetitions):
    _, records = _construct_chosen(abtem.Potential, _bn(repetitions), **STRAIN_GRID)
    assert len(records) == 1


@ignore_strain_warning
@pytest.mark.parametrize("repetitions", EXACT)
def test_exact_default_box_is_silent(repetitions):
    # (3, 2, 1), (2, 3, 1), ...: different repetitions along x and y whose
    # lattice vectors still make an exact orthogonal supercell.
    _, records = _construct_chosen(abtem.Potential, _bn(repetitions), **STRAIN_GRID)
    assert records == []


@ignore_strain_warning
def test_default_box_with_different_repetitions_along_x_and_y_is_judged_by_the_cell():
    _, strained = _construct_chosen(abtem.Potential, _bn((4, 1, 1)), **STRAIN_GRID)
    _, exact = _construct_chosen(abtem.Potential, _bn((3, 2, 1)), **STRAIN_GRID)
    assert len(strained) == 1
    assert exact == []


@ignore_strain_warning
def test_non_periodic_default_box_is_silent():
    for repetitions in STRAINED:
        _, records = _construct_chosen(
            abtem.Potential, _bn(repetitions), periodic=False, **STRAIN_GRID
        )
        assert records == []


@ignore_strain_warning
def test_box_given_by_the_user_is_not_reported_as_chosen_by_abtem():
    atoms = _bn((3, 1, 1))
    default = abtem.Potential(atoms, **STRAIN_GRID).box
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        abtem.Potential(atoms, box=(default[0] * 1.01, default[1], 4.0), **STRAIN_GRID)
    assert _chosen_box_warnings(records) == []
    assert [r for r in records if str(r.message).startswith("The box")]


@float64_devices
@ignore_strain_warning
def test_warning_is_given_once_by_a_frozen_phonon_potential_and_its_blocks():
    frozen_phonons = abtem.FrozenPhonons(
        _bn((3, 1, 1)), num_configs=2, sigmas=0.05, seed=1
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = abtem.Potential(frozen_phonons, **STRAIN_GRID)
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_chosen_box_warnings(records)) == 1


@ignore_strain_warning
def test_orthogonalize_cell_without_a_box_does_not_warn():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        orthogonalize_cell(_bn((3, 1, 1)))
    assert _chosen_box_warnings(records) == []
    assert not [r for r in records if "strained" in str(r.message)]


_GPAW_FAMILY = [
    pytest.param(
        lambda atoms: ChargeDensityPotential(
            atoms, _charge_density(), sampling=0.2, slice_thickness=1.0
        ),
        id="charge-density",
    ),
    pytest.param(
        lambda atoms: GPAWMagneticField(
            _FakeCalculator(atoms), sampling=0.2, slice_thickness=1.0
        ),
        id="magnetic-field",
    ),
    pytest.param(
        lambda atoms: GPAWVectorPotential(
            _FakeCalculator(atoms), sampling=0.2, slice_thickness=1.0
        ),
        id="vector-potential",
    ),
]


@ignore_strain_warning
@pytest.mark.parametrize("build", _GPAW_FAMILY)
@pytest.mark.parametrize("repetitions", [(3, 1, 1), (5, 1, 1), (4, 1, 1)])
def test_gpaw_family_warns_for_a_strained_default_box(build, repetitions):
    _, records = _construct_chosen(build, _bn(repetitions))
    assert len(records) == 1


@ignore_strain_warning
@pytest.mark.parametrize("build", _GPAW_FAMILY)
@pytest.mark.parametrize("repetitions", [(1, 1, 1), (2, 1, 1), (3, 2, 1)])
def test_gpaw_family_is_silent_for_an_exact_default_box(build, repetitions):
    _, records = _construct_chosen(build, _bn(repetitions))
    assert records == []


@ignore_strain_warning
@pytest.mark.parametrize("repetitions", [(3, 1, 1), (4, 1, 1)])
def test_charge_density_repetitions_are_judged_by_the_repeated_cell(repetitions):
    _, records = _construct_chosen(
        ChargeDensityPotential,
        _bn(),
        _charge_density(),
        sampling=0.2,
        slice_thickness=1.0,
        repetitions=repetitions,
    )
    assert len(records) == 1
    assert "[[1, 0, 0], [1, " in str(records[0].message)


@float64_devices
@ignore_strain_warning
def test_charge_density_does_not_repeat_the_warning_when_it_builds_the_ewald_field():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = ChargeDensityPotential(
            _bn(),
            _charge_density(),
            sampling=0.2,
            slice_thickness=1.0,
            repetitions=(3, 1, 1),
        )
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_chosen_box_warnings(records)) == 1


@ignore_strain_warning
@pytest.mark.parametrize("plane", ["xz", "yz"])
@pytest.mark.parametrize(
    "build",
    [
        lambda atoms, plane: abtem.Potential(atoms, plane=plane, **STRAIN_GRID),
        lambda atoms, plane: abtem.Potential(
            atoms, plane=plane, periodic=False, box=(5.0, 5.0, 5.0), **STRAIN_GRID
        ),
        lambda atoms, plane: ChargeDensityPotential(
            atoms, _charge_density(), plane=plane, **STRAIN_GRID
        ),
        lambda atoms, plane: GPAWMagneticField(
            _FakeCalculator(atoms), plane=plane, **STRAIN_GRID
        ),
    ],
    ids=["potential", "non-periodic-box", "charge-density", "gpaw-magnetic-field"],
)
def test_cell_that_cannot_be_rotated_to_the_plane_raises_at_construction(build, plane):
    # The hexagonal cell has no lattice vector along y, the beam direction of
    # "xz", and is not orthogonal once rotated to "yz". Building it in either
    # plane raises, so constructing it does.
    with pytest.raises(RuntimeError, match=f"cannot be rotated to plane='{plane}'"):
        build(_bn(), plane)


@ignore_strain_warning
@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_orthogonalized_cell_can_be_rotated_to_the_plane(plane):
    potential = abtem.Potential(orthogonalize_cell(_bn()), plane=plane, **STRAIN_GRID)
    assert potential.build(lazy=False).shape[0] == len(potential)


@ignore_strain_warning
@pytest.mark.parametrize(
    "build",
    [
        lambda atoms, box: ChargeDensityPotential(
            atoms, _charge_density(), box=box, sampling=0.2, slice_thickness=1.0
        ),
        lambda atoms, box: GPAWMagneticField(
            _FakeCalculator(atoms), box=box, sampling=0.2, slice_thickness=1.0
        ),
        lambda atoms, box: GPAWVectorPotential(
            _FakeCalculator(atoms), box=box, sampling=0.2, slice_thickness=1.0
        ),
    ],
    ids=["charge-density", "magnetic-field", "vector-potential"],
)
def test_gpaw_family_does_not_report_its_default_box_when_it_is_given(build):
    atoms = _bn((3, 1, 1))
    default = abtem.Potential(atoms, **STRAIN_GRID).box
    _, records = _construct_chosen(build, atoms, default)
    assert records == []


# sampling="auto" targets 0.05 A on the extent the potential really has.


NON_PERIODIC_XY = (False, False, True)


def _co(pbc):
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=pbc,
    )


def _bn_with_pbc(pbc):
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = pbc
    return atoms


def _ensemble(atoms):
    other = atoms.copy()
    other.positions[0] += 0.1
    return abtem.AtomsEnsemble([atoms, other])


AUTO_GRID_CASES = [
    ("CO, pbc (F, F, T), plane xy", lambda: _co(NON_PERIODIC_XY), {}),
    ("CO, pbc (F, F, T), plane xz", lambda: _co(NON_PERIODIC_XY), dict(plane="xz")),
    ("CO, pbc (F, F, T), plane yz", lambda: _co(NON_PERIODIC_XY), dict(plane="yz")),
    ("CO, pbc (T, F, T), plane xz", lambda: _co((True, False, True)), dict(plane="xz")),
    ("BN, pbc (F, F, T), plane xy", lambda: _bn_with_pbc(NON_PERIODIC_XY), {}),
    ("CO, pbc (F, F, T), box", lambda: _co(NON_PERIODIC_XY), dict(box=(8.0, 9.0, 10.0))),
    ("CO, pbc (F, F, T), origin", lambda: _co(NON_PERIODIC_XY), dict(origin=(1.0, 0.0, 0.0))),
    (
        "AtomsEnsemble of CO, plane xz",
        lambda: _ensemble(_co(True)),
        dict(plane="xz"),
    ),
    ("AtomsEnsemble of BN", lambda: _ensemble(_bn_with_pbc(True)), {}),
]


@pytest.mark.parametrize("case", AUTO_GRID_CASES, ids=lambda c: c[0])
def test_auto_grid_of_non_periodic_atoms_follows_the_extent(case):
    name, make_atoms, kwargs = case
    potential = abtem.Potential(make_atoms(), sampling="auto", **kwargs)

    # The grid the target gives for the extent of the potential, which is the
    # box, or the cell rotated to the plane and made orthogonal.
    expected = tuple(int(np.ceil(e / 0.05)) for e in potential.extent)
    if round_auto_derived_gpts():
        expected = tuple(next_fast_fft_size(n) for n in expected)
    assert potential.gpts == expected
    assert max(potential.sampling) <= 0.05 + 1e-12


def test_auto_grid_of_non_periodic_atoms_with_a_box_is_unchanged():
    # The box was already the extent of this branch.
    potential = abtem.Potential(
        _co(NON_PERIODIC_XY), sampling="auto", box=(8.0, 9.0, 10.0), periodic=False
    )
    assert potential.gpts == (160, 180)


@pytest.mark.parametrize(
    "pbc, plane, gpts",
    [(NON_PERIODIC_XY, "xy", (80, 60)), (True, "xy", (80, 60)), (True, "xz", (80, 100))],
)
def test_auto_grid_that_was_right_is_unchanged(pbc, plane, gpts):
    potential = abtem.Potential(_co(pbc), sampling="auto", plane=plane)
    assert potential.gpts == gpts
