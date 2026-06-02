import hypothesis.strategies as st
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


def test_non_orthogonal_requires_z_separable():
    """A fully triclinic cell (z not perpendicular to xy) is rejected."""
    import numpy as np
    from ase import Atoms

    cell = np.array([[4.0, 0, 0], [1.0, 4.0, 0], [0.5, 0.0, 6.0]])  # z-x coupling
    atoms = Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)])
    with pytest.raises(NotImplementedError):
        Potential(atoms, gpts=(40, 40), slice_thickness=6.0, non_orthogonal=True)


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


def test_waves_cell_survives_metadata_roundtrip():
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
