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
    # at the float precision floor (max over all above-threshold pixels)
    rel_FP = np.abs(IF[mask] - IP[mask]) / IF[mask]
    assert rel_FP.max() < 1e-4, (
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
