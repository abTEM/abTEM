import warnings

import dask.array as da
import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from ase.build import bulk, graphene
from hypothesis import assume, given
from hypothesis import strategies as st
from utils import gpu

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.dynamical import BlochwaveEnsemble, calculate_structure_factors
from abtem.bloch.utils import (
    auto_detect_centering,
    get_reflection_condition,
    relative_positions_for_centering,
    wrapped_is_close,
)
from abtem.core.backend import cp
from abtem.core.fft import fft_interpolate
from abtem.parametrizations import LobatoParametrization


@st.composite
def basis_and_positions(draw):
    length = draw(st.integers(min_value=1, max_value=5))
    basis_numbers = draw(
        st.lists(
            st.integers(min_value=1, max_value=5), min_size=length, max_size=length
        )
    )
    basis = draw(
        st.lists(
            st.tuples(st.floats(0.1, 1), st.floats(0.1, 1), st.floats(0.1, 1)),
            min_size=length,
            max_size=length,
        )
    )
    return np.array(basis_numbers), np.array(basis)


def basis_match_template(basis, template):
    n = len(basis) / len(template)
    if not np.isclose(n, np.round(n), atol=1e-6):
        return False

    shifted_basis = (basis - basis[0]) % 1.0
    is_close = wrapped_is_close(shifted_basis, template)
    return is_close.all(-1).any(axis=1).all()


def basis_match_templates(basis):
    template_bases = relative_positions_for_centering()
    bases_to_check = set(template_bases.keys())
    for centering, template_basis in template_bases.items():
        if not basis_match_template(basis, template_basis):
            bases_to_check.remove(centering)
    return bases_to_check


@pytest.mark.parametrize("centering", ["P", "F", "I", "A", "B", "C"])
@pytest.mark.filterwarnings("ignore:Something went wrong with the centering detection")
@given(
    data=basis_and_positions(),
    cell=st.tuples(st.floats(1, 2), st.floats(1, 2), st.floats(1, 2)),
)
def test_auto_detect_centering(data, cell, centering):
    basis_numbers, basis = data

    lattice = np.array(relative_positions_for_centering()[centering])

    positions = (lattice[:, None] + basis[None]).reshape((-1, 3))
    numbers = np.tile(basis_numbers, len(lattice))

    assume(np.isclose(basis[:, None], basis[None]).sum(-1).sum() == len(basis) * 3)

    atoms = Atoms(numbers, positions=positions, cell=(1, 1, 1), pbc=True)
    atoms.set_cell(cell, scale_atoms=True)

    assume(centering != "P" or basis_match_templates(basis) == {"P"})
    assert auto_detect_centering(atoms) == centering


REFLECTION_CONDITIONS = {
    "P": lambda h, k, l: np.ones_like(h, dtype=bool),
    "A": lambda h, k, l: (k + l) % 2 == 0,
    "B": lambda h, k, l: (h + l) % 2 == 0,
    "C": lambda h, k, l: (h + k) % 2 == 0,
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: (h % 2 == k % 2) & (k % 2 == l % 2),
}


def _hkl_cube(n=3):
    hkl = np.stack(
        np.meshgrid(*(np.arange(-n, n + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    return hkl[(hkl != 0).any(axis=1)]


@pytest.mark.parametrize("centering", ["P", "A", "B", "C", "I", "F"])
def test_reflection_condition_matches_crystallographic_rules(centering):
    # Regression: the A, B and C branches applied .all(axis=1) to a 1-D mask
    # and raised numpy.exceptions.AxisError.
    hkl = _hkl_cube()
    mask = get_reflection_condition(hkl, centering)
    assert mask.shape == (len(hkl),)
    np.testing.assert_array_equal(mask, REFLECTION_CONDITIONS[centering](*hkl.T))


@pytest.mark.parametrize("centering", ["A", "B", "C", "I", "F"])
def test_centering_templates_are_consistent_with_reflection_conditions(centering):
    # A crystal built from the centering translations returned by
    # relative_positions_for_centering must be auto-detected as that centering, and
    # exactly the reflections removed by get_reflection_condition must be
    # systematically absent. The A, B and C templates previously encoded
    # half-translations along a single lattice vector, so the reflection
    # conditions would have removed allowed reflections.
    basis = np.array([[0.1, 0.2, 0.3], [0.27, 0.13, 0.41]])
    lattice = relative_positions_for_centering()[centering]
    scaled_positions = (lattice[:, None] + basis[None]).reshape((-1, 3)) % 1.0
    atoms = Atoms(
        np.tile([6, 8], len(lattice)),
        scaled_positions=scaled_positions,
        cell=[3.1, 3.7, 4.3],
        pbc=True,
    )
    assert auto_detect_centering(atoms) == centering

    hkl = _hkl_cube()
    F = calculate_structure_factors(
        hkl,
        atoms,
        parametrization="lobato",
        g_max=10.0,
        thermal_sigma=0.0,
        occupancy=1.0,
        device="cpu",
    )
    allowed = get_reflection_condition(hkl, centering)
    assert np.abs(F[~allowed]).max() < 1e-6 * np.abs(F).max()
    assert (np.abs(F[allowed]) > 1e-4 * np.abs(F).max()).all()

    structure_factor = StructureFactor(atoms, g_max=2.0)
    assert structure_factor.centering == centering
    structure_factor_p = StructureFactor(atoms, g_max=2.0, centering="P")
    expected = structure_factor_p.hkl[
        get_reflection_condition(structure_factor_p.hkl, centering)
    ]
    np.testing.assert_array_equal(structure_factor.hkl, expected)


def test_auto_detect_centering_prefers_face_centering():
    # F-centering includes the translations of A-, B- and C-centering; they are
    # not additional centerings of the cell.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert auto_detect_centering(bulk("Si", cubic=True)) == "F"


def test_structure_factor_auto_centering_keeps_allowed_reflections():
    # Regression: graphene repeated along c used to be auto-detected as "C" (from a
    # (0, 0, 1/2) translation) and then crashed in get_reflection_condition; with
    # only the crash fixed, h + k odd reflections such as (1, 0, 0) would have been
    # silently dropped.
    atoms = graphene(vacuum=2.0)
    atoms.pbc = True
    atoms = atoms.repeat((1, 1, 20))

    structure_factor = StructureFactor(atoms, g_max=2.0)
    structure_factor_p = StructureFactor(atoms, g_max=2.0, centering="P")

    F = np.asarray(structure_factor_p.build(lazy=False).array)
    kept = (structure_factor_p.hkl[:, None] == structure_factor.hkl[None]).all(-1)
    kept = kept.any(axis=1)
    assert np.abs(F[~kept]).max(initial=0.0) < 1e-6 * np.abs(F).max()
    assert [1, 0, 0] in structure_factor.hkl.tolist()


@pytest.mark.parametrize("built_lazy", [True, False], ids=["lazy", "eager"])
def test_structure_factor_array_projected_potential_honors_lazy(built_lazy):
    # Regression: StructureFactorArray.get_projected_potential ignored its lazy
    # argument and followed self.is_lazy.
    structure_factor = StructureFactor(bulk("Si", cubic=True), g_max=4.0).build(
        lazy=built_lazy
    )

    potentials = {
        lazy: structure_factor.get_projected_potential(slice_thickness=1.0, lazy=lazy)
        for lazy in (True, False, None)
    }

    assert potentials[True].is_lazy
    assert not potentials[False].is_lazy
    assert potentials[None].is_lazy == built_lazy
    assert structure_factor.is_lazy == built_lazy

    expected = potentials[False].array
    for potential in potentials.values():
        assert potential.slice_thickness == potentials[False].slice_thickness
        np.testing.assert_allclose(potential.compute().array, expected, atol=1e-6)


@given(
    atoms=abtem_st.atoms(min_thickness=1.0, max_atomic_number=20),
    sampling=abtem_st.sampling(min_value=0.02, max_value=0.1),
    thermal_sigma=st.floats(min_value=0.03, max_value=0.1),
    g_max=st.floats(min_value=8, max_value=16),
    slice_thickness=st.floats(min_value=1, max_value=2.0),
)
@pytest.mark.filterwarnings("ignore:Something went wrong with the centering detection")
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_potential_from_structure_factor(
    atoms, sampling, thermal_sigma, g_max, slice_thickness, lazy
):
    structure_factor = abtem.StructureFactor(
        atoms, g_max=g_max, thermal_sigma=thermal_sigma
    )

    structure_factor_potential = structure_factor.get_projected_potential(
        slice_thickness=slice_thickness, sampling=sampling, lazy=lazy
    )

    parametrization = abtem.parametrizations.LobatoParametrization(sigmas=thermal_sigma)

    # Compare the two on the requested grid, band-limited the same way:
    #
    # - The structure-factor potential there is the exact potential, Fourier
    #   cropped to the grid. Potential's infinite projection is not: it
    #   deposits each atom bilinearly onto the four nearest grid points and
    #   only divides out the average (sinc) response of that, an inexact
    #   sub-pixel shift whose error grows with the sampling relative to the
    #   width of the thermally smeared atomic potential (up to ~5% at 0.1 Å
    #   with the sharpest potentials here, ~0.05% with the atoms on grid
    #   points). So the reference is built on a grid fine enough for that to
    #   be negligible and Fourier cropped to the requested grid.
    # - The structure factors stop at g_max (a steep taper at 0.95 g_max), but
    #   a fine grid holds frequencies far beyond it (Nyquist 25 1/Å at
    #   0.02 Å), which the reference keeps: up to ~2.5% with the sharpest
    #   potentials and the smallest g_max. So both are low-passed to within
    #   g_max, below the taper.
    #
    # They then agree to ~0.1% at worst (over 1500 examples), against up to
    # ~5% without either step.
    potential = abtem.Potential(
        atoms,
        sampling=0.01,
        slice_thickness=structure_factor_potential.slice_thickness,
        parametrization=parametrization,
        projection="infinite",
    )

    structure_factor_potential = structure_factor_potential.project().compute()
    potential = potential.build(lazy=lazy).project().compute()

    # sampling is honoured (to within fitting the grid to the cell)
    assert np.allclose(structure_factor_potential.sampling, sampling, rtol=0.05)

    array1 = structure_factor_potential.array
    array2 = fft_interpolate(potential.array, array1.shape).real

    grid_sampling = structure_factor_potential.sampling
    kx, ky = (np.fft.fftfreq(n, d) for n, d in zip(array1.shape, grid_sampling))
    within = kx[:, None] ** 2 + ky[None] ** 2 <= (0.9 * g_max) ** 2
    array1 = np.fft.ifft2(np.fft.fft2(array1) * within).real
    array2 = np.fft.ifft2(np.fft.fft2(array2) * within).real
    array1 -= array1.min()
    array2 -= array2.min()

    error = np.abs(array2 - array1).sum() / array1.sum() * 100
    assert error < 0.5


def test_structure_matrix_with_read_only_structure_factor_values(monkeypatch):
    # recent pandas versions return read-only arrays from Series.to_numpy()
    from ase.build import bulk

    from abtem.bloch import dynamical
    from abtem.bloch.utils import retrieve_structure_factor_values

    def read_only_values(*args, **kwargs):
        values = retrieve_structure_factor_values(*args, **kwargs)
        values.flags.writeable = False
        return values

    structure_factor = abtem.StructureFactor(bulk("Si", cubic=True), g_max=3)
    bloch_waves = abtem.BlochWaves(structure_factor, energy=100e3, sg_max=0.1)
    expected = bloch_waves.calculate_structure_matrix(lazy=False)

    monkeypatch.setattr(dynamical, "retrieve_structure_factor_values", read_only_values)
    structure_matrix = bloch_waves.calculate_structure_matrix(lazy=False)

    assert np.allclose(structure_matrix, expected)


def test_structure_factor_matches_standard_crystallographic_sign_convention():
    # calculate_structure_factors must compute F(g) = sum_j f_j(g) * exp(+2pi i
    # g . r_j) / V, the standard International-Tables convention. abTEM previously
    # computed the complex conjugate (exp(-2pi i ...)), which structure_factor_to_
    # potential's ifftn happened to cancel when reconstructing a real-space potential
    # (see test_potential_from_structure_factor), but calculate_structure_matrix
    # consumes F(g) directly with no such cancellation -- so the wrong sign there
    # gave Bloch-wave results inconsistent with multislice for non-centrosymmetric
    # structures. Verified here against a fully independent hand computation on an
    # asymmetric, low-symmetry two-atom cell, where the two sign conventions give
    # very different (not just complex-conjugate-equal) answers at general hkl.
    atoms = Atoms(
        "CO",
        positions=[[0.3, 0.1, 0.0], [1.1, 0.4, 0.2]],
        cell=[3.0, 3.5, 4.0],
        pbc=True,
    )
    hkl = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, -1, 1]])

    F_code = calculate_structure_factors(
        hkl,
        atoms,
        parametrization="lobato",
        g_max=3.0,
        thermal_sigma=0.0,
        occupancy=1.0,
        device="cpu",
    )

    parametrization = LobatoParametrization()
    reciprocal_cell = np.linalg.inv(np.asarray(atoms.cell)).T
    g_vec = hkl @ reciprocal_cell
    g_length_sq = (g_vec**2).sum(-1)

    F_hand = np.zeros(len(hkl), dtype=complex)
    for number, position in zip(atoms.numbers, atoms.positions):
        f = parametrization.scattering_factor(int(number))(g_length_sq)
        F_hand += f * np.exp(2.0j * np.pi * (g_vec @ position))
    F_hand /= atoms.cell.volume

    np.testing.assert_allclose(F_code, F_hand, atol=1e-7)

    # the wrong (conjugate) sign would disagree by far more than float noise
    F_hand_wrong_sign = np.zeros(len(hkl), dtype=complex)
    for number, position in zip(atoms.numbers, atoms.positions):
        f = parametrization.scattering_factor(int(number))(g_length_sq)
        F_hand_wrong_sign += f * np.exp(-2.0j * np.pi * (g_vec @ position))
    F_hand_wrong_sign /= atoms.cell.volume
    assert np.max(np.abs(F_code - F_hand_wrong_sign)) > 1e-3


def test_bloch_waves_on_skewed_cell_at_tilt_matches_orthogonalized_supercell():
    # Regression test for a crash (ValueError: invalid entry in coordinates array,
    # from np.ravel_multi_index in abtem.bloch.utils.ravel_hkl) that occurred
    # reliably for non-orthogonal cells (hexagonal/rhombohedral angles near 120
    # degrees are the worst case) once the tilt/beam selection was wide enough
    # that calculate_structure_matrix needed structure-factor values at reflection
    # differences outside the array bounds reciprocal_space_gpts sized for.
    #
    # Beyond "does it crash", this checks the fix is quantitatively correct: the
    # hexagonal primitive cell's diffraction pattern, at a general (non-zone-axis)
    # tilt, must match an orthogonalized supercell of the exact same crystal.
    hex_atoms = bulk("Mg", "hcp", a=3.21, c=5.21)
    hex_atoms.set_cell(
        [hex_atoms.cell[0], hex_atoms.cell[1], [0.0, 0.0, 7.5]], scale_atoms=True
    )
    ortho_atoms = orthogonalize_cell(hex_atoms, max_repetitions=6)

    orientation_matrix, _ = np.linalg.qr(
        np.array([[0.95, 0.05, 0.1], [-0.05, 0.98, 0.15], [-0.1, -0.15, 0.97]])
    )

    def diffraction_pattern(atoms):
        structure_factor = StructureFactor(
            atoms, g_max=6.0, parametrization="lobato", thermal_sigma=0.0
        )
        bloch_waves = BlochWaves(
            structure_factor=structure_factor,
            energy=200e3,
            sg_max=0.15,
            orientation_matrix=orientation_matrix,
            use_wave_eq=True,
        )
        return (
            bloch_waves.calculate_diffraction_patterns(thicknesses=[50.0])
            .to_cpu()
            .compute()
        )

    dp_hex = diffraction_pattern(hex_atoms)
    dp_ortho = diffraction_pattern(ortho_atoms)

    g_hex = np.asarray(dp_hex.positions)[0]
    g_ortho = np.asarray(dp_ortho.positions)[0]
    intensity_hex = np.asarray(dp_hex.array)[0]
    intensity_ortho = np.asarray(dp_ortho.array)[0]

    # match each primitive-cell reflection to its supercell counterpart by
    # physical reciprocal-space position (the two cells don't share hkl labels)
    distances = np.linalg.norm(g_hex[:, None, :] - g_ortho[None, :, :], axis=-1)
    nearest = distances.argmin(axis=1)
    matched = distances[np.arange(len(g_hex)), nearest] < 1e-4
    assert matched.mean() > 0.95  # nearly every primitive-cell reflection matches

    a = intensity_hex[matched]
    b = intensity_ortho[nearest[matched]]
    keep = (a > 1e-9) | (b > 1e-9)
    r1 = np.abs(a[keep] - b[keep]).sum() / b[keep].sum()
    assert r1 < 1e-4


@pytest.fixture(scope="module")
def silicon_bloch_waves():
    structure_factor = StructureFactor(
        bulk("Si", "diamond", a=5.43, cubic=True),
        g_max=4.0,
        parametrization="lobato",
        thermal_sigma=0.0,
    )
    return BlochWaves(structure_factor=structure_factor, energy=200e3, sg_max=0.1)


def _intensities_by_hkl(diffraction_patterns):
    # map each Miller index to its intensities; ensembles and single orientations
    # generally include different sets of beams
    diffraction_patterns = diffraction_patterns.to_cpu().compute()
    array = np.asarray(diffraction_patterns.array)
    return {
        tuple(int(i) for i in hkl): array[..., j]
        for j, hkl in enumerate(diffraction_patterns.miller_indices)
    }


def _assert_matches_single_orientation(ensemble_pattern, single_pattern):
    ensemble = _intensities_by_hkl(ensemble_pattern)
    single = _intensities_by_hkl(single_pattern)
    assert set(single) <= set(ensemble)
    for hkl, value in ensemble.items():
        np.testing.assert_allclose(
            value, single.get(hkl, np.zeros_like(value)), atol=1e-6
        )


@pytest.mark.parametrize("lazy", [True, False])
def test_zero_width_rotation_ensemble_diffraction_matches_unrotated(
    silicon_bloch_waves, lazy
):
    thicknesses = [50.0, 100.0]
    ensemble = silicon_bloch_waves.rotate("x", abtem.distributions.uniform(0.0, 0.0, 2))
    assert isinstance(ensemble, BlochwaveEnsemble)

    patterns = ensemble.calculate_diffraction_patterns(thicknesses, lazy=lazy)
    assert patterns.is_lazy == lazy
    assert [axis.label for axis in patterns.ensemble_axes_metadata] == [
        "x_rotation",
        "z",
    ]
    patterns = patterns.compute()

    reference = silicon_bloch_waves.calculate_diffraction_patterns(
        thicknesses, lazy=False
    )
    assert patterns.shape == (2,) + reference.shape
    np.testing.assert_array_equal(patterns.miller_indices, reference.miller_indices)
    for i in range(2):
        np.testing.assert_allclose(patterns.array[i], reference.array, atol=1e-6)


@pytest.mark.parametrize("lazy", [True, False])
def test_zero_width_rotation_ensemble_exit_waves_match_unrotated(
    silicon_bloch_waves, lazy
):
    ensemble = silicon_bloch_waves.rotate("x", abtem.distributions.uniform(0.0, 0.0, 2))

    exit_waves = ensemble.calculate_exit_waves([50.0], gpts=(32, 32), lazy=lazy)
    assert exit_waves.is_lazy == lazy
    exit_waves = exit_waves.compute()

    reference = silicon_bloch_waves.calculate_exit_waves(
        [50.0], gpts=(32, 32), lazy=False
    )
    assert exit_waves.shape == (2,) + reference.shape
    for i in range(2):
        np.testing.assert_allclose(exit_waves.array[i], reference.array, atol=1e-6)


@pytest.mark.parametrize("lazy", [True, False])
def test_rotation_ensemble_matches_individually_rotated_bloch_waves(
    silicon_bloch_waves, lazy
):
    # Regression test for SciPy >= 1.18, which rejects 1-D angle arrays for a
    # single-axis Euler sequence. Two distributions also check that the ensemble
    # composes the rotations in the same order as a non-ensemble rotate call.
    x_angles = np.array([0.0, 0.01])
    y_angles = np.array([-0.005, 0.0, 0.008])
    ensemble = silicon_bloch_waves.rotate(
        "x",
        abtem.distributions.from_values(x_angles),
        "y",
        abtem.distributions.from_values(y_angles),
    )

    patterns = ensemble.calculate_diffraction_patterns([50.0], lazy=lazy).compute()
    assert patterns.shape[:3] == (2, 3, 1)
    # with extent=None the ensemble uses the largest rotated-cell bounds of all its
    # members, so fix the extent to compare with single orientations
    exit_wave_kwargs = {"gpts": (32, 32), "extent": (5.43, 5.43)}
    exit_waves = ensemble.calculate_exit_waves(
        [50.0], lazy=lazy, **exit_wave_kwargs
    ).compute()
    assert exit_waves.shape == (2, 3, 1, 32, 32)

    for i, x in enumerate(x_angles):
        for j, y in enumerate(y_angles):
            single = silicon_bloch_waves.rotate("x", x, "y", y)
            _assert_matches_single_orientation(
                patterns[i, j], single.calculate_diffraction_patterns([50.0])
            )
            np.testing.assert_allclose(
                exit_waves.array[i, j],
                single.calculate_exit_waves([50.0], **exit_wave_kwargs).compute().array,
                atol=1e-5,
            )


def test_rotation_ensemble_with_multi_axis_sequence(silicon_bloch_waves):
    angles = np.array([[0.0, 0.01], [0.01, -0.005]])
    ensemble = silicon_bloch_waves.rotate("xy", angles)
    assert isinstance(ensemble, BlochwaveEnsemble)

    patterns = ensemble.calculate_diffraction_patterns([50.0]).compute()
    assert patterns.shape[:2] == (2, 1)

    for i, (x, y) in enumerate(angles):
        single = silicon_bloch_waves.rotate("xy", np.array([x, y]))
        _assert_matches_single_orientation(
            patterns[i], single.calculate_diffraction_patterns([50.0])
        )


def _si_structure_factor():
    return StructureFactor(bulk("Si", cubic=True), g_max=4.0)


@pytest.mark.parametrize("criterion", ["intensity", "distance"])
def test_sort_indexed_diffraction_patterns(criterion):
    bloch_waves = BlochWaves(_si_structure_factor(), energy=100e3, sg_max=0.1)
    spots = bloch_waves.calculate_diffraction_patterns([10.0, 20.0], lazy=False)

    sorted_spots = spots.sort(criterion)

    if criterion == "intensity":
        key = np.asarray(sorted_spots.array).max(axis=0)
    else:
        key = np.linalg.norm(sorted_spots.positions, axis=-1).max(axis=0)
    assert np.all(np.diff(key) <= 1e-12)  # descending
    np.testing.assert_array_equal(
        sorted_spots.reciprocal_lattice_vectors, spots.reciprocal_lattice_vectors
    )
    # the same spots, reordered
    before = dict(zip(map(tuple, spots.miller_indices), np.asarray(spots.array).T))
    for hkl, values in zip(
        map(tuple, sorted_spots.miller_indices), np.asarray(sorted_spots.array).T
    ):
        np.testing.assert_array_equal(values, before[hkl])

    with pytest.raises(RuntimeError, match="lazy"):
        bloch_waves.calculate_diffraction_patterns(10.0, lazy=True).sort(criterion)


@pytest.mark.parametrize("thicknesses", [10.0, [10.0, 20.0]])
def test_positions_dict_maps_every_spot_to_its_position(thicknesses):
    bloch_waves = BlochWaves(_si_structure_factor(), energy=100e3, sg_max=0.1)
    spots = bloch_waves.calculate_diffraction_patterns(thicknesses, lazy=False)
    positions = spots.positions_dict
    assert len(positions) == len(spots.miller_indices)
    for hkl, position in zip(
        map(tuple, spots.miller_indices), np.moveaxis(spots.positions, -2, 0)
    ):
        np.testing.assert_array_equal(positions[hkl], position)
    np.testing.assert_array_equal(positions[(0, 0, 0)], 0.0)


@pytest.mark.parametrize("lazy", [False, True])
def test_projected_potential_sequence_slice_thickness(lazy):
    sf = _si_structure_factor()
    depth = sf.cell[2, 2]
    potential = sf.get_projected_potential(
        slice_thickness=[2.0, 2.0, depth - 4.0], lazy=lazy
    )
    assert len(potential.slice_thickness) == 3
    assert np.isclose(sum(potential.slice_thickness), depth)
    # slicing only redistributes the potential along z
    reference = sf.get_projected_potential(slice_thickness=2.0, lazy=lazy)
    np.testing.assert_allclose(
        np.asarray(potential.compute().array).sum(0),
        np.asarray(reference.compute().array).sum(0),
        rtol=1e-5,
        atol=1e-6,
    )

    with pytest.raises(ValueError, match="add up"):
        sf.get_projected_potential(slice_thickness=[2.0, 2.0], lazy=lazy)
    with pytest.raises(ValueError, match="at least one z grid point"):
        sf.get_projected_potential(slice_thickness=[depth - 1e-3, 1e-3], lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_projected_potential_sampling_and_integer_gpts(lazy):
    sf = _si_structure_factor()
    default = sf.get_projected_potential(slice_thickness=2.0, lazy=lazy)
    sampled = sf.get_projected_potential(slice_thickness=2.0, sampling=0.05, lazy=lazy)
    assert sampled.gpts != default.gpts
    np.testing.assert_allclose(sampled.sampling, 0.05, rtol=0.01)
    assert sf.get_projected_potential(slice_thickness=2.0, gpts=64, lazy=lazy).gpts == (
        64,
        64,
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_kinematical_pattern_and_eager_exit_waves_on_device(device):
    # NumPy excitation errors / reciprocal vectors used to meet CuPy data
    from abtem.core.backend import asnumpy

    sf = StructureFactor(bulk("Si", cubic=True), g_max=4.0, device=device)
    reference_sf = StructureFactor(bulk("Si", cubic=True), g_max=4.0, device="cpu")
    for method, kwargs in (
        ("get_kinematical_diffraction_pattern", {}),
        ("calculate_exit_waves", dict(thicknesses=[10.0], gpts=(16, 16), lazy=False)),
    ):
        result = getattr(
            BlochWaves(sf, energy=100e3, sg_max=0.1, device=device), method
        )(**kwargs)
        expected = getattr(BlochWaves(reference_sf, energy=100e3, sg_max=0.1), method)(
            **kwargs
        )
        np.testing.assert_allclose(
            asnumpy(result.array), asnumpy(expected.array), rtol=1e-5, atol=1e-7
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("radius", [None, 0.25])
def test_spot_indexing_stays_on_device(device, lazy, radius):
    # Every array that indexing combines with the patterns must be on their device
    # (CuPy refuses the implicit conversion of NumPy arrays), and the result stays
    # there and matches the CPU result.
    from scipy.spatial.transform import Rotation

    from abtem.core.axes import NonLinearAxis
    from abtem.core.backend import asnumpy, copy_to_device, get_array_module
    from abtem.measurements import DiffractionPatterns

    # a single member: the eager path indexes with one orientation matrix
    rotations = (10.0,)
    cpu_array = np.random.default_rng(0).random((len(rotations), 49, 49))
    orientation_matrices = Rotation.from_euler(
        "x", np.asarray(rotations)[:, None], degrees=True
    ).as_matrix()

    def index(array):
        patterns = DiffractionPatterns(
            array,
            sampling=(0.1, 0.1),
            fftshift=True,
            ensemble_axes_metadata=[
                NonLinearAxis(label="x_rotation", units="deg", values=rotations)
            ],
        )
        return patterns.index_diffraction_spots(
            cell=bulk("Si", cubic=True).cell,
            orientation_matrices=orientation_matrices,
            centering="F",
            radius=radius,
            energy=100e3,
        )

    array = copy_to_device(cpu_array, device)
    if lazy:
        array = da.from_array(array, chunks=(1, 49, 49))
    result = index(array)
    if lazy:
        result = result.compute()

    assert get_array_module(result.array) is get_array_module(device)
    np.testing.assert_allclose(asnumpy(result.array), index(cpu_array).array, rtol=1e-6)


def test_eager_exit_waves_for_scalar_thickness_have_no_thickness_axis():
    # The eager path returned a (1, nx, ny) array for a scalar thickness while the
    # axes metadata only described the two spatial axes, which made Waves raise a
    # RuntimeError; the lazy path always returned a 2D array.
    structure_factor = StructureFactor(bulk("Si", cubic=True), g_max=3.0)
    bloch_waves = BlochWaves(structure_factor, energy=100e3, sg_max=0.1)

    eager = bloch_waves.calculate_exit_waves(50.0, lazy=False)
    lazy = bloch_waves.calculate_exit_waves(50.0, lazy=True)

    assert eager.array.ndim == 2
    assert eager.shape == lazy.shape
    assert np.allclose(eager.array, lazy.array.compute())


def _hermitian_propagator_argument():
    # i*H for a Hermitian H, the form expm takes in calculate_scattering_matrix,
    # with a norm large enough that scaling and squaring is exercised.
    rng = np.random.default_rng(0)
    H = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    return 1j * (H + H.conj().T)


def test_cupy_expm_algorithm_matches_scipy(monkeypatch):
    # The CuPy matrix exponential used b[8] * A where the [13/13] Pade approximant
    # needs b[8] * A2. Here the module is loaded with NumPy standing in for CuPy,
    # so the algorithm itself is checked without a GPU.
    import importlib.util
    from pathlib import Path

    from scipy.linalg import expm as expm_scipy

    class NumpyAsCupy:
        fuse = staticmethod(lambda function: function)

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr("abtem.core.backend.cp", NumpyAsCupy())
    path = Path(abtem.__file__).parent / "bloch" / "matrix_exponential.py"
    spec = importlib.util.spec_from_file_location("_expm_numpy", path)
    expm_numpy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expm_numpy)

    a = _hermitian_propagator_argument()
    reference = expm_scipy(a)
    np.testing.assert_allclose(
        expm_numpy.expm(a.copy()),
        reference,
        rtol=0,
        atol=1e-11 * np.abs(reference).max(),
    )


@pytest.mark.skipif(cp is None, reason="no gpu")
def test_cupy_expm_matches_scipy():
    from scipy.linalg import expm as expm_scipy

    from abtem.bloch.matrix_exponential import expm as expm_cupy

    a = _hermitian_propagator_argument()
    reference = expm_scipy(a)
    result = cp.asnumpy(expm_cupy(cp.asarray(a)))
    np.testing.assert_allclose(
        result, reference, rtol=0, atol=1e-10 * np.abs(reference).max()
    )


@pytest.mark.parametrize(
    "hkl, expected",
    [
        # Planes: (h k l) -> (h k i l), i = -(h + k), derived by hand.
        ((1, 0, 0), (1, 0, -1, 0)),  # a prism plane, (10-10)
        ((1, 0, 1), (1, 0, -1, 1)),  # a pyramidal plane, (10-11)
        ((2, 1, 3), (2, 1, -3, 3)),
        ((1, 1, 0), (1, 1, -2, 0)),  # (11-20)
        ((0, 0, 1), (0, 0, 0, 1)),  # the basal plane, (0001)
    ],
)
def test_miller_bravais_known_values(hkl, expected):
    from abtem.bloch.indexing import miller_to_miller_bravais

    assert tuple(miller_to_miller_bravais(hkl)) == expected


@pytest.mark.parametrize("hkl", [(1, 0, 0), (0, 1, 2), (2, -1, 1), (-3, 2, 4)])
def test_miller_bravais_indices_are_the_plane_intercepts(hkl):
    """Geometric oracle: a plane's Miller(-Bravais) indices are g . a_j for
    its reciprocal-lattice vector g and each axis a_j. Using a real
    hexagonal cell with the three basal axes a1, a2, a3 = -(a1 + a2) and c,
    the four indices must be (g.a1, g.a2, g.a3, g.c)."""
    from ase.cell import Cell

    from abtem.bloch.indexing import miller_to_miller_bravais

    cell = Cell.new([3.2, 3.2, 5.2, 90, 90, 120])
    a1, a2, c = np.array(cell)
    a3 = -(a1 + a2)
    g = np.array(hkl) @ np.linalg.inv(np.array(cell)).T
    intercepts = [g @ axis for axis in (a1, a2, a3, c)]
    np.testing.assert_allclose(miller_to_miller_bravais(hkl), intercepts, atol=1e-9)
