import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from ase.build import bulk
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.dynamical import calculate_structure_factors
from abtem.bloch.utils import (
    auto_detect_centering,
    relative_positions_for_centering,
    wrapped_is_close,
)
from abtem.parametrizations import LobatoParametrization
from utils import gpu


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


@settings(max_examples=5)
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


@settings(max_examples=5)
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

    potential = abtem.Potential(
        atoms,
        gpts=structure_factor_potential.gpts,
        slice_thickness=structure_factor_potential.slice_thickness,
        parametrization=parametrization,
        projection="infinite",
    )

    structure_factor_potential = structure_factor_potential.project().compute()
    potential = potential.build(lazy=lazy).project().compute()

    array1 = structure_factor_potential.array
    array2 = potential.array
    array1 -= array1.min()
    array2 -= array2.min()

    error = np.abs(array2 - array1).sum() / array1.sum() * 100
    assert error < 2.5


def test_structure_factor_potential_requests_the_slow_fft_diagnostic():
    # This 3D transform cannot go through abtem.core.fft.ifftn (the FFTW
    # backend there transforms only the trailing two axes, silently making it
    # 2D on CPU), so it must ask for the diagnostic explicitly or escape it.
    from unittest import mock

    import abtem.bloch.dynamical as dynamical

    with mock.patch.object(dynamical, "warn_if_slow_gpu_fft") as warner:
        potential = dynamical.structure_factor_to_potential(
            np.zeros(4, dtype=np.complex64),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            gpts=(4, 4, 4),
        )

    assert warner.call_count == 1
    assert potential.shape == (4, 4, 4)


def test_structure_factor_matches_standard_crystallographic_sign_convention():
    # calculate_structure_factors must compute F(g) = sum_j f_j(g) * exp(+2pi i
    # g . r_j) / V, the standard International-Tables convention. abTEM previously
    # computed the complex conjugate (exp(-2pi i ...)), which structure_factor_to_
    # potential's ifftn happened to cancel when reconstructing a real-space potential
    # (see test_potential_from_structure_factor), but calculate_structure_matrix
    # consumes F(g) directly with no such cancellation -- so the wrong sign there
    # gave Bloch-wave results inconsistent with multislice for non-centrosymmetric
    # structures (reported from the 3DED project's independent cross-check against
    # multislice). Verified here against a fully independent hand computation on an
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
    # Reported from the 3DED project's Bloch-wave cross-checks against multislice.
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


# --- abTEM/abTEM#455 --------------------------------------------------------


def _si_structure_factor():
    return StructureFactor(bulk("Si", cubic=True), g_max=4.0)


@pytest.mark.parametrize(
    "method, args",
    [("calculate_structure_matrix", ()), ("calculate_scattering_matrix", (50.0,))],
)
def test_matrix_methods_need_a_single_energy(method, args):
    # the energies' beam sets differ in size, so there is no common matrix to
    # stack; they used to answer silently for the first energy
    sf = _si_structure_factor()
    multi = BlochWaves(sf, energy=[100e3, 200e3], sg_max=0.1)
    with pytest.raises(ValueError, match="select_energy"):
        getattr(multi, method)(*args)
    getattr(multi.select_energy(200e3), method)(*args)


ENERGIES = (100e3, 200e3)


def test_select_energy_equals_single_energy_bloch_waves():
    sf = _si_structure_factor()
    multi = BlochWaves(sf, energy=list(ENERGIES), sg_max=0.1)
    for energy in ENERGIES:
        selected = multi.select_energy(energy)
        single = BlochWaves(sf, energy=energy, sg_max=0.1)
        assert selected.energy == energy
        np.testing.assert_array_equal(selected.hkl, single.hkl)
        np.testing.assert_allclose(
            selected.calculate_structure_matrix(lazy=False),
            single.calculate_structure_matrix(lazy=False),
        )
    with pytest.raises(ValueError, match="not one of"):
        multi.select_energy(300e3)
    single = BlochWaves(sf, energy=100e3, sg_max=0.1)
    assert single.select_energy(100e3) is single


def _per_energy_rows(multi_hkl, single_hkl):
    index = {tuple(h): i for i, h in enumerate(multi_hkl)}
    return np.array([index[tuple(h)] for h in single_hkl])


def test_excitation_errors_with_several_energies():
    sf = _si_structure_factor()
    multi = BlochWaves(sf, energy=list(ENERGIES), sg_max=0.1)
    sg = multi.excitation_errors()
    assert sg.shape == (len(ENERGIES), len(multi.hkl))
    for row, energy in zip(sg, ENERGIES):
        single = BlochWaves(sf, energy=energy, sg_max=0.1)
        rows = _per_energy_rows(multi.hkl, single.hkl)
        np.testing.assert_allclose(row[rows], single.excitation_errors())
    assert not np.allclose(sg[0], sg[1])  # really per energy


def test_kinematical_pattern_with_several_energies():
    from abtem.core.axes import EnergyAxis

    sf = _si_structure_factor()
    multi = BlochWaves(sf, energy=list(ENERGIES), sg_max=0.1)
    pattern = multi.get_kinematical_diffraction_pattern()
    assert isinstance(pattern.ensemble_axes_metadata[0], EnergyAxis)
    assert pattern.array.shape == (len(ENERGIES), len(multi.hkl))
    for member, energy in zip(np.asarray(pattern.array), ENERGIES):
        single = BlochWaves(sf, energy=energy, sg_max=0.1)
        rows = _per_energy_rows(multi.hkl, single.hkl)
        np.testing.assert_allclose(
            member[rows], single.get_kinematical_diffraction_pattern().array
        )
        others = np.setdiff1d(np.arange(len(multi.hkl)), rows)
        assert np.all(member[others] == 0)


def test_several_energies_honor_lazy_false():
    bloch_waves = BlochWaves(_si_structure_factor(), energy=[100e3, 200e3], sg_max=0.1)
    thicknesses = [10.0, 20.0]

    eager = bloch_waves.calculate_diffraction_patterns(thicknesses, lazy=False)
    lazy = bloch_waves.calculate_diffraction_patterns(thicknesses, lazy=True).compute()
    assert not eager.is_lazy
    np.testing.assert_allclose(eager.array, lazy.array)

    eager = bloch_waves.calculate_exit_waves(thicknesses, gpts=(16, 16), lazy=False)
    lazy = bloch_waves.calculate_exit_waves(thicknesses, gpts=(16, 16)).compute()
    assert not eager.is_lazy
    np.testing.assert_allclose(eager.array, lazy.array)


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
    for hkl, values in zip(map(tuple, sorted_spots.miller_indices),
                           np.asarray(sorted_spots.array).T):
        np.testing.assert_array_equal(values, before[hkl])

    with pytest.raises(RuntimeError, match="lazy"):
        bloch_waves.calculate_diffraction_patterns(10.0, lazy=True).sort(criterion)


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
    assert sf.get_projected_potential(slice_thickness=2.0, gpts=64, lazy=lazy).gpts == (64, 64)


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
        result = getattr(BlochWaves(sf, energy=100e3, sg_max=0.1, device=device), method)(**kwargs)
        expected = getattr(BlochWaves(reference_sf, energy=100e3, sg_max=0.1), method)(**kwargs)
        np.testing.assert_allclose(
            asnumpy(result.array), asnumpy(expected.array), rtol=1e-5, atol=1e-7
        )
