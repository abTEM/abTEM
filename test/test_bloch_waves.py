import warnings

import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from ase.build import bulk
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from utils import gpu, requires_gpu

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.bloch import BlochWavePrecisionWarning, BlochWaves, StructureFactor
from abtem.bloch.dynamical import BlochwaveEnsemble, calculate_structure_factors
from abtem.bloch.utils import (
    auto_detect_centering,
    relative_positions_for_centering,
    wrapped_is_close,
)
from abtem.core.backend import asnumpy
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
    # The structure factor is truncated at g_max, which leaves a systematic error
    # versus the real-space potential that grows as thermal_sigma shrinks (less
    # Debye-Waller damping at high g). At thermal_sigma=0.03 the error is ~3% for
    # g_max=8, ~1.4% for g_max=10 and below 1% from g_max=11, so g_max < 10 would
    # not test the implementation against the 2.5% tolerance below.
    g_max=st.floats(min_value=10, max_value=16),
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


def test_exact_excitation_errors_match_the_nonparaxial_dispersion():
    # use_wave_eq="exact": -g_z + k (sqrt(1 - (lambda g_perp)^2) - 1), the
    # counterpart of the exact multislice propagator; to first order in
    # (lambda g_perp)^2 it is the paraxial use_wave_eq=True form
    from abtem.bloch.utils import excitation_errors
    from abtem.core.energy import energy2wavelength

    energy = 100e3
    k = 1 / energy2wavelength(energy)
    g = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [6.0, 0.0, -0.3], [1e-3, 0, 0]])

    exact = excitation_errors(g, energy, use_wave_eq="exact")
    g_perp_sq = g[:, 0] ** 2 + g[:, 1] ** 2
    np.testing.assert_allclose(
        exact, -g[:, 2] + np.sqrt(k**2 - g_perp_sq) - k, rtol=1e-12, atol=1e-12
    )

    paraxial = excitation_errors(g, energy, use_wave_eq=True)
    np.testing.assert_allclose(exact[-1], paraxial[-1], rtol=1e-9)
    assert np.all(exact[1:3] < paraxial[1:3])  # the sphere lies below the paraboloid

    with pytest.raises(ValueError, match="evanescent|g_perp"):
        excitation_errors(np.array([[1.1 * k, 0.0, 0.0]]), energy, use_wave_eq="exact")
    with pytest.raises(ValueError, match="use_wave_eq"):
        excitation_errors(g, energy, use_wave_eq="paraxial")


@pytest.mark.parametrize("use_wave_eq", ["paraxial", "True", None])
def test_invalid_use_wave_eq_is_rejected_at_construction(use_wave_eq):
    # not only later, at compute time inside dask
    from abtem.bloch.dynamical import BlochwaveEnsemble

    structure_factor = StructureFactor(bulk("Si", cubic=True), g_max=2.0)
    with pytest.raises(ValueError, match="use_wave_eq"):
        BlochWaves(structure_factor, energy=100e3, sg_max=0.1, use_wave_eq=use_wave_eq)
    with pytest.raises(ValueError, match="use_wave_eq"):
        BlochwaveEnsemble(
            "x",
            np.array([0.0, 0.01]),
            structure_factor=structure_factor,
            energy=100e3,
            sg_max=0.1,
            g_max=1.0,
            use_wave_eq=use_wave_eq,
        )
    for valid in (False, True, "exact", np.bool_(True)):
        bloch_waves = BlochWaves(
            structure_factor, energy=100e3, sg_max=0.1, use_wave_eq=valid
        )
        assert bloch_waves.use_wave_eq == valid


@pytest.mark.slow
# order=1 at 10 keV is used deliberately, as the paraxial reference
@pytest.mark.filterwarnings("ignore:Maximum propagator phase error")
def test_exact_bloch_waves_pair_with_exact_multislice():
    # Bloch waves with use_wave_eq=True solve the paraxial equation that
    # multislice solves with the first-order propagator; use_wave_eq="exact"
    # the non-paraxial one of FourierMultislice(order="exact"). The two place
    # the Ewald sphere differently, by ~lambda^3 g^4 / 8, which matters most at
    # low energy and for the first-order Laue zone (FOLZ). A weakly scattering
    # crystal (one C atom per 4 x 4 x 5 Å cell) keeps the scattering nearly
    # kinematic, so a small beam set converges, while the long c puts its FOLZ
    # ring (~1.8 1/Å at 10 keV) inside the compared beams; there each Bloch-wave
    # variant must agree with its own multislice counterpart, and clearly not
    # with the other.
    from abtem.multislice import FourierMultislice

    energy = 10e3
    atoms = Atoms("C", positions=[(0, 0, 0)], cell=[4.0, 4.0, 5.0], pbc=True)
    potential = abtem.Potential(
        atoms.repeat((1, 1, 40)),
        sampling=0.05,
        slice_thickness=0.25,
        parametrization="lobato",
        projection="finite",  # the 3D potential, as the structure factor has it
    )
    structure_factor = StructureFactor(
        atoms, g_max=4.0, parametrization="lobato", centering="P"
    )

    # The beams sharing a g_perp = (h, k) / a, one per Laue zone l, all fall on
    # the same multislice pixel; at a thickness of whole unit cells their phases
    # exp(2 pi i l z / c) are 1, so the pixel is their coherent sum. Comparing
    # |sum over l|^2 with the pixel intensity avoids assigning it to a single l.
    def multislice(order):
        waves = abtem.PlaneWave(energy=energy).multislice(
            potential, algorithm=FourierMultislice(order=order), lazy=False
        )
        array = np.asarray(waves.array)
        return np.fft.fft2(array) / array.size

    def bloch_waves(use_wave_eq):
        bloch_waves = BlochWaves(
            structure_factor=structure_factor,
            energy=energy,
            sg_max=0.3,
            use_wave_eq=use_wave_eq,
        )
        psi = bloch_waves.calculate_diffraction_patterns(
            potential.thickness, return_complex=True, lazy=False
        )
        summed = {}
        for (h, k, _), value in zip(bloch_waves.hkl, np.asarray(psi.array)):
            summed[(h, k)] = summed.get((h, k), 0.0) + value
        return summed

    ms = {order: multislice(order) for order in (1, "exact")}
    bw = {w: bloch_waves(w) for w in (True, "exact")}

    hk = [(h, k) for h, k in bw[True] if (h, k) != (0, 0) and np.hypot(h, k) / 4.0 <= 2]

    def r_factor(ms, bw):
        a = np.array([abs(bw[h, k]) ** 2 for h, k in hk])
        b = np.array([abs(ms[h % ms.shape[0], k % ms.shape[1]]) ** 2 for h, k in hk])
        return np.abs(a - b).sum() / b.sum()

    # R ~ 1 % for the matched pairs, ~ 10 % for the mixed ones
    for order, use_wave_eq, other in ((1, True, "exact"), ("exact", "exact", True)):
        matched = r_factor(ms[order], bw[use_wave_eq])
        assert matched < 0.03
        assert r_factor(ms[order], bw[other]) > 3 * matched


# --- abTEM/abTEM#455 --------------------------------------------------------


def _si_structure_factor():
    return StructureFactor(bulk("Si", cubic=True), g_max=4.0)


# With sg_max=0.02 these energies select 21 and 37 beams of Si (at sg_max=0.1
# both select the same 97, so each energy's beams embedded into the union beam
# set would go untested); _multi_energy_bloch_waves checks that they differ.
ENERGIES = (100e3, 200e3)
SG_MAX = 0.02


def _multi_energy_bloch_waves(sf):
    multi = BlochWaves(sf, energy=list(ENERGIES), sg_max=SG_MAX)
    beam_sets = {frozenset(map(tuple, multi.select_energy(e).hkl)) for e in ENERGIES}
    assert len(beam_sets) == len(ENERGIES), "the energies select the same beams"
    return multi


@pytest.mark.parametrize(
    "method, args",
    [("calculate_structure_matrix", ()), ("calculate_scattering_matrix", (50.0,))],
)
def test_matrix_methods_need_a_single_energy(method, args):
    # the energies' beam sets differ in size, so there is no common matrix to
    # stack; they used to answer silently for the first energy
    sf = _si_structure_factor()
    multi = _multi_energy_bloch_waves(sf)
    with pytest.raises(ValueError, match="select_energy"):
        getattr(multi, method)(*args)
    getattr(multi.select_energy(200e3), method)(*args)


def test_select_energy_equals_single_energy_bloch_waves():
    sf = _si_structure_factor()
    multi = _multi_energy_bloch_waves(sf)
    for energy in ENERGIES:
        selected = multi.select_energy(energy)
        single = BlochWaves(sf, energy=energy, sg_max=SG_MAX)
        assert selected.energy == energy
        np.testing.assert_array_equal(selected.hkl, single.hkl)
        np.testing.assert_allclose(
            selected.calculate_structure_matrix(lazy=False),
            single.calculate_structure_matrix(lazy=False),
        )
    with pytest.raises(ValueError, match="not one of"):
        multi.select_energy(300e3)
    single = BlochWaves(sf, energy=100e3, sg_max=SG_MAX)
    assert single.select_energy(100e3) is single


def _per_energy_rows(multi_hkl, single_hkl):
    index = {tuple(h): i for i, h in enumerate(multi_hkl)}
    return np.array([index[tuple(h)] for h in single_hkl])


def test_excitation_errors_with_several_energies():
    sf = _si_structure_factor()
    multi = _multi_energy_bloch_waves(sf)
    sg = multi.excitation_errors()
    assert sg.shape == (len(ENERGIES), len(multi.hkl))
    for row, energy in zip(sg, ENERGIES):
        single = BlochWaves(sf, energy=energy, sg_max=SG_MAX)
        rows = _per_energy_rows(multi.hkl, single.hkl)
        np.testing.assert_allclose(row[rows], single.excitation_errors())
    assert not np.allclose(sg[0], sg[1])  # really per energy


def test_kinematical_pattern_with_several_energies():
    from abtem.core.axes import EnergyAxis

    sf = _si_structure_factor()
    multi = _multi_energy_bloch_waves(sf)
    pattern = multi.get_kinematical_diffraction_pattern()
    assert isinstance(pattern.ensemble_axes_metadata[0], EnergyAxis)
    assert pattern.array.shape == (len(ENERGIES), len(multi.hkl))
    for member, energy in zip(np.asarray(pattern.array), ENERGIES):
        single = BlochWaves(sf, energy=energy, sg_max=SG_MAX)
        rows = _per_energy_rows(multi.hkl, single.hkl)
        np.testing.assert_allclose(
            member[rows], single.get_kinematical_diffraction_pattern().array
        )
        others = np.setdiff1d(np.arange(len(multi.hkl)), rows)
        assert np.all(member[others] == 0)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_diffraction_patterns_with_several_energies(lazy):
    # each energy on its own beams, embedded into the union beam set
    sf = _si_structure_factor()
    multi = _multi_energy_bloch_waves(sf)
    thicknesses = [10.0, 20.0]
    patterns = multi.calculate_diffraction_patterns(thicknesses, lazy=lazy)
    assert patterns.array.shape == (len(ENERGIES), len(thicknesses), len(multi.hkl))
    for member, energy in zip(np.asarray(patterns.compute().array), ENERGIES):
        single = BlochWaves(sf, energy=energy, sg_max=SG_MAX)
        rows = _per_energy_rows(multi.hkl, single.hkl)
        expected = single.calculate_diffraction_patterns(thicknesses, lazy=False)
        np.testing.assert_allclose(member[:, rows], expected.array, atol=1e-12)
        others = np.setdiff1d(np.arange(len(multi.hkl)), rows)
        assert np.all(member[:, others] == 0)


def test_several_energies_honor_lazy_false():
    bloch_waves = _multi_energy_bloch_waves(_si_structure_factor())
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
def test_zero_width_rotation_ensemble_diffraction_matches_unrotated(lazy):
    silicon_bloch_waves = _silicon_bloch_waves("cpu")
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
def test_zero_width_rotation_ensemble_exit_waves_match_unrotated(lazy):
    silicon_bloch_waves = _silicon_bloch_waves("cpu")
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
def test_rotation_ensemble_matches_individually_rotated_bloch_waves(lazy):
    silicon_bloch_waves = _silicon_bloch_waves("cpu")
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


def test_rotation_ensemble_with_multi_axis_sequence():
    silicon_bloch_waves = _silicon_bloch_waves("cpu")
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


def test_rotation_ensemble_with_fixed_and_distributed_rotations():
    silicon_bloch_waves = _silicon_bloch_waves("cpu")
    y_angles = np.array([0.0, 0.3, 0.5])
    ensemble = silicon_bloch_waves.rotate(
        "x", 0.4, "y", abtem.distributions.from_values(y_angles), degrees=True
    )
    assert ensemble.ensemble_shape == (3,)

    patterns = ensemble.calculate_diffraction_patterns([50.0]).compute()
    assert patterns.shape[:2] == (3, 1)

    for i, y in enumerate(y_angles):
        single = silicon_bloch_waves.rotate("x", 0.4, "y", y, degrees=True)
        _assert_matches_single_orientation(
            patterns[i], single.calculate_diffraction_patterns([50.0])
        )


def _silicon_bloch_waves(device, g_max=4.0, sg_max=0.1):
    structure_factor = StructureFactor(
        bulk("Si", "diamond", a=5.43, cubic=True),
        g_max=g_max,
        parametrization="lobato",
        thermal_sigma=0.0,
        device=device,
    )
    return BlochWaves(
        structure_factor=structure_factor, energy=200e3, sg_max=sg_max, device=device
    )


# Bloch waves honor the 'precision' setting, so the host reference is pinned to
# double precision here: the tests then measure the accelerator's error against
# an accurate result, not against a second single-precision one. An accelerator
# may well be single precision -- Metal always is -- so the tolerances allow for
# float32 against float64, not merely for a different order of operations.
@pytest.mark.parametrize("device", [gpu])
def test_bloch_wave_diffraction_patterns_match_cpu(device):
    thicknesses = [50.0, 200.0, 1000.0]
    with abtem.config.set({"precision": "float64"}):
        reference = asnumpy(
            _silicon_bloch_waves("cpu")
            .calculate_diffraction_patterns(thicknesses=thicknesses)
            .compute()
            .array
        )
    result = asnumpy(
        _silicon_bloch_waves(device)
        .calculate_diffraction_patterns(thicknesses=thicknesses)
        .compute()
        .array
    )

    assert result.shape == reference.shape
    np.testing.assert_allclose(result, reference, atol=1e-5 * np.abs(reference).max())


@pytest.mark.parametrize("device", [gpu])
def test_bloch_wave_scattering_matrix_matches_cpu(device):
    with abtem.config.set({"precision": "float64"}):
        reference = asnumpy(
            _silicon_bloch_waves("cpu").calculate_scattering_matrix(50.0)
        )
    result = asnumpy(_silicon_bloch_waves(device).calculate_scattering_matrix(50.0))

    np.testing.assert_allclose(result, reference, atol=3e-5 * np.abs(reference).max())


# 721 beams at 5000 Å put the norm of the exponent in the thousands, where
# scaling and squaring breaks down at single precision. Exponentiated in
# complex64, Metal's scattering matrix was off by 9e-3; the matrix exponential
# is now done in double, leaving the 8e-4 that the single-precision structure
# matrix alone accounts for.
@pytest.mark.parametrize("device", [gpu])
def test_many_beam_scattering_matrix_matches_cpu(device):
    with abtem.config.set({"precision": "float64"}):
        reference = asnumpy(
            _silicon_bloch_waves(
                "cpu", g_max=6.0, sg_max=0.3
            ).calculate_scattering_matrix(5000.0)
        )
    result = asnumpy(
        _silicon_bloch_waves(device, g_max=6.0, sg_max=0.3).calculate_scattering_matrix(
            5000.0
        )
    )

    assert reference.shape == (721, 721)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, reference, atol=2e-3)


# The same 721 beams on the CPU, at 1000 Å. Exponentiated in complex64, the
# scattering matrix came out NaN; with the exponent formed and exponentiated in
# double, float32 agrees with float64 to 6.5e-5, the share of the
# single-precision structure matrix (2.1e-4 if the exponent is formed in single
# precision before the exponential).
def test_single_precision_scattering_matrix_matches_double():
    scattering_matrices = {}
    for precision in ("float32", "float64"):
        with abtem.config.set({"precision": precision}):
            scattering_matrices[precision] = _silicon_bloch_waves(
                "cpu", g_max=6.0, sg_max=0.3
            ).calculate_scattering_matrix(1000.0)

    result = scattering_matrices["float32"]
    reference = scattering_matrices["float64"]
    assert reference.shape == (721, 721)
    assert result.dtype == np.complex64
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, reference, atol=1.5e-4)


# Every lazy Bloch-wave array must declare the dtype its blocks actually
# compute, and both must follow the 'precision' setting. NumPy float64 scalars
# and host arrays (the cell volume, the M matrix, the excitation errors, the
# thicknesses) used to widen a float32 calculation to complex128 on the host,
# leaving a float32 meta over float64 data.
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_bloch_wave_dtypes_follow_precision(device, precision):
    real = np.dtype(precision)
    complex_ = np.result_type(real, np.complex64)
    with abtem.config.set({"precision": precision}):
        bloch_waves = _silicon_bloch_waves(device)
        structure_factor = bloch_waves.structure_factor
        calculations = {
            "structure factors": (
                lambda lazy: structure_factor.build(lazy=lazy).array,
                complex_,
            ),
            "structure matrix": (
                lambda lazy: bloch_waves.calculate_structure_matrix(lazy=lazy),
                complex_,
            ),
            "intensities": (
                lambda lazy: (
                    bloch_waves.calculate_diffraction_patterns(
                        [50.0, 1000.0], lazy=lazy
                    ).array
                ),
                real,
            ),
            "intensities, scalar thickness": (
                lambda lazy: (
                    bloch_waves.calculate_diffraction_patterns(50.0, lazy=lazy).array
                ),
                real,
            ),
            "amplitudes": (
                lambda lazy: (
                    bloch_waves.calculate_diffraction_patterns(
                        [50.0], return_complex=True, lazy=lazy
                    ).array
                ),
                complex_,
            ),
            "exit waves": (
                lambda lazy: (
                    bloch_waves.calculate_exit_waves(
                        [50.0, 1000.0], gpts=(16, 16), lazy=lazy
                    ).array
                ),
                complex_,
            ),
            "exit waves, scalar thickness": (
                lambda lazy: (
                    bloch_waves.calculate_exit_waves(
                        50.0, gpts=(16, 16), lazy=lazy
                    ).array
                ),
                complex_,
            ),
        }

        for name, (calculate, expected) in calculations.items():
            lazy_array = calculate(True)
            assert lazy_array.dtype == expected, f"{name}: declared meta"
            assert lazy_array.compute().dtype == expected, f"{name}: computed"
            assert calculate(False).dtype == expected, f"{name}: eager"

        assert bloch_waves.calculate_scattering_matrix(50.0).dtype == complex_


# The cell depth came in as a NumPy float64 scalar, widening a float32
# potential to float64 under a float32 meta.
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_potential_from_structure_factor_follows_precision(precision, lazy):
    with abtem.config.set({"precision": precision}):
        structure_factor = _silicon_bloch_waves("cpu").structure_factor
        potential = structure_factor.get_projected_potential(
            slice_thickness=5.43 / 4, lazy=lazy
        )
        assert potential.array.dtype == precision
        assert potential.compute().array.dtype == precision


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_bloch_waves_warn_in_single_precision(device, precision, monkeypatch):
    with abtem.config.set({"precision": precision}):
        bloch_waves = _silicon_bloch_waves(device)
        calculations = [
            lambda: bloch_waves.calculate_diffraction_patterns([50.0]).compute(),
            lambda: bloch_waves.calculate_exit_waves(50.0, gpts=(16, 16)).compute(),
            lambda: bloch_waves.calculate_scattering_matrix(50.0),
        ]

        for calculate in calculations:
            # The warning is shown once per process; forget earlier ones.
            monkeypatch.setattr(
                abtem.bloch.dynamical, "_issued_precision_warnings", set()
            )
            if precision == "float32":
                # Metal has no double precision to switch to, so the advice
                # there is to check on another device.
                check = "'cpu' or 'gpu' device" if device == "mps" else "precision"
                with pytest.warns(BlochWavePrecisionWarning, match=check):
                    calculate()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", BlochWavePrecisionWarning)
                    calculate()


def test_bloch_wave_precision_warning_is_shown_once():
    with abtem.config.set({"precision": "float32"}):
        bloch_waves = _silicon_bloch_waves("cpu")
        abtem.bloch.dynamical._issued_precision_warnings.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for thickness in (50.0, [50.0, 100.0]):
                bloch_waves.calculate_diffraction_patterns(thickness).compute()
            bloch_waves.calculate_scattering_matrix(50.0)

    assert [w.category for w in caught].count(BlochWavePrecisionWarning) == 1


def _hermitian_propagator_argument(dtype):
    # i*H for a Hermitian H, the form expm takes in calculate_scattering_matrix,
    # with a norm large enough that scaling and squaring is exercised.
    rng = np.random.default_rng(0)
    H = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    return (1j * (H + H.conj().T)).astype(dtype)


@requires_gpu
@pytest.mark.parametrize(
    "dtype, tolerance", [(np.complex128, 1e-10), (np.complex64, 1e-4)]
)
def test_cupy_expm_matches_scipy(dtype, tolerance):
    import cupy as cp
    from scipy.linalg import expm as expm_scipy

    from abtem.bloch.matrix_exponential import expm as expm_cupy

    a = _hermitian_propagator_argument(dtype)
    reference = expm_scipy(a.astype(np.complex128))
    result = expm_cupy(cp.asarray(a))

    # The input precision is kept, not widened by float64 constants.
    assert result.dtype == dtype
    result = cp.asnumpy(result)
    np.testing.assert_allclose(
        result, reference, atol=tolerance * np.abs(reference).max()
    )
