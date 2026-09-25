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


@pytest.mark.slow
# order=1 at 10 keV is used deliberately, as the paraxial reference
@pytest.mark.filterwarnings("ignore:Maximum propagator phase error")
def test_exact_bloch_waves_pair_with_exact_multislice():
    # Bloch waves with use_wave_eq=True solve the paraxial equation that
    # multislice solves with the first-order propagator; use_wave_eq="exact"
    # the non-paraxial one of FourierMultislice(order="exact"). At low energy,
    # where the two differ most (~lambda^3), each Bloch-wave variant must agree
    # better with its own multislice counterpart than with the other.
    from abtem.multislice import FourierMultislice

    atoms = bulk("Si", cubic=True)
    energy = 10e3
    potential = abtem.Potential(
        atoms.repeat((2, 2, int(150 / atoms.cell[2, 2]))),
        sampling=0.05,
        slice_thickness=0.25,
        parametrization="lobato",
    )

    def multislice(order):
        return (
            abtem.PlaneWave(energy=energy)
            .multislice(potential, algorithm=FourierMultislice(order=order), lazy=False)
            .diffraction_patterns(max_angle=None)
            .index_diffraction_spots(cell=atoms)
            .to_data_array()
        )

    structure_factor = StructureFactor(
        atoms, g_max=4.0, parametrization="lobato", centering="F"
    )

    def bloch_waves(use_wave_eq):
        return (
            BlochWaves(
                structure_factor=structure_factor,
                energy=energy,
                sg_max=1.5,
                use_wave_eq=use_wave_eq,
            )
            .calculate_diffraction_patterns([potential.thickness], lazy=False)
            .crop(k_max=1.5)
            .to_data_array()
        )

    def r_factor(a, b):
        hkl = sorted(set(a["hkl"].data) & set(b["hkl"].data) - {"0 0 0"})
        a = np.asarray(a.sel(hkl=hkl).data).ravel()
        b = np.asarray(b.sel(hkl=hkl).data).ravel()
        return np.abs(a - b).sum() / b.sum()

    ms = {order: multislice(order) for order in (1, "exact")}
    bw = {w: bloch_waves(w) for w in (True, "exact")}

    assert r_factor(ms[1], bw[True]) < 0.7 * r_factor(ms[1], bw["exact"])
    assert r_factor(ms["exact"], bw["exact"]) < 0.9 * r_factor(ms["exact"], bw[True])
