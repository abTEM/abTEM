import warnings

import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from ase.build import bulk
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from utils import gpu

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.bloch import BlochWavePrecisionWarning, BlochWaves, StructureFactor
from abtem.bloch.dynamical import calculate_structure_factors
from abtem.bloch.utils import (
    auto_detect_centering,
    relative_positions_for_centering,
    wrapped_is_close,
)
from abtem.core.backend import asnumpy
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


def _silicon_bloch_waves(device):
    structure_factor = StructureFactor(
        bulk("Si", "diamond", a=5.43, cubic=True),
        g_max=4.0,
        parametrization="lobato",
        thermal_sigma=0.0,
        device=device,
    )
    return BlochWaves(
        structure_factor=structure_factor, energy=200e3, sg_max=0.1, device=device
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
