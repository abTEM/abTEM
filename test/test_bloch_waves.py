import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from ase.build import bulk
from hypothesis import assume, given
from hypothesis import strategies as st

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.dynamical import BlochwaveEnsemble, calculate_structure_factors
from abtem.bloch.utils import (
    auto_detect_centering,
    relative_positions_for_centering,
    wrapped_is_close,
)
from abtem.core.backend import cp
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
