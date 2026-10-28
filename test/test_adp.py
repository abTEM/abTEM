"""Anisotropic displacement parameters: one set of tensors for Bloch waves, the
averaged multislice potential and frozen phonons, in any frame and cell shape."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from utils import gpu

import abtem
from abtem.adp import (
    ADP_ARRAY,
    adp_from_cif_tags,
    adp_from_phonopy,
    cartesian_adp,
    crystallographic_adp,
    debye_adp,
    get_adp,
    read_cif_adp,
    set_adp,
)
from abtem.atoms import euler_to_rotation, rotate_atoms
from abtem.bloch.dynamical import calculate_structure_factors
from abtem.core.backend import asnumpy
from abtem.core.fft import fft_interpolate
from abtem.integrals import GaussianProjectionIntegrals
from abtem.parametrizations import LobatoParametrization

# Rutile with illustrative ADPs. By the site symmetry of P4_2/mnm, U12 is free on
# both sites, so the O tensors are not diagonal along the cell axes, and the 4_2
# screw axis relating O at (x, x, 0) and (1/2 - x, 1/2 + x, 1/2) flips its sign.
RUTILE_CIF = """data_rutile
_cell_length_a 4.5937
_cell_length_b 4.5937
_cell_length_c 2.9587
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 42/m n m'
_space_group_IT_number 136
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
Ti1 Ti 0.0 0.0 0.0 0.00553
O1 O 0.30479 0.30479 0.0 0.00637
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
Ti1 0.00578 0.00578 0.00503 -0.00028 0.0 0.0
O1 0.00620 0.00620 0.00671 -0.00230 0.0 0.0
"""


@pytest.fixture
def rutile(tmp_path):
    path = tmp_path / "rutile.cif"
    path.write_text(RUTILE_CIF)
    return read_cif_adp(str(path))


def exaggerated(atoms, factor=6.0):
    """The tensors scaled up, so that an error in their frame is large."""
    return set_adp(atoms, get_adp(atoms, cartesian=True) * factor)


def random_tensors(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 3, 3)) * 0.12
    return np.einsum("nij,nkj->nik", a, a) + 0.002 * np.eye(3)


def test_read_cif_expands_tensors_with_the_space_group(rutile):
    adp = get_adp(rutile, cartesian=True)
    scaled = rutile.get_scaled_positions()

    # U_eq = tr(U) / 3 is invariant, and the CIF gives it per site
    u_eq = np.trace(adp, axis1=1, axis2=2) / 3
    expected = np.where(rutile.numbers == 22, 0.00553, 0.00637)
    np.testing.assert_allclose(u_eq, expected, atol=1e-8)

    # U_xy of O at (x, x, 0) is U12 of the CIF; the 4_2 screw flips it
    oxygen = rutile.numbers == 8
    on_diagonal = np.isclose(scaled[:, 2], 0.0) & oxygen
    off_diagonal = np.isclose(scaled[:, 2], 0.5) & oxygen
    np.testing.assert_allclose(adp[on_diagonal, 0, 1], -0.0023, atol=1e-10)
    np.testing.assert_allclose(adp[off_diagonal, 0, 1], 0.0023, atol=1e-10)
    np.testing.assert_allclose(adp[oxygen, 2, 2], 0.00671, atol=1e-10)


def write_cif(tmp_path, text, name="crystal.cif"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_cif_is_read_through_ase(tmp_path, rutile):
    import ase.io

    path = write_cif(tmp_path, RUTILE_CIF)
    atoms = ase.io.read(path, store_tags=True)
    np.testing.assert_array_equal(
        get_adp(adp_from_cif_tags(atoms), cartesian=True),
        get_adp(rutile, cartesian=True),
    )

    with pytest.raises(ValueError, match="store_tags=True"):
        adp_from_cif_tags(ase.io.read(path))

    # the same crystal with its symmetry operations listed, not named
    operations = [
        "x,y,z",
        "-x,-y,z",
        "-y+1/2,x+1/2,z+1/2",
        "y+1/2,-x+1/2,z+1/2",
        "-x+1/2,y+1/2,-z+1/2",
        "x+1/2,-y+1/2,-z+1/2",
        "y,x,-z",
        "-y,-x,-z",
        "-x,-y,-z",
        "x,y,-z",
        "y+1/2,-x+1/2,-z+1/2",
        "-y+1/2,x+1/2,-z+1/2",
        "x+1/2,-y+1/2,z+1/2",
        "-x+1/2,y+1/2,z+1/2",
        "-y,-x,z",
        "y,x,z",
    ]
    # (ASE needs the number of the space group with them)
    listed = RUTILE_CIF.replace(
        "_symmetry_space_group_name_H-M 'P 42/m n m'\n",
        "loop_\n_space_group_symop_operation_xyz\n"
        + "".join(f"'{operation}'\n" for operation in operations),
    )
    listed_atoms = read_cif_adp(write_cif(tmp_path, listed, "listed.cif"))
    order = [
        np.argmin(np.linalg.norm(listed_atoms.positions - position, axis=1))
        for position in rutile.positions
    ]
    np.testing.assert_allclose(
        get_adp(listed_atoms, cartesian=True)[order],
        get_adp(rutile, cartesian=True),
        atol=1e-15,
    )


def test_cif_conventions_and_missing_values(tmp_path, rutile):
    expected = get_adp(rutile, cartesian=True)
    titanium = rutile.numbers == 22

    # B_ij = 8 pi^2 U_ij
    aniso = RUTILE_CIF[RUTILE_CIF.index("Ti1 0.00578") :]
    b_values = "\n".join(
        " ".join(
            [row.split()[0]] + [repr(8 * np.pi**2 * float(v)) for v in row.split()[1:]]
        )
        for row in aniso.strip().splitlines()
    )
    b_cif = RUTILE_CIF.replace("_atom_site_aniso_U_", "_atom_site_aniso_B_").replace(
        aniso.strip(), b_values
    )
    np.testing.assert_allclose(
        get_adp(read_cif_adp(write_cif(tmp_path, b_cif, "b.cif")), cartesian=True),
        expected,
        atol=1e-15,
    )

    # standard uncertainties are dropped; an unknown component falls back to U_iso
    uncertain = RUTILE_CIF.replace(
        "Ti1 0.00578 0.00578 0.00503 -0.00028",
        "Ti1 0.00578(12) 0.00578(12) ? -0.00028(3)",
    ).replace("O1 0.00620 0.00620", "O1 0.00620(9) 0.00620(9)")
    with pytest.warns(UserWarning, match="unknown components"):
        uncertain_atoms = read_cif_adp(write_cif(tmp_path, uncertain, "uncertain.cif"))
    adp = get_adp(uncertain_atoms, cartesian=True)
    np.testing.assert_allclose(adp[~titanium], expected[~titanium], atol=1e-15)
    np.testing.assert_allclose(
        adp[titanium], np.broadcast_to(0.00553 * np.eye(3), adp[titanium].shape)
    )

    # isotropic only
    isotropic = RUTILE_CIF[: RUTILE_CIF.index("loop_\n_atom_site_aniso_label")]
    adp = get_adp(
        read_cif_adp(write_cif(tmp_path, isotropic, "iso.cif")), cartesian=True
    )
    np.testing.assert_allclose(
        np.trace(adp, axis1=1, axis2=2) / 3, np.where(titanium, 0.00553, 0.00637)
    )
    np.testing.assert_allclose(
        adp - np.trace(adp, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3),
        0.0,
        atol=1e-15,
    )

    # a site without ADPs is not smeared, and says so
    partial = isotropic.replace(
        "O1 O 0.30479 0.30479 0.0 0.00637", "O1 O 0.30479 0.30479 0.0 ?"
    )
    with pytest.warns(UserWarning, match="O1"):
        adp = get_adp(
            read_cif_adp(write_cif(tmp_path, partial, "partial.cif")), cartesian=True
        )
    np.testing.assert_allclose(adp[~titanium], 0.0)

    # no ADPs at all: no tensors, so standard deviations may still be given
    bare = partial.replace("0.0 0.00553", "0.0 ?")
    with pytest.warns(UserWarning, match="no ADPs"):
        atoms = read_cif_adp(write_cif(tmp_path, bare, "bare.cif"))
    assert get_adp(atoms, cartesian=True) is None


WURTZITE_CIF = """data_ZnO
_cell_length_a 3.2495
_cell_length_b 3.2495
_cell_length_c 5.2069
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'P 63 m c'
_space_group_IT_number 186
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zn1 Zn 0.33333333 0.66666667 0.0
O1 O 0.33333333 0.66666667 0.3821
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
Zn1 0.0100 0.0100 0.0120 0.0050 0.0 0.0
O1 0.0090 0.0090 0.0150 0.0045 0.0 0.0
"""


def test_cif_adp_types_and_npd_sites(tmp_path):
    # Uani without an aniso row falls back with a warning, Uiso does not warn;
    # an NPD site is refused by its label.
    typed = (
        RUTILE_CIF.replace(
            "_atom_site_U_iso_or_equiv\n",
            "_atom_site_U_iso_or_equiv\n_atom_site_adp_type\n",
        )
        .replace("0.0 0.00553", "0.0 0.00553 Uani")
        .replace("0.0 0.00637", "0.0 0.00637 Uiso")
    )
    typed = typed.replace("Ti1 0.00578 0.00578 0.00503 -0.00028 0.0 0.0\n", "")
    with pytest.warns(UserWarning, match=r"\['Ti1'\] are anisotropic"):
        atoms = read_cif_adp(write_cif(tmp_path, typed, "typed.cif"))
    titanium = atoms.numbers == 22
    np.testing.assert_allclose(
        get_adp(atoms, cartesian=True)[titanium],
        np.broadcast_to(0.00553 * np.eye(3), (titanium.sum(), 3, 3)),
    )

    npd = RUTILE_CIF.replace("O1 0.00620 0.00620", "O1 -0.00620 0.00620")
    with pytest.raises(ValueError, match="O1"):
        read_cif_adp(write_cif(tmp_path, npd, "npd.cif"))


def test_cif_tensors_are_symmetrized_over_their_sites(tmp_path):
    # Ti in rutile sits on a site with mmm symmetry along [110], which requires
    # U11 = U22 and U13 = U23 = 0. A CIF that breaks it is reported and averaged
    # over the site's operations: U11 and U22 become their mean, U13 vanishes.
    broken = RUTILE_CIF.replace(
        "Ti1 0.00578 0.00578 0.00503 -0.00028 0.0 0.0",
        "Ti1 0.00600 0.00556 0.00503 -0.00028 0.0010 0.0",
    )
    with pytest.warns(UserWarning, match="symmetry of their sites"):
        atoms = read_cif_adp(write_cif(tmp_path, broken, "broken.cif"))
    titanium = get_adp(atoms)[atoms.numbers == 22]
    np.testing.assert_allclose(titanium[:, 0, 0], 0.00578, atol=1e-12)
    np.testing.assert_allclose(titanium[:, 1, 1], 0.00578, atol=1e-12)
    np.testing.assert_allclose(titanium[:, 0, 2], 0.0, atol=1e-12)


def test_deprecated_names_and_per_axis_sigmas():
    atoms = bulk("Si", cubic=True)
    hkl = np.array([[2, 2, 0], [4, 0, 0]])
    expected = calculate_structure_factors(hkl, atoms, "lobato", 100, sigmas=0.07)
    with pytest.warns(FutureWarning, match="thermal_sigma"):
        got = calculate_structure_factors(hkl, atoms, "lobato", 100, thermal_sigma=0.07)
    np.testing.assert_array_equal(got, expected)
    with pytest.warns(FutureWarning, match="thermal_sigma"):
        structure_factor = abtem.StructureFactor(atoms, g_max=2.0, thermal_sigma=0.07)
    assert structure_factor.sigmas["Si"] == pytest.approx(0.07)
    with pytest.warns(FutureWarning, match="thermal_sigma"):
        structure_factor.thermal_sigma

    with pytest.warns(FutureWarning, match="Cartesian axes"):
        abtem.StructureFactor(atoms, g_max=2.0, sigmas=(0.05, 0.06, 0.07))

    # warned once where the frozen phonons are made, not again when a lazy
    # potential rebuilds them for each block
    with pytest.warns(FutureWarning, match="Cartesian axes") as record:
        frozen_phonons = abtem.FrozenPhonons(atoms, 2, sigmas=(0.05, 0.06, 0.07))
    assert len(record) == 1
    abtem.Potential(frozen_phonons, gpts=16, slice_thickness=2.0).build().compute()


def test_cif_tensors_have_the_symmetry_of_their_sites(tmp_path):
    # On the 3m sites of wurtzite U11 = U22 = 2 U12 and U13 = U23 = 0, which in
    # Cartesian coordinates is U_xx = U_yy = U11 (with a* = 2 / (sqrt(3) a)), U_xy
    # = 0 and U_zz = U33. Every operation of the space group that maps an atom
    # onto itself must leave its tensor unchanged.
    atoms = read_cif_adp(write_cif(tmp_path, WURTZITE_CIF))
    adp = get_adp(atoms, cartesian=True)
    zinc = atoms.numbers == 30
    for selected, u11, u33 in ((zinc, 0.0100, 0.0120), (~zinc, 0.0090, 0.0150)):
        expected = np.diag([u11, u11, u33])
        np.testing.assert_allclose(
            adp[selected], np.broadcast_to(expected, adp[selected].shape), atol=1e-15
        )

    cell = np.array(atoms.cell)
    scaled = atoms.get_scaled_positions()
    for rotation, translation in atoms.info["spacegroup"].get_symop():
        image = scaled @ rotation.T + translation
        cartesian = cell.T @ rotation @ np.linalg.inv(cell.T)
        for i in range(len(atoms)):
            difference = image[i] - scaled[i]
            if np.allclose(difference, np.round(difference), atol=1e-6):
                np.testing.assert_allclose(
                    cartesian @ adp[i] @ cartesian.T, adp[i], atol=1e-15
                )


def test_debye_adp_integrates_the_debye_density_of_states():
    # Oracle: U_iso = (1 / 3) hbar / (2 m) integral g(w) / w coth(hbar w / 2 k T)
    # with g(w) = 9 w^2 / w_D^3, three modes per atom, evaluated by quadrature.
    from scipy import integrate
    from scipy.constants import atomic_mass, hbar, k

    atoms = Atoms("CuO", positions=[(0, 0, 0), (1, 1, 1)], cell=[3, 3, 3], pbc=True)
    debye_temperatures = {"Cu": 315.0, "O": 500.0}
    for temperature in (0.0, 50.0, 300.0, 1000.0):
        u_iso = debye_adp(atoms, temperature, debye_temperatures)
        for i, symbol in enumerate(atoms.get_chemical_symbols()):
            w_d = k * debye_temperatures[symbol] / hbar
            mass = atoms.get_masses()[i] * atomic_mass

            def integrand(w):
                occupation = 1.0
                if temperature > 0:
                    occupation = 1 / np.tanh(hbar * w / (2 * k * temperature))
                return 9 * w / w_d**3 * occupation

            expected = hbar / (6 * mass) * integrate.quad(integrand, 0, w_d)[0] * 1e20
            np.testing.assert_allclose(u_iso[i], expected, rtol=1e-8)


def _emt_phonopy(atoms, supercell):
    """A phonopy calculation with the EMT potential of ASE."""
    phonopy = pytest.importorskip("phonopy")
    from ase.calculators.emt import EMT
    from phonopy.structure.atoms import PhonopyAtoms

    phonon = phonopy.Phonopy(
        PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=np.array(atoms.cell),
            scaled_positions=atoms.get_scaled_positions(),
        ),
        supercell_matrix=np.diag(supercell),
        primitive_matrix="auto",
    )
    phonon.generate_displacements(distance=0.01)
    forces = []
    for displaced in phonon.supercells_with_displacements:
        displaced = Atoms(
            displaced.symbols,
            cell=displaced.cell,
            scaled_positions=displaced.scaled_positions,
            pbc=True,
        )
        displaced.calc = EMT()
        forces.append(displaced.get_forces())
    phonon.forces = forces
    phonon.produce_force_constants()
    return phonon


def test_adp_from_phonopy_is_the_thermal_displacement_sum(tmp_path):
    # Oracle: U = hbar / (2 m N_q) sum e e^dagger coth(hbar w / 2 k T) / w over the
    # modes phonopy reports, in SI units, which checks that phonopy's matrices are
    # in Å² and assigned to the right atoms of the conventional cell.
    from scipy.constants import atomic_mass, hbar, k

    phonon = _emt_phonopy(bulk("Cu", cubic=True), (2, 2, 2))
    temperature, mesh = 300.0, (8, 8, 8)
    atoms = adp_from_phonopy(phonon, temperature, mesh=mesh)
    adp = get_adp(atoms, cartesian=True)
    assert len(atoms) == 4 and len(phonon.primitive) == 1

    phonon.run_mesh(
        mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    modes = phonon.get_mesh_dict()
    mass = phonon.primitive.masses[0] * atomic_mass
    expected = np.zeros((3, 3), dtype=complex)
    for frequencies, vectors, weight in zip(
        modes["frequencies"], modes["eigenvectors"], modes["weights"]
    ):
        for nu, frequency in enumerate(frequencies):
            if frequency < 1e-3:
                continue
            omega = 2 * np.pi * frequency * 1e12
            e = vectors[:3, nu]
            occupation = 1 / np.tanh(hbar * omega / (2 * k * temperature))
            expected += weight * np.outer(e, e.conj()) * occupation / omega
    expected = (hbar / (2 * mass * modes["weights"].sum()) * expected).real * 1e20

    np.testing.assert_allclose(
        adp, np.broadcast_to(expected, adp.shape), rtol=1e-5, atol=1e-12
    )
    # a cubic crystal on a Gamma-centred mesh is isotropic
    np.testing.assert_allclose(expected, expected[0, 0] * np.eye(3), atol=1e-15)

    # the files phonondb distributes are read by phonopy.load, which symmetrizes
    # the force constants: the same tensors to ~1e-6
    path = str(tmp_path / "phonopy_params.yaml")
    phonon.save(path)
    np.testing.assert_allclose(
        get_adp(adp_from_phonopy(path, temperature, mesh=mesh), cartesian=True),
        adp,
        rtol=1e-5,
        atol=1e-15,
    )


def test_adp_from_phonopy_turns_with_the_crystal():
    # hcp has U_zz != U_xx (by 4.5 % here, 450 times the tolerance below);
    # computing the phonons of the rotated crystal gives the rotated tensors.
    hcp = bulk("Cu", "hcp", a=2.55, c=4.16)
    rotation = euler_to_rotation(0.3, 0.5, 0.0, axes="zxz", convention="intrinsic")
    rotated = rotate_atoms(hcp, angles=(0.3, 0.5, 0.0))

    adp = get_adp(
        adp_from_phonopy(_emt_phonopy(hcp, (3, 3, 2)), 300.0, mesh=(6, 6, 4)),
        cartesian=True,
    )
    adp_rotated = get_adp(
        adp_from_phonopy(_emt_phonopy(rotated, (3, 3, 2)), 300.0, mesh=(6, 6, 4)),
        cartesian=True,
    )
    assert abs(adp[0, 2, 2] - adp[0, 0, 0]) > 0.02 * adp[0, 0, 0]
    np.testing.assert_allclose(
        adp_rotated,
        np.einsum("ia,nab,jb->nij", rotation, adp, rotation),
        atol=1e-4 * np.abs(adp).max(),
    )


def test_cartesian_adp_of_a_skewed_cell():
    # Oracle: an isotropic U_iso in crystallographic form,
    # U^ij = U_iso G*_ij / (|a*_i| |a*_j|), is U_iso I in Cartesian coordinates,
    # and U_eq = (1/3) sum_ij U^ij a*_i a*_j a_i . a_j for any tensor.
    cell = np.array([[5.1, 0.0, 0.0], [1.3, 6.2, 0.0], [-0.8, 0.9, 7.3]])
    reciprocal = np.linalg.inv(cell).T
    lengths = np.linalg.norm(reciprocal, axis=1)
    metric = reciprocal @ reciprocal.T

    u_iso = 0.012
    crystallographic = u_iso * metric / np.outer(lengths, lengths)
    np.testing.assert_allclose(
        cartesian_adp(cell, crystallographic), u_iso * np.eye(3), atol=1e-15
    )

    u = np.array([0.011, 0.017, 0.009, 0.002, -0.003, 0.001])
    u_ij = np.array([[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]])
    u_eq = np.einsum("ij,i,j,ij->", u_ij, lengths, lengths, cell @ cell.T) / 3
    cartesian = cartesian_adp(cell, u)
    assert np.isclose(np.trace(cartesian) / 3, u_eq, rtol=1e-12)

    # B = 8 pi^2 U and beta_ij = 2 pi^2 a*_i a*_j U^ij are the same tensor
    np.testing.assert_allclose(
        cartesian_adp(cell, 8 * np.pi**2 * u_ij, "B"), cartesian, rtol=1e-12
    )
    beta = 2 * np.pi**2 * np.outer(lengths, lengths) * u_ij
    np.testing.assert_allclose(cartesian_adp(cell, beta, "beta"), cartesian, rtol=1e-12)


MONOCLINIC = Atoms(
    "SiO2",
    scaled_positions=[(0.1, 0.2, 0.3), (0.6, 0.5, 0.8), (0.3, 0.7, 0.1)],
    cell=[[4.9, 0.0, 0.0], [0.0, 5.3, 0.0], [-1.6, 0.0, 6.1]],
    pbc=True,
)


def test_set_adp_takes_crystallographic_u_ij():
    # U^ij in the CIF order, as copied from a paper, on a monoclinic cell, where
    # they are not Cartesian. The oracle is cartesian_adp, which
    # test_cartesian_adp_of_a_skewed_cell checks independently.
    u_ij = np.array([0.011, 0.017, 0.009, 0.0, 0.003, 0.0])
    atoms = set_adp(MONOCLINIC, u_ij)
    expected = cartesian_adp(MONOCLINIC.cell, u_ij)
    np.testing.assert_allclose(
        get_adp(atoms, cartesian=True),
        np.broadcast_to(expected, (3, 3, 3)),
        atol=1e-15,
    )
    assert not np.allclose(expected, cartesian_adp(np.diag([1.0, 1, 1]), u_ij))

    # and get_adp gives them back, as do the (3, 3) and per-atom forms
    np.testing.assert_allclose(
        get_adp(atoms)[0], cartesian_adp(np.eye(3), u_ij), atol=1e-15
    )
    np.testing.assert_allclose(
        get_adp(set_adp(MONOCLINIC, np.tile(u_ij, (3, 1))), cartesian=True),
        get_adp(atoms, cartesian=True),
        atol=1e-15,
    )

    # U_iso is isotropic in any cell; a per-element dict mixes the forms
    np.testing.assert_allclose(
        get_adp(set_adp(MONOCLINIC, 0.01), cartesian=True),
        np.broadcast_to(0.01 * np.eye(3), (3, 3, 3)),
        atol=1e-15,
    )
    mixed = get_adp(set_adp(MONOCLINIC, {"Si": u_ij, "O": 0.02}), cartesian=True)
    np.testing.assert_allclose(mixed[0], expected, atol=1e-15)
    np.testing.assert_allclose(mixed[1], 0.02 * np.eye(3), atol=1e-15)

    # the inverse conversion, also to B and beta
    for convention in ("U", "B", "beta"):
        converted = crystallographic_adp(MONOCLINIC.cell, expected, convention)
        np.testing.assert_allclose(
            cartesian_adp(MONOCLINIC.cell, converted, convention), expected, atol=1e-15
        )

    with pytest.raises(ValueError, match="cartesian=True"):
        set_adp(Atoms("Si"), np.eye(3) * 0.01)


def test_tensors_follow_a_cell_rotated_outside_abtem():
    # ASE rotates the cell with the atoms but not the tensors; abTEM finds the
    # rotation from the cell. Oracle: R U R^T with ASE's own rotation matrix.
    tensor = random_tensors(1)[0]
    atoms = set_adp(MONOCLINIC, tensor, cartesian=True)

    rotated = atoms.copy()
    rotated.rotate(35, (1, 2, 3), rotate_cell=True)
    probe = Atoms(positions=np.eye(3))
    probe.rotate(35, (1, 2, 3))
    R = probe.positions.T  # columns are the images of the unit vectors
    expected = R @ tensor @ R.T

    np.testing.assert_allclose(
        get_adp(rotated, cartesian=True)[0], expected, atol=1e-12
    )
    # with vacuum added as well, and in what the simulations use
    rotated.cell[2] *= 1.5
    np.testing.assert_allclose(
        get_adp(rotated, cartesian=True)[0], expected, atol=1e-12
    )
    for resolved in (
        abtem.FrozenPhonons(rotated, num_configs=1).atoms,
        abtem.StructureFactor(rotated, g_max=1.0).atoms,
    ):
        np.testing.assert_allclose(
            get_adp(resolved, cartesian=True)[0], expected, atol=1e-12
        )

    # a supercell, reordered axes or added vacuum is the same frame
    for same_frame in (atoms * (2, 1, 3), atoms[[2, 0, 1]]):
        np.testing.assert_allclose(
            get_adp(same_frame, cartesian=True)[0], tensor, atol=1e-15
        )
    padded = atoms.copy()
    padded.center(vacuum=4.0, axis=2)
    np.testing.assert_allclose(get_adp(padded, cartesian=True)[0], tensor, atol=1e-15)


def test_conversions_match_cctbx():
    # Reference values from cctbx.adptbx (cctbx-base 2026.9, computed once and
    # hard-coded): u_cif_as_u_cart, u_star_as_beta(u_cif_as_u_star),
    # u_cart_as_u_iso and u_star_as_u_cif(u_iso_as_u_star), for a triclinic cell
    # in cctbx's Cartesian frame (the rows of its orthogonalization matrix
    # transposed).
    cell = [
        [5.1, 0.0, 0.0],
        [-1.6305599841458813, 6.085332705621131, 0.0],
        [-0.9018331411980907, 0.8245657513037988, 7.298409984867499],
    ]
    u_cif = np.array([0.0112, 0.0175, 0.0091, 0.0021, -0.0033, 0.0014])
    u_cart = [
        0.013276798868149416,
        0.018157881304740785,
        0.009100000000000006,
        -0.0032835156320729407,
        -0.004932218223286159,
        0.0024370138844963204,
    ]
    beta = [
        0.00918400969702981,
        0.009447295107445382,
        0.003372210073260614,
        0.0013972083317255572,
        -0.0018191047583099376,
        0.0006261802515264758,
    ]
    u_iso_cif = [
        0.013,
        0.012999999999999992,
        0.012999999999999994,
        0.003198898962056658,
        0.001166757209366182,
        -0.0014594398986865732,
    ]

    def as_matrix(components):
        u11, u22, u33, u12, u13, u23 = components
        return np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])

    cartesian = cartesian_adp(cell, u_cif)
    np.testing.assert_allclose(cartesian, as_matrix(u_cart), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        crystallographic_adp(cell, cartesian, "beta"), as_matrix(beta), atol=1e-15
    )
    assert np.trace(cartesian) / 3 == pytest.approx(0.013511560057630068, abs=1e-15)

    # U_iso as crystal-axis components, and back to U_iso I in Cartesian
    np.testing.assert_allclose(
        crystallographic_adp(cell, 0.013 * np.eye(3)), as_matrix(u_iso_cif), atol=1e-15
    )
    atoms = Atoms("Si", cell=cell, pbc=True)
    np.testing.assert_allclose(
        get_adp(set_adp(atoms, u_cif), cartesian=True)[0],
        as_matrix(u_cart),
        atol=1e-15,
    )


def test_site_symmetrization_matches_cctbx(tmp_path):
    # cctbx: site_symmetry((0, 0.31, 1/4)).average_u_star for C 1 2/c 1, a site
    # on the 2-fold axis along b (U12 = U23 = 0), of a tensor that breaks it.
    cif = """data_c2c
_cell_length_a 9.2
_cell_length_b 6.1
_cell_length_c 7.7
_cell_angle_alpha 90
_cell_angle_beta 112
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'C 1 2/c 1'
_space_group_IT_number 15
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0 0.31 0.25
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
Si1 0.0121 0.0098 0.0143 0.0017 0.0026 -0.0011
"""
    with pytest.warns(UserWarning, match="symmetry of their sites"):
        atoms = read_cif_adp(write_cif(tmp_path, cif))
    site = np.argmin(
        np.linalg.norm(atoms.get_scaled_positions() - [0.0, 0.31, 0.25], axis=1)
    )
    expected = np.array(
        [[0.0121, 0.0, 0.0026], [0.0, 0.0098, 0.0], [0.0026, 0.0, 0.0143]]
    )
    np.testing.assert_allclose(get_adp(atoms)[site], expected, atol=1e-12)


def test_validate_adp_forms():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    na = atoms.numbers == 11
    tensor = random_tensors(1)[0]

    adp = get_adp(
        set_adp(atoms, {"Na": 0.01, "Cl": (0.01, 0.02, 0.03)}, cartesian=True),
        cartesian=True,
    )
    np.testing.assert_allclose(
        adp[na], np.broadcast_to(0.01 * np.eye(3), adp[na].shape)
    )
    np.testing.assert_allclose(
        adp[~na], np.broadcast_to(np.diag([0.01, 0.02, 0.03]), adp[~na].shape)
    )

    adp = get_adp(set_adp(atoms, {11: tensor, 17: 0.0}, cartesian=True), cartesian=True)
    np.testing.assert_allclose(adp[na], np.broadcast_to(tensor, adp[na].shape))
    np.testing.assert_allclose(adp[~na], 0.0)

    # one U_iso per atom, also for exactly three atoms (no diagonal shorthand)
    u_iso = np.linspace(0.01, 0.02, len(atoms))
    np.testing.assert_allclose(
        get_adp(set_adp(atoms, u_iso, cartesian=True), cartesian=True),
        u_iso[:, None, None] * np.eye(3),
    )
    three = atoms[:3]
    np.testing.assert_allclose(
        get_adp(set_adp(three, [0.01, 0.02, 0.03], cartesian=True), cartesian=True),
        np.array([0.01, 0.02, 0.03])[:, None, None] * np.eye(3),
    )

    with pytest.raises(ValueError, match="missing"):
        set_adp(atoms, {"Na": 0.01}, cartesian=True)
    with pytest.raises(ValueError, match="symmetric"):
        set_adp(atoms, np.triu(tensor), cartesian=True)
    with pytest.raises(ValueError, match="positive semi-definite"):
        set_adp(atoms, -tensor, cartesian=True)

    assert get_adp(set_adp(set_adp(atoms, 0.01), None), cartesian=True) is None


@pytest.mark.filterwarnings(
    "ignore:Standard deviations along the Cartesian axes:FutureWarning"
)
def test_bloch_debye_waller_factor_is_the_quadratic_form():
    # One atom at the origin, so the structure factor is its scattering factor
    # times exp(-2 pi^2 g^T U g); the isotropic and diagonal forms of the
    # standard deviations are the special cases.
    atoms = Atoms("Si", cell=[4.1, 5.3, 6.2], pbc=True)
    hkl = np.array(
        [[h, k, l] for h in range(-3, 4) for k in (-2, 0, 1) for l in (-1, 2)]
    )
    g_vec = hkl @ atoms.cell.reciprocal()
    f = LobatoParametrization().scattering_factor("Si")(np.sum(g_vec**2, axis=1))

    def structure_factor(**kwargs):
        return calculate_structure_factors(
            hkl, g_max=100, parametrization="lobato", **kwargs
        )

    tensor = random_tensors(1)[0]
    dwf = np.exp(-2 * np.pi**2 * np.einsum("ga,ab,gb->g", g_vec, tensor, g_vec))
    np.testing.assert_allclose(
        structure_factor(atoms=set_adp(atoms, tensor, cartesian=True)),
        f * dwf / atoms.cell.volume,
        rtol=1e-5,
    )

    sigmas = np.array([0.05, 0.08, 0.11])
    np.testing.assert_allclose(
        structure_factor(atoms=atoms, sigmas=tuple(sigmas)),
        structure_factor(atoms=set_adp(atoms, np.diag(sigmas**2), cartesian=True)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        structure_factor(atoms=atoms, sigmas=0.07),
        structure_factor(atoms=set_adp(atoms, 0.07**2, cartesian=True)),
        rtol=1e-6,
    )


def test_bloch_structure_factors_are_invariant_under_rotating_the_crystal(rutile):
    # Rotating the atoms rotates their cell, so the same hkl is the same
    # reflection; its structure factor is unchanged only if the tensors turn
    # with the positions.
    atoms = exaggerated(rutile)
    hkl = np.array(
        [[h, k, l] for h in range(-3, 4) for k in range(-3, 4) for l in (-1, 0, 1)]
    )
    angles = (0.4, 1.1, -0.7)
    rotated = rotate_atoms(atoms, angles=angles)
    R = euler_to_rotation(*angles, axes="zxz", convention="intrinsic")

    np.testing.assert_allclose(
        get_adp(rotated, cartesian=True),
        np.einsum("ia,nab,jb->nij", R, get_adp(atoms, cartesian=True), R),
        atol=1e-15,
    )

    def structure_factors(atoms):
        return calculate_structure_factors(hkl, atoms, "lobato", g_max=100)

    expected = structure_factors(atoms)
    np.testing.assert_allclose(structure_factors(rotated), expected, atol=1e-6)

    # U^ij refer to the cell, which turned with the atoms: the same U^ij on the
    # rotated crystal are the rotated tensors. The same Cartesian tensors are
    # left in the old frame and give clearly different structure factors.
    same_u_ij = set_adp(rotated, get_adp(atoms))
    np.testing.assert_allclose(structure_factors(same_u_ij), expected, atol=1e-6)
    stale = set_adp(rotated, get_adp(atoms, cartesian=True), cartesian=True)
    assert np.abs(structure_factors(stale) - expected).max() > 1e-3


def test_centering_is_detected_by_kind():
    # Body-centred Fe with different tensors on the two sites is primitive: the
    # reflections with h + k + l odd are not extinct, and must not be dropped.
    atoms = Atoms(
        "Fe2", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=[2.87] * 3, pbc=True
    )
    tensors = np.array([0.01 * np.eye(3), np.diag([0.03, 0.005, 0.005])])
    odd = np.array([[1, 0, 0]])
    assert (
        abs(
            calculate_structure_factors(
                odd, set_adp(atoms, tensors, cartesian=True), "lobato", 100
            )[0]
        )
        > 1e-3
    )

    assert (
        abtem.StructureFactor(
            set_adp(atoms, tensors, cartesian=True), g_max=3
        ).centering
        == "P"
    )
    assert abtem.StructureFactor(atoms, g_max=3, occupancy=[1.0, 0.5]).centering == "P"
    assert (
        abtem.StructureFactor(set_adp(atoms, 0.01, cartesian=True), g_max=3).centering
        == "I"
    )


def test_finite_projection_shares_a_buffer_between_elements():
    # Na and Cl with the same tensor share one buffer; the potential is linear
    # in the atoms, so the oracle is the sum of the two elements built apart.
    atoms = set_adp(
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        random_tensors(1)[0],
        cartesian=True,
    )
    symbols = np.array(atoms.get_chemical_symbols())

    def build(atoms):
        return (
            abtem.Potential(
                atoms, sampling=0.1, slice_thickness=1.41, projection="finite"
            )
            .build(lazy=False)
            .array
        )

    both = build(atoms)
    separate = build(atoms[symbols == "Na"]) + build(atoms[symbols == "Cl"])
    np.testing.assert_allclose(both, separate, rtol=0, atol=1e-5 * np.abs(both).max())


def test_flip_atoms_mirrors_the_tensors():
    from abtem.atoms import flip_atoms

    tensor = random_tensors(1)[0]
    flipped = get_adp(
        flip_atoms(set_adp(bulk("Si", cubic=True), tensor, cartesian=True), axis=0),
        cartesian=True,
    )
    mirror = np.diag([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(flipped[0], mirror @ tensor @ mirror, atol=1e-15)
    assert np.sign(flipped[0, 0, 1]) == -np.sign(tensor[0, 1])


def test_thermal_sigma_and_tensors_are_not_combined(rutile):
    with pytest.raises(ValueError, match="carry ADP tensors"):
        abtem.StructureFactor(rutile, g_max=2.0, sigmas=0.1)

    with pytest.raises(ValueError, match="carry ADP tensors"):
        abtem.Potential(
            rutile, gpts=32, parametrization=LobatoParametrization(sigmas=0.1)
        )

    with pytest.raises(ValueError, match="carry ADP tensors"):
        abtem.FrozenPhonons(rutile, num_configs=2, sigmas=0.1)

    with pytest.raises(ValueError, match="sigmas"):
        abtem.FrozenPhonons(set_adp(rutile, None), num_configs=2)

    with pytest.raises(ValueError, match="electron density"):
        from abtem.potentials.charge_density import ChargeDensityPotential

        ChargeDensityPotential(rutile, charge_density=np.zeros((8, 8, 8)), gpts=16)

    # an explicit zero asks for no smearing, which contradicts the tensors too
    hkl = np.array([[1, 1, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="explicit zero"):
        calculate_structure_factors(hkl, rutile, "lobato", 100, sigmas=0.0)
    with pytest.raises(ValueError, match="carry ADP tensors"):
        abtem.Potential(
            rutile, gpts=32, parametrization=LobatoParametrization(sigmas=0.0)
        )
    with pytest.raises(ValueError, match="carry ADP tensors"):
        abtem.FrozenPhonons(rutile, num_configs=2, sigmas=0.0)


def _projected_potentials(atoms, structure_factor_atoms, device, integrator):
    """The L1 difference of the projected potential of `atoms` from `Potential` and
    the one of `structure_factor_atoms` from their Bloch-wave structure factors,
    compared on one grid and band-limited the same way (see
    test_potential_from_structure_factor in test_bloch_waves.py). The tensors are
    exaggerated, so the potentials are smooth enough for a 0.03 Å grid."""
    g_max = 6.0
    reference = (
        abtem.StructureFactor(structure_factor_atoms, g_max=g_max)
        .get_projected_potential(
            slice_thickness=structure_factor_atoms.cell[2, 2], sampling=0.05, lazy=False
        )
        .project()
        .compute()
    )
    potential = (
        abtem.Potential(
            atoms,
            sampling=0.03,
            slice_thickness=atoms.cell[2, 2] / 4,
            integrator=integrator,
            device=device,
        )
        .build(lazy=False)
        .project()
        .compute()
    )

    array1 = reference.array
    array2 = fft_interpolate(np.asarray(asnumpy(potential.array)), array1.shape).real

    kx, ky = (np.fft.fftfreq(n, d) for n, d in zip(array1.shape, reference.sampling))
    within = kx[:, None] ** 2 + ky[None] ** 2 <= (0.9 * g_max) ** 2
    array1 = np.fft.ifft2(np.fft.fft2(array1) * within).real
    array2 = np.fft.ifft2(np.fft.fft2(array2) * within).real
    array1 -= array1.min()
    array2 -= array2.min()
    return np.abs(array2 - array1).sum() / array1.sum()


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "integrator, tolerance",
    [
        # measured 1.1e-4
        (abtem.integrals.ScatteringFactorProjectionIntegrals, 1e-3),
        # measured 1.1e-3
        (abtem.integrals.QuadratureProjectionIntegrals, 3e-3),
        # measured 5.2e-3; 7.1e-3 without any ADPs, from its short-range
        # correction on this grid
        (GaussianProjectionIntegrals, 1e-2),
    ],
    ids=["infinite", "finite", "gaussian"],
)
def test_multislice_potential_matches_bloch_wave_potential(
    rutile, device, integrator, tolerance
):
    # The Bloch-wave structure factors apply the full 3D Debye-Waller factor,
    # whose projection along z is the in-plane one each multislice integrator
    # applies. Flipping the sign of U_xy in the multislice potential gives an L1
    # difference of 6.3e-2 for all three.
    atoms = exaggerated(rutile)
    assert _projected_potentials(atoms, atoms, device, integrator()) < tolerance

    flipped = get_adp(atoms, cartesian=True)
    flipped[:, [0, 1], [1, 0]] *= -1
    error = _projected_potentials(
        set_adp(atoms, flipped, cartesian=True), atoms, device, integrator()
    )
    assert error > 3e-2


# The linear maps of the positions to the planes, acting on row vectors: the
# first axis named becomes x, the second y (see rotate_atoms_to_plane).
PLANE_MAPS = {
    "xy": np.eye(3),
    "zx": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    "yz": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}


@pytest.mark.parametrize("plane", ["zx", "yz"])
def test_potential_rotates_tensors_to_its_plane(rutile, plane):
    # The oracle rotates the positions and tensors by hand.
    atoms = exaggerated(rutile)
    linear_map = PLANE_MAPS[plane]
    by_hand = Atoms(
        atoms.numbers,
        positions=atoms.positions @ linear_map,
        cell=np.array(atoms.cell) @ linear_map,
        pbc=True,
    )
    by_hand = set_adp(
        by_hand,
        np.einsum(
            "ai,nab,bj->nij", linear_map, get_adp(atoms, cartesian=True), linear_map
        ),
        cartesian=True,
    )

    transformed = abtem.Potential(
        atoms, gpts=32, slice_thickness=0.5, plane=plane
    ).get_transformed_atoms()
    # the box: abTEM orders the rotated lattice vectors along x, y and z
    np.testing.assert_allclose(
        np.diag(transformed.cell), np.abs(by_hand.cell).sum(axis=0), atol=1e-12
    )

    distances = np.linalg.norm(
        transformed.positions[:, None] - by_hand.positions[None], axis=-1
    )
    match = distances.argmin(axis=1)
    assert np.all(distances[np.arange(len(match)), match] < 1e-8)
    np.testing.assert_allclose(
        get_adp(transformed, cartesian=True),
        get_adp(by_hand, cartesian=True)[match],
        atol=1e-15,
    )


@pytest.mark.parametrize("plane", ["xy", "zx", "yz"])
def test_orthogonalized_cell_has_the_structure_factors_of_the_skewed_cell(plane):
    # A hexagonal cell is orthogonalized by repeating it, so every reflection of
    # the primitive cell is a reflection of the orthogonal one with the same
    # structure factor -- if the tensors follow the atoms into it. The tensors
    # are random, so no symmetry hides an error.
    hexagonal = bulk("ZnO", "wurtzite", a=3.25, c=5.2)
    hexagonal = set_adp(hexagonal, random_tensors(len(hexagonal)), cartesian=True)
    source = hexagonal if plane == "xy" else abtem.orthogonalize_cell(hexagonal)

    potential = abtem.Potential(source, sampling=0.1, slice_thickness=1.0, plane=plane)
    orthogonal = potential.get_transformed_atoms()

    hkl = np.array(
        [[h, k, l] for h in range(-3, 4) for k in range(-3, 4) for l in range(-2, 3)]
    )
    # reciprocal vectors are rows, and turn with the positions
    g = hkl @ hexagonal.cell.reciprocal() @ PLANE_MAPS[plane]
    hkl_orthogonal = g @ np.array(orthogonal.cell).T
    assert np.allclose(hkl_orthogonal, np.round(hkl_orthogonal), atol=1e-6)

    expected = calculate_structure_factors(hkl, hexagonal, "lobato", g_max=100)
    got = calculate_structure_factors(
        np.round(hkl_orthogonal), orthogonal, "lobato", g_max=100
    )
    np.testing.assert_allclose(np.abs(got), np.abs(expected), atol=1e-6)


@pytest.mark.filterwarnings(
    "ignore:Standard deviations along the Cartesian axes:FutureWarning"
)
@pytest.mark.filterwarnings("ignore:The box .* is not a whole supercell")
@pytest.mark.parametrize(
    "case", ["cubic", "cubic zx", "hexagonal", "monoclinic strained"]
)
def test_frozen_phonons_from_diagonal_tensors_equal_standard_deviations(case):
    # Both draw along the axes of the given atoms and map the displacements to
    # the potential's frame, here a rotation, the repetition of a hexagonal cell
    # and a 20 % strain; the oracle is the standard-deviation path, which
    # test_frozen_phonons_frame.py checks on its own.
    monoclinic = Atoms(
        "SiO",
        scaled_positions=[(0.1, 0.2, 0.3), (0.6, 0.5, 0.8)],
        cell=[[4.0, 0.0, 0.0], [0.9, 4.3, 0.0], [0.0, 0.0, 5.1]],
        pbc=True,
    )
    atoms, kwargs = {
        "cubic": (bulk("Si", cubic=True), {}),
        "cubic zx": (bulk("Si", cubic=True), {"plane": "zx"}),
        "hexagonal": (bulk("ZnO", "wurtzite", a=3.25, c=5.2), {}),
        "monoclinic strained": (monoclinic, {"box": (4.0, 8.6, 5.1)}),
    }[case]
    sigmas = np.array([0.05, 0.1, 0.2])
    kwargs = dict(gpts=32, slice_thickness=1.0, **kwargs)

    with abtem.config.set({"precision": "float64"}):
        by_sigmas = abtem.Potential(
            abtem.FrozenPhonons(atoms, 2, sigmas=tuple(sigmas), seed=3), **kwargs
        ).to_atoms_ensemble()
        by_tensors = abtem.Potential(
            abtem.FrozenPhonons(
                set_adp(atoms, np.diag(sigmas**2), cartesian=True), 2, seed=3
            ),
            **kwargs,
        ).to_atoms_ensemble()

    for a, b in zip(by_sigmas.trajectory, by_tensors.trajectory):
        np.testing.assert_allclose(b.positions, a.positions, atol=1e-12)
        # the displaced atoms are not smeared again
        assert not b.has(ADP_ARRAY)


def test_frozen_phonon_covariance_is_the_transformed_tensor():
    # The sample covariance of 13824 displacements, against the tensor the
    # averaged potential uses in the same frame. Its statistical error is
    # ~sqrt(2 / N) ~ 1.2 % of U; using the untransformed tensor is off by ~100 %.
    tensor = random_tensors(1, seed=1)[0]
    atoms = set_adp(bulk("Si", cubic=True) * (12, 12, 12), tensor, cartesian=True)
    kwargs = dict(gpts=32, slice_thickness=2.0, plane="zx")

    reference = abtem.Potential(atoms, **kwargs).get_transformed_atoms()
    displaced = (
        abtem.Potential(abtem.FrozenPhonons(atoms, 1, seed=5), **kwargs)
        .to_atoms_ensemble()
        .trajectory[0]
    )

    box = np.diag(reference.cell)
    displacements = displaced.positions - reference.positions
    displacements -= np.round(displacements / box) * box
    covariance = displacements.T @ displacements / len(displacements)

    transformed = get_adp(reference, cartesian=True)[0]
    assert not np.allclose(transformed, tensor, atol=0.1 * np.abs(tensor).max())
    assert np.abs(covariance - transformed).max() < 0.05 * np.abs(transformed).max()


def test_ebsd_slab_carries_rotated_tensors(rutile):
    from abtem.ebsd.orientations import rotated_slab

    atoms = exaggerated(rutile)
    atoms.set_array("cell_index", np.arange(len(atoms)))
    slab, rotation = rotated_slab(
        atoms, np.array([1.0, 2.0, 3.0]), cell=(8.0, 8.0, 8.0)
    )

    # directions turn as R @ d, so the tensors as R U R^T
    source = get_adp(atoms, cartesian=True)[slab.arrays["cell_index"]]
    np.testing.assert_allclose(
        get_adp(slab, cartesian=True),
        np.einsum("ia,nab,jb->nij", rotation, source, rotation),
        atol=1e-15,
    )


@pytest.mark.slow
def test_bloch_waves_pair_with_multislice_for_full_tensors(rutile):
    # Paraxial Bloch waves against first-order multislice, as in
    # test_exact_bloch_waves_pair_with_exact_multislice: R = 1.6 % with the same
    # tensors in both, 3.8 % with isotropic tensors of the same U_eq in the
    # multislice and 4.7 % with U_xy flipped there.
    from abtem.bloch import BlochWaves, StructureFactor

    atoms = exaggerated(rutile)
    repetitions = 20
    energy = 100e3

    structure_factor = StructureFactor(atoms, g_max=6.0, centering="P")
    bloch_waves = BlochWaves(
        structure_factor=structure_factor, energy=energy, sg_max=0.2, use_wave_eq=True
    )
    psi = bloch_waves.calculate_diffraction_patterns(
        [atoms.cell[2, 2] * repetitions], return_complex=True, lazy=False
    )
    summed: dict = {}
    for (h, k, _), value in zip(bloch_waves.hkl, np.asarray(psi.array).ravel()):
        summed[h, k] = summed.get((h, k), 0.0) + value
    lengths = atoms.cell.lengths()[:2]
    hk = [
        (h, k)
        for h, k in summed
        if (h, k) != (0, 0) and np.linalg.norm(np.array([h, k]) / lengths) <= 1.5
    ]

    def r_factor(atoms):
        potential = abtem.Potential(
            atoms.repeat((1, 1, repetitions)),
            sampling=0.04,
            slice_thickness=atoms.cell[2, 2] / 4,
        )
        waves = abtem.PlaneWave(energy=energy).multislice(potential, lazy=False)
        array = np.asarray(waves.array)
        spectrum = np.fft.fft2(array) / array.size
        a = np.array([abs(summed[h, k]) ** 2 for h, k in hk])
        b = np.array(
            [abs(spectrum[h % array.shape[0], k % array.shape[1]]) ** 2 for h, k in hk]
        )
        return np.abs(a - b).sum() / b.sum()

    adp = get_adp(atoms, cartesian=True)
    isotropic = np.trace(adp, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)

    matched = r_factor(atoms)
    assert matched < 0.025
    assert r_factor(set_adp(atoms, isotropic, cartesian=True)) > 2 * matched
