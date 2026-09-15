import numpy as np
import pytest
from ase import Atoms

from abtem.potentials.vasp import (
    VASPPotential,
    core_density_fourier_transform,
    get_core_density_fourier_interpolator,
    parse_potcar,
)


def _format_radial_block(values, per_line=5):
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("  " + "  ".join(f"{v:.12E}" for v in chunk))
    return lines


def _write_fake_potcar(path, elements):
    """Write a minimal POTCAR-like file containing only the sections
    `parse_potcar` actually reads (VRHFIN, PAW radial sets/grid/core
    charge-density, End of Dataset), for one or more elements.

    `elements` maps chemical symbol to (r, core_density) arrays, where
    `core_density` follows the real POTCAR convention `sqrt(4 pi) * r**2 * n(r)`.
    """
    lines = []
    for symbol, (r, core_density) in elements.items():
        lines.append(f"   VRHFIN ={symbol}: fake")
        lines.append(" PAW radial sets")
        lines.append(f"         {len(r)}   1.0")
        lines.append("(5E20.12)")
        lines.append(" grid")
        lines.extend(_format_radial_block(r))
        lines.append(" core charge-density")
        lines.extend(_format_radial_block(core_density))
        lines.append(" End of Dataset")
    path.write_text("\n".join(lines) + "\n")


def _gaussian_core_density(Nc, a, n_r=2000, r_max_factor=10):
    """A core density shaped as an isotropic Gaussian with total charge `Nc` and
    width `a`, whose l=0 Fourier transform has the known closed form
    `Nc * exp(-(a * G / 2) ** 2)` -- used to check the Fourier transform
    implementation against an analytic reference."""
    r = np.linspace(1e-4, r_max_factor * a, n_r)
    n = Nc / (np.pi**1.5 * a**3) * np.exp(-((r / a) ** 2))
    core_density = np.sqrt(4 * np.pi) * r**2 * n
    return r, core_density


@pytest.fixture
def fake_potcar(tmp_path):
    r_c, rho_c = _gaussian_core_density(Nc=2.0, a=0.8)
    r_o, rho_o = _gaussian_core_density(Nc=6.0, a=1.3)
    path = tmp_path / "POTCAR_fake"
    _write_fake_potcar(path, {"C": (r_c, rho_c), "O": (r_o, rho_o)})
    return path


def test_parse_potcar_roundtrip(fake_potcar):
    elements = parse_potcar(fake_potcar)
    assert set(elements) == {"C", "O"}
    for symbol in ("C", "O"):
        assert elements[symbol]["grid"].shape == (2000,)
        assert elements[symbol]["core_density"].shape == (2000,)
        assert elements[symbol]["nmax"] == 2000


def test_core_density_fourier_transform_matches_gaussian_analytic():
    """The l=0 spherical Fourier transform of an isotropic Gaussian core density
    has a known closed form; check the numerical implementation against it."""
    Nc, a = 6.0, 1.3
    r, core_density = _gaussian_core_density(Nc, a)

    G = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    f_numeric = core_density_fourier_transform(r, core_density, G)
    f_analytic = Nc * np.exp(-((a * G / 2) ** 2))

    assert f_numeric == pytest.approx(f_analytic, rel=1e-5)


def test_fourier_transform_at_zero_equals_core_electron_count(fake_potcar):
    """f(G=0) integrates the core density over all space, i.e. the core electron
    count -- this is what makes the correction physically meaningful."""
    elements = parse_potcar(fake_potcar)
    interp_c, Nc_c = get_core_density_fourier_interpolator("C", elements)
    interp_o, Nc_o = get_core_density_fourier_interpolator("O", elements)

    assert Nc_c == pytest.approx(2.0, rel=1e-4)
    assert Nc_o == pytest.approx(6.0, rel=1e-4)
    assert interp_c(0.0) == pytest.approx(Nc_c)
    assert interp_o(0.0) == pytest.approx(Nc_o)


def test_interpolator_decays_with_increasing_g(fake_potcar):
    elements = parse_potcar(fake_potcar)
    interp, Nc = get_core_density_fourier_interpolator("O", elements)
    G = np.array([0.0, 1.0, 5.0, 20.0])
    f = interp(G)
    assert f[0] == pytest.approx(Nc)
    assert np.all(np.diff(f) < 0)


@pytest.fixture
def carbon_atoms():
    return Atoms("C", positions=[(2.5, 2.5, 2.5)], cell=(5, 5, 5), pbc=True)


@pytest.fixture
def charge_density_3d():
    return np.random.RandomState(0).rand(32, 32, 32).astype(np.float32) * 0.1


def test_vasp_potential_build(carbon_atoms, charge_density_3d, fake_potcar):
    pot = VASPPotential(
        carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
    )
    result = pot.build(lazy=False)
    assert result.array.shape[-2:] == pot.gpts
    assert np.all(np.isfinite(result.array))


def test_vasp_potential_build_lazy(carbon_atoms, charge_density_3d, fake_potcar):
    pot = VASPPotential(
        carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
    )
    result = pot.build(lazy=True).compute()
    assert result.array.shape[-2:] == pot.gpts
    assert np.all(np.isfinite(result.array))


def test_vasp_potential_accepts_preparsed_potcar(
    carbon_atoms, charge_density_3d, fake_potcar
):
    elements = parse_potcar(fake_potcar)
    pot = VASPPotential(carbon_atoms, charge_density_3d, potcar=elements, sampling=0.1)
    result = pot.build(lazy=False)
    assert result.array.shape[-2:] == pot.gpts


def test_vasp_potential_missing_element_raises(
    carbon_atoms, charge_density_3d, fake_potcar
):
    elements = parse_potcar(fake_potcar)
    del elements["C"]
    with pytest.raises(ValueError, match="missing"):
        VASPPotential(carbon_atoms, charge_density_3d, potcar=elements, sampling=0.1)


def test_vasp_potential_requires_potcar(carbon_atoms, charge_density_3d):
    with pytest.raises(ValueError, match="potcar"):
        VASPPotential(carbon_atoms, charge_density_3d, sampling=0.1)


def test_vasp_potential_differs_from_crude_point_charge(
    carbon_atoms, charge_density_3d, fake_potcar
):
    """The whole point of VASPPotential is to replace the crude Gaussian
    point-charge core correction with the POTCAR's actual radial core density --
    the two should give different potentials for the same input density."""
    from abtem.potentials.charge_density import ChargeDensityPotential

    vasp_pot = VASPPotential(
        carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
    )
    crude_pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)

    vasp_result = vasp_pot.build(lazy=False)
    crude_result = crude_pot.build(lazy=False)

    assert not np.allclose(vasp_result.array, crude_result.array)


def test_vasp_potential_subtract_min_defaults_to_false(
    carbon_atoms, charge_density_3d, fake_potcar
):
    pot = VASPPotential(
        carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
    )
    assert pot.subtract_min is False

    slices = [slic.array[0] for slic in pot.generate_slices()]
    assert any(not np.isclose(s.min(), 0.0) for s in slices)


def test_vasp_potential_subtract_min_true_zeros_each_slice_minimum(
    carbon_atoms, charge_density_3d, fake_potcar
):
    pot = VASPPotential(
        carbon_atoms,
        charge_density_3d,
        potcar=fake_potcar,
        sampling=0.1,
        subtract_min=True,
    )
    assert pot.subtract_min is True

    for slic in pot.generate_slices():
        assert np.isclose(slic.array[0].min(), 0.0, atol=1e-6)


def test_vasp_potential_on_skew_cell(charge_density_3d, fake_potcar):
    """VASPPotential inherits ChargeDensityPotential's non-orthogonal (skewed)
    in-plane grid support unchanged -- the core-density correction is injected
    into the same reciprocal-space charge array as the crude point charges, so
    it must work equally well on a skewed cell."""
    a, c = 2.46, 3.35
    atoms = Atoms(
        "C2",
        cell=[
            [a, 0, 0],
            [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0],
            [0, 0, c],
        ],
        pbc=True,
        scaled_positions=[(0, 0, 0), (1 / 3, 1 / 3, 0.5)],
    )
    pot = VASPPotential(atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1)
    assert not pot.grid.is_orthogonal

    built = pot.build(lazy=False)
    assert np.all(np.isfinite(built.array))
    assert built.cell is not None
    assert not built.grid.is_orthogonal
