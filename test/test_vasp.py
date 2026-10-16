import warnings

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


#: The fake POTCAR's species, as (core electron count Nc, Gaussian width a [Å]).
_FAKE_CORES = {"C": (2.0, 0.8), "O": (6.0, 1.3)}


@pytest.fixture
def fake_potcar(tmp_path):
    path = tmp_path / "POTCAR_fake"
    cores = {s: _gaussian_core_density(Nc, a) for s, (Nc, a) in _FAKE_CORES.items()}
    _write_fake_potcar(path, cores)
    return path


def test_parse_potcar_roundtrip(fake_potcar):
    """The parsed grid and core density are the arrays written, to the 12
    significant digits the file stores."""
    elements = parse_potcar(fake_potcar)
    assert set(elements) == set(_FAKE_CORES)
    for symbol, (Nc, a) in _FAKE_CORES.items():
        r, core_density = _gaussian_core_density(Nc, a)
        assert elements[symbol]["nmax"] == len(r)
        np.testing.assert_allclose(elements[symbol]["grid"], r, rtol=1e-11)
        np.testing.assert_allclose(
            elements[symbol]["core_density"], core_density, rtol=1e-11, atol=1e-11
        )


def test_core_density_fourier_transform_matches_gaussian_analytic():
    """The l=0 spherical Fourier transform of an isotropic Gaussian core density
    has a known closed form; check the numerical implementation against it."""
    Nc, a = 6.0, 1.3
    r, core_density = _gaussian_core_density(Nc, a)

    G = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    f_numeric = core_density_fourier_transform(r, core_density, G)
    f_analytic = Nc * np.exp(-((a * G / 2) ** 2))

    assert f_numeric == pytest.approx(f_analytic, rel=1e-5)


def test_core_density_interpolator_is_the_gaussian_transform_in_inverse_angstrom(
    fake_potcar,
):
    """For each fake species the interpolator reproduces the analytic transform
    `Nc * exp(-(a * G / 2) ** 2)`, with G in 1 / Å, out to G = 20 / Å where it
    has decayed to zero.

    f(G=0) integrates the core density over all space, i.e. the core electron
    count -- this is what makes the correction physically meaningful.

    The POTCAR radial grid is in Ångström (VASP's own internal length unit), so
    the interpolator's argument must be an angular wavenumber in 1 / Å. Reading
    the grid as Bohr instead would rescale the G axis by 1 / 0.529, compressing
    the core density in real space and driving `V_core(0)` ~1.9x too high.
    """
    elements = parse_potcar(fake_potcar)
    G = np.linspace(0.0, 20.0, 81)  # 1 / Å

    for symbol, (Nc_written, a) in _FAKE_CORES.items():
        interp, Nc = get_core_density_fourier_interpolator(symbol, elements)

        assert Nc == pytest.approx(Nc_written, rel=1e-4), symbol
        assert interp(0.0) == pytest.approx(Nc), symbol
        # The fixture's trapezoid quadrature is good to 3e-5 here; misreading
        # the grid as Bohr would miss by 0.9 (C) and 2.6 (O).
        assert interp(G) == pytest.approx(
            Nc * np.exp(-((a * G / 2) ** 2)), abs=2e-3
        ), symbol


@pytest.fixture
def carbon_atoms():
    return Atoms("C", positions=[(2.5, 2.5, 2.5)], cell=(5, 5, 5), pbc=True)


@pytest.fixture
def charge_density_3d(carbon_atoms):
    """Random, but normalised to the valence-electron count `fake_potcar` implies
    for a single carbon atom (Z=6, Nc=2 -> 4 valence electrons), so that
    `VASPPotential`'s POTCAR-based electron-count check is satisfied."""
    rho = np.random.RandomState(0).rand(32, 32, 32).astype(np.float64) * 0.1
    voxel_volume = carbon_atoms.cell.volume / rho.size
    return (rho * (4.0 / (rho.sum() * voxel_volume))).astype(np.float32)


def test_vasp_potential_build(carbon_atoms, charge_density_3d, fake_potcar):
    """An eager build from a POTCAR path and a lazy build from the pre-parsed
    POTCAR give the same finite potential.

    The whole point of VASPPotential is to replace the crude Gaussian
    point-charge core correction with the POTCAR's actual radial core density --
    the two should give different potentials for the same input density."""
    from abtem.potentials.charge_density import ChargeDensityPotential

    pot = VASPPotential(
        carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
    )
    eager = pot.build(lazy=False)
    assert eager.array.shape[-2:] == pot.gpts
    assert np.all(np.isfinite(eager.array))
    # subtract_min is off by default, so the slice minima are left alone
    assert any(not np.isclose(s.min(), 0.0) for s in eager.array)

    preparsed = VASPPotential(
        carbon_atoms,
        charge_density_3d,
        potcar=parse_potcar(fake_potcar),
        sampling=0.1,
    )
    lazy = preparsed.build(lazy=True).compute()
    scale = np.abs(eager.array).max()
    np.testing.assert_allclose(lazy.array, eager.array, rtol=0, atol=1e-6 * scale)

    crude = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    assert not np.allclose(eager.array, crude.build(lazy=False).array)


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


def test_vasp_potential_subtract_min(carbon_atoms, charge_density_3d, fake_potcar):
    """subtract_min defaults to False (test_vasp_potential_build checks that the
    default build leaves the slice minima alone); when True, VASPPotential passes
    it on to the slice generator, which zeros each slice's minimum."""
    default = VASPPotential(
        carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
    )
    assert default.subtract_min is False

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


def test_vasp_potential_on_skew_cell(fake_potcar):
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
    # normalised to this cell's own valence count: 2 C atoms, Z-Nc = 4 each
    rho = np.random.RandomState(0).rand(32, 32, 32).astype(np.float64) * 0.1
    rho *= 8.0 / (rho.sum() * atoms.cell.volume / rho.size)

    # 13 x 13 gpts: the skew-grid interpolation is per pixel, and a coarse grid
    # takes the same path as a fine one
    pot = VASPPotential(atoms, rho.astype(np.float32), potcar=fake_potcar, sampling=0.2)
    assert not pot.grid.is_orthogonal

    built = pot.build(lazy=False)
    assert np.all(np.isfinite(built.array))
    assert built.cell is not None
    assert not built.grid.is_orthogonal


def test_warns_when_density_misses_potcar_valence_count(carbon_atoms, fake_potcar):
    """POTCAR states each species' core-electron count, so the valence count the
    density must integrate to is known exactly (sum of Z - Nc). A density carrying
    the core electrons too -- e.g. VASP's AECCAR0+AECCAR2 sum -- integrates to
    roughly the total atomic number instead, and must be flagged."""
    rho = np.random.RandomState(0).rand(32, 32, 32).astype(np.float64) * 0.1
    voxel_volume = carbon_atoms.cell.volume / rho.size
    all_electron = rho * (6.0 / (rho.sum() * voxel_volume))  # Z=6, not Z-Nc=4

    with pytest.warns(UserWarning, match="valence electrons"):
        VASPPotential(
            carbon_atoms, all_electron, potcar=fake_potcar, sampling=0.1
        )


def test_no_warning_when_density_matches_potcar_valence_count(
    carbon_atoms, charge_density_3d, fake_potcar
):
    """The correctly normalised valence density must not trip the check."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        VASPPotential(
            carbon_atoms, charge_density_3d, potcar=fake_potcar, sampling=0.1
        )
