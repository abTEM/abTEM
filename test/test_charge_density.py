import numpy as np
import pytest
from ase import Atoms

import abtem
from abtem.potentials.charge_density import ChargeDensityPotential


@pytest.fixture
def carbon_atoms():
    return Atoms("C", positions=[(2.5, 2.5, 2.5)], cell=(5, 5, 5), pbc=True)


@pytest.fixture
def charge_density_3d():
    return np.random.RandomState(0).rand(32, 32, 32).astype(np.float32) * 0.1


def test_build_lazy(carbon_atoms, charge_density_3d):
    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    result = pot.build().compute()
    assert result.array.shape[-2:] == pot.gpts


def test_build_eager(carbon_atoms, charge_density_3d):
    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    result = pot.build(lazy=False)
    assert result.array.shape[-2:] == pot.gpts


def test_generate_slices(carbon_atoms, charge_density_3d):
    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    slices = list(pot.generate_slices())
    assert len(slices) == len(pot)


def test_4d_charge_density(carbon_atoms, charge_density_3d):
    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d[None], sampling=0.1)
    result = pot.build(lazy=False)
    assert result.array.shape[-2:] == pot.gpts


@pytest.mark.parametrize("slice_thickness", [0.5, 1.0, 2.0])
def test_various_slice_thicknesses(carbon_atoms, charge_density_3d, slice_thickness):
    pot = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, slice_thickness=slice_thickness
    )
    result = pot.build(lazy=False)
    assert result.array.shape[-2:] == pot.gpts
    assert result.array.shape[0] == len(pot)


def test_thin_slice_thickness(carbon_atoms, charge_density_3d):
    """Slice thickness equal to z-sampling of the charge density grid."""
    cell_z = carbon_atoms.cell[2, 2]
    dz = cell_z / charge_density_3d.shape[2]
    pot = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, slice_thickness=dz
    )
    result = pot.build(lazy=False)
    assert result.array.shape[-2:] == pot.gpts
    assert result.array.shape[0] == len(pot)


@pytest.mark.parametrize("repetitions", [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)])
def test_repetitions_cell(carbon_atoms, charge_density_3d, repetitions):
    """Repeating the cell should scale the box accordingly."""
    pot = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, repetitions=repetitions
    )
    base_cell = carbon_atoms.cell.diagonal()
    expected = tuple(base_cell[i] * repetitions[i] for i in range(3))
    assert np.allclose(pot.box, expected[:2] + (expected[2],), atol=1e-5)


def test_repetitions_build(carbon_atoms, charge_density_3d):
    """Building with repetitions should succeed and tile the potential."""
    pot_1x1 = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, repetitions=(1, 1, 1)
    )
    pot_2x2 = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, repetitions=(2, 2, 1)
    )
    result_1x1 = pot_1x1.build(lazy=False)
    result_2x2 = pot_2x2.build(lazy=False)

    assert result_2x2.array.shape[-2:] == pot_2x2.gpts
    assert result_2x2.array.shape[0] == len(pot_2x2)
    # The tiled potential should have ~4x more grid points in x and y
    assert result_2x2.array.shape[-2] == pytest.approx(result_1x1.array.shape[-2] * 2, abs=1)
    assert result_2x2.array.shape[-1] == pytest.approx(result_1x1.array.shape[-1] * 2, abs=1)


def test_num_frozen_phonons(carbon_atoms, charge_density_3d):
    """num_frozen_phonons should match the number of ensemble configurations."""
    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    assert pot.num_frozen_phonons == pot.num_configurations == 1


def test_repetitions_property(carbon_atoms, charge_density_3d):
    """repetitions property should return the stored tuple."""
    reps = (2, 3, 1)
    pot = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, repetitions=reps
    )
    assert pot.repetitions == reps


def test_float64_precision_keeps_charge_density(carbon_atoms):
    """The charge density must not be rounded to float32 when the configured
    precision is float64."""
    charge_density = np.random.RandomState(0).rand(32, 32, 32) * 0.1

    with abtem.config.set({"precision": "float64"}):
        pot = ChargeDensityPotential(carbon_atoms, charge_density, sampling=0.1)
        np.testing.assert_array_equal(pot._charge_density, charge_density)
        assert pot.build(lazy=False).array.dtype == np.float64


_CD_CELL = 8.0
_CD_GRID = 64  # charge-density grid points per axis (0.125 A)


def _gaussian_electrons(Z, z0, sigma):
    """-Z electrons in a normalised 3D Gaussian of std `sigma` on the atom
    (minimum-image in all directions, so the density is periodic)."""
    x = np.arange(_CD_GRID) * _CD_CELL / _CD_GRID
    d = [
        (x - c + _CD_CELL / 2) % _CD_CELL - _CD_CELL / 2
        for c in (_CD_CELL / 2,) * 2 + (z0,)
    ]
    r2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
    return Z * np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2) ** 1.5


@pytest.mark.parametrize("z0", (3.3, 4.0))
def test_neutral_atom_matches_screened_coulomb_per_slice(z0):
    """Nucleus Z plus -Z electrons in a Gaussian of std 0.7 A on it: a neutral
    atom, whose potential is V(r) = Z/(4 pi eps0) erfc(r / (sqrt(2) s)) / r.

    Oracle: that V integrated over each slice along z with scipy.integrate.quad
    (plus the +-Lz images). The long-range part of every slice used to be
    sampled from the slice's left edge instead of at midpoints, which shifted it
    by dz/2 against the short-range part: the slices above and below an atom on
    a slice boundary differed and the potential 3 A from the atom was not ~0.
    Summed over a full period the error cancels, so only per-slice values see it.
    """
    from scipy import integrate
    from scipy.special import erfc

    from abtem.core.constants import eps0

    Z, sigma, dz = 6, 0.7, 0.5
    atoms = Atoms(
        [Z], positions=[(_CD_CELL / 2, _CD_CELL / 2, z0)], cell=[_CD_CELL] * 3, pbc=True
    )
    with abtem.config.set({"precision": "float64"}):
        potential = ChargeDensityPotential(
            atoms, _gaussian_electrons(Z, z0, sigma), sampling=0.05, slice_thickness=dz
        )
        array = potential.build(lazy=False).array

    C = Z / (4 * np.pi * eps0)

    def V(r):
        return C * erfc(r / (np.sqrt(2) * sigma)) / r

    s = potential.sampling
    i0 = int(round(_CD_CELL / 2 / s[0]))
    offsets = np.array([4, 10, 20, 40, 60])  # 0.2, 0.5, 1, 2, 3 A
    numeric = array[:, i0 + offsets, i0]
    oracle = np.zeros_like(numeric)
    for j, (a, b) in enumerate(potential.slice_limits):
        for k, offset in enumerate(offsets):
            rho = offset * s[0]
            for image in (-1, 0, 1):
                lo, hi = a - z0 - image * _CD_CELL, b - z0 - image * _CD_CELL
                oracle[j, k] += integrate.quad(
                    lambda z: V(np.hypot(rho, z)), lo, hi, limit=200
                )[0]
    tolerance = 2e-3 * oracle.max()
    assert np.abs(numeric - oracle).max() < tolerance
    # Far from the atom a neutral atom's potential is ~0 in every slice.
    assert np.abs(numeric[:, -1]).max() < tolerance
