import numpy as np
import pytest
from ase import Atoms

from abtem.potentials.charge_density import (
    ChargeDensityPotential,
    _interpolate_between_cells,
)


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


# A periodic field on a hexagonal cell, sampled on a grid whose size differs along every
# axis. Its exact values are known everywhere, so the interpolation can be checked
# against the field itself and against the invariances of a periodic interpolant.
_HEX_A, _HEX_C = 2.5, 4.0
_HEX_CELL = np.array(
    [[_HEX_A, 0, 0], [-_HEX_A / 2, _HEX_A * np.sqrt(3) / 2, 0], [0, 0, _HEX_C]]
)
_HEX_BOX = np.diag([_HEX_A, _HEX_A * np.sqrt(3), _HEX_C])
_HEX_GPTS = (24, 20, 40)


def _periodic_field(fractional):
    reciprocal = 2 * np.pi * np.linalg.inv(_HEX_CELL).T
    terms = zip(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
        [1.0, 0.7, 0.5, 0.4],
        [0.3, 1.1, 2.0, 0.7],
    )
    positions = fractional @ _HEX_CELL
    return sum(
        amplitude * np.cos(positions @ (np.array(hkl) @ reciprocal) + phase)
        for hkl, amplitude, phase in terms
    )


def _fractional_grid(shape):
    axes = [np.arange(n) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


@pytest.fixture(scope="module")
def hexagonal_field():
    return _periodic_field(_fractional_grid(_HEX_GPTS)).reshape(_HEX_GPTS)


def test_interpolation_between_cells_reproduces_a_periodic_field(hexagonal_field):
    """Against the exact field, including the target points next to the cell faces."""
    shape = (25, 43, 40)
    interpolated = _interpolate_between_cells(
        hexagonal_field, shape, _HEX_CELL, _HEX_BOX
    )
    fractional = _fractional_grid(shape) @ _HEX_BOX @ np.linalg.inv(_HEX_CELL)
    exact = _periodic_field(fractional % 1.0).reshape(shape)

    scale = np.abs(hexagonal_field).max()
    assert np.abs(interpolated - exact).max() < 3e-4 * scale


def test_interpolation_between_cells_is_invariant_to_whole_point_rolls(
    hexagonal_field,
):
    shape = (25, 43, 40)
    rolls = (5, 7, 9)
    shift = np.array(rolls) / np.array(_HEX_GPTS) @ _HEX_CELL

    rolled = _interpolate_between_cells(
        np.roll(hexagonal_field, rolls, axis=(0, 1, 2)),
        shape,
        _HEX_CELL,
        _HEX_BOX,
        offset=shift,
    )
    original = _interpolate_between_cells(hexagonal_field, shape, _HEX_CELL, _HEX_BOX)

    scale = np.abs(hexagonal_field).max()
    np.testing.assert_allclose(rolled, original, rtol=0, atol=1e-12 * scale)


def test_interpolation_between_cells_is_invariant_to_tiling(hexagonal_field):
    shape = (25, 43, 40)
    tiled_cell = _HEX_CELL * np.array([[2], [1], [1]])

    tiled = _interpolate_between_cells(
        np.tile(hexagonal_field, (2, 1, 1)), shape, tiled_cell, _HEX_BOX
    )
    original = _interpolate_between_cells(hexagonal_field, shape, _HEX_CELL, _HEX_BOX)

    scale = np.abs(hexagonal_field).max()
    np.testing.assert_allclose(tiled, original, rtol=0, atol=1e-12 * scale)
