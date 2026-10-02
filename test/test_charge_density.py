import numpy as np
import pytest
from ase import Atoms

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


def test_charge_density_potential_on_skew_cell_preserves_cell():
    """ChargeDensityPotential.build on a non-orthogonal in-plane cell must produce a
    PotentialArray on the same skew grid (not silently rectified onto a Cartesian
    rectangle). Before the slicer was generalised, _interpolate_slice built the
    target slice on a diagonal slice_box, so the resulting PotentialArray came back
    with ``grid.is_orthogonal == True`` even on a skew input."""
    import numpy as np
    from ase import Atoms

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
    density = np.random.RandomState(0).rand(32, 32, 32).astype(np.float32) * 0.1

    pot = ChargeDensityPotential(atoms, density, sampling=0.1)
    assert not pot.grid.is_orthogonal
    assert pot.cell is not None

    built = pot.build(lazy=False)
    arr = np.asarray(built.array)
    assert np.all(np.isfinite(arr))
    # the cell metadata must propagate through the slicer to the built array
    assert built.cell is not None, (
        "ChargeDensityPotential.build silently lost the non-orthogonal cell metadata"
    )
    assert not built.grid.is_orthogonal


def test_point_charges_general_cell_conserves_nuclear_charge():
    """The pixel-volume normalisation in _add_point_charges_real_space /
    _add_point_charges_fourier must be the true parallelepiped volume |det(cell)|,
    not the product of diagonal entries. For an ASE-standard cell (a along x, b in
    the xy-plane) the cell matrix is upper-triangular and the two formulas happen
    to coincide. They diverge when the cell is rotated -- e.g. a not along x --
    which abTEM accepts through ChargeDensityPotential. Test with such a cell that
    the total integrated nuclear charge equals sum(atomic_numbers) (a
    method-independent sanity check)."""
    import numpy as np
    from ase import Atoms

    from abtem.potentials.charge_density import _add_point_charges_real_space

    # rotated cell where prod(diag) = 60 but |det| = 55 (a NOT along x)
    cell = [[3.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    parallelepiped = abs(np.linalg.det(np.array(cell)))
    prod_diag = float(np.prod(np.diag(np.array(cell))))
    # sanity: the two normalisations differ here (so we're testing the real fix)
    assert abs(prod_diag - parallelepiped) / parallelepiped > 0.05

    atoms = Atoms(
        "C2", cell=cell, pbc=True,
        scaled_positions=[(0.1, 0.2, 0.3), (0.6, 0.7, 0.7)],
    )
    expected_total_charge = float(np.sum(atoms.numbers))

    shape = (32, 32, 32)
    array = np.zeros(shape, dtype=np.float32)
    array = _add_point_charges_real_space(array, atoms)
    pixel_volume = parallelepiped / np.prod(shape)
    total_real = float(array.sum()) * pixel_volume
    assert abs(total_real - expected_total_charge) / expected_total_charge < 1e-5, (
        f"nuclear-charge normalisation broken on rotated cell: got {total_real:.4f} "
        f"expected {expected_total_charge:.4f} -- pixel_volume should be |det(cell)|/N, "
        f"not prod(diag(cell))/N"
    )
