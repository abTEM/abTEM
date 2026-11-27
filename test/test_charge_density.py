import numpy as np
import pytest
from ase import Atoms

import abtem
from abtem.inelastic.phonons import FrozenPhonons
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


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "charge_densities", [1, 3], ids=["shared", "per_configuration"]
)
def test_build_from_frozen_phonons(
    carbon_atoms, charge_density_3d, lazy, charge_densities
):
    """Each configuration equals the potential of its displaced atoms built alone,
    with one charge density for every configuration or one for each."""
    frozen_phonons = FrozenPhonons(
        carbon_atoms, num_configs=3, sigmas=0.1, seed=4, ensemble_mean=False
    )
    densities = [charge_density_3d * (1 + 0.1 * i) for i in range(charge_densities)]
    charge_density = densities[0] if charge_densities == 1 else np.stack(densities)

    potential = ChargeDensityPotential(frozen_phonons, charge_density, sampling=0.2)
    built = potential.build(lazy=lazy)
    if lazy:
        built = built.compute()

    assert built.ensemble_shape == (3,)
    for i, atoms in enumerate(frozen_phonons):
        density = densities[i if charge_densities > 1 else 0]
        expected = ChargeDensityPotential(atoms, density, sampling=0.2)
        expected = expected.build(lazy=False)
        np.testing.assert_allclose(
            built.array[i], expected.array, rtol=0, atol=1e-6 * expected.array.max()
        )


@pytest.mark.parametrize("cls", ["ChargeDensityPotential", "GPAWPotential"])
def test_documented_slice_thickness_default_is_the_default(cls):
    # the docstrings said 0.5 Å where the default is 1.0 Å
    import inspect

    from abtem.potentials import charge_density, gpaw

    builder = getattr(charge_density, cls, None) or getattr(gpaw, cls)
    default = inspect.signature(builder).parameters["slice_thickness"].default

    assert f"(default is {default} Å)" in builder.__doc__


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_repeated_cell_displaces_each_frozen_phonon_configuration(
    carbon_atoms, charge_density_3d, lazy
):
    """With `repetitions` the potential was made from the undisplaced repeated atoms,
    so every configuration was the same."""

    def build(sigmas, seed=4):
        frozen_phonons = FrozenPhonons(
            carbon_atoms, num_configs=3, sigmas=sigmas, seed=seed, ensemble_mean=False
        )
        potential = ChargeDensityPotential(
            frozen_phonons, charge_density_3d, sampling=0.4, repetitions=(2, 1, 1)
        )
        built = potential.build(lazy=lazy)
        return (built.compute() if lazy else built).array

    displaced = build(0.1)
    undisplaced = build(0.0)
    scale = np.abs(undisplaced).max()

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert np.abs(displaced[i] - displaced[j]).max() > 1e-3 * scale
    # no displacement: the configurations are the repeated cell, all alike
    np.testing.assert_allclose(undisplaced[0], undisplaced[1], atol=1e-6 * scale)
    assert np.abs(displaced[0] - undisplaced[0]).max() > 1e-3 * scale
    # a seed gives the same configurations again, another seed other ones
    np.testing.assert_allclose(displaced, build(0.1), rtol=0, atol=1e-10 * scale)
    assert np.abs(displaced - build(0.1, seed=5)).max() > 1e-3 * scale


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
# uneven chunks of several configurations; equal ones and one block add no case
@pytest.mark.parametrize("chunks", [((1, 3),)], ids=["uneven"])
@pytest.mark.parametrize(
    "charge_densities", [1, 4], ids=["shared", "per_configuration"]
)
def test_chunks_of_several_configurations_get_their_own_charge_densities(
    carbon_atoms, charge_density_3d, lazy, chunks, charge_densities
):
    """A chunk of configurations holds the charge densities of those configurations
    (it used to get density k for chunk k, and drop the rest)."""
    frozen_phonons = FrozenPhonons(
        carbon_atoms, num_configs=4, sigmas=0.1, seed=4, ensemble_mean=False
    )
    densities = [charge_density_3d * (1 + 0.5 * i) for i in range(charge_densities)]
    charge_density = densities[0] if charge_densities == 1 else np.stack(densities)
    potential = ChargeDensityPotential(frozen_phonons, charge_density, sampling=0.2)

    if lazy:
        blocks = list(potential.ensemble_blocks(chunks).compute())
    else:
        blocks = [block.item() for *_, block in potential.generate_blocks(chunks)]
    built = np.concatenate(
        [
            block.build(lazy=False).array.reshape((-1,) + block.base_shape)
            for block in blocks
        ]
    )

    assert [block.ensemble_shape[0] for block in blocks] == list(
        potential._validate_ensemble_chunks(chunks)[0]
    )
    for i, atoms in enumerate(frozen_phonons):
        density = densities[i if charge_densities > 1 else 0]
        expected = ChargeDensityPotential(atoms, density, sampling=0.2)
        expected = expected.build(lazy=False).array
        np.testing.assert_allclose(
            built[i], expected, rtol=0, atol=1e-6 * expected.max()
        )


def test_repeated_cell_configuration_equals_the_potential_of_its_displaced_atoms(
    carbon_atoms, charge_density_3d
):
    """Each configuration of a repeated cell is the potential of the displaced
    repeated atoms with the repeated density."""
    frozen_phonons = FrozenPhonons(
        carbon_atoms, num_configs=3, sigmas=0.1, seed=4, ensemble_mean=False
    )
    built = ChargeDensityPotential(
        frozen_phonons, charge_density_3d, sampling=0.4, repetitions=(2, 1, 1)
    ).build(lazy=False)
    tiled = np.tile(charge_density_3d, (2, 1, 1))
    for i, config in enumerate(frozen_phonons.generate_blocks(1)):
        fp = config[2].item()
        atoms = fp.randomize(carbon_atoms * (2, 1, 1))
        expected = ChargeDensityPotential(atoms, tiled, sampling=0.4).build(lazy=False)
        np.testing.assert_allclose(
            built.array[i], expected.array, rtol=0, atol=1e-6 * expected.array.max()
        )
