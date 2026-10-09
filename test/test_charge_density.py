import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene
from utils import float64_devices

import abtem
from abtem.core.backend import asnumpy
from abtem.inelastic.phonons import FrozenPhonons
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


def test_chunks_of_frozen_phonons_hold_every_configuration(
    carbon_atoms, charge_density_3d
):
    """The chunks of an unbuilt potential with frozen phonons hold every
    configuration, each equal to the potential of its displaced atoms built alone."""
    frozen_phonons = FrozenPhonons(
        carbon_atoms, num_configs=3, sigmas=0.1, seed=4, ensemble_mean=False
    )
    potential = ChargeDensityPotential(
        frozen_phonons, charge_density_3d, sampling=0.2, slice_thickness=1.0
    )

    chunks = list(potential.generate_chunked_slices(chunk_size=2))

    assert len(potential) == 5
    assert [chunk.shape[:2] for chunk in chunks] == [(3, 1), (3, 2), (3, 2)]
    for i, atoms in enumerate(frozen_phonons):
        expected = ChargeDensityPotential(
            atoms, charge_density_3d, sampling=0.2, slice_thickness=1.0
        ).build(lazy=False)
        np.testing.assert_allclose(
            np.concatenate([chunk.array[i] for chunk in chunks]),
            expected.array,
            rtol=0,
            atol=1e-6 * expected.array.max(),
        )


@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_anisotropic_sigmas_follow_the_axes_of_the_input_atoms(
    charge_density_3d, plane
):
    """The point charges of frozen phonons in a potential rotated to another plane
    are displaced along the axes of the input atoms."""
    atoms = Atoms("C", positions=[(2.0, 2.5, 3.0)], cell=(5, 6, 7), pbc=True)
    sigmas = (0.05, 0.10, 0.20)
    frozen_phonons = FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=4)
    r = np.random.default_rng(frozen_phonons.seed[0]).normal(size=(1, 3))
    displaced = atoms.copy()
    displaced.positions += np.array(sigmas, dtype=np.float32) * r

    actual = ChargeDensityPotential(
        frozen_phonons, charge_density_3d, sampling=0.2, plane=plane
    ).build(lazy=False)
    expected = ChargeDensityPotential(
        displaced, charge_density_3d, sampling=0.2, plane=plane
    ).build(lazy=False)

    np.testing.assert_allclose(
        actual.array[0], expected.array, rtol=0, atol=1e-5 * expected.array.max()
    )


@pytest.mark.parametrize("num_densities", [2, 4])
def test_a_charge_density_count_other_than_one_or_the_configurations_raises(
    carbon_atoms, charge_density_3d, num_densities
):
    """Three configurations take one charge density or three: two leave one
    configuration without a density, and four leave one density unused."""
    frozen_phonons = FrozenPhonons(carbon_atoms, num_configs=3, sigmas=0.1, seed=4)
    densities = np.stack([charge_density_3d] * num_densities)
    with pytest.raises(ValueError, match="charge densities were given for 3"):
        ChargeDensityPotential(frozen_phonons, densities, sampling=0.2)


def test_several_charge_densities_without_frozen_phonons_raise(
    carbon_atoms, charge_density_3d
):
    densities = np.stack([charge_density_3d] * 2)
    with pytest.raises(ValueError, match="charge densities were given for 1"):
        ChargeDensityPotential(carbon_atoms, densities, sampling=0.2)


# `repetitions` repeats the lattice vectors of the cell, not their components.


def _bn():
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    return atoms


def _two_atoms():
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )


def _charge_density():
    shape = (16, 12, 20)
    x, y, z = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
    return (
        0.3
        + 0.1 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        + 0.05 * np.cos(4 * np.pi * z)
    )


# The orthogonal cell is the baseline accuracy of the comparison with the tiled
# one-cell potential (1.4e-5 of the maximum); the others are of the same order.
REPETITION_CASES = [
    ("orthogonal x (2, 3, 2)", _two_atoms, (2, 3, 2), (40, 30)),
    ("BN x (2, 1, 1)", _bn, (2, 1, 1), (25, 45)),
    ("BN x (2, 3, 2)", _bn, (2, 3, 2), (25, 45)),
    ("BN x (1, 2, 1)", _bn, (1, 2, 1), (25, 45)),
]


@float64_devices
@pytest.mark.parametrize("case", REPETITION_CASES, ids=lambda c: c[0])
def test_charge_density_potential_repetitions_tile_the_one_cell_potential(case):
    name, make_atoms, repetitions, unit_gpts = case
    atoms, rho = make_atoms(), _charge_density()

    unit = ChargeDensityPotential(atoms, rho, gpts=unit_gpts, slice_thickness=1.0)
    unit_array = asnumpy(unit.build(lazy=False).array)
    repeated = ChargeDensityPotential(
        atoms,
        rho,
        sampling=unit.sampling,
        slice_thickness=1.0,
        repetitions=repetitions,
    )

    # The box of the repeated crystal.
    assert repeated.box == pytest.approx(
        abtem.Potential(atoms * repetitions, sampling=0.1).box
    )

    nx, ny, nz = (round(float(repeated.box[i] / unit.box[i])) for i in range(3))
    tiled = np.tile(unit_array, (nz, nx, ny))
    actual = asnumpy(repeated.build(lazy=False).array)
    assert actual.shape == tiled.shape
    np.testing.assert_allclose(actual, tiled, rtol=0, atol=1e-4 * np.abs(tiled).max())


@float64_devices
def test_charge_density_potential_repetitions_lazy_equals_eager_on_bn():
    atoms, rho = _bn(), _charge_density()
    potential = ChargeDensityPotential(
        atoms, rho, gpts=(50, 135), slice_thickness=1.0, repetitions=(2, 3, 2)
    )
    eager = asnumpy(potential.build(lazy=False).array)
    lazy = asnumpy(potential.build(lazy=True).compute().array)
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-10 * np.abs(eager).max())


@float64_devices
def test_charge_density_potential_with_an_approximate_default_box_reports_it_once():
    # The default box of BN x (3, 1, 1) is reached by a strain of about 1 %. The
    # potential reports it when it is constructed; the Ewald potential it builds
    # from its own box does not repeat it.
    atoms, rho = _bn(), _charge_density()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = ChargeDensityPotential(
            atoms, rho, sampling=0.2, slice_thickness=1.0, repetitions=(3, 1, 1)
        )
        eager = asnumpy(potential.build(lazy=False).array)
        lazy = asnumpy(potential.build(lazy=True).compute().array)
    assert len([r for r in records if "abTEM chose" in str(r.message)]) == 1
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-10 * np.abs(eager).max())
