"""`repetitions` repeats the lattice vectors of the cell, not their components."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene

import abtem
from abtem.potentials.charge_density import ChargeDensityPotential


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64", "fft": "numpy"}):
        yield


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
CASES = [
    ("orthogonal x (2, 3, 2)", _two_atoms, (2, 3, 2), (40, 30)),
    ("BN x (2, 1, 1)", _bn, (2, 1, 1), (25, 45)),
    ("BN x (2, 3, 2)", _bn, (2, 3, 2), (25, 45)),
    ("BN x (1, 2, 1)", _bn, (1, 2, 1), (25, 45)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_charge_density_potential_repetitions_tile_the_one_cell_potential(case):
    name, make_atoms, repetitions, unit_gpts = case
    atoms, rho = make_atoms(), _charge_density()

    unit = ChargeDensityPotential(atoms, rho, gpts=unit_gpts, slice_thickness=1.0)
    unit_array = unit.build(lazy=False).array
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
    actual = repeated.build(lazy=False).array
    assert actual.shape == tiled.shape
    np.testing.assert_allclose(actual, tiled, rtol=0, atol=1e-4 * np.abs(tiled).max())


def test_charge_density_potential_repetitions_lazy_equals_eager_on_bn():
    atoms, rho = _bn(), _charge_density()
    potential = ChargeDensityPotential(
        atoms, rho, gpts=(50, 135), slice_thickness=1.0, repetitions=(2, 3, 2)
    )
    eager = potential.build(lazy=False).array
    lazy = potential.build(lazy=True).compute().array
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-10 * np.abs(eager).max())


def test_charge_density_potential_with_an_approximate_default_box_builds_silently():
    # The default box of BN x (3, 1, 1) is reached by a strain of about 1 %; it
    # is the potential's own box, and the Ewald potential it builds from it does
    # not report it (the test suite turns warnings into errors).
    atoms, rho = _bn(), _charge_density()
    potential = ChargeDensityPotential(
        atoms, rho, sampling=0.2, slice_thickness=1.0, repetitions=(3, 1, 1)
    )
    eager = potential.build(lazy=False).array
    lazy = potential.build(lazy=True).compute().array
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-10 * np.abs(eager).max())
