"""A potential's `box`, `origin` and auto grid follow the arguments given."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene

import abtem
from abtem.atoms import best_orthogonal_cell, orthogonalize_cell
from abtem.magnetism.gpaw import GPAWMagneticField, GPAWVectorPotential
from abtem.potentials.charge_density import ChargeDensityPotential

GRID = dict(sampling=0.1, slice_thickness=1.0)


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64", "fft": "numpy"}):
        yield


def _two_atoms():
    # Different lengths along x, y, z, and atoms off every symmetry position, so
    # no two axes can be confused.
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )


def _assert_same(actual, expected):
    assert actual.shape == expected.shape
    scale = np.abs(expected).max()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10 * scale)


def _charge_density():
    shape = (16, 12, 20)
    x, y, z = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
    return (
        0.3
        + 0.1 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        + 0.05 * np.cos(4 * np.pi * z)
    )


class _FakeCalculator:
    # The part of a GPAW calculator that the GPAW magnetics read when they are
    # constructed.
    def __init__(self, atoms):
        self.atoms = atoms

    def get_number_of_grid_points(self):
        return np.array([16, 12, 20])


@pytest.mark.parametrize(
    "box", [(1.5, 3.0, 5.0), (4.0, 1.0, 5.0), (4.0, 3.0, 2.0)], ids=["x", "y", "z"]
)
def test_orthogonalize_cell_box_with_no_whole_period_raises(box):
    with pytest.raises(ValueError, match="no whole repetition"):
        orthogonalize_cell(_two_atoms(), box=box)


def test_orthogonalize_cell_box_with_one_period_strains_it():
    # A box of 2.1 A holds one period of the 4 A axis, compressed by 47.5 %.
    box = (2.1, 3.0, 5.0)
    atoms = orthogonalize_cell(_two_atoms(), box=box)
    assert atoms.cell.lengths() == pytest.approx(box)
    assert len(atoms) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(box=(8.0, 9.0, 10.0)),
        dict(box=(4.0, 3.0, 5.5)),
        dict(origin=(1.0, 0.75, 0.0)),
        dict(origin=(0.0, 0.0, 1e-3)),
    ],
    ids=["box", "box-z", "origin", "origin-z"],
)
def test_charge_density_potential_rejects_a_box_or_origin(kwargs):
    with pytest.raises(NotImplementedError, match="default box"):
        ChargeDensityPotential(_two_atoms(), _charge_density(), **GRID, **kwargs)


def test_charge_density_potential_accepts_its_own_box():
    atoms = _two_atoms()
    rho = _charge_density()
    potential = ChargeDensityPotential(atoms, rho, box=(4.0, 3.0, 5.0), **GRID)
    assert potential.box == (4.0, 3.0, 5.0)
    _assert_same(
        potential.build(lazy=False).array,
        ChargeDensityPotential(atoms, rho, **GRID).build(lazy=False).array,
    )


def test_charge_density_potential_default_box_follows_the_plane_and_cell():
    rho = _charge_density()
    atoms = _two_atoms()

    # plane="xz": the box is the cell with y and z exchanged.
    in_plane = ChargeDensityPotential(
        atoms, rho, plane="xz", box=(4.0, 5.0, 3.0), **GRID
    )
    assert in_plane.box == (4.0, 5.0, 3.0)
    with pytest.raises(NotImplementedError, match="default box"):
        ChargeDensityPotential(atoms, rho, plane="xz", box=(4.0, 3.0, 5.0), **GRID)

    # A non-orthogonal cell: the default box is its best orthogonal cell.
    skewed = graphene(vacuum=2.0)
    default = tuple(best_orthogonal_cell(skewed.cell))
    assert ChargeDensityPotential(skewed, rho, **GRID).box == pytest.approx(default)
    assert ChargeDensityPotential(skewed, rho, box=default, **GRID).box == (
        pytest.approx(default)
    )
    with pytest.raises(NotImplementedError, match="default box"):
        ChargeDensityPotential(
            skewed, rho, box=(default[0] * 2, default[1], default[2]), **GRID
        )


@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
@pytest.mark.parametrize(
    "kwargs",
    [dict(box=(8.0, 9.0, 10.0)), dict(origin=(1.0, 0.75, 0.0))],
    ids=["box", "origin"],
)
def test_gpaw_magnetics_reject_a_box_or_origin(builder, kwargs):
    calculator = _FakeCalculator(_two_atoms())
    with pytest.raises(NotImplementedError, match="default box"):
        builder(calculator, **GRID, **kwargs)


@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_magnetics_accept_their_own_box(builder):
    calculator = _FakeCalculator(_two_atoms())
    assert builder(calculator, box=(4.0, 3.0, 5.0), **GRID).box == (4.0, 3.0, 5.0)
    assert builder(calculator, origin=(0.0, 0.0, 0.0), **GRID).box == (4.0, 3.0, 5.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda **kwargs: ChargeDensityPotential(
            _two_atoms(), _charge_density(), **GRID, **kwargs
        ),
        lambda **kwargs: GPAWMagneticField(
            _FakeCalculator(_two_atoms()), **GRID, **kwargs
        ),
        lambda **kwargs: GPAWVectorPotential(
            _FakeCalculator(_two_atoms()), **GRID, **kwargs
        ),
    ],
    ids=["charge-density", "magnetic-field", "vector-potential"],
)
class TestRejectingBuildersValidateTheirArguments:
    def test_origin_none_is_the_zero_origin(self, build):
        assert build(origin=None).box == (4.0, 3.0, 5.0)

    @pytest.mark.parametrize(
        "origin", [(1.0, 0.5), ("1", "0", "0"), (np.nan, 0.0, 0.0), 1.0]
    )
    def test_invalid_origin_raises(self, build, origin):
        with pytest.raises(ValueError, match="origin"):
            build(origin=origin)

    @pytest.mark.parametrize("box", [("4", "3", "5"), (4.0, 3.0), (4.0, np.nan, 5.0)])
    def test_invalid_box_raises(self, build, box):
        with pytest.raises(ValueError, match="box"):
            build(box=box)
