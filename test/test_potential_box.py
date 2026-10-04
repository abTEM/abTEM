"""A potential's `box`, `origin` and auto grid follow the arguments given."""

import pickle
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene

import abtem
from abtem.atoms import (
    best_orthogonal_cell,
    cut_cell,
    orthogonalize_cell,
)
from abtem.magnetism.gpaw import GPAWMagneticField, GPAWVectorPotential
from abtem.inelastic.phonons import FrozenPhonons
from abtem.magnetism.iam import MagneticField
from abtem.potentials.charge_density import ChargeDensityPotential

GRID = dict(sampling=0.1, slice_thickness=1.0)

# The strained boxes of these tests are reported by a warning, which the tests of
# the warning itself collect; the others do not look at it.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The box .* is not a whole supercell:UserWarning"
)


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


def _permuted_two_atoms():
    # `_two_atoms()` with the y and z axes exchanged, as plane="xz" sees it.
    atoms = _two_atoms()
    return Atoms(
        atoms.numbers,
        positions=atoms.positions[:, [0, 2, 1]],
        cell=[4.0, 5.0, 3.0],
        pbc=True,
    )


def _array(atoms, **kwargs):
    return abtem.Potential(atoms, **kwargs).build(lazy=False).array


def _assert_same(actual, expected):
    assert actual.shape == expected.shape
    scale = np.abs(expected).max()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10 * scale)


def _integral(potential):
    return potential.build(lazy=False).array.sum() * np.prod(potential.sampling)


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


def _supercell_cases():
    atoms = _two_atoms()
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    orthogonal_graphene = orthogonalize_cell(graphene(vacuum=2.0))
    return [
        ("orthogonal 2x3x2", atoms, (2, 3, 2)),
        ("cubic Si 2x3x1", si, (2, 3, 1)),
        ("graphene 3x1x1", graphene(vacuum=2.0), (3, 1, 1), orthogonal_graphene),
    ]


@pytest.mark.parametrize("projection", ["infinite", "finite"])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("case", _supercell_cases(), ids=lambda c: c[0])
def test_supercell_box_matches_repeated_atoms(case, periodic, projection):
    name, atoms, repetitions, *orthogonal = case
    repeated = (orthogonal[0] if orthogonal else atoms) * repetitions
    box = tuple(np.diag(repeated.cell))

    potential = abtem.Potential(
        atoms, box=box, periodic=periodic, projection=projection, **GRID
    )

    assert potential.box == pytest.approx(box, rel=1e-12)
    _assert_same(
        potential.build(lazy=False).array,
        _array(repeated, projection=projection, **GRID),
    )


@pytest.mark.parametrize("projection", ["infinite", "finite"])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("box, cells", [((8.0, 9.0, 10.0), 12), ((12.0, 3.0, 5.0), 3)])
def test_projected_potential_integral_counts_the_cells_in_the_box(
    box, cells, periodic, projection
):
    # Each atom's projected potential integrates to the same value wherever it
    # sits, so the integral over a box that holds whole cells counts them.
    atoms = _two_atoms()
    one_cell = abtem.Potential(atoms, projection=projection, **GRID)
    in_box = abtem.Potential(
        atoms, box=box, periodic=periodic, projection=projection, **GRID
    )

    assert _integral(in_box) / _integral(one_cell) == pytest.approx(cells, rel=1e-9)


def test_projected_potential_integral_of_a_strained_box_counts_its_periods():
    # A 9 A box holds 2 periods of the 4 A axis and 3 of the 3 A axis, strained
    # onto the box, so the integral counts 2 x 3 x 2 cells. The infinite
    # projection does not depend on where the strain puts an atom relative to a
    # slice boundary.
    atoms = _two_atoms()
    one_cell = abtem.Potential(atoms, **GRID)
    in_box = abtem.Potential(atoms, box=(9.0, 9.0, 10.0), **GRID)

    assert _integral(in_box) / _integral(one_cell) == pytest.approx(12, rel=1e-9)


def test_box_that_is_not_a_supercell_strains_the_atoms_onto_it():
    atoms = _two_atoms()
    box = (9.0, 9.0, 10.0)
    potential = abtem.Potential(atoms, box=box, **GRID)
    assert potential.extent == pytest.approx(box[:2])
    _assert_same(
        potential.build(lazy=False).array,
        _array(orthogonalize_cell(atoms, box=box), **GRID),
    )


def test_box_with_one_period_compresses_the_cell_onto_it():
    atoms = _two_atoms()
    box = (2.1, 3.0, 5.0)
    potential = abtem.Potential(atoms, box=box, **GRID)
    assert potential.box == pytest.approx(box)
    assert potential.get_transformed_atoms().cell.lengths() == pytest.approx(box)
    _assert_same(
        potential.build(lazy=False).array,
        _array(orthogonalize_cell(atoms, box=box), **GRID),
    )


@pytest.mark.parametrize("box", [None, (4.0, 3.0, 5.0)])
@pytest.mark.parametrize("kind", [tuple, list, np.array])
def test_origin_translates_an_orthogonal_cell(box, kind):
    atoms = _two_atoms()
    origin = (1.0, 0.5, 0.0)
    translated = atoms.copy()
    translated.translate(-np.array(origin))
    translated.wrap()

    _assert_same(
        _array(atoms, origin=kind(origin), box=box, **GRID),
        _array(translated, **GRID),
    )


def test_origin_with_a_plane_translates_the_permuted_atoms():
    atoms = _two_atoms()
    origin = (1.0, 0.5, 0.25)
    # The origin is given relative to the atoms as provided: they are translated
    # first, then the plane is mapped to xy.
    translated = atoms.copy()
    translated.translate(-np.array(origin))
    translated.wrap()
    permuted = Atoms(
        translated.numbers,
        positions=translated.positions[:, [0, 2, 1]],
        cell=[4.0, 5.0, 3.0],
        pbc=True,
    )

    _assert_same(
        _array(atoms, plane="xz", origin=origin, **GRID), _array(permuted, **GRID)
    )


def test_zero_origin_given_as_a_list_changes_nothing():
    atoms = _two_atoms()
    _assert_same(_array(atoms, origin=[0.0, 0.0, 0.0], **GRID), _array(atoms, **GRID))


def test_plane_and_box_together():
    atoms = _two_atoms()
    potential = abtem.Potential(atoms, plane="xz", box=(8.0, 10.0, 3.0), **GRID)
    assert potential.box == (8.0, 10.0, 3.0)
    _assert_same(
        potential.build(lazy=False).array,
        _array(_permuted_two_atoms() * (2, 2, 1), **GRID),
    )


def test_box_equal_to_the_rotated_cell_with_a_plane_changes_nothing():
    # The box that a plane gives by default describes the rotated cell, not the
    # cell as it is given.
    atoms = _two_atoms()
    _assert_same(
        _array(atoms, plane="xz", box=(4.0, 5.0, 3.0), **GRID),
        _array(atoms, plane="xz", **GRID),
    )


def test_box_equal_to_the_unrotated_cell_with_a_plane_strains_the_rotated_cell():
    # (4, 3, 5) is the cell's own diagonal, but with plane="xz" the potential's
    # axes are the cell's x, z, y: the rotated 4 x 5 x 3 A cell is strained onto
    # it.
    atoms = _two_atoms()
    box = (4.0, 3.0, 5.0)
    potential = abtem.Potential(atoms, plane="xz", box=box, **GRID)
    assert potential.box == box
    _assert_same(
        potential.build(lazy=False).array,
        _array(orthogonalize_cell(_permuted_two_atoms(), box=box), **GRID),
    )


def test_box_equal_to_the_cell_changes_nothing():
    atoms = _two_atoms()
    _assert_same(_array(atoms, box=(4.0, 3.0, 5.0), **GRID), _array(atoms, **GRID))


def test_auto_sampling_and_slice_thickness_follow_the_box():
    atoms = _two_atoms()
    repeated = atoms * (2, 3, 2)
    in_box = abtem.Potential(
        atoms, box=(8.0, 9.0, 10.0), sampling="auto", slice_thickness="auto"
    )
    reference = abtem.Potential(repeated, sampling="auto", slice_thickness="auto")
    assert in_box.extent == pytest.approx((8.0, 9.0))
    assert in_box.gpts == reference.gpts
    assert in_box.slice_thickness == pytest.approx(reference.slice_thickness)


def test_auto_sampling_follows_the_box_of_a_strained_cell():
    atoms = _two_atoms()
    box = (9.0, 9.0, 10.0)
    in_box = abtem.Potential(atoms, box=box, sampling="auto", slice_thickness="auto")
    reference = abtem.Potential(
        orthogonalize_cell(atoms, box=box), sampling="auto", slice_thickness="auto"
    )
    assert in_box.gpts == reference.gpts
    assert in_box.slice_thickness == pytest.approx(reference.slice_thickness)
    assert sum(in_box.slice_thickness) == pytest.approx(box[2])


def test_auto_slice_thickness_follows_the_plane():
    atoms = _two_atoms()
    potential = abtem.Potential(atoms, plane="xz", sampling=0.1, slice_thickness="auto")
    reference = abtem.Potential(
        _permuted_two_atoms(), sampling=0.1, slice_thickness="auto"
    )
    assert potential.slice_thickness == pytest.approx(reference.slice_thickness)


def test_auto_slice_thickness_of_a_primitive_fcc_cell_fills_its_box():
    # The primitive cell is non-orthogonal; the slices fill the best orthogonal
    # cell the potential is built in.
    atoms = bulk("Si", "diamond", a=5.431)
    potential = abtem.Potential(atoms, sampling=0.1, slice_thickness="auto")
    assert sum(potential.slice_thickness) == pytest.approx(potential.box[2])


@pytest.mark.parametrize(
    "box", [(8.0, 9.0), (8.0, 0.0, 10.0), (8.0, -9.0, 10.0), (8.0, np.nan, 10.0), "abc"]
)
def test_invalid_box_raises(box):
    with pytest.raises(ValueError):
        abtem.Potential(_two_atoms(), box=box, **GRID)


@pytest.mark.parametrize("sampling", [0.1, "auto"])
def test_periodic_box_with_no_whole_period_raises_at_construction(sampling):
    with pytest.raises(ValueError, match="no whole repetition"):
        abtem.Potential(_two_atoms(), box=(1.5, 3.0, 5.0), sampling=sampling)


def test_whole_period_of_a_box_is_counted_in_the_frame_of_the_plane():
    # With plane="xz" the potential's axes are the cell's x, z, y (4, 5, 3 A), so
    # the 2 A along y holds no period of the 5 A axis it is laid over; in the
    # unrotated frame it holds a period of the 3 A axis.
    atoms = _two_atoms()
    box = (4.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="no whole repetition"):
        abtem.Potential(atoms, plane="xz", box=box, **GRID)
    assert abtem.Potential(atoms, box=box, **GRID).box == box


def test_non_periodic_box_with_no_whole_period_is_cut_out():
    atoms = _two_atoms()
    box = (1.5, 3.0, 5.0)
    potential = abtem.Potential(atoms, box=box, periodic=False, **GRID)
    array = potential.build(lazy=False).array
    assert array.shape == (5, 15, 30)
    _assert_same(array, _array(cut_cell(atoms, cell=box), **GRID))


@pytest.mark.parametrize("box", [(1.5, 3.0, 5.0), (9.0, 9.0, 10.0), (8.0, 9.0, 10.0)])
def test_non_periodic_auto_grid_follows_the_atoms_cut_out_of_the_box(box):
    # The cut-out atoms are not strained, so the grid commensurate with them is
    # the one of the atoms cut out of the repeated structure.
    atoms = _two_atoms()
    potential = abtem.Potential(
        atoms, box=box, periodic=False, sampling="auto", slice_thickness="auto"
    )
    reference = abtem.Potential(
        cut_cell(atoms, cell=box), sampling="auto", slice_thickness="auto"
    )
    assert potential.gpts == reference.gpts
    assert potential.sampling == pytest.approx(reference.sampling)
    assert potential.slice_thickness == pytest.approx(reference.slice_thickness)


def test_invalid_origin_raises():
    for origin in [(1.0, 0.5), ("1", "0", "0"), (np.nan, 0.0, 0.0)]:
        with pytest.raises(ValueError, match="origin"):
            abtem.Potential(_two_atoms(), origin=origin, **GRID)


def test_origin_none_is_the_zero_origin():
    atoms = _two_atoms()
    _assert_same(_array(atoms, origin=None, **GRID), _array(atoms, **GRID))


def test_box_of_strings_raises():
    with pytest.raises(ValueError, match="box"):
        abtem.Potential(_two_atoms(), box=("8", "9", "10"), **GRID)


def test_plane_box_and_origin_together():
    atoms = _two_atoms()
    origin = (1.0, 0.5, 0.25)
    # The origin translates the atoms as provided, then the plane maps y and z.
    translated = atoms.copy()
    translated.translate(-np.array(origin))
    translated.wrap()
    permuted = Atoms(
        translated.numbers,
        positions=translated.positions[:, [0, 2, 1]],
        cell=[4.0, 5.0, 3.0],
        pbc=True,
    )

    potential = abtem.Potential(
        atoms, plane="xz", box=(8.0, 10.0, 3.0), origin=origin, **GRID
    )
    assert potential.box == (8.0, 10.0, 3.0)
    _assert_same(
        potential.build(lazy=False).array, _array(permuted * (2, 2, 1), **GRID)
    )


def test_frozen_phonons_through_a_box():
    atoms = _two_atoms()
    frozen_phonons = FrozenPhonons(atoms, 3, sigmas=0.1, seed=1)
    potential = abtem.Potential(frozen_phonons, box=(8.0, 9.0, 10.0), **GRID)

    eager = potential.build(lazy=False).array
    lazy = potential.build(lazy=True).compute().array

    assert eager.shape == (3, 10, 80, 90)
    _assert_same(lazy, eager)
    assert not np.allclose(eager[0], eager[1])

    # Each configuration holds the 12 cells of the box, displaced.
    cell = abtem.Potential(atoms, **GRID)
    for configuration in eager:
        integral = configuration.sum() * np.prod(potential.sampling)
        assert integral / _integral(cell) == pytest.approx(12, rel=1e-9)


def test_crystal_potential_of_a_unit_with_a_box():
    atoms = _two_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0)
    unit = abtem.Potential(atoms, box=(8.0, 9.0, 10.0), **kwargs)
    crystal = abtem.CrystalPotential(unit, repetitions=(2, 1, 1))
    reference = abtem.Potential(atoms * (4, 3, 2), **kwargs)

    assert crystal.box == pytest.approx((16.0, 9.0, 10.0))
    _assert_same(crystal.build(lazy=False).array, reference.build(lazy=False).array)


def test_non_periodic_box_is_cut_out_of_the_repeated_atoms():
    potential = abtem.Potential(
        _two_atoms(),
        box=(8.0, 9.0, 10.0),
        periodic=False,
        projection="finite",
        **GRID,
    )
    assert potential.box == (8.0, 9.0, 10.0)
    assert potential.get_transformed_atoms().cell.lengths() == pytest.approx(
        (8.0, 9.0, 10.0)
    )


def test_multislice_through_a_box_matches_the_repeated_atoms():
    atoms = _two_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0)
    in_box = abtem.Potential(atoms, box=(8.0, 9.0, 10.0), **kwargs)
    repeated = abtem.Potential(atoms * (2, 3, 2), **kwargs)
    wave = abtem.PlaneWave(energy=100e3)
    _assert_same(
        wave.multislice(in_box, lazy=False).array,
        wave.multislice(repeated, lazy=False).array,
    )


def test_magnetic_field_box_matches_repeated_atoms():
    atoms = _two_atoms()
    atoms.set_chemical_symbols(["Fe", "O"])
    atoms.set_array("magnetic_moments", np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
    kwargs = dict(sampling=0.2, slice_thickness=1.0)
    _assert_same(
        MagneticField(atoms, box=(8.0, 9.0, 10.0), **kwargs).build(lazy=False).array,
        MagneticField(atoms * (2, 3, 2), **kwargs).build(lazy=False).array,
    )


@pytest.mark.parametrize("stored", [None, [0.0, 0.0, 0.0], np.zeros(3)])
def test_potential_restored_with_the_origin_as_it_was_passed_builds(stored):
    # A potential restored from a pickle skips `__init__`, so it may hold the
    # origin exactly as the user passed it.
    atoms = _two_atoms()
    potential = abtem.Potential(atoms, **GRID)
    restored = pickle.loads(pickle.dumps(potential))
    restored._origin = stored

    _assert_same(restored.build(lazy=False).array, _array(atoms, **GRID))


def test_potential_restored_with_an_invalid_origin_raises_on_build():
    restored = pickle.loads(pickle.dumps(abtem.Potential(_two_atoms(), **GRID)))
    restored._origin = (1.0, 0.0)
    with pytest.raises(ValueError, match="origin"):
        restored.build(lazy=False)


def _strain_warnings(records):
    return [r for r in records if str(r.message).startswith("The box")]


def _construct(*args, **kwargs):
    """The potential, and the box-strain warnings its construction gave."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = abtem.Potential(*args, **kwargs)
    return potential, _strain_warnings(records)


def _strain_from_the_transform(atoms, box):
    """Stretch of each supercell vector onto the box, and the cosines of the angles
    between them, from the affine map `orthogonalize_cell` applies: it takes the
    supercell vectors v to the box edges, v @ A = diag(box)."""
    _, transform = orthogonalize_cell(atoms, box=box, return_transform_matrix=True)
    supercell = np.diag(box) @ np.linalg.inv(transform)
    lengths = np.linalg.norm(supercell, axis=1)
    unit = supercell / lengths[:, None]
    cosines = [unit[1] @ unit[2], unit[0] @ unit[2], unit[0] @ unit[1]]
    return np.asarray(box) / lengths - 1.0, np.array(cosines), lengths


def test_strain_warning_threshold_is_a_tenth_of_a_percent():
    assert abtem.atoms.BOX_STRAIN_WARNING_THRESHOLD == 1e-3


def test_box_that_strains_the_atoms_warns_with_the_numbers():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    potential, records = _construct(si, box=(20.0, 5.431, 5.431), sampling=0.2)

    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)
    assert records[0].filename == __file__
    message = str(records[0].message)
    # 4 periods of 5.431 A make 21.724 A; 20 A compresses them.
    stretch = 100 * (20.0 / (4 * 5.431) - 1.0)
    assert f"{stretch:+.3f} %" in message
    assert "-7.936 %" in message
    assert "+0.000 %" in message
    assert "(4, 1, 1) periods" in message
    assert "21.724" in message
    assert "20.0" in message
    assert potential.box == (20.0, 5.431, 5.431)


@pytest.mark.parametrize(
    "factor, warns",
    [(1.0011, True), (0.9989, True), (1.0009, False), (0.9991, False), (1.0, False)],
)
def test_strain_warning_threshold_applies_to_the_stretch(factor, warns):
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    box = (4 * 5.431 * factor, 5.431, 5.431)
    _, records = _construct(si, box=box, sampling=0.2)
    assert bool(records) == warns
    if warns:
        assert f"{100 * (factor - 1):+.3f} %" in str(records[0].message)


def test_a_slightly_off_box_of_four_periods_is_silent():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    _, records = _construct(si, box=(21.72, 5.431, 5.431), sampling=0.2)
    assert records == []


def test_strain_warning_quotes_the_shear_of_a_hexagonal_supercell():
    graphene_cell = graphene(vacuum=2.0)
    box = (20.0, 20.0, 4.0)
    _, records = _construct(graphene_cell, box=box, sampling=0.2)

    assert len(records) == 1
    message = str(records[0].message)
    stretch, cosines, lengths = _strain_from_the_transform(graphene_cell, box)
    assert cosines[2] == pytest.approx(0.064, abs=1e-3)
    for value in stretch:
        assert f"{100 * value:+.3f} %" in message
    for value in cosines:
        assert f"{value:.2e}" in message
    for value in np.degrees(np.arccos(cosines)):
        assert f"{value:.3f}°" in message
    assert str(round(float(lengths[0]), 6)) in message
    assert "[[8, 0, 0], [5, 9, 0], [0, 0, 1]]" in message


def test_boxes_that_are_whole_supercells_up_to_round_off_are_silent():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    hexagonal = graphene(vacuum=2.0)
    a = 2.46
    cases = [(si, (n * 5.431, m * 5.431, 5.431)) for n in (1, 2, 3, 7) for m in (1, 3)]
    # Boxes of hexagonal supercells computed in floating point.
    cases += [
        (hexagonal, (n * a, m * a * np.sqrt(3.0), 4.0))
        for n in (1, 2, 3, 5)
        for m in (1, 2, 3)
    ]
    cases += [
        (hexagonal, (n * a, m * 3.0 * a / np.sqrt(3.0), 4.0))
        for n in (2, 4)
        for m in (1, 3)
    ]
    for atoms, box in cases:
        _, records = _construct(atoms, box=box, sampling=0.5)
        assert records == [], (box, [str(r.message)[:80] for r in records])


def test_exact_default_box_of_a_non_orthogonal_cell_is_silent():
    _, records = _construct(bulk("Si", "diamond", a=5.431), sampling=0.2)
    assert records == []


def test_default_box_given_explicitly_is_not_checked():
    # The default box of this supercell is itself reached by a strain of 1.4 %,
    # 0.7 % and a shear of 0.11; it is the default, however it is spelled.
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0) * (3, 1, 1)
    default = abtem.Potential(atoms, sampling=0.2).box
    assert abs(100 * (default[0] / 7.5 - 1)) > 0.5

    _, records = _construct(atoms, box=default, sampling=0.2)
    assert records == []

    _, records = _construct(atoms, box=(default[0] * 1.01, default[1], 4.0))
    assert len(records) == 1


def test_non_periodic_box_is_not_strained_and_is_silent():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    _, records = _construct(si, box=(20.0, 5.431, 5.431), periodic=False, sampling=0.2)
    assert records == []


@pytest.mark.parametrize("builder", [abtem.Potential, MagneticField])
def test_strain_warning_is_given_once_per_construction(builder):
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    si.set_array("magnetic_moments", np.zeros((len(si), 3)))
    kwargs = dict(box=(20.0, 5.431, 5.431), sampling=0.2, slice_thickness=2.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = builder(si, **kwargs)
        potential.copy()
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_strain_warnings(records)) == 1


def test_strain_warning_is_not_repeated_by_frozen_phonons_or_lazy_blocks():
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    frozen_phonons = FrozenPhonons(si, 3, sigmas=0.05, seed=1)
    kwargs = dict(box=(20.0, 5.431, 5.431), sampling=0.2, slice_thickness=2.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = abtem.Potential(frozen_phonons, **kwargs)
        assert len(_strain_warnings(records)) == 1
        lazy = potential.build(lazy=True).compute()
        eager = potential.build(lazy=False)
        wave = abtem.PlaneWave(energy=100e3)
        wave.multislice(potential, lazy=True).compute()
    assert len(_strain_warnings(records)) == 1
    assert lazy.array.shape == eager.array.shape
    assert lazy.array.shape[:3] == (3, 3, 100)


def test_strain_warning_does_not_hide_an_error_for_a_box_with_no_whole_period():
    with pytest.raises(ValueError, match="no whole repetition"):
        abtem.Potential(_two_atoms(), box=(1.5, 3.0, 5.0), **GRID)


def test_strain_warning_is_not_repeated_by_a_crystal_potential():
    # CrystalPotential rebuilds its unit with its own frozen-phonon pool per
    # member and per enlarged pool; the unit's box was reported when it was made.
    two = _two_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        unit = abtem.Potential(
            FrozenPhonons(two, 2, sigmas=0.1, seed=1), box=(9.0, 9.0, 10.0), **kwargs
        )
        assert len(_strain_warnings(records)) == 1

        # The pool (2) is smaller than the 4 lateral tiles and is enlarged.
        tiled = abtem.CrystalPotential(unit, (2, 2, 1))
        tiled.build(lazy=False)
        tiled.build(lazy=True).compute()

        # An ensemble of members, each with its own pool.
        members = abtem.CrystalPotential(
            unit, (1, 1, 2), num_frozen_phonons=2, seeds=(5, 6)
        )
        members.build(lazy=False)
        members.build(lazy=True).compute()

    assert len(_strain_warnings(records)) == 1


@pytest.mark.parametrize("repetitions, reported", [((2, 3, 2), 0), ((3, 1, 1), 1)])
def test_charge_density_potential_reports_its_default_box_once(repetitions, reported):
    # The default box of BN x (3, 1, 1) is reached by a strain; the potential
    # reports it when it is constructed, and the Ewald potential it builds from
    # its own box does not repeat it.
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = ChargeDensityPotential(
            atoms,
            _charge_density(),
            sampling=0.2,
            slice_thickness=1.0,
            repetitions=repetitions,
        )
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_strain_warnings(records)) == reported
