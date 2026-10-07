"""The box that abTEM picks itself is reported when it strains the atoms."""

import warnings

import numpy as np
import pytest
from ase.build import graphene

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.magnetism.gpaw import GPAWMagneticField, GPAWVectorPotential
from abtem.potentials.charge_density import ChargeDensityPotential

GRID = dict(sampling=0.2, slice_thickness=1.0)

# The silent cases are asserted from the warnings the construction gives, so none
# of them may raise; a warning of another kind is not looked at.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The box .* is not a whole supercell:UserWarning"
)

# Hexagonal BN repeated along x, as (nx, ny, 1), with the default box its
# repetitions give: exact for (1, 1, 1) and (2, 1, 1), strained for the others.
STRAINED = [(3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1)]
EXACT = [(1, 1, 1), (2, 1, 1), (3, 2, 1), (2, 3, 1), (4, 2, 1), (5, 2, 1), (3, 3, 1)]


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64", "fft": "numpy"}):
        yield


def _bn(repetitions=(1, 1, 1)):
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    return atoms * repetitions


def _chosen_box_warnings(records):
    return [r for r in records if "abTEM chose" in str(r.message)]


def _construct(builder, *args, **kwargs):
    """The builder, and the warnings about the box abTEM chose that its
    construction gave."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        built = builder(*args, **kwargs)
    return built, _chosen_box_warnings(records)


def _strain_from_the_transform(atoms, box):
    """Stretch of each supercell vector onto the box, and the cosines of the
    angles between them, from the affine map `orthogonalize_cell` applies: it
    takes the supercell vectors v to the box edges, v @ A = diag(box)."""
    _, transform = orthogonalize_cell(atoms, box=box, return_transform_matrix=True)
    supercell = np.diag(box) @ np.linalg.inv(transform)
    lengths = np.linalg.norm(supercell, axis=1)
    unit = supercell / lengths[:, None]
    cosines = [unit[1] @ unit[2], unit[0] @ unit[2], unit[0] @ unit[1]]
    return np.asarray(box) / lengths - 1.0, np.array(cosines), lengths


def test_default_box_of_bn_repeated_three_times_warns_with_the_numbers():
    atoms = _bn((3, 1, 1))
    potential, records = _construct(abtem.Potential, atoms, **GRID)

    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)
    assert records[0].filename == __file__
    message = str(records[0].message)
    stretch, cosines, lengths = _strain_from_the_transform(atoms, potential.box)
    assert 100 * stretch == pytest.approx([1.379, -0.660, 0.0], abs=1e-3)
    assert cosines[2] == pytest.approx(0.115, abs=1e-3)
    for value in stretch:
        assert f"{100 * value:+.3f} %" in message
    for value in cosines:
        assert f"{value:.2e}" in message
    for value in np.degrees(np.arccos(cosines)):
        assert f"{value:.3f}°" in message
    assert str(round(float(lengths[0]), 6)) in message
    assert "[[1, 0, 0], [1, 5, 0], [0, 0, 1]]" in message
    assert str(tuple(float(b) for b in potential.box)) in message


def test_warning_says_who_chose_the_box_and_names_the_remedies():
    _, records = _construct(abtem.Potential, _bn((3, 1, 1)), **GRID)

    message = str(records[0].message)
    assert "abTEM chose because none was given" in message
    assert "Pass a `box` that is a whole supercell" in message
    assert "repeat the atoms' cell so that an orthogonal supercell" in message
    assert "at most 5 repetitions" in message
    assert f"{abtem.atoms.BOX_STRAIN_WARNING_THRESHOLD:.1e}" in message


def test_default_box_of_bn_repeated_five_times_warns():
    atoms = _bn((5, 1, 1))
    potential, records = _construct(abtem.Potential, atoms, **GRID)

    assert len(records) == 1
    stretch, _, _ = _strain_from_the_transform(atoms, potential.box)
    assert 100 * stretch[1] == pytest.approx(-13.397, abs=1e-3)
    assert f"{100 * stretch[1]:+.3f} %" in str(records[0].message)


@pytest.mark.parametrize("repetitions", STRAINED)
def test_strained_default_box_warns_once(repetitions):
    _, records = _construct(abtem.Potential, _bn(repetitions), **GRID)
    assert len(records) == 1


@pytest.mark.parametrize("repetitions", EXACT)
def test_exact_default_box_is_silent(repetitions):
    # (3, 2, 1), (2, 3, 1), ...: different repetitions along x and y whose
    # lattice vectors still make an exact orthogonal supercell.
    _, records = _construct(abtem.Potential, _bn(repetitions), **GRID)
    assert records == []


def test_default_box_with_different_repetitions_along_x_and_y_is_judged_by_the_cell():
    _, strained = _construct(abtem.Potential, _bn((4, 1, 1)), **GRID)
    _, exact = _construct(abtem.Potential, _bn((3, 2, 1)), **GRID)
    assert len(strained) == 1
    assert exact == []


def test_non_periodic_default_box_is_silent():
    for repetitions in STRAINED:
        _, records = _construct(
            abtem.Potential, _bn(repetitions), periodic=False, **GRID
        )
        assert records == []


def test_box_given_by_the_user_is_not_reported_as_chosen_by_abtem():
    atoms = _bn((3, 1, 1))
    default = abtem.Potential(atoms, **GRID).box
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        abtem.Potential(atoms, box=(default[0] * 1.01, default[1], 4.0), **GRID)
    assert _chosen_box_warnings(records) == []
    assert [r for r in records if str(r.message).startswith("The box")]


def test_warning_is_given_once_by_a_frozen_phonon_potential_and_its_blocks():
    frozen_phonons = abtem.FrozenPhonons(
        _bn((3, 1, 1)), num_configs=2, sigmas=0.05, seed=1
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = abtem.Potential(frozen_phonons, **GRID)
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_chosen_box_warnings(records)) == 1


def test_orthogonalize_cell_without_a_box_does_not_warn():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        orthogonalize_cell(_bn((3, 1, 1)))
    assert _chosen_box_warnings(records) == []
    assert not [r for r in records if "strained" in str(r.message)]


def _charge_density():
    shape = (16, 12, 20)
    x, y, z = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
    return 0.3 + 0.1 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


class _FakeCalculator:
    # The part of a GPAW calculator that the GPAW magnetics read when they are
    # constructed.
    def __init__(self, atoms):
        self.atoms = atoms

    def get_number_of_grid_points(self):
        return np.array([16, 12, 20])


_GPAW_FAMILY = [
    pytest.param(
        lambda atoms: ChargeDensityPotential(
            atoms, _charge_density(), sampling=0.2, slice_thickness=1.0
        ),
        id="charge-density",
    ),
    pytest.param(
        lambda atoms: GPAWMagneticField(
            _FakeCalculator(atoms), sampling=0.2, slice_thickness=1.0
        ),
        id="magnetic-field",
    ),
    pytest.param(
        lambda atoms: GPAWVectorPotential(
            _FakeCalculator(atoms), sampling=0.2, slice_thickness=1.0
        ),
        id="vector-potential",
    ),
]


@pytest.mark.parametrize("build", _GPAW_FAMILY)
@pytest.mark.parametrize("repetitions", [(3, 1, 1), (5, 1, 1), (4, 1, 1)])
def test_gpaw_family_warns_for_a_strained_default_box(build, repetitions):
    _, records = _construct(build, _bn(repetitions))
    assert len(records) == 1


@pytest.mark.parametrize("build", _GPAW_FAMILY)
@pytest.mark.parametrize("repetitions", [(1, 1, 1), (2, 1, 1), (3, 2, 1)])
def test_gpaw_family_is_silent_for_an_exact_default_box(build, repetitions):
    _, records = _construct(build, _bn(repetitions))
    assert records == []


@pytest.mark.parametrize("repetitions", [(3, 1, 1), (4, 1, 1)])
def test_charge_density_repetitions_are_judged_by_the_repeated_cell(repetitions):
    _, records = _construct(
        ChargeDensityPotential,
        _bn(),
        _charge_density(),
        sampling=0.2,
        slice_thickness=1.0,
        repetitions=repetitions,
    )
    assert len(records) == 1
    assert "[[1, 0, 0], [1, " in str(records[0].message)


def test_charge_density_does_not_repeat_the_warning_when_it_builds_the_ewald_field():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = ChargeDensityPotential(
            _bn(),
            _charge_density(),
            sampling=0.2,
            slice_thickness=1.0,
            repetitions=(3, 1, 1),
        )
        potential.build(lazy=False)
        potential.build(lazy=True).compute()
    assert len(_chosen_box_warnings(records)) == 1


def test_cell_that_cannot_be_rotated_to_the_plane_is_constructed_silently():
    # The hexagonal cell has no vertical lattice vector once rotated to xz; the
    # potential is constructed as before, and the check of the default box does
    # not raise in its place.
    atoms = _bn()
    atoms.pbc = (False, False, True)
    _, records = _construct(abtem.Potential, atoms, plane="xz", **GRID)
    assert records == []


@pytest.mark.parametrize(
    "build",
    [
        lambda atoms, box: ChargeDensityPotential(
            atoms, _charge_density(), box=box, sampling=0.2, slice_thickness=1.0
        ),
        lambda atoms, box: GPAWMagneticField(
            _FakeCalculator(atoms), box=box, sampling=0.2, slice_thickness=1.0
        ),
        lambda atoms, box: GPAWVectorPotential(
            _FakeCalculator(atoms), box=box, sampling=0.2, slice_thickness=1.0
        ),
    ],
    ids=["charge-density", "magnetic-field", "vector-potential"],
)
def test_gpaw_family_does_not_report_its_default_box_when_it_is_given(build):
    atoms = _bn((3, 1, 1))
    default = abtem.Potential(atoms, **GRID).box
    _, records = _construct(build, atoms, default)
    assert records == []
