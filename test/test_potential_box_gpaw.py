"""`GPAWPotential` places its field in the default box at the default origin."""

import warnings

import pytest
from ase import Atoms

pytest.importorskip("gpaw")

import abtem  # noqa: E402
from abtem.potentials.gpaw import GPAWPotential  # noqa: E402

CELL = (3.2, 2.8, 3.6)


@pytest.fixture(scope="module")
def calculator():
    from gpaw import GPAW, PW

    atoms = Atoms(
        "CO",
        positions=[(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)],
        cell=CELL,
        pbc=True,
    )
    atoms.calc = GPAW(mode=PW(250), h=0.2, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture(autouse=True)
def _cpu():
    with abtem.config.set({"device": "cpu"}):
        yield


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(box=(6.4, 2.8, 3.6)),
        dict(box=(3.2, 2.8, 4.0)),
        dict(origin=(1.0, 0.0, 0.0)),
        dict(origin=(0.0, 0.0, 0.5)),
    ],
    ids=["box", "box-z", "origin", "origin-z"],
)
def test_gpaw_potential_rejects_a_box_or_origin(calculator, kwargs):
    with pytest.raises(NotImplementedError, match="default box"):
        GPAWPotential(calculator, sampling=0.1, **kwargs)


def test_gpaw_potential_accepts_its_own_box(calculator):
    default = GPAWPotential(calculator, sampling=0.1)
    own = GPAWPotential(calculator, box=CELL, sampling=0.1)
    assert own.box == default.box == CELL


def test_gpaw_potential_box_follows_the_repetitions(calculator):
    repeated = (6.4, 2.8, 3.6)
    potential = GPAWPotential(
        calculator, repetitions=(2, 1, 1), box=repeated, sampling=0.1
    )
    assert potential.box == pytest.approx(repeated)
    with pytest.raises(NotImplementedError, match="default box"):
        GPAWPotential(calculator, repetitions=(2, 1, 1), box=CELL, sampling=0.1)


def test_gpaw_potential_origin_none_is_the_zero_origin(calculator):
    assert GPAWPotential(calculator, origin=None, sampling=0.1).box == CELL


@pytest.mark.parametrize("origin", [(1.0, 0.5), ("1", "0", "0"), (float("nan"), 0, 0)])
def test_gpaw_potential_invalid_origin_raises(calculator, origin):
    with pytest.raises(ValueError, match="origin"):
        GPAWPotential(calculator, origin=origin, sampling=0.1)


def test_gpaw_potential_box_of_strings_raises(calculator):
    with pytest.raises(ValueError, match="box"):
        GPAWPotential(calculator, box=("3.2", "2.8", "3.6"), sampling=0.1)


@pytest.fixture(scope="module")
def hexagonal_calculator():
    from ase.build import graphene
    from gpaw import GPAW, PW

    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    atoms.calc = GPAW(mode=PW(250), h=0.25, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


def _chosen_box_warnings(records):
    return [r for r in records if "abTEM chose" in str(r.message)]


@pytest.mark.filterwarnings("ignore:The box .* is not a whole supercell:UserWarning")
@pytest.mark.parametrize(
    "repetitions, warns",
    [((1, 1, 1), False), ((2, 1, 1), False), ((3, 1, 1), True), ((4, 1, 1), True)],
)
def test_gpaw_potential_reports_a_strained_default_box(
    hexagonal_calculator, repetitions, warns
):
    # The calculator's cell is repeated, and the default box of the repeated
    # hexagonal cell is exact for (1, 1, 1) and (2, 1, 1) only.
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = GPAWPotential(
            hexagonal_calculator, repetitions=repetitions, sampling=0.2
        )
        potential.build(lazy=True).compute()
    assert len(_chosen_box_warnings(records)) == int(warns)


@pytest.fixture(scope="module")
def spinpolarised_calculator():
    from ase.build import graphene
    from gpaw import GPAW, PW

    atoms = graphene(formula="BN", a=2.5, vacuum=2.0) * (3, 1, 1)
    atoms.pbc = True
    atoms.set_initial_magnetic_moments([0.3] * len(atoms))
    atoms.calc = GPAW(
        mode=PW(200),
        kpts=(1, 2, 1),
        spinpol=True,
        txt=None,
        symmetry="off",
        convergence={"density": 1e-3},
        maxiter=60,
    )
    atoms.get_potential_energy()
    return atoms.calc


@pytest.mark.filterwarnings("ignore:The box .* is not a whole supercell:UserWarning")
@pytest.mark.parametrize("include_magnetic_field", [False, True])
def test_gpaw_magnetic_fields_report_the_strained_default_box_once(
    spinpolarised_calculator, include_magnetic_field
):
    # The potential and the vector potential (and the magnetic field) are built
    # from one calculator in one call and share one default box, so the strain
    # of that box is reported once, at the call.
    from abtem.magnetism.gpaw import gpaw_magnetic_fields

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        gpaw_magnetic_fields(
            spinpolarised_calculator,
            sampling=0.2,
            include_magnetic_field=include_magnetic_field,
        )
    chosen = _chosen_box_warnings(records)
    assert len(chosen) == 1
    assert chosen[0].filename == __file__
