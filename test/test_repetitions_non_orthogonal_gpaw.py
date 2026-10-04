"""`GPAWPotential(repetitions=...)` lays out the box of the repeated crystal."""

import pytest
from ase.build import graphene

pytest.importorskip("gpaw")

import abtem  # noqa: E402
from abtem.potentials.gpaw import GPAWPotential  # noqa: E402


@pytest.fixture(scope="module")
def calculator():
    from gpaw import GPAW, PW

    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    atoms.calc = GPAW(mode=PW(250), kpts=(2, 2, 1), txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@pytest.mark.parametrize("repetitions", [(2, 1, 1), (2, 3, 2), (1, 2, 1)])
def test_gpaw_potential_box_is_that_of_the_repeated_crystal(calculator, repetitions):
    # Only the layout is tested here: the values of GPAWPotential(repetitions)
    # are a separate matter.
    atoms = calculator.atoms
    with abtem.config.set({"device": "cpu"}):
        potential = GPAWPotential(calculator, repetitions=repetitions, sampling=0.1)
        reference = abtem.Potential(atoms * repetitions, sampling=0.1)

    assert potential.box == pytest.approx(reference.box)
    assert potential.extent == pytest.approx(reference.extent)
