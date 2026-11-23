"""What `==` on abTEM objects (`safe_equality`) tells apart."""

import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem.core.utils import EqualityMixin


def _atoms():
    return bulk("Si", cubic=True)


def _potential(**kwargs):
    kwargs.setdefault("gpts", (64, 64))
    kwargs.setdefault("slice_thickness", 1.0)
    return abtem.Potential(_atoms(), **kwargs)


class Holder(EqualityMixin):
    def __init__(self, value):
        self._value = value


def test_a_held_object_compares_as_it_does_on_its_own_in_both_orders():
    """A nested object is compared with `==`, so a subclass of the other operand's
    class is dispatched as at the top level, and the verdict does not depend on
    which of two held objects is on the left. A `PotentialArray` and a
    `TransmissionFunction` (its subclass) with the same array differ, whichever way
    round. Device-independent."""
    from abtem.multislice import MultisliceTransform

    transmission = _potential(gpts=(16, 16)).build(lazy=False)
    transmission = transmission.transmission_function(100e3)
    potential = abtem.PotentialArray(
        transmission.array.copy(),
        slice_thickness=transmission.slice_thickness,
        extent=transmission.extent,
    )

    assert potential != transmission
    assert transmission != potential
    assert Holder(potential) != Holder(transmission)
    assert Holder(transmission) != Holder(potential)
    assert MultisliceTransform(potential) != MultisliceTransform(transmission)
    assert MultisliceTransform(transmission) != MultisliceTransform(potential)


def test_a_held_object_is_compared_with_its_own_eq():
    """A check that the held object's class adds to `__eq__` runs when it is held
    by another object. Device-independent."""

    class Strict(EqualityMixin):
        def __init__(self, tag):
            self.tag = tag

        def __eq__(self, other):
            return False

    assert Strict(1) != Strict(1)
    assert Holder(Strict(1)) != Holder(Strict(1))


@pytest.mark.parametrize("error", [KeyError, TypeError, ValueError])
def test_a_held_object_whose_own_comparison_raises_is_a_difference(error):
    """A `KeyError`, `TypeError` or `ValueError` from a held object's `__eq__` is a
    difference. Device-independent."""

    class Inner(EqualityMixin):
        def __eq__(self, other):
            raise error("not comparable")

    assert Holder(Inner()) != Holder(Inner())


@pytest.mark.parametrize(
    "kind", ["FrozenPhonons", "Potential", "CrystalPotential", "NoiseTransform"]
)
def test_seeds_a_few_parts_per_million_apart_are_different(kind):
    """Integers compare exactly. `np.allclose` takes seeds that differ by one in a
    million as equal, so objects that draw different random numbers compared
    equal. Device-independent."""
    from abtem.noise import NoiseTransform

    seeds = (1_000_000, 2_000_000)
    near = (1_000_001, 2_000_000)

    def make(seeds):
        atoms = _atoms()
        if kind == "NoiseTransform":
            return NoiseTransform(dose=1e3, seeds=seeds)
        frozen_phonons = abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=seeds)
        if kind == "FrozenPhonons":
            return frozen_phonons
        if kind == "Potential":
            return abtem.Potential(
                frozen_phonons, gpts=(16, 16), slice_thickness=atoms.cell[2, 2] / 4
            )
        unit = abtem.Potential(
            abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1),
            gpts=(16, 16),
            slice_thickness=atoms.cell[2, 2] / 4,
        )
        return abtem.CrystalPotential(unit, (2, 2, 1), seeds=seeds)

    assert make(seeds) != make(near)
    assert make(seeds) == make(seeds)


@pytest.mark.parametrize(
    "a, b",
    [
        ({"energy": 100_000}, {"energy": 100e3}),
        ({"sampling": 1, "gpts": None}, {"sampling": 1.0, "gpts": None}),
        ({"semiangle_cutoff": 20}, {"semiangle_cutoff": 20.0}),
    ],
    ids=["energy", "sampling", "semiangle_cutoff"],
)
def test_an_integer_and_the_same_float_still_compare_equal(a, b):
    """Only a pair of integers compares exactly; an integer against a float keeps
    the tolerance of `np.allclose`. Device-independent."""

    def probe(**kwargs):
        kwargs = {"semiangle_cutoff": 20, "gpts": 64, "extent": 10, **kwargs}
        return abtem.Probe(**{k: v for k, v in kwargs.items() if v is not None})

    assert probe(**a) == probe(**b)


def test_integers_compare_by_value_whatever_the_container():
    """A list and a tuple of the same integers are equal, as are an array and a
    list, and differing integers are not. Device-independent."""
    assert Holder([7]) == Holder((7,))
    assert Holder(np.array([7, 8])) == Holder([7, 8])
    assert Holder(1_000_000) != Holder(1_000_001)
    assert Holder((1_000_000, 2)) != Holder((1_000_001, 2))
    assert Holder(np.array([1_000_000, 2])) != Holder(np.array([1_000_001, 2]))
    assert Holder(True) != Holder(False)
