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
