"""What `==` on abTEM objects (`safe_equality`) tells apart."""

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

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


@pytest.mark.parametrize(
    "a, b",
    [
        ((), (1,)),
        ([], [5]),
        (7, (7, 7)),
        (7, (7,)),
        ((5,), (5, 5)),
        ((1.0,), (1.0, 1.0)),
        (np.arange(7), np.tile(np.arange(7), (7, 1))),
        (np.zeros(3), np.zeros((2, 3))),
        (np.array(["a"]), np.array(["a", "a"])),
    ],
    ids=[
        "empty-vs-one",
        "empty-list-vs-one",
        "number-vs-pair",
        "number-vs-one",
        "one-vs-two",
        "float-one-vs-two",
        "int-array-vs-tile",
        "float-array-vs-stack",
        "string-array",
    ],
)
def test_values_of_different_shapes_are_different(a, b):
    """`np.equal` and `np.allclose` broadcast, so `(5,)` equalled `(5, 5)` and an
    empty sequence equalled anything. The array pairs differ in size along both
    axes. Device-independent."""
    assert Holder(a) != Holder(b)
    assert Holder(b) != Holder(a)


@pytest.mark.parametrize(
    "a, b",
    [
        ([7], (7,)),
        (np.arange(3), [0, 1, 2]),
        (((1, 2), (3,)), ((1, 2), (3,))),
        ((1.0,), (1.0 + 1e-9,)),
        (np.zeros((2, 3)), np.zeros((2, 3))),
    ],
    ids=["list-vs-tuple", "array-vs-list", "ragged", "float-tolerance", "array"],
)
def test_values_of_the_same_shape_still_compare_equal(a, b):
    """The shape check adds no difference between values of one shape: a list and a
    tuple, an array and a list, identical ragged tuples and floats within
    `np.allclose`'s tolerance stay equal, and different ragged tuples stay
    different. Device-independent."""
    assert Holder(a) == Holder(b)
    assert Holder(((1, 2), (3,))) != Holder(((1, 2), (4,)))


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_arrays_of_different_shapes_are_different_on_any_device(device, dtype):
    """`(3, 4)` broadcasts against `(1, 4)`, so the values alone cannot tell them
    apart, and `(5, 4)` does not broadcast at all; the shapes decide, whichever
    side the host array is on."""
    from abtem.measurements import Images

    def images(shape, on_device=True):
        images = Images(np.zeros(shape, dtype=dtype), sampling=0.1)
        return images.copy_to_device(device) if on_device else images

    for shape in ((1, 4), (5, 4)):
        a, b = images((3, 4)), images(shape)
        host = images(shape, on_device=False)
        assert a != b
        assert b != a
        assert host != a
        assert a != host
    assert images((3, 4)) == images((3, 4))


@pytest.mark.parametrize("kind", ["FrozenPhonons", "NoiseTransform"])
def test_one_seed_against_two_equal_seeds_is_different(kind):
    """A `FrozenPhonons` of one configuration and one of two with the same seed, and
    noise transforms with seeds `(5,)` and `(5, 5)`, draw different numbers of
    random streams. Device-independent."""
    from abtem.noise import NoiseTransform

    def make(seeds):
        if kind == "NoiseTransform":
            return NoiseTransform(dose=1e3, seeds=seeds)
        return abtem.FrozenPhonons(_atoms(), len(seeds), sigmas=0.1, seed=seeds)

    assert make((5,)) != make((5, 5))
    assert make((5, 5)) != make((5,))
    assert make((5, 5)) == make((5, 5))
