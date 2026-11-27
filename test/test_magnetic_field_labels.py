"""The component axis of an unbuilt magnetic field or vector potential is labelled like
the array it builds."""

import numpy as np
import pytest
from ase import Atoms

from utils import float64_devices, lazy_params, to_host_array

from abtem import FrozenPhonons
from abtem.magnetism.iam import MagneticField, VectorPotential


@pytest.mark.parametrize(
    "cls, labels",
    [(MagneticField, ("Bx", "By", "Bz")), (VectorPotential, ("Ax", "Ay", "Az"))],
)
def test_unbuilt_component_axis_matches_the_built_array(cls, labels):
    atoms = Atoms(
        "Fe2",
        positions=[(1.0, 1.0, 1.0), (3.0, 2.0, 2.0)],
        cell=(4.0, 3.0, 3.0),
        pbc=True,
    )
    atoms.set_array("magnetic_moments", np.tile([0.0, 0.0, 2.0], (len(atoms), 1)))

    builder = cls(atoms, gpts=(10, 12), num_slices=2)
    built = builder.build()

    assert builder.base_axes_metadata[1].values == labels
    assert built.axes_metadata[1].values == labels
    assert builder.base_shape == built.shape


def _iron_atoms(depth=3.0):
    atoms = Atoms(
        "Fe2",
        positions=[(1.0, 1.0, 1.0), (3.0, 2.0, 2.0)],
        cell=(4.0, 3.0, depth),
        pbc=True,
    )
    atoms.set_array("magnetic_moments", np.tile([0.0, 0.0, 2.0], (len(atoms), 1)))
    return atoms


@pytest.mark.parametrize(
    "cls, labels",
    [(MagneticField, ("Bx", "By", "Bz")), (VectorPotential, ("Ax", "Ay", "Az"))],
)
@float64_devices
@lazy_params
@pytest.mark.parametrize("projection_axis", ["y", "x"])
@pytest.mark.parametrize("frozen_phonons", [False, True], ids=["static", "phonons"])
def test_depth_profile_of_a_vector_field_keeps_its_components(
    cls, labels, projection_axis, frozen_phonons, lazy, device
):
    # the component axis between depth and space was swapped with the spatial axis,
    # and the profile then had too few axes metadata. The cell holds 5 slices against
    # 3 components, and the frozen phonons add an ensemble axis in front, so a depth
    # axis moved to the wrong place gives a different shape
    atoms = _iron_atoms(depth=5.0)
    if frozen_phonons:
        atoms = FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=1)
    built = cls(atoms, gpts=(10, 12), num_slices=5, device=device).build(
        lazy=lazy
    )
    # (ensemble, z, component, x, y); the profile of a lazy field is lazy and equals
    # the eager one, summed here from the array
    array = to_host_array(built.array.compute() if lazy else built.array)

    profile = built.depth_profile(projection_axis=projection_axis)
    assert profile.is_lazy == lazy
    profile = profile.compute() if lazy else profile

    summed = array.sum(axis=-1 if projection_axis == "y" else -2)  # (.., z, c, space)
    expected = np.moveaxis(summed, -3, -1)
    assert expected.shape[-1] == 5 and expected.shape[-3] == 3
    np.testing.assert_allclose(
        to_host_array(profile.array),
        expected,
        rtol=0,
        atol=1e-10 * np.abs(expected).max(),
    )
    assert profile.ensemble_axes_metadata[-1].values == labels
    assert profile.base_shape == (summed.shape[-1], array.shape[-4])


@pytest.mark.parametrize("cls", ["GPAWMagneticField", "GPAWVectorPotential"])
def test_gpaw_fields_keep_the_labelled_component_axis_of_their_base_class(cls):
    # `_GPAWMagnetics` defined a base_axes_metadata that returned None and came first
    # in the method resolution order
    import inspect

    from abtem.magnetism import gpaw

    builder = getattr(gpaw, cls)
    base = next(b for b in builder.__mro__ if b.__name__.startswith("Base"))

    assert inspect.getattr_static(builder, "base_axes_metadata") is (
        base.__dict__["base_axes_metadata"]
    )
