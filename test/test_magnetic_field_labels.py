"""The component axis of an unbuilt magnetic field or vector potential is labelled like
the array it builds."""

import numpy as np
import pytest
from ase import Atoms

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

    builder = cls(atoms, gpts=(10, 12), slice_thickness=1.5)
    built = builder.build()

    assert builder.base_axes_metadata[1].values == labels
    assert built.axes_metadata[1].values == labels
    assert builder.base_shape == built.shape
