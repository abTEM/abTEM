"""Skew-cell support for the magnetic-field IAM module.

``abtem.magnetism.iam._superpose_field_3d`` samples the atomic magnetic field on a
parallelepiped grid for a non-orthogonal cell (pixel ``(i, j, k)`` sits at Cartesian
``i*a/Ni + j*b/Nj + k*c/Nk``) and reduces to a Cartesian rectangle for an orthogonal
cell, so the result is correctly placed on the skew lattice in both cases.
"""

import numpy as np
import pytest
from ase import Atoms

from abtem.magnetism.iam import magnetic_field_3d


def _fe_atoms(cell, position=(0.0, 0.0, 0.0)):
    atoms = Atoms("Fe", cell=cell, pbc=True, positions=[position])
    atoms.set_array("magnetic_moments", np.array([[0.0, 0.0, 1.0]]))
    return atoms


_A = 3.0
_CELLS = {
    "orthogonal": np.diag([_A, _A, _A]),
    "hexagonal": np.array(
        [
            [_A, 0.0, 0.0],
            [_A * np.cos(np.deg2rad(60)), _A * np.sin(np.deg2rad(60)), 0.0],
            [0.0, 0.0, _A],
        ]
    ),
}


@pytest.mark.parametrize("cell", list(_CELLS.values()), ids=list(_CELLS))
def test_magnetic_field_3d_follows_the_atom_along_the_lattice_vectors(cell):
    """Each pixel must sit at ``i*a/N + j*b/N + k*c/N``, so moving the atom by one
    pixel step along a or b moves the field by exactly one pixel along that axis.

    On the hexagonal cell this is what tells the parallelepiped grid from a Cartesian
    rectangle: sampling the hexagonal cell on a rectangle of the same side lengths
    (the bug the skew branch fixes) misses the shift along b by 31% of the peak field.
    On the orthogonal cell it pins the legacy rectangular branch.
    """
    n = 16
    position = np.array([0.5, 0.5, 1.5])  # away from the corner
    field = magnetic_field_3d(_fe_atoms(cell, position), gpts=(n,) * 3, cutoff=2.0)
    assert field.shape == (3, n, n, n)
    assert np.all(np.isfinite(field))
    peak = np.abs(field).max()
    assert peak > 0.0

    for axis in (0, 1):
        shifted = magnetic_field_3d(
            _fe_atoms(cell, position + cell[axis] / n), gpts=(n,) * 3, cutoff=2.0
        )
        # the field is not periodic, so compare the overlap of the two grids
        before = [slice(None)] * 4
        after = [slice(None)] * 4
        before[axis + 1] = slice(0, n - 1)
        after[axis + 1] = slice(1, n)
        error = np.abs(shifted[tuple(after)] - field[tuple(before)]).max() / peak
        assert error < 1e-6, f"shift along lattice vector {axis}: {error:.2e}"
