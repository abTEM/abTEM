"""Skew-cell gating for the magnetic-field IAM module.

The atom-field accumulator in ``abtem.magnetism.iam._superpose_field_3d`` builds the
field on a rectangular Cartesian grid sized by ``atoms.cell.array.diagonal()``. That
rectangle does not match a skewed parallelepiped (atoms at ``a + b`` fall outside
it for a non-orthogonal cell), and periodic images do not tile correctly, so the
resulting field would be silently wrong. The module raises NotImplementedError on
non-orthogonal cells until a parallelepiped-aware accumulator is implemented.
"""

import numpy as np
import pytest
from ase import Atoms

from abtem.magnetism.iam import magnetic_field_3d


def _fe_atoms(cell):
    atoms = Atoms("Fe", cell=cell, pbc=True, positions=[(0.0, 0.0, 0.0)])
    atoms.set_array("magnetic_moments", np.array([[0.0, 0.0, 1.0]]))
    return atoms


def test_magnetic_field_3d_orthogonal_cell_runs():
    """Sanity: an orthogonal cell still computes a finite magnetic field."""
    atoms = _fe_atoms([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    field = magnetic_field_3d(atoms, gpts=(16, 16, 16), cutoff=2.0)
    assert field.shape == (3, 16, 16, 16)
    assert np.all(np.isfinite(field))


def test_magnetic_field_3d_rejects_non_orthogonal_cell():
    """A 60-deg hex cell is rejected with a clear error message."""
    a = 3.0
    atoms = _fe_atoms(
        [[a, 0.0, 0.0], [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0.0],
         [0.0, 0.0, 3.0]]
    )
    with pytest.raises(NotImplementedError, match="non-orthogonal cells"):
        magnetic_field_3d(atoms, gpts=(16, 16, 16), cutoff=2.0)
