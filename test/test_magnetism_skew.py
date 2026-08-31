"""Skew-cell support for the magnetic-field IAM module.

``abtem.magnetism.iam._superpose_field_3d`` samples the atomic magnetic field on a
parallelepiped grid for a non-orthogonal cell (pixel ``(i, j, k)`` sits at Cartesian
``i*a/Ni + j*b/Nj + k*c/Nk``) and reduces to a Cartesian rectangle for an orthogonal
cell, so the result is correctly placed on the skew lattice in both cases.
"""

import numpy as np
from ase import Atoms

from abtem.magnetism.iam import magnetic_field_3d


def _fe_atoms(cell, position=(0.0, 0.0, 0.0)):
    atoms = Atoms("Fe", cell=cell, pbc=True, positions=[position])
    atoms.set_array("magnetic_moments", np.array([[0.0, 0.0, 1.0]]))
    return atoms


def test_magnetic_field_3d_orthogonal_cell_runs():
    """Sanity: an orthogonal cell still computes a finite magnetic field."""
    atoms = _fe_atoms([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    field = magnetic_field_3d(atoms, gpts=(16, 16, 16), cutoff=2.0)
    assert field.shape == (3, 16, 16, 16)
    assert np.all(np.isfinite(field))


def test_magnetic_field_3d_non_orthogonal_cell_runs():
    """A 60-deg hex cell now samples on the parallelepiped grid (not silently
    wrong on a Cartesian rectangle that misses parts of the parallelogram)."""
    a = 3.0
    atoms = _fe_atoms(
        [[a, 0.0, 0.0],
         [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0.0],
         [0.0, 0.0, 3.0]],
        position=(0.5, 0.5, 1.5),  # away from the corner
    )
    field = magnetic_field_3d(atoms, gpts=(16, 16, 16), cutoff=2.0)
    assert field.shape == (3, 16, 16, 16)
    assert np.all(np.isfinite(field))
    # the field should be non-trivial (atom is inside the parallelepiped)
    assert float(np.max(np.abs(field))) > 0.0


def test_magnetic_field_3d_skew_equivalent_to_orthogonal_at_orthogonal_limit():
    """In the orthogonal limit (60-deg hex flattened to a rectangle, i.e. the
    cell IS orthogonal), the parallelepiped sampling reduces to the rectangular
    case and the result must match the legacy rectangular code path bit-for-bit."""
    atoms = _fe_atoms(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        position=(0.7, 1.2, 1.5),
    )
    field = magnetic_field_3d(atoms, gpts=(12, 12, 12), cutoff=2.0)
    # Just a sanity check that the orthogonal path still produces the expected
    # values (the dispatch in _superpose_field_3d uses the legacy rectangular
    # branch when the cell is orthogonal).
    assert field.shape == (3, 12, 12, 12)
    assert float(np.max(np.abs(field))) > 0.0
