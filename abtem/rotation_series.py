"""Module for building a rotation (tilt) series of a fixed finite crystallite, e.g.
for simulating 3D electron diffraction (3DED)."""

from __future__ import annotations

from typing import Sequence

import dask
import numpy as np
from ase import Atoms

from abtem.atoms import atoms_in_cell, euler_to_rotation
from abtem.core.axes import NonLinearAxis
from abtem.inelastic.phonons import AtomsEnsemble


def _rotation_matrix(angle: float, rotation_axis: float = 0.0) -> np.ndarray:
    """The rotation matrix for tilting by `angle` [deg] about an axis at azimuth
    `rotation_axis` [deg] about `z`, without any net rotation about `z` -- the
    zxz Euler sequence `(rotation_axis, angle, -rotation_axis)` used throughout
    this module."""
    return euler_to_rotation(
        np.deg2rad(rotation_axis),
        np.deg2rad(angle),
        -np.deg2rad(rotation_axis),
        axes="zxz",
    )


def _rotate_and_crop_to_cell(
    atoms: Atoms, angle: float, rotation_axis: float = 0.0
) -> Atoms:
    """Rotate `atoms`'s positions -- not its cell, which stays the fixed
    simulation box -- about the center of its cell, then crop back to it."""
    center = np.asarray(atoms.cell).sum(0) / 2
    R = _rotation_matrix(angle, rotation_axis)

    atoms = atoms.copy()
    atoms.positions[:] = (atoms.positions - center) @ R.T + center
    return atoms_in_cell(atoms)


def rotated_atoms_ensemble(
    atoms: Atoms,
    angles: Sequence[float],
    rotation_axis: float = 0.0,
) -> AtomsEnsemble:
    """
    Build a rotation series of a finite crystallite as an `AtomsEnsemble`, for
    multislice: at each angle, the crystal is rotated about the center of its
    cell and cropped back to it, leaving the cell (the simulation box) fixed.
    Pair with :func:`rotation_series_orientation_matrices` for indexing the
    resulting diffraction patterns onto the same `hkl` grid a Bloch-wave
    calculation on the same series would produce.

    Parameters
    ----------
    atoms : ase.Atoms
        The (typically finite, non-periodic) crystallite to rotate, e.g. from
        :func:`cut_disk` or :func:`cut_ball`. Its cell is used as the fixed box
        every rotated configuration is cropped back to.
    angles : sequence of float
        The rotation angles [deg].
    rotation_axis : float, optional
        Azimuthal angle about `z` (i.e. in the `x`-`y` plane), in degrees, of
        the axis to rotate about (default is 0.0). Match this to whatever the
        crystallite was built with (e.g. `cut_disk`'s `rotation_axis`) so a
        large fraction of it survives being cropped back to the box at every
        angle.

    Returns
    -------
    atoms_ensemble : AtomsEnsemble
        The rotation series, with a `NonLinearAxis` ensemble axis labelled
        `"x_rotation"` (units `"deg"`, values `angles`).
    """
    func = dask.delayed(_rotate_and_crop_to_cell)
    trajectory = [func(atoms, angle, rotation_axis=rotation_axis) for angle in angles]
    axis_metadata = NonLinearAxis(
        label="x_rotation", units="deg", values=tuple(float(a) for a in angles)
    )
    return AtomsEnsemble(
        trajectory,
        ensemble_mean=False,
        ensemble_axes_metadata=axis_metadata,
        cell=atoms.cell,
    )


def rotation_series_orientation_matrices(
    angles: Sequence[float], rotation_axis: float = 0.0
) -> np.ndarray:
    """
    Orientation matrices matching :func:`rotated_atoms_ensemble`, for
    `DiffractionPatterns.index_diffraction_spots`, so a multislice rotation
    series is indexed onto the same `hkl` grid a Bloch-wave calculation on the
    same series (`BlochWaves.rotate("zxz", [(rotation_axis, angle,
    -rotation_axis) for angle in angles], degrees=True)`) would produce.

    Parameters
    ----------
    angles : sequence of float
        The rotation angles [deg], matching `rotated_atoms_ensemble`.
    rotation_axis : float, optional
        Azimuthal angle about `z`, in degrees, matching
        `rotated_atoms_ensemble` (default is 0.0).

    Returns
    -------
    orientation_matrices : numpy.ndarray
        Array of shape `(len(angles), 3, 3)`.
    """
    return np.array([_rotation_matrix(angle, rotation_axis) for angle in angles])
