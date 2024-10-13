import numpy as np

bohr_magneton = 9.2740100657e-24 * 1e20  # A * Å^2
vacuum_permeability = 1.25663706127e-6 * 1e10  # T * Å / A


def saturation_magnetization(magnetic_moments, volume):
    return bohr_magneton * vacuum_permeability * np.sum(magnetic_moments) / volume


def set_magnetic_moments(atoms, magnetic_moments):
    magnetic_moments = np.array(magnetic_moments)
    if magnetic_moments.ndim == 1:
        magnetic_moments = np.tile(magnetic_moments, (len(atoms), 1))

    assert magnetic_moments.shape == (len(atoms), 3)
    atoms.set_array("magnetic_moments", magnetic_moments)
    return atoms
