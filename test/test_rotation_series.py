import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem.atoms import cut_disk
from abtem.bloch import BlochWaves, StructureFactor
from abtem.rotation_series import (
    rotated_atoms_ensemble,
    rotation_series_orientation_matrices,
)


@pytest.fixture
def si_disk():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    disk = cut_disk(atoms, (25.0, 25.0, 50.0), rotation_axis=25.0)
    return atoms, disk


def test_rotated_atoms_ensemble_shape_and_axis_metadata(si_disk):
    _, disk = si_disk
    angles = (0.0, 1.0, 2.0)
    ensemble = rotated_atoms_ensemble(disk, angles, rotation_axis=25.0)

    assert ensemble.ensemble_shape == (len(angles),)
    axis = ensemble.ensemble_axes_metadata[0]
    assert axis.label == "x_rotation"
    assert axis.units == "deg"
    assert axis.values == angles


def test_rotated_atoms_ensemble_conserves_atom_count_across_angles(si_disk):
    # cut_disk's rotation_axis is chosen to match the intended tilt axis
    # precisely so that a rotation series stays similarly populated at every
    # angle -- this is the property the disk shape (over a plain box) exists
    # to guarantee. (Rotating and cropping back to the cell always loses most
    # atoms relative to the raw, uncropped disk -- even at angle=0.0, since
    # cut_disk's own output is not itself cropped to the box -- so the
    # comparison here is between angles, not against the raw disk.)
    _, disk = si_disk
    counts = [
        len(
            rotated_atoms_ensemble(disk, [angle], rotation_axis=25.0)
            .trajectory[0]
            .compute()
        )
        for angle in (0.0, 1.0, 2.0, -1.5)
    ]
    assert max(counts) / min(counts) < 1.1


def test_rotation_series_orientation_matrices_shape(si_disk):
    angles = (0.0, 1.0, 2.0)
    R = rotation_series_orientation_matrices(angles, rotation_axis=25.0)
    assert R.shape == (3, 3, 3)
    for Ri in R:
        np.testing.assert_allclose(Ri @ Ri.T, np.eye(3), atol=1e-10)


def test_rotation_series_orientation_matrices_identity_at_zero_angle():
    R = rotation_series_orientation_matrices([0.0], rotation_axis=37.0)
    np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)


def test_rotation_series_ms_and_bw_agree(si_disk):
    # The property rotated_atoms_ensemble and rotation_series_orientation_matrices
    # exist to guarantee: multislice and Bloch wave over the same rotation series
    # land on the same indexed hkl grid, and their intensities agree.
    atoms, disk = si_disk
    angles = [0.0, 1.0, 2.0]
    rotation_axis = 25.0
    energy = 200e3
    g_max_store = 2.0

    abtem.config.set(
        {"diagnostics.task_progress": False, "diagnostics.progress_bar": False}
    )

    potential = abtem.Potential(
        rotated_atoms_ensemble(disk, angles, rotation_axis=rotation_axis),
        sampling=0.12,
        slice_thickness=2.0,
        exit_planes=8,
    )
    diffraction = (
        abtem.PlaneWave(energy=energy)
        .multislice(potential=potential)
        .diffraction_patterns(max_angle=None)
    )

    orientation_matrices = rotation_series_orientation_matrices(
        angles, rotation_axis=rotation_axis
    )[:, None]
    ms = (
        diffraction.to_cpu()
        .index_diffraction_spots(
            cell=atoms,
            orientation_matrices=orientation_matrices,
            centering="F",
            sg_max=0.2,
            g_max=3.0,
            radius=0.02,
        )
        .crop(k_max=g_max_store)
        .compute()
    )

    structure_factor = StructureFactor(
        atoms, g_max=6.0, parametrization="lobato", thermal_sigma=0.078, centering="F"
    )
    bloch_waves = BlochWaves(
        structure_factor=structure_factor, energy=energy, sg_max=0.2, use_wave_eq=True
    )
    all_angles = np.array([[rotation_axis, angle, -rotation_axis] for angle in angles])
    bw = (
        bloch_waves.rotate("zxz", all_angles, degrees=True)
        .calculate_diffraction_patterns(thicknesses=potential.exit_thicknesses)
        .to_cpu()
        .crop(k_max=g_max_store)
        .compute()
    )

    assert np.array_equal(np.asarray(ms.miller_indices), np.asarray(bw.miller_indices))

    ms_array = np.asarray(ms.array)[:, -1]
    bw_array = np.asarray(bw.array)[:, -1]
    keep = (ms_array > 1e-9) | (bw_array > 1e-9)
    ms_norm = ms_array[keep] / ms_array[keep].sum()
    bw_norm = bw_array[keep] / bw_array[keep].sum()

    r1 = np.abs(ms_norm - bw_norm).sum() / bw_norm.sum()
    correlation = np.corrcoef(ms_array[keep], bw_array[keep])[0, 1]
    assert r1 < 0.05
    assert correlation > 0.999
