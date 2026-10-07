"""Frozen-phonon displacements in a potential that transforms the atoms.

A potential may rotate the atoms to another `plane`, orthogonalize their cell or cut
them into a box. Anisotropic standard deviations refer to the axes of the atoms as
given, so the displacements are rotated with the atoms; `directions` refers to the
axes of the potential. Per-atom standard deviations follow their atoms into a
transformed structure with a different number of atoms, and
`Potential.to_atoms_ensemble()` returns the configurations as simulated.
"""

import warnings

import ase
import ase.build
import numpy as np
import pytest

import abtem
from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis
from abtem.inelastic.phonons import (
    SOURCE_INDEX,
    DummyFrozenPhonons,
    EnergyResolvedAtomsEnsemble,
)


def _rotated(atoms, angle):
    atoms = atoms.copy()
    atoms.rotate(angle, "z", rotate_cell=True)
    return atoms


def _single_atom():
    return ase.Atoms("Au", positions=[(2.0, 2.0, 2.0)], cell=(4.0, 4.0, 4.0), pbc=True)


def _rectangular_rotated_about_y(num_atoms):
    """A rectangular cell rotated about y: with plane="xz" this is a rotation about
    the potential's z, which standardize_cell undoes."""
    atoms = ase.Atoms(
        "Si" * num_atoms,
        scaled_positions=[(i / num_atoms, 0.5, 0.5) for i in range(num_atoms)],
        cell=(4.0, 4.0, 5.0),
        pbc=True,
    )
    atoms.rotate(30, "y", rotate_cell=True)
    return atoms


def _with_cell_noise(atoms):
    """The cell with 1e-8 A added to a component that is zero, below the 1e-6 A
    that orthogonalize_cell zeroes before it rotates the atoms."""
    atoms = atoms.copy()
    cell = np.array(atoms.cell)
    cell[1, 0] += 1e-8
    atoms.set_cell(cell)
    return atoms


TRANSFORMS = {
    "plane_xz": (ase.build.bulk("Si", cubic=True), {"plane": "xz"}),
    # three atoms: with any other number standardize_cell raises an IndexError
    "rotated_cell_plane_xz": (_rectangular_rotated_about_y(3), {"plane": "xz"}),
    "plane_yz": (ase.build.bulk("Si", cubic=True), {"plane": "yz"}),
    "hexagonal": (ase.build.mx2("WSe2", vacuum=2), {}),
    "hexagonal_rotated_17": (_rotated(ase.build.mx2("WSe2", vacuum=2), 17), {}),
    "hexagonal_non_periodic": (ase.build.mx2("WSe2", vacuum=2), {"periodic": False}),
    "graphene_rotated_17_non_periodic": (
        _rotated(ase.build.graphene(vacuum=2), 17),
        {"periodic": False},
    ),
    # numerical noise makes the cell non-orthogonal, so it is orthogonalized
    "noisy_rotated_cell_plane_xz": (
        _with_cell_noise(_rectangular_rotated_about_y(3)),
        {"plane": "xz"},
    ),
    # a non-periodic cut of a cell rotated to another plane
    "rotated_cell_plane_xz_non_periodic": (
        _rectangular_rotated_about_y(3),
        {"plane": "xz", "periodic": False},
    ),
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_frame_is_the_map_the_transform_applies_to_positions(name):
    """Shifting every atom by v moves each transformed atom by v @ frame.

    v is chosen so that every component of its image, v @ frame, is larger than
    orthogonalize_cell's snapping tolerance at the cell boundary. Each transformed
    atom is paired with the nearest transformed copy of the same atom, so periodic
    images cannot be mistaken for each other.
    """
    atoms, kwargs = TRANSFORMS[name]

    potential = abtem.Potential(atoms, sampling=0.1, **kwargs)
    reference, _, frame = potential._transform_atoms()
    v = np.array([0.03, 0.07, 0.05]) @ np.linalg.inv(frame)
    shifted_atoms = atoms.copy()
    shifted_atoms.positions += v
    shifted_potential = abtem.Potential(shifted_atoms, sampling=0.1, **kwargs)
    shifted = shifted_potential._transform_atoms()[0]

    lengths = np.diag(reference.cell)
    paired = 0
    for position, source in zip(shifted.positions, shifted.arrays[SOURCE_INDEX]):
        moves = position - reference.positions[reference.arrays[SOURCE_INDEX] == source]
        if kwargs.get("periodic", True):
            moves -= lengths * np.round(moves / lengths)
        move = moves[np.argmin(np.linalg.norm(moves, axis=1))]
        if np.linalg.norm(move) < 0.3:
            np.testing.assert_allclose(move, v @ frame, rtol=0, atol=1e-12)
            paired += 1

    assert paired >= len(atoms)
    if "plane" in name:
        assert not np.allclose(frame, np.eye(3))
    if name.startswith("rotated_cell_plane_xz"):
        # not a permutation: standardize_cell's rotation about z is included
        assert not np.allclose(np.abs(frame), np.round(np.abs(frame)))


def test_frame_of_a_rotated_cell_is_not_a_permutation():
    """The 17 degree case exercises a frame that orthogonalize_cell computes."""
    atoms, kwargs = TRANSFORMS["hexagonal_rotated_17"]
    frame = abtem.Potential(atoms, sampling=0.1, **kwargs)._transform_atoms()[2]
    assert np.abs(frame[0, 1]) > 1e-3


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "non_periodic"])
@pytest.mark.parametrize("directions", ["xyz", "xy"])
def test_anisotropic_sigmas_follow_the_axes_of_the_input_atoms(directions, periodic):
    """With plane="xz" the beam runs along the input y axis, the potential's z.

    The displacement drawn as sigmas * r along the input axes appears in the
    potential permuted by the frame, and `directions` drops components along the
    potential's own axes. Sizes all differ, so a swapped axis cannot pass. A
    non-periodic potential displaces its atoms on a separate path, after padding.
    """
    sigmas, seed = (0.05, 0.10, 0.20), 11
    fp = abtem.FrozenPhonons(
        _single_atom(), num_configs=1, sigmas=sigmas, directions=directions, seed=seed
    )
    potential = abtem.Potential(fp, sampling=0.1, plane="xz", periodic=periodic)

    configuration = potential.to_atoms_ensemble().trajectory[0]
    undisplaced = potential.get_transformed_atoms()
    displacement = configuration.positions[0] - undisplaced.positions[0]

    r = np.random.default_rng(fp.seed[0]).normal(size=(1, 3))[0]
    in_input_axes = np.array(sigmas, dtype=np.float32) * r
    # potential axes (x, y, z) are input axes (x, z, y)
    expected = in_input_axes[[0, 2, 1]]
    if directions == "xy":
        expected[2] = 0.0
    np.testing.assert_allclose(displacement, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_rotated_potential_equals_displacing_the_input_atoms(plane):
    """Public API only: frozen phonons in a potential rotated to another plane
    give the potential of the input atoms displaced along their own axes."""
    atoms, sigmas = _single_atom(), (0.05, 0.10, 0.20)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=11)
    r = np.random.default_rng(fp.seed[0]).normal(size=(1, 3))
    displaced = atoms.copy()
    displaced.positions += np.array(sigmas, dtype=np.float32) * r

    actual = abtem.Potential(fp, sampling=0.05, plane=plane).build().compute()
    expected = abtem.Potential(displaced, sampling=0.05, plane=plane).build().compute()

    np.testing.assert_allclose(actual.array[0], expected.array, rtol=1e-6, atol=1e-6)


def _equal_per_atom_rows(atoms):
    return np.repeat(np.linspace(0.05, 0.1, len(atoms))[:, None], 3, axis=1)


def _magnetic_iron():
    atoms = ase.build.bulk("Fe", "bcc", a=2.87, cubic=True)
    atoms.set_array("magnetic_moments", np.tile([0.0, 0.0, 2.0], (len(atoms), 1)))
    return atoms


@pytest.mark.parametrize("field", ["MagneticField", "VectorPotential"])
def test_rotated_magnetic_field_equals_displacing_the_input_atoms(field):
    """Public API only: the magnetic fields share the potential's transform, so
    frozen phonons with anisotropic sigmas and plane="xz" give the field of the
    input atoms displaced along their own axes."""
    from abtem.magnetism import iam as magnetism

    field = getattr(magnetism, field)
    atoms, sigmas = _magnetic_iron(), (0.03, 0.05, 0.07)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=1)
    r = np.random.default_rng(fp.seed[0]).normal(size=(len(atoms), 3))
    displaced = atoms.copy()
    displaced.positions += np.array(sigmas, dtype=np.float32) * r

    kwargs = dict(sampling=0.2, slice_thickness=1.5, plane="xz")
    actual = field(fp, **kwargs).build(lazy=False).array[0]
    expected = field(displaced, **kwargs).build(lazy=False).array

    scale = np.abs(expected).max()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * scale)


@pytest.mark.parametrize(
    "sigmas",
    [0.1, {"W": 0.08, "Se": 0.09}, (0.1, 0.1, 0.1), _equal_per_atom_rows],
    ids=["float", "dict", "equal_tuple", "equal_per_atom_rows"],
)
@pytest.mark.parametrize("name", ["plane_xz", "hexagonal_rotated_17"])
def test_isotropic_sigmas_give_the_seeded_displacements_unchanged(name, sigmas):
    """An isotropic Gaussian is the same distribution along any rotated axes, so
    sigmas equal in all three directions are not mapped by the frame: the
    displacements are sigma * r along the potential's axes for any frame. That
    includes a 3-tuple or per-atom rows with three equal components."""
    atoms, kwargs = TRANSFORMS[name]
    if isinstance(sigmas, dict) and "W" not in atoms.get_chemical_symbols():
        sigmas = {"Si": 0.07}
    if callable(sigmas):
        sigmas = sigmas(atoms)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=5)
    potential = abtem.Potential(fp, sampling=0.1, **kwargs)
    transformed, _, frame = potential._transform_atoms()

    displaced = fp.randomize(transformed, frame=frame)

    per_atom = fp._sigmas_of(transformed)
    if per_atom.ndim == 2:
        per_atom = per_atom[:, 0]
    r = np.random.default_rng(fp.seed[0]).normal(size=(len(transformed), 3))
    expected = transformed.positions.copy()
    for axis in range(3):
        expected[:, axis] += per_atom * r[:, axis]
    np.testing.assert_array_equal(displaced.positions, expected)
    np.testing.assert_array_equal(
        displaced.positions, fp.randomize(transformed).positions
    )


def test_only_the_anisotropic_atoms_are_mapped():
    """Per-species sigmas with one isotropic species: its atoms are displaced by
    sigma * r along the potential's axes, unmapped, while the anisotropic
    species is mapped by the frame."""
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    sigmas = {"Na": (0.1, 0.1, 0.1), "Cl": (0.05, 0.10, 0.20)}
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=5)
    transformed, _, frame = abtem.Potential(
        fp, sampling=0.1, plane="xz"
    )._transform_atoms()

    displaced = fp.randomize(transformed, frame=frame)

    r = np.random.default_rng(fp.seed[0]).normal(size=(len(transformed), 3))
    per_atom = fp._sigmas_of(transformed)
    expected = per_atom * r
    chlorine = transformed.numbers == 17
    expected[chlorine] = expected[chlorine] @ frame
    np.testing.assert_array_equal(displaced.positions, transformed.positions + expected)
    assert not np.allclose(frame, np.eye(3))


def test_anisotropic_sigmas_with_an_identity_frame_are_unchanged():
    atoms = ase.build.bulk("Si", cubic=True)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=(0.05, 0.1, 0.2), seed=3)
    r = np.random.default_rng(fp.seed[0]).normal(size=(len(atoms), 3))
    expected = atoms.positions + np.array((0.05, 0.1, 0.2), dtype=np.float32) * r

    np.testing.assert_array_equal(fp.randomize(atoms).positions, expected)
    np.testing.assert_array_equal(
        fp.randomize(atoms, frame=np.eye(3)).positions, expected
    )


def test_a_frame_off_the_identity_by_rounding_is_the_identity():
    """Orthogonalizing a hexagonal cell without rotating it gives a frame that
    differs from the identity by rounding only; anisotropic displacements then
    stay sigma * r exactly, as without a frame."""
    atoms = ase.build.mx2("WSe2", vacuum=2)
    fp = abtem.FrozenPhonons(
        atoms,
        num_configs=1,
        sigmas={"W": (0.05, 0.1, 0.2), "Se": (0.1, 0.07, 0.03)},
        seed=3,
    )
    transformed, _, frame = abtem.Potential(fp, sampling=0.1)._transform_atoms()
    assert not np.array_equal(frame, np.eye(3))
    np.testing.assert_allclose(frame, np.eye(3), rtol=0, atol=1e-12)

    np.testing.assert_array_equal(
        fp.randomize(transformed, frame=frame).positions,
        fp.randomize(transformed).positions,
    )


@pytest.mark.filterwarnings(
    "ignore:The box .*, which abTEM chose because none was given:UserWarning"
)
def test_anisotropic_sigmas_follow_the_strain_of_an_orthogonalization():
    """A sheared cell that orthogonalize_cell straightens without repeating it:
    anisotropic displacements get the same linear map as the positions, so the
    frozen phonons equal the potential of the input atoms displaced along their
    own axes."""
    atoms = ase.Atoms(
        "Au",
        positions=[(2.0, 2.0, 2.5)],
        cell=[[4.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    sigmas = (0.05, 0.10, 0.20)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=11)
    frame = abtem.Potential(fp, sampling=0.05)._transform_atoms()[2]
    assert np.abs(frame[1, 0]) > 0.05

    r = np.random.default_rng(fp.seed[0]).normal(size=(1, 3))
    displaced = atoms.copy()
    displaced.positions += np.array(sigmas, dtype=np.float32) * r

    actual = abtem.Potential(fp, sampling=0.05).build().compute().array[0]
    expected = abtem.Potential(displaced, sampling=0.05).build().compute().array
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5 * expected.max())


def test_per_atom_sigmas_follow_their_atoms_into_a_transformed_cell():
    """The orthogonalized cell holds two copies of each atom of the hexagonal one;
    every copy is displaced with its own atom's standard deviation."""
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    sigmas = np.linspace(0.05, 0.10, len(atoms))
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=2)
    potential = abtem.Potential(fp, sampling=0.1)

    transformed, _, frame = potential._transform_atoms()
    assert len(transformed) != len(atoms)
    displaced = fp.randomize(transformed, frame=frame)

    r = np.random.default_rng(fp.seed[0]).normal(size=(len(transformed), 3))
    per_atom = sigmas.astype(np.float32)[transformed.arrays[SOURCE_INDEX]]
    np.testing.assert_array_equal(
        displaced.positions, transformed.positions + per_atom[:, None] * r
    )
    assert potential.build().compute().array.shape[-3] == len(potential)


def test_per_atom_sigmas_build_a_potential_with_a_transformed_cell():
    """Public API only: a hexagonal cell with per-atom sigmas, which the potential
    orthogonalizes into a cell with more atoms."""
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    sigmas = np.linspace(0.05, 0.10, len(atoms))
    fp = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=sigmas, seed=2)
    potential = abtem.Potential(fp, sampling=0.1, slice_thickness=2)
    assert potential.build().compute().array.shape[-3] == len(potential)


def test_per_atom_sigmas_for_other_atoms_raise():
    atoms = ase.build.bulk("Si", cubic=True)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=[0.1] * len(atoms), seed=1)
    with pytest.raises(RuntimeError, match="give them per species"):
        fp.randomize(atoms * (2, 1, 1))


def test_sigmas_without_a_species_of_the_atoms_raise():
    atoms = ase.build.bulk("Si", cubic=True)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas={"Si": 0.1}, seed=1)
    germanium = atoms.copy()
    germanium.numbers[:] = 32
    with pytest.raises(RuntimeError, match="provided for all atomic species"):
        fp.randomize(germanium)


def _rotation(axis, degrees):
    """The rotation as a 3x3 matrix acting on row vectors."""
    unit_vectors = ase.Atoms(positions=np.eye(3))
    unit_vectors.rotate(degrees, axis)
    return unit_vectors.positions


def _many_atoms():
    return ase.Atoms(
        "Au50",
        scaled_positions=np.random.default_rng(1).uniform(0.1, 0.9, (50, 3)),
        cell=(5.0, 6.0, 7.0),
        pbc=True,
    )


@pytest.mark.parametrize("sheared", [False, True], ids=["rotated", "rotated_sheared"])
def test_directions_along_rotated_axes_drop_the_rotated_component(sheared):
    """With directions referring to axes rotated about x, and also sheared, the
    displacement keeps exactly its components along the new x and y and none
    along the new z, whatever the input axes those are. Under a shear the
    projection that does this is not symmetric."""
    atoms = _many_atoms()
    sigmas = (0.05, 0.10, 0.20)
    fp = abtem.FrozenPhonons(
        atoms, num_configs=1, sigmas=sigmas, directions="xy", seed=3
    )
    frame = _rotation("x", 30)
    if sheared:
        frame = frame @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]])

    displaced = fp._randomize_transformed(atoms, directions_frame=frame)

    r = np.random.default_rng(fp.seed[0]).normal(size=(len(atoms), 3))
    drawn = np.array(sigmas, dtype=np.float32) * r
    displacement = (displaced.positions - atoms.positions) @ frame
    np.testing.assert_allclose(displacement[:, 2], 0.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        displacement[:, :2], (drawn @ frame)[:, :2], rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("degrees", [4, 17, 33, 61, 122])
def test_directions_along_axes_rotated_in_the_plane_are_kept_exactly(degrees):
    """A rotation about z keeps the plane of x and y: directions="xy" then keeps
    the x and y components of sigma * r exactly, as without a frame, although
    the projection onto the rotated plane is diagonal only to rounding, which
    for most angles would move some coordinates by a few 1e-16 A."""
    atoms = _many_atoms()
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=0.1, directions="xy", seed=3)
    frame = _rotation("z", degrees)

    displaced = fp._randomize_transformed(atoms, directions_frame=frame)

    np.testing.assert_array_equal(displaced.positions, fp.randomize(atoms).positions)


def test_a_frame_that_is_not_3x3_raises():
    atoms = ase.build.bulk("Si", cubic=True)
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=(0.1, 0.2, 0.3), seed=1)
    with pytest.raises(ValueError, match="3x3"):
        fp.randomize(atoms, frame=np.eye(2))


REBUILT = {
    "hexagonal": (ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1), {}),
    "orthogonal": (abtem.orthogonalize_cell(ase.build.mx2("WSe2", vacuum=2)), {}),
    "plane_xz": (ase.build.bulk("Si", cubic=True), {"plane": "xz"}),
    "hexagonal_non_periodic": (ase.build.mx2("WSe2", vacuum=2), {"periodic": False}),
}


@pytest.mark.parametrize(
    "name, projection",
    [
        (name, projection)
        for name in REBUILT
        for projection in ("infinite", "finite")
        # see test_non_periodic_finite_configuration_holds_the_box
        if not (name.endswith("non_periodic") and projection == "finite")
    ],
)
def test_potential_to_atoms_ensemble_gives_the_simulated_configurations(
    name, projection
):
    """Each configuration, built on its own with the same projection and
    `periodic`, gives the ensemble member it belongs to, exactly. A configuration
    holds the atoms of the box only: a finite projection's images of the atoms
    within its margin, added again by the rebuild, would count them twice."""
    atoms, kwargs = REBUILT[name]
    periodic = kwargs.get("periodic", True)
    sigmas = {s: 0.08 for s in set(atoms.get_chemical_symbols())}
    fp = abtem.FrozenPhonons(
        atoms, num_configs=3, sigmas=sigmas, seed=7, ensemble_mean=False
    )
    potential = abtem.Potential(
        fp, sampling=0.05, slice_thickness=2, projection=projection, **kwargs
    )
    ensemble = potential.build().compute().array

    configurations = potential.to_atoms_ensemble()
    assert len(configurations) == 3
    for i, configuration in enumerate(configurations):
        assert SOURCE_INDEX not in configuration.arrays
        assert len(configuration) == len(potential.get_transformed_atoms())
        single = abtem.Potential(
            configuration,
            sampling=0.05,
            slice_thickness=2,
            projection=projection,
            periodic=periodic,
        )
        np.testing.assert_array_equal(single.build().compute().array, ensemble[i])


@pytest.mark.parametrize(
    "atoms",
    [ase.build.mx2("WSe2", vacuum=2), ase.build.bulk("Si", cubic=True)],
    ids=["cut_hexagonal", "padded_orthogonal"],
)
def test_non_periodic_finite_configuration_holds_the_box(atoms):
    """A non-periodic potential with a finite projection also integrates the atoms
    within its cutoff outside the box, each displaced independently of the box's
    atoms. A configuration holds the box's atoms, the same ones as with an infinite
    projection, without those. The hexagonal cell is cut out of a repeated one, the
    orthogonal one padded with images."""
    fp = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.08, seed=7)

    def configurations(projection):
        potential = abtem.Potential(
            fp, sampling=0.1, periodic=False, projection=projection
        )
        return potential, potential.to_atoms_ensemble()

    finite, finite_configurations = configurations("finite")
    infinite, infinite_configurations = configurations("infinite")
    in_box = len(infinite.get_transformed_atoms())
    assert len(finite.get_sliced_atoms().atoms) > in_box
    for a, b in zip(finite_configurations, infinite_configurations):
        assert len(a) == in_box
        np.testing.assert_array_equal(a.numbers, b.numbers)


def test_potential_to_atoms_ensemble_keeps_an_energy_resolved_ensemble():
    """Two energies by three configurations stay a 2 x 3 ensemble with the same
    energies and axes, and the potential of the returned ensemble is the one of
    the original."""
    base = ase.build.mx2("WSe2", vacuum=2)
    rng = np.random.default_rng(0)
    snapshots = np.empty((2, 3), dtype=object)
    for index in np.ndindex(snapshots.shape):
        snapshot = base.copy()
        snapshot.positions += rng.normal(scale=0.05, size=snapshot.positions.shape)
        snapshots[index] = snapshot
    ensemble = EnergyResolvedAtomsEnsemble(snapshots, energies=[0.0, 0.02])
    potential = abtem.Potential(ensemble, sampling=0.1, slice_thickness=2)

    configurations = potential.to_atoms_ensemble()

    assert isinstance(configurations, EnergyResolvedAtomsEnsemble)
    assert configurations.ensemble_shape == (2, 3)
    np.testing.assert_array_equal(configurations.energies, ensemble.energies)
    assert [type(axis) for axis in configurations.ensemble_axes_metadata] == [
        type(axis) for axis in ensemble.ensemble_axes_metadata
    ]
    rebuilt = abtem.Potential(configurations, sampling=0.1, slice_thickness=2)
    np.testing.assert_array_equal(
        rebuilt.build().compute().array, potential.build().compute().array
    )


def _energy_resolved_snapshots(shape):
    base = ase.build.mx2("WSe2", vacuum=2)
    rng = np.random.default_rng(0)
    snapshots = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        snapshot = base.copy()
        snapshot.positions += rng.normal(scale=0.05, size=snapshot.positions.shape)
        snapshots[index] = snapshot
    return snapshots


@pytest.mark.parametrize("ensemble_mean", [True, False])
def test_potential_to_atoms_ensemble_keeps_the_ensemble_mean_of_energy_resolved(
    ensemble_mean,
):
    ensemble = EnergyResolvedAtomsEnsemble(
        _energy_resolved_snapshots((2, 3)),
        energies=[0.0, 0.02],
        ensemble_mean=ensemble_mean,
    )
    configurations = abtem.Potential(ensemble, sampling=0.2).to_atoms_ensemble()

    assert isinstance(configurations, EnergyResolvedAtomsEnsemble)
    assert configurations.ensemble_mean is ensemble_mean


def test_potential_to_atoms_ensemble_keeps_the_axes_of_an_energy_resolved_ensemble():
    """The returned ensemble carries the axes metadata of the original, not the
    default ones, as copies."""
    axes = [
        EnergyLossAxis(values=(0.0, 0.02), units="meV", label="Custom loss"),
        FrozenPhononsAxis(label="Snapshot", _ensemble_mean=True),
    ]
    ensemble = EnergyResolvedAtomsEnsemble(
        _energy_resolved_snapshots((2, 3)),
        energies=[0.0, 0.02],
        ensemble_axes_metadata=axes,
    )
    configurations = abtem.Potential(ensemble, sampling=0.2).to_atoms_ensemble()

    assert configurations.ensemble_axes_metadata == axes
    for returned, original in zip(configurations.ensemble_axes_metadata, axes):
        assert returned is not original


def test_potential_to_atoms_ensemble_keeps_the_axes_of_a_one_axis_ensemble():
    base = ase.build.bulk("Si", cubic=True)
    ensemble = abtem.AtomsEnsemble(
        [base, base.copy(), base.copy()],
        ensemble_axes_metadata=[FrozenPhononsAxis(label="Snapshot")],
    )
    configurations = abtem.Potential(ensemble, sampling=0.2).to_atoms_ensemble()

    assert configurations.ensemble_axes_metadata == ensemble.ensemble_axes_metadata
    assert configurations.ensemble_axes_metadata[0].label == "Snapshot"


@pytest.mark.parametrize("ensemble_mean", [True, False])
def test_potential_to_atoms_ensemble_keeps_ensemble_mean(ensemble_mean):
    atoms = ase.build.bulk("Si", cubic=True)
    fp = abtem.FrozenPhonons(
        atoms, num_configs=2, sigmas=0.05, seed=1, ensemble_mean=ensemble_mean
    )
    configurations = abtem.Potential(fp, sampling=0.2).to_atoms_ensemble()
    assert configurations.ensemble_mean is ensemble_mean


class _TwoAxes(abtem.AtomsEnsemble):
    """A 2 x 2 ensemble of configurations that is not energy resolved."""

    def __init__(self, atoms):
        super().__init__([atoms] * 4)
        self._shape = (2, 2)

    @property
    def ensemble_shape(self):
        return self._shape

    @property
    def ensemble_axes_metadata(self):
        return [FrozenPhononsAxis(), FrozenPhononsAxis()]


def test_potential_to_atoms_ensemble_rejects_other_ensembles_of_two_axes():
    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(_TwoAxes(atoms), sampling=0.2)
    assert potential.ensemble_shape == (2, 2)
    with pytest.raises(NotImplementedError, match="2 ensemble axes"):
        potential.to_atoms_ensemble()


def test_potential_without_frozen_phonons_has_one_configuration():
    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(atoms, sampling=0.1, plane="xz")
    (configuration,) = potential.to_atoms_ensemble()
    np.testing.assert_array_equal(
        configuration.positions, potential.get_transformed_atoms().positions
    )


@pytest.mark.parametrize("projection", ["infinite", "finite"])
def test_the_atoms_of_a_potential_do_not_expose_the_source_index(projection):
    """get_transformed_atoms and get_sliced_atoms (which the core-loss sites use)."""
    atoms = ase.build.mx2("WSe2", vacuum=2)
    potential = abtem.Potential(atoms, sampling=0.1, projection=projection)
    assert SOURCE_INDEX not in potential.get_transformed_atoms().arrays
    assert SOURCE_INDEX not in potential.get_sliced_atoms().atoms.arrays


class _OwnRandomize(abtem.FrozenPhonons):
    """Written against a randomize that takes the atoms only; it shifts every atom
    by a constant, which the built-in randomize never does."""

    shift = np.array([0.11, 0.07, 0.03])

    def randomize(self, atoms):
        atoms = atoms.copy()
        atoms.positions += self.shift
        return atoms


class _OwnRandomizeWithFrame(abtem.FrozenPhonons):
    def randomize(self, atoms, frame=None):
        return super().randomize(atoms, frame=frame)


@pytest.mark.parametrize("plane", ["xy", "xz"])
def test_a_subclass_with_its_own_randomize_keeps_working(plane):
    """A subclass whose randomize takes the atoms only is called with the atoms
    only; one that takes a frame gets it."""
    atoms = ase.build.bulk("Si", cubic=True)
    own = _OwnRandomize(atoms, num_configs=2, sigmas=0.05, seed=1)
    potential = abtem.Potential(own, sampling=0.2, plane=plane)
    transformed = potential.get_transformed_atoms()
    expected = np.mod(transformed.positions + own.shift, np.diag(transformed.cell))
    for configuration in potential.to_atoms_ensemble():
        np.testing.assert_allclose(configuration.positions, expected, atol=1e-10)
    assert potential.build(lazy=False).array.shape[0] == 2

    sigmas = (0.05, 0.10, 0.20)
    with_frame = _OwnRandomizeWithFrame(atoms, num_configs=1, sigmas=sigmas, seed=1)
    plain = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=1)
    np.testing.assert_array_equal(
        abtem.Potential(with_frame, sampling=0.2, plane=plane)
        .to_atoms_ensemble()
        .trajectory[0]
        .positions,
        abtem.Potential(plain, sampling=0.2, plane=plane)
        .to_atoms_ensemble()
        .trajectory[0]
        .positions,
    )


class _StoredConfiguration(DummyFrozenPhonons):
    """Returns the configuration it stores, whatever atoms it is given."""

    def randomize(self, atoms):
        return self.atoms


def _stored_configuration():
    return ase.Atoms(
        "Si2", positions=[(-0.3, 1.0, 1.0), (2.0, 5.3, 1.0)], cell=(4.0, 5.0, 4.0)
    )


def test_a_stored_configuration_returned_by_randomize_is_not_changed():
    """The potential wraps the atoms randomize returns into the cell; it must not
    do so in the object the frozen phonons keep."""
    stored = _stored_configuration()
    before = stored.positions.copy()

    abtem.Potential(_StoredConfiguration(stored), sampling=0.1).build(lazy=False)

    np.testing.assert_array_equal(stored.positions, before)


def test_potential_to_atoms_ensemble_takes_a_stored_configuration():
    """Atoms that randomize returns need not carry the potential's own arrays."""
    stored = _stored_configuration()
    potential = abtem.Potential(_StoredConfiguration(stored), sampling=0.1)

    (configuration,) = potential.to_atoms_ensemble()

    np.testing.assert_allclose(
        configuration.positions, np.mod(stored.positions, (4.0, 5.0, 4.0)), atol=1e-12
    )
    assert SOURCE_INDEX not in configuration.arrays


class _GPAWStub:
    """Runs GPAWPotential.generate_slices without GPAW and records the displaced
    atoms it hands to the slicing."""

    def __init__(self, monkeypatch):
        import abtem.potentials.gpaw as gpaw_module

        self.module = gpaw_module
        self.displaced = []

        class Calculator(gpaw_module._DummyGPAW):
            setups = None

        def generate_slices(*args, atoms, **kwargs):
            self.displaced.append(atoms.copy())
            return iter(())

        monkeypatch.setattr(gpaw_module, "GPAW", object)
        monkeypatch.setattr(
            gpaw_module, "get_core_correction_interpolators", lambda *args: []
        )
        monkeypatch.setattr(gpaw_module, "_generate_slices", generate_slices)
        self.calculator_class = Calculator

    def displace(self, atoms, frozen_phonons, **kwargs):
        calculator = self.calculator_class(
            setup_mode=None,
            setup_xc=None,
            nt_sG=None,
            gd=None,
            D_asp={},
            atoms=atoms,
            Q_aL={},
            valence_potential=None,
        )
        potential = self.module.GPAWPotential(
            calculator, sampling=0.2, frozen_phonons=frozen_phonons, **kwargs
        )
        list(potential.generate_slices())
        return self.displaced[-1]


@pytest.mark.parametrize("directions", ["xyz", "xy"])
@pytest.mark.parametrize("plane", ["xy", "xz"])
def test_gpaw_potential_drops_the_directions_of_the_potential(
    monkeypatch, plane, directions
):
    """GPAWPotential displaces the atoms along their own axes before it rotates
    them; `directions` still refers to the axes of the potential, so with
    plane="xz" and directions="xy" the input y axis, the beam, is dropped."""
    stub = _GPAWStub(monkeypatch)
    atoms = _single_atom()
    sigmas = (0.05, 0.10, 0.20)
    fp = abtem.FrozenPhonons(
        atoms, num_configs=1, sigmas=sigmas, directions=directions, seed=3
    )

    displaced = stub.displace(atoms, fp, plane=plane)

    r = np.random.default_rng(fp.seed[0]).normal(size=(1, 3))
    expected = atoms.positions + np.array(sigmas, dtype=np.float32) * r
    if directions == "xy":
        dropped = 1 if plane == "xz" else 2
        expected[:, dropped] = atoms.positions[:, dropped]
    np.testing.assert_array_equal(displaced.positions, expected)


@pytest.mark.filterwarnings(
    "ignore:The box .*, which abTEM chose because none was given:UserWarning"
)
@pytest.mark.parametrize(
    "sigmas", [0.1, (0.05, 0.10, 0.20)], ids=["isotropic", "anisotropic"]
)
def test_gpaw_directions_with_a_strained_frame_are_unchanged(monkeypatch, sigmas):
    """A sheared cell, which the potentials GPAWPotential builds orthogonalize by
    a strain in the plane: directions="xy" still drops exactly the z component,
    and the x and y components stay sigma * r, bit for bit."""
    stub = _GPAWStub(monkeypatch)
    # Twenty atoms, so that mapping the displacements to the strained axes and
    # back would move at least one coordinate by rounding.
    atoms = ase.Atoms(
        "Au20",
        scaled_positions=np.random.default_rng(0).uniform(0.1, 0.9, (20, 3)),
        cell=[[4.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    fp = abtem.FrozenPhonons(
        atoms, num_configs=1, sigmas=sigmas, directions="xy", seed=3
    )

    displaced = stub.displace(atoms, fp)

    r = np.random.default_rng(fp.seed[0]).normal(size=(len(atoms), 3))
    expected = atoms.positions.copy()
    expected[:, :2] += (np.array(sigmas, dtype=np.float32) * r)[:, :2]
    np.testing.assert_array_equal(displaced.positions, expected)


def test_gpaw_directions_frame_does_not_repeat_the_box_warning():
    """The frame of the potentials GPAWPotential builds comes from a Potential of
    its atoms, whose default box strains a sheared cell; the GPAWPotential
    reported that box when it was constructed, so building the frame must not
    report it again."""
    from abtem.potentials.gpaw import _slice_axes_frame

    atoms = ase.Atoms(
        "Au",
        positions=[(2.0, 2.0, 2.5)],
        cell=[[4.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = _slice_axes_frame(atoms, "xy", (40, 40))

    assert [str(warning.message) for warning in caught] == []
    assert frame.shape == (3, 3)


@pytest.mark.parametrize("directions", ["xyz", "zyx"])
def test_gpaw_needs_no_directions_frame_for_all_directions(monkeypatch, directions):
    """The frame of the potentials GPAWPotential builds is needed only to drop
    directions; with all three it is not computed."""
    stub = _GPAWStub(monkeypatch)

    def unused(*args, **kwargs):
        raise AssertionError("the directions frame was computed")

    monkeypatch.setattr(stub.module, "_slice_axes_frame", unused)
    atoms = _single_atom()
    fp = abtem.FrozenPhonons(
        atoms, num_configs=1, sigmas=0.1, directions=directions, seed=3
    )

    displaced = stub.displace(atoms, fp, plane="xz")

    np.testing.assert_array_equal(displaced.positions, fp.randomize(atoms).positions)


def test_gpaw_per_atom_sigmas_follow_their_atoms_into_the_repetitions(monkeypatch):
    stub = _GPAWStub(monkeypatch)
    atoms = ase.build.bulk("Si", cubic=True)
    sigmas = np.linspace(0.05, 0.10, len(atoms))
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=3)

    displaced = stub.displace(atoms, fp, repetitions=(2, 1, 1))

    repeated = atoms * (2, 1, 1)
    r = np.random.default_rng(fp.seed[0]).normal(size=(len(repeated), 3))
    per_atom = np.tile(sigmas.astype(np.float32), 2)
    np.testing.assert_array_equal(
        displaced.positions, repeated.positions + per_atom[:, None] * r
    )
