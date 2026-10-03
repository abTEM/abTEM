"""Frozen-phonon displacements in a potential that transforms the atoms.

A potential may rotate the atoms to another `plane`, orthogonalize their cell or cut
them into a box. Anisotropic standard deviations refer to the axes of the atoms as
given, so the displacements are rotated with the atoms; `directions` refers to the
axes of the potential. Per-atom standard deviations follow their atoms into a
transformed structure with a different number of atoms, and
`Potential.to_atoms_ensemble()` returns the configurations as simulated.
"""

import ase
import ase.build
import numpy as np
import pytest

import abtem
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
    if name == "rotated_cell_plane_xz":
        # not a permutation: standardize_cell's rotation about z is included
        assert not np.allclose(np.abs(frame), np.round(np.abs(frame)))


def test_frame_of_a_rotated_cell_is_not_a_permutation():
    """The 17 degree case exercises a frame that orthogonalize_cell computes."""
    atoms, kwargs = TRANSFORMS["hexagonal_rotated_17"]
    frame = abtem.Potential(atoms, sampling=0.1, **kwargs)._transform_atoms()[2]
    assert np.abs(frame[0, 1]) > 1e-3


@pytest.mark.parametrize("directions", ["xyz", "xy"])
def test_anisotropic_sigmas_follow_the_axes_of_the_input_atoms(directions):
    """With plane="xz" the beam runs along the input y axis, the potential's z.

    The displacement drawn as sigmas * r along the input axes appears in the
    potential permuted by the frame, and `directions` drops components along the
    potential's own axes. Sizes all differ, so a swapped axis cannot pass.
    """
    sigmas, seed = (0.05, 0.10, 0.20), 11
    fp = abtem.FrozenPhonons(
        _single_atom(), num_configs=1, sigmas=sigmas, directions=directions, seed=seed
    )
    potential = abtem.Potential(fp, sampling=0.1, plane="xz")

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


def test_non_periodic_finite_configuration_holds_the_box():
    """A non-periodic potential with a finite projection also integrates the atoms
    within its cutoff outside the box, each displaced independently of the box's
    atoms. A configuration holds the box's atoms, the same ones as with an infinite
    projection, without those."""
    atoms = ase.build.mx2("WSe2", vacuum=2)
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
    """Written against a randomize that takes the atoms only."""

    def randomize(self, atoms):
        atoms = atoms.copy()
        rng = np.random.default_rng(self.seed[0])
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
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
    for seed, configuration in zip(own.seed, potential.to_atoms_ensemble()):
        rng = np.random.default_rng(seed)
        expected = transformed.positions + rng.normal(
            scale=0.05, size=transformed.positions.shape
        )
        expected = np.mod(expected, np.diag(transformed.cell))
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
