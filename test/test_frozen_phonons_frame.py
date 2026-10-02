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
from abtem.inelastic.phonons import SOURCE_INDEX


def _rotated(atoms, angle):
    atoms = atoms.copy()
    atoms.rotate(angle, "z", rotate_cell=True)
    return atoms


def _single_atom():
    return ase.Atoms("Au", positions=[(2.0, 2.0, 2.0)], cell=(4.0, 4.0, 4.0), pbc=True)


TRANSFORMS = {
    "plane_xz": (ase.build.bulk("Si", cubic=True), {"plane": "xz"}),
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

    The shift is larger than orthogonalize_cell's snapping tolerance at the cell
    boundary, and each transformed atom is paired with the nearest transformed copy
    of the same atom, so periodic images cannot be mistaken for each other.
    """
    atoms, kwargs = TRANSFORMS[name]
    v = np.array([0.03, 0.07, 0.05])

    potential = abtem.Potential(atoms, sampling=0.1, **kwargs)
    reference, _, frame = potential._transform_atoms()
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
    if name.startswith("plane"):
        assert not np.allclose(frame, np.eye(3))


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


@pytest.mark.parametrize(
    "sigmas", [0.1, {"W": 0.08, "Se": 0.09}], ids=["float", "dict"]
)
@pytest.mark.parametrize("name", ["plane_xz", "hexagonal_rotated_17"])
def test_isotropic_sigmas_give_the_seeded_displacements_unchanged(name, sigmas):
    """An isotropic Gaussian needs no rotation: the displacements are sigma * r along
    the potential's axes, as before the frame was introduced, for any frame."""
    atoms, kwargs = TRANSFORMS[name]
    if isinstance(sigmas, dict) and "W" not in atoms.get_chemical_symbols():
        sigmas = {"Si": 0.07}
    fp = abtem.FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=5)
    potential = abtem.Potential(fp, sampling=0.1, **kwargs)
    transformed, _, frame = potential._transform_atoms()

    displaced = fp.randomize(transformed, frame=frame)

    per_atom = fp._sigmas_of(transformed)
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


@pytest.mark.parametrize(
    "atoms",
    [
        ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1),
        abtem.orthogonalize_cell(ase.build.mx2("WSe2", vacuum=2)),
    ],
    ids=["hexagonal", "orthogonal"],
)
def test_potential_to_atoms_ensemble_gives_the_simulated_configurations(atoms):
    """Each configuration, built on its own, gives the ensemble member it belongs
    to. Iterating the frozen phonons does so only for a cell the potential keeps."""
    fp = abtem.FrozenPhonons(
        atoms,
        num_configs=3,
        sigmas={"W": 0.08, "Se": 0.09},
        seed=7,
        ensemble_mean=False,
    )
    potential = abtem.Potential(fp, sampling=0.05, slice_thickness=2)
    ensemble = potential.build().compute().array

    configurations = potential.to_atoms_ensemble()
    assert len(configurations) == 3
    for i, configuration in enumerate(configurations):
        assert SOURCE_INDEX not in configuration.arrays
        single = abtem.Potential(configuration, sampling=0.05, slice_thickness=2)
        np.testing.assert_allclose(
            single.build().compute().array, ensemble[i], rtol=1e-6, atol=1e-6
        )


def test_potential_without_frozen_phonons_has_one_configuration():
    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(atoms, sampling=0.1, plane="xz")
    (configuration,) = potential.to_atoms_ensemble()
    np.testing.assert_array_equal(
        configuration.positions, potential.get_transformed_atoms().positions
    )


def test_get_transformed_atoms_does_not_expose_the_source_index():
    atoms = ase.build.mx2("WSe2", vacuum=2)
    transformed = abtem.Potential(atoms, sampling=0.1).get_transformed_atoms()
    assert SOURCE_INDEX not in transformed.arrays
