"""Device and precision handling of `MagneticField` and `VectorPotential`.

`QuasiDipoleProjections.integrate_on_grid` builds each slice with a host-side
numba kernel. It used to return that NumPy array whatever `device` was asked
for, so on an accelerator `generate_slices` added a NumPy array into a device
array and the build raised (issue #541). It also pinned every array to
float32, ignoring ``abtem.config["precision"]``.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene
from utils import gpu, plane_rotation_by_hand, rotated_by_hand

import abtem
from abtem.atoms import orthogonalize_cell
from abtem.core.backend import asnumpy, get_array_module
from abtem.magnetism.iam import MagneticField, VectorPotential


def _atoms(height: float = 5.0, symbols: str = "FeO") -> Atoms:
    atoms = Atoms(
        symbols,
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, height],
        pbc=True,
    )
    moments = np.array([[0.3, -0.4, 2.0], [0, 0, 0]])
    if symbols == "Fe2":
        moments[1] = (0.0, 1.0, -1.5)
    atoms.set_array("magnetic_moments", moments)
    return atoms


def _build(cls, atoms, device, lazy):
    field = cls(atoms, sampling=0.2, slice_thickness=1.0, device=device)
    return field.build(lazy=lazy).compute()


_cpu_references = {}


def _cpu_reference(cls, symbols, precision):
    """The eager CPU build, once per worker: most of a build is the integral
    table, which every new field computes again."""
    key = (cls, symbols, precision)
    if key not in _cpu_references:
        with abtem.config.set({"precision": precision}):
            _cpu_references[key] = _build(cls, _atoms(symbols=symbols), "cpu", False)
    return _cpu_references[key]


@pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_build_on_device_matches_cpu(cls, device, precision):
    """A build on `device`, eager and lazy, equals the eager CPU build. On the CPU
    that leaves the lazy build to compare."""
    # One element: generate_slices yields the integrator's array as it is and
    # build() copies it into its buffer. Two elements: generate_slices adds it into
    # a buffer.
    for symbols in ("Fe2", "FeO"):
        reference = _cpu_reference(cls, symbols, precision)
        assert reference.array.dtype == np.dtype(precision)
        assert np.abs(reference.array).max() > 0

        for lazy in (True,) if device == "cpu" else (False, True):
            case = f"{symbols}, lazy={lazy}"
            with abtem.config.set({"precision": precision}):
                field = _build(cls, _atoms(symbols=symbols), device, lazy=lazy)

            assert get_array_module(field.array) is get_array_module(device), case
            assert field.array.dtype == np.dtype(precision), case
            assert field.array.shape == reference.array.shape, case

            rtol = 1e-5 if precision == "float32" else 1e-12
            np.testing.assert_allclose(
                asnumpy(field.array),
                reference.array,
                rtol=rtol,
                atol=rtol * np.abs(reference.array).max(),
                err_msg=case,
            )


@pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_slices_without_atoms_on_device(cls, device):
    # A tall cell leaves slices with no atom within the cutoff, which takes
    # integrate_on_grid's empty-slice return. Slice 11 (11-12 Å) is more than
    # 4.25 Å from both atoms and from their periodic images.
    atoms = _atoms(height=20.0)
    field = _build(cls, atoms, device, lazy=False)
    array = asnumpy(field.array)

    assert get_array_module(field.array) is get_array_module(device)
    assert np.all(array[11] == 0)
    assert np.abs(array[0]).max() > 0


@pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
def test_integral_table_follows_precision_change(cls):
    # The integral-table cache is shared across builds; one filled under
    # float32 must not be handed back after switching to float64.
    field = cls(_atoms(), sampling=0.2, slice_thickness=1.0)
    with abtem.config.set({"precision": "float32"}):
        assert field.build(lazy=False).array.dtype == np.float32
    with abtem.config.set({"precision": "float64"}):
        assert field._integrator.get_integral_table("Fe").dtype == np.float64
        assert field.build(lazy=False).array.dtype == np.float64


# Magnetic moments rotated with the atoms


def _chiral_magnetic_atoms() -> Atoms:
    # Three iron atoms spanning a triangle off every symmetry plane of a cell of
    # three different lengths, with moments along no axis, so that a moment left
    # unrotated, or rotated about the wrong axis, changes the field.
    atoms = Atoms(
        "Fe3",
        positions=[(0.6, 0.5, 1.1), (2.9, 2.1, 1.6), (1.4, 0.9, 3.9)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )
    atoms.set_array(
        "magnetic_moments",
        np.array([[0.3, -0.4, 2.0], [0.0, 1.0, -1.5], [-1.2, 0.5, 0.7]]),
    )
    return atoms


@pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
@pytest.mark.parametrize("plane", ["yz", "zx", "xz", "yx", "zy"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_field_in_a_plane_is_that_of_the_atoms_and_moments_rotated_by_hand(
    cls, plane, device
):
    """The field built with `plane` is the field of the atoms rotated by hand to
    that plane, their positions and their magnetic moments, built in the xy plane.
    A moment left along its input axes gives the field of a different magnet."""
    atoms = _chiral_magnetic_atoms()
    kwargs = dict(sampling=0.2, slice_thickness=1.0, device=device)

    actual = cls(atoms, plane=plane, **kwargs).build(lazy=False)
    expected = cls(rotated_by_hand(atoms, plane), **kwargs).build(lazy=False)

    actual, expected = asnumpy(actual.array), asnumpy(expected.array)
    assert actual.shape == expected.shape
    scale = np.abs(expected).max()
    assert scale > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4 * scale)


def _rotation_about(axis, angle):
    unit_vectors = Atoms(positions=np.eye(3))
    unit_vectors.rotate(angle, axis)
    return unit_vectors.positions


def test_orthogonalize_cell_rotates_the_moments_of_a_rotated_cell():
    """A hexagonal cell rotated about z by 17 degrees: orthogonalize_cell repeats
    it into a supercell close to the axes, then turns and strains that onto its
    box. Each moment of the result is the moment of the atom it is a copy of,
    turned by the rotation part of that map (here about 0.9 degrees)."""
    atoms = graphene(vacuum=2.0)
    atoms.rotate(17.0, "z", rotate_cell=True)
    moments = np.array([[1.0, 0.0, 0.5], [0.0, -1.0, 0.25]])
    atoms.set_array("magnetic_moments", moments)
    atoms.set_array("source", np.arange(len(atoms)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        orthogonal, transform = orthogonalize_cell(
            atoms, return_transform_matrix=True
        )

    u, _, vt = np.linalg.svd(transform)
    rotation = u @ vt
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # a rotation about z, and not the identity, so a moment left alone fails
    np.testing.assert_allclose(rotation[2], [0.0, 0.0, 1.0], atol=1e-12)
    assert abs(rotation[0, 1]) > 1e-3
    assert len(orthogonal) > len(atoms)
    np.testing.assert_allclose(
        orthogonal.get_array("magnetic_moments"),
        moments[orthogonal.get_array("source")] @ rotation,
        rtol=0,
        atol=1e-10,
    )


def test_orthogonalize_cell_rotates_the_moments_in_another_plane():
    """An orthorhombic cell rotated about y by 30 degrees, rotated to plane "xz":
    the rotation about the beam that `standardize_cell` undoes turns the moments
    too, so they are the moments of the unrotated cell rotated to "xz"."""
    atoms = _chiral_magnetic_atoms()
    moments = atoms.get_array("magnetic_moments")
    rotated = atoms.copy()
    rotated.rotate(30.0, "y", rotate_cell=True)
    rotated.set_array("magnetic_moments", moments @ _rotation_about("y", 30.0))

    orthogonal = orthogonalize_cell(rotated, plane="xz")

    np.testing.assert_allclose(
        orthogonal.get_array("magnetic_moments"),
        moments @ plane_rotation_by_hand("xz"),
        rtol=0,
        atol=1e-10,
    )


def test_a_strain_leaves_the_moments_and_a_rotation_turns_them():
    """A sheared cell strained into its box: the strain has no rule for a moment,
    so only the rotation part of the map turns it, and its length is kept."""
    atoms = Atoms(
        "Fe",
        positions=[(2.0, 2.0, 2.5)],
        cell=[[4.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    moments = np.array([[0.6, -0.8, 0.0]])
    atoms.set_array("magnetic_moments", moments)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        strained, transform = orthogonalize_cell(
            atoms, box=(4.0, 4.0, 5.0), return_transform_matrix=True
        )

    # the map from the sheared cell onto the box: a rotation about z times a
    # strain in the xy plane
    assert not np.allclose(transform, np.eye(3))
    rotated = strained.get_array("magnetic_moments")
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(rotated[:, 2], 0.0, atol=1e-12)
    # the rotation part is the closest rotation to the map
    u, _, vt = np.linalg.svd(transform)
    np.testing.assert_allclose(rotated, moments @ (u @ vt), atol=1e-12)
