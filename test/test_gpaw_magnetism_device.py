import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms

from abtem.bloch.dynamical import equal_slice_thicknesses
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.utils import get_dtype
from abtem.magnetism.gpaw import (
    GPAWMagneticField,
    GPAWMagneticFields,
    GPAWVectorPotential,
    _fourier_slice_integrals,
    _supercell_of_box,
    get_magnetic_field_from_gpaw,
    get_vector_potential_from_gpaw,
)
from abtem.potentials.iam import PotentialArray
from utils import devices


class _SpinPolarizedCalculator:
    """The two methods of a spin-polarized GPAW calculator that the GPAW
    magnetic builders call (the `GPAW` protocol of abtem.magnetism.gpaw)."""

    atoms = Atoms("Fe", positions=[(1.0, 1.2, 1.4)], cell=[3.0, 3.5, 4.0], pbc=True)

    def get_number_of_grid_points(self):
        return np.array([6, 7, 8])

    def get_all_electron_density(self, spin, gridrefinement):
        shape = tuple(self.get_number_of_grid_points() * gridrefinement)
        x, y, z = (np.arange(n) / n for n in shape)
        up = (
            1.0
            + 0.3
            * np.sin(2 * np.pi * x)[:, None, None]
            * np.cos(2 * np.pi * y)[None, :, None]
            * np.cos(2 * np.pi * z)[None, None, :]
        )
        # GPAW indexes its (spin, x, y, z) density with `spin`; True adds an axis.
        return np.stack([up, 0.5 * up])[spin]


@devices
@pytest.mark.parametrize("projection", ["fft", "real_space"])
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_built_on_a_device_matches_the_cpu_build(
    builder, projection, device
):
    kwargs = dict(
        sampling=0.25, slice_thickness=1.0, gridrefinement=2, projection=projection
    )

    expected = asnumpy(
        builder(_SpinPolarizedCalculator(), device="cpu", **kwargs).build().array
    )
    built = builder(_SpinPolarizedCalculator(), device=device, **kwargs).build()

    assert get_array_module(built.array) is get_array_module(device)
    scale = np.abs(expected).max()
    assert scale > 0
    np.testing.assert_allclose(
        asnumpy(built.array), expected, rtol=0, atol=1e-6 * scale
    )


@devices
def test_show_draws_fields_built_on_a_device(device):
    kwargs = dict(sampling=0.25, slice_thickness=1.0, gridrefinement=2, device=device)
    vector_potential = GPAWVectorPotential(_SpinPolarizedCalculator(), **kwargs).build()
    xp = get_array_module(device)
    num_slices, _, *gpts = vector_potential.array.shape
    fields = GPAWMagneticFields(
        potential=PotentialArray(
            xp.ones((num_slices, *gpts), dtype=get_dtype()),
            slice_thickness=1.0,
            extent=vector_potential.extent,
        ),
        vector_potential=vector_potential,
        magnetic_field=GPAWMagneticField(_SpinPolarizedCalculator(), **kwargs).build(),
    )

    fig = fields.show()

    assert len(fig.axes) == 10
    plt.close(fig)


@pytest.mark.parametrize(
    "projection, slice_thickness, num_thicknesses",
    [
        # real-space slices are whole z pixels: 16 pixels over 5 slices are
        # 3, 3, 3, 3, 4
        ("real_space", 0.8, 2),
        # ... or any thicknesses whose boundaries are on z pixels
        ("real_space", (0.75, 1.0, 0.75, 0.75, 0.75), 2),
        ("fft", 0.8, 1),
        # the fft projection integrates between any slice limits
        ("fft", (0.4, 1.2, 0.8, 1.0, 0.6), 5),
    ],
)
@pytest.mark.parametrize("first_slice, last_slice", [(1, 4), (2, None)])
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_slice_range_matches_the_full_build(
    builder, projection, slice_thickness, num_thicknesses, first_slice, last_slice
):
    field = builder(
        _SpinPolarizedCalculator(),
        sampling=0.25,
        slice_thickness=slice_thickness,
        gridrefinement=2,
        projection=projection,
    )

    full = field.build()
    part = field.build(first_slice=first_slice, last_slice=last_slice)

    assert full.array.shape[0] == 5
    assert len(set(full.slice_thickness)) == num_thicknesses
    scale = np.abs(full.array).max()
    assert scale > 0
    np.testing.assert_allclose(
        part.array, full.array[first_slice:last_slice], rtol=0, atol=1e-6 * scale
    )
    # `build` takes the thicknesses from the builder, so read the generated slices
    generated = [
        s.slice_thickness for s in field.generate_slices(first_slice, last_slice)
    ]
    assert generated == [(t,) for t in full.slice_thickness[first_slice:last_slice]]


class _NonDividingCalculator(_SpinPolarizedCalculator):
    """A calculator whose 17 z planes over 2 Å, sliced at 0.34 Å, are 2, 3, 3, 3,
    3, 3 planes per slice. With `axis=1` the y axis is the one of 17 planes, the
    slicing axis of plane="xz"."""

    def __init__(self, axis=2):
        cell, gpts = [3.0, 3.5, 3.5], [6, 7, 7]
        cell[axis], gpts[axis] = 2.0, 17
        self.atoms = Atoms("Fe", positions=[(1.0, 1.2, 0.7)], cell=cell, pbc=True)
        self._gpts = np.array(gpts)

    def get_number_of_grid_points(self):
        return self._gpts

    def get_all_electron_density(self, spin, gridrefinement):
        shape = tuple(self.get_number_of_grid_points() * gridrefinement)
        x, y, z = np.meshgrid(
            *(2 * np.pi * np.arange(n) / n for n in shape), indexing="ij"
        )
        # Terms constant along z and along y give the field a nonzero average
        # along either slicing axis; the second harmonics vary it over slices.
        up = (
            1.0
            + 0.3 * np.sin(x) * np.cos(y)
            + 0.3 * np.sin(x) * np.cos(z)
            + 0.2 * np.cos(x) * np.sin(2 * y) * np.sin(2 * z)
        )
        return np.stack([up, 0.5 * up])[spin]


_REAL_SPACE_PLANES = {"xy": ((0, 1, 2), 2), "xz": ((0, 2, 1), 1)}


@devices
@pytest.mark.parametrize("plane", list(_REAL_SPACE_PLANES))
@pytest.mark.parametrize(
    "builder, raw_field",
    [
        (GPAWMagneticField, get_magnetic_field_from_gpaw),
        (GPAWVectorPotential, get_vector_potential_from_gpaw),
    ],
)
def test_gpaw_real_space_slices_use_every_z_plane_once(
    builder, raw_field, plane, device
):
    axes, axis = _REAL_SPACE_PLANES[plane]
    calculator = _NonDividingCalculator(axis)
    field = builder(
        calculator,
        gpts=tuple(int(calculator.get_number_of_grid_points()[a]) for a in axes[:2]),
        slice_thickness=0.34,
        gridrefinement=1,
        projection="real_space",
        plane=plane,
        rotate_field=None,
        device=device,
    )
    thicknesses, planes = equal_slice_thicknesses(17, 0.34, depth=2.0)
    assert tuple(planes) == (2, 3, 3, 3, 3, 3)
    assert field.slice_thickness == pytest.approx(thicknesses, rel=1e-12)

    # The field in the frame of `plane`: its components and axes in the order of
    # `axes`, sliced along the last axis.
    expected = np.moveaxis(raw_field(calculator, gridrefinement=1), axis + 1, -1)
    expected = expected[list(axes)]
    dz = 2.0 / 17
    bounds = np.cumsum((0,) + tuple(planes))
    expected_slices = np.stack(
        [expected[..., a:b].sum(-1) * dz for a, b in zip(bounds[:-1], bounds[1:])]
    )

    built = field.build()
    assert get_array_module(built.array) is get_array_module(device)
    array = asnumpy(built.array)
    scale = np.abs(expected_slices).max()
    assert scale > 0
    atol = 1e-5 * scale
    np.testing.assert_allclose(array, expected_slices, rtol=0, atol=atol)
    # The slices add up to the integral of the field through the whole cell.
    integral = expected.sum(-1) * dz
    assert np.abs(integral).max() > 0.1 * scale
    np.testing.assert_allclose(array.sum(0), integral, rtol=0, atol=atol)

    part = asnumpy(field.build(first_slice=2, last_slice=5).array)
    np.testing.assert_allclose(part, array[2:5], rtol=0, atol=atol)


@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_real_space_builder_rebuilds_from_its_slice_thickness(builder):
    calculator = _NonDividingCalculator()
    kwargs = dict(gpts=(6, 7), gridrefinement=1, projection="real_space")
    field = builder(calculator, slice_thickness=0.34, **kwargs)

    rebuilt = builder(calculator, slice_thickness=field.slice_thickness, **kwargs)
    copied = field.copy()

    expected = asnumpy(field.build().array)
    for other in (rebuilt, copied):
        assert other.slice_thickness == field.slice_thickness
        np.testing.assert_array_equal(asnumpy(other.build().array), expected)


@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_real_space_projection_rejects_thicknesses_off_the_z_grid(builder):
    # The z planes of _SpinPolarizedCalculator are 0.25 Å apart.
    with pytest.raises(NotImplementedError, match="whole numbers of the z grid"):
        builder(
            _SpinPolarizedCalculator(),
            sampling=0.25,
            slice_thickness=(0.4, 1.2, 0.8, 1.0, 0.6),
            gridrefinement=2,
            projection="real_space",
        )


def test_fourier_slice_integrals_are_exact_for_a_trigonometric_polynomial():
    depth, n = 4.0, 8
    k = 2 * np.pi / depth
    z = np.arange(n) * depth / n
    # 4 k is the Nyquist frequency of 8 samples, which is taken as a cosine.
    samples = (
        1.0
        + 0.5 * np.sin(k * z)
        + 0.3 * np.cos(2 * k * z + 0.4)
        + 0.2 * np.cos(4 * k * z)
    )

    def antiderivative(z):
        return (
            z
            - 0.5 / k * np.cos(k * z)
            + 0.3 / (2 * k) * np.sin(2 * k * z + 0.4)
            + 0.2 / (4 * k) * np.sin(4 * k * z)
        )

    limits = [(0.0, 0.3), (0.3, 1.7), (1.7, 4.0)]
    integrals = _fourier_slice_integrals(samples[None], limits, depth)

    assert integrals.shape == (3, 1)
    np.testing.assert_allclose(
        integrals[:, 0],
        [antiderivative(b) - antiderivative(a) for a, b in limits],
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "builder, raw_field",
    [
        (GPAWMagneticField, get_magnetic_field_from_gpaw),
        (GPAWVectorPotential, get_vector_potential_from_gpaw),
    ],
)
def test_gpaw_fft_projection_integrates_through_the_slices(builder, raw_field):
    # A slice is the field integrated through it, in field units times Å, as the
    # slices of MagneticField and VectorPotential and the projected potential that
    # adjust_coulomb_potential subtracts A_z from. It used to be a point sample of
    # the field at the slice's entrance.
    calculator = _SpinPolarizedCalculator()
    kwargs = dict(gpts=(12, 14), gridrefinement=2, rotate_field=None)

    def build(projection, slice_thickness):
        field = builder(
            calculator,
            projection=projection,
            slice_thickness=slice_thickness,
            **kwargs,
        )
        return asnumpy(field.build().array).astype(np.float64)

    # The varying part of the density goes as cos(q z), q = 2 pi / depth, so the
    # field is C cos(q z) + S sin(q z) (the z derivative of the curl gives the
    # sine), with C the field at z = 0 and S that at a quarter of the depth, the
    # 4th of 16 planes. A slice from a to b is the integral of that.
    depth = 4.0
    thicknesses = (0.4, 1.2, 0.8, 1.0, 0.6)
    limits = np.cumsum((0.0,) + thicknesses)
    a, b = limits[:-1, None, None, None], limits[1:, None, None, None]
    q = 2 * np.pi / depth
    field = raw_field(calculator, gridrefinement=2)
    assert field.shape[-1] == 16
    cosine, sine = field[..., 0], field[..., 4]
    expected = (
        cosine * (np.sin(q * b) - np.sin(q * a))
        + sine * (np.cos(q * a) - np.cos(q * b))
    ) / q
    scale = np.abs(expected).max()
    assert scale > 0
    fft = build("fft", thicknesses)
    np.testing.assert_allclose(fft, expected, rtol=0, atol=1e-5 * scale)

    # Integrals add up: each slice is the sum of the 0.2 Å slices it spans.
    fine = build("fft", 0.2)
    bounds = np.rint(limits / 0.2).astype(int)
    np.testing.assert_allclose(
        fft,
        np.stack([fine[a:b].sum(0) for a, b in zip(bounds[:-1], bounds[1:])]),
        rtol=0,
        atol=1e-5 * scale,
    )

    # The real-space projection sums the z planes of the density times their
    # spacing, which through the whole cell is the same integral.
    slicing = (0.75, 1.0, 0.75, 0.75, 0.75)
    np.testing.assert_allclose(
        build("fft", slicing).sum(0),
        build("real_space", slicing).sum(0),
        rtol=0,
        atol=1e-5 * scale,
    )


class _CountingCalculator(_SpinPolarizedCalculator):
    def __init__(self):
        self.density_calls = 0

    def get_all_electron_density(self, spin, gridrefinement):
        self.density_calls += 1
        return super().get_all_electron_density(spin, gridrefinement)


@pytest.mark.parametrize("projection", ["fft", "real_space"])
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_is_computed_once_for_every_slice_range(builder, projection):
    calculator = _CountingCalculator()
    field = builder(
        calculator, sampling=0.25, slice_thickness=0.5, projection=projection
    )
    full = asnumpy(field.build().array)

    for i in range(len(field)):
        (slic,) = field.generate_slices(i, i + 1)
        np.testing.assert_array_equal(asnumpy(slic.array), full[i : i + 1])
    chunks = list(field.generate_chunked_slices(chunk_size=1))
    np.testing.assert_array_equal(
        np.concatenate([asnumpy(chunk.array) for chunk in chunks]), full
    )

    assert len(chunks) == len(field) == 8
    assert calculator.density_calls == 1

    # The slices handed out are not the ones the builder keeps.
    (slic,) = field.generate_slices(0, 1)
    slic.array[:] = 0.0
    np.testing.assert_array_equal(asnumpy(field.build().array), full)
    # The kept slices are not part of what the builder is.
    assert field == builder(
        calculator, sampling=0.25, slice_thickness=0.5, projection=projection
    )


# A hexagonal cell and its orthogonal supercell (a, sqrt(3) a, c), which is the
# default box of the hexagonal cell, with the same spin density.
_A, _C = 2.5, 4.0
_HEXAGONAL = np.array([[_A, 0, 0], [-_A / 2, _A * np.sqrt(3) / 2, 0], [0, 0, _C]])
_ORTHOGONAL = np.diag([_A, _A * np.sqrt(3), _C])


class _LatticeCalculator:
    """A spin-polarized calculator of any cell whose density is a few Fourier
    components in the fractional coordinates of `lattice`, a cell of the same
    lattice as `cell`, so that calculators of equivalent cells describe the same
    field. Each component is resolved by the grids used below."""

    _components = [
        ((1, 0, 0), 0.3, 0.7),
        ((0, 1, 0), 0.2, 0.0),
        ((1, 1, 1), 0.25, 0.4),
        ((2, -1, 0), 0.15, 1.4),
        ((-1, 2, 1), 0.1, -0.7),
    ]

    def __init__(self, cell, gpts, lattice=None):
        cell = np.array(cell, dtype=float)
        self.atoms = Atoms("Co", positions=[(0.1, 0.2, 0.3)], cell=cell, pbc=True)
        self._gpts = np.array(gpts)
        self._lattice = cell if lattice is None else np.array(lattice, dtype=float)

    def get_number_of_grid_points(self):
        return self._gpts

    def get_all_electron_density(self, spin, gridrefinement):
        shape = tuple(self._gpts * gridrefinement)
        s = np.stack(
            np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij"), axis=-1
        )
        fractional = s @ np.array(self.atoms.cell) @ np.linalg.inv(self._lattice)
        up = np.ones(shape)
        for m, amplitude, phase in self._components:
            up += amplitude * np.cos(2 * np.pi * fractional @ np.array(m) + phase)
        return np.stack([up, 0.5 * up])[spin]


@pytest.mark.parametrize("projection", ["fft", "real_space"])
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_of_a_hexagonal_cell_is_that_of_its_orthogonal_supercell(
    builder, projection
):
    # The grid of a hexagonal calculator runs along its lattice vectors; the field
    # used to be placed on the box as if they were Cartesian.
    kwargs = dict(
        sampling=0.25,
        slice_thickness=1.0,
        gridrefinement=1,
        projection=projection,
        rotate_field=None,
    )
    hexagonal = builder(_LatticeCalculator(_HEXAGONAL, (12, 12, 8)), **kwargs).build()
    orthogonal = builder(
        _LatticeCalculator(_ORTHOGONAL, (12, 24, 8), lattice=_HEXAGONAL), **kwargs
    ).build()

    assert hexagonal.shape == orthogonal.shape == (4, 3, 10, 18)
    expected = asnumpy(orthogonal.array)
    scale = np.abs(expected).max()
    assert scale > 0
    np.testing.assert_allclose(
        asnumpy(hexagonal.array), expected, rtol=0, atol=1e-5 * scale
    )


def test_box_of_a_rotated_cell_turns_the_field_back():
    # A field placed on the box as orthogonalize_cell places the atoms turns its
    # vectors by the rotation that maps the supercell onto the box.
    angle = 0.2
    rotation = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    vectors, turn = _supercell_of_box(_ORTHOGONAL @ rotation, np.diag(_ORTHOGONAL))

    np.testing.assert_array_equal(vectors, np.eye(3, dtype=int))
    np.testing.assert_allclose(turn, rotation.T, rtol=0, atol=1e-12)
    vectors, turn = _supercell_of_box(_HEXAGONAL, np.diag(_ORTHOGONAL))
    np.testing.assert_array_equal(vectors, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    np.testing.assert_allclose(turn, np.eye(3), rtol=0, atol=1e-12)
