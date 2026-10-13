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
        # the fft projection takes uniform slices only
        ("fft", 0.8, 1),
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


@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_fft_projection_takes_uniform_slice_thicknesses_only(builder):
    kwargs = dict(sampling=0.25, gridrefinement=2, projection="fft")
    with pytest.raises(NotImplementedError, match="Non-uniform slice thicknesses"):
        builder(
            _SpinPolarizedCalculator(),
            slice_thickness=(0.4, 1.2, 0.8, 1.0, 0.6),
            **kwargs,
        )

    uniform = builder(_SpinPolarizedCalculator(), slice_thickness=(0.8,) * 5, **kwargs)
    np.testing.assert_array_equal(
        asnumpy(uniform.build().array),
        asnumpy(
            builder(_SpinPolarizedCalculator(), slice_thickness=0.8, **kwargs)
            .build()
            .array
        ),
    )


def _local_exit_planes(global_exit_planes, offset, length):
    return tuple(
        i - offset for i in global_exit_planes if offset <= i < offset + length
    )


@devices
@pytest.mark.parametrize(
    "projection, slice_thickness",
    [
        # 3, 4, 3, 3, 3 z pixels
        ("real_space", (0.75, 1.0, 0.75, 0.75, 0.75)),
        ("fft", 0.8),
    ],
)
@pytest.mark.parametrize(
    "first_slice, last_slice, chunk_size", [(0, None, 2), (2, 5, 2)]
)
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_chunked_slices_match_the_full_build(
    builder, projection, slice_thickness, first_slice, last_slice, chunk_size, device
):
    xp = get_array_module(device)
    field = builder(
        _SpinPolarizedCalculator(),
        sampling=0.25,
        slice_thickness=slice_thickness,
        gridrefinement=2,
        projection=projection,
        exit_planes=2,
        device=device,
    )
    full = field.build()
    stop = len(full) if last_slice is None else last_slice

    chunks = list(
        field.generate_chunked_slices(first_slice, last_slice, chunk_size=chunk_size)
    )

    assert len(full) == 5
    assert len(chunks) > 1
    assert len({len(chunk) for chunk in chunks}) > 1
    assert all(type(chunk) is type(full) for chunk in chunks)
    assert all(get_array_module(chunk.array) is xp for chunk in chunks)
    assert all(chunk.sampling == full.sampling for chunk in chunks)
    scale = np.abs(asnumpy(full.array)).max()
    assert scale > 0
    np.testing.assert_allclose(
        asnumpy(xp.concatenate([chunk.array for chunk in chunks])),
        asnumpy(full.array[first_slice:stop]),
        rtol=0,
        atol=1e-6 * scale,
    )
    assert (
        sum((chunk.slice_thickness for chunk in chunks), ())
        == full.slice_thickness[first_slice:stop]
    )
    offset = first_slice
    for chunk in chunks:
        assert chunk.exit_planes == _local_exit_planes(
            full.exit_planes, offset, len(chunk)
        )
        offset += len(chunk)


@devices
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_built_gpaw_field_iterates_and_chunks_its_own_slices(builder, device):
    xp = get_array_module(device)
    full = builder(
        _SpinPolarizedCalculator(),
        sampling=0.25,
        slice_thickness=(0.75, 1.0, 0.75, 0.75, 0.75),
        gridrefinement=2,
        projection="real_space",
        device=device,
    ).build()

    slices = list(full)

    assert len(set(full.slice_thickness)) == 2
    assert [type(s) for s in slices] == [type(full)] * len(full)
    np.testing.assert_array_equal(
        asnumpy(xp.concatenate([s.array for s in slices])), asnumpy(full.array)
    )
    assert [s.slice_thickness for s in slices] == [(t,) for t in full.slice_thickness]

    chunks = list(full.generate_chunked_slices(2, 5, chunk_size=2))

    assert [len(c) for c in chunks] == [1, 2]
    assert [type(c) for c in chunks] == [type(full)] * len(chunks)
    np.testing.assert_array_equal(
        asnumpy(xp.concatenate([c.array for c in chunks])), asnumpy(full.array[2:5])
    )
    assert sum((c.slice_thickness for c in chunks), ()) == full.slice_thickness[2:5]
