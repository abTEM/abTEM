import numpy as np
import pytest
from ase import Atoms
from utils import gpu

from abtem.bloch.dynamical import equal_slice_thicknesses
from abtem.core.backend import asnumpy, get_array_module
from abtem.magnetism.gpaw import (
    GPAWMagneticField,
    GPAWVectorPotential,
    _fourier_slice_integrals,
    get_magnetic_field_from_gpaw,
    get_vector_potential_from_gpaw,
)


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


@pytest.mark.parametrize("device", ["cpu", gpu])
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


@pytest.mark.parametrize("projection", ["real_space", "fft"])
@pytest.mark.parametrize("first_slice, last_slice", [(1, 4), (2, None)])
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_slice_range_matches_the_full_build(
    builder, projection, first_slice, last_slice
):
    # real-space slices are whole z pixels: 16 pixels over 5 slices are 3, 3, 3, 3, 4
    field = builder(
        _SpinPolarizedCalculator(),
        sampling=0.25,
        slice_thickness=0.8,
        gridrefinement=2,
        projection=projection,
    )

    full = field.build()
    part = field.build(first_slice=first_slice, last_slice=last_slice)

    assert full.array.shape[0] == 5
    scale = np.abs(full.array).max()
    assert scale > 0
    np.testing.assert_allclose(
        part.array, full.array[first_slice:last_slice], rtol=0, atol=1e-6 * scale
    )


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


@pytest.mark.parametrize("device", ["cpu", gpu])
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
    # Plane k stands for the field from (k - 1/2) dz to (k + 1/2) dz, so a slice
    # takes its inner planes whole and half of each plane on its limits; the last
    # slice ends on plane 17, which is plane 0.
    dz = 2.0 / 17
    bounds = np.cumsum((0,) + tuple(planes))
    weights = np.zeros((len(planes), 17))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        weights[i, a:b] = dz
        weights[i, a] -= dz / 2
        weights[i, b % 17] += dz / 2
    np.testing.assert_allclose(weights.sum(0), dz, rtol=1e-12)
    expected_slices = np.moveaxis(expected @ weights.T, -1, 0)

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

    # The real-space projection integrates the z planes of the density, which
    # through the whole cell is the same integral.
    np.testing.assert_allclose(
        build("fft", 0.8).sum(0),
        build("real_space", 0.8).sum(0),
        rtol=0,
        atol=1e-5 * scale,
    )


@pytest.mark.parametrize(
    "builder, raw_field",
    [
        (GPAWMagneticField, get_magnetic_field_from_gpaw),
        (GPAWVectorPotential, get_vector_potential_from_gpaw),
    ],
)
def test_gpaw_real_space_slices_are_integrals_between_the_slice_limits(
    builder, raw_field
):
    # Plane k of the density is the sample at z = k dz and stands for the field
    # from k dz - dz / 2 to k dz + dz / 2, so a slice is the integral of the field
    # between its limits, as the fft projection computes it, up to the error of
    # integrating from the planes. Summing whole planes from each lower limit (a
    # left Riemann sum) put every slice integral dz / 2 below its slice, 20% of the
    # largest slice off here.
    calculator = _SpinPolarizedCalculator()
    field = builder(
        calculator,
        gpts=(12, 14),
        gridrefinement=2,
        slice_thickness=0.8,
        projection="real_space",
        rotate_field=None,
    )
    real_space = asnumpy(field.build().array).astype(np.float64)

    depth = 4.0
    raw = raw_field(calculator, gridrefinement=2)
    assert raw.shape[-1] == 16
    limits = np.cumsum((0.0,) + tuple(field.slice_thickness))
    exact = _fourier_slice_integrals(raw, list(zip(limits[:-1], limits[1:])), depth)
    scale = np.abs(exact).max()
    assert scale > 0

    # The field is a single harmonic along z, q = 2 pi / depth, with q dz = pi / 8.
    # Integrated from its planes by the trapezoidal rule, its integral between any
    # two planes is (q dz / 2) cot(q dz / 2) = 0.987 times the exact one: scaled,
    # not shifted.
    half_step = np.pi / 16
    quadrature = half_step / np.tan(half_step)
    np.testing.assert_allclose(
        real_space, quadrature * exact, rtol=0, atol=1e-5 * scale
    )
    np.testing.assert_allclose(
        real_space, exact, rtol=0, atol=(1 - quadrature) * scale + 1e-5 * scale
    )
