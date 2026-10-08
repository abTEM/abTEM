import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms

from abtem.core.backend import asnumpy, get_array_module
from abtem.magnetism.gpaw import (
    GPAWMagneticField,
    GPAWMagneticFields,
    GPAWVectorPotential,
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
            xp.ones((num_slices, *gpts), dtype=np.float32),
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
    "projection, slice_thickness",
    [
        # real-space slices are whole z pixels: 16 pixels over 5 slices are
        # 3, 3, 3, 3, 4
        ("real_space", 0.8),
        ("fft", (0.4, 1.2, 0.8, 1.0, 0.6)),
    ],
)
@pytest.mark.parametrize("first_slice, last_slice", [(1, 4), (2, None)])
@pytest.mark.parametrize("builder", [GPAWMagneticField, GPAWVectorPotential])
def test_gpaw_field_slice_range_matches_the_full_build(
    builder, projection, slice_thickness, first_slice, last_slice
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
    assert len(set(full.slice_thickness)) > 1
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
