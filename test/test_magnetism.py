import numpy as np
import pytest
from ase import Atoms
from scipy.integrate import trapezoid

from abtem.magnetism.iam import (
    QuasiDipoleMagneticFieldProjections,
    QuasiDipoleVectorPotentialProjections,
    MagneticField,
    VectorPotential,
    radial_prefactor_b1,
    radial_prefactor_b2,
    unit_vector_from_angles,
    magnetic_field_3d,
    vector_potential_3d,
)
from abtem.magnetism.parametrizations import LyonParametrization


def integrate_magnetic_field(quasi_dipole_projector, symbol, a, b, magnetic_moment):
    r = np.linspace(0, quasi_dipole_projector.cutoff(symbol), 100)
    parameters = np.array(quasi_dipole_projector.parametrization.parameters[symbol])
    b1 = radial_prefactor_b1(r, parameters)
    b2 = radial_prefactor_b2(r, parameters)

    x = y = quasi_dipole_projector._xy_coordinates(symbol)
    z = np.arange(
        a, b + quasi_dipole_projector._step_size / 2, quasi_dipole_projector._step_size
    )

    r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
    mr = (
        x[:, None, None] * magnetic_moment[0]
        + y[None, :, None] * magnetic_moment[1]
        + z[None, None] * magnetic_moment[2]
    )

    integrals = trapezoid(b1(r) * mr, x=z, axis=-1)
    integrals2 = trapezoid(b2(r), x=z, axis=-1)

    B = np.zeros((3,) + (len(x),) * 2)

    B[0] = integrals * x[:, None] + magnetic_moment[0] * integrals2
    B[1] = integrals * y[None, :] + magnetic_moment[1] * integrals2

    integrals = trapezoid(b1(r) * mr * z[None, None], x=z, axis=-1)
    B[2] = integrals + magnetic_moment[2] * integrals2
    return B


def test_magnetic_field_projections():
    atoms = Atoms("Fe", positions=[(2, 2, 0.0)], cell=[4] * 3) * (1, 1, 1)

    m = np.array([[0, 0, 2.33]])

    atoms.set_array("magnetic_moments", m)

    quasi_dipole_projector = QuasiDipoleMagneticFieldProjections(
        cutoff=2,
        sampling=0.05,
        slice_thickness=0.01,
        integration_steps=0.01,
    )

    magnetic_field = MagneticField(
        atoms, sampling=0.05, integrator=quasi_dipole_projector, slice_thickness=0.5
    )

    magnetic_field_array = magnetic_field.build(lazy=False).compute()
    B = integrate_magnetic_field(quasi_dipole_projector, "Fe", 0.0, 0.5, m[0])

    assert np.allclose(B[0, :-1, :-1], magnetic_field_array.array[0, 0])


def test_projected():
    parametrization = LyonParametrization()
    parameters = np.array(parametrization.parameters["Fe"])
    theta = np.pi * 0.0
    phi = 0.0
    L = 6
    gpts = (64,) * 2
    dz = 0.05
    nz = int(np.ceil(L / dz))
    r_cut = L / 2
    origin = (L / 2, L / 2, 0)

    magnetic_moment = unit_vector_from_angles(theta, phi) * 2.33


@pytest.mark.parametrize(
    "field_objects",
    [
        (magnetic_field_3d, QuasiDipoleMagneticFieldProjections, MagneticField),
        (vector_potential_3d, QuasiDipoleVectorPotentialProjections, VectorPotential),
    ],
)
def test_projected_vs_3d(field_objects):
    field_3d, QuasiDipoleFieldProjections, Field = field_objects

    parametrization = LyonParametrization()
    parameters = np.array(parametrization.parameters["Fe"])

    theta = np.pi * 0.0
    phi = 0.0
    L = 6
    gpts = (64,) * 2
    dz = 0.05
    nz = int(np.ceil(L / dz))
    r_cut = L / 2
    origin = (L / 2, L / 2, 0)
    z = 0.25

    magnetic_moment = unit_vector_from_angles(theta, phi) * 2.33

    B_grid = field_3d(
        (L, L, L), gpts + (nz,), origin, magnetic_moment, parameters, r_cut
    )
    B_grid = B_grid[..., int(z / dz), 0]

    atoms = Atoms("Fe", positions=[origin], cell=[L, L, L])
    atoms.set_array("magnetic_moments", magnetic_moment[None])

    integrator = QuasiDipoleFieldProjections(
        cutoff=4,
        sampling=0.1,
        slice_thickness=0.01,
        integration_steps=0.001,
    )

    magnetic_field = Field(
        atoms, gpts=gpts[0], integrator=integrator, slice_thickness=dz
    )

    magnetic_field_array = magnetic_field.build(lazy=False)

    B_sliced = magnetic_field_array.array[int(z / dz), 0] / dz

    assert ((B_sliced - B_grid) / B_sliced.max()).max() < 5
