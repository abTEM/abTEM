import numpy as np
from ase import Atoms
from scipy.integrate import trapezoid

from abtem.magnetism.iam import QuasiDipoleProjections, MagneticField


def integrate_magnetic_field(quasi_dipole_projector, symbol, a, b, magnetic_moment):
    b1 = quasi_dipole_projector._radial_prefactor_b1(symbol)
    b2 = quasi_dipole_projector._radial_prefactor_b2(symbol)

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


def test_magnetic_field():
    atoms = Atoms("Fe", positions=[(2, 2, 0.0)], cell=[4] * 3) * (1, 1, 1)

    m = np.array([[0, 0, 2.33]])

    atoms.set_array("magnetic_moments", m)

    quasi_dipole_projector = QuasiDipoleProjections(
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
