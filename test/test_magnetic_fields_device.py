"""Device and precision handling of `MagneticField` and `VectorPotential`.

`QuasiDipoleProjections.integrate_on_grid` builds each slice with a host-side
numba kernel. It used to return that NumPy array whatever `device` was asked
for, so on an accelerator `generate_slices` added a NumPy array into a device
array and the build raised (issue #541). It also pinned every array to
float32, ignoring ``abtem.config["precision"]``.
"""

import numpy as np
import pytest
from ase import Atoms
from utils import gpu

import abtem
from abtem.core.backend import asnumpy, get_array_module
from abtem.magnetism.iam import MagneticField, VectorPotential


def _atoms(height: float = 5.0) -> Atoms:
    atoms = Atoms(
        "FeO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, height],
        pbc=True,
    )
    atoms.set_array("magnetic_moments", np.array([[0.3, -0.4, 2.0], [0, 0, 0]]))
    return atoms


def _build(cls, atoms, device, lazy):
    field = cls(atoms, sampling=0.2, slice_thickness=1.0, device=device)
    return field.build(lazy=lazy).compute()


@pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("lazy", [False, True])
def test_build_on_device_matches_cpu(cls, device, precision, lazy):
    atoms = _atoms()
    with abtem.config.set({"precision": precision}):
        reference = _build(cls, atoms, "cpu", lazy=False)
        field = _build(cls, atoms, device, lazy=lazy)

    assert get_array_module(field.array) is get_array_module(device)
    assert field.array.dtype == np.dtype(precision)
    assert field.array.shape == reference.array.shape
    assert np.abs(reference.array).max() > 0

    rtol = 1e-5 if precision == "float32" else 1e-12
    np.testing.assert_allclose(
        asnumpy(field.array),
        reference.array,
        rtol=rtol,
        atol=rtol * np.abs(reference.array).max(),
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
