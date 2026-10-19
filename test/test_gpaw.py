import os
import sys

import numpy as np
import pytest
from ase import Atoms, units
from utils import gpu

from abtem.core.backend import asnumpy, get_array_module
from abtem.inelastic.phonons import FrozenPhonons
from abtem.potentials.iam import Potential

try:
    from gpaw import GPAW, PW
    from gpaw.utilities.ps2ae import PS2AE

    from abtem.potentials.gpaw import GPAWPotential
except ImportError:
    pass


@pytest.fixture
def gpaw_calculator_no_bonding():
    atoms = Atoms("C", positions=[(0, 0, 0)], cell=(5.0,) * 3, pbc=True)
    # h=0.2 makes the new PW backend of GPAW 26 pick a real-space grid that
    # get_electrostatic_potential() rejects as "too small"; 0.18 avoids this.
    atoms.calc = GPAW(mode=PW(500), h=0.18, txt=None, kpts=(3, 3, 3))
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture
def gpaw_calculator_bonding():
    atoms = Atoms("C", positions=[(0, 0, 0)], cell=(2.0,) * 3, pbc=True)
    atoms.calc = GPAW(mode=PW(500), h=0.18, txt=None, kpts=(3, 3, 3))
    atoms.get_potential_energy()
    return atoms.calc


# @pytest.mark.skipif('gpaw' not in sys.modules, reason="requires gpaw")
# def test_all_electron_density(gpaw_calculator_no_bonding):
#     abtem_ae_density = GPAWPotential(gpaw_calculator_no_bonding)._get_all_electron_density()
#     gpaw_ae_density = gpaw_calculator_no_bonding.get_all_electron_density(gridrefinement=4)
#     assert np.all(abtem_ae_density == gpaw_ae_density)


def assert_psae_matches_abtem(calc):
    ps2ae_potential = PS2AE(calc, grid_spacing=0.02)
    ps2ae_potential = ps2ae_potential.get_electrostatic_potential(
        rcgauss=0.01 * units.Bohr, ae=True
    )
    ps2ae_potential = (
        -ps2ae_potential.sum(-1) * calc.atoms.cell[2, 2] / ps2ae_potential.shape[-1]
    )
    ps2ae_potential -= ps2ae_potential.min()

    gpaw_potential = GPAWPotential(calc, gpts=ps2ae_potential.shape)
    gpaw_potential = gpaw_potential.build().project().compute().array
    gpaw_potential -= gpaw_potential.min()

    assert np.allclose(ps2ae_potential[1:], gpaw_potential[1:], rtol=1e-2, atol=1)


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_compare_ps2ae_to_abtem_no_bonding(gpaw_calculator_no_bonding):
    assert_psae_matches_abtem(gpaw_calculator_no_bonding)


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_compare_ps2ae_to_abtem_bonding(gpaw_calculator_bonding):
    assert_psae_matches_abtem(gpaw_calculator_bonding)


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_gpaw_potential_with_frozen_phonons(gpaw_calculator_bonding):
    frozen_phonons = FrozenPhonons(
        gpaw_calculator_bonding.atoms, num_configs=2, sigmas=0.1
    )
    gpaw_potential = GPAWPotential(
        gpaw_calculator_bonding, sampling=0.05, frozen_phonons=frozen_phonons
    )
    assert gpaw_potential.ensemble_shape == (2,)
    assert gpaw_potential.build().ensemble_shape == (2,)
    gpaw_potential = gpaw_potential.build().compute()
    assert gpaw_potential.ensemble_shape == (2,)
    assert not np.allclose(gpaw_potential.array[0], gpaw_potential.array[1])


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_gpaw_potential_multiple_calculators(gpaw_calculator_bonding):
    gpaw_potential = GPAWPotential([gpaw_calculator_bonding] * 2, sampling=0.05)
    assert gpaw_potential.ensemble_shape == (2,)
    assert gpaw_potential.build().ensemble_shape == (2,)
    gpaw_potential = gpaw_potential.build().compute()
    assert gpaw_potential.ensemble_shape == (2,)
    assert np.all(gpaw_potential.array[0] == gpaw_potential.array[1])


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_gpaw_vs_iam(gpaw_calculator_no_bonding):
    gpaw_potential = (
        GPAWPotential(gpaw_calculator_no_bonding, gpts=128).build().project().array
    )
    gpaw_potential -= gpaw_potential.min()

    iam_potential = (
        Potential(
            gpaw_calculator_no_bonding.atoms,
            gpts=gpaw_potential.shape,
            projection="finite",
        )
        .build()
        .project()
        .array
    )

    iam_potential -= iam_potential.min()
    assert np.allclose(iam_potential, gpaw_potential, rtol=1e-3, atol=5)


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_gpaw_potential_from_disk(gpaw_calculator_bonding, tmpdir):
    path = os.path.join(str(tmpdir), "test.gpw")
    gpaw_calculator_bonding.write(path)

    gpaw_potential = GPAWPotential(gpaw_calculator_bonding, gpts=(32, 32))
    gpaw_potential = gpaw_potential.build().compute()

    gpaw_potential_from_disk = GPAWPotential(path, gpts=(32, 32))
    gpaw_potential_from_disk = gpaw_potential_from_disk.build().compute()
    assert gpaw_potential_from_disk == gpaw_potential

    gpaw_potential_from_disk_with_fp = GPAWPotential([path] * 2, gpts=(32, 32))
    gpaw_potential_from_disk_with_fp = (
        gpaw_potential_from_disk_with_fp.build().compute()
    )

    assert gpaw_potential_from_disk_with_fp.ensemble_shape == (2,)
    assert np.all(
        gpaw_potential_from_disk_with_fp.array[0]
        == gpaw_potential_from_disk_with_fp.array[1]
    )


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_charge_density_potential(gpaw_calculator_bonding, tmpdir):
    gpaw_potential = GPAWPotential(gpaw_calculator_bonding, sampling=0.05)


# A single element, two elements (their potentials are summed per slice), and a
# non-orthogonal cell (the valence potential is interpolated onto each slice).
_DEVICE_CELLS = {
    "C": (["C"], [(0.6, 0.8, 1.0)], (3.2, 2.8, 3.6)),
    "CO": (["C", "O"], [(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)], (3.2, 2.8, 3.6)),
    "C-nonorthogonal": (
        ["C"],
        [(0.6, 0.8, 1.0)],
        [[3.2, 0.0, 0.0], [1.0, 2.8, 0.0], [0.0, 0.0, 3.6]],
    ),
}


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("cell_name", list(_DEVICE_CELLS))
def test_gpaw_potential_built_on_a_device_matches_the_cpu_build(
    cell_name, lazy, device
):
    symbols, positions, cell = _DEVICE_CELLS[cell_name]
    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    atoms.calc = GPAW(mode=PW(250), h=0.2, txt=None, symmetry="off")
    atoms.get_potential_energy()

    kwargs = dict(gpts=(32, 28), slice_thickness=0.9)
    expected = asnumpy(
        GPAWPotential(atoms.calc, device="cpu", **kwargs).build(lazy=False).array
    )
    built = GPAWPotential(atoms.calc, device=device, **kwargs).build(lazy=lazy)
    built = built.compute()

    assert get_array_module(built.array) is get_array_module(device)
    assert built.array.shape[0] > 1
    scale = np.abs(expected).max()
    assert scale > 0
    np.testing.assert_allclose(
        asnumpy(built.array), expected, rtol=0, atol=1e-5 * scale
    )


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_gpaw_potential_gives_its_device_to_the_core_correction_potentials(
    gpaw_calculator_bonding, monkeypatch
):
    # Without a device the inner potentials follow abtem.config instead of the
    # device of the GPAWPotential, which on a GPU gives host slices that the
    # builder cannot assign into its device buffer.
    import abtem.potentials.gpaw

    devices = []

    class SpyPotential(Potential):
        def __init__(self, *args, **kwargs):
            devices.append(kwargs.get("device"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(abtem.potentials.gpaw, "Potential", SpyPotential)

    potential = GPAWPotential(gpaw_calculator_bonding, gpts=(16, 16), device="cpu")
    potential.build(lazy=False)

    assert devices == ["cpu"]
