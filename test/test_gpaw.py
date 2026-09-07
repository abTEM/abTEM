import os
import sys

import numpy as np
import pytest
from ase import Atoms, units

from abtem.inelastic.phonons import FrozenPhonons
from abtem.potentials.iam import Potential

try:
    from gpaw import GPAW, PW
    from gpaw.utilities.ps2ae import PS2AE

    from abtem.potentials.gpaw import GPAWPotential

    # GPAW's own gpaw/utilities/ps2ae.py (add_potential_correction) is
    # written against the old-style Density API and unconditionally does:
    #
    #     dens.Q_aL.redistribute(dens.atom_partition.as_serial())
    #     ...
    #     dens.Q_aL.redistribute(dens.atom_partition)
    #
    # `calc.density` (a @property) builds a fresh `FakeDensity` compat shim
    # (gpaw.new.backwards_compatibility.FakeDensity) for GPAW's new PW
    # backend, and that shim never defines `Q_aL` at all -- only its
    # renamed replacement `ccc_aL` (see FakeDensity.__init__: `self.ccc_aL =
    # density.calculate_compensation_charge_coefficients()`). This is a gap
    # in GPAW's own ps2ae.py/FakeDensity compatibility layer, not an abTEM
    # bug (abtem/potentials/gpaw.py already has its own Q_aL/ccc_aL
    # fallback for its own code paths; this shim is only needed because the
    # tests call GPAW's PS2AE utility directly).
    #
    # `.redistribute()` exists to gather each atom's data onto rank 0
    # before a local calculation and scatter it back afterwards, for MPI
    # runs where atoms are split across domains/ranks. In a genuinely
    # serial (single-process) run -- as in these tests -- every atom is
    # already local to the one and only rank (atom_partition.rank_a is all
    # zeros, comm.size == 1), so redistributing is provably a no-op: there
    # is nothing to gather or scatter. The shim below asserts that
    # invariant explicitly and refuses to silently no-op (raising instead)
    # if it is ever exercised under real multi-rank parallelism, where
    # skipping the actual gather would silently corrupt the result.
    from gpaw.new.backwards_compatibility import FakeDensity

    class _SerialCompensationChargeCoefficientsAsQaL:
        """Adapts a new-backend `ccc_aL` (AtomArrays) to look like the
        old-style `Q_aL` (dict-like ArrayDict) that both ps2ae.py and
        abtem/potentials/gpaw.py (`dict(Q_aL)`) expect, valid only for
        single-process (serial) GPAW calculations."""

        def __init__(self, ccc_aL):
            self._ccc_aL = ccc_aL

        def __getitem__(self, a):
            return self._ccc_aL[a]

        def __getattr__(self, name):
            # Delegate dict-like access (keys/items/get/values/...) to the
            # underlying AtomArrays so e.g. `dict(Q_aL)` (used in
            # abtem/potentials/gpaw.py) keeps working. `redistribute`
            # below is defined on the class, so it takes precedence over
            # this fallback rather than being delegated.
            return getattr(self._ccc_aL, name)

        def redistribute(self, partition):
            if partition.comm.size != 1:
                raise NotImplementedError(
                    "Q_aL/ccc_aL compatibility shim (see test_gpaw.py) only "
                    "supports serial (single-process) GPAW calculations; "
                    f"got comm.size={partition.comm.size}. Redistributing "
                    "compensation charge coefficients across real MPI "
                    "ranks is not implemented here."
                )
            # comm.size == 1 => every atom is already local to the only
            # rank, so redistributing to any partition on this comm is a
            # genuine no-op.
            return self

    if not hasattr(FakeDensity, "Q_aL"):
        FakeDensity.Q_aL = property(
            lambda self: _SerialCompensationChargeCoefficientsAsQaL(
                self.ccc_aL
            )
        )
except ImportError:
    pass


@pytest.fixture
def gpaw_calculator_no_bonding():
    atoms = Atoms("C", positions=[(0, 0, 0)], cell=(5.0,) * 3, pbc=True)
    # h=0.2 makes GPAW's "new" PW-mode backend pick a real-space FFT grid
    # (50x50x26 for this cell) that its own PWDesc.indices() rejects as
    # "too small" as soon as get_electrostatic_potential() is called
    # (gpaw/core/plane_waves.py). This is a GPAW grid-size quirk in the new
    # backend, not an abTEM bug; a slightly finer h picks a larger, clean
    # cubic grid (28^3 density / 56^3 fine grid) that avoids it with margin
    # (confirmed stable for h in [0.15, 0.19], not just barely under 0.2).
    atoms.calc = GPAW(mode=PW(500), h=0.18, txt=None, kpts=(3, 3, 3))
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture
def gpaw_calculator_bonding():
    atoms = Atoms("C", positions=[(0, 0, 0)], cell=(2.0,) * 3, pbc=True)
    # See gpaw_calculator_no_bonding above: h=0.2 triggers GPAW's new PW
    # backend "20x20x11 grid too small!" error from get_electrostatic_
    # potential(); h=0.18 gives a clean 12^3 density / 24^3 fine grid instead.
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
