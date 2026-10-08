import os
import sys
import warnings

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import graphene

import abtem

from abtem.core.backend import asnumpy, get_array_module
from abtem.inelastic.phonons import FrozenPhonons
from abtem.potentials.iam import Potential
from utils import devices, ignore_strain_warning

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

pytestmark = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")


@pytest.fixture
def _cpu_device():
    with abtem.config.set({"device": "cpu"}):
        yield


cpu_device = pytest.mark.usefixtures("_cpu_device")


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.mark.slow
def test_compare_ps2ae_to_abtem_no_bonding(gpaw_calculator_no_bonding):
    assert_psae_matches_abtem(gpaw_calculator_no_bonding)


def test_compare_ps2ae_to_abtem_bonding(gpaw_calculator_bonding):
    assert_psae_matches_abtem(gpaw_calculator_bonding)


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


def test_gpaw_frozen_phonon_directions_are_the_axes_of_the_potential(
    gpaw_calculator_bonding,
):
    """With plane="xz" the beam runs along the input y axis, so directions="xy"
    drops the displacement along input y, which equals a zero sigma there."""
    atoms = gpaw_calculator_bonding.atoms

    def build(sigmas, directions):
        frozen_phonons = FrozenPhonons(
            atoms, num_configs=1, sigmas=sigmas, directions=directions, seed=3
        )
        return (
            GPAWPotential(
                gpaw_calculator_bonding,
                sampling=0.1,
                frozen_phonons=frozen_phonons,
                plane="xz",
            )
            .build(lazy=False)
            .array
        )

    dropped = build((0.05, 0.10, 0.20), "xy")
    zero_sigma = build((0.05, 0.0, 0.20), "xyz")
    np.testing.assert_array_equal(dropped, zero_sigma)


def test_gpaw_potential_multiple_calculators(gpaw_calculator_bonding):
    gpaw_potential = GPAWPotential([gpaw_calculator_bonding] * 2, sampling=0.05)
    assert gpaw_potential.ensemble_shape == (2,)
    assert gpaw_potential.build().ensemble_shape == (2,)
    gpaw_potential = gpaw_potential.build().compute()
    assert gpaw_potential.ensemble_shape == (2,)
    assert np.all(gpaw_potential.array[0] == gpaw_potential.array[1])


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


@ignore_strain_warning
@devices
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


# `GPAWPotential` places its field in the default box at the default origin.


CELL = (3.2, 2.8, 3.6)


@pytest.fixture(scope="module")
def box_calculator():
    from gpaw import GPAW, PW

    atoms = Atoms(
        "CO",
        positions=[(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)],
        cell=CELL,
        pbc=True,
    )
    atoms.calc = GPAW(mode=PW(250), h=0.2, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@cpu_device
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(box=(6.4, 2.8, 3.6)),
        dict(box=(3.2, 2.8, 4.0)),
        dict(origin=(1.0, 0.0, 0.0)),
        dict(origin=(0.0, 0.0, 0.5)),
    ],
    ids=["box", "box-z", "origin", "origin-z"],
)
def test_gpaw_potential_rejects_a_box_or_origin(box_calculator, kwargs):
    with pytest.raises(NotImplementedError, match="default box"):
        GPAWPotential(box_calculator, sampling=0.1, **kwargs)


@cpu_device
def test_gpaw_potential_accepts_its_own_box(box_calculator):
    default = GPAWPotential(box_calculator, sampling=0.1)
    own = GPAWPotential(box_calculator, box=CELL, sampling=0.1)
    assert own.box == default.box == CELL


@cpu_device
def test_gpaw_potential_box_follows_the_repetitions(box_calculator):
    repeated = (6.4, 2.8, 3.6)
    potential = GPAWPotential(
        box_calculator, repetitions=(2, 1, 1), box=repeated, sampling=0.1
    )
    assert potential.box == pytest.approx(repeated)
    with pytest.raises(NotImplementedError, match="default box"):
        GPAWPotential(box_calculator, repetitions=(2, 1, 1), box=CELL, sampling=0.1)


@cpu_device
def test_gpaw_potential_origin_none_is_the_zero_origin(box_calculator):
    assert GPAWPotential(box_calculator, origin=None, sampling=0.1).box == CELL


@cpu_device
@pytest.mark.parametrize("origin", [(1.0, 0.5), ("1", "0", "0"), (float("nan"), 0, 0)])
def test_gpaw_potential_invalid_origin_raises(box_calculator, origin):
    with pytest.raises(ValueError, match="origin"):
        GPAWPotential(box_calculator, origin=origin, sampling=0.1)


@cpu_device
def test_gpaw_potential_box_of_strings_raises(box_calculator):
    with pytest.raises(ValueError, match="box"):
        GPAWPotential(box_calculator, box=("3.2", "2.8", "3.6"), sampling=0.1)


@pytest.fixture(scope="module")
def hexagonal_calculator():
    from ase.build import graphene
    from gpaw import GPAW, PW

    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    atoms.calc = GPAW(mode=PW(250), h=0.25, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


def _chosen_box_warnings(records):
    return [r for r in records if "abTEM chose" in str(r.message)]


@cpu_device
@pytest.mark.filterwarnings("ignore:The box .* is not a whole supercell:UserWarning")
@pytest.mark.parametrize(
    "repetitions, warns",
    [((1, 1, 1), False), ((2, 1, 1), False), ((3, 1, 1), True), ((4, 1, 1), True)],
)
def test_gpaw_potential_reports_a_strained_default_box(
    hexagonal_calculator, repetitions, warns
):
    # The calculator's cell is repeated, and the default box of the repeated
    # hexagonal cell is exact for (1, 1, 1) and (2, 1, 1) only.
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = GPAWPotential(
            hexagonal_calculator, repetitions=repetitions, sampling=0.2
        )
        potential.build(lazy=True).compute()
    assert len(_chosen_box_warnings(records)) == int(warns)


@pytest.fixture(scope="module")
def spinpolarised_calculator():
    from ase.build import graphene
    from gpaw import GPAW, PW

    atoms = graphene(formula="BN", a=2.5, vacuum=2.0) * (3, 1, 1)
    atoms.pbc = True
    atoms.set_initial_magnetic_moments([0.3] * len(atoms))
    atoms.calc = GPAW(
        mode=PW(200),
        kpts=(1, 2, 1),
        spinpol=True,
        txt=None,
        symmetry="off",
        convergence={"density": 1e-3},
        maxiter=60,
    )
    atoms.get_potential_energy()
    return atoms.calc


@cpu_device
@pytest.mark.filterwarnings("ignore:The box .* is not a whole supercell:UserWarning")
@pytest.mark.parametrize("include_magnetic_field", [False, True])
def test_gpaw_magnetic_fields_report_the_strained_default_box_once(
    spinpolarised_calculator, include_magnetic_field
):
    # The potential and the vector potential (and the magnetic field) are built
    # from one box_calculator in one call and share one default box, so the strain
    # of that box is reported once, at the call.
    from abtem.magnetism.gpaw import gpaw_magnetic_fields

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        gpaw_magnetic_fields(
            spinpolarised_calculator,
            sampling=0.2,
            include_magnetic_field=include_magnetic_field,
        )
    chosen = _chosen_box_warnings(records)
    assert len(chosen) == 1
    assert chosen[0].filename == __file__


# `GPAWPotential(repetitions=...)` lays out the box of the repeated crystal.


@pytest.fixture(scope="module")
def bn_calculator():
    from gpaw import GPAW, PW

    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    atoms.calc = GPAW(mode=PW(250), kpts=(2, 2, 1), txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@pytest.mark.parametrize("repetitions", [(2, 1, 1), (2, 3, 2), (1, 2, 1)])
def test_gpaw_potential_box_is_that_of_the_repeated_crystal(bn_calculator, repetitions):
    # Only the layout is tested here: the values of GPAWPotential(repetitions)
    # are a separate matter.
    atoms = bn_calculator.atoms
    with abtem.config.set({"device": "cpu"}):
        potential = GPAWPotential(bn_calculator, repetitions=repetitions, sampling=0.1)
        reference = abtem.Potential(atoms * repetitions, sampling=0.1)

    assert potential.box == pytest.approx(reference.box)
    assert potential.extent == pytest.approx(reference.extent)
