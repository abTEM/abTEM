"""GPAWPotential(repetitions=...) must equal the one-cell potential repeated.

The oracle is the potential of the calculator's own cell, tiled: the core
corrections of every repeated cell are present and the valence potential is
tiled rather than stretched over the repeated cell. Orthogonal cells only; the
row scaling of a non-orthogonal cell is a separate mechanism.
"""

import sys

import numpy as np
import pytest
from ase import Atoms

import abtem
from abtem.inelastic.phonons import FrozenPhonons

try:
    from gpaw import GPAW, PW

    from abtem.potentials.gpaw import GPAWPotential
except ImportError:
    pass

pytestmark = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")

GPTS = (32, 28)


@pytest.fixture(scope="module")
def calculator():
    # Orthogonal 3.2 x 2.8 x 3.6 A cell with different lengths along x, y and z
    # and two atoms off the cell's symmetry planes. GPAW's valence potential
    # has 36 planes along z.
    atoms = Atoms(
        "CO",
        positions=[(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)],
        cell=(3.2, 2.8, 3.6),
        pbc=True,
    )
    atoms.calc = GPAW(mode=PW(250), h=0.2, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture(autouse=True)
def double_precision_numpy_fft():
    with abtem.config.set({"precision": "float64", "fft": "numpy"}):
        yield


def _integral(potential):
    return potential.array.sum() * np.prod(potential.sampling)


def _repeated(calculator, reps, slice_thickness, **kwargs):
    gpts = (GPTS[0] * reps[0], GPTS[1] * reps[1])
    return GPAWPotential(
        calculator,
        gpts=gpts,
        slice_thickness=slice_thickness,
        repetitions=reps,
        **kwargs,
    )


@pytest.mark.parametrize("reps", [(2, 1, 1), (1, 1, 2), (2, 3, 2), (1, 3, 1)])
@pytest.mark.parametrize("slice_thickness", [0.9, 1.2])
def test_repetitions_equal_the_one_cell_potential_tiled(
    calculator, reps, slice_thickness
):
    unit = GPAWPotential(calculator, gpts=GPTS, slice_thickness=slice_thickness).build(
        lazy=False
    )
    repeated = _repeated(calculator, reps, slice_thickness).build(lazy=False)

    # (slice, x, y): the cell repeats reps[2] times along the slices.
    tiled = np.tile(unit.array, (reps[2], reps[0], reps[1]))

    assert repeated.array.shape == tiled.shape
    np.testing.assert_allclose(
        repeated.array, tiled, rtol=0, atol=1e-10 * np.abs(tiled).max()
    )


@pytest.mark.parametrize("reps", [(2, 1, 1), (1, 1, 2), (2, 3, 2)])
def test_integral_scales_with_the_number_of_cells(calculator, reps):
    unit = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9).build(lazy=False)
    repeated = _repeated(calculator, reps, 0.9).build(lazy=False)

    assert _integral(repeated) / _integral(unit) == pytest.approx(
        np.prod(reps), rel=1e-9
    )


# The slicing axis is the third entry; the first two are the axes of each slice.
# The slice thicknesses divide the length along the slicing axis into whole numbers
# of the valence potential's planes.
@pytest.mark.parametrize(
    "plane, gpts, slice_thickness, axes",
    [("xz", (32, 36), 0.4, (0, 2, 1)), ("yz", (28, 36), 1.6, (1, 2, 0))],
)
@pytest.mark.parametrize("reps", [(2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)])
def test_repetitions_equal_the_one_cell_potential_tiled_in_other_planes(
    calculator, plane, gpts, slice_thickness, axes, reps
):
    unit = GPAWPotential(
        calculator, gpts=gpts, slice_thickness=slice_thickness, plane=plane
    ).build(lazy=False)
    repeated = GPAWPotential(
        calculator,
        gpts=(gpts[0] * reps[axes[0]], gpts[1] * reps[axes[1]]),
        slice_thickness=slice_thickness,
        plane=plane,
        repetitions=reps,
    ).build(lazy=False)

    tiled = np.tile(unit.array, (reps[axes[2]], reps[axes[0]], reps[axes[1]]))
    assert repeated.array.shape == tiled.shape
    np.testing.assert_allclose(
        repeated.array, tiled, rtol=0, atol=1e-10 * np.abs(tiled).max()
    )


def test_repetitions_given_as_a_list(calculator):
    unit = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9).build(lazy=False)
    repeated = _repeated(calculator, [2, 1, 2], 0.9).build(lazy=False)

    tiled = np.tile(unit.array, (2, 2, 1))
    np.testing.assert_allclose(
        repeated.array, tiled, rtol=0, atol=1e-10 * np.abs(tiled).max()
    )


def test_one_repetition_is_the_default(calculator):
    default = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9)
    explicit = GPAWPotential(
        calculator, gpts=GPTS, slice_thickness=0.9, repetitions=(1, 1, 1)
    )
    np.testing.assert_array_equal(
        default.build(lazy=False).array, explicit.build(lazy=False).array
    )


# Displacement of atom j of the calculator's cell. The overriding randomize applies
# the displacement of atom j % 2 to atom j of any atoms, so every repeated cell is
# displaced alike and the one-cell frozen-phonon potential tiled is the oracle.
_DISPLACEMENTS = np.array([[0.07, -0.05, 0.04], [-0.06, 0.08, -0.03]])


class _DisplaceEveryCellAlike(FrozenPhonons):
    def randomize(self, atoms):
        atoms = atoms.copy()
        atoms.positions += _DISPLACEMENTS[np.arange(len(atoms)) % 2]
        return atoms


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("reps", [(2, 1, 1), (1, 1, 2), (2, 3, 2)])
def test_displaced_atoms_of_every_repeated_cell_keep_their_core_correction(
    calculator, lazy, reps
):
    frozen_phonons = _DisplaceEveryCellAlike(
        calculator.atoms, num_configs=1, sigmas=0.1
    )
    unit = GPAWPotential(
        calculator, gpts=GPTS, slice_thickness=0.9, frozen_phonons=frozen_phonons
    ).build(lazy=False)
    repeated = _repeated(calculator, reps, 0.9, frozen_phonons=frozen_phonons).build(
        lazy=lazy
    )
    repeated = repeated.compute() if lazy else repeated

    # The displacement is not the identity: the displaced one-cell potential is
    # not the equilibrium one.
    equilibrium = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9).build(
        lazy=False
    )
    assert (
        np.abs(unit.array - equilibrium.array).max() > 1e-2 * np.abs(unit.array).max()
    )

    tiled = np.tile(unit.array, (1, reps[2], reps[0], reps[1]))
    assert repeated.array.shape == tiled.shape
    np.testing.assert_allclose(
        repeated.array, tiled, rtol=0, atol=1e-10 * np.abs(tiled).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_frozen_phonons_with_repetitions_displace_every_cell_independently(
    calculator, lazy
):
    reps = (2, 3, 2)
    frozen_phonons = FrozenPhonons(calculator.atoms, num_configs=2, sigmas=0.05)
    potential = _repeated(calculator, reps, 0.9, frozen_phonons=frozen_phonons)
    assert potential.ensemble_shape == (2,)

    built = potential.build(lazy=lazy)
    built = built.compute() if lazy else built
    assert built.array.shape == (2, 8, GPTS[0] * reps[0], GPTS[1] * reps[1])

    unit = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9).build(lazy=False)
    tiled = np.tile(unit.array, (reps[2], reps[0], reps[1]))
    scale = np.abs(tiled).max()
    assert np.abs(built.array[0] - tiled).max() > 1e-3 * scale
    assert np.abs(built.array[0] - built.array[1]).max() > 1e-3 * scale


def test_frozen_phonons_with_repetitions_lazy_and_eager_agree(calculator):
    frozen_phonons = FrozenPhonons(
        calculator.atoms, num_configs=2, sigmas=0.05, seed=(3, 4)
    )
    potential = _repeated(calculator, (2, 1, 2), 0.9, frozen_phonons=frozen_phonons)

    eager = potential.build(lazy=False).array
    lazy = potential.build().compute().array
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-12 * np.abs(eager).max())


def test_zero_displacement_frozen_phonons_with_repetitions_are_the_tiled_cell(
    calculator,
):
    reps = (2, 3, 2)
    unit = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9).build(lazy=False)
    frozen_phonons = FrozenPhonons(calculator.atoms, num_configs=1, sigmas=0.0)
    built = _repeated(calculator, reps, 0.9, frozen_phonons=frozen_phonons).build(
        lazy=False
    )

    tiled = np.tile(unit.array, (reps[2], reps[0], reps[1]))
    np.testing.assert_allclose(
        built.array[0], tiled, rtol=0, atol=1e-10 * np.abs(tiled).max()
    )


def test_repetitions_over_a_calculator_list(calculator):
    reps = (2, 1, 2)
    unit = GPAWPotential(calculator, gpts=GPTS, slice_thickness=0.9).build(lazy=False)
    potential = GPAWPotential(
        [calculator, calculator],
        gpts=(GPTS[0] * 2, GPTS[1]),
        slice_thickness=0.9,
        repetitions=reps,
    )
    built = potential.build().compute()

    tiled = np.tile(unit.array, (reps[2], reps[0], reps[1]))
    assert built.array.shape == (2,) + tiled.shape
    for config in range(2):
        np.testing.assert_allclose(
            built.array[config], tiled, rtol=0, atol=1e-10 * np.abs(tiled).max()
        )
