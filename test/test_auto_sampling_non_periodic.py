"""sampling="auto" targets 0.05 A on the extent the potential really has."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene

import abtem
from abtem.core.fft import next_fast_fft_size
from abtem.core.grid import round_auto_derived_gpts

FFT = (False, False, True)


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64", "fft": "numpy"}):
        yield


def _co(pbc):
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=pbc,
    )


def _bn(pbc):
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = pbc
    return atoms


def _ensemble(atoms):
    other = atoms.copy()
    other.positions[0] += 0.1
    return abtem.AtomsEnsemble([atoms, other])


CASES = [
    ("CO, pbc (F, F, T), plane xy", lambda: _co(FFT), {}),
    ("CO, pbc (F, F, T), plane xz", lambda: _co(FFT), dict(plane="xz")),
    ("CO, pbc (F, F, T), plane yz", lambda: _co(FFT), dict(plane="yz")),
    ("CO, pbc (T, F, T), plane xz", lambda: _co((True, False, True)), dict(plane="xz")),
    ("BN, pbc (F, F, T), plane xy", lambda: _bn(FFT), {}),
    ("BN, pbc (F, F, T), plane xz", lambda: _bn(FFT), dict(plane="xz")),
    ("CO, pbc (F, F, T), box", lambda: _co(FFT), dict(box=(8.0, 9.0, 10.0))),
    ("CO, pbc (F, F, T), origin", lambda: _co(FFT), dict(origin=(1.0, 0.0, 0.0))),
    (
        "AtomsEnsemble of CO, plane xz",
        lambda: _ensemble(_co(True)),
        dict(plane="xz"),
    ),
    ("AtomsEnsemble of BN", lambda: _ensemble(_bn(True)), {}),
    (
        "AtomsEnsemble of BN, plane xz",
        lambda: _ensemble(_bn(True)),
        dict(plane="xz"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_auto_grid_of_non_periodic_atoms_follows_the_extent(case):
    name, make_atoms, kwargs = case
    potential = abtem.Potential(make_atoms(), sampling="auto", **kwargs)

    # The grid the target gives for the extent of the potential, which is the
    # box, or the cell rotated to the plane and made orthogonal.
    expected = tuple(int(np.ceil(e / 0.05)) for e in potential.extent)
    if round_auto_derived_gpts():
        expected = tuple(next_fast_fft_size(n) for n in expected)
    assert potential.gpts == expected
    assert max(potential.sampling) <= 0.05 + 1e-12


def test_auto_grid_of_non_periodic_atoms_with_a_box_is_unchanged():
    # The box was already the extent of this branch.
    potential = abtem.Potential(
        _co(FFT), sampling="auto", box=(8.0, 9.0, 10.0), periodic=False
    )
    assert potential.gpts == (160, 180)


@pytest.mark.parametrize(
    "pbc, plane, gpts",
    [(FFT, "xy", (80, 60)), (True, "xy", (80, 60)), (True, "xz", (80, 100))],
)
def test_auto_grid_that_was_right_is_unchanged(pbc, plane, gpts):
    potential = abtem.Potential(_co(pbc), sampling="auto", plane=plane)
    assert potential.gpts == gpts
