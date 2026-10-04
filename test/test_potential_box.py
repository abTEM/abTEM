"""A potential's `box`, `origin` and auto grid follow the arguments given."""

import pytest
from ase import Atoms

import abtem
from abtem.atoms import orthogonalize_cell


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64", "fft": "numpy"}):
        yield


def _two_atoms():
    # Different lengths along x, y, z, and atoms off every symmetry position, so
    # no two axes can be confused.
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )


@pytest.mark.parametrize(
    "box", [(1.5, 3.0, 5.0), (4.0, 1.0, 5.0), (4.0, 3.0, 2.0)], ids=["x", "y", "z"]
)
def test_orthogonalize_cell_box_with_no_whole_period_raises(box):
    with pytest.raises(ValueError, match="no whole repetition"):
        orthogonalize_cell(_two_atoms(), box=box)


def test_orthogonalize_cell_box_with_one_period_strains_it():
    # A box of 2.1 A holds one period of the 4 A axis, compressed by 47.5 %.
    box = (2.1, 3.0, 5.0)
    atoms = orthogonalize_cell(_two_atoms(), box=box)
    assert atoms.cell.lengths() == pytest.approx(box)
    assert len(atoms) == 2
