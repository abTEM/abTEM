import hypothesis.strategies as st
import numpy as np
from ase.build import bulk
from ase.data import chemical_symbols
from ase.data import reference_states
from hypothesis import given

from abtem.structures import orthogonalize_cell


@given(number=st.integers(min_value=1, max_value=102))
def test_orthogonalize_atoms(number):
    if not reference_states[number - 1] in ('sc', 'fcc', 'bcc', 'diamond', 'hcp'):
        return

    symbol = chemical_symbols[number]
    atoms = bulk(symbol)

    cubic_atoms = bulk(symbol, cubic=True)
    orthogonalized_atoms = orthogonalize_cell(atoms)

    cubic_positions = cubic_atoms.positions[np.lexsort(np.rot90(cubic_atoms.positions))]
    orthogonalized_positions = orthogonalized_atoms.positions[np.lexsort(np.rot90(orthogonalized_atoms.positions))]

    cubic_cell = cubic_atoms.cell[np.lexsort(np.rot90(cubic_atoms.cell))]
    orthogonalized_cell = orthogonalized_atoms.cell[np.lexsort(np.rot90(orthogonalized_atoms.cell))]

    assert np.allclose(cubic_positions, orthogonalized_positions)
    assert np.allclose(cubic_cell, orthogonalized_cell)