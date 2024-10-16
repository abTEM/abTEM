from functools import reduce
from operator import mul

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, reproduce_failure

import strategies as abtem_st
import dask.array as da
import ase.build

from abtem import FrozenPhonons


@given(data=st.data())
@pytest.mark.parametrize('frozen_phonons', [
    abtem_st.dummy_frozen_phonons,
    abtem_st.frozen_phonons,
    abtem_st.md_frozen_phonons,
])
@pytest.mark.parametrize('lazy', [
    True, False,
])
def test_frozen_phonons_as_ensembles(data, frozen_phonons, lazy):
    frozen_phonons = data.draw(frozen_phonons(lazy=lazy))

    if len(frozen_phonons.ensemble_shape) > 0:
        chunks = data.draw(st.integers(min_value=1, max_value=reduce(mul, frozen_phonons.ensemble_shape)))
    else:
        chunks = ()

    blocks = frozen_phonons.ensemble_blocks(chunks).compute()

    #assert all([not block.is_lazy for block in blocks])

    for i, _, fp in frozen_phonons.generate_blocks(chunks):
        fp = fp.item()

        assert blocks[i] == fp

    #assert all(isinstance(array, da.core.Array) for array in frozen_phonons._partition_args(lazy=True))
    #assert all(not isinstance(array, da.core.Array) for array in frozen_phonons._partition_args(lazy=False))



def test_sigmas():
    atoms = ase.build.bulk("Au", cubic=True) * (2, 2, 2)

    frozen_phonons = FrozenPhonons(atoms, num_configs=1000, sigmas=.1, seed=None, directions="xyz")

    positions = np.stack([atoms.positions for atoms in frozen_phonons.to_atoms_ensemble().trajectory])
    positions = positions - positions.mean(axis=0)

    assert np.abs(positions.std() - .1) < .001