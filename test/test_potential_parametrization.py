import hypothesis.strategies as st
import numpy as np
import pytest
from ase.data import chemical_symbols
from hypothesis import given, settings

from abtem.potentials.parametrizations import LobatoParametrization, KirklandParametrization
import itertools

from utils import array_is_close


@settings(deadline=None, max_examples=1)
@given(atomic_number=st.integers(1, 102))
@pytest.mark.parametrize("func", ['potential',
                                  'scattering_factor',
                                  'projected_potential',
                                  'projected_scattering_factor'
                                  ])
def test_lobato_kirkland_match(atomic_number, func):
    r = np.linspace(0.01, 4., 10)
    kirkland = KirklandParametrization()
    lobato = LobatoParametrization()
    f1 = getattr(kirkland, func)(chemical_symbols[atomic_number])(r)
    f2 = getattr(lobato, func)(chemical_symbols[atomic_number])(r)
    assert array_is_close(f1, f2, rel_tol=0.05, check_above_rel=.02)
