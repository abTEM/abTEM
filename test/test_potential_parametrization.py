import sys

import hypothesis.strategies as st
import numpy as np
import pytest
from ase.data import chemical_symbols
from hypothesis import given, settings

from abtem.potentials.parametrizations import LobatoParametrization, KirklandParametrization
from utils import array_is_close

try:
    from gpaw import GPAW
    from abtem.potentials.gpaw.parametrization import GPAWParametrization
except ImportError:
    pass


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


@settings(deadline=None, max_examples=2)
@given(atomic_number=st.integers(1, 102))
@pytest.mark.parametrize("func", ['potential'])
@pytest.mark.skipif('gpaw' not in sys.modules, reason="requires gpaw")
def test_lobato_gpaw_match(atomic_number, func):
    r = np.linspace(0.01, 4., 10)
    gpaw = GPAWParametrization()
    lobato = LobatoParametrization()
    f1 = getattr(gpaw, func)(chemical_symbols[atomic_number])(r)
    f2 = getattr(lobato, func)(chemical_symbols[atomic_number])(r)
    assert array_is_close(f1, f2, rel_tol=0.05, check_above_rel=.02)
