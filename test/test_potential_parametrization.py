import itertools

import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, assume, settings

import strats as abst
from abtem import Probe, PlaneWave
from abtem.core.backend import get_array_module
from abtem.core.energy import EnergyUndefinedError, energy2wavelength
from abtem.core.grid import GridUndefinedError
from test_grid import grid_data, _test_grid_consistent
from utils import ensure_is_tuple, check_array_matches_device, check_array_matches_laziness

from abtem.potentials.parametrizations import lobato, kirkland
from ase.data import chemical_symbols
from utils import array_is_close

parametrizations = (lobato, kirkland)


@settings(deadline=None, max_examples=1)
@given(atomic_number=st.integers(1, 80))
@pytest.mark.parametrize("func", ['potential',
                                  'potential_derivative',
                                  'projected_potential',
                                  'projected_scattering_factor'
                                  ])
def test_parametrizations_match(atomic_number, func):

    values = []
    r = np.linspace(0.01, 4., 10)
    for parametrization in parametrizations:
        if func == 'scattering_factor' and parametrization is lobato:
            parameters = parametrization.load_parameters(scale_parameters=False)[chemical_symbols[atomic_number]]
        else:
            parameters = parametrization.load_parameters()[chemical_symbols[atomic_number]]

        values.append(getattr(parametrization, func)(r, parameters))

    for p1, p2, in itertools.combinations(values, 2):
        assert array_is_close(p1, p2, rel_tol=0.05, check_above_rel=.02)