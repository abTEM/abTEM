from itertools import combinations

import numpy as np

from ..parametrizations import lobato, lobato_parameters, convert_lobato, kirkland, kirkland_parameters, \
    convert_kirkland
from ..potentials import kappa


def test_similar():
    r = np.linspace(.1, 2, 5)

    assert np.allclose(lobato(r, *convert_lobato(lobato_parameters[47])),
                       kirkland(r, *convert_kirkland(kirkland_parameters[47])), rtol=.1)


def test_values():
    r = np.array([1., 1.316074, 1.7320508, 2.2795072, 3.])

    print(lobato(r, *convert_lobato(lobato_parameters[47])))
    assert np.allclose(
        lobato(r, *convert_lobato(lobato_parameters[47])) / kappa,
        [10.877785, 3.5969334, 1.1213292, 0.29497656, 0.05587856])

    assert np.allclose(
        kirkland(r, *convert_kirkland(kirkland_parameters[47])) / kappa,
        [10.966407, 3.7869546, 1.1616056, 0.2839873, 0.04958321])
