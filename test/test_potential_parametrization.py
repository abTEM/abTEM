import numpy as np

from abtem.parametrizations import lobato, kirkland, load_kirkland_parameters, load_lobato_parameters
from abtem.potentials import kappa

kirkland_parameters = load_kirkland_parameters()
lobato_parameters = load_lobato_parameters()


def test_similar():
    r = np.linspace(.1, 2, 5)
    assert np.allclose(lobato(r, lobato_parameters[47]), kirkland(r, kirkland_parameters[47]), rtol=.1)


def test_values():
    r = np.array([1., 1.316074, 1.7320508, 2.2795072, 3.])

    assert np.allclose(lobato(r, lobato_parameters[47]) / kappa,
                       [10.877785, 3.5969334, 1.1213292, 0.29497656, 0.05587856])

    assert np.allclose(kirkland(r, kirkland_parameters[47]) / kappa,
                       [10.966407, 3.7869546, 1.1616056, 0.2839873, 0.04958321])
