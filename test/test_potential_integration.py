from abtem.potentials import PotentialIntegrator
from abtem.parametrizations import kirkland_projected, kirkland, load_kirkland_parameters, lobato, dvdr_lobato, \
    load_lobato_parameters
import numpy as np
from abtem.potentials import PotentialIntegrator, Potential, kappa
from ase import Atoms


def test_gaussian_integral():
    sigma = 3
    f = lambda r: np.exp(-r ** 2 / (2 * sigma ** 2))
    r = np.array([0, 30])
    integrator = PotentialIntegrator(f, r)
    value = integrator.integrate(-30, 30)
    assert np.isclose(value[0], sigma * np.sqrt(2 * np.pi))


def test_projected_kirkland():
    f = lambda r: kirkland(r, parameters[6])
    parameters = load_kirkland_parameters()
    r = np.geomspace(.01, 10, 100)
    integrator = PotentialIntegrator(f, r)
    assert np.allclose(integrator.integrate(-10, 10), kirkland_projected(r, parameters[6]))


def test_cutoff():
    tolerance = 1e-5
    atoms = Atoms([6], [(0, 0, 0)], cell=(1, 1, 1))
    potential = Potential(atoms, cutoff_tolerance=tolerance)
    assert np.isclose(potential.function(potential._get_cutoff(6), potential.parameters[6]), tolerance)


def test_interpolation():
    sampling = .01
    L = 20
    atoms = Atoms('C', positions=[(0, 0, 1.5)], cell=(L, L, 4))

    potential = Potential(atoms, sampling=sampling, cutoff_tolerance=1e-7, slice_thickness=4)

    interpolated = potential.calculate_slice(0)[0]
    integrator = potential._integrators[6][0]
    integrated = np.sum(list(integrator.cache._cache.values()), axis=0)

    r = np.linspace(0, L, len(interpolated))

    x = np.linspace(.1, 1.5, 10)
    relative_errors = (np.interp(x, integrator.r, integrated) / kappa -
                       np.interp(x, r, interpolated)) / np.interp(x, r, interpolated)
    # print(relative_errors)
    assert np.all(np.abs(relative_errors) < sampling)


def test_geomspace():
    sampling = .1
    rc = 10
    n = 100

    for rf in np.linspace(.01, rc + 1, 1000):
        r = np.geomspace(sampling, rc, n)
        i = max(np.searchsorted(r, rf) - 1, 0)
        dt = np.log(rc / sampling) / (n - 1)
        j = min(max(np.floor(np.log(rf / sampling) / dt), 0), len(r) - 1)
        assert i == j
