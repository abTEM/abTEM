from abtem.potentials import PotentialIntegrator
from abtem.parametrizations import kirkland_projected, kirkland, load_kirkland_parameters, lobato, dvdr_lobato, \
    load_lobato_parameters
import numpy as np
from abtem.potentials import PotentialInterpolator


def test_gaussian_integral():
    sigma = 3
    f = lambda r: np.exp(-r ** 2 / (2 * sigma ** 2))
    r = np.array([0])
    integrator = PotentialIntegrator(f, r, 1e-6, 30)
    value = integrator.integrate(-30, 30)
    assert np.isclose(value, sigma * np.sqrt(2 * np.pi))


def test_projected_kirkland():
    f = lambda r: kirkland(r, parameters[6])
    parameters = load_kirkland_parameters()
    r = np.geomspace(.01, 10, 100)
    integrator = PotentialIntegrator(f, r, 1e-8, cutoff=10)
    assert np.allclose(integrator.integrate(-10, 10), kirkland_projected(r, parameters[6]))


def test_cutoff():
    tolerance = 1e-5
    interpolator = PotentialInterpolator(6, 'lobato', tolerance=tolerance, sampling=.01)
    cutoff = interpolator.cutoff
    assert np.isclose(interpolator.function(cutoff), tolerance)


def test_interpolation():
    sampling = .001
    gpts = np.array((3 / sampling, 3 / sampling)).astype(np.int)

    positions = np.array([[0., 0.]])
    a = np.array([-1.5])
    b = np.array([2])

    interpolator = PotentialInterpolator(6, 'lobato', 1e-5, sampling)

    projected = interpolator.projector.integrate(-1.5, 2)
    interpolated = interpolator.interpolate(gpts, positions, (a, b))[0]

    r = np.linspace(0, 3, gpts[0], endpoint=False)

    x = np.linspace(.1, 1.5, 10)
    relative_errors = (np.interp(x, interpolator.projector.r, projected) -
                       np.interp(x, r, interpolated)) / np.interp(x, r, interpolated)

    assert np.all(np.abs(relative_errors) < sampling)
