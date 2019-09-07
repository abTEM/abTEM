import numpy as np
from scipy.special import kn

from abtem.parametrizations import kirkland_projected_finite, load_parameters, convert_kirkland, interpolation_kernel


def kirkland_project_infinite(r, a, b, c, d):
    v = (2 * a[0] * kn(0, b[0] * r) + np.sqrt(np.pi / d[0]) * c[0] * np.exp(-d[0] * r ** 2.) +
         2 * a[1] * kn(0, b[1] * r) + np.sqrt(np.pi / d[1]) * c[1] * np.exp(-d[1] * r ** 2.) +
         2 * a[2] * kn(0, b[2] * r) + np.sqrt(np.pi / d[2]) * c[2] * np.exp(-d[2] * r ** 2.))
    return v


def test_project():
    parameters = load_parameters('data/kirkland.txt')
    a, b, c, d = convert_kirkland(parameters[47])

    n = 200
    r_cut = 20
    r = np.linspace(r_cut / n, r_cut, n)
    z0 = np.array([-20.])
    z1 = np.array([20.])
    samples = 100000

    inifite = kirkland_project_infinite(r, a, b, c, d)
    finite = kirkland_projected_finite(r, r_cut, z0, z1, samples, a, b, c, d)[0]

    assert np.all(np.isclose(inifite, finite))


def test_interpolation():
    parameters = load_parameters('data/kirkland.txt')
    a, b, c, d = convert_kirkland(parameters[47])

    n = 200
    r_cut = 1.8
    r = np.linspace(r_cut / n, r_cut, n)
    z0 = np.array([-20.])
    z1 = np.array([20.])
    samples = 10000

    vr = kirkland_projected_finite(r, r_cut, z0, z1, samples, a, b, c, d)

    extent = r_cut * 2
    m = 2 * n + 1

    v = np.zeros((m, m))
    corner_positions = np.array([[0, 0]])
    block_positions = np.array([[extent / 2, extent / 2]])
    x = np.linspace(0., extent, m)
    y = np.linspace(0., extent, m)
    colors = np.array([0])

    interpolation_kernel(v, r, vr, r_cut, corner_positions, block_positions, x, y, colors)

    assert np.all(np.isclose(vr[0], v[m // 2, m // 2 + 1:]))
