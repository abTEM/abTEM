import numpy as np
from scipy.special import kn

from abtem.interpolation import interpolation_kernel_parallel
from abtem.parametrizations import kirkland_projected_finite_riemann, kirkland_parameters, convert_kirkland, kirkland, \
    dvdr_kirkland, kirkland_projected_finite_tanh_sinh
from abtem.potentials import tanh_sinh_quadrature, Potential
from ase import Atoms


def kirkland_project_infinite(r, a, b, c, d):
    v = (2 * a[0] * kn(0, b[0] * r) + np.sqrt(np.pi / d[0]) * c[0] * np.exp(-d[0] * r ** 2.) +
         2 * a[1] * kn(0, b[1] * r) + np.sqrt(np.pi / d[1]) * c[1] * np.exp(-d[1] * r ** 2.) +
         2 * a[2] * kn(0, b[2] * r) + np.sqrt(np.pi / d[2]) * c[2] * np.exp(-d[2] * r ** 2.))
    return v


def test_tanh_sinh_quadrature():
    for sigma in [.5, .05, .005]:
        def gaussian(x):
            return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- x ** 2 / sigma ** 2 / 2)

        a = -20
        b = 20
        k = 30

        xk, wk = tanh_sinh_quadrature(int(k / sigma), sigma / k)
        c = (b - a) / 2.
        d = (b + a) / 2.

        xk = xk * c + d
        wk = wk * c

        assert np.sum(gaussian(xk) * wk) - 1 < 1e-7
        assert np.sum(gaussian(xk) * wk) - 1 > 1e-9


def test_tanh_sinh_vs_riemann():
    atoms = Atoms('Ag', positions=[(4, 4, .5)], cell=(8, 8, 4))
    potential = Potential(atoms=atoms, sampling=.02, num_slices=1, parametrization='kirkland', quadrature_order=1000)

    atomic_number = 47
    i = 0
    slice_thickness = potential.slice_thickness(i)

    positions = potential._get_atomic_positions(atomic_number)
    parameters = potential._get_adjusted_parameters(atomic_number)
    cutoff = potential.get_cutoff(atomic_number)
    cutoff_value = potential._get_cutoff_value(atomic_number)
    derivative_cutoff_value = potential._get_derivative_cutoff_value(atomic_number)
    r = potential._get_radial_coordinates(atomic_number)
    xk, wk = potential._get_quadrature()

    positions = positions[np.abs((i + .5) * slice_thickness - positions[:, 2]) < (cutoff + slice_thickness / 2)]

    z0 = i * potential.slice_thickness(i) - positions[:, 2]
    z1 = (i + 1) * potential.slice_thickness(i) - positions[:, 2]

    v_riemann = kirkland_projected_finite_riemann(r, cutoff, cutoff_value, derivative_cutoff_value, z0, z1, 10000,
                                                  *parameters)
    v_tanh_sinh = kirkland_projected_finite_tanh_sinh(r, cutoff, cutoff_value, derivative_cutoff_value, z0, z1, xk, wk,
                                                      *parameters)
    assert np.allclose(v_riemann, v_tanh_sinh, atol=1e-8)


def test_project():
    a, b, c, d = convert_kirkland(kirkland_parameters[47])

    n = 200
    r_cut = 20
    r = np.linspace(r_cut / n, r_cut, n)
    z0 = np.array([-20.])
    z1 = np.array([20.])
    samples = 100000

    inifite = kirkland_project_infinite(r, a, b, c, d)

    v_cut = kirkland(r_cut, a, b, c, d)
    dvdr_cut = dvdr_kirkland(r_cut, a, b, c, d)

    finite = kirkland_projected_finite_riemann(r, r_cut, v_cut, dvdr_cut, z0, z1, samples, a, b, c, d)[0]

    assert np.all(np.isclose(inifite, finite))


def test_interpolation():
    a, b, c, d = convert_kirkland(kirkland_parameters[47])

    n = 200
    r_cut = 1.8
    r = np.linspace(r_cut / n, r_cut, n)
    z0 = np.array([-20.])
    z1 = np.array([20.])
    samples = 10000

    v_cut = kirkland(r_cut, a, b, c, d)
    dvdr_cut = dvdr_kirkland(r_cut, a, b, c, d)

    vr = kirkland_projected_finite_riemann(r, r_cut, v_cut, dvdr_cut, z0, z1, samples, a, b, c, d)

    extent = r_cut * 2
    m = 2 * n + 1

    v = np.zeros((m, m))
    corner_positions = np.array([[0, 0]])
    block_positions = np.array([[extent / 2, extent / 2]])
    x = np.linspace(0., extent, m)
    y = np.linspace(0., extent, m)

    interpolation_kernel_parallel(v, r, vr, corner_positions, block_positions, x, y, thread_safe=True)

    assert np.all(np.isclose(vr[0], v[m // 2, m // 2 + 1:]))
