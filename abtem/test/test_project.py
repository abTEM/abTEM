import numpy as np
from scipy.special import kn

from abtem.interpolation import interpolation_kernel_parallel
from abtem.parametrizations import project_riemann, kirkland, dvdr_kirkland, project_tanh_sinh, kirkland_soft, \
    load_kirkland_parameters
from abtem.potentials import tanh_sinh_quadrature, Potential
from ase import Atoms


def kirkland_project_infinite(r, p):
    v = (2 * p[0, 0] * kn(0, p[1, 0] * r) + np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.exp(-p[3, 0] * r ** 2.) +
         2 * p[0, 1] * kn(0, p[1, 1] * r) + np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.exp(-p[3, 1] * r ** 2.) +
         2 * p[0, 2] * kn(0, p[1, 2] * r) + np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.exp(-p[3, 2] * r ** 2.))
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

    v_riemann = project_riemann(r, cutoff, cutoff_value, derivative_cutoff_value, z0, z1, 10000, kirkland_soft,
                                parameters)
    v_tanh_sinh = project_tanh_sinh(r, cutoff, cutoff_value, derivative_cutoff_value, z0, z1, xk, wk, kirkland_soft,
                                    parameters)
    assert np.allclose(v_riemann, v_tanh_sinh, atol=1e-8)


def test_project():
    parameters = load_kirkland_parameters()[47]

    n = 200
    r_cut = 20
    r = np.linspace(r_cut / n, r_cut, n)
    z0 = np.array([-20.])
    z1 = np.array([20.])
    samples = 100000

    inifite = kirkland_project_infinite(r, parameters)

    v_cut = kirkland(r_cut, parameters)
    dvdr_cut = dvdr_kirkland(r_cut, parameters)

    finite = project_riemann(r, r_cut, v_cut, dvdr_cut, z0, z1, samples, kirkland_soft, parameters)[0]

    assert np.all(np.isclose(inifite, finite))


def test_interpolation():
    parameters = load_kirkland_parameters()[47]

    n = 10
    r_cut = .5
    r = np.linspace(0, r_cut, n)
    z0 = np.array([-10.])
    z1 = np.array([10.])
    samples = 10000

    v_cut = kirkland(r_cut, parameters)
    dvdr_cut = dvdr_kirkland(r_cut, parameters)

    vr = project_riemann(r, r_cut, v_cut, dvdr_cut, z0, z1, samples, kirkland_soft, parameters)

    extent = (r_cut) * 2
    m = 2 * n - 1

    v = np.zeros((m, m))
    corner_positions = np.array([[0, 0]])
    block_positions = np.array([[0, 0]])  # + r_cut / n / 2
    x = np.linspace(0., extent, m)
    y = np.linspace(0., extent, m)

    interpolation_kernel_parallel(v, r, vr, corner_positions, block_positions, x, y, thread_safe=True)

    np.allclose(vr[0], v[0, :10])

