import numpy as np

from abtem.potentials import KirklandPotential, project_riemann


def test_project():
    kirkland = KirklandPotential()

    func = kirkland.get_potential(47)
    projected_func = kirkland.get_projected_potential(47)

    r = np.linspace(.5, 3, 10)
    r_cut = 100.

    a = np.array([-15.])
    b = np.array([15.])

    assert np.allclose(project_riemann(func, r, r_cut, a, b, 100000)[0], projected_func(r))

    a = np.array([-10., -9])
    b = np.array([9., 10])

    assert np.all(np.abs(np.diff(project_riemann(func, r, r_cut, a, b, 10000), axis=0)) < 1e-10)
