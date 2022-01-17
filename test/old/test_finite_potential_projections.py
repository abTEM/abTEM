import numpy as np

from abtem.potentials.atom import AtomicPotential
from abtem.potentials.parametrizations import lobato


def test():
    p = AtomicPotential('H', cutoff_tolerance=1e-4, num_integration_limits=20)
    p.build_integral_table()

    gpts = 32
    sampling = .025
    array = np.zeros((gpts,) * 2, dtype=np.float32)
    positions = np.array([[0., 0., 0.]], dtype=np.float32)
    a = np.array([-10.], dtype=np.float32)
    b = np.array([10.], dtype=np.float32)

    r = np.linspace(0, gpts * sampling, gpts, endpoint=False)
    vr = p.project_on_grid(array, (sampling,) * 2, positions, a, b)

    analytical = lobato.projected_potential(r[1:], lobato.load_parameters()['H'])
    numerical = vr[0][1:]
    relative_error = (numerical - analytical) / numerical * 100

    assert np.all(relative_error < 1)
