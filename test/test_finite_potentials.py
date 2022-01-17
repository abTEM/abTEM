import numpy as np

from abtem.potentials.atom import AtomicPotential
from abtem.potentials.parametrizations import lobato
from hypothesis import given, settings, assume, Verbosity, Phase
from ase.data import chemical_symbols
import hypothesis.strategies as st




@given(Z=st.integers(min_value=1, max_value=90),
       gpts=st.integers(min_value=16, max_value=64),
       sampling=st.floats(min_value=.02, max_value=.05))
@settings(max_examples=5, deadline=None)
def test_potential_projection(Z, gpts, sampling):
    p = AtomicPotential(Z, cutoff_tolerance=1e-4, num_integration_limits=20)
    p.build_integral_table()

    position_x = sampling * 3
    position_y = sampling * 2

    array = np.zeros((gpts,) * 2, dtype=np.float32)

    positions = np.array([[0., 0., 0.]], dtype=np.float32)
    a = np.array([-10.], dtype=np.float32)
    b = np.array([10.], dtype=np.float32)

    r = np.linspace(0, gpts * sampling, gpts, endpoint=False)
    vr = p.project_on_grid(array, (sampling,) * 2, positions, a, b)

    analytical = lobato.projected_potential(r[1:], lobato.load_parameters()[chemical_symbols[Z]])
    numerical = vr[0][1:]

    assert array_is_close(analytical, numerical, rel_tol=.01, check_above_rel=.01)

    positions = np.array([[0., 0., 0.]] * 4, dtype=np.float32)
    a = np.array([-10., -5., -1., 0.], dtype=np.float32)
    b = np.array([-5., -1., 0., 10.], dtype=np.float32)

    array = np.zeros((gpts,) * 2, dtype=np.float32)
    vr2 = p.project_on_grid(array, (sampling,) * 2, positions, a, b)
    assert np.allclose(vr2, vr)

    positions = np.array([[position_x, position_y, 0.]], dtype=np.float32)
    a = np.array([-10.], dtype=np.float32)
    b = np.array([10.], dtype=np.float32)

    array = np.zeros((gpts,) * 2, dtype=np.float32)
    vr3 = p.project_on_grid(array, (sampling,) * 2, positions, a, b)

    max_x, max_y = np.where(np.max(vr3) == vr3)

    assert len(max_x) == len(max_y) == 1
    assert np.allclose(max_x, position_x / sampling)
    assert np.allclose(max_y, position_y / sampling)
