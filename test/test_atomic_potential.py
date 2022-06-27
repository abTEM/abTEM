import numpy as np
import pytest

from abtem.potentials.atom import AtomicPotential
from abtem.potentials.parametrizations import LobatoParametrization, KirklandParametrization


@pytest.mark.parametrize('parametrization',
                         [LobatoParametrization(),
                          KirklandParametrization()])
def test_project(parametrization):
    atomic_potential = AtomicPotential(102,
                                       parametrization=parametrization,
                                       cutoff_tolerance=1e-7,
                                       quad_order=20,
                                       taper=1)

    analytical = parametrization.projected_potential(102)(atomic_potential.radial_gpts(inner_cutoff=0.01))

    table = atomic_potential.build_integral_table(inner_limit=0.01)

    assert np.allclose(analytical, table.project(-atomic_potential.cutoff, atomic_potential.cutoff))


@pytest.mark.parametrize('parametrization',
                         [LobatoParametrization(),
                          KirklandParametrization()])
def test_project_on_grid(parametrization):
    Z = 102
    sampling = (.02, .02)
    array = np.zeros((100, 1), dtype=float)
    positions = np.zeros((1, 2), dtype=float)
    inner_cutoff = 0.01

    atomic_potential = AtomicPotential(Z,
                                       parametrization=parametrization,
                                       cutoff_tolerance=1e-7,
                                       quad_order=20,
                                       taper=1)

    table = atomic_potential.build_integral_table(inner_limit=inner_cutoff)

    a = np.array([-atomic_potential.cutoff])
    b = np.array([atomic_potential.cutoff])

    projected = table.project_on_grid(array, sampling, positions, a, b)

    r = np.linspace(sampling[0], array.shape[0] * sampling[0], array.shape[0] - 1, endpoint=False)
    analytical = parametrization.projected_potential(Z)(r)

    assert np.allclose(analytical, projected[1:, 0], rtol=1e-3)
