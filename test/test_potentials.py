import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, settings

import strats as abst
from abtem.potentials.parametrizations import LobatoParametrization, KirklandParametrization
from abtem.potentials.potentials import Potential
from utils import array_is_close


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(),
       gpts=abst.gpts(),
       slice_thickness=st.floats(min_value=.1, max_value=2.)
       )
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['gpu', 'cpu'])
@pytest.mark.parametrize('parametrization', ['kirkland', 'lobato'])
@pytest.mark.parametrize('projection', ['finite', 'infinite'])
def test_build_is_lazy(atom_data, gpts, slice_thickness, lazy, device, parametrization, projection):
    atoms = Atoms(**atom_data)
    potential = Potential(atoms, gpts=gpts, device=device, slice_thickness=slice_thickness,
                          parametrization=parametrization, projection=projection)
    potential.build(lazy=lazy).compute()


@settings(deadline=None, max_examples=1)
@given(Z=st.integers(1, 102),
       slice_thickness=st.floats(min_value=.1, max_value=2.)
       )
@pytest.mark.parametrize('parametrization', ['kirkland', 'lobato'])
def test_finite_infinite_projected_match(Z, slice_thickness, parametrization):
    atoms = Atoms([Z], positions=[(0., 0., 2.5)], cell=[5., 5., 5.])

    finite_potential = Potential(atoms, gpts=512, projection='finite', slice_thickness=slice_thickness,
                                 parametrization=parametrization)
    finite_potential = finite_potential.build(lazy=False).project()
    infinite_potential = Potential(atoms, gpts=512, projection='infinite', slice_thickness=slice_thickness,
                                   parametrization=parametrization)
    infinite_potential = infinite_potential.build(lazy=False).project()

    mask = np.ones_like(finite_potential.array, dtype=bool)
    mask[0, 0] = 0

    assert array_is_close(finite_potential.array, infinite_potential.array, rel_tol=.02, check_above_rel=.02, mask=mask)


@settings(deadline=None, max_examples=1)
@given(Z=st.integers(1, 102),
       slice_thickness=st.floats(min_value=.1, max_value=2.)
       )
@pytest.mark.parametrize('parametrization', [LobatoParametrization(), KirklandParametrization()])
def test_finite_infinite_projected_match(Z, slice_thickness, parametrization):
    L = 8
    gpts = 512

    atoms = Atoms([Z], positions=[(0., 0., L / 2)], cell=[L, L, L])

    finite_potential = Potential(atoms, slice_thickness=slice_thickness, gpts=gpts, projection='finite',
                                 parametrization=parametrization, cutoff_tolerance=1e-5)
    finite_potential = finite_potential.build(lazy=False).project().array[0, 1:]

    infinite_potential = Potential(atoms, slice_thickness=slice_thickness, gpts=gpts, projection='infinite',
                                   parametrization=parametrization,
                                   cutoff_tolerance=1e-5)
    infinite_potential = infinite_potential.build(lazy=False).project().array[0, 1:]

    r = np.linspace(0, L, gpts, endpoint=False)[1:]
    analytical_potential = parametrization.projected_potential(Z)(r)

    assert array_is_close(infinite_potential, analytical_potential, rel_tol=.01, check_above_rel=.01)
    assert array_is_close(finite_potential, analytical_potential, rel_tol=.01, check_above_rel=.01)
