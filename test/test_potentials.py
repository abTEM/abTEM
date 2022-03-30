import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, settings

from abtem.potentials.parametrizations import LobatoParametrization, KirklandParametrization
from abtem.potentials.potentials import Potential
from strategies import atoms as atoms_st
from strategies import core as core_st
from utils import array_is_close, gpu


@given(atom_data=atoms_st.empty_atoms_data(),
       gpts=core_st.gpts(),
       slice_thickness=st.floats(min_value=.1, max_value=2.)
       )
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', [gpu, 'cpu'])
@pytest.mark.parametrize('parametrization', ['kirkland', 'lobato'])
@pytest.mark.parametrize('projection', ['finite', 'infinite'])
def test_build_is_lazy(atom_data, gpts, slice_thickness, lazy, device, parametrization, projection):
    atoms = Atoms(**atom_data)
    potential = Potential(atoms, gpts=gpts, device=device, slice_thickness=slice_thickness,
                          parametrization=parametrization, projection=projection)
    potential.build(lazy=lazy).compute()


@settings(max_examples=2)
@given(Z=st.integers(1, 102),
       slice_thickness=st.floats(min_value=2., max_value=4.)
       )
@pytest.mark.parametrize('parametrization', ['kirkland', 'lobato'])
def test_finite_infinite_projected_match(Z, slice_thickness, parametrization):
    atoms = Atoms([Z], positions=[(0., 0., 4)], cell=[8., 8., 8.])
    print(Z)
    finite_potential = Potential(atoms,
                                 sampling=0.01,
                                 projection='finite',
                                 slice_thickness=slice_thickness,
                                 parametrization=parametrization,
                                 cutoff_tolerance=1e-5)

    finite_potential = finite_potential.build(lazy=False).project()
    infinite_potential = Potential(atoms,
                                   sampling=0.01,
                                   projection='infinite',
                                   slice_thickness=slice_thickness,
                                   parametrization=parametrization)
    infinite_potential = infinite_potential.build(lazy=False).project()

    mask = np.ones_like(finite_potential.array, dtype=bool)
    mask[0, 0] = 0
    assert array_is_close(finite_potential.array, infinite_potential.array, rel_tol=.01, check_above_rel=.1, mask=mask)


@given(Z=st.integers(1, 102),
       slice_thickness=st.floats(min_value=2, max_value=4.),
       sampling=st.floats(min_value=0.01, max_value=0.02))
@pytest.mark.parametrize('parametrization', [LobatoParametrization(), KirklandParametrization()])
def test_infinite_projected_match(Z, slice_thickness, parametrization, sampling):
    sidelength = 8

    atoms = Atoms([Z], positions=[(0., 0., sidelength / 2)], cell=[sidelength, sidelength, sidelength])

    potential = Potential(atoms,
                          slice_thickness=slice_thickness,
                          sampling=sampling,
                          projection='infinite',
                          parametrization=parametrization)

    r = np.linspace(0, sidelength, potential.gpts[0], endpoint=False)[1:]
    analytical_potential = parametrization.projected_potential(Z)(r)

    potential = potential.build(lazy=False).project().array[0, 1:]
    assert array_is_close(potential, analytical_potential, rel_tol=.01, check_above_rel=.01)


@settings(max_examples=2)
@given(Z=st.integers(1, 102),
       slice_thickness=st.floats(min_value=2, max_value=4.),
       sampling=st.floats(min_value=0.02, max_value=0.04))
@pytest.mark.parametrize('parametrization', [LobatoParametrization(), KirklandParametrization()])
def test_finite_projected_match(Z, slice_thickness, parametrization, sampling):
    sidelength = 8
    atoms = Atoms([Z], positions=[(0., 0., sidelength / 2)], cell=[sidelength, sidelength, sidelength])

    potential = Potential(atoms,
                          slice_thickness=slice_thickness,
                          sampling=sampling,
                          projection='finite',
                          parametrization=parametrization)

    r = np.linspace(0, sidelength, potential.gpts[0], endpoint=False)[1:]
    analytical_potential = parametrization.projected_potential(Z)(r)

    potential = potential.build(lazy=False).project().array[0, 1:]
    assert array_is_close(potential, analytical_potential, rel_tol=.01, check_above_rel=.01)


# def test_atom_position():
#     from ase import Atoms
#
#     L = 8.0
#     z1 = 0
#     z2 = L / 2
#
#     atoms1 = Atoms('C', [(L / 2, L / 2, z1)], cell=(L,) * 3)
#     atoms2 = Atoms('C', [(L / 2, L / 2, z2)], cell=(L,) * 3)
#
#     potential1 = Potential(atoms1, sampling=.1, projection='finite', slice_thickness=L)
#     potential2 = Potential(atoms2, sampling=.1, projection='finite', slice_thickness=L)
#
#     # print(potential1.num_slices, potential2.num_slices)
#
#     potential1 = potential1.build(lazy=False)
#     potential2 = potential2.build(lazy=False)