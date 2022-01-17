import numpy as np
from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st
from hypothesis.strategies import integers, floats, composite, booleans, lists
from abtem import PlaneWave, Potential, FrozenPhonons
from ase import Atoms
from hypothesis.extra.numpy import arrays, array_shapes

# @composite
# def plane_wave(draw,
#                gpts=integers(min_value=1, max_value=1024),
#                sampling=floats(min_value=.01, max_value=1),
#                energy=floats(min_value=50e3, max_value=500e3),
#                normalize=booleans()
#                ):
#     gpts = draw(gpts)
#     sampling = draw(sampling)
#     energy = draw(energy)
#     normalize = draw(normalize)
#     plane_wave = PlaneWave(gpts=gpts, sampling=sampling, energy=energy, normalize=normalize)
#     return plane_wave
from abtem.core.backend import get_array_module


@composite
def atoms(draw):
    n = draw(st.integers(1, 10))
    numbers = draw(arrays(int, n, elements=st.integers(min_value=1, max_value=80)))
    cell = draw(arrays(float, 3, elements=st.floats(min_value=1, max_value=10)))
    positions = draw(arrays(float, (n, 3), elements=st.floats(min_value=1, max_value=max(cell))))
    atoms = Atoms(numbers, positions=positions, cell=cell)
    atoms.wrap()
    return atoms


@composite
def frozen_phonons(draw, atoms=atoms()):
    atoms = draw(atoms)
    num_configs = draw(st.integers(min_value=1, max_value=4))
    atoms = FrozenPhonons(atoms, sigmas=.1, num_configs=num_configs)
    return atoms


@given(atoms=frozen_phonons(),
       gpts=st.integers(min_value=32, max_value=128),
       slice_thickness=st.floats(min_value=.1, max_value=10.),
       projection=st.sampled_from(['finite', 'infinite']),
       parametrization=st.sampled_from(['kirkland', 'lobato']),
       device=st.sampled_from(['cpu', 'gpu']))
@settings(deadline=None, print_blob=True, max_examples=6)
def test_build_potential(atoms,
                         gpts,
                         slice_thickness,
                         projection,
                         parametrization,
                         device):

    potential = Potential(atoms,
                          gpts=gpts,
                          slice_thickness=slice_thickness,
                          projection=projection,
                          parametrization=parametrization,
                          device=device,
                          )
    potential_array = potential.build()

    xp = get_array_module(potential_array.array)

    assert potential.num_slices > 0
    assert np.isclose(potential.thickness, atoms.cell[2, 2])
    assert xp.any(potential_array.array)

    # assert all([xp.all(potential_array.array[i] == potential[i].array) for i in range(len(potential))])

    copied_potential_array = potential.copy().build()
    assert xp.all(copied_potential_array.array == potential_array.array)
