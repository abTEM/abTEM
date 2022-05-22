import hypothesis.strategies as st
from ase import Atoms

from abtem.potentials.temperature import FrozenPhonons
from hypothesis.extra import numpy as numpy_st
from .core import sensible_floats


@st.composite
def empty_atoms_data(draw,
                     min_side_length=1.,
                     max_side_length=5.,
                     min_thickness=.5,
                     max_thickness=5.):
    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))
    return {
        'numbers': [],
        'positions': [],
        'cell': cell
    }


@st.composite
def random_atoms_data(draw,
                      min_side_length=1.,
                      max_side_length=5.,
                      min_thickness=.5,
                      max_thickness=5.,
                      max_atoms=5):
    n = draw(st.integers(1, max_atoms))

    numbers = draw(st.lists(elements=st.integers(min_value=1, max_value=80), min_size=n, max_size=n))

    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))

    position = st.tuples(st.floats(min_value=0, max_value=cell[0]),
                         st.floats(min_value=0, max_value=cell[1]),
                         st.floats(min_value=0, max_value=cell[2]))

    positions = draw(st.lists(elements=position, min_size=n, max_size=n))

    return {
        'numbers': numbers,
        'positions': positions,
        'cell': cell
    }


@st.composite
def random_atoms(draw,
                 min_side_length=1.,
                 max_side_length=5.,
                 min_thickness=.5,
                 max_thickness=4.,
                 max_atoms=10):
    n = draw(st.integers(1, max_atoms))

    numbers = st.lists(elements=st.integers(min_value=1, max_value=80), min_size=n, max_size=n)

    cell = st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                     st.floats(min_value=min_side_length, max_value=max_side_length),
                     st.floats(min_value=min_thickness, max_value=max_thickness))

    positions = numpy_st.arrays(dtype=float,
                                shape=(n, 3),
                                elements=sensible_floats(min_value=0, max_value=max_side_length))

    return Atoms(numbers=draw(numbers), positions=draw(positions), cell=draw(cell))


@st.composite
def random_frozen_phonons(draw,
                          min_side_length=1.,
                          max_side_length=5.,
                          min_thickness=.5,
                          max_thickness=5.,
                          max_atoms=10,
                          max_configs=2):
    atoms = draw(random_atoms(min_side_length=min_side_length,
                              max_side_length=max_side_length,
                              min_thickness=min_thickness,
                              max_thickness=max_thickness,
                              max_atoms=max_atoms))
    num_configs = draw(st.integers(min_value=1, max_value=max_configs))
    sigmas = draw(st.floats(min_value=0., max_value=.2))
    return FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas)
