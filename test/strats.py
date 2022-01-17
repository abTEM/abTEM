import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays, array_shapes


@st.composite
def gpts(draw, min_value=8, max_value=128, allow_none=False):
    gpts = st.integers(min_value=min_value, max_value=max_value)
    gpts = gpts | st.tuples(gpts, gpts)
    if allow_none:
        gpts = gpts | st.none()
    gpts = st.one_of(gpts)
    gpts = draw(gpts)
    return gpts


@st.composite
def sampling(draw, min_value=0.01, max_value=0.1, allow_none=False):
    sampling = st.floats(min_value=min_value, max_value=max_value)
    sampling = sampling | st.tuples(sampling, sampling)
    if allow_none:
        sampling = sampling | st.none()
    sampling = st.one_of(sampling)
    sampling = draw(sampling)
    return sampling


@st.composite
def extent(draw, min_value=1., max_value=10., allow_none=False):
    extent = st.floats(min_value=min_value, max_value=max_value)
    extent = extent | st.tuples(extent, extent)
    if allow_none:
        extent = extent | st.none()
    extent = st.one_of(extent)
    extent = draw(extent)
    return extent


@st.composite
def energy(draw, min_value=80e3, max_value=300e3, allow_none=False):
    energy = st.floats(min_value=min_value, max_value=max_value)
    if allow_none:
        energy = energy | st.none()
    energy = draw(energy)
    return energy


# @st.composite
# def tilt(draw, min_value=0., max_value=10.):
#     energy = st.floats(min_value=min_value, max_value=max_value)
#     if allow_none:
#         energy = energy | st.none()
#     energy = draw(energy)
#     return energy


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
                      max_atoms=10):

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


device = st.sampled_from(['cpu', 'gpu'])
