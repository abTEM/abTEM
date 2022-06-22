import hypothesis.strategies as st

from abtem.potentials.potentials import Potential
from . import atoms as atoms_st
from . import core as core_st
from ase.build import bulk


@st.composite
def random_potential(draw):
    atoms = draw(atoms_st.random_atoms(min_atomic_number=10))
    gpts = draw(core_st.gpts())

    potential = Potential(atoms, gpts=gpts)
    return potential


@st.composite
def gold_potential(draw):
    atoms = bulk('Au', cubic=True) * (2, 2, 1)
    gpts = draw(core_st.gpts())
    potential = Potential(atoms, gpts=gpts)
    return potential


@st.composite
def random_potential_array(draw):
    return draw(random_potential()).build(lazy=False)
