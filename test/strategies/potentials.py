import hypothesis.strategies as st
import numpy as np
from ase import Atoms
from ase.build import bulk
from hypothesis.extra import numpy as numpy_st

from abtem.potentials.potentials import Potential, PotentialArray
from abtem.potentials.temperature import FrozenPhonons, DummyFrozenPhonons
from . import core as core_st


@st.composite
def atoms(draw,
          min_side_length=1.,
          max_side_length=5.,
          min_thickness=.5,
          max_thickness=4.,
          min_atoms=5,
          max_atoms=10,
          min_atomic_number=1,
          max_atomic_number=102):
    n = draw(st.integers(min_atoms, max_atoms))

    numbers = st.lists(elements=st.integers(min_value=min_atomic_number, max_value=max_atomic_number),
                       min_size=n, max_size=n)

    cell = st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                     st.floats(min_value=min_side_length, max_value=max_side_length),
                     st.floats(min_value=min_thickness, max_value=max_thickness))

    positions = numpy_st.arrays(dtype=float,
                                shape=(n, 3),
                                elements=core_st.sensible_floats(min_value=0, max_value=max_side_length))

    return Atoms(numbers=draw(numbers), positions=draw(positions), cell=draw(cell))


@st.composite
def frozen_phonons(draw,
                   min_side_length=1.,
                   max_side_length=5.,
                   min_thickness=.5,
                   max_thickness=5.,
                   max_atoms=10,
                   ensemble_mean=True,
                   min_configs=1,
                   max_configs=5):
    drawn_atoms = draw(atoms(min_side_length=min_side_length,
                             max_side_length=max_side_length,
                             min_thickness=min_thickness,
                             max_thickness=max_thickness,
                             max_atoms=max_atoms))
    num_configs = draw(st.integers(min_value=min_configs, max_value=max_configs))
    sigmas = draw(st.floats(min_value=0., max_value=.2))
    seeds = draw(st.one_of(st.none(), st.integers(min_value=0)))
    return FrozenPhonons(drawn_atoms,
                         num_configs=num_configs,
                         sigmas=sigmas,
                         seeds=seeds,
                         ensemble_mean=ensemble_mean)


@st.composite
def dummy_frozen_phonons(draw,
                         min_side_length=1.,
                         max_side_length=5.,
                         min_thickness=.5,
                         max_thickness=5.,
                         max_atoms=10, ):
    drawn_atoms = draw(atoms(min_side_length=min_side_length,
                             max_side_length=max_side_length,
                             min_thickness=min_thickness,
                             max_thickness=max_thickness,
                             max_atoms=max_atoms))

    return DummyFrozenPhonons(drawn_atoms)


@st.composite
def potential(draw,
              no_frozen_phonons=False,
              min_frozen_phonons=1,
              max_frozen_phonons=1,
              exit_planes=False,
              ensemble_mean=True,
              device='cpu'):
    gpts = draw(core_st.gpts())

    if exit_planes:
        exit_planes = draw(st.integers(min_value=1, max_value=2))
    else:
        exit_planes = None

    if no_frozen_phonons:
        fp = draw(atoms())
    else:
        fp = draw(frozen_phonons(min_configs=min_frozen_phonons,
                                 max_configs=max_frozen_phonons,
                                 ensemble_mean=ensemble_mean))

    potential = Potential(fp,
                          exit_planes=exit_planes,
                          gpts=gpts,
                          device=device)
    return potential


@st.composite
def gold_potential(draw):
    atoms = bulk('Au', cubic=True) * (2, 2, 1)
    gpts = draw(core_st.gpts())
    potential = Potential(atoms, gpts=gpts)
    return potential


@st.composite
def potential_array(draw, lazy=True, device='cpu', min_base_side=8):
    shape = draw(core_st.shape(base_dims=3, min_base_side=min_base_side, min_ensemble_dims=0, max_ensemble_dims=1))
    axes = draw(st.lists(core_st.ordinal_axis_metadata(shape[0]), min_size=len(shape) - 3, max_size=len(shape) - 3))

    if lazy:
        chunks = draw(core_st.chunks(shape, (1,) * (len(shape) - 3) + (-1,) * 3))
    else:
        chunks = None

    array = core_st.random_array(shape, chunks=chunks, device=device, min_value=0., dtype=np.complex64)
    sampling = draw(core_st.sampling())
    slice_thickness = draw(st.floats(min_value=0.1, max_value=2.))
    return PotentialArray(array=array, sampling=sampling, ensemble_axes_metadata=axes, slice_thickness=slice_thickness)
