from functools import reduce
from operator import mul

import ase.build
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import given

from abtem import FrozenPhonons


@given(data=st.data())
@pytest.mark.parametrize(
    "frozen_phonons",
    [
        abtem_st.dummy_frozen_phonons,
        abtem_st.frozen_phonons,
        abtem_st.md_frozen_phonons,
    ],
)
@pytest.mark.parametrize(
    "lazy",
    [
        True,
        False,
    ],
)
def test_frozen_phonons_as_ensembles(data, frozen_phonons, lazy):
    frozen_phonons = data.draw(frozen_phonons(lazy=lazy))

    if len(frozen_phonons.ensemble_shape) > 0:
        chunks = data.draw(
            st.integers(
                min_value=1, max_value=reduce(mul, frozen_phonons.ensemble_shape)
            )
        )
    else:
        chunks = ()

    blocks = frozen_phonons.ensemble_blocks(chunks).compute()

    # assert all([not block.is_lazy for block in blocks])

    for i, _, fp in frozen_phonons.generate_blocks(chunks):
        fp = fp.item()

        assert blocks[i] == fp

    # assert all(isinstance(array, da.core.Array) for array in frozen_phonons._partition_args(lazy=True))
    # assert all(not isinstance(array, da.core.Array) for array in frozen_phonons._partition_args(lazy=False))


def test_sigmas():
    atoms = ase.build.bulk("Au", cubic=True) * (2, 2, 2)

    frozen_phonons = FrozenPhonons(
        atoms, num_configs=1000, sigmas=0.1, seed=None, directions="xyz"
    )

    positions = np.stack(
        [atoms.positions for atoms in frozen_phonons.to_atoms_ensemble().trajectory]
    )
    positions = positions - positions.mean(axis=0)

    assert np.abs(positions.std() - 0.1) < 0.001


def test_default_directions_displace_all_three_axes():
    atoms = ase.build.bulk("Au", cubic=True) * (2, 2, 2)

    for directions, displaced in ((None, (0, 1, 2)), ("xy", (0, 1)), ("z", (2,))):
        kwargs = {} if directions is None else {"directions": directions}
        frozen_phonons = FrozenPhonons(
            atoms, num_configs=1, sigmas=0.1, seed=1, **kwargs
        )
        displacement = (
            frozen_phonons.to_atoms_ensemble().trajectory[0].positions - atoms.positions
        )
        for axis in range(3):
            assert np.any(displacement[:, axis] != 0) == (axis in displaced)


@pytest.mark.parametrize("directions", ["xq", "q", "x y"])
def test_invalid_direction_is_named_at_construction(directions):
    atoms = ase.build.bulk("Au", cubic=True)
    bad = next(d for d in directions if d not in "xyz")

    with pytest.raises(RuntimeError, match=f"not '{bad}'"):
        FrozenPhonons(atoms, num_configs=1, sigmas=0.1, directions=directions)


def test_lazy_partition_args_embed_atoms_once():
    # One Atoms node shared by every configuration chunk, not a copy per chunk.
    atoms = ase.build.bulk("Au", cubic=True) * (2, 2, 2)
    frozen_phonons = FrozenPhonons(atoms, num_configs=6, sigmas=0.1, seed=1)

    (array,) = frozen_phonons._partition_args(chunks=1, lazy=True)

    graph = dict(array.__dask_graph__())
    # Newer dask wraps literal graph values in a DataNode holding ``.value``.
    atoms_keys = [
        key
        for key, value in graph.items()
        if isinstance(getattr(value, "value", value), ase.Atoms)
    ]
    assert len(array.chunks[0]) == 6
    assert len(atoms_keys) == 1


def test_gpaw_lazy_partition_args_ship_the_calculator_once():
    # The calculator (density and potential grids) is one graph node shared by
    # every configuration chunk, not a literal argument pickled into each task
    # the scheduler sends to a worker. GPAW is not needed: _partition_args only
    # reads the calculator, and a stand-in object marks where it ends up.
    import cloudpickle

    from abtem.potentials.gpaw import GPAWPotential

    class StandInCalculator:
        def __init__(self):
            self.stand_in_calculator_grid = np.zeros(1000)

    calculator = StandInCalculator()
    potential = GPAWPotential.__new__(GPAWPotential)
    potential._calculators = calculator
    potential._frozen_phonons = FrozenPhonons(
        ase.build.bulk("Au", cubic=True), num_configs=4, sigmas=0.1, seed=1
    )

    (array,) = potential._partition_args(chunks=1, lazy=True)

    tasks_with_calculator = [
        key
        for key, task in dict(array.__dask_graph__()).items()
        if b"stand_in_calculator_grid" in cloudpickle.dumps(task)
    ]
    assert len(array.chunks[0]) == 4
    assert len(tasks_with_calculator) == 1

    (eager,) = potential._partition_args(chunks=1, lazy=False)
    for lazy_args, eager_args in zip(array.compute(), eager):
        eager_args = eager_args.item()
        assert lazy_args["calculators"] is calculator
        assert eager_args["calculators"] is calculator
        # the lazy chunk arrives wrapped in a one-element object array
        lazy_atoms, lazy_seeds = np.asarray(lazy_args["frozen_phonons"]).item()
        eager_atoms, eager_seeds = eager_args["frozen_phonons"]
        assert lazy_atoms == eager_atoms
        assert tuple(lazy_seeds) == tuple(eager_seeds)
